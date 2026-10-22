"""Tests for CefpycoTransport against a fake cefpyco handle (no root, no real cefnetd)."""

from __future__ import annotations

from dataclasses import dataclass

from semindex import holder_state, schema
from semindex.cefpyco_transport import CefpycoTransport
from semindex.transport import Response


@dataclass
class _FakePacket:
    """A stand-in for cefpyco.CcnPacketInfo: only the fields CefpycoTransport reads."""

    is_interest: bool = False
    is_data: bool = False
    name: str | None = None
    payload: bytes | None = None
    chunk_num: int | None = None
    end_chunk_num: int | None = None


_NOTHING = _FakePacket()  # what a bare receive() timeout looks like: no Interest, no Data


class _FakeClock:
    """A monotonic-clock stand-in that only advances when told to.

    `tick()` is called from `_FakeHandle.receive()` when a handle is wired
    to one (mirrors cefpyco's own semindex_smoke test doubles): a real
    receive() call costs up to its own timeout_ms of wall-clock time, so a
    fake handle advancing this clock by that same amount per call is what
    lets a deadline-based test resolve in a bounded number of iterations
    without actually sleeping.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def tick(self, step: float) -> None:
        self.value += step


class _FakeHandle:
    """A controllable stand-in for CefpycoTransport's CefpycoHandle protocol.

    `packets` is drained by receive(), oldest first; once empty, every
    further receive() returns _NOTHING, matching a real handle when
    nothing arrives within its own timeout_ms. If `clock` is given,
    receive() advances it by `step_s` on every call, so a test that waits
    out a deadline via CefpycoTransport's own internal polling loop
    (fetch_content) terminates without an actual sleep.
    """

    def __init__(self, packets=None, *, clock: _FakeClock | None = None, step_s: float = 0.1) -> None:
        self.packets = list(packets or [])
        self._clock = clock
        self._step_s = step_s
        self.registered: list[str] = []
        self.sent_interests: list[tuple[str, int, int]] = []  # (name, chunk_num, lifetime)
        self.sent_data: list[tuple[str, bytes, int, int, int]] = []  # (name, payload, chunk_num, expiry, cache_time)

    def register(self, name: str) -> None:
        """Record that `name` was registered."""
        self.registered.append(name)

    def send_interest(self, name: str, chunk_num: int, lifetime: int) -> None:
        """Record the Interest that was sent; no delivery happens here."""
        self.sent_interests.append((name, chunk_num, lifetime))

    def send_data(self, name: str, payload: bytes, chunk_num: int, expiry: int, cache_time: int) -> None:
        """Record the Data that was sent."""
        self.sent_data.append((name, payload, chunk_num, expiry, cache_time))

    def receive(self, timeout_ms: int) -> _FakePacket:
        """Pop and return the next queued packet, or _NOTHING once the queue is empty."""
        if self._clock is not None:
            self._clock.tick(self._step_s)
        if self.packets:
            return self.packets.pop(0)
        return _NOTHING


class _StubEngine:
    """A minimal SyncEngine double: records every call and returns whatever the test configured."""

    def __init__(self) -> None:
        self.handle_interest_calls: list[str] = []
        self.handle_interest_result = None
        self.on_state_calls: list[tuple[str, bytes, float]] = []
        self.on_record_calls: list[tuple[str, bytes, float]] = []
        self.on_fetch_timeout_calls: list[tuple[str, float]] = []

    def handle_interest(self, name: str):
        """Record the call and either return the configured Response/None, or raise the configured exception."""
        self.handle_interest_calls.append(name)
        if isinstance(self.handle_interest_result, BaseException):
            raise self.handle_interest_result
        return self.handle_interest_result

    def on_state(self, neighbor: str, payload: bytes, now: float) -> None:
        """Record the call."""
        self.on_state_calls.append((neighbor, payload, now))

    def on_record(self, name: str, payload: bytes, now: float) -> None:
        """Record the call."""
        self.on_record_calls.append((name, payload, now))

    def on_fetch_timeout(self, name: str, now: float) -> None:
        """Record the call."""
        self.on_fetch_timeout_calls.append((name, now))


def _record_payload(origin: str = "h0", seq: int = 3) -> bytes:
    """Build one encoded, well-formed record, for tests that only need a valid Data payload."""
    record = schema.IndexRecord(
        origin=origin,
        seq=seq,
        content_name=f"ccnx:/content/{origin}/{seq}",
        zone="zone1",
        disaster="flood",
        media="image",
        observed_at=1,
    )
    return schema.encode(record)


def test_register_delegates_to_the_handle():
    """register() is a pass-through to handle.register."""
    handle = _FakeHandle()
    transport = CefpycoTransport(handle, _StubEngine())
    transport.register("ccnx:/semidx/g/h0")
    assert handle.registered == ["ccnx:/semidx/g/h0"]


def test_send_interest_sends_without_a_chunk_tlv_and_counts_the_name_bytes():
    """send_interest passes chunk_num=-1 (no chunk TLV) and counts the name's UTF-8 length as sent."""
    name = "ccnx:/semidx/g/h1/sv/h0/1000"
    handle = _FakeHandle()
    transport = CefpycoTransport(handle, _StubEngine(), clock=_FakeClock(100.0))
    transport.send_interest(name, 2000)
    assert handle.sent_interests == [(name, -1, 2000)]
    assert transport.bytes_sent == len(name.encode("utf-8"))


def test_poll_dispatches_an_incoming_interest_to_a_response_as_data():
    """poll() answers an incoming Interest whose handle_interest call returns a Response, as Data."""
    name = "ccnx:/semidx/g/h0/rec/h0/1"
    handle = _FakeHandle([_FakePacket(is_interest=True, name=name)])
    engine = _StubEngine()
    engine.handle_interest_result = Response(payload=b'{"x":1}', expiry_s=3600.0, cache_time_s=300.0)
    transport = CefpycoTransport(handle, engine, clock=_FakeClock(0.0))
    transport.poll()
    assert engine.handle_interest_calls == [name]
    assert handle.sent_data == [(name, b'{"x":1}', -1, 3600000, 300000)]
    assert transport.bytes_sent == len(name.encode("utf-8")) + len(b'{"x":1}')
    assert transport.bytes_received == len(name.encode("utf-8"))


def test_poll_lets_an_interest_expire_when_handle_interest_returns_none():
    """No Response (unparsable name, wrong holder, fault.drop_rec, record not held) sends no Data."""
    handle = _FakeHandle([_FakePacket(is_interest=True, name="ccnx:/semidx/g/h0/rec/h0/9")])
    transport = CefpycoTransport(handle, _StubEngine(), clock=_FakeClock(0.0))
    transport.poll()
    assert handle.sent_data == []


def test_poll_catches_a_holder_state_error_from_handle_interest_and_lets_the_interest_expire():
    """holder_state.encode can raise HolderStateError when it cannot fit even with every
    origin's extra truncated away (map.md "ページングは無し"); this must not crash the loop
    that also serves every other neighbor.
    """
    handle = _FakeHandle([_FakePacket(is_interest=True, name="ccnx:/semidx/g/h0/sv/h1/5")])
    engine = _StubEngine()
    engine.handle_interest_result = holder_state.HolderStateError("too big")
    transport = CefpycoTransport(handle, engine, clock=_FakeClock(0.0))
    transport.poll()  # must not raise
    assert handle.sent_data == []


def test_poll_routes_data_with_no_payload_tlv_without_crashing():
    """cefpyco's CcnPacketInfo.payload is None for a Data TLV with no payload at all
    (its own payload_s property guards the same way); len(None) must not reach
    _route_data's byte-counting and crash the loop.
    """
    name = "ccnx:/semidx/g/h1/rec/h0/3"
    handle = _FakeHandle([_FakePacket(is_data=True, name=name, payload=None)])
    engine = _StubEngine()
    transport = CefpycoTransport(handle, engine, clock=_FakeClock(5.0))
    transport.poll()  # must not raise
    assert engine.on_record_calls == [(name, b"", 5.0)]
    assert transport.bytes_received == len(name.encode("utf-8"))


def test_poll_refuses_to_send_a_response_payload_over_the_wire_limit_for_its_kind():
    """A record Response over schema.MAX_RECORD_BYTES is dropped rather than handed to
    send_data -- the last line of defense map.md calls out (cefpyco's send_data itself
    does not check payload length).
    """
    name = "ccnx:/semidx/g/h0/rec/h0/1"
    oversized = b"x" * (schema.MAX_RECORD_BYTES + 1)
    handle = _FakeHandle([_FakePacket(is_interest=True, name=name)])
    engine = _StubEngine()
    engine.handle_interest_result = Response(payload=oversized, expiry_s=3600.0, cache_time_s=300.0)
    transport = CefpycoTransport(handle, engine, clock=_FakeClock(0.0))
    transport.poll()
    assert handle.sent_data == []


def test_poll_refuses_a_state_response_payload_over_the_wire_limit_for_its_kind():
    """The same wire-limit refusal as the record case above, but for a state name --
    exercises the *other* branch of the isinstance(parsed, names.StateName) limit
    selection, which the record-only test above cannot: a regression that always
    applied schema.MAX_RECORD_BYTES (2 KiB) regardless of kind would still pass
    that test but must fail this one.
    """
    name = "ccnx:/semidx/g/h0/sv/h1/5"
    oversized = b"x" * (holder_state.MAX_ENCODED_BYTES + 1)
    handle = _FakeHandle([_FakePacket(is_interest=True, name=name)])
    engine = _StubEngine()
    engine.handle_interest_result = Response(payload=oversized, expiry_s=1.0, cache_time_s=0.0)
    transport = CefpycoTransport(handle, engine, clock=_FakeClock(0.0))
    transport.poll()
    assert handle.sent_data == []


def test_poll_sends_a_state_response_payload_between_the_record_and_state_limits():
    """A state Response over schema.MAX_RECORD_BYTES but within
    holder_state.MAX_ENCODED_BYTES is sent, not dropped -- the companion to the
    test above. Together the two prove the limit selection actually picks the
    *state* limit for a state name rather than always applying the smaller
    *record* one (which this test alone would catch) or always applying the
    larger *state* one (which the test above alone would catch).
    """
    name = "ccnx:/semidx/g/h0/sv/h1/5"
    payload = b"x" * (schema.MAX_RECORD_BYTES + 1)
    assert len(payload) <= holder_state.MAX_ENCODED_BYTES
    handle = _FakeHandle([_FakePacket(is_interest=True, name=name)])
    engine = _StubEngine()
    engine.handle_interest_result = Response(payload=payload, expiry_s=1.0, cache_time_s=0.0)
    transport = CefpycoTransport(handle, engine, clock=_FakeClock(0.0))
    transport.poll()
    assert handle.sent_data == [(name, payload, -1, 1000, 0)]


def test_poll_routes_state_data_to_on_state_with_the_holder_from_the_name():
    """A Data reply for a state name is routed to on_state with the holder segment as the neighbor."""
    name = "ccnx:/semidx/g/h1/sv/h0/1000"
    payload = holder_state.encode({})
    handle = _FakeHandle([_FakePacket(is_data=True, name=name, payload=payload)])
    engine = _StubEngine()
    transport = CefpycoTransport(handle, engine, clock=_FakeClock(5.0))
    transport.poll()
    assert engine.on_state_calls == [("h1", payload, 5.0)]
    assert transport.bytes_received == len(name.encode("utf-8")) + len(payload)


def test_poll_routes_record_data_to_on_record_with_the_full_name():
    """A Data reply for a record name is routed to on_record with the name itself
    (on_record derives the neighbor from the name's own holder segment).
    """
    name = "ccnx:/semidx/g/h1/rec/h0/3"
    payload = _record_payload()
    handle = _FakeHandle([_FakePacket(is_data=True, name=name, payload=payload)])
    engine = _StubEngine()
    transport = CefpycoTransport(handle, engine, clock=_FakeClock(5.0))
    transport.poll()
    assert engine.on_record_calls == [(name, payload, 5.0)]


def test_send_interest_then_poll_past_the_deadline_with_no_data_reports_a_timeout_exactly_once():
    """An Interest whose lifetime elapses with no Data calls the engine's timeout path exactly once."""
    name = "ccnx:/semidx/g/h1/rec/h0/3"
    handle = _FakeHandle()
    engine = _StubEngine()
    clock = _FakeClock(0.0)
    transport = CefpycoTransport(handle, engine, clock=clock)
    transport.send_interest(name, 2000)  # lifetime_ms=2000 -> deadline at clock=2.0
    clock.value = 1.0
    transport.poll()
    assert engine.on_fetch_timeout_calls == []
    clock.value = 2.5
    transport.poll()
    assert engine.on_fetch_timeout_calls == [(name, 2.5)]
    # The deadline is consumed: a further poll at a later time must not re-report it.
    clock.value = 3.0
    transport.poll()
    assert engine.on_fetch_timeout_calls == [(name, 2.5)]


def test_late_data_after_the_deadline_already_fired_is_still_routed_not_dropped():
    """Data arriving after its own deadline already fired is still a valid reply: the
    tracker already released that fetch's in-flight slot on the timeout, so a late
    on_record call is exactly as valid as an on-time one.
    """
    name = "ccnx:/semidx/g/h1/rec/h0/3"
    payload = _record_payload()
    handle = _FakeHandle()
    engine = _StubEngine()
    clock = _FakeClock(0.0)
    transport = CefpycoTransport(handle, engine, clock=clock)
    transport.send_interest(name, 1000)  # deadline at clock=1.0
    clock.value = 1.5
    transport.poll()  # nothing has arrived yet -> this is the timeout
    assert engine.on_fetch_timeout_calls == [(name, 1.5)]
    handle.packets.append(_FakePacket(is_data=True, name=name, payload=payload))
    clock.value = 1.6
    transport.poll()
    assert engine.on_record_calls == [(name, payload, 1.6)]


def test_fetch_content_reassembles_every_chunk_and_stops_at_end_chunk_num():
    """Fetches chunk 0..end one Interest at a time and counts bytes both ways."""
    name = "ccnx:/content/h0/body"
    clock = _FakeClock(0.0)
    handle = _FakeHandle(
        [
            _FakePacket(is_data=True, name=name, payload=b"hello-", chunk_num=0, end_chunk_num=None),
            _FakePacket(is_data=True, name=name, payload=b"world!", chunk_num=1, end_chunk_num=1),
        ],
        clock=clock,
    )
    transport = CefpycoTransport(handle, _StubEngine(), clock=clock)
    ok, total = transport.fetch_content(name, chunk_lifetime_ms=2000)
    assert (ok, total) == (True, len(b"hello-world!"))
    assert [chunk_num for (_name, chunk_num, _lifetime) in handle.sent_interests] == [0, 1]


def test_fetch_content_fails_if_a_chunk_never_arrives_within_its_lifetime():
    """A chunk whose Interest lifetime elapses with no matching Data fails the fetch."""
    name = "ccnx:/content/h0/body"
    clock = _FakeClock(0.0)
    handle = _FakeHandle([], clock=clock, step_s=0.2)
    transport = CefpycoTransport(handle, _StubEngine(), clock=clock)
    ok, total = transport.fetch_content(name, chunk_lifetime_ms=500)
    assert (ok, total) == (False, 0)


def test_fetch_content_dispatches_sync_traffic_seen_while_waiting_for_a_chunk():
    """A neighbor Interest arriving mid-fetch is still served, not dropped: cefpyco's
    receive buffer is a single process-wide static one, so a packet not read here is
    gone, not merely delayed (module docstring).
    """
    content_name = "ccnx:/content/h0/body"
    sync_name = "ccnx:/semidx/g/h0/rec/h0/1"
    clock = _FakeClock(0.0)
    handle = _FakeHandle(
        [
            _FakePacket(is_interest=True, name=sync_name),
            _FakePacket(is_data=True, name=content_name, payload=b"x", chunk_num=0, end_chunk_num=0),
        ],
        clock=clock,
    )
    engine = _StubEngine()
    engine.handle_interest_result = Response(payload=b"{}", expiry_s=1.0, cache_time_s=0.0)
    transport = CefpycoTransport(handle, engine, clock=clock)
    ok, total = transport.fetch_content(content_name, chunk_lifetime_ms=2000)
    assert (ok, total) == (True, 1)
    assert engine.handle_interest_calls == [sync_name]
    assert handle.sent_data == [(sync_name, b"{}", -1, 1000, 0)]


def test_fetch_content_treats_a_chunk_with_no_payload_tlv_as_empty_not_a_crash():
    """The same payload=None case as the poll()/on_record test above, but on
    fetch_content's own chunk-matching path: a chunk Data with no payload TLV
    (chunk_num present, payload None) must count as 0 bytes, not raise.
    """
    name = "ccnx:/content/h0/body"
    clock = _FakeClock(0.0)
    handle = _FakeHandle(
        [_FakePacket(is_data=True, name=name, payload=None, chunk_num=0, end_chunk_num=0)],
        clock=clock,
    )
    transport = CefpycoTransport(handle, _StubEngine(), clock=clock)
    ok, total = transport.fetch_content(name, chunk_lifetime_ms=2000)
    assert (ok, total) == (True, 0)


def test_fetch_content_stops_promptly_when_should_stop_becomes_true_mid_wait():
    """A chunk wait that would otherwise burn its full chunk_lifetime_ms exits as
    soon as should_stop() reports true, instead of only being noticed on the agent
    loop's next step() -- the SIGTERM-while-mid-fetch case ticket requirement 5's
    well-under-2s shutdown budget depends on (map.md's T1 partition-phase probe is
    exactly this: content that is unreachable by design).
    """
    name = "ccnx:/content/h0/body"
    clock = _FakeClock(0.0)
    handle = _FakeHandle([], clock=clock, step_s=0.1)
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] >= 3  # let one receive() happen first, then stop mid-wait

    transport = CefpycoTransport(handle, _StubEngine(), clock=clock, should_stop=should_stop)
    ok, total = transport.fetch_content(name, chunk_lifetime_ms=2000)
    assert (ok, total) == (False, 0)
    # Stopped after one 0.1s receive tick, nowhere near the 2s chunk_lifetime_ms deadline.
    assert clock.value < 1.0
