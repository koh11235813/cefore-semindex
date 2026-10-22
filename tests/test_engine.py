"""Tests for SyncEngine against a bare recording Transport (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from semindex import holder_state, names, schema
from semindex.engine import SyncConfig, SyncEngine
from semindex.store import OriginProgress, Store


class RecordingTransport:
    """A Transport that just remembers what was sent; no delivery at all."""

    def __init__(self) -> None:
        self.registered: list[str] = []
        self.sent: list[tuple[str, int]] = []

    def register(self, prefix: str) -> None:
        self.registered.append(prefix)

    def send_interest(self, name: str, lifetime_ms: int) -> None:
        self.sent.append((name, lifetime_ms))


def _engine(tmp_path: Path, node: str = "h0", neighbors: tuple[str, ...] = ("h1",), **overrides) -> tuple[SyncEngine, RecordingTransport, Store]:
    store = Store(tmp_path / f"{node}.db")
    transport = RecordingTransport()
    config = SyncConfig(group="g", node=node, neighbors=neighbors, **overrides)
    engine = SyncEngine(config, store, transport)
    return engine, transport, store


def test_registers_its_own_prefix_on_construction(tmp_path: Path):
    _, transport, _ = _engine(tmp_path)
    assert transport.registered == ["ccnx:/semidx/g/h0"]


def test_construction_rejects_a_neighbor_containing_a_slash(tmp_path: Path):
    """2026-10-22 regression: a malformed neighbor used to pass
    unchecked through __init__ and only surface once some later
    on_tick tried to build a name with it -- and then with a
    "holder must be..." diagnostic that names the wrong thing from
    this config's point of view. It must now fail at construction,
    before anything is registered, with a diagnostic that names the
    neighbor.
    """
    store = Store(tmp_path / "x.db")
    config = SyncConfig(group="g", node="h0", neighbors=("h1/evil",))
    transport = RecordingTransport()
    with pytest.raises(names.MalformedNameError) as exc_info:
        SyncEngine(config, store, transport)
    assert str(exc_info.value) == "neighbor must be a non-empty name component without '/'"
    assert transport.registered == []


def test_create_record_allocates_sequential_seqs_and_emits_events(tmp_path: Path):
    engine, _, _ = _engine(tmp_path)
    r1 = engine.create_record(content_name="a", zone="zone1", disaster="flood", media="image", observed_at=1)
    r2 = engine.create_record(content_name="b", zone="zone1", disaster="flood", media="image", observed_at=2)
    assert (r1.seq, r2.seq) == (1, 2)
    assert engine.drain_events() == [
        {"event": "record_created", "origin": "h0", "seq": 1},
        {"event": "record_created", "origin": "h0", "seq": 2},
    ]


def test_on_record_rejects_name_payload_mismatch(tmp_path: Path):
    engine, _, store = _engine(tmp_path)
    record = schema.IndexRecord(
        origin="h1", seq=1, content_name="x", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    payload = schema.encode(record)
    # Name claims seq 2, payload says seq 1.
    mismatched_name = names.build_record_name("g", "h0", "h1", 2)
    engine.on_record(mismatched_name, payload, now=0.0)
    assert store.get_record("h1", 1) is None
    assert store.get_record("h1", 2) is None
    assert engine.drain_events() == []


def test_on_record_releases_the_fetch_for_retry_on_name_payload_mismatch(tmp_path: Path):
    # 2026-10-22 fix regression: a mismatched reply used to just be
    # dropped, leaving the fetch marked in-flight forever (no
    # on_fetch_timeout is ever coming for an Interest that already got
    # a Data reply). It must instead back off and let the next on_tick
    # retry it, exactly like a timeout would.
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",))
    engine._tracker.observe_state("origin-x", "h1", hi=1, extra=())
    engine.on_tick(now=0.0)  # claims the fetch for origin-x/1 against h1
    fetch_name = names.build_record_name("g", "h1", "origin-x", 1)
    assert fetch_name in [n for n, _ in transport.sent]
    transport.sent.clear()

    record = schema.IndexRecord(
        origin="origin-x", seq=2, content_name="x", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    mismatched_payload = schema.encode(record)  # payload says seq 2, but fetch_name says seq 1
    engine.on_record(fetch_name, mismatched_payload, now=0.0)

    # The state poll is not due again yet (interval is 2.0s by
    # default); only the released fetch retry should fire.
    engine.on_tick(now=1.0)
    assert transport.sent == [(fetch_name, engine._config.interest_lifetime_ms)]


def test_on_record_releases_the_fetch_for_retry_on_malformed_payload(tmp_path: Path):
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",))
    engine._tracker.observe_state("origin-x", "h1", hi=1, extra=())
    engine.on_tick(now=0.0)
    fetch_name = names.build_record_name("g", "h1", "origin-x", 1)
    assert fetch_name in [n for n, _ in transport.sent]
    transport.sent.clear()

    engine.on_record(fetch_name, b"not json", now=0.0)

    engine.on_tick(now=1.0)
    assert transport.sent == [(fetch_name, engine._config.interest_lifetime_ms)]


def test_on_record_stores_a_valid_record_and_emits_record_stored(tmp_path: Path):
    engine, _, store = _engine(tmp_path)
    record = schema.IndexRecord(
        origin="h1", seq=1, content_name="x", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    payload = schema.encode(record)
    name = names.build_record_name("g", "h1", "h1", 1)
    engine.on_record(name, payload, now=0.0)
    assert store.get_record("h1", 1) == payload
    assert engine.drain_events() == [{"event": "record_stored", "origin": "h1", "seq": 1, "holder": "h1"}]


def test_on_record_duplicate_is_idempotent_and_silent(tmp_path: Path):
    engine, _, _ = _engine(tmp_path)
    record = schema.IndexRecord(
        origin="h1", seq=1, content_name="x", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    payload = schema.encode(record)
    name = names.build_record_name("g", "h0", "h1", 1)
    engine.on_record(name, payload, now=0.0)
    engine.drain_events()
    engine.on_record(name, payload, now=1.0)
    assert engine.drain_events() == []


def test_on_record_conflict_keeps_first_and_emits_conflict_event(tmp_path: Path):
    engine, _, store = _engine(tmp_path)
    name = names.build_record_name("g", "h0", "h1", 1)
    record_a = schema.IndexRecord(
        origin="h1", seq=1, content_name="a", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    payload_a = schema.encode(record_a)
    engine.on_record(name, payload_a, now=0.0)
    engine.drain_events()

    record_b = schema.IndexRecord(
        origin="h1", seq=1, content_name="b", zone="zone2", disaster="fire", media="video", observed_at=2
    )
    payload_b = schema.encode(record_b)
    engine.on_record(name, payload_b, now=1.0)

    events = engine.drain_events()
    assert events == [
        {
            "event": "conflict",
            "origin": "h1",
            "seq": 1,
            "kept_hash": schema.content_hash(payload_a),
            "rejected_hash": schema.content_hash(payload_b),
        }
    ]
    assert store.get_record("h1", 1) == payload_a


def test_handle_interest_serves_own_state(tmp_path: Path):
    engine, _, _ = _engine(tmp_path)
    engine.create_record(content_name="a", zone="zone1", disaster="flood", media="image", observed_at=1)
    name = names.build_state_name("g", "h0", "h1", 1700000000000)
    response = engine.handle_interest(name)
    assert response is not None
    decoded = holder_state.decode(response.payload)
    assert decoded == {"h0": {"hi": 1, "extra": [], "truncated": False}}


def test_handle_interest_serves_own_record(tmp_path: Path):
    engine, _, _ = _engine(tmp_path)
    created = engine.create_record(
        content_name="a", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    name = names.build_record_name("g", "h0", "h0", created.seq)
    response = engine.handle_interest(name)
    assert response is not None
    assert schema.decode(response.payload) == created


def test_handle_interest_returns_none_for_unheld_record(tmp_path: Path):
    engine, _, _ = _engine(tmp_path)
    name = names.build_record_name("g", "h0", "h0", 99)
    assert engine.handle_interest(name) is None


def test_handle_interest_returns_none_when_holder_segment_is_not_self(tmp_path: Path):
    """2026-10-22 regression: this used to ask for (h0, 1) without the
    engine ever having created it, so the test passed even with the
    holder check deleted -- the "not held" branch alone already
    returns None for an unheld record, regardless of the holder
    segment. Holding the record this engine is asked about isolates
    the holder check itself.
    """
    engine, _, _ = _engine(tmp_path)
    created = engine.create_record(
        content_name="a", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    name = names.build_record_name("g", "someone-else", "h0", created.seq)
    assert engine.handle_interest(name) is None


def test_handle_interest_returns_none_for_a_state_request_not_addressed_to_self(tmp_path: Path):
    """The state branch has its own holder check, mirroring the record
    branch's; nothing elsewhere exercises holder != self for a state
    request, only holder == self.
    """
    engine, _, _ = _engine(tmp_path)
    name = names.build_state_name("g", "someone-else", "h1", 1700000000000)
    assert engine.handle_interest(name) is None


def test_fault_drop_rec_withholds_then_serves(tmp_path: Path):
    engine, _, _ = _engine(tmp_path, fault_drop_rec={("h0", 1): 2})
    created = engine.create_record(
        content_name="a", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    assert created.seq == 1
    name = names.build_record_name("g", "h0", "h0", 1)
    assert engine.handle_interest(name) is None  # drop 1 of 2
    assert engine.handle_interest(name) is None  # drop 2 of 2
    response = engine.handle_interest(name)  # count exhausted; now served
    assert response is not None
    assert schema.decode(response.payload) == created


def test_on_tick_sends_a_state_request_to_every_neighbor(tmp_path: Path):
    engine, transport, _ = _engine(tmp_path, neighbors=("h1", "h2"))
    engine.on_tick(now=0.0)
    requesters = [names.parse(n).requester for n, _ in transport.sent]
    holders = [names.parse(n).holder for n, _ in transport.sent]
    assert holders == ["h1", "h2"]
    assert requesters == ["h0", "h0"]


def test_on_tick_does_not_resend_state_request_before_next_poll_is_due(tmp_path: Path):
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",), state_poll_interval_s=2.0)
    engine.on_tick(now=0.0)
    assert len(transport.sent) == 1
    engine.on_tick(now=1.0)
    assert len(transport.sent) == 1  # not due yet (interval is 2.0s, no jitter by default)
    engine.on_tick(now=2.0)
    assert len(transport.sent) == 2


def test_on_state_decodes_and_feeds_the_tracker_then_plans_fetches(tmp_path: Path):
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",))
    payload = holder_state.encode({"origin-x": OriginProgress(hi=2, extra=())})
    engine.on_state("h1", payload, now=0.0)
    events = engine.drain_events()
    assert events[0] == {"event": "state_rx", "neighbor": "h1"}
    sent_names = [n for n, _ in transport.sent]
    assert sent_names == [
        names.build_record_name("g", "h1", "origin-x", 1),
        names.build_record_name("g", "h1", "origin-x", 2),
    ]


def test_on_state_drops_malformed_payload_silently(tmp_path: Path):
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",))
    engine.on_state("h1", b"not json", now=0.0)
    events = engine.drain_events()
    assert events == [{"event": "state_rx", "neighbor": "h1"}]
    assert transport.sent == []


def test_on_state_drops_payload_with_a_malformed_origin_without_crashing(tmp_path: Path):
    # 2026-10-22 fix regression: holder_state.decode used to accept any
    # non-empty origin key, including one containing '/'. That reached
    # names.build_record_name while planning a fetch for it and raised
    # MalformedNameError uncaught, escaping on_state entirely. Decoding
    # must now reject the whole state instead, the same as any other
    # malformed reply.
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",))
    payload = b'{"bad/origin":{"extra":[],"hi":1,"truncated":false}}'
    engine.on_state("h1", payload, now=0.0)  # must not raise
    assert engine.drain_events() == [{"event": "state_rx", "neighbor": "h1"}]
    assert transport.sent == []


def test_on_fetch_timeout_emits_event_and_backs_off(tmp_path: Path):
    engine, _, _ = _engine(tmp_path, neighbors=("h1",))
    name = names.build_record_name("g", "h1", "origin-x", 3)
    engine.on_fetch_timeout(name, now=0.0)
    assert engine.drain_events() == [
        {"event": "fetch_timeout", "origin": "origin-x", "seq": 3, "neighbor": "h1"}
    ]


def test_on_fetch_timeout_releases_the_fetch_for_retry(tmp_path: Path):
    """2026-10-22 regression: before the neighbor was derived from the
    name's holder segment, a caller passing a neighbor that disagreed
    with it left the fetch wedged in-flight forever, since
    Tracker._fail no-ops on a mismatched neighbor and no further
    timeout is ever coming for an Interest already resolved. With the
    separate argument gone that mismatch cannot be constructed; this
    pins that a real timeout still frees the slot so the next on_tick
    retries it, instead of leaving it stuck.
    """
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",))
    engine._tracker.observe_state("origin-x", "h1", hi=1, extra=())
    engine.on_tick(now=0.0)  # claims the fetch for origin-x/1 against h1
    fetch_name = names.build_record_name("g", "h1", "origin-x", 1)
    assert fetch_name in [n for n, _ in transport.sent]
    transport.sent.clear()

    engine.on_fetch_timeout(fetch_name, now=0.0)

    # The state poll is not due again yet; only the released fetch
    # retry (backoff_initial_s defaults to 1.0) should fire at now=1.0.
    engine.on_tick(now=1.0)
    assert transport.sent == [(fetch_name, engine._config.interest_lifetime_ms)]
