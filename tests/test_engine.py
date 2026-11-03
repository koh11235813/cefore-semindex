"""Tests for SyncEngine against a bare recording Transport (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from semindex import holder_state, names, schema
from semindex.engine import StatePollPolicy, SyncConfig, SyncEngine, _PollReplies
from semindex.store import OriginProgress, Store


class RecordingTransport:
    """A Transport that just remembers what was sent; no delivery at all."""

    def __init__(self) -> None:
        """Start with nothing registered and nothing sent."""
        self.registered: list[str] = []
        self.sent: list[tuple[str, int]] = []

    def register(self, prefix: str) -> None:
        """Remember `prefix` was registered."""
        self.registered.append(prefix)

    def send_interest(self, name: str, lifetime_ms: int) -> None:
        """Remember `name` and `lifetime_ms` were sent; never resolves it."""
        self.sent.append((name, lifetime_ms))


def _engine(tmp_path: Path, node: str = "h0", neighbors: tuple[str, ...] = ("h1",), **overrides) -> tuple[SyncEngine, RecordingTransport, Store]:
    """Build a bare SyncEngine wired to a RecordingTransport, for tests that only check what it sends."""
    store = Store(tmp_path / f"{node}.db")
    transport = RecordingTransport()
    config = SyncConfig(group="g", node=node, neighbors=neighbors, **overrides)
    engine = SyncEngine(config, store, transport)
    return engine, transport, store


def test_registers_its_own_prefix_on_construction(tmp_path: Path):
    """Construction registers exactly the node's own prefix, once."""
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
    """Each create_record call allocates the next seq for this node's own origin and emits record_created."""
    engine, _, _ = _engine(tmp_path)
    r1 = engine.create_record(content_name="a", zone="zone1", disaster="flood", media="image", observed_at=1)
    r2 = engine.create_record(content_name="b", zone="zone1", disaster="flood", media="image", observed_at=2)
    assert (r1.seq, r2.seq) == (1, 2)
    assert engine.drain_events() == [
        {"event": "record_created", "ts_ms": 1, "origin": "h0", "seq": 1},
        {"event": "record_created", "ts_ms": 2, "origin": "h0", "seq": 2},
    ]


def test_on_record_rejects_name_payload_mismatch(tmp_path: Path):
    """A reply whose payload's (origin, seq) disagrees with the name's is rejected without being stored."""
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
    """A name/payload mismatch still releases the in-flight fetch, so the next on_tick retries it."""
    # 2026-10-22 fix regression: a mismatched reply used to just be
    # dropped, leaving the fetch marked in-flight forever (no
    # on_fetch_timeout is ever coming for an Interest that already got
    # a Data reply). It must instead back off and let the next on_tick
    # retry it, exactly like a timeout would.
    # fetch_backoff_initial_s is pinned to 1.0: this test is about the
    # mismatch-triggers-retry path, not the live backoff default.
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",), fetch_backoff_initial_s=1.0)
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
    """Malformed JSON in a reply still releases the in-flight fetch, so the next on_tick retries it."""
    # fetch_backoff_initial_s is pinned to 1.0: this test is about the
    # malformed-payload-triggers-retry path, not the live backoff default.
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",), fetch_backoff_initial_s=1.0)
    engine._tracker.observe_state("origin-x", "h1", hi=1, extra=())
    engine.on_tick(now=0.0)
    fetch_name = names.build_record_name("g", "h1", "origin-x", 1)
    assert fetch_name in [n for n, _ in transport.sent]
    transport.sent.clear()

    engine.on_record(fetch_name, b"not json", now=0.0)

    engine.on_tick(now=1.0)
    assert transport.sent == [(fetch_name, engine._config.interest_lifetime_ms)]


def test_on_record_stores_a_valid_record_and_emits_record_stored(tmp_path: Path):
    """A valid reply is stored and emits record_stored naming the holder it came from."""
    engine, _, store = _engine(tmp_path)
    record = schema.IndexRecord(
        origin="h1", seq=1, content_name="x", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    payload = schema.encode(record)
    name = names.build_record_name("g", "h1", "h1", 1)
    engine.on_record(name, payload, now=0.0)
    assert store.get_record("h1", 1) == payload
    assert engine.drain_events() == [{"event": "record_stored", "ts_ms": 0, "origin": "h1", "seq": 1, "holder": "h1"}]


def test_on_record_duplicate_is_idempotent_and_silent(tmp_path: Path):
    """Re-receiving the exact same record a second time stores nothing and emits no event."""
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
    """A second record at the same (origin, seq) with different content is kept out, and the conflict is reported with both hashes."""
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
            "ts_ms": 1000,
            "origin": "h1",
            "seq": 1,
            "kept_hash": schema.content_hash(payload_a),
            "rejected_hash": schema.content_hash(payload_b),
        }
    ]
    assert store.get_record("h1", 1) == payload_a


def test_handle_interest_serves_own_state(tmp_path: Path):
    """Serving a self-addressed state request returns this node's own progress, encoded."""
    engine, _, _ = _engine(tmp_path)
    engine.create_record(content_name="a", zone="zone1", disaster="flood", media="image", observed_at=1)
    name = names.build_state_name("g", "h0", "h1", 1700000000000)
    response = engine.handle_interest(name)
    assert response is not None
    decoded = holder_state.decode(response.payload)
    assert decoded == {"h0": {"hi": 1, "extra": [], "truncated": False}}


def test_handle_interest_serves_own_record(tmp_path: Path):
    """Serving a self-addressed record request returns exactly the record this node created."""
    engine, _, _ = _engine(tmp_path)
    created = engine.create_record(
        content_name="a", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    name = names.build_record_name("g", "h0", "h0", created.seq)
    response = engine.handle_interest(name)
    assert response is not None
    assert schema.decode(response.payload) == created


def test_handle_interest_returns_none_for_unheld_record(tmp_path: Path):
    """Asking for a record this node never created or received returns None."""
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
    """fault_drop_rec withholds exactly the configured number of replies before serving the record normally."""
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
    """on_tick sends one state request addressed to each configured neighbor."""
    engine, transport, _ = _engine(tmp_path, neighbors=("h1", "h2"))
    engine.on_tick(now=0.0)
    requesters = [names.parse(n).requester for n, _ in transport.sent]
    holders = [names.parse(n).holder for n, _ in transport.sent]
    assert holders == ["h1", "h2"]
    assert requesters == ["h0", "h0"]


def test_on_tick_does_not_resend_state_request_before_next_poll_is_due(tmp_path: Path):
    """A neighbor is not re-polled before its state_poll_interval_s has elapsed."""
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",), state_poll_interval_s=2.0)
    engine.on_tick(now=0.0)
    assert len(transport.sent) == 1
    engine.on_tick(now=1.0)
    assert len(transport.sent) == 1  # not due yet (interval is 2.0s, no jitter by default)
    engine.on_tick(now=2.0)
    assert len(transport.sent) == 2


def test_on_state_decodes_and_feeds_the_tracker_then_plans_fetches(tmp_path: Path):
    """A decoded state reply feeds the tracker and immediately queues fetch Interests for what it reveals as missing."""
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",))
    payload = holder_state.encode({"origin-x": OriginProgress(hi=2, extra=())})
    engine.on_state("h1", payload, now=0.0, poll_epoch_ms=0)
    events = engine.drain_events()
    assert events[0] == {"event": "state_rx", "ts_ms": 0, "neighbor": "h1", "hi": {"origin-x": 2}}
    sent_names = [n for n, _ in transport.sent]
    assert sent_names == [
        names.build_record_name("g", "h1", "origin-x", 1),
        names.build_record_name("g", "h1", "origin-x", 2),
    ]


def test_on_state_drops_malformed_payload_silently(tmp_path: Path):
    """A malformed state reply is dropped after the state_rx event, with no fetches planned from it."""
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",))
    engine.on_state("h1", b"not json", now=0.0, poll_epoch_ms=0)
    events = engine.drain_events()
    assert events == [{"event": "state_rx", "ts_ms": 0, "neighbor": "h1", "hi": {}}]
    assert transport.sent == []


def test_on_state_drops_payload_with_a_malformed_origin_without_crashing(tmp_path: Path):
    """A state reply naming a malformed origin is dropped like any other malformed reply, not propagated as an uncaught error."""
    # 2026-10-22 fix regression: holder_state.decode used to accept any
    # non-empty origin key, including one containing '/'. That reached
    # names.build_record_name while planning a fetch for it and raised
    # MalformedNameError uncaught, escaping on_state entirely. Decoding
    # must now reject the whole state instead, the same as any other
    # malformed reply.
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",))
    payload = b'{"bad/origin":{"extra":[],"hi":1,"truncated":false}}'
    engine.on_state("h1", payload, now=0.0, poll_epoch_ms=0)  # must not raise
    assert engine.drain_events() == [{"event": "state_rx", "ts_ms": 0, "neighbor": "h1", "hi": {}}]
    assert transport.sent == []


def test_on_fetch_timeout_emits_event_and_backs_off(tmp_path: Path):
    """on_fetch_timeout always emits a fetch_timeout event naming the origin, seq and neighbor from the name."""
    engine, _, _ = _engine(tmp_path, neighbors=("h1",))
    name = names.build_record_name("g", "h1", "origin-x", 3)
    engine.on_fetch_timeout(name, now=0.0)
    assert engine.drain_events() == [
        {"event": "fetch_timeout", "ts_ms": 0, "origin": "origin-x", "seq": 3, "neighbor": "h1"}
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
    # fetch_backoff_initial_s is pinned to 1.0: this test is about the
    # timeout-frees-the-slot-for-retry path, not the live backoff default.
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",), fetch_backoff_initial_s=1.0)
    engine._tracker.observe_state("origin-x", "h1", hi=1, extra=())
    engine.on_tick(now=0.0)  # claims the fetch for origin-x/1 against h1
    fetch_name = names.build_record_name("g", "h1", "origin-x", 1)
    assert fetch_name in [n for n, _ in transport.sent]
    transport.sent.clear()

    engine.on_fetch_timeout(fetch_name, now=0.0)

    # The state poll is not due again yet; only the released fetch
    # retry (fetch_backoff_initial_s pinned to 1.0 above) should fire at now=1.0.
    engine.on_tick(now=1.0)
    assert transport.sent == [(fetch_name, engine._config.interest_lifetime_ms)]


def test_default_fetch_backoff_initial_clears_cefores_pit_grace_window(tmp_path: Path):
    """Pins SyncConfig's own live default (2026-10-23: 2s, not 1s) -- the production value the
    agent actually runs with (engine.py passes config.fetch_backoff_initial_s straight to the
    Tracker; see tracker.py's identical pin of its own DEFAULT_BACKOFF_INITIAL_S). The first
    same-name retry after a fetch_timeout must not be planned before 2s: a 1s retry would land
    inside Cefore v0.12's own 1s PIT grace window after an expired Interest and be silently
    absorbed there, never reaching the producer at all."""
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",))  # no override: exercises SyncConfig's own default
    engine._tracker.observe_state("origin-x", "h1", hi=1, extra=())
    engine.on_tick(now=0.0)  # claims the fetch for origin-x/1 against h1
    fetch_name = names.build_record_name("g", "h1", "origin-x", 1)
    assert fetch_name in [n for n, _ in transport.sent]
    transport.sent.clear()

    engine.on_fetch_timeout(fetch_name, now=0.0)

    # Strictly after the 1s PIT grace: nothing is due yet at 1.0s (the
    # state poll isn't due again yet either -- its own interval is also
    # 2.0s by default).
    engine.on_tick(now=1.0)
    assert transport.sent == []

    engine.on_tick(now=2.0)
    assert fetch_name in [n for n, _ in transport.sent]


# -- decision events (map.md "判断の記録") -----------------------------------------


def test_on_tick_emits_a_state_poll_decision_per_due_neighbor(tmp_path: Path):
    """on_tick's state-poll send logs a `decision` event naming the polled neighbor and
    the jittered interval chosen for its next poll (action_prob 1.0: stage 1's policy
    is deterministic). With no tracker state at all, decision_state's `origins` is
    empty for both neighbors, and with no earlier poll last_poll_status is null."""
    engine, _, _ = _engine(tmp_path, neighbors=("h1", "h2"))  # default interval 2.0s, no jitter offset
    engine.on_tick(now=0.0)
    assert engine.drain_events() == [
        {
            "event": "decision",
            "ts_ms": 0,
            "kind": "state_poll",
            "state": {
                "in_flight": 0,
                "last_timeout_s": None,
                "truncated": False,
                "origins": {},
                "last_poll_status": None,
                "consecutive_poll_timeouts": 0,
            },
            "action": {"neighbor": "h1", "interval_s": 2.0},
            "action_prob": 1.0,
        },
        {
            "event": "decision",
            "ts_ms": 0,
            "kind": "state_poll",
            "state": {
                "in_flight": 0,
                "last_timeout_s": None,
                "truncated": False,
                "origins": {},
                "last_poll_status": None,
                "consecutive_poll_timeouts": 0,
            },
            "action": {"neighbor": "h2", "interval_s": 2.0},
            "action_prob": 1.0,
        },
    ]


def test_send_fetch_plan_emits_a_fetch_plan_decision_only_when_something_is_due(tmp_path: Path):
    """_send_fetch_plan logs a `decision` event only when it actually hands out new fetch
    Interests: on_tick calls this every receive-poll cycle (about 10/s live), so an empty
    plan -- the overwhelmingly common case -- must produce no line, or the log's size would
    track tick frequency rather than actual fetch activity."""
    engine, transport, _ = _engine(tmp_path, neighbors=("h1",))
    engine._tracker.observe_state("origin-x", "h1", hi=2, extra=())

    engine.on_tick(now=0.0)
    events = engine.drain_events()
    fetch_plan_decisions = [e for e in events if e["event"] == "decision" and e["kind"] == "fetch_plan"]
    assert fetch_plan_decisions == [
        {
            "event": "decision",
            "ts_ms": 0,
            "kind": "fetch_plan",
            "state": {
                "in_flight": 0,
                "last_timeout_s": None,
                "truncated": False,
                "origins": {"origin-x": {"hi": 0, "missing": 2, "advertised": {"h1": {"hi": 2, "notified_max": 2}}}},
            },
            "action": {"neighbor": "h1", "fetched": [["origin-x", 1], ["origin-x", 2]]},
            "action_prob": 1.0,
        }
    ]

    # Both seqs are now in flight and the state poll just fired too; nothing new is
    # due, so a second on_tick at the same instant logs and sends nothing at all.
    engine.on_tick(now=0.0)
    assert engine.drain_events() == []
    assert len(transport.sent) == 3  # 1 state request + 2 fetch Interests, from the first on_tick only


def test_on_fetch_timeout_emits_a_fetch_backoff_decision_for_a_real_in_flight_fetch(tmp_path: Path):
    """on_fetch_timeout logs a `fetch_backoff` decision giving the pre-failure state and the
    chosen wait, but only for a real in-flight fetch -- test_on_fetch_timeout_emits_event_and_backs_off's
    orphan name (never actually fetched) stays a bare fetch_timeout with no decision line,
    since Tracker._fail has nothing to resolve for it."""
    engine, _, _ = _engine(tmp_path, neighbors=("h1",), fetch_backoff_initial_s=3.0)
    engine._tracker.observe_state("origin-x", "h1", hi=1, extra=())
    engine.on_tick(now=0.0)  # claims the fetch for origin-x/1 against h1
    engine.drain_events()
    fetch_name = names.build_record_name("g", "h1", "origin-x", 1)

    engine.on_fetch_timeout(fetch_name, now=5.0)
    assert engine.drain_events() == [
        {"event": "fetch_timeout", "ts_ms": 5000, "origin": "origin-x", "seq": 1, "neighbor": "h1"},
        {
            "event": "decision",
            "ts_ms": 5000,
            "kind": "fetch_backoff",
            "state": {
                "in_flight": 1,
                "last_timeout_s": None,
                "truncated": False,
                "origins": {"origin-x": {"hi": 0, "missing": 1, "advertised": {"h1": {"hi": 1, "notified_max": 1}}}},
            },
            "action": {"neighbor": "h1", "origin": "origin-x", "seq": 1, "wait_s": 3.0, "cause": "timeout"},
            "action_prob": 1.0,
        },
    ]


def test_on_record_malformed_payload_emits_a_fetch_backoff_decision(tmp_path: Path):
    """A malformed reply resolves via Tracker.on_fetch_rejected, logged the same way as a
    timeout but with cause="rejected"."""
    engine, _, _ = _engine(tmp_path, neighbors=("h1",), fetch_backoff_initial_s=4.0)
    engine._tracker.observe_state("origin-x", "h1", hi=1, extra=())
    engine.on_tick(now=0.0)
    engine.drain_events()
    fetch_name = names.build_record_name("g", "h1", "origin-x", 1)

    engine.on_record(fetch_name, b"not json", now=2.0)
    assert engine.drain_events() == [
        {
            "event": "decision",
            "ts_ms": 2000,
            "kind": "fetch_backoff",
            "state": {
                "in_flight": 1,
                "last_timeout_s": None,
                "truncated": False,
                "origins": {"origin-x": {"hi": 0, "missing": 1, "advertised": {"h1": {"hi": 1, "notified_max": 1}}}},
            },
            "action": {"neighbor": "h1", "origin": "origin-x", "seq": 1, "wait_s": 4.0, "cause": "rejected"},
            "action_prob": 1.0,
        },
    ]


def test_on_record_name_payload_mismatch_emits_a_fetch_backoff_decision(tmp_path: Path):
    """A name/payload mismatch is the other on_fetch_rejected call site in on_record; it
    logs the same decision shape as the malformed-payload case."""
    engine, _, _ = _engine(tmp_path, neighbors=("h1",), fetch_backoff_initial_s=1.5)
    engine._tracker.observe_state("origin-x", "h1", hi=1, extra=())
    engine.on_tick(now=0.0)
    engine.drain_events()
    fetch_name = names.build_record_name("g", "h1", "origin-x", 1)  # claims seq 1

    record = schema.IndexRecord(
        origin="origin-x", seq=2, content_name="x", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    mismatched_payload = schema.encode(record)  # payload says seq 2, name says seq 1
    engine.on_record(fetch_name, mismatched_payload, now=1.0)

    decisions = [e for e in engine.drain_events() if e["event"] == "decision"]
    assert decisions == [
        {
            "event": "decision",
            "ts_ms": 1000,
            "kind": "fetch_backoff",
            "state": {
                "in_flight": 1,
                "last_timeout_s": None,
                "truncated": False,
                "origins": {"origin-x": {"hi": 0, "missing": 1, "advertised": {"h1": {"hi": 1, "notified_max": 1}}}},
            },
            "action": {"neighbor": "h1", "origin": "origin-x", "seq": 1, "wait_s": 1.5, "cause": "rejected"},
            "action_prob": 1.0,
        },
    ]


def test_log_decisions_false_emits_no_decision_events(tmp_path: Path):
    """SyncConfig(log_decisions=False) -- the default is True, and this effort's live
    configs leave it there -- drops every `decision` event while every other event and
    every sent Interest stays exactly as it would with logging on (see the fuller proof
    below)."""
    engine, transport, _ = _engine(tmp_path, neighbors=("h1", "h2"), log_decisions=False)
    engine._tracker.observe_state("origin-x", "h1", hi=1, extra=())
    engine.on_tick(now=0.0)
    events = engine.drain_events()
    assert events == []
    # The fetch Interest for origin-x/1 and both state requests were still sent.
    assert len(transport.sent) == 3


def test_decision_logging_leaves_interests_and_other_events_unchanged(tmp_path: Path):
    """The "do NOT change behaviour" proof (map.md "判断の記録"): with log_decisions
    True vs False, an otherwise identical engine sends the exact same Interests at the
    exact same times and emits the exact same non-`decision` events. Exact-count
    assertions on the jitter/backoff callables (poll=2, backoff=1 -- see the two
    on_tick calls and the one timeout below) confirm logging draws neither rng a
    second time, and ticking once strictly before the backoff deadline and once at
    it directly shows a retry is not sent early.

    2026-10-23 bug fix: the prior version of this test returned a *constant* from
    both jitter callables and only compared call counts between the True/False
    runs, never against a fixed expected count. A constant jitter makes a hidden
    double-draw invisible here: calling it twice instead of once still returns the
    same value both times, so scheduling comes out identical either way and the
    comparison passes even with the bug. Cycling through distinct values makes an
    extra draw shift every value drawn after it, in only the buggy run.
    """

    def _run(label: str, log_decisions: bool):
        """Build one engine (log_decisions True or False) and drive it through a
        state poll, a fetch timeout, a tick strictly before the resulting backoff's
        deadline, and a tick at it, returning what was sent, what was logged, and
        how many times each rng was actually drawn.
        """
        # Same node identity ("h0") in both runs -- only the db file (named by
        # `label`, not `node`) differs -- so a divergent wire name would signal a
        # real behavior change, not just a different store path.
        store = Store(tmp_path / f"{label}.db")
        transport = RecordingTransport()
        poll_calls: list[None] = []
        backoff_calls: list[None] = []
        # Distinct, non-repeating values (not a constant): a hidden extra draw
        # from either sequence returns a value it would not otherwise have
        # reached, shifting every interval/wait computed after it -- a constant
        # jitter cannot expose that, since drawing it twice looks the same as once.
        poll_jitter_values = (0.2, 0.8)
        backoff_jitter_values = (0.25, 0.75)

        def poll_jitter() -> float:
            """Return the next of poll_jitter_values, cycling; record that it was drawn."""
            value = poll_jitter_values[len(poll_calls) % len(poll_jitter_values)]
            poll_calls.append(None)
            return value

        def backoff_jitter() -> float:
            """Return the next of backoff_jitter_values, cycling; record that it was drawn."""
            value = backoff_jitter_values[len(backoff_calls) % len(backoff_jitter_values)]
            backoff_calls.append(None)
            return value

        config = SyncConfig(
            group="g", node="h0", neighbors=("h1",), fetch_backoff_initial_s=1.0, log_decisions=log_decisions
        )
        engine = SyncEngine(config, store, transport, poll_jitter_rng=poll_jitter, backoff_jitter=backoff_jitter)
        engine._tracker.observe_state("origin-x", "h1", hi=1, extra=())

        all_events: list[dict] = []
        engine.on_tick(now=0.0)  # state poll (due now; poll draw #1) + fetch plan (claims origin-x/1)
        all_events += engine.drain_events()

        fetch_name = names.build_record_name("g", "h1", "origin-x", 1)
        engine.on_fetch_timeout(fetch_name, now=0.5)  # backoff draw #1: wait = 1.0 + 0.25 = 1.25s, deadline 1.75
        all_events += engine.drain_events()
        sent_count_after_timeout = len(transport.sent)

        engine.on_tick(now=1.0)  # strictly before the 1.75 deadline and the next poll's own due time (>=1.6)
        all_events += engine.drain_events()
        sent_count_before_deadline = len(transport.sent)

        # 2026-10-23 test fix: the 1.75s backoff deadline and the next poll's own
        # ~1.76s due time are only a hundredth of a second apart, so a tick at 1.0
        # and the next one at 5.0 cannot tell "the retry resent exactly on time" from
        # "a state poll fired instead" -- nor would either catch a bug that nudges
        # either deadline by a few hundredths of a second (e.g. only when
        # log_decisions draws an extra, unaccounted-for rng value). Ticking once
        # more, strictly before 1.75, and once exactly at it, isolates the fetch
        # retry from the still-not-due poll and pins the exact Interest each sends.
        engine.on_tick(now=1.7)  # strictly before both the 1.75 and ~1.76 deadlines
        all_events += engine.drain_events()
        sent_before_1_75 = list(transport.sent)

        engine.on_tick(now=1.75)  # exactly the backoff deadline; the poll is not due until 1.76
        all_events += engine.drain_events()
        sent_at_1_75 = list(transport.sent)

        # poll_jitter_values[0]=0.2 maps to offset -0.12 (state_poll_jitter_frac's
        # own default 0.2 -> 0.2*(2*0.2-1)), so the first poll's own interval is
        # exactly 2.0*(1-0.12)=1.76s: tick there, not at some later catch-all time,
        # so a bug that nudges this deadline (not just the backoff one above) by a
        # few hundredths of a second is caught at the instant it would first show.
        engine.on_tick(now=1.76)  # exactly the next poll's own due time: poll draw #2
        all_events += engine.drain_events()

        return (
            transport.sent,
            all_events,
            len(poll_calls),
            len(backoff_calls),
            sent_count_after_timeout,
            sent_count_before_deadline,
            sent_before_1_75,
            sent_at_1_75,
            fetch_name,
            config.interest_lifetime_ms,
        )

    (
        sent_on, events_on, polls_on, backoffs_on, after_timeout_on, before_deadline_on,
        before_1_75_on, at_1_75_on, fetch_name_on, lifetime_on,
    ) = _run("on", True)
    (
        sent_off, events_off, polls_off, backoffs_off, after_timeout_off, before_deadline_off,
        before_1_75_off, at_1_75_off, fetch_name_off, lifetime_off,
    ) = _run("off", False)

    assert sent_on == sent_off
    assert polls_on == 2
    assert backoffs_on == 1
    assert polls_off == 2
    assert backoffs_off == 1

    # The tick strictly before the backoff deadline (now=1.0) sends nothing new in
    # either run -- the retry is not resent early.
    assert before_deadline_on == after_timeout_on
    assert before_deadline_off == after_timeout_off

    # Still nothing new at 1.7s: strictly before both the backoff deadline and the
    # next poll's own due time.
    assert before_1_75_on == sent_on[:before_deadline_on]
    assert before_1_75_off == sent_off[:before_deadline_off]

    # Exactly at the 1.75s deadline, the retry -- and only the retry, not a poll
    # that is not due until 1.76s -- is resent.
    assert at_1_75_on[len(before_1_75_on):] == [(fetch_name_on, lifetime_on)]
    assert at_1_75_off[len(before_1_75_off):] == [(fetch_name_off, lifetime_off)]

    # Exactly at the poll's own 1.76s due time, exactly one more Interest goes
    # out: the state-request poll itself, not a second copy of the
    # already-in-flight retry (checked, separately from the retry above, at the
    # instant the poll is actually due rather than at some later catch-all tick).
    poll_name = names.build_state_name("g", "h1", "h0", round(1.76 * 1000))
    assert sent_on[len(at_1_75_on):] == [(poll_name, lifetime_on)]
    assert sent_off[len(at_1_75_off):] == [(poll_name, lifetime_off)]

    non_decision_on = [e for e in events_on if e["event"] != "decision"]
    non_decision_off = [e for e in events_off if e["event"] != "decision"]
    assert non_decision_on == non_decision_off
    assert any(e["event"] == "decision" for e in events_on)
    assert not any(e["event"] == "decision" for e in events_off)


# -- state-poll interval choices (stage-2 map "行動と確率") ---------------------------


class _Cycle:
    """A [0, 1) rng stand-in that returns `values` in order, cycling, and counts its draws.

    Distinct values (not a constant) so a hidden extra draw shifts every later
    value and shows up in the chosen intervals, not just in the draw count.
    """

    def __init__(self, *values: float) -> None:
        """Remember the values to cycle through; no draw has happened yet."""
        self._values = values
        self.calls = 0

    def __call__(self) -> float:
        """Return the next value and count the draw."""
        value = self._values[self.calls % len(self._values)]
        self.calls += 1
        return value


def _poll_decisions(engine: SyncEngine, ticks: list[float]) -> list[tuple[float, dict, float]]:
    """Tick `engine` at each time in `ticks` and return (ts, action, action_prob) of
    every state_poll decision -- not the whole line, so these tests pin only the
    policy's own output and not the decision state's shape."""
    rows = []
    for now in ticks:
        engine.on_tick(now)
        for event in engine.drain_events():
            if event["event"] == "decision" and event["kind"] == "state_poll":
                rows.append((event["ts_ms"], event["action"], event["action_prob"]))
    return rows


def test_unset_choices_keeps_stage_1_jittered_intervals_and_action_prob_1(tmp_path: Path):
    """With no state_poll_interval_choices_s, the interval is still 2 s +/- 20% from
    poll_jitter_rng (0.2 -> 1.76 s, 0.8 -> 2.24 s), action_prob is still 1.0, and the
    choice rng is never drawn -- stage 1's policy, intervals, action and rng draws are
    unchanged. The decision line's state does gain the two poll-reply fields (pinned in
    test_on_tick_emits_a_state_poll_decision_per_due_neighbor), so this compares only
    ts_ms, action and action_prob."""
    store = Store(tmp_path / "h0.db")
    jitter = _Cycle(0.2, 0.8)
    choice = _Cycle(0.5)
    engine = SyncEngine(
        SyncConfig(group="g", node="h0", neighbors=("h1",)),
        store,
        RecordingTransport(),
        poll_jitter_rng=jitter,
        poll_choice_rng=choice,
    )
    assert _poll_decisions(engine, [0.0, 1.76, 4.0]) == [
        (0, {"neighbor": "h1", "interval_s": 1.76}, 1.0),
        (1760, {"neighbor": "h1", "interval_s": 2.24}, 1.0),
        (4000, {"neighbor": "h1", "interval_s": 1.76}, 1.0),
    ]
    assert jitter.calls == 3
    assert choice.calls == 0


def test_choices_draw_one_value_per_poll_with_action_prob_one_over_len(tmp_path: Path):
    """With choices (1, 2, 4), each poll's interval is the choice its rng draw indexes
    (u in [0,1/3) -> 1, [1/3,2/3) -> 2, [2/3,1) -> 4), the next poll is scheduled that
    far ahead, action_prob is exactly 1/3, and the jitter rng is never drawn."""
    store = Store(tmp_path / "h0.db")
    transport = RecordingTransport()
    jitter = _Cycle(0.2)
    choice = _Cycle(0.9, 0.0, 0.5)
    engine = SyncEngine(
        SyncConfig(group="g", node="h0", neighbors=("h1",), state_poll_interval_choices_s=(1.0, 2.0, 4.0)),
        store,
        transport,
        poll_jitter_rng=jitter,
        poll_choice_rng=choice,
    )
    # 3.9 and 4.9 are strictly before the 4 s and then 1 s intervals elapse, so
    # neither tick polls; 4.0 and 5.0 are exactly when each comes due.
    assert _poll_decisions(engine, [0.0, 3.9, 4.0, 4.9, 5.0]) == [
        (0, {"neighbor": "h1", "interval_s": 4.0, "applied_interval_s": 4.0}, 1 / 3),
        (4000, {"neighbor": "h1", "interval_s": 1.0, "applied_interval_s": 1.0}, 1 / 3),
        (5000, {"neighbor": "h1", "interval_s": 2.0, "applied_interval_s": 2.0}, 1 / 3),
    ]
    assert [n for n, _ in transport.sent] == [
        names.build_state_name("g", "h1", "h0", 0),
        names.build_state_name("g", "h1", "h0", 4000),
        names.build_state_name("g", "h1", "h0", 5000),
    ]
    assert choice.calls == 3
    assert jitter.calls == 0


def test_a_single_choice_is_a_fixed_interval_policy_with_action_prob_1(tmp_path: Path):
    """One choice is a deterministic fixed-interval policy: every interval is that
    value and action_prob is 1.0."""
    engine, _, _ = _engine(tmp_path, state_poll_interval_choices_s=(0.5,))
    assert _poll_decisions(engine, [0.0, 0.5, 1.0]) == [
        (0, {"neighbor": "h1", "interval_s": 0.5, "applied_interval_s": 0.5}, 1.0),
        (500, {"neighbor": "h1", "interval_s": 0.5, "applied_interval_s": 0.5}, 1.0),
        (1000, {"neighbor": "h1", "interval_s": 0.5, "applied_interval_s": 0.5}, 1.0),
    ]


def test_choice_jitter_scales_the_applied_interval_but_not_the_action(tmp_path: Path):
    """With j = 0.1, each poll's applied interval is the chosen value times
    1 + 0.1 * (2u - 1) for the jitter draw u: u = 0 -> 0.9, u = 0.75 -> 1.05,
    u = 0.5 -> 1.0. The action keeps the chosen value with action_prob 1/2, the
    applied interval is recorded beside it, and the next poll is due that far
    after this one's own send time (jitter accumulates, it is not undone)."""
    store = Store(tmp_path / "h0.db")
    transport = RecordingTransport()
    engine = SyncEngine(
        SyncConfig(
            group="g",
            node="h0",
            neighbors=("h1",),
            state_poll_interval_choices_s=(1.0, 2.0),
            state_poll_choice_jitter_frac=0.1,
        ),
        store,
        transport,
        poll_choice_rng=_Cycle(0.0, 0.9),
        poll_choice_jitter_rng=_Cycle(0.0, 0.75, 0.5),
    )
    first, second = 1.0 * 0.9, 2.0 * 1.05
    ticks = [0.0, first - 0.01, first, first + second - 0.01, first + second]
    assert _poll_decisions(engine, ticks) == [
        (0, {"neighbor": "h1", "interval_s": 1.0, "applied_interval_s": first}, 0.5),
        (round(first * 1000), {"neighbor": "h1", "interval_s": 2.0, "applied_interval_s": second}, 0.5),
        (round((first + second) * 1000), {"neighbor": "h1", "interval_s": 1.0, "applied_interval_s": 1.0}, 0.5),
    ]
    assert len(transport.sent) == 3


def test_choice_jitter_zero_reproduces_jitter_free_choices(tmp_path: Path):
    """j = 0: whatever the jitter rng returns, the applied interval is exactly the chosen
    value, so polls go out at exactly the times the choices alone give (4, 1, 2 s)."""
    store = Store(tmp_path / "h0.db")
    transport = RecordingTransport()
    jitter = _Cycle(0.0, 0.99)
    engine = SyncEngine(
        SyncConfig(
            group="g",
            node="h0",
            neighbors=("h1",),
            state_poll_interval_choices_s=(1.0, 2.0, 4.0),
            state_poll_choice_jitter_frac=0.0,
        ),
        store,
        transport,
        poll_choice_rng=_Cycle(0.9, 0.0, 0.5),
        poll_choice_jitter_rng=jitter,
    )
    assert _poll_decisions(engine, [0.0, 3.9, 4.0, 4.9, 5.0]) == [
        (0, {"neighbor": "h1", "interval_s": 4.0, "applied_interval_s": 4.0}, 1 / 3),
        (4000, {"neighbor": "h1", "interval_s": 1.0, "applied_interval_s": 1.0}, 1 / 3),
        (5000, {"neighbor": "h1", "interval_s": 2.0, "applied_interval_s": 2.0}, 1 / 3),
    ]
    assert [n for n, _ in transport.sent] == [
        names.build_state_name("g", "h1", "h0", 0),
        names.build_state_name("g", "h1", "h0", 4000),
        names.build_state_name("g", "h1", "h0", 5000),
    ]
    assert jitter.calls == 3


def test_decision_logging_leaves_interests_and_other_events_unchanged_under_choices(tmp_path: Path):
    """Stage 1's "logging adds no behavior" property, for the choices policy: with
    log_decisions True vs False, the same Interests go out at the same times, the
    non-`decision` events match, and the choice rng and the choices-mode jitter rng
    are each drawn exactly once per poll while stage 1's jitter rng is never drawn."""

    def _run(label: str, log_decisions: bool):
        """Drive one engine through polls, a reply and a fetch timeout; return what it sent,
        emitted, and how many times each rng was drawn."""
        store = Store(tmp_path / f"{label}.db")
        transport = RecordingTransport()
        jitter = _Cycle(0.2)
        choice = _Cycle(0.1, 0.7, 0.4)
        choice_jitter = _Cycle(0.1, 0.9, 0.3)
        config = SyncConfig(
            group="g",
            node="h0",
            neighbors=("h1", "h2"),
            state_poll_interval_choices_s=(0.5, 1.0, 2.0),
            log_decisions=log_decisions,
        )
        engine = SyncEngine(
            store=store,
            transport=transport,
            config=config,
            poll_jitter_rng=jitter,
            poll_choice_rng=choice,
            poll_choice_jitter_rng=choice_jitter,
        )
        events: list[dict] = []
        engine.on_tick(0.0)
        engine.on_state("h1", holder_state.encode({"origin-x": OriginProgress(hi=1, extra=())}), 0.1, poll_epoch_ms=0)
        events += engine.drain_events()
        engine.on_fetch_timeout(names.build_record_name("g", "h1", "origin-x", 1), 0.3)
        # 0.05 s ticks: the jittered due times are off the 0.5 s grid.
        for i in range(10, 81):
            engine.on_tick(i * 0.05)
            events += engine.drain_events()
        return transport.sent, events, choice.calls, jitter.calls, choice_jitter.calls

    sent_on, events_on, choice_on, jitter_on, choice_jitter_on = _run("on", True)
    sent_off, events_off, choice_off, jitter_off, choice_jitter_off = _run("off", False)
    assert sent_on == sent_off
    assert [e for e in events_on if e["event"] != "decision"] == [e for e in events_off if e["event"] != "decision"]
    polls = sum(1 for n, _ in sent_on if isinstance(names.parse(n), names.StateName))
    assert choice_on == polls
    assert choice_off == polls
    assert choice_jitter_on == polls
    assert choice_jitter_off == polls
    assert jitter_on == 0
    assert jitter_off == 0
    assert any(e["event"] == "decision" for e in events_on)
    assert not any(e["event"] == "decision" for e in events_off)


# -- poll-reply status in the state_poll decision state (stage-2 map "判断の状態に足すもの") --


_EMPTY_STATE_PAYLOAD = holder_state.encode({})


def _poll_reply_states(engine: SyncEngine, steps: list[tuple]) -> list[tuple[int, str | None, int]]:
    """Run `steps` -- ("tick", now), or ("reply", now, epoch_ms) for h1's reply to the
    poll sent at epoch_ms -- and return each state_poll decision's
    (ts_ms, last_poll_status, consecutive_poll_timeouts)."""
    rows = []
    for step in steps:
        if step[0] == "tick":
            engine.on_tick(step[1])
        else:
            engine.on_state("h1", _EMPTY_STATE_PAYLOAD, step[1], poll_epoch_ms=step[2])
        for event in engine.drain_events():
            if event["event"] == "decision" and event["kind"] == "state_poll":
                rows.append(
                    (event["ts_ms"], event["state"]["last_poll_status"], event["state"]["consecutive_poll_timeouts"])
                )
    return rows


def test_poll_reply_status_answered(tmp_path: Path):
    """A reply to the previous poll shows as "answered" with no timeouts in a row."""
    engine, _, _ = _engine(tmp_path)  # interval exactly 2.0 s, lifetime 2000 ms
    assert _poll_reply_states(engine, [("tick", 0.0), ("reply", 0.1, 0), ("tick", 2.0)]) == [
        (0, None, 0),
        (2000, "answered", 0),
    ]


def test_poll_reply_status_timed_out_once_the_lifetime_has_passed(tmp_path: Path):
    """With no reply, the previous poll is "timed_out" as soon as its 2 s lifetime has
    passed (deadline <= now, the transport's own rule), and each further unanswered
    poll adds one to the count."""
    engine, _, _ = _engine(tmp_path)
    assert _poll_reply_states(engine, [("tick", 0.0), ("tick", 2.0), ("tick", 4.0)]) == [
        (0, None, 0),
        (2000, "timed_out", 1),
        (4000, "timed_out", 2),
    ]


def test_poll_reply_status_pending_while_the_lifetime_has_not_passed(tmp_path: Path):
    """With a 1 s interval and a 2 s lifetime, the previous poll is still "pending" at
    the next decision. At 2.0 the first poll's lifetime has passed (counted), while
    the one sent at 1.0 is the previous poll and still pending."""
    engine, _, _ = _engine(tmp_path, state_poll_interval_choices_s=(1.0,))
    assert _poll_reply_states(engine, [("tick", 0.0), ("tick", 1.0), ("tick", 2.0), ("tick", 3.0)]) == [
        (0, None, 0),
        (1000, "pending", 0),
        (2000, "pending", 1),
        (3000, "pending", 2),
    ]


def test_poll_reply_status_answered_after_consecutive_timeouts_resets_the_count(tmp_path: Path):
    """Two polls time out in a row, then the next one is answered: the decision after
    it shows "answered" and the count back at 0."""
    engine, _, _ = _engine(tmp_path)
    steps = [("tick", 0.0), ("tick", 2.0), ("tick", 4.0), ("reply", 4.1, 4000), ("tick", 6.0)]
    assert _poll_reply_states(engine, steps) == [
        (0, None, 0),
        (2000, "timed_out", 1),
        (4000, "timed_out", 2),
        (6000, "answered", 0),
    ]


def test_a_reply_to_an_older_poll_does_not_mark_the_newer_one_answered(tmp_path: Path):
    """With a 1 s interval, the reply to the poll sent at 0 arrives after the poll at
    1.0 went out: it marks only the poll at 0 answered, so the decision at 2.0 still
    sees its previous poll (sent at 1.0) as pending. Once that poll times out
    unanswered, the count is 1 -- the older answer does not hide it."""
    engine, _, _ = _engine(tmp_path, state_poll_interval_choices_s=(1.0,))
    steps = [("tick", 0.0), ("tick", 1.0), ("reply", 1.05, 0), ("tick", 2.0), ("tick", 3.0)]
    assert _poll_reply_states(engine, steps) == [
        (0, None, 0),
        (1000, "pending", 0),
        (2000, "pending", 0),
        (3000, "pending", 1),
    ]


def test_interleaved_half_second_polls_track_each_reply_by_its_own_poll(tmp_path: Path):
    """0.5 s polls with a 2 s lifetime keep four polls outstanding at once.

    - The reply to the poll at 500 (arriving at 1.1) answers that poll only.
    - The polls at 1000 and 1500 then time out unanswered (at 3.0 and 3.5).
    - A late reply to the poll at 1000 (at 3.6) does not reset the count, because the
      newer poll at 1500 has already timed out, and by 4.0 so has the one at 2000:
      two timeouts since the newest answered poll.
    - The reply to the poll at 3500 then brings the count back to 0; the decision at
      4.5 shows its previous poll (sent at 4.0) as pending, not answered.
    """
    engine, _, _ = _engine(tmp_path, state_poll_interval_choices_s=(0.5,))
    steps = [
        ("tick", 0.0),
        ("tick", 0.5),
        ("tick", 1.0),
        ("reply", 1.1, 500),
        ("tick", 1.5),
        ("tick", 2.0),
        ("tick", 2.5),
        ("tick", 3.0),
        ("tick", 3.5),
        ("reply", 3.6, 1000),
        ("tick", 4.0),
        ("reply", 4.1, 3500),
        ("tick", 4.5),
    ]
    assert _poll_reply_states(engine, steps) == [
        (0, None, 0),
        (500, "pending", 0),
        (1000, "pending", 0),
        (1500, "pending", 0),
        (2000, "pending", 0),
        (2500, "pending", 0),
        (3000, "pending", 1),
        (3500, "pending", 2),
        (4000, "pending", 2),
        (4500, "pending", 0),
    ]


def test_a_late_reply_after_the_lifetime_still_counts_as_answered(tmp_path: Path):
    """A reply that arrives after its poll's lifetime has passed is still that poll's
    reply: the neighbor is answering, so the status is "answered" and the count 0."""
    engine, _, _ = _engine(tmp_path, state_poll_interval_choices_s=(4.0,))
    assert _poll_reply_states(engine, [("tick", 0.0), ("reply", 3.0, 0), ("tick", 4.0)]) == [
        (0, None, 0),
        (4000, "answered", 0),
    ]


def test_a_long_silent_outage_keeps_the_poll_history_bounded(tmp_path: Path):
    """2000 s of 0.5 s polls with no reply: the count keeps growing (every poll sent at
    least 2 s before the decision has timed out), but only the polls still within
    lifetime + 1 s grace are held -- the 6 sent in [T - 3, T - 0.5] (a reply exactly at
    send + 3 s is still accepted), plus the poll just sent."""
    engine, _, _ = _engine(tmp_path, state_poll_interval_choices_s=(0.5,))
    rows = _poll_reply_states(engine, [("tick", i * 0.5) for i in range(4001)])
    assert rows[-1] == (2_000_000, "pending", 3997)
    assert len(engine._poll_replies["h1"].polls) == 7


def test_a_reply_later_than_lifetime_plus_grace_to_an_older_poll_is_ignored(tmp_path: Path):
    """Cefore drops a PIT entry about 1 s after its lifetime, so a poll unanswered for
    lifetime + 1 s (and no longer the newest) is counted as timed out for good: a
    reply to it after that changes neither the status nor the count."""
    engine, _, _ = _engine(tmp_path, state_poll_interval_choices_s=(0.5,))
    steps = [("tick", i * 0.5) for i in range(8)] + [("reply", 3.6, 0), ("tick", 4.0)]
    assert _poll_reply_states(engine, steps)[-2:] == [
        (3500, "pending", 4),
        (4000, "pending", 5),
    ]


def test_a_reply_after_grace_to_a_poll_no_longer_newest_is_ignored_even_before_expiry(tmp_path: Path):
    """Lifetime 2 s, 4 s interval: polls at 0 and 4. The reply to poll 0 arrives at 4.1,
    when poll 0 is no longer the newest and 4.1 > 0 + 2 + 1, so it is ignored even
    though no decision has dropped poll 0 yet: at 8 both polls count as timed out."""
    engine, _, _ = _engine(tmp_path, state_poll_interval_choices_s=(4.0,))
    steps = [("tick", 0.0), ("tick", 4.0), ("reply", 4.1, 0), ("tick", 8.0)]
    assert _poll_reply_states(engine, steps) == [
        (0, None, 0),
        (4000, "timed_out", 1),
        (8000, "timed_out", 2),
    ]


def test_a_reply_exactly_at_lifetime_plus_grace_is_accepted(tmp_path: Path):
    """0.5 s polls: the reply to poll 0 at exactly 0 + 2 + 1 = 3.0 s is accepted, so
    only the polls after it (0.5, 1.0, 1.5) count as timed out at 3.5."""
    engine, _, _ = _engine(tmp_path, state_poll_interval_choices_s=(0.5,))
    steps = [("tick", i * 0.5) for i in range(7)] + [("reply", 3.0, 0), ("tick", 3.5)]
    assert _poll_reply_states(engine, steps)[-1] == (3500, "pending", 3)


def test_a_reply_just_after_lifetime_plus_grace_is_ignored(tmp_path: Path):
    """The same reply at 3.01 s is past the grace, so poll 0 stays timed out: 4 at 3.5."""
    engine, _, _ = _engine(tmp_path, state_poll_interval_choices_s=(0.5,))
    steps = [("tick", i * 0.5) for i in range(7)] + [("reply", 3.01, 0), ("tick", 3.5)]
    assert _poll_reply_states(engine, steps)[-1] == (3500, "pending", 4)


def test_a_reply_to_the_newest_poll_is_accepted_however_late(tmp_path: Path):
    """While a poll is still the newest one sent, its reply is accepted even past
    lifetime + grace (3.9 s here); the same poll stops being so once the next is sent."""
    engine, _, _ = _engine(tmp_path, state_poll_interval_choices_s=(4.0,))
    steps = [("tick", 0.0), ("reply", 3.9, 0), ("tick", 4.0)]
    assert _poll_reply_states(engine, steps) == [
        (0, None, 0),
        (4000, "answered", 0),
    ]


def test_bulk_expiry_drops_the_whole_stale_prefix_at_once(tmp_path: Path):
    """After 1000 polls 0.5 s apart with no reply, one decision 100 s later drops every
    stale poll in one go, keeping only the newest, and counts all 1000 as timed out."""
    engine, _, _ = _engine(tmp_path, state_poll_interval_choices_s=(0.5,))
    replies = engine._poll_replies.setdefault("h1", _PollReplies())
    for i in range(1000):
        replies.sent(i * 500, i * 0.5)
    assert replies.snapshot(600.0, 2.0) == ("timed_out", 1000)
    assert len(replies.polls) == 1


def test_a_reply_that_fails_to_decode_still_counts_as_answered(tmp_path: Path):
    """A reply whose payload cannot be decoded still shows the neighbor is answering."""
    engine, _, _ = _engine(tmp_path)
    engine.on_tick(0.0)
    engine.on_state("h1", b"not json", 0.1, poll_epoch_ms=0)
    engine.drain_events()
    engine.on_tick(2.0)
    [decision] = [e for e in engine.drain_events() if e["event"] == "decision"]
    assert (decision["state"]["last_poll_status"], decision["state"]["consecutive_poll_timeouts"]) == ("answered", 0)


def test_poll_reply_status_leaves_interests_and_other_events_unchanged(tmp_path: Path):
    """The poll-reply bookkeeping only feeds the decision state: with log_decisions
    True vs False, the same scenario of interleaved 0.5 s polls, timeouts and replies
    (including one to an older poll) sends the same Interests at the same times and
    emits the same non-`decision` events."""

    def _run(label: str, log_decisions: bool):
        """Drive one engine through the scenario; return what it sent and emitted."""
        engine, transport, _ = _engine(
            tmp_path / label, state_poll_interval_choices_s=(0.5,), log_decisions=log_decisions
        )
        payload = holder_state.encode({"origin-x": OriginProgress(hi=1, extra=())})
        events: list[dict] = []
        steps = [("tick", 0.0), ("tick", 0.5), ("reply", 0.6, 0), ("tick", 1.0), ("tick", 2.5), ("reply", 2.6, 1000)]
        for step in steps + [("tick", 3.0), ("tick", 3.5)]:
            if step[0] == "tick":
                engine.on_tick(step[1])
            else:
                engine.on_state("h1", payload, step[1], poll_epoch_ms=step[2])
            events += engine.drain_events()
        return transport.sent, events

    for label in ("on", "off"):
        (tmp_path / label).mkdir()
    sent_on, events_on = _run("on", True)
    sent_off, events_off = _run("off", False)
    assert sent_on == sent_off
    assert [e for e in events_on if e["event"] != "decision"] == [e for e in events_off if e["event"] != "decision"]
    assert not any(e["event"] == "decision" for e in events_off)


# -- re-deciding the next poll when the newest poll resolves (ADR 0004, state_poll_policy) --


def _policy_engine(
    tmp_path: Path, intervals: tuple[float, ...], jitter: tuple[float, ...], u: "_Cycle", **overrides
) -> tuple[SyncEngine, RecordingTransport, dict[str, _Cycle]]:
    """A one-neighbor (h1) engine under a state_poll_policy table, its jitter u drawn
    from `u`; the other two poll rngs are spies that must stay undrawn."""
    store = Store(tmp_path / "h0.db")
    transport = RecordingTransport()
    spies = {"jitter": _Cycle(0.2), "choice": _Cycle(0.5)}
    config = SyncConfig(
        group="g",
        node="h0",
        neighbors=("h1",),
        state_poll_policy=StatePollPolicy(intervals_s=intervals, jitter_frac=jitter),
        **overrides,
    )
    engine = SyncEngine(
        config,
        store,
        transport,
        poll_jitter_rng=spies["jitter"],
        poll_choice_rng=spies["choice"],
        poll_choice_jitter_rng=u,
    )
    return engine, transport, spies


def _drive(engine: SyncEngine, steps: list[tuple]) -> list[dict]:
    """Run ("tick", now) / ("reply", now, epoch_ms) steps against h1 and return every event."""
    events: list[dict] = []
    for step in steps:
        if step[0] == "tick":
            engine.on_tick(step[1])
        else:
            engine.on_state("h1", _EMPTY_STATE_PAYLOAD, step[1], poll_epoch_ms=step[2])
        events += engine.drain_events()
    return events


def _decisions(events: list[dict], kind: str) -> list[dict]:
    """The `decision` events of one kind, in order."""
    return [e for e in events if e["event"] == "decision" and e["kind"] == kind]


def _poll_names(transport: RecordingTransport) -> list[str]:
    """Every state-request Interest name sent, in order."""
    return [n for n, _ in transport.sent if isinstance(names.parse(n), names.StateName)]


def _state_name(epoch_ms: int) -> str:
    """h0's state-request name toward h1 for the poll sent at epoch_ms."""
    return names.build_state_name("g", "h1", "h0", epoch_ms)


def test_policy_reschedules_on_a_timeout_from_the_poll_s_own_send_time(tmp_path: Path):
    """Table (10, 4), j 0.1, u 0.75 -> factor 1.05. The poll at 0 is sent with the
    after-success interval 10.5 s; at 2.0 its lifetime has passed, so the pending next
    poll is re-decided from the after-failure row with the same u: 0 + 4 * 1.05 = 4.2.
    The second poll draws its own u (0.25); the first poll's u is never drawn again."""
    u = _Cycle(0.75, 0.25)
    engine, transport, spies = _policy_engine(tmp_path, (10.0, 4.0), (0.1, 0.1), u)
    events = _drive(engine, [("tick", 0.0), ("tick", 1.99)])
    assert _decisions(events, "state_poll_reschedule") == []
    events = _drive(engine, [("tick", 2.0)])
    assert _decisions(events, "state_poll_reschedule") == [
        {
            "event": "decision",
            "ts_ms": 2000,
            "kind": "state_poll_reschedule",
            "state": {"last_poll_status": "timed_out", "consecutive_poll_timeouts": 1},
            "action": {
                "neighbor": "h1",
                "poll_epoch_ms": 0,
                "trigger": "timed_out",
                "interval_s": 4.0,
                "applied_interval_s": 4.0 * 1.05,
                "clamped": False,
            },
            "action_prob": 1.0,
        }
    ]
    _drive(engine, [("tick", 4.0 * 1.05 - 0.01)])
    assert _poll_names(transport) == [_state_name(0)]
    events = _drive(engine, [("tick", 4.0 * 1.05)])
    assert _poll_names(transport) == [_state_name(0), _state_name(4200)]
    # The second poll's provisional interval comes from the same table, looked up
    # with the pre-send state (one timeout), and its own fresh u.
    [poll] = _decisions(events, "state_poll")
    assert poll["action"] == {"neighbor": "h1", "interval_s": 4.0, "applied_interval_s": 4.0 * 0.95}
    assert poll["action_prob"] == 1.0
    assert u.calls == 2
    assert spies["jitter"].calls == 0
    assert spies["choice"].calls == 0


def test_policy_reschedules_on_an_answer_and_records_the_poll_it_resolved(tmp_path: Path):
    """After a timeout the poll at 4.2 is sent with the after-failure interval (4 *
    1.05); its reply at 4.3 re-decides the next poll from the after-success row with
    that poll's own u: 4.2 + 10 * 1.05 = 14.7, not 8.4."""
    engine, transport, _ = _policy_engine(tmp_path, (10.0, 4.0), (0.1, 0.1), _Cycle(0.75))
    sent_at = 4.0 * 1.05
    _drive(engine, [("tick", 0.0), ("tick", 2.0), ("tick", sent_at)])
    events = _drive(engine, [("reply", 4.3, 4200)])
    assert _decisions(events, "state_poll_reschedule") == [
        {
            "event": "decision",
            "ts_ms": 4300,
            "kind": "state_poll_reschedule",
            "state": {"last_poll_status": "answered", "consecutive_poll_timeouts": 0},
            "action": {
                "neighbor": "h1",
                "poll_epoch_ms": 4200,
                "trigger": "answered",
                "interval_s": 10.0,
                "applied_interval_s": 10.0 * 1.05,
                "clamped": False,
            },
            "action_prob": 1.0,
        }
    ]
    _drive(engine, [("tick", 8.4), ("tick", sent_at + 10.0 * 1.05 - 0.01)])
    assert _poll_names(transport) == [_state_name(0), _state_name(4200)]
    _drive(engine, [("tick", sent_at + 10.0 * 1.05)])
    assert _poll_names(transport)[-1] == _state_name(round((sent_at + 10.0 * 1.05) * 1000))


def test_policy_resolves_each_poll_once(tmp_path: Path):
    """A resolved poll is not resolved again: further ticks before the next send, and a
    duplicate reply to an answered poll, add no reschedule row."""
    engine, _, _ = _policy_engine(tmp_path, (10.0, 4.0), (0.0, 0.0), _Cycle(0.5))
    events = _drive(engine, [("tick", 0.0), ("tick", 2.0), ("tick", 2.5), ("tick", 3.9)])
    assert len(_decisions(events, "state_poll_reschedule")) == 1
    events = _drive(engine, [("tick", 4.0), ("reply", 4.1, 4000), ("reply", 4.2, 4000), ("tick", 6.5)])
    assert [d["action"]["trigger"] for d in _decisions(events, "state_poll_reschedule")] == ["answered"]


def test_policy_table_rows_saturate_at_the_last_row_and_return_to_row_0_on_success(tmp_path: Path):
    """A 4-row table (10, 3, 5, 7), no jitter, h1 silent: the timeouts re-decide with
    counts 1, 2, 3, 4 -> rows 1, 2, 3, 3 (the last row covers 3+), the count still
    growing once older polls are folded into expired_timeouts; an answer then
    re-decides from row 0. Sends: 0, 3, 8, 15, 22, then 22 + 10 = 32."""
    engine, transport, _ = _policy_engine(tmp_path, (10.0, 3.0, 5.0, 7.0), (0.0, 0.0, 0.0, 0.0), _Cycle(0.5))
    steps = [("tick", t) for t in (0.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 17.0, 22.0)] + [("reply", 22.1, 22000)]
    events = _drive(engine, steps)
    assert engine._poll_replies["h1"].polls[-1].replied
    rows = [
        (d["action"]["trigger"], d["state"]["consecutive_poll_timeouts"], d["action"]["interval_s"])
        for d in _decisions(events, "state_poll_reschedule")
    ]
    assert rows == [
        ("timed_out", 1, 3.0),
        ("timed_out", 2, 5.0),
        ("timed_out", 3, 7.0),
        ("timed_out", 4, 7.0),
        ("answered", 0, 10.0),
    ]
    assert _poll_names(transport) == [_state_name(ms) for ms in (0, 3000, 8000, 15000, 22000)]
    _drive(engine, [("tick", 31.9), ("tick", 32.0)])
    assert _poll_names(transport)[-1] == _state_name(32000)


def test_policy_folded_timeouts_still_pick_the_saturated_row(tmp_path: Path):
    """The count the table reads includes polls _expire has folded into
    expired_timeouts: by the timeout of the poll at 15, the polls at 0, 3 and 8 are
    past lifetime + grace and folded, and the count is still 4 (row 3)."""
    engine, _, _ = _policy_engine(tmp_path, (10.0, 3.0, 5.0, 7.0), (0.0, 0.0, 0.0, 0.0), _Cycle(0.5))
    events = _drive(engine, [("tick", t) for t in (0.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 17.0)])
    assert engine._poll_replies["h1"].expired_timeouts == 3
    last = _decisions(events, "state_poll_reschedule")[-1]
    assert (last["state"]["consecutive_poll_timeouts"], last["action"]["interval_s"]) == (4, 7.0)


def test_policy_an_undecodable_reply_still_resolves_the_poll_as_answered(tmp_path: Path):
    """A reply whose payload fails to decode still shows h1 answering: after a timeout
    (next poll at 4.0 + 4 = 8.0), the undecodable reply to the poll at 4.0 re-decides
    from row 0, moving the next poll to 4.0 + 10 = 14.0."""
    engine, transport, _ = _policy_engine(tmp_path, (10.0, 4.0), (0.0, 0.0), _Cycle(0.5))
    _drive(engine, [("tick", 0.0), ("tick", 2.0), ("tick", 4.0)])
    engine.on_state("h1", b"not json", 4.1, poll_epoch_ms=4000)
    [reschedule] = _decisions(engine.drain_events(), "state_poll_reschedule")
    assert (reschedule["action"]["trigger"], reschedule["action"]["interval_s"]) == ("answered", 10.0)
    _drive(engine, [("tick", 8.0), ("tick", 13.9)])
    assert _poll_names(transport) == [_state_name(0), _state_name(4000)]
    _drive(engine, [("tick", 14.0)])
    assert _poll_names(transport)[-1] == _state_name(14000)


def test_policy_a_reply_after_the_lifetime_but_before_any_tick_resolves_as_answered(tmp_path: Path):
    """The poll at 0 is answered at 2.05, past its 2 s lifetime but before any tick has
    resolved it as timed out: it is resolved once, as answered (count 0, next at 0 +
    10), with no timed-out row at the 2.1 tick and no late-reply event."""
    engine, transport, _ = _policy_engine(tmp_path, (10.0, 4.0), (0.0, 0.0), _Cycle(0.5))
    events = _drive(engine, [("tick", 0.0), ("reply", 2.05, 0), ("tick", 2.1), ("tick", 4.0), ("tick", 9.99)])
    assert [
        (d["ts_ms"], d["action"]["trigger"], d["state"]["consecutive_poll_timeouts"], d["action"]["applied_interval_s"])
        for d in _decisions(events, "state_poll_reschedule")
    ] == [(2050, "answered", 0, 10.0)]
    assert [e for e in events if e["event"] == "state_poll_late_reply"] == []
    assert _poll_names(transport) == [_state_name(0)]
    _drive(engine, [("tick", 10.0)])
    assert _poll_names(transport) == [_state_name(0), _state_name(10000)]


def test_policy_a_late_reply_to_a_timed_out_poll_does_not_re_decide(tmp_path: Path):
    """The poll at 0 is resolved as timed out at 2.0 (next poll at 4.0). Its reply at
    2.5 is still accepted -- the poll is answered, as before -- but the next poll stays
    at 4.0 (an answered re-decision would have moved it to 10) and a
    state_poll_late_reply event records the reply instead."""
    engine, transport, _ = _policy_engine(tmp_path, (10.0, 4.0), (0.0, 0.0), _Cycle(0.5))
    _drive(engine, [("tick", 0.0), ("tick", 2.0)])
    events = _drive(engine, [("reply", 2.5, 0)])
    assert _decisions(events, "state_poll_reschedule") == []
    assert [e for e in events if e["event"] == "state_poll_late_reply"] == [
        {"event": "state_poll_late_reply", "ts_ms": 2500, "neighbor": "h1", "poll_epoch_ms": 0}
    ]
    assert engine._poll_replies["h1"].snapshot(2.5, 2.0) == ("answered", 0)
    _drive(engine, [("tick", 3.99)])
    assert _poll_names(transport) == [_state_name(0)]
    events = _drive(engine, [("tick", 4.0)])
    assert _poll_names(transport) == [_state_name(0), _state_name(4000)]
    [poll] = _decisions(events, "state_poll")
    assert (poll["state"]["last_poll_status"], poll["state"]["consecutive_poll_timeouts"]) == ("answered", 0)


def test_policy_a_late_reply_to_a_timed_out_poll_no_longer_newest_is_still_counted(tmp_path: Path):
    """Table (10, 2.5): the poll at 0 times out at 2.0, the next goes at 2.5, and the
    reply to the poll at 0 arrives at 2.9 (within lifetime + grace, so accepted). It is
    a late reply to a timed-out poll: counted, never re-decided."""
    engine, transport, _ = _policy_engine(tmp_path, (10.0, 2.5), (0.0, 0.0), _Cycle(0.5))
    _drive(engine, [("tick", 0.0), ("tick", 2.0), ("tick", 2.5)])
    events = _drive(engine, [("reply", 2.9, 0)])
    assert _decisions(events, "state_poll_reschedule") == []
    assert [e for e in events if e["event"] == "state_poll_late_reply"] == [
        {"event": "state_poll_late_reply", "ts_ms": 2900, "neighbor": "h1", "poll_epoch_ms": 0}
    ]
    _drive(engine, [("tick", 4.49)])
    assert _poll_names(transport) == [_state_name(0), _state_name(2500)]


def test_policy_a_reply_to_a_poll_that_is_not_the_newest_does_not_re_decide(tmp_path: Path):
    """One-row table (1.0): the reply to the poll at 0 arrives after the poll at 1.0
    went out. It is accepted but changes no schedule (poll 0 was never resolved, so no
    late-reply event either); the reply to the newest poll then does re-decide."""
    engine, _, _ = _policy_engine(tmp_path, (1.0,), (0.0,), _Cycle(0.5))
    events = _drive(engine, [("tick", 0.0), ("tick", 1.0), ("reply", 1.05, 0)])
    assert _decisions(events, "state_poll_reschedule") == []
    assert [e for e in events if e["event"] == "state_poll_late_reply"] == []
    events = _drive(engine, [("reply", 1.1, 1000)])
    assert [d["action"]["poll_epoch_ms"] for d in _decisions(events, "state_poll_reschedule")] == [1000]


def test_policy_clamps_a_past_next_time_to_now_and_polls_in_the_same_tick(tmp_path: Path):
    """Table (10, 1): at 2.0 the poll at 0 times out and its after-failure next time,
    0 + 1 = 1.0, is already past. It is clamped to now and, because the timeout is
    resolved before the due check, the next poll goes out in that same tick."""
    engine, transport, _ = _policy_engine(tmp_path, (10.0, 1.0), (0.0, 0.0), _Cycle(0.5))
    _drive(engine, [("tick", 0.0)])
    events = _drive(engine, [("tick", 2.0)])
    assert _poll_names(transport) == [_state_name(0), _state_name(2000)]
    assert [(e["kind"], e["ts_ms"]) for e in events if e["event"] == "decision"] == [
        ("state_poll_reschedule", 2000),
        ("state_poll", 2000),
    ]
    [reschedule] = _decisions(events, "state_poll_reschedule")
    assert reschedule["action"] == {
        "neighbor": "h1",
        "poll_epoch_ms": 0,
        "trigger": "timed_out",
        "interval_s": 1.0,
        "applied_interval_s": 1.0,
        "clamped": True,
    }


def test_policy_clamps_an_answer_that_arrives_after_the_next_time_has_passed(tmp_path: Path):
    """One-row table (1.0): the reply to the poll at 0 arrives at 1.5, before any tick
    has sent the poll that was due at 1.0. The re-decided time 0 + 1.0 is past, so it
    is recorded as clamped and held as now, 1.5."""
    engine, _, _ = _policy_engine(tmp_path, (1.0,), (0.0,), _Cycle(0.5))
    _drive(engine, [("tick", 0.0)])
    [reschedule] = _decisions(_drive(engine, [("reply", 1.5, 0)]), "state_poll_reschedule")
    assert (reschedule["action"]["trigger"], reschedule["action"]["clamped"]) == ("answered", True)
    assert engine._next_state_poll["h1"] == 1.5


def test_policy_resolution_never_sends_or_cancels_an_interest_itself(tmp_path: Path):
    """Re-deciding only moves the pending next-poll time: neither the reply that
    resolves a poll nor the tick that resolves its timeout sends anything, and every
    Interest sent is a state poll at its own (re-decided) time, once."""
    engine, transport, _ = _policy_engine(tmp_path, (10.0, 4.0), (0.0, 0.0), _Cycle(0.5))
    _drive(engine, [("tick", 0.0)])
    before = list(transport.sent)
    _drive(engine, [("tick", 2.0)])  # resolves poll 0 as timed out; next at 4.0
    assert transport.sent == before
    _drive(engine, [("tick", 4.0)])
    before = list(transport.sent)
    _drive(engine, [("reply", 4.1, 4000)])  # resolves poll 4000 as answered; next at 14.0
    assert transport.sent == before
    _drive(engine, [("tick", t / 10) for t in range(41, 141)])
    assert transport.sent == [(_state_name(ms), 2000) for ms in (0, 4000, 14000)]


@pytest.mark.parametrize("interval", [0.5, 3.695])
def test_a_one_row_policy_polls_exactly_like_the_same_single_choice(tmp_path: Path, interval: float):
    """A one-row table (x, j) re-decides to the value it chose at send time with the
    same u, so over a FakeNetwork run with replies, a partition and a heal it sends the
    same Interests at the same times as choices mode with the one choice x, choice
    jitter j and the same jitter stream -- for x shorter and longer than the lifetime."""
    from fakes import FakeNetwork

    class _Tap:
        """Record every Interest name an engine sends, then pass it on."""

        def __init__(self, inner) -> None:
            """Wrap `inner`, a FakeNetwork node transport."""
            self.inner = inner
            self.sent: list[str] = []

        def register(self, prefix: str) -> None:
            """Pass through."""
            self.inner.register(prefix)

        def send_interest(self, name: str, lifetime_ms: int) -> None:
            """Record and pass through."""
            self.sent.append(name)
            self.inner.send_interest(name, lifetime_ms)

    def _run(label: str, **policy) -> tuple[list[str], list[dict]]:
        """Run h0 polling h1 for 40 s, h1 unreachable over [10, 20); return h0's sends and events."""
        network = FakeNetwork()
        tap = _Tap(network.transport_for("h0"))
        h0 = SyncEngine(
            SyncConfig(group="g", node="h0", neighbors=("h1",), **policy),
            Store(tmp_path / f"{label}-h0.db"),
            tap,
            poll_choice_rng=_Cycle(0.3, 0.9),
            poll_choice_jitter_rng=_Cycle(0.1, 0.9, 0.35, 0.6),
        )
        h1 = SyncEngine(SyncConfig(group="g", node="h1", neighbors=()), Store(tmp_path / f"{label}-h1.db"), network.transport_for("h1"))
        network.add_node("h0", h0)
        network.add_node("h1", h1)
        events: list[dict] = []
        for i in range(800):
            now = i * 0.05
            if i == 200:
                network.partition("h0", "h1")
            if i == 400:
                network.heal("h0", "h1")
            network.pump(now)
            h0.on_tick(now)
            events += h0.drain_events()
        return tap.sent, events

    sent_policy, events_policy = _run(
        "policy", state_poll_policy=StatePollPolicy(intervals_s=(interval,), jitter_frac=(0.1,))
    )
    sent_choice, _ = _run("choice", state_poll_interval_choices_s=(interval,), state_poll_choice_jitter_frac=0.1)
    assert sent_policy == sent_choice
    assert len(sent_policy) > 40 / (interval * 1.1)
    # Every reschedule kept the time chosen at send.
    assert {d["action"]["clamped"] for d in _decisions(events_policy, "state_poll_reschedule")} <= {False}


def _snapshot_spy(monkeypatch) -> list[float]:
    """Count every _PollReplies.snapshot call (the times it was called at)."""
    calls: list[float] = []
    original = _PollReplies.snapshot

    def spy(self, now, lifetime_s):
        """Record the call's time, then run the real snapshot."""
        calls.append(now)
        return original(self, now, lifetime_s)

    monkeypatch.setattr(_PollReplies, "snapshot", spy)
    return calls


@pytest.mark.parametrize("mode", ["stage1", "choices"])
def test_without_a_policy_nothing_new_runs(tmp_path: Path, monkeypatch, mode: str):
    """The gate: with no state_poll_policy the poll-reply history is read only when a
    poll is sent (once per poll, as before -- never from the new timeout check in
    on_tick nor from on_state), each rng is drawn exactly as before, and no
    reschedule or late-reply event is emitted, through replies, a late reply after
    the lifetime and a stretch of timeouts."""
    calls = _snapshot_spy(monkeypatch)
    overrides = {} if mode == "stage1" else {"state_poll_interval_choices_s": (1.0, 3.0)}
    store = Store(tmp_path / "h0.db")
    transport = RecordingTransport()
    jitter, choice, choice_jitter = _Cycle(0.2, 0.8), _Cycle(0.1, 0.7), _Cycle(0.3, 0.6)
    engine = SyncEngine(
        SyncConfig(group="g", node="h0", neighbors=("h1",), **overrides),
        store,
        transport,
        poll_jitter_rng=jitter,
        poll_choice_rng=choice,
        poll_choice_jitter_rng=choice_jitter,
    )
    steps = [("tick", i * 0.05) for i in range(400)]
    steps.insert(3, ("reply", 0.12, 0))
    steps.insert(60, ("reply", 2.95, 0))
    events = _drive(engine, steps)
    polls = len(_poll_names(transport))
    assert polls > 5
    assert len(calls) == polls
    assert {e["kind"] for e in events if e["event"] == "decision"} == {"state_poll"}
    assert {e["event"] for e in events} == {"decision", "state_rx"}
    if mode == "stage1":
        assert (jitter.calls, choice.calls, choice_jitter.calls) == (polls, 0, 0)
    else:
        assert (jitter.calls, choice.calls, choice_jitter.calls) == (0, polls, polls)


def test_policy_logging_adds_no_behaviour(tmp_path: Path):
    """With a policy, log_decisions True vs False sends the same Interests at the same
    times and emits the same non-`decision` events (the late-reply event included);
    the jitter u is drawn once per poll and the two other poll rngs never."""

    def _run(label: str, log_decisions: bool):
        """Drive one policy engine through answers, timeouts, a clamp and a late reply."""
        (tmp_path / label).mkdir()
        engine, transport, spies = _policy_engine(
            tmp_path / label, (3.0, 1.0), (0.1, 0.0), _Cycle(0.1, 0.9, 0.4), log_decisions=log_decisions
        )
        # Poll 0 is answered; poll 2800 (sent on the first 0.05 s tick after its
        # 2.76 s due time) times out at 4.8, which clamps the next poll to 4.8; the
        # reply to poll 2800 at 4.9 is then a late reply to a timed-out poll.
        replies = [("reply", 0.06, 0), ("reply", 4.9, 2800)]
        steps = sorted([("tick", i * 0.05) for i in range(300)] + replies, key=lambda step: (step[1], step[0] == "reply"))
        events = _drive(engine, steps)
        return transport.sent, events, engine._poll_choice_jitter_rng.calls, spies

    sent_on, events_on, u_on, spies_on = _run("on", True)
    sent_off, events_off, u_off, spies_off = _run("off", False)
    assert sent_on == sent_off
    assert [e for e in events_on if e["event"] != "decision"] == [e for e in events_off if e["event"] != "decision"]
    assert any(e["event"] == "state_poll_late_reply" for e in events_on)
    reschedules = _decisions(events_on, "state_poll_reschedule")
    assert {d["action"]["trigger"] for d in reschedules} == {"answered", "timed_out"}
    assert any(d["action"]["clamped"] for d in reschedules)
    assert not any(e["event"] == "decision" for e in events_off)
    polls = sum(1 for n, _ in sent_on if isinstance(names.parse(n), names.StateName))
    assert u_on == u_off == polls
    assert [s.calls for s in (*spies_on.values(), *spies_off.values())] == [0, 0, 0, 0]


_SHORTEN = ((10.043, 3.695), (0.1, 0.1))
_WAIT_5S = ((10.043, 5.0), (0.1, 0.0))


def test_shorten_policy_retries_after_3_695_and_returns_to_10_043(tmp_path: Path):
    """The "shorten" table, measured from each resolved poll's own send time, as
    replay.py does: poll 0 (u 0.3) times out, so the next goes at 0 + 3.695 * (1 +
    0.1 * (2 * 0.3 - 1)); that poll (u 0.8) is answered, so the one after goes at its
    send time + 10.043 * (1 + 0.1 * (2 * 0.8 - 1))."""
    engine, transport, _ = _policy_engine(tmp_path, *_SHORTEN, _Cycle(0.3, 0.8))
    retry = 0 + 3.695 * (1 + 0.1 * (2 * 0.3 - 1))
    back = retry + 10.043 * (1 + 0.1 * (2 * 0.8 - 1))
    _drive(engine, [("tick", 0.0), ("tick", 2.0), ("tick", retry - 0.001)])
    assert _poll_names(transport) == [_state_name(0)]
    _drive(engine, [("tick", retry), ("reply", retry + 0.01, round(retry * 1000)), ("tick", back - 0.001)])
    assert _poll_names(transport) == [_state_name(0), _state_name(round(retry * 1000))]
    _drive(engine, [("tick", back)])
    assert _poll_names(transport)[-1] == _state_name(round(back * 1000))


def test_five_second_wait_policy_retries_exactly_5_s_after_a_failed_poll(tmp_path: Path):
    """The "5 s wait" table: after a timeout the next poll goes exactly 5.0 s after the
    failed poll's send time (no jitter on that row, whatever u = 0.9 was drawn); after
    an answer it goes 10.043 * (1 + 0.1 * (2u - 1)) after the answered poll's send."""
    engine, transport, _ = _policy_engine(tmp_path, *_WAIT_5S, _Cycle(0.9, 0.2))
    _drive(engine, [("tick", 0.0), ("tick", 2.0), ("tick", 4.999)])
    assert _poll_names(transport) == [_state_name(0)]
    back = 5.0 + 10.043 * (1 + 0.1 * (2 * 0.2 - 1))
    _drive(engine, [("tick", 5.0), ("reply", 5.01, 5000), ("tick", back - 0.001)])
    assert _poll_names(transport) == [_state_name(0), _state_name(5000)]
    _drive(engine, [("tick", back)])
    assert _poll_names(transport)[-1] == _state_name(round(back * 1000))
