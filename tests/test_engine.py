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
    engine.on_state("h1", payload, now=0.0)
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
    engine.on_state("h1", b"not json", now=0.0)
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
    engine.on_state("h1", payload, now=0.0)  # must not raise
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
    empty for both neighbors."""
    engine, _, _ = _engine(tmp_path, neighbors=("h1", "h2"))  # default interval 2.0s, no jitter offset
    engine.on_tick(now=0.0)
    assert engine.drain_events() == [
        {
            "event": "decision",
            "ts_ms": 0,
            "kind": "state_poll",
            "state": {"in_flight": 0, "last_timeout_s": None, "truncated": False, "origins": {}},
            "action": {"neighbor": "h1", "interval_s": 2.0},
            "action_prob": 1.0,
        },
        {
            "event": "decision",
            "ts_ms": 0,
            "kind": "state_poll",
            "state": {"in_flight": 0, "last_timeout_s": None, "truncated": False, "origins": {}},
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
    choice rng is never drawn -- stage 1's policy and decision lines, unchanged."""
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
        (0, {"neighbor": "h1", "interval_s": 4.0}, 1 / 3),
        (4000, {"neighbor": "h1", "interval_s": 1.0}, 1 / 3),
        (5000, {"neighbor": "h1", "interval_s": 2.0}, 1 / 3),
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
    value and action_prob is 1.0 (the trial runs these to compare intervals)."""
    engine, _, _ = _engine(tmp_path, state_poll_interval_choices_s=(0.5,))
    assert _poll_decisions(engine, [0.0, 0.5, 1.0]) == [
        (0, {"neighbor": "h1", "interval_s": 0.5}, 1.0),
        (500, {"neighbor": "h1", "interval_s": 0.5}, 1.0),
        (1000, {"neighbor": "h1", "interval_s": 0.5}, 1.0),
    ]


def test_decision_logging_leaves_interests_and_other_events_unchanged_under_choices(tmp_path: Path):
    """Stage 1's "logging adds no behavior" property, for the choices policy: with
    log_decisions True vs False, the same Interests go out at the same times, the
    non-`decision` events match, and the choice rng is drawn the same number of
    times (once per poll) while the jitter rng is never drawn at all."""

    def _run(label: str, log_decisions: bool):
        """Drive one engine through polls, a reply and a fetch timeout; return what it sent,
        emitted, and how many times each rng was drawn."""
        store = Store(tmp_path / f"{label}.db")
        transport = RecordingTransport()
        jitter = _Cycle(0.2)
        choice = _Cycle(0.1, 0.7, 0.4)
        config = SyncConfig(
            group="g",
            node="h0",
            neighbors=("h1", "h2"),
            state_poll_interval_choices_s=(0.5, 1.0, 2.0),
            log_decisions=log_decisions,
        )
        engine = SyncEngine(store=store, transport=transport, config=config, poll_jitter_rng=jitter, poll_choice_rng=choice)
        events: list[dict] = []
        engine.on_tick(0.0)
        engine.on_state("h1", holder_state.encode({"origin-x": OriginProgress(hi=1, extra=())}), 0.1)
        events += engine.drain_events()
        engine.on_fetch_timeout(names.build_record_name("g", "h1", "origin-x", 1), 0.3)
        for now in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
            engine.on_tick(now)
            events += engine.drain_events()
        return transport.sent, events, choice.calls, jitter.calls

    sent_on, events_on, choice_on, jitter_on = _run("on", True)
    sent_off, events_off, choice_off, jitter_off = _run("off", False)
    assert sent_on == sent_off
    assert [e for e in events_on if e["event"] != "decision"] == [e for e in events_off if e["event"] != "decision"]
    polls = sum(1 for n, _ in sent_on if isinstance(names.parse(n), names.StateName))
    assert choice_on == polls
    assert choice_off == polls
    assert jitter_on == 0
    assert jitter_off == 0
    assert any(e["event"] == "decision" for e in events_on)
    assert not any(e["event"] == "decision" for e in events_off)
