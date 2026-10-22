"""Tests for SyncEngine against a bare recording Transport (no network)."""

from __future__ import annotations

from pathlib import Path

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
    engine.on_record("h1", mismatched_name, payload, now=0.0)
    assert store.get_record("h1", 1) is None
    assert store.get_record("h1", 2) is None
    assert engine.drain_events() == []


def test_on_record_stores_a_valid_record_and_emits_record_stored(tmp_path: Path):
    engine, _, store = _engine(tmp_path)
    record = schema.IndexRecord(
        origin="h1", seq=1, content_name="x", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    payload = schema.encode(record)
    name = names.build_record_name("g", "h0", "h1", 1)
    engine.on_record("h1", name, payload, now=0.0)
    assert store.get_record("h1", 1) == payload
    assert engine.drain_events() == [{"event": "record_stored", "origin": "h1", "seq": 1, "holder": "h1"}]


def test_on_record_duplicate_is_idempotent_and_silent(tmp_path: Path):
    engine, _, _ = _engine(tmp_path)
    record = schema.IndexRecord(
        origin="h1", seq=1, content_name="x", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    payload = schema.encode(record)
    name = names.build_record_name("g", "h0", "h1", 1)
    engine.on_record("h1", name, payload, now=0.0)
    engine.drain_events()
    engine.on_record("h1", name, payload, now=1.0)
    assert engine.drain_events() == []


def test_on_record_conflict_keeps_first_and_emits_conflict_event(tmp_path: Path):
    engine, _, store = _engine(tmp_path)
    name = names.build_record_name("g", "h0", "h1", 1)
    record_a = schema.IndexRecord(
        origin="h1", seq=1, content_name="a", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    payload_a = schema.encode(record_a)
    engine.on_record("h1", name, payload_a, now=0.0)
    engine.drain_events()

    record_b = schema.IndexRecord(
        origin="h1", seq=1, content_name="b", zone="zone2", disaster="fire", media="video", observed_at=2
    )
    payload_b = schema.encode(record_b)
    engine.on_record("h1", name, payload_b, now=1.0)

    events = engine.drain_events()
    assert len(events) == 1
    assert events[0]["event"] == "conflict"
    assert events[0]["origin"] == "h1"
    assert events[0]["seq"] == 1
    assert events[0]["kept_hash"] == schema.content_hash(payload_a)
    assert events[0]["rejected_hash"] == schema.content_hash(payload_b)
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
    engine, _, _ = _engine(tmp_path)
    name = names.build_record_name("g", "someone-else", "h0", 1)
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


def test_on_fetch_timeout_emits_event_and_backs_off(tmp_path: Path):
    engine, _, _ = _engine(tmp_path, neighbors=("h1",))
    name = names.build_record_name("g", "h1", "origin-x", 3)
    engine.on_fetch_timeout("h1", name, now=0.0)
    assert engine.drain_events() == [
        {"event": "fetch_timeout", "origin": "origin-x", "seq": 3, "neighbor": "h1"}
    ]
