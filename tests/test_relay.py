"""Integration tests driving multiple SyncEngines over tests.fakes.FakeNetwork."""

from __future__ import annotations

from pathlib import Path

from semindex.engine import SyncConfig, SyncEngine
from semindex.store import Store

from fakes import FakeNetwork

ROUND_INTERVAL_S = 2.0  # matches SyncConfig.state_poll_interval_s default


def _build_node(tmp_path: Path, network: FakeNetwork, node: str, neighbors: tuple[str, ...], **overrides) -> SyncEngine:
    store = Store(tmp_path / f"{node}.db")
    config = SyncConfig(group="g", node=node, neighbors=neighbors, **overrides)
    engine = SyncEngine(config, store, network.transport_for(node))
    network.add_node(node, engine)
    return engine


def _run_round(engines: list[SyncEngine], network: FakeNetwork, now: float) -> None:
    """One simulated round: every engine ticks, then the network drains fully.

    Draining with several pump calls (rather than one) resolves cascades
    within the round -- a state reply can itself trigger fetch Interests
    (SyncEngine.on_state's immediate "fetch plan"), which land in a fresh
    queue that only the *next* pump call resolves.
    """
    for engine in engines:
        engine.on_tick(now)
    for _ in range(4):
        network.pump(now)


def test_a_to_b_to_c_relay_survives_a_partition_after_b_synced(tmp_path: Path):
    network = FakeNetwork()
    a = _build_node(tmp_path, network, "A", neighbors=("B",))
    b = _build_node(tmp_path, network, "B", neighbors=("A", "C"))
    c = _build_node(tmp_path, network, "C", neighbors=("B",))
    engines = [a, b, c]

    # A is the Index origin for 3 records, created before any sync starts.
    for i in range(3):
        a.create_record(
            content_name=f"ccnx:/content/A/{i + 1}", zone="zone1", disaster="flood", media="image", observed_at=i
        )

    now = 0.0
    for _ in range(3):
        _run_round(engines, network, now)
        now += ROUND_INTERVAL_S

    # Direct neighbor B is fully caught up...
    assert b._store.get_progress("A").hi == 3
    # ...and it has relayed to C, which is not A's direct neighbor.
    assert c._store.get_progress("A").hi == 3

    # A is now cut off from B. While partitioned, A creates 2 more
    # records that must not reach B or C at all.
    network.partition("A", "B")
    for i in range(3, 5):
        a.create_record(
            content_name=f"ccnx:/content/A/{i + 1}", zone="zone1", disaster="flood", media="image", observed_at=i
        )

    for _ in range(3):
        _run_round(engines, network, now)
        now += ROUND_INTERVAL_S
    assert b._store.get_progress("A").hi == 3
    assert c._store.get_progress("A").hi == 3

    # Heal the link; both B and then C should catch all the way up to 5.
    network.heal("A", "B")
    for _ in range(4):
        _run_round(engines, network, now)
        now += ROUND_INTERVAL_S

    assert b._store.get_progress("A") == a._store.get_progress("A")
    assert c._store.get_progress("A").hi == 5
    for seq in range(1, 6):
        assert c._store.get_record("A", seq) == a._store.get_record("A", seq)


def test_drop_rec_fault_delays_but_does_not_prevent_recovery(tmp_path: Path):
    network = FakeNetwork()
    # B is the holder being asked for (origin=B, seq=3); B's own
    # handle_interest is what fault_drop_rec gates, so it is B's config
    # that withholds the reply, twice, before A can converge.
    a = _build_node(tmp_path, network, "A", neighbors=("B",))
    b = _build_node(tmp_path, network, "B", neighbors=("A",), fault_drop_rec={("B", 3): 2})
    engines = [a, b]

    for i in range(3):
        b.create_record(
            content_name=f"ccnx:/content/B/{i + 1}", zone="zone1", disaster="flood", media="image", observed_at=i
        )

    now = 0.0
    timeouts_for_seq_3 = 0
    stored_seq_3 = False
    for _ in range(12):
        _run_round(engines, network, now)
        for event in a.drain_events():
            if event["event"] == "fetch_timeout" and event["origin"] == "B" and event["seq"] == 3:
                timeouts_for_seq_3 += 1
            if event["event"] == "record_stored" and event["origin"] == "B" and event["seq"] == 3:
                stored_seq_3 = True
        now += ROUND_INTERVAL_S
        if stored_seq_3:
            break

    assert timeouts_for_seq_3 == 2
    assert stored_seq_3 is True
    assert a._store.get_progress("B").hi == 3


def test_requester_retries_via_a_second_holder_once_the_first_partitions(tmp_path: Path):
    # R has two direct neighbors that both can answer for origin A's
    # records: A itself, and B (already synced from A). R's config lists
    # A before B, so its first fetch attempt lands on A. A then
    # partitions from R specifically (B does not), and every one of
    # those fetches must eventually be retried against B instead of
    # being retried against A forever (tracker.py's failed_neighbors
    # bookkeeping; see test_tracker.py for the mechanism in isolation).
    network = FakeNetwork()
    a = _build_node(tmp_path, network, "A", neighbors=())
    b = _build_node(tmp_path, network, "B", neighbors=("A",))
    r = _build_node(tmp_path, network, "R", neighbors=("A", "B"))

    for i in range(3):
        a.create_record(
            content_name=f"ccnx:/content/A/{i + 1}", zone="zone1", disaster="flood", media="image", observed_at=i
        )

    # Sync A -> B first, with R not yet ticking, so B is fully caught up
    # before R asks anyone.
    now = 0.0
    for _ in range(3):
        _run_round([a, b], network, now)
        now += ROUND_INTERVAL_S
    assert b._store.get_progress("A").hi == 3

    # R ticks once: it learns from both A and B that origin A has hi=3,
    # and claims its first fetch attempt for seq 1..3 against A (the
    # first neighbor state request to resolve, since A is listed first).
    r.on_tick(now)
    network.pump(now)  # resolves the two state replies; queues the fetch Interests to holder A
    r.drain_events()

    # Cut A off from R right after it claimed those fetches but before
    # any of them are resolved, so every one of them times out.
    network.partition("A", "R")
    network.pump(now)  # resolves the queued fetch Interests to A: all 3 time out
    timeouts = [e for e in r.drain_events() if e["event"] == "fetch_timeout" and e["origin"] == "A"]
    assert {e["seq"] for e in timeouts} == {1, 2, 3}

    stored: dict[int, str] = {}
    for _ in range(4):
        _run_round([a, b, r], network, now)
        for event in r.drain_events():
            if event["event"] == "record_stored" and event["origin"] == "A":
                stored[event["seq"]] = event["holder"]
        now += ROUND_INTERVAL_S

    assert r._store.get_progress("A").hi == 3
    assert stored == {1: "B", 2: "B", 3: "B"}
