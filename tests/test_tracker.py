"""Tests for the pure missing-record tracker and fetch scheduler."""

from __future__ import annotations

from semindex.tracker import DEFAULT_MAX_IN_FLIGHT_PER_NEIGHBOR, Tracker


def test_gaps_1_2_4_give_contiguous_2_and_missing_3():
    tracker = Tracker()
    # This node already holds 1,2 contiguously and 4 sparsely; a
    # neighbor has told it seq 4 is the highest it knows of.
    tracker.observe_held("h0", hi=2, extra=(4,))
    tracker.observe_state("h0", "n1", hi=4, extra=())
    assert tracker.contiguous("h0") == 2
    assert tracker.notified_max("h0") == 4
    # 3 is the only seq in (contiguous, notified_max] not already held.
    assert tracker.missing("h0") == [3]


def test_missing_is_empty_when_caught_up():
    tracker = Tracker()
    tracker.observe_held("h0", hi=3, extra=())
    tracker.observe_state("h0", "n1", hi=3, extra=())
    assert tracker.missing("h0") == []


def test_plan_fetches_only_targets_a_neighbor_that_advertised_the_seq():
    tracker = Tracker()
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=())  # n1 only has seq 1
    tracker.observe_state("h0", "n2", hi=3, extra=())  # n2 has 1..3
    # notified_max is 3 (max across neighbors), so missing is [1, 2, 3],
    # but n1 only actually holds seq 1.
    assert tracker.missing("h0") == [1, 2, 3]
    assert tracker.plan_fetches("n1", now=0.0) == [("h0", 1)]
    assert tracker.plan_fetches("n2", now=0.0) == [("h0", 2), ("h0", 3)]


def test_in_flight_cap_per_neighbor():
    tracker = Tracker()
    many = tuple(range(2, 2 + 20))
    tracker.observe_held("h0", hi=1, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=many)
    due = tracker.plan_fetches("n1", now=0.0)
    assert len(due) == DEFAULT_MAX_IN_FLIGHT_PER_NEIGHBOR == 8
    # Still at the cap: no more are handed out until some are released.
    assert tracker.plan_fetches("n1", now=0.0) == []


def test_backoff_schedule_doubles_from_1s_capped_at_16s():
    # map.md "再送間隔 1 秒から倍々、16 秒で頭打ち": the wait *after* a
    # timeout is the backoff that was in effect for that attempt (1s the
    # first time), and only the backoff for the *next* attempt doubles.
    tracker = Tracker()
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=())

    # First attempt is due immediately.
    assert tracker.plan_fetches("n1", now=0.0) == [("h0", 1)]
    tracker.on_fetch_timeout("h0", 1, "n1", now=0.0)  # waits 1s, due at 1.0

    assert tracker.plan_fetches("n1", now=0.9) == []
    assert tracker.plan_fetches("n1", now=1.0) == [("h0", 1)]
    tracker.on_fetch_timeout("h0", 1, "n1", now=1.0)  # waits 2s, due at 3.0

    assert tracker.plan_fetches("n1", now=2.9) == []
    assert tracker.plan_fetches("n1", now=3.0) == [("h0", 1)]
    tracker.on_fetch_timeout("h0", 1, "n1", now=3.0)  # waits 4s, due at 7.0

    assert tracker.plan_fetches("n1", now=6.9) == []
    assert tracker.plan_fetches("n1", now=7.0) == [("h0", 1)]
    tracker.on_fetch_timeout("h0", 1, "n1", now=7.0)  # waits 8s, due at 15.0

    assert tracker.plan_fetches("n1", now=14.9) == []
    assert tracker.plan_fetches("n1", now=15.0) == [("h0", 1)]
    tracker.on_fetch_timeout("h0", 1, "n1", now=15.0)  # waits 16s (uncapped would also be 16), due at 31.0

    assert tracker.plan_fetches("n1", now=30.9) == []
    assert tracker.plan_fetches("n1", now=31.0) == [("h0", 1)]
    tracker.on_fetch_timeout("h0", 1, "n1", now=31.0)  # waits min(32, 16) = 16s, due at 47.0: the cap holds

    assert tracker.plan_fetches("n1", now=46.9) == []
    assert tracker.plan_fetches("n1", now=47.0) == [("h0", 1)]


def test_observe_held_releases_in_flight_slot_when_seq_arrives_from_elsewhere():
    tracker = Tracker()
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=())
    assert tracker.plan_fetches("n1", now=0.0) == [("h0", 1)]

    # Seq 1 arrives (e.g. via a different neighbor, or locally); the
    # in-flight slot on n1 must be released, not leaked.
    tracker.observe_held("h0", hi=1, extra=())
    assert tracker.missing("h0") == []

    # n1's slot is free again: a brand new missing seq can use it.
    tracker.observe_state("h0", "n1", hi=2, extra=())
    assert tracker.plan_fetches("n1", now=0.0) == [("h0", 2)]


def test_stale_fetch_timeout_from_a_reassigned_neighbor_is_ignored():
    tracker = Tracker()
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=())
    tracker.plan_fetches("n1", now=0.0)

    # The seq got satisfied by some other means before the timeout for
    # n1 arrived; the stale timeout must not resurrect tracking for it.
    tracker.observe_held("h0", hi=1, extra=())
    tracker.on_fetch_timeout("h0", 1, "n1", now=5.0)
    assert tracker.missing("h0") == []
