"""Tests for the pure missing-record tracker and fetch scheduler."""

from __future__ import annotations

from semindex.tracker import (
    DEFAULT_MAX_IN_FLIGHT_PER_NEIGHBOR,
    DEFAULT_MAX_MISSING_PER_ORIGIN,
    Tracker,
)


def test_gaps_1_2_4_give_contiguous_2_and_missing_3():
    """Holding 1, 2 contiguously and 4 sparsely, with a neighbor advertising up to 4, leaves exactly 3 missing."""
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
    """Nothing is missing once held and notified are the same seq."""
    tracker = Tracker()
    tracker.observe_held("h0", hi=3, extra=())
    tracker.observe_state("h0", "n1", hi=3, extra=())
    assert tracker.missing("h0") == []


def test_plan_fetches_only_targets_a_neighbor_that_advertised_the_seq():
    """plan_fetches only ever assigns a seq to a neighbor that itself advertised holding it, never to one that merely raised notified_max."""
    tracker = Tracker()
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=())  # n1 only has seq 1
    tracker.observe_state("h0", "n2", hi=3, extra=())  # n2 has 1..3
    # notified_max is 3 (max across neighbors), so missing is [1, 2, 3],
    # but n1 only actually holds seq 1.
    assert tracker.missing("h0") == [1, 2, 3]
    assert tracker.plan_fetches("n1", now=0.0) == [("h0", 1)]
    assert tracker.plan_fetches("n2", now=0.0) == [("h0", 2), ("h0", 3)]


def test_missing_is_capped_at_max_missing_per_origin():
    """map.md "origin ごとの未取得管理上限 1024": a huge gap must not make
    missing() (or how many fetches plan_fetches hands out from it) grow
    unbounded with the gap size.
    """
    tracker = Tracker()
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=5000, extra=())

    missing = tracker.missing("h0")
    assert len(missing) == DEFAULT_MAX_MISSING_PER_ORIGIN == 1024
    assert missing == list(range(1, 1025))

    # plan_fetches is capped independently, by max_in_flight_per_neighbor:
    # only 8 of those 1024 missing seqs are handed out in one call, even
    # though n1 advertised holding all the way up to seq 5000.
    due = tracker.plan_fetches("n1", now=0.0)
    assert due == [("h0", seq) for seq in range(1, 1 + DEFAULT_MAX_IN_FLIGHT_PER_NEIGHBOR)]


def test_in_flight_cap_per_neighbor():
    """plan_fetches never exceeds max_in_flight_per_neighbor fetches outstanding on one neighbor at a time."""
    tracker = Tracker()
    many = tuple(range(2, 2 + 20))
    tracker.observe_held("h0", hi=1, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=many)
    due = tracker.plan_fetches("n1", now=0.0)
    assert len(due) == DEFAULT_MAX_IN_FLIGHT_PER_NEIGHBOR == 8
    # Still at the cap: no more are handed out until some are released.
    assert tracker.plan_fetches("n1", now=0.0) == []


def test_backoff_schedule_doubles_from_1s_capped_at_16s():
    """Each timeout's wait is the backoff already in effect; only the next attempt's backoff doubles, capped at 16s."""
    # map.md "再送間隔 1 秒から倍々、16 秒で頭打ち": the wait *after* a
    # timeout is the backoff that was in effect for that attempt (1s the
    # first time), and only the backoff for the *next* attempt doubles.
    # This test is about the doubling/cap mechanics, not the live default
    # (2s as of 2026-10-23, see tracker.DEFAULT_BACKOFF_INITIAL_S), so it
    # pins backoff_initial_s=1.0 explicitly to keep its own numbers true
    # regardless of what the default is.
    tracker = Tracker(backoff_initial_s=1.0)
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
    """A seq that arrives by any means releases its in-flight slot on whichever neighbor it was outstanding on, freeing that slot for a new fetch."""
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


def test_plan_fetches_gives_the_retry_to_another_advertiser_once_one_neighbor_fails():
    """Once a neighbor fails a seq, its retry goes to another neighbor that also advertised it, not back to the one that just failed."""
    # Two neighbors both advertise seq 1; n1 is the one that keeps
    # failing. Without failed-neighbor exclusion, the retry would go
    # straight back to n1 every time (plan_fetches is called for n1
    # before n2 on every real on_tick, since neighbor order is fixed),
    # even though n2 also advertised holding it and could serve it.
    # This test is about failover-to-another-advertiser logic, not the
    # live backoff default, so backoff_initial_s=1.0 is pinned explicitly
    # to keep its timing numbers decoupled from that default.
    tracker = Tracker(backoff_initial_s=1.0)
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=())
    tracker.observe_state("h0", "n2", hi=1, extra=())

    assert tracker.plan_fetches("n1", now=0.0) == [("h0", 1)]
    assert tracker.plan_fetches("n2", now=0.0) == []  # n1 already claimed it this round
    tracker.on_fetch_timeout("h0", 1, "n1", now=0.0)  # waits 1s, due at 1.0

    # At the next due time, n1 is excluded because n2 also advertised
    # the seq and has not failed it; n2 gets this retry instead.
    assert tracker.plan_fetches("n1", now=1.0) == []
    assert tracker.plan_fetches("n2", now=1.0) == [("h0", 1)]


def test_plan_fetches_cycles_through_three_advertisers_that_all_keep_failing():
    """Once every advertiser of a seq has failed it once, the failed set resets so the round-robin keeps cycling instead of locking up."""
    # Three neighbors advertise the same seq and all three keep timing
    # out. A failed set that only ever grows would, once it covered all
    # three, make the exclusion check vacuous and hand every further
    # retry back to whichever neighbor on_tick visits first -- the same
    # starvation this bookkeeping exists to prevent. It must instead
    # reset once everyone has failed once, so the round-robin continues.
    # Pinned to 1.0: this test is about the round-robin reset, not the
    # live backoff default.
    tracker = Tracker(backoff_initial_s=1.0)
    tracker.observe_held("h0", hi=0, extra=())
    for neighbor in ("n1", "n2", "n3"):
        tracker.observe_state("h0", neighbor, hi=1, extra=())

    assert tracker.plan_fetches("n1", now=0.0) == [("h0", 1)]
    tracker.on_fetch_timeout("h0", 1, "n1", now=0.0)  # failed={n1}, due at 1.0

    assert tracker.plan_fetches("n1", now=1.0) == []
    assert tracker.plan_fetches("n2", now=1.0) == [("h0", 1)]
    tracker.on_fetch_timeout("h0", 1, "n2", now=1.0)  # failed={n1,n2}, due at 3.0

    assert tracker.plan_fetches("n1", now=3.0) == []
    assert tracker.plan_fetches("n2", now=3.0) == []
    assert tracker.plan_fetches("n3", now=3.0) == [("h0", 1)]  # the only one not yet failed
    tracker.on_fetch_timeout("h0", 1, "n3", now=3.0)  # every advertiser has now failed once: reset

    # The failed set reset, so n1 is eligible again and the cycle
    # continues rather than locking up on n2/n3 forever.
    assert tracker.plan_fetches("n1", now=7.0) == [("h0", 1)]


def test_on_fetch_rejected_releases_the_slot_and_backs_off_like_a_timeout():
    """A rejected reply (failed validation, not a timeout) still releases the in-flight slot and backs off exactly like a timeout would."""
    # A reply that fails validation (schema.py / engine.on_record) is a
    # different outcome from a network timeout -- Data did arrive -- but
    # it must free the in-flight slot and apply the same backoff, or the
    # fetch would stay marked in-flight forever (no further timeout is
    # ever coming for an Interest that was already answered).
    # Pinned to 1.0: this test is about the rejected-reply path reusing
    # the timeout backoff logic, not the live backoff default.
    tracker = Tracker(backoff_initial_s=1.0)
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=())
    assert tracker.plan_fetches("n1", now=0.0) == [("h0", 1)]

    tracker.on_fetch_rejected("h0", 1, "n1", now=0.0)
    assert tracker.plan_fetches("n1", now=0.9) == []
    assert tracker.plan_fetches("n1", now=1.0) == [("h0", 1)]


def test_stale_fetch_timeout_from_a_reassigned_neighbor_is_ignored():
    """A timeout for a fetch already satisfied by other means is stale and must not resurrect tracking for it."""
    tracker = Tracker()
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=())
    tracker.plan_fetches("n1", now=0.0)

    # The seq got satisfied by some other means before the timeout for
    # n1 arrived; the stale timeout must not resurrect tracking for it.
    tracker.observe_held("h0", hi=1, extra=())
    tracker.on_fetch_timeout("h0", 1, "n1", now=5.0)
    assert tracker.missing("h0") == []


def test_default_backoff_initial_clears_cefores_pit_grace_window():
    """Pins the live default: the first same-name retry after a fetch_timeout must not be
    planned before 2s, strictly after the 1s grace Cefore v0.12's PIT keeps a cleaned-up entry
    for past an expired Interest's own lifetime (cef_pit.h CefC_Pit_CleaningTime). A retry
    scheduled inside that window would be silently absorbed by the PIT and never reach the
    producer at all -- this guarantees the tracker's own default never regresses back to a value
    that wastes a retry that way."""
    tracker = Tracker()  # no override: exercises DEFAULT_BACKOFF_INITIAL_S itself
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=())
    assert tracker.plan_fetches("n1", now=0.0) == [("h0", 1)]

    tracker.on_fetch_timeout("h0", 1, "n1", now=0.0)

    # Strictly after the 1s PIT grace: nothing is due yet at 1.0s.
    assert tracker.plan_fetches("n1", now=1.0) == []
    assert tracker.plan_fetches("n1", now=2.0) == [("h0", 1)]


# -- decision-event support (map.md "判断の記録") ---------------------------------


def test_on_fetch_timeout_returns_the_chosen_wait():
    """on_fetch_timeout returns the backoff wait it just scheduled: the seconds chosen
    before the next attempt, for a real, in-flight fetch that actually failed."""
    tracker = Tracker(backoff_initial_s=3.0, jitter=lambda: 0.0)
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=())
    tracker.plan_fetches("n1", now=0.0)
    assert tracker.on_fetch_timeout("h0", 1, "n1", now=10.0) == 3.0


def test_on_fetch_rejected_returns_the_chosen_wait():
    """on_fetch_rejected shares _fail with on_fetch_timeout, so it returns the same way."""
    tracker = Tracker(backoff_initial_s=2.0, jitter=lambda: 0.0)
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=())
    tracker.plan_fetches("n1", now=0.0)
    assert tracker.on_fetch_rejected("h0", 1, "n1", now=10.0) == 2.0


def test_stale_fetch_timeout_returns_none():
    """A timeout/rejection that _fail treats as stale (nothing to resolve) returns None, so
    SyncEngine logs no `decision` event for it -- no choice was actually made."""
    tracker = Tracker()
    assert tracker.on_fetch_timeout("never-seen", 1, "n1", now=0.0) is None

    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=())
    tracker.plan_fetches("n1", now=0.0)
    tracker.observe_held("h0", hi=1, extra=())  # satisfied by other means
    assert tracker.on_fetch_timeout("h0", 1, "n1", now=5.0) is None


def test_in_flight_count_reflects_outstanding_fetches_on_a_neighbor():
    """in_flight_count mirrors plan_fetches'/_fail's own bookkeeping, read-only."""
    tracker = Tracker()
    assert tracker.in_flight_count("n1") == 0
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=2, extra=())
    tracker.plan_fetches("n1", now=0.0)
    assert tracker.in_flight_count("n1") == 2
    tracker.on_fetch_timeout("h0", 1, "n1", now=0.0)
    assert tracker.in_flight_count("n1") == 1


def test_last_timeout_at_is_none_until_a_real_timeout_then_tracks_it():
    """last_timeout_at(neighbor) is None until _fail actually resolves a timeout on it,
    then holds that timeout's `now` -- a stale timeout (see
    test_stale_fetch_timeout_returns_none) must not set it, since nothing actually
    failed."""
    tracker = Tracker()
    assert tracker.last_timeout_at("n1") is None
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=())
    tracker.plan_fetches("n1", now=0.0)
    tracker.on_fetch_timeout("h0", 1, "n1", now=7.5)
    assert tracker.last_timeout_at("n1") == 7.5

    # A later stale timeout (already resolved) must not overwrite it.
    tracker.on_fetch_timeout("h0", 1, "n1", now=99.0)
    assert tracker.last_timeout_at("n1") == 7.5


def test_last_timeout_at_ignores_a_rejected_reply():
    """A rejected reply must not count as a timeout for last_timeout_at: timing out at
    now=3 and then, on a re-fetch, rejecting a bad reply at now=6 must still report
    the timeout's own time (3.0), not the rejection's.

    2026-10-23 bug fix: _fail used to record
    `now` in the same dict for both a timeout and a rejected reply, so the later
    rejection silently overwrote the timeout's own time, even though decision_state
    reports this value as "last_timeout_s" -- map.md "判断の記録"'s "直前のタイムア
    ウトの時刻" means a timeout specifically.

    2026-10-23 test fix: the re-fetch used to be planned at now=3.0, the exact
    instant the timeout itself fired -- but backoff_initial_s=1.0 means the retry
    is not actually due until now=4.0, so plan_fetches at 3.0 claimed nothing
    ([]), on_fetch_rejected at 6.0 found no in-flight fetch to resolve and
    returned None through _fail's own stale early-return, and the final assert
    passed on a value the rejection had never touched at all -- it would have
    passed identically against the pre-fix code this test exists to pin.
    """
    tracker = Tracker(backoff_initial_s=1.0, jitter=lambda: 0.0)
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=1, extra=())

    tracker.plan_fetches("n1", now=0.0)
    tracker.on_fetch_timeout("h0", 1, "n1", now=3.0)
    assert tracker.last_timeout_at("n1") == 3.0

    # Due only once the 1s backoff has actually elapsed (now=4.0, not 3.0).
    assert tracker.plan_fetches("n1", now=4.0) == [("h0", 1)]
    wait = tracker.on_fetch_rejected("h0", 1, "n1", now=6.0)
    assert wait is not None  # proves this rejection actually resolved a real in-flight fetch
    assert tracker.last_timeout_at("n1") == 3.0


def test_decision_state_missing_is_not_capped_by_the_fetch_planning_limit():
    """decision_state's own `missing` count must be the true backlog size, not
    missing()'s own list, which fetch planning deliberately caps at
    max_missing_per_origin (DEFAULT_MAX_MISSING_PER_ORIGIN=1024) so a single due
    pass stays bounded.

    2026-10-23 bug fix: decision_state used to
    report len(missing()), so a run with a backlog beyond the cap showed exactly
    1024 forever -- indistinguishable from a backlog of exactly 1024 -- hiding how
    much worse a run that has fallen far behind actually is.
    """
    tracker = Tracker(max_missing_per_origin=1024)
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=2000, extra=())
    assert len(tracker.missing("h0")) == 1024  # fetch planning's own cap, unchanged
    state = tracker.decision_state("n1")
    assert state["origins"]["h0"]["missing"] == 2000


def test_missing_total_excludes_only_the_held_extras_inside_the_range():
    """missing_total's arithmetic form (span size minus held extras within that span)
    must agree with missing()'s own seq-by-seq enumeration, including a held extra
    that falls outside (held_contiguous, notified_max] -- seq 1, already covered by
    held_contiguous itself -- which must not be subtracted a second time."""
    tracker = Tracker()
    tracker.observe_held("h0", hi=2, extra=(1, 5, 9))
    tracker.observe_state("h0", "n1", hi=10, extra=())
    assert tracker.missing("h0") == [3, 4, 6, 7, 8, 10]
    assert tracker.missing_total("h0") == 6


def test_missing_total_runs_in_constant_time_regardless_of_notified_max():
    """missing_total must not enumerate every seq between held_contiguous and
    notified_max to count them: on an empty local state, a single neighbor
    advertising hi=10_000_000 used to take on the order of a tenth of a second to
    answer this one call -- and on_tick calls it, through decision_state, for
    every neighbor on every tick of this single-threaded agent's loop, whether or
    not anything is actually due. The arithmetic form answers in microseconds no
    matter how high hi is.
    """
    import time

    tracker = Tracker()
    tracker.observe_state("h0", "n1", hi=10_000_000, extra=())
    started = time.perf_counter()
    total = tracker.missing_total("h0")
    elapsed = time.perf_counter() - started
    assert total == 10_000_000
    assert elapsed < 0.05


def test_decision_state_reports_per_origin_hi_missing_and_every_neighbors_advertisement():
    """decision_state(neighbor) summarizes, per origin this tracker knows of: this node's
    own contiguous hold (`hi`), the origin's true missing count, and `advertised` --
    every neighbor that has advertised this origin, including ones other than
    `neighbor` itself (origin-y here was only ever advertised by n2, not n1), each
    with its own {hi, notified_max} -- plus `neighbor`'s own in_flight count and
    last_timeout_at."""
    tracker = Tracker()
    tracker.observe_held("h0", hi=1, extra=())
    tracker.observe_state("h0", "n1", hi=3, extra=())  # n1 advertised up to 3
    tracker.observe_held("origin-y", hi=0, extra=())
    tracker.observe_state("origin-y", "n2", hi=5, extra=())  # n1 never advertised origin-y

    state = tracker.decision_state("n1")
    assert state == {
        "in_flight": 0,
        "last_timeout_s": None,
        "truncated": False,
        "origins": {
            "h0": {"hi": 1, "missing": 2, "advertised": {"n1": {"hi": 3, "notified_max": 3}}},
            "origin-y": {"hi": 0, "missing": 5, "advertised": {"n2": {"hi": 5, "notified_max": 5}}},
        },
    }


def test_decision_state_distinguishes_contiguous_from_sparse_advertisements():
    """Two neighbors advertising the same origin can reach the same highest known
    seq by different means -- n1 holds 1..5 contiguously, n2 holds only the single
    sparse seq 5 -- and a fetch from n2 for seq 3 would fail where one from n1
    would not (see Tracker.plan_fetches's own `seq > hi and seq not in extra_set`
    check), so decision_state must keep `hi` (contiguous) and `notified_max`
    (highest known at all) distinct for every neighbor shown, and show every
    neighbor's advertisement, not only the one this decision concerns.

    2026-10-23 bug fix: decision_state used to report `neighbor`-only, collapsing
    (hi, extra) into one max() -- these two advertisements produced the identical
    adv=5, and n2's row (and n1's, had the decision concerned n2) was absent.
    """
    tracker = Tracker()
    tracker.observe_held("h0", hi=0, extra=())
    tracker.observe_state("h0", "n1", hi=5, extra=())  # n1: contiguous 1..5
    tracker.observe_state("h0", "n2", hi=0, extra=(5,))  # n2: only sparse seq 5

    state = tracker.decision_state("n1")
    assert state["origins"]["h0"]["advertised"] == {
        "n1": {"hi": 5, "notified_max": 5},
        "n2": {"hi": 0, "notified_max": 5},
    }


def test_decision_state_truncates_beyond_the_origin_cap():
    """decision_state caps how many origins it reports (DECISION_STATE_MAX_ORIGINS, lowest
    name first) so a `decision` event's size cannot grow with how many origins a run
    tracks; `truncated` says whether any were left out."""
    tracker = Tracker()
    cap = Tracker.DECISION_STATE_MAX_ORIGINS
    for i in range(cap + 3):
        origin = f"o{i:02d}"
        tracker.observe_held(origin, hi=0, extra=())
        tracker.observe_state(origin, "n1", hi=1, extra=())

    state = tracker.decision_state("n1")
    assert state["truncated"] is True
    assert sorted(state["origins"]) == [f"o{i:02d}" for i in range(cap)]
    assert len(state["origins"]) == cap
