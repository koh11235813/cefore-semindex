"""Per-origin missing-record tracking and fetch scheduling.

Pure: no I/O, no wall-clock reads. Every method that needs "now" takes it
as an argument, and the only randomness (retry jitter) comes from an
injected callable, so this module is driven byte-for-byte by tests and by
the real event loop alike.

Three distinct quantities per origin (CONTEXT.md "Holder state", map.md
"同期プロトコル"), never conflated:
  - notified_max: the highest seq this node has been told about by any
    neighbor (the max of every advertised hi and extra entry).
  - contiguous: the highest seq this node holds contiguously from 1
    (mirrors store.OriginProgress.hi for that origin).
  - missing: seqs between contiguous+1 and notified_max that this node
    neither holds contiguously nor already holds sparsely (its own
    extra). Receiving 1, 2 and 4 must never mark 3 as held.

A fetch is only ever sent to a neighbor that actually advertised holding
that seq (its hi or extra): a neighbor's advertised (hi, extra) is kept
per neighbor, not merged away, precisely so plan_fetches can tell who can
answer -- fetching from a neighbor that cannot would just manufacture
fetch_timeout events and backoff with no chance of ever succeeding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

# 2026-10-23: live T2a showed Cefore v0.12 keeps a PIT entry for
# interest_lifetime + CefC_Pit_CleaningTime (1s; cefore
# src/include/cefore/cef_pit.h:55, lib/cef_pit.c:744) after the Interest
# that created it expires. With a 2s interest_lifetime and a 1s initial
# backoff, the first same-name retry landed inside that window and was
# silently absorbed by the PIT -- never reaching the producer -- wasting
# one retry on every single fetch. 2s clears the PIT's own grace window.
DEFAULT_BACKOFF_INITIAL_S = 2.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0
DEFAULT_BACKOFF_MAX_S = 16.0
DEFAULT_MAX_MISSING_PER_ORIGIN = 1024
DEFAULT_MAX_IN_FLIGHT_PER_NEIGHBOR = 8

# 2026-10-23 (map.md "判断の記録"): Tracker.decision_state reports at most this many
# origins (lowest name first) so one `decision` event's size cannot grow with how many
# origins a run tracks -- a run config is expected to have a handful of origins (this
# effort's own baseline has 2), so the cap is generous headroom, not a tuned limit.
DECISION_STATE_MAX_ORIGINS = 8


def no_jitter() -> float:
    """Default jitter callable: always zero, for deterministic callers and tests."""
    return 0.0


@dataclass
class _FetchState:
    """Retry bookkeeping for one (origin, seq) this node is trying to fetch.

    `in_flight_neighbor` is set while a fetch Interest is outstanding on
    that neighbor and cleared on timeout; `next_attempt_at` is only
    meaningful while it is None (not currently in flight).

    `failed_neighbors` is the set of neighbors that have failed this
    exact (origin, seq) since the last time the set covered every
    current advertiser. It exists to stop a single unreachable neighbor
    from permanently hogging every retry of a seq that another neighbor
    also advertises: without it, plan_fetches would hand the retry back
    to whichever neighbor happens to come first in on_tick's fixed
    iteration order every time, even though that neighbor is the one
    that just failed.
    """

    next_attempt_at: float
    backoff_s: float
    in_flight_neighbor: str | None = None
    failed_neighbors: frozenset[str] = frozenset()


@dataclass
class _OriginState:
    """Per-origin bookkeeping: what each neighbor advertised, what we hold, outstanding fetches."""

    advertised: dict[str, tuple[int, tuple[int, ...]]] = field(default_factory=dict)  # neighbor -> (hi, extra)
    held_contiguous: int = 0
    held_extra: frozenset[int] = frozenset()
    fetches: dict[int, _FetchState] = field(default_factory=dict)  # seq -> fetch state


class Tracker:
    """Tracks missing Index records per origin and schedules their retrieval."""

    # Exposed as a class attribute (mirrors DECISION_STATE_MAX_ORIGINS above) so a
    # caller -- or a test pinning the cap in its own assertions -- can read it off the
    # class without reaching into the module's globals.
    DECISION_STATE_MAX_ORIGINS = DECISION_STATE_MAX_ORIGINS

    def __init__(
        self,
        *,
        backoff_initial_s: float = DEFAULT_BACKOFF_INITIAL_S,
        backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
        backoff_max_s: float = DEFAULT_BACKOFF_MAX_S,
        max_missing_per_origin: int = DEFAULT_MAX_MISSING_PER_ORIGIN,
        max_in_flight_per_neighbor: int = DEFAULT_MAX_IN_FLIGHT_PER_NEIGHBOR,
        jitter: Callable[[], float] = no_jitter,
    ) -> None:
        """Store the retry/backoff/cap configuration; all per-origin state starts empty."""
        self._backoff_initial_s = backoff_initial_s
        self._backoff_multiplier = backoff_multiplier
        self._backoff_max_s = backoff_max_s
        self._max_missing_per_origin = max_missing_per_origin
        self._max_in_flight_per_neighbor = max_in_flight_per_neighbor
        self._jitter = jitter
        self._origins: dict[str, _OriginState] = {}
        self._in_flight_count: dict[str, int] = {}  # neighbor -> count, across all origins
        # 2026-10-23 (map.md "判断の記録"): the `now` of the most recent fetch
        # failure (timeout or rejected reply) _fail actually resolved on this
        # neighbor, across any origin/seq -- read-only bookkeeping for
        # Tracker.decision_state; never consulted by any scheduling decision.
        self._last_fail_at: dict[str, float] = {}

    def _origin(self, origin: str) -> _OriginState:
        """Return this origin's bookkeeping, creating empty state on first mention."""
        return self._origins.setdefault(origin, _OriginState())

    # -- observing ------------------------------------------------------------

    def observe_state(self, origin: str, neighbor: str, hi: int, extra: Iterable[int]) -> None:
        """Record what `neighbor` just advertised holding for `origin`."""
        self._origin(origin).advertised[neighbor] = (hi, tuple(extra))

    def observe_held(self, origin: str, hi: int, extra: Iterable[int]) -> None:
        """Sync this node's own holdings for `origin` from store.OriginProgress.

        Call this after any successful store write (own or received) so
        `missing` never includes a seq already on disk, and any
        in-flight fetch for a seq that just landed -- even if it arrived
        from a different neighbor than the one it was requested from --
        is retired.
        """
        state = self._origin(origin)
        state.held_contiguous = hi
        state.held_extra = frozenset(extra)
        for seq in list(state.fetches):
            if seq <= state.held_contiguous or seq in state.held_extra:
                self._release(state, seq)

    # -- derived quantities -----------------------------------------------------

    def notified_max(self, origin: str) -> int:
        """The highest seq this node has been told about by any neighbor, for `origin`."""
        state = self._origins.get(origin)
        if state is None or not state.advertised:
            return 0
        return max(max(hi, max(extra, default=0)) for hi, extra in state.advertised.values())

    def contiguous(self, origin: str) -> int:
        """The highest seq this node holds contiguously from 1, for `origin`."""
        state = self._origins.get(origin)
        return state.held_contiguous if state is not None else 0

    def missing(self, origin: str) -> list[int]:
        """Seqs this node knows exist (<= notified_max) but does not hold, lowest first.

        Capped at max_missing_per_origin: the lowest seqs are kept
        because fetching them first is what grows `contiguous`, the
        quantity that matters for catching this origin up.
        """
        state = self._origins.get(origin)
        if state is None:
            return []
        upper = self.notified_max(origin)
        result: list[int] = []
        for seq in range(state.held_contiguous + 1, upper + 1):
            if seq not in state.held_extra:
                result.append(seq)
                if len(result) >= self._max_missing_per_origin:
                    break
        return result

    def origins(self) -> list[str]:
        """Every origin this tracker has any advertised or held state for."""
        return sorted(self._origins)

    def in_flight_count(self, neighbor: str) -> int:
        """How many fetches are currently outstanding on `neighbor`, across every origin.

        2026-10-23 (map.md "判断の記録"): read-only accessor onto the same counter
        plan_fetches/_fail already maintain, for SyncEngine to log in a `decision`
        event; it changes no scheduling behavior.
        """
        return self._in_flight_count.get(neighbor, 0)

    def last_fail_at(self, neighbor: str) -> float | None:
        """The `now` of the most recent fetch failure _fail actually resolved on
        `neighbor`, or None if it has never failed one (map.md "判断の記録"'s "直前
        のタイムアウトの時刻"). A stale timeout/rejection -- one _fail treats as
        nothing to resolve -- never sets this, since nothing actually failed.
        """
        return self._last_fail_at.get(neighbor)

    def decision_state(self, neighbor: str) -> dict:
        """A compact snapshot of what is known about `neighbor`, for one `decision`
        event (map.md "判断の記録"): this node's own per-origin progress, what
        `neighbor` itself has advertised, how many fetches are outstanding on it, and
        when it last failed one.

        Capped at DECISION_STATE_MAX_ORIGINS origins (lowest name first) so a
        decision line's size cannot grow with how many origins a run tracks;
        `truncated` says whether any were left out. Per origin: `hi` is this node's
        own contiguous hold (Store.OriginProgress.hi's own meaning), `adv` is the
        highest seq `neighbor` has itself advertised holding for that origin (None if
        it never has), and `missing` is Tracker.missing's own count for that origin
        (not neighbor-specific: it is what this node still lacks, regardless of who
        can supply it).
        """
        origins = self.origins()
        shown = origins[:DECISION_STATE_MAX_ORIGINS]
        origin_rows: dict[str, dict] = {}
        for origin in shown:
            state = self._origins[origin]
            advertised = state.advertised.get(neighbor)
            adv = max(advertised[0], max(advertised[1], default=0)) if advertised is not None else None
            origin_rows[origin] = {
                "hi": state.held_contiguous,
                "adv": adv,
                "missing": len(self.missing(origin)),
            }
        return {
            "in_flight": self.in_flight_count(neighbor),
            "last_timeout_s": self.last_fail_at(neighbor),
            "truncated": len(origins) > DECISION_STATE_MAX_ORIGINS,
            "origins": origin_rows,
        }

    # -- fetch scheduling -------------------------------------------------------

    def _has_advertiser_outside(self, state: _OriginState, seq: int, excluded: frozenset[str]) -> bool:
        """Whether some neighbor not in `excluded` has advertised holding `seq`."""
        for other, (hi, extra) in state.advertised.items():
            if other in excluded:
                continue
            if seq <= hi or seq in extra:
                return True
        return False

    def plan_fetches(self, neighbor: str, now: float) -> list[tuple[str, int]]:
        """Due, not-in-flight (origin, seq) fetches to send to `neighbor` now.

        Honors max_in_flight_per_neighbor across all origins fetched
        from that neighbor, and only ever picks a seq that `neighbor`
        itself advertised holding. Marks each returned (origin, seq) in
        flight on `neighbor`, so calling this again for the same `now`
        does not return it a second time.
        """
        headroom = self._max_in_flight_per_neighbor - self._in_flight_count.get(neighbor, 0)
        if headroom <= 0:
            return []

        due: list[tuple[str, int]] = []
        for origin in self.origins():
            if len(due) >= headroom:
                break
            state = self._origins[origin]
            advertisement = state.advertised.get(neighbor)
            if advertisement is None:
                continue
            hi, extra = advertisement
            extra_set = set(extra)
            for seq in self.missing(origin):
                if len(due) >= headroom:
                    break
                if seq > hi and seq not in extra_set:
                    continue  # this neighbor never claimed to hold it
                fetch = state.fetches.get(seq)
                if fetch is not None:
                    if fetch.in_flight_neighbor is not None or fetch.next_attempt_at > now:
                        continue
                    # 2026-10-22 fix: a seq whose in-flight neighbor keeps
                    # failing used to be reassigned straight back to that
                    # same neighbor every retry, because on_tick visits
                    # neighbors in a fixed order -- starving any other
                    # neighbor that also advertised it. Give an advertiser
                    # outside the failed set first claim on this retry.
                    if neighbor in fetch.failed_neighbors and self._has_advertiser_outside(
                        state, seq, fetch.failed_neighbors
                    ):
                        continue
                due.append((origin, seq))

        for origin, seq in due:
            state = self._origins[origin]
            existing = state.fetches.get(seq)
            backoff = existing.backoff_s if existing is not None else self._backoff_initial_s
            failed = existing.failed_neighbors if existing is not None else frozenset()
            state.fetches[seq] = _FetchState(
                next_attempt_at=now, backoff_s=backoff, in_flight_neighbor=neighbor, failed_neighbors=failed
            )
            self._in_flight_count[neighbor] = self._in_flight_count.get(neighbor, 0) + 1
        return due

    def on_fetch_timeout(self, origin: str, seq: int, neighbor: str, now: float) -> float | None:
        """A fetch sent to `neighbor` for (origin, seq) got no reply in time.

        Releases the in-flight slot and waits the *current* backoff
        before the next attempt (map.md "再送間隔 ... 倍々、16 秒で頭打
        ち": the wait is the backoff already in effect for this attempt,
        not the doubled value computed for the attempt after it), doubling
        the backoff for the attempt after that, capped. A timeout for a
        fetch already resolved or reassigned elsewhere is stale and
        ignored.

        Returns the chosen wait (seconds), or None if this timeout was
        stale (nothing was actually resolved) -- see _fail's own return.
        """
        return self._fail(origin, seq, neighbor, now)

    def on_fetch_rejected(self, origin: str, seq: int, neighbor: str, now: float) -> float | None:
        """A Data reply for (origin, seq) arrived from `neighbor` but failed
        validation (malformed payload, or name/payload mismatch).

        2026-10-22 fix: on_record used to just drop a bad reply and
        return, leaving the fetch marked in_flight forever -- Data had
        already arrived, so no on_fetch_timeout would ever follow to
        release it. This applies the exact same backoff-and-release
        treatment as a timeout, through the same failed_neighbors
        bookkeeping, so a neighbor that keeps answering with garbage is
        rotated away from exactly like one that stops answering at all.

        Returns the chosen wait (seconds), or None if this rejection was
        stale -- see _fail's own return.
        """
        return self._fail(origin, seq, neighbor, now)

    def _fail(self, origin: str, seq: int, neighbor: str, now: float) -> float | None:
        """Shared outcome of a timeout or a rejected reply: release the
        in-flight slot, back off, and record `neighbor` as having failed
        this seq.

        If every neighbor currently advertising this seq has now failed
        it (the failed set would cover them all), the set is reset to
        just `{neighbor}` instead of kept full: a full set would make
        plan_fetches's exclusion check vacuous (no neighbor left
        "outside" it), which would silently hand every subsequent retry
        back to whichever neighbor on_tick happens to visit first --
        recreating the starvation this bookkeeping exists to prevent.
        Resetting instead restarts the round-robin over the same
        advertisers for as long as they all keep failing.

        2026-10-23 (map.md "判断の記録"): returns the chosen wait
        (seconds), or None on the stale early-return below, so
        SyncEngine can tell a real decision from a no-op and log a
        `fetch_backoff` decision event only for the former. Also
        records `now` in _last_fail_at[neighbor] -- but only past that
        same early-return, so a stale call never fakes a failure that
        did not happen.
        """
        state = self._origins.get(origin)
        if state is None:
            return None
        fetch = state.fetches.get(seq)
        if fetch is None or fetch.in_flight_neighbor != neighbor:
            return None
        wait = fetch.backoff_s + self._jitter()
        next_backoff = min(fetch.backoff_s * self._backoff_multiplier, self._backoff_max_s)
        failed = fetch.failed_neighbors | {neighbor}
        if not self._has_advertiser_outside(state, seq, failed):
            failed = frozenset({neighbor})
        state.fetches[seq] = _FetchState(
            next_attempt_at=now + wait, backoff_s=next_backoff, failed_neighbors=failed
        )
        self._in_flight_count[neighbor] = max(0, self._in_flight_count.get(neighbor, 0) - 1)
        self._last_fail_at[neighbor] = now
        return wait

    def _release(self, state: _OriginState, seq: int) -> None:
        """Stop tracking `seq` as a fetch and free its in-flight slot, if any."""
        fetch = state.fetches.pop(seq, None)
        if fetch is not None and fetch.in_flight_neighbor is not None:
            neighbor = fetch.in_flight_neighbor
            self._in_flight_count[neighbor] = max(0, self._in_flight_count.get(neighbor, 0) - 1)
