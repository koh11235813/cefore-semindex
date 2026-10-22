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

DEFAULT_BACKOFF_INITIAL_S = 1.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0
DEFAULT_BACKOFF_MAX_S = 16.0
DEFAULT_MAX_MISSING_PER_ORIGIN = 1024
DEFAULT_MAX_IN_FLIGHT_PER_NEIGHBOR = 8


def no_jitter() -> float:
    """Default jitter callable: always zero, for deterministic callers and tests."""
    return 0.0


@dataclass
class _FetchState:
    """Retry bookkeeping for one (origin, seq) this node is trying to fetch.

    `in_flight_neighbor` is set while a fetch Interest is outstanding on
    that neighbor and cleared on timeout; `next_attempt_at` is only
    meaningful while it is None (not currently in flight).
    """

    next_attempt_at: float
    backoff_s: float
    in_flight_neighbor: str | None = None


@dataclass
class _OriginState:
    """Per-origin bookkeeping: what each neighbor advertised, what we hold, outstanding fetches."""

    advertised: dict[str, tuple[int, tuple[int, ...]]] = field(default_factory=dict)  # neighbor -> (hi, extra)
    held_contiguous: int = 0
    held_extra: frozenset[int] = frozenset()
    fetches: dict[int, _FetchState] = field(default_factory=dict)  # seq -> fetch state


class Tracker:
    """Tracks missing Index records per origin and schedules their retrieval."""

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
        self._backoff_initial_s = backoff_initial_s
        self._backoff_multiplier = backoff_multiplier
        self._backoff_max_s = backoff_max_s
        self._max_missing_per_origin = max_missing_per_origin
        self._max_in_flight_per_neighbor = max_in_flight_per_neighbor
        self._jitter = jitter
        self._origins: dict[str, _OriginState] = {}
        self._in_flight_count: dict[str, int] = {}  # neighbor -> count, across all origins

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

    # -- fetch scheduling -------------------------------------------------------

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
                if fetch is not None and (fetch.in_flight_neighbor is not None or fetch.next_attempt_at > now):
                    continue
                due.append((origin, seq))

        for origin, seq in due:
            state = self._origins[origin]
            existing = state.fetches.get(seq)
            backoff = existing.backoff_s if existing is not None else self._backoff_initial_s
            state.fetches[seq] = _FetchState(next_attempt_at=now, backoff_s=backoff, in_flight_neighbor=neighbor)
            self._in_flight_count[neighbor] = self._in_flight_count.get(neighbor, 0) + 1
        return due

    def on_fetch_timeout(self, origin: str, seq: int, neighbor: str, now: float) -> None:
        """A fetch sent to `neighbor` for (origin, seq) got no reply in time.

        Releases the in-flight slot and waits the *current* backoff
        before the next attempt (map.md "再送間隔 1 秒から倍々、16 秒で
        頭打ち": the first wait is 1 s, not 2 s), doubling the backoff
        for the attempt after that, capped. A timeout for a fetch
        already resolved or reassigned elsewhere is stale and ignored.
        """
        state = self._origins.get(origin)
        if state is None:
            return
        fetch = state.fetches.get(seq)
        if fetch is None or fetch.in_flight_neighbor != neighbor:
            return
        wait = fetch.backoff_s + self._jitter()
        next_backoff = min(fetch.backoff_s * self._backoff_multiplier, self._backoff_max_s)
        state.fetches[seq] = _FetchState(next_attempt_at=now + wait, backoff_s=next_backoff)
        self._in_flight_count[neighbor] = max(0, self._in_flight_count.get(neighbor, 0) - 1)

    def _release(self, state: _OriginState, seq: int) -> None:
        """Stop tracking `seq` as a fetch and free its in-flight slot, if any."""
        fetch = state.fetches.pop(seq, None)
        if fetch is not None and fetch.in_flight_neighbor is not None:
            neighbor = fetch.in_flight_neighbor
            self._in_flight_count[neighbor] = max(0, self._in_flight_count.get(neighbor, 0) - 1)
