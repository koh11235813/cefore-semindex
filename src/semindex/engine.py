"""SyncEngine: the transport-independent core of the Sync agent.

Owns record creation, neighbor polling, fetch planning and serving as a
holder, but performs no I/O itself beyond the two Transport calls
(register, send_interest -- both fire-and-forget). Everything else --
replies, incoming Interests, timeouts -- is delivered by calling the
on_* / handle_interest methods below, so the engine is driven identically
by any network, in-memory or Cefore-backed, that resolves Interests to
Data or timeouts through this same contract.

Records and holder state accepted here are schema-validated only (see
schema.py, holder_state.py): this baseline has no signatures, so
"validated" never means "authenticated".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from semindex import holder_state, names, schema
from semindex.store import Store
from semindex.tracker import Tracker
from semindex.transport import Response, Transport


@dataclass(frozen=True)
class StatePollPolicy:
    """A deterministic state-poll interval table (ADR 0004): row k is the interval, and
    its jitter fraction, for a poll whose neighbor has k consecutive timed-out polls
    behind it, the last row covering every larger count (the 0/1/2/3+ buckets).

    Trusted as given: agent.py's config validation enforces equal lengths of 1 to 4,
    positive finite intervals and finite jitter fractions in [0, 1).
    """

    intervals_s: tuple[float, ...]
    jitter_frac: tuple[float, ...]

    def row(self, consecutive_poll_timeouts: int) -> tuple[float, float]:
        """(interval, jitter fraction) for that many consecutive timeouts."""
        index = min(consecutive_poll_timeouts, len(self.intervals_s) - 1)
        return self.intervals_s[index], self.jitter_frac[index]


@dataclass(frozen=True)
class StatePollRule:
    """A per-neighbor state-poll interval that moves with what the polling learns:
    one interval per neighbor, starting at `i_min_s`, multiplied by
    `factor` up to `i_max_s` each time a reply says the neighbor has nothing this node
    does not already hold, and dropped back to `i_min_s` as soon as one brings
    something new. `on_timeout` is the one thing the two configured rules differ in:
    "keep" leaves the interval where it is when a poll goes unanswered, "reset" drops
    it back to `i_min_s`, so a neighbor that has just come back is polled soon rather
    than after the long interval the silence was reached with. `jitter_frac` spreads
    the applied interval the same way a StatePollPolicy row's does, so two agents'
    poll phases drift apart instead of locking together.

    A table keyed on consecutive timeouts (StatePollPolicy) cannot express this: the
    interval here depends on what the replies carried, which that table never sees.

    Trusted as given, like StatePollPolicy: agent.py's config validation enforces the
    bounds and that no state_poll_policy or interval choices are set alongside it.
    """

    i_min_s: float
    factor: float
    i_max_s: float
    jitter_frac: float
    on_timeout: str


@dataclass(frozen=True)
class SyncConfig:
    """Tunable values for one SyncEngine, defaulted from map.md "時間・サイズの初期値".

    `fault_drop_rec` is test-only fault injection (map.md "試験用の故障
    注入"): a holder consulting this engine's `handle_interest` for a
    record Interest whose (origin, seq) key has a positive count here
    withholds that many replies before answering normally. It must never
    be set in a production configuration.
    """

    group: str
    node: str
    neighbors: tuple[str, ...]
    state_poll_interval_s: float = 2.0
    state_poll_jitter_frac: float = 0.2
    # 2026-10-26 (stage-2 map "行動と確率"): when set, each state poll's next
    # interval is drawn uniformly from these discrete values instead of
    # state_poll_interval_s +/- state_poll_jitter_frac, and the decision line's
    # action_prob is 1/len. Discrete, not continuous, so the recorded
    # probability is a real probability an off-policy estimator can divide by,
    # not a density. None (the default) keeps stage 1's jittered policy, its
    # intervals and its action_prob 1.0 exactly as before.
    state_poll_interval_choices_s: tuple[float, ...] | None = None
    # 2026-10-26 (stage-2 map "選択肢の間隔のゆらぎ"): with choices, the interval
    # actually applied is the chosen value times a uniform factor in [1 - j, 1 + j].
    # A trial with one fixed 0.5 s choice showed every agent's polls in lock-step:
    # agents start within ~17 ms of each other, and an identical period with no
    # jitter never lets their phases drift apart. Ignored without choices (stage
    # 1's policy has its own state_poll_jitter_frac); 0 reproduces jitter-free
    # choices exactly.
    state_poll_choice_jitter_frac: float = 0.1
    # 2026-11-22: with choices, re-decide the pending next-poll time when the newest
    # poll is answered or passes its lifetime, instead of leaving the draw made at
    # the previous poll's send time standing. Without it the choices draw and the
    # state recorded beside it sit one poll earlier than a state_poll_rule's, so a
    # rule and a uniform draw cannot be compared as two answers to the same
    # question asked at the same moment. False (the default) runs none of that
    # code, so choices mode keeps its exact behaviour and event streams. Only
    # meaningful with state_poll_interval_choices_s, which agent.py enforces.
    state_poll_choice_redecide: bool = False
    # 2026-11-03 (ADR 0004 "Consequences", re-decision): when set, each poll's
    # interval comes from this table, and the pending next-poll time is decided
    # again when the newest poll is answered or passes its lifetime -- a policy
    # that reacts to a failed poll otherwise acts one whole interval late. None
    # (the default) runs none of that code, so stages A, B and C (choices or
    # stage 1's policy) keep their exact behaviour and event streams. Mutually
    # exclusive with the choices and stage-1 interval settings (agent.py).
    state_poll_policy: StatePollPolicy | None = None
    # 2026-11-18: when set, the next-poll interval comes from the
    # per-neighbor rule above instead of a table, re-decided at the same two points a
    # state_poll_policy is. None (the default) runs none of that code, so every stage
    # before this one keeps its exact behaviour and event streams. Mutually exclusive
    # with state_poll_policy, the choices and stage-1 interval settings (agent.py).
    state_poll_rule: StatePollRule | None = None
    # 2026-11-27: when set, the next-poll interval comes from an external decider
    # object injected at construction (SyncEngine.__init__'s poll_decider) instead of
    # any of the five policies above. The engine never reads a key of this value --
    # it exists so the run's own config names which decider its engine is running --
    # and only ever asks whether it is None, which is what turns decider mode on.
    # The spec here and the object on the engine are two sides of one invariant,
    # checked at construction: one is never set without the other. Trusted as given
    # otherwise: agent.py's config validation enforces that decider mode comes with
    # state_poll_interval_choices_s and state_poll_choice_redecide = True, since a
    # decider that is never asked at a resolution could not be the thing polling.
    state_poll_decider: dict | None = None
    interest_lifetime_ms: int = 2000
    # 2026-10-23: matches tracker.DEFAULT_BACKOFF_INITIAL_S -- see that
    # constant's comment for why 1s let the first retry land inside
    # Cefore's own PIT grace window and get absorbed before reaching the
    # producer.
    fetch_backoff_initial_s: float = 2.0
    fetch_backoff_multiplier: float = 2.0
    fetch_backoff_max_s: float = 16.0
    max_in_flight_per_neighbor: int = 8
    max_missing_per_origin: int = 1024
    state_data_expiry_s: float = 1.0
    state_data_cache_time_s: float = 0.0
    record_data_expiry_s: float = 3600.0
    record_data_cache_time_s: float = 300.0
    fault_drop_rec: dict[tuple[str, int], int] = field(default_factory=dict)
    # 2026-10-23 (map.md "判断の記録"): emit a `decision` event at each of the three
    # points SyncEngine actually chooses something (state-poll target/time, fetch
    # backoff wait, fetch-plan target/count) -- four with a state_poll_policy
    # (2026-11-03) or a state_poll_rule (2026-11-18), which also logs each re-decided next-poll time
    # (`state_poll_reschedule`) -- and four in choices mode too once
    # state_poll_choice_redecide is on (2026-11-22). True in every production config; a test
    # sets it False on an otherwise-identical engine to prove logging adds no
    # behavior of its own.
    log_decisions: bool = True


# 2026-10-26 fix: the poll-reply history grew by one entry per poll for a silent
# neighbor. How long past its Interest lifetime a state poll can still be
# answered: Cefore v0.12 keeps a PIT entry for interest_lifetime +
# CefC_Pit_CleaningTime (1 s; see tracker.DEFAULT_BACKOFF_INITIAL_S's comment),
# so Data for an older poll has nowhere to go. _PollReplies ignores replies to,
# and drops, unanswered polls past this unless they are still the newest poll.
POLL_REPLY_GRACE_S = 1.0


@dataclass
class _Poll:
    """One state poll sent to a neighbor: its name's epoch_ms, when it was sent,
    whether its own reply has arrived and, except under stage 1's jittered interval,
    its send-time jitter draw and how it was resolved."""

    epoch_ms: int
    sent_at: float
    replied: bool = False
    # 2026-11-03: state_poll_policy, since 2026-11-18 state_poll_rule and since
    # 2026-11-22 choices mode (None only under stage 1's interval, whose draw is a
    # +/- offset on one fixed value and is never reused). The jitter
    # draw made when this poll was sent, reused unchanged when its next-poll
    # time is re-decided -- redrawing would make a one-row table diverge from
    # the single-choice policy it equals, and shift every later draw -- and how
    # this poll was resolved ("answered"/"timed_out"), so it is resolved once.
    jitter_u: float | None = None
    resolved: str | None = None


@dataclass
class _PollReplies:
    """Whether one neighbor has been answering this node's state polls (stage-2 map
    "判断の状態に足すもの") -- bookkeeping for the `state_poll` decision state. Without a
    state_poll_policy it is never consulted by any scheduling decision; with one
    (2026-11-03), its snapshot also feeds the policy table, at send time and when
    the newest poll is resolved (see SyncEngine._redecide_next_poll). A
    state_poll_rule (2026-11-18) is re-decided at the same two points from these
    polls' resolution, but reads no part of the snapshot; choices mode with
    state_poll_choice_redecide (2026-11-22) re-decides there too, drawing uniformly,
    and records the snapshot without reading it.

    2026-10-26 fix: replies used to be attributed to the latest poll. Each reply
    is now matched to its own poll by the epoch_ms in the reply's name: with
    an interval shorter than the Interest lifetime several polls are outstanding at
    once, and a late reply to poll N must not mark poll N+1 answered.

    A reply to a poll is accepted iff, at reply time, that poll is still the
    newest poll sent or the reply time is <= its send time + lifetime +
    POLL_REPLY_GRACE_S; otherwise it is ignored (the poll stays timed out). A
    poll's status is judged at decision time: "answered" once an accepted reply
    has arrived (even after its lifetime), else "timed_out" once send time +
    interest_lifetime_ms <= now -- CefpycoTransport's own deadline rule -- else
    "pending". It is judged here from the clock rather than from the transport's timeout callback because
    Agent.step() runs that callback in poll() against a `now` read *before*
    on_tick's: a deadline falling between the two reads would still look pending
    to a callback-driven status although its lifetime had passed when the decision
    was made. This also makes FakeNetwork (which reports timeouts at pump time, not
    after the lifetime) agree with the real transport.
    """

    # Send order, holding only what the two reported values still depend on:
    # polls sent after the newest answered one, minus those folded into
    # `expired_timeouts` below. An answered poll survives only while it is the
    # newest poll (its status is then last_poll_status).
    polls: list[_Poll] = field(default_factory=list)
    # Polls sent after the newest answered one that went past lifetime +
    # POLL_REPLY_GRACE_S unanswered while not the newest, and were dropped from `polls`. They count as
    # timed out for good, which keeps `polls` bounded by how many polls fit in
    # lifetime + grace, however long a neighbor stays silent.
    expired_timeouts: int = 0

    def sent(self, epoch_ms: int, now: float, *, jitter_u: float | None = None) -> None:
        """Record a new poll sent at `now`; it is pending until answered or expired."""
        self.polls.append(_Poll(epoch_ms, now, jitter_u=jitter_u))

    def replied(self, epoch_ms: int, now: float, lifetime_s: float) -> tuple[_Poll, bool] | None:
        """Apply a reply that arrived at `now` to the poll whose name carried `epoch_ms`.

        2026-10-26 fix: acceptance used to depend on whether `_expire` had
        happened to run since the poll went stale; it is now decided from the
        time alone, by the rule in the class docstring. An accepted reply marks
        the poll answered and drops everything older. Every expired poll is
        older than any poll still held, so it also clears `expired_timeouts`. A
        reply to a poll no longer held (older than an answered one, or dropped
        as expired -- which the time rule would reject anyway) changes nothing.

        2026-11-03: returns (the accepted poll, whether it is the newest poll) so
        a state_poll_policy can re-decide on it, or None when the reply was not
        accepted. Read by every mode that re-decides on resolution (see
        SyncEngine._redecides_on_resolution; since 2026-11-22 choices mode with
        state_poll_choice_redecide reads it too); a caller in any other mode ignores
        it, as it did the old None.
        """
        for index, poll in enumerate(self.polls):
            if poll.epoch_ms != epoch_ms:
                continue
            newest = index == len(self.polls) - 1
            if newest or now <= poll.sent_at + lifetime_s + POLL_REPLY_GRACE_S:
                poll.replied = True
                del self.polls[:index]
                self.expired_timeouts = 0
                return poll, newest
            return None
        return None

    def _expire(self, now: float, lifetime_s: float) -> None:
        """Drop the oldest polls the reported values no longer need, in one slice
        deletion; the newest poll is always kept, since last_poll_status is its status.

        An answered poll at the front is no longer needed once a newer poll exists.
        An unanswered one whose reply would now be rejected (now > send time +
        lifetime + POLL_REPLY_GRACE_S) is folded into `expired_timeouts`. Strictly
        greater, matching `replied`: a reply exactly at that boundary is accepted.

        2026-10-26 fix: dropping with pop(0) per poll moved every remaining entry
        each time, quadratic when many polls expire at once (a long tick delay, or
        a long interval chosen after short ones).
        """
        cut = 0
        while cut < len(self.polls) - 1:
            poll = self.polls[cut]
            if poll.replied:
                cut += 1
            elif poll.sent_at + lifetime_s + POLL_REPLY_GRACE_S < now:
                cut += 1
                self.expired_timeouts += 1
            else:
                break
        del self.polls[:cut]

    def snapshot(self, now: float, lifetime_s: float) -> tuple[str | None, int]:
        """(status of the most recent poll, consecutive timed-out polls) as of `now`.

        The count is the number of timed-out polls since the newest answered one,
        not counting still-pending polls: those can only be the newest ones (every
        poll shares one lifetime), so they are skipped rather than ending the run.
        A reply to an older poll therefore does not reset the count when newer
        polls have already timed out. None before the first poll.
        """
        self._expire(now, lifetime_s)
        if not self.polls:
            return None, 0
        statuses = [self._status(poll, now, lifetime_s) for poll in self.polls]
        count = self.expired_timeouts + sum(1 for status in statuses if status == "timed_out")
        return statuses[-1], count

    @staticmethod
    def _status(poll: _Poll, now: float, lifetime_s: float) -> str:
        """One poll's status at `now` (see the class docstring)."""
        if poll.replied:
            return "answered"
        if poll.sent_at + lifetime_s <= now:
            return "timed_out"
        return "pending"


class SyncEngine:
    """The per-node Sync agent core: requester toward its neighbors, holder for itself."""

    def __init__(
        self,
        config: SyncConfig,
        store: Store,
        transport: Transport,
        *,
        poll_jitter_rng: Callable[[], float] = lambda: 0.5,
        poll_choice_rng: Callable[[], float] = lambda: 0.0,
        poll_choice_jitter_rng: Callable[[], float] = lambda: 0.5,
        backoff_jitter: Callable[[], float] = lambda: 0.0,
        poll_decider: object | None = None,
    ) -> None:
        """`poll_jitter_rng` must return a float in [0, 1); it is mapped to
        +/- state_poll_jitter_frac around state_poll_interval_s (map.md
        "状態確認の周期 2 秒 ±20%"). The default is a constant 0.5, which
        maps to zero offset, so engines are deterministic unless a real
        source of randomness is injected. `backoff_jitter` is passed
        straight through to the Tracker.

        `poll_choice_rng` must also return a float in [0, 1); it picks the
        index into config.state_poll_interval_choices_s and is drawn only
        when that is set -- poll_jitter_rng is then never drawn, so the two
        policies never share or perturb one stream. Once per poll, and
        (2026-11-22) once more per resolved poll with
        config.state_poll_choice_redecide, from this same stream: a second
        stream would be one more thing a replay has to reproduce in order.
        Its constant default
        0.0 always picks the first choice, deterministic like the jitter
        default. `poll_choice_jitter_rng` (also [0, 1)) is the separate stream
        for config.state_poll_choice_jitter_frac, drawn once per poll in
        choices mode; its default 0.5 maps to a factor of exactly 1. With
        config.state_poll_policy (2026-11-03), and with config.state_poll_rule
        (2026-11-18), it is the only poll rng drawn, once per poll at send time,
        for that table row's or that neighbor's interval jitter; neither
        poll_choice_rng nor poll_jitter_rng is drawn.

        2026-11-27: `poll_decider`, when given, replaces the choice draw itself (not
        the jitter draw) with a call to its own `decide(obs)` -- config.state_poll_decider
        must then be set too, since the config is what a run's own record says it asked
        for and the object is what actually ran; one set without the other is refused
        below rather than silently running the wrong mode or silently not running the
        one that was asked for.

        2026-10-22 fix: `config.group` and `config.node` already failed
        here with an exact diagnostic -- the registration_prefix call
        below validates both -- but a malformed neighbor went
        unchecked until whichever later on_tick first tried to build a
        name with it, raising "holder must be..." from inside
        names.build_state_name: a diagnostic that names the wrong
        thing from this config's point of view. Validating every
        neighbor up front, before that first on_tick, surfaces the
        same error at construction instead.
        """
        for neighbor in config.neighbors:
            names.validate_component(neighbor, "neighbor")
        if (config.state_poll_decider is None) != (poll_decider is None):
            raise ValueError("state_poll_decider and poll_decider must be both set or both unset")
        self._config = config
        self._store = store
        self._transport = transport
        self._poll_jitter_rng = poll_jitter_rng
        self._poll_choice_rng = poll_choice_rng
        self._poll_choice_jitter_rng = poll_choice_jitter_rng
        self._poll_decider = poll_decider
        self._tracker = Tracker(
            backoff_initial_s=config.fetch_backoff_initial_s,
            backoff_multiplier=config.fetch_backoff_multiplier,
            backoff_max_s=config.fetch_backoff_max_s,
            max_missing_per_origin=config.max_missing_per_origin,
            max_in_flight_per_neighbor=config.max_in_flight_per_neighbor,
            jitter=backoff_jitter,
        )
        self._next_state_poll: dict[str, float] = {}
        # 2026-11-18: state_poll_rule only -- the one interval the rule
        # keeps per neighbor, seeded with i_min_s at that neighbor's first poll.
        self._rule_interval_s: dict[str, float] = {}
        self._poll_replies: dict[str, _PollReplies] = {}
        # 2026-11-27: poll_decider only -- the per-neighbor history a decider is
        # handed beside its state: its pair's last 8 resolved polls, oldest first.
        # Appended only from a state_poll_reschedule row
        # (_redecide_next_poll), after that row's own decision has been asked, so a
        # decision is never asked about itself.
        self._poll_hist: dict[str, deque] = {}
        self._events: list[dict] = []
        self._transport.register(names.registration_prefix(config.group, config.node))
        # Seed the tracker with whatever this node already holds (e.g.
        # after reopening its store on restart) before the first tick.
        for origin, progress in store.all_progress().items():
            self._tracker.observe_held(origin, progress.hi, progress.extra)

    # -- events ------------------------------------------------------------------

    def _emit(self, event: dict) -> None:
        """Queue one event dict for the next drain_events call."""
        self._events.append(event)

    def drain_events(self) -> list[dict]:
        """Return and clear every event emitted since the last drain."""
        events, self._events = self._events, []
        return events

    def _emit_decision(self, kind: str, state: dict, action: dict, now: float, action_prob: float = 1.0) -> None:
        """Queue one `decision` event (map.md "判断の記録"): the state visible at the
        moment this engine actually chose `action`, plus action_prob -- the
        probability the policy gave `action`. 1.0 for every stage-1 policy (each
        picks its one deterministic action); only a draw from
        state_poll_interval_choices_s passes anything else -- at send time, and
        (2026-11-22, state_poll_choice_redecide) on the state_poll_reschedule row of a
        re-decided poll, which carries the same 1/len.

        A no-op when config.log_decisions is False (test/analysis-only; every
        production config leaves it at its default True). Assumes a real decision
        has already been made -- it never re-checks that for itself, so it must only
        be called once something was actually chosen, not on a no-op.

        2026-10-23 fix: `ts_ms` is stamped here from `now` -- the processing
        time this decision was actually made at, i.e. whatever clock reading
        the caller (on_tick/on_state/_fail_fetch, all driven by the
        transport's own clock) was already holding -- rather than left for
        EventWriter.write to stamp at flush time. A decision made while this
        node's Agent was blocked inside a probe's synchronous content fetch
        (agent.py's _run_probe, until 2026-11-02 when that fetch stopped
        blocking the loop) stayed queued, not flushed, until that probe
        returned, so the write-time stamp used to run however many seconds
        behind the decision itself the probe wait happened to take. Events
        are still flushed later than they are made (once per step), so the
        stamp stays here. The same
        epoch-ms formula EventWriter uses (round(clock_reading * 1000)) is
        reused here so a pre-set ts_ms and a stamp-at-write-time one are
        never computed two different ways.
        """
        if not self._config.log_decisions:
            return
        self._emit(
            {
                "event": "decision",
                "ts_ms": round(now * 1000),
                "kind": kind,
                "state": state,
                "action": action,
                "action_prob": action_prob,
            }
        )

    # -- own-record creation --------------------------------------------------------

    def create_record(
        self,
        *,
        content_name: str,
        zone: str,
        disaster: str,
        media: str,
        observed_at: int,
        model_id: str = schema.DEFAULT_MODEL_ID,
        supersedes: schema.RecordKey | None = None,
    ) -> schema.IndexRecord:
        """Allocate the next seq for this node's own origin and store the record.

        This is how the agent's own content enters the Semantic Index.
        """
        record = self._store.append_own(
            origin=self._config.node,
            content_name=content_name,
            zone=zone,
            disaster=disaster,
            media=media,
            observed_at=observed_at,
            model_id=model_id,
            supersedes=supersedes,
        )
        progress = self._store.get_progress(record.origin)
        self._tracker.observe_held(record.origin, progress.hi, progress.extra)
        # 2026-10-23 fix: ts_ms is `observed_at` itself, not left for
        # EventWriter.write to stamp at flush time -- the caller
        # (agent.py's _drain_schedule) already computed observed_at as
        # round(now * 1000) from the same processing `now` this record was
        # actually created at, so reusing it here needs no separate clock
        # reading and guarantees the two can never read differently for the
        # same record. Without this, a record created in a step() that then
        # blocked inside a probe's synchronous fetch (drained only after that
        # probe returned) logged a ts_ms seconds later than it was actually
        # created. 2026-11-02: the probe fetch no longer blocks the step, but
        # events are still drained after the step's other work, so the stamp
        # stays here rather than at write time.
        self._emit({"event": "record_created", "ts_ms": observed_at, "origin": record.origin, "seq": record.seq})
        return record

    # -- periodic housekeeping -----------------------------------------------------

    def on_tick(self, now: float) -> None:
        """Send due state requests and due record fetches to every neighbor.

        In a mode that re-decides on resolution, a newest poll whose lifetime
        has passed is resolved first, so a re-decided next time already in the past
        (clamped to `now`) is polled in this same tick rather than one tick late.
        """
        for neighbor in self._config.neighbors:
            if self._redecides_on_resolution():
                self._resolve_timed_out_poll(neighbor, now)
            due_at = self._next_state_poll.get(neighbor, now)
            if now >= due_at:
                self._send_state_request(neighbor, now)
            self._send_fetch_plan(neighbor, now)

    def _send_state_request(self, neighbor: str, now: float) -> None:
        """Send one state-request Interest to `neighbor` and schedule its next poll."""
        epoch_ms = round(now * 1000)
        name = names.build_state_name(self._config.group, neighbor, self._config.node, epoch_ms)
        self._transport.send_interest(name, self._config.interest_lifetime_ms)
        replies = self._poll_replies.setdefault(neighbor, _PollReplies())
        lifetime_s = self._config.interest_lifetime_ms / 1000.0
        # The reply status is snapshotted before this poll itself is recorded as
        # sent, so the decision shows how the *previous* poll to this neighbor
        # fared, not the trivially-pending poll being sent right now.
        if self._poll_decider is not None:
            # 2026-11-27: the state the decider is asked with has to be the very
            # dict this row ends up logging (see _decide's own mutation contract),
            # so it is built here, before the call, rather than left to the shared
            # line below -- which would rebuild it from a decision_state call that
            # allocates a fresh dict every time, handing the decider one object and
            # the row another.
            last_poll_status, consecutive_poll_timeouts = replies.snapshot(now, lifetime_s)
            state = {
                **self._tracker.decision_state(neighbor),
                "last_poll_status": last_poll_status,
                "consecutive_poll_timeouts": consecutive_poll_timeouts,
            }
            interval = self._decide(neighbor, "state_poll", now, state, None)
            jitter_u = self._poll_choice_jitter_rng()
            applied_interval = interval * (1 + self._config.state_poll_choice_jitter_frac * (2 * jitter_u - 1))
            action = {"neighbor": neighbor, "interval_s": interval, "applied_interval_s": applied_interval}
            action_prob = 1.0
        elif self._config.state_poll_policy is None and self._config.state_poll_rule is None:
            action, applied_interval, action_prob, jitter_u = self._choose_poll_interval(neighbor)
            last_poll_status, consecutive_poll_timeouts = replies.snapshot(now, lifetime_s)
        else:
            # 2026-11-03: the table is looked up with that previous poll's state,
            # so here the snapshot has to come before the choice. Kept as its own
            # branch so the other two policies' order stays as it was.
            last_poll_status, consecutive_poll_timeouts = replies.snapshot(now, lifetime_s)
            jitter_u = self._poll_choice_jitter_rng()
            if self._config.state_poll_rule is None:
                interval, applied_interval = self._policy_interval(consecutive_poll_timeouts, jitter_u)
            else:
                # 2026-11-18: the rule reads no part of that snapshot, but
                # draws its jitter at the same point and logs the same action keys, so
                # the two mechanisms stay comparable line for line.
                interval, applied_interval = self._rule_interval(neighbor, jitter_u)
            action = {"neighbor": neighbor, "interval_s": interval, "applied_interval_s": applied_interval}
            action_prob = 1.0
        if self._poll_decider is None:
            # The one line decider mode guards rather than skips: with a decider, `state`
            # above is already the dict the decider was handed, and rebuilding it here
            # from a fresh decision_state call would log a different object from the
            # one the decision was actually made from.
            state = {
                **self._tracker.decision_state(neighbor),
                "last_poll_status": last_poll_status,
                "consecutive_poll_timeouts": consecutive_poll_timeouts,
            }
        # map.md "判断の記録", kind 1: this *is* the decision -- which
        # neighbor, and the interval chosen for its next poll. Always made
        # (a neighbor due for a poll is always polled), so always logged
        # when log_decisions is on.
        self._emit_decision("state_poll", state, action, now, action_prob)
        replies.sent(epoch_ms, now, jitter_u=jitter_u)
        # The jittered interval runs from this poll's own send time, so jitter
        # accumulates poll after poll instead of being pulled back to a grid --
        # the point is that two agents' phases drift apart. Final in stage 1, and in
        # choices mode with state_poll_choice_redecide off; with a state_poll_policy, a
        # state_poll_rule or (2026-11-22) that knob on, only provisional, until this
        # poll is resolved (_redecide_next_poll), which also measures from here.
        self._next_state_poll[neighbor] = now + applied_interval

    def _decide(self, neighbor: str, kind: str, now: float, state: dict, resolved: dict | None) -> float:
        """Ask `self._poll_decider` for `neighbor`'s next interval and return it, or
        raise if the answer cannot be used (poll_decider only).

        `state` is handed over by reference, not copied: it is the same dict this
        decision's own row logs, and a decider that mutates it corrupts exactly that
        one row and nothing later (decision_state allocates a fresh dict on every
        call, so the next call's `state` is never this one). `history` is the
        opposite -- the engine keeps that list for the rest of the run, so each row
        handed to the decider is its own copy, made here, and `obs["where"]` carries
        no ordinal: nothing a node could run reads one call's position in a stream of
        decisions across every other node, and a decider that depended on it could
        not run on a real node at all.

        `None` is the shape of an answer that would come later, which is not
        supported: at a resolution it is refused by name below. At a
        send a deferred answer means nothing -- the row cannot report what it does not
        yet have -- so it is left to fall through to the one check every other invalid
        answer hits, `interval not in choices`, which None always fails.
        """
        obs = {
            "state": state,
            "history": [dict(row) for row in self._poll_hist.get(neighbor, ())],
            "resolved": resolved,
            "where": {"node": self._config.node, "neighbor": neighbor, "kind": kind, "ts_ms": round(now * 1000)},
        }
        interval = self._poll_decider.decide(obs)
        if interval is None and kind != "state_poll":
            # 2026-11-27: refused by name here, before the generic choice-set check
            # below could fail on it for the wrong reason -- the engine keeps no
            # pending entry for a resolution and arms nothing from a None answer, so
            # deferring here would leave the pair's next poll unarmed with no row
            # logged to say why.
            raise ValueError(f"{self._config.node}->{neighbor} {kind}: deferred answers are not supported yet")
        choices = self._config.state_poll_interval_choices_s
        if interval not in choices:
            raise ValueError(
                f"{self._config.node}->{neighbor} {kind}: the decider returned {interval!r}, "
                f"which is not one of {choices!r}"
            )
        return interval

    def _policy_interval(self, consecutive_poll_timeouts: int, jitter_u: float) -> tuple[float, float]:
        """(table interval, applied interval) for that many consecutive timeouts and
        the poll's jitter draw `jitter_u`.

        The same factor arithmetic as choices mode (value * (1 + j * (2u - 1))), so
        a one-row table and the single choice it equals give bit-identical times.
        """
        interval, jitter_frac = self._config.state_poll_policy.row(consecutive_poll_timeouts)
        factor = 1 + jitter_frac * (2 * jitter_u - 1)
        return interval, interval * factor

    def _rule_interval(self, neighbor: str, jitter_u: float) -> tuple[float, float]:
        """(`neighbor`'s current rule interval, that interval applied with `jitter_u`).

        The interval starts at i_min_s the first time this neighbor is polled and from
        then on sits wherever _advance_rule_interval last left it. The same factor
        arithmetic as _policy_interval, so a rule whose interval can never move polls
        at exactly the times the one-row table with that value does.
        """
        rule = self._config.state_poll_rule
        interval = self._rule_interval_s.setdefault(neighbor, rule.i_min_s)
        return interval, interval * (1 + rule.jitter_frac * (2 * jitter_u - 1))

    def _advance_rule_interval(self, neighbor: str, trigger: str, empty: bool | None) -> float:
        """Move `neighbor`'s rule interval for a poll that resolved as `trigger`, and
        return where it now stands (state_poll_rule only).

        `empty` is whether the reply held nothing this node did not already have, or
        None when there was no reply to read: a timeout, or a reply that failed to
        decode. Nothing read means nothing learned about whether the neighbor has news,
        so the interval stays where it is rather than being guessed in either
        direction. An answer that brought nothing new backs the polling off by `factor`
        up to the ceiling; one that brought something new drops it to the floor, since
        a neighbor with news is likely to have more. What a timeout does is the one
        thing the two configured rules disagree about, and the whole point of the
        comparison this rule exists for: "keep" leaves a backed-off interval backed
        off, "reset" polls the returning neighbor soon.
        """
        rule = self._config.state_poll_rule
        interval = self._rule_interval_s.get(neighbor, rule.i_min_s)
        if trigger == "timed_out":
            if rule.on_timeout == "reset":
                interval = rule.i_min_s
        elif empty is True:
            interval = min(interval * rule.factor, rule.i_max_s)
        elif empty is False:
            interval = rule.i_min_s
        self._rule_interval_s[neighbor] = interval
        return interval

    def _redecides_after_decode(self) -> bool:
        """Whether a reply's own poll is resolved only after the reply has been read.

        A state_poll_rule asks what the reply held, so it has to wait for the decode.
        2026-11-22: choices mode with state_poll_choice_redecide waits for the decode
        as well, and then for one step more -- on_state resolves it only once the reply
        has been folded into the tracker, because its reschedule row reports the
        tracker's state and that state has to include what this reply said. A
        state_poll_policy resolves before the decode, as it has from the start.
        """
        return self._config.state_poll_rule is not None or self._config.state_poll_choice_redecide

    def _redecides_on_resolution(self) -> bool:
        """Whether a resolved poll re-decides its neighbor's pending next-poll time at all.

        Asked in one place so the gates that let a resolution through cannot drift
        apart from _redecide_next_poll's own branches: a mode reaching that method
        without a branch there would be scheduled from the wrong setting.
        """
        return self._config.state_poll_policy is not None or self._redecides_after_decode()

    def _resolve_timed_out_poll(self, neighbor: str, now: float) -> None:
        """Re-decide `neighbor`'s next poll if its newest poll has just passed its
        lifetime unresolved (only in a mode that re-decides at all, see
        _redecides_on_resolution).

        The same clock rule as _PollReplies._status (send time + lifetime <= now),
        not the transport's timeout callback, for the reason its docstring gives.
        Only the newest poll is resolved: an older one's outcome no longer decides
        when the next poll goes, since that poll has already been sent.
        """
        replies = self._poll_replies.get(neighbor)
        if replies is None or not replies.polls:
            return
        poll = replies.polls[-1]
        if poll.resolved is None and poll.sent_at + self._config.interest_lifetime_ms / 1000.0 <= now:
            self._redecide_next_poll(neighbor, replies, poll, "timed_out", now)

    def _redecide_next_poll(
        self,
        neighbor: str,
        replies: _PollReplies,
        poll: _Poll,
        trigger: str,
        now: float,
        *,
        empty: bool | None = None,
    ) -> None:
        """Resolve `poll` (the newest to `neighbor`) as `trigger` and decide its next
        poll's time again from the policy table, from the rule's per-neighbor
        interval, or (2026-11-22, state_poll_choice_redecide) from a fresh uniform
        draw over the interval choices, logging a `state_poll_reschedule`.

        The state is snapshotted *after* the resolution -- the opposite of send time
        -- since the outcome just learned is what the policy reacts to. The interval
        runs from the poll's own send time with its own send-time jitter draw, as
        replay.py measures it; a time already past is clamped to `now` (a table row
        shorter than the lifetime cannot be met after a timeout). Only the pending
        next-poll time moves: no Interest is sent or withdrawn here.

        `empty` is what a rule's reply held (see _advance_rule_interval) and is ignored
        under a policy, which never asks.
        """
        poll.resolved = trigger
        last_poll_status, consecutive_poll_timeouts = replies.snapshot(now, self._config.interest_lifetime_ms / 1000.0)
        rule = self._config.state_poll_rule
        action_prob = 1.0
        # The table and the rule read none of the tracker's per-origin state, and
        # carrying it would make every reschedule row as large as a state_poll row.
        state = {"last_poll_status": last_poll_status, "consecutive_poll_timeouts": consecutive_poll_timeouts}
        if self._poll_decider is not None:
            # 2026-11-27: checked before the redecide branch below, and not after it
            # -- decider mode always has state_poll_choice_redecide on (agent.py's
            # config validation), so that branch's own test would also be true here,
            # and a decider check placed after it would never run.
            state = {
                **self._tracker.decision_state(neighbor),
                "last_poll_status": last_poll_status,
                "consecutive_poll_timeouts": consecutive_poll_timeouts,
            }
            resolved = {"poll_epoch_ms": poll.epoch_ms, "trigger": trigger, "empty": empty}
            interval = self._decide(neighbor, "state_poll_reschedule", now, state, resolved)
            applied_interval = interval * (1 + self._config.state_poll_choice_jitter_frac * (2 * poll.jitter_u - 1))
        elif self._config.state_poll_choice_redecide:
            # 2026-11-22: checked first, and before the table's branch, because that
            # branch reads self._config.state_poll_policy, which choices mode leaves
            # None. Exactly one draw from the choice stream per resolution, made
            # before _emit_decision so logging never changes which value is drawn.
            choices = self._config.state_poll_interval_choices_s
            interval = choices[int(self._poll_choice_rng() * len(choices))]
            applied_interval = interval * (1 + self._config.state_poll_choice_jitter_frac * (2 * poll.jitter_u - 1))
            action_prob = 1.0 / len(choices)
            # Every field the send-time row carries, so the two rows of one poll can
            # be read as the same question asked twice. The learner this row exists
            # for needs the per-origin progress the two fields above cannot show.
            state = {
                **self._tracker.decision_state(neighbor),
                "last_poll_status": last_poll_status,
                "consecutive_poll_timeouts": consecutive_poll_timeouts,
            }
        elif rule is None:
            interval, applied_interval = self._policy_interval(consecutive_poll_timeouts, poll.jitter_u)
        else:
            interval = self._advance_rule_interval(neighbor, trigger, empty)
            applied_interval = interval * (1 + rule.jitter_frac * (2 * poll.jitter_u - 1))
        due_at = poll.sent_at + applied_interval
        self._next_state_poll[neighbor] = max(due_at, now)
        action = {
            "neighbor": neighbor,
            "poll_epoch_ms": poll.epoch_ms,
            "trigger": trigger,
            "interval_s": interval,
            "applied_interval_s": applied_interval,
            "clamped": due_at < now,
        }
        if rule is not None or self._config.state_poll_choice_redecide:
            # 2026-11-18: on an answer, what the reply held is what moves a rule's
            # interval, so a row without it cannot be read back. Not added under a
            # policy: its rows are compared whole by tests that pin the shape its
            # analysis already expects. 2026-11-22: carried in choices mode too --
            # the uniform draw does not react to it, but a learner reading these
            # rows beside a rule's needs the same column there.
            action["empty"] = empty
        if self._poll_decider is not None:
            # 2026-11-27: `decided_at_ms` is when the answer was actually applied --
            # this phase never defers, so that is always this same instant, equal to
            # `ts_ms`; `late` is always False for the same reason (both gain meaning
            # once a deferred answer can arrive after the pair's next poll). Written
            # only in decider mode: the five existing modes' rows gain nothing.
            action["decided_at_ms"] = round(now * 1000)
            action["late"] = False
        self._emit_decision("state_poll_reschedule", state, action, now, action_prob)
        if self._poll_decider is not None:
            # 2026-11-27: appended from this row's own action, after it has been
            # asked for and emitted (or would have been, with logging on) -- never
            # from inside _emit_decision, which is a no-op with log_decisions off, so
            # turning logging off must not change what the next decision is asked
            # with.
            self._poll_hist.setdefault(neighbor, deque(maxlen=8)).append(
                {
                    "ts_ms": round(now * 1000),
                    "poll_epoch_ms": action["poll_epoch_ms"],
                    "trigger": action["trigger"],
                    "empty": action["empty"],
                    "interval_s": action["interval_s"],
                    "applied_interval_s": action["applied_interval_s"],
                    "clamped": action["clamped"],
                }
            )

    def _choose_poll_interval(self, neighbor: str) -> tuple[dict, float, float, float | None]:
        """Draw the interval until `neighbor`'s next state poll: the decision's
        `action`, the interval actually applied, the probability the policy gave
        the action, and the jitter draw it was applied with.

        2026-11-22: the jitter draw is returned so the poll can keep it, which is
        what a re-decision under state_poll_choice_redecide applies -- redrawing it
        there would shift every later draw in the stream. None under stage 1's
        interval: that draw is a +/- offset on one fixed value, nothing re-decides
        it, and reusing it would mean nothing.

        In choices mode the action is the chosen value (`interval_s`, the thing
        action_prob is the probability of) and the jitter only perturbs how it is
        applied, recorded as `applied_interval_s`; the jitter is not part of the
        action, so it does not change action_prob.

        Each rng is drawn exactly once per poll, whether or not log_decisions is
        on: the `decision` event only reports the value drawn here, it never draws
        its own -- otherwise turning logging off would shift every later draw and
        change which Interests go out when.
        """
        choices = self._config.state_poll_interval_choices_s
        if choices is not None:
            # A uniform u in [0, 1) gives every index the same 1/len share,
            # which is exactly what action_prob claims.
            index = int(self._poll_choice_rng() * len(choices))
            chosen = choices[index]
            jitter_u = self._poll_choice_jitter_rng()
            factor = 1 + self._config.state_poll_choice_jitter_frac * (2 * jitter_u - 1)
            applied = chosen * factor
            return (
                {"neighbor": neighbor, "interval_s": chosen, "applied_interval_s": applied},
                applied,
                1.0 / len(choices),
                jitter_u,
            )
        # Map the injected [0, 1) rng to +/- state_poll_jitter_frac around the
        # configured interval, so "2s +/-20%" is reproducible in tests yet real
        # under a real rng. action_prob stays 1.0, as in stage 1's data (stage-1
        # map's 2026-10-26 correction: those lines are not off-policy usable).
        offset = self._config.state_poll_jitter_frac * (2 * self._poll_jitter_rng() - 1)
        interval = self._config.state_poll_interval_s * (1 + offset)
        return {"neighbor": neighbor, "interval_s": interval}, interval, 1.0, None

    def _send_fetch_plan(self, neighbor: str, now: float) -> None:
        """Send a fetch Interest for every due, not-in-flight missing record on `neighbor`.

        2026-10-23 (map.md "判断の記録", kind 3): the state snapshot is taken
        *before* plan_fetches, which mutates the Tracker's own in-flight count for
        `neighbor` -- the snapshot must show the headroom that was actually available
        when this decision was made, not what is left after it. on_tick calls this
        every receive-poll cycle (about 10/s live) for every neighbor, so logging is
        skipped entirely when nothing is actually due: an empty plan is not a
        decision, it is the overwhelming common case, and logging it would make the
        decision log's size track tick frequency rather than fetch activity.
        """
        state = self._tracker.decision_state(neighbor)
        due = self._tracker.plan_fetches(neighbor, now)
        if due:
            self._emit_decision(
                "fetch_plan", state, {"neighbor": neighbor, "fetched": [[origin, seq] for origin, seq in due]}, now
            )
        for origin, seq in due:
            name = names.build_record_name(self._config.group, neighbor, origin, seq)
            self._transport.send_interest(name, self._config.interest_lifetime_ms)

    # -- requester-side reply handling -----------------------------------------------

    def on_state(self, neighbor: str, payload: bytes, now: float, *, poll_epoch_ms: int) -> None:
        """A neighbor's reply to a state-request Interest arrived.

        `poll_epoch_ms` is the epoch_ms segment of the reply's own name, i.e. which
        of this node's polls it answers -- the transport has already parsed it.

        Decodes it, folds every origin's advertised (hi, extra) into the
        tracker, and immediately plans any newly-due fetches from that
        neighbor rather than waiting for the next tick, so a discovered
        gap starts closing as soon as it is known.

        2026-10-22 fix: the state_rx event used to carry only
        `neighbor`, with no record of what was actually advertised -- a
        restarted holder (map.md T2(c)) that advertises a higher hi than
        it has actually stored could only be caught indirectly, through a
        fetch_timeout that follows much later, never through the
        advertisement itself. `hi` (per-origin, decode failure -> `{}`)
        lets a checker compare what a holder claimed against what its own
        events.jsonl shows it had actually stored as of this same reply.
        """
        # 2026-10-23 fix: ts_ms is stamped here from `now` -- see
        # _emit_decision's own comment on why a queued event's processing
        # time, not its later flush time, is what belongs in ts_ms.
        ts_ms = round(now * 1000)
        # Any reply, even one that fails to decode, shows the neighbor is
        # answering -- what the poll-reply status is there to tell the policy.
        accepted = None
        replies = self._poll_replies.get(neighbor)
        if replies is not None:
            accepted = replies.replied(poll_epoch_ms, now, self._config.interest_lifetime_ms / 1000.0)
            if accepted is not None and self._config.state_poll_policy is not None:
                self._on_policy_reply(neighbor, replies, *accepted, now)
        try:
            decoded = holder_state.decode(payload)
        except holder_state.HolderStateError:
            self._emit({"event": "state_rx", "ts_ms": ts_ms, "neighbor": neighbor, "hi": {}})
            if replies is not None and accepted is not None and self._redecides_after_decode():
                # Nothing could be read off this reply, so the rule has no emptiness to
                # react to -- but the poll is still resolved as answered, exactly as
                # under a policy, since Data did come back from this neighbor.
                self._on_policy_reply(neighbor, replies, *accepted, now, empty=None)
            return  # malformed/oversize reply from a neighbor; drop silently
        self._emit(
            {"event": "state_rx", "ts_ms": ts_ms, "neighbor": neighbor, "hi": {o: e["hi"] for o, e in decoded.items()}}
        )
        empty: bool | None = None
        # 2026-11-22: choices mode resolves the poll one step later than a rule does,
        # past the fold below, so both gates are decided here, before anything moves.
        resolves_here = replies is not None and accepted is not None and self._redecides_after_decode()
        waits_for_fold = resolves_here and self._config.state_poll_choice_redecide
        if resolves_here:
            # Emptiness compares against what this node holds, which the fold below
            # never touches (that records what the neighbor advertised), so it is asked
            # here, where the answer is plainly about the holdings this poll was
            # answered against.
            empty = self._reply_is_empty(decoded)
            if not waits_for_fold:
                # 2026-11-18: a rule asks what the reply brought, so its
                # re-decision waits until the reply has been read -- the one thing it does
                # differently from a policy, which re-decides above, before the decode, as
                # it has from the start. It stays ahead of the fold: its reschedule row
                # carries no tracker state, so no part of what it logs could move.
                self._on_policy_reply(neighbor, replies, *accepted, now, empty=empty)
        for origin, entry in decoded.items():
            self._tracker.observe_state(origin, neighbor, entry["hi"], entry["extra"])
        if waits_for_fold:
            # 2026-11-22: choices mode re-decides only here, after the fold, because its
            # reschedule row does carry the tracker's decision_state, and a learner reads
            # that row as everything the node could observe at the instant it chose the
            # next interval. By that instant the node has read this reply, so a pre-fold
            # snapshot would hide what this very reply advertised. Still ahead of
            # _send_fetch_plan, which puts fetches in flight: the send-time state_poll
            # row is snapshotted before the fetch plan too, so `in_flight` counts the
            # same thing in both rows of one poll.
            self._on_policy_reply(neighbor, replies, *accepted, now, empty=empty)
        self._send_fetch_plan(neighbor, now)

    def _reply_is_empty(self, decoded: dict[str, dict]) -> bool:
        """Whether `decoded` advertises nothing this node does not already hold: for
        every origin in it, this node holds every seq from 1 up to that origin's hi
        (evidence 11's `rx_no_new`, the quantity the replayed comparison was built on).

        The advertised `extra` seqs are deliberately not read: they are the holder's
        sparse seqs above its own hi, and the replayed comparison this rule's arms are
        predicted from ignored them too (evidence 13 found none in 137659 replies). A
        reply carrying no origin at all is empty by the same reading -- there is
        nothing in it to be behind on.
        """
        return all(self._tracker.holds_through(origin, entry["hi"]) for origin, entry in decoded.items())

    def _on_policy_reply(
        self,
        neighbor: str,
        replies: _PollReplies,
        poll: _Poll,
        newest: bool,
        now: float,
        *,
        empty: bool | None = None,
    ) -> None:
        """Act on an accepted reply under a state_poll_policy (2026-11-03), a
        state_poll_rule (2026-11-18) or state_poll_choice_redecide (2026-11-22); the
        latter two pass what the reply held as `empty`.

        An unresolved newest poll is resolved as answered. A reply to a poll already
        resolved as timed out does not re-decide (one resolution per poll; the
        schedule it set stands) but is logged as `state_poll_late_reply`, so
        analysis can count these without inferring them -- replay.py has no such
        event. A reply to an older, unresolved poll changes nothing: the poll after
        it has already been sent. In neither of those two cases does a rule's interval
        move: an interval is only ever moved by the poll it is re-decided from.
        """
        if poll.resolved == "timed_out":
            self._emit(
                {
                    "event": "state_poll_late_reply",
                    "ts_ms": round(now * 1000),
                    "neighbor": neighbor,
                    "poll_epoch_ms": poll.epoch_ms,
                }
            )
        elif poll.resolved is None and newest:
            self._redecide_next_poll(neighbor, replies, poll, "answered", now, empty=empty)

    def on_record(self, name: str, payload: bytes, now: float) -> None:
        """A neighbor's reply to a record-fetch Interest arrived.

        Validates the payload, checks it against the (origin, seq)
        carried by the name itself (map.md "名前の origin/seq とペイロー
        ドの値が違えば拒否"), stores it, and updates the tracker so this
        seq stops being tracked as outstanding.

        2026-10-22 fix: a payload that fails validation now also calls
        Tracker.on_fetch_rejected rather than being silently dropped --
        Data had already arrived for this Interest, so no
        on_fetch_timeout would ever follow to release the in-flight
        slot, and a bad reply used to mark the fetch in-flight forever.

        2026-10-22 fix: `neighbor` used to be a separate argument from
        this same name's holder segment, with nothing checking the two
        agreed. _send_fetch_plan always builds the fetch name with
        holder=neighbor, so the two can never legitimately differ; a
        caller that passed a disagreeing value just hit
        Tracker._fail's `in_flight_neighbor != neighbor` early-return
        silently, leaving the seq marked in-flight forever. Deriving it
        from the name removes the possibility outright instead of
        adding a check for it.
        """
        try:
            parsed = names.parse(name)
        except names.MalformedNameError:
            return
        if not isinstance(parsed, names.RecordName):
            return
        neighbor = parsed.holder

        try:
            record = schema.decode(payload)
        except schema.RecordError:
            self._fail_fetch(parsed.origin, parsed.seq, neighbor, now)
            return
        if record.origin != parsed.origin or record.seq != parsed.seq:
            # name/payload mismatch; reject without storing
            self._fail_fetch(parsed.origin, parsed.seq, neighbor, now)
            return

        digest = schema.content_hash(payload)
        result = self._store.append_received(record.origin, record.seq, payload, digest)
        # 2026-10-23 fix: ts_ms is stamped here from `now` -- the engine's
        # own processing time for this store/conflict -- rather than left for
        # EventWriter.write to stamp whenever this queued event is actually
        # flushed. A record that landed while this node's Agent.step() was
        # blocked inside a probe's synchronous content fetch (agent.py's
        # _run_probe, until 2026-11-02 when that fetch stopped blocking) was
        # stored and queued right here, but not drained to events.jsonl
        # until that probe returned; without a
        # pre-set ts_ms, evalagg's own discovery-latency/consistency metrics
        # (semindex.evalagg._analyze_run_metrics) would read a ts_ms up to
        # one probe-wait's worth of seconds later than this store actually
        # happened, over-measuring propagation time and, for a store near
        # the measurement window's end, misclassifying an in-window store as
        # censored.
        ts_ms = round(now * 1000)
        if result.conflict:
            self._emit(
                {
                    "event": "conflict",
                    "ts_ms": ts_ms,
                    "origin": record.origin,
                    "seq": record.seq,
                    "kept_hash": result.existing_hash,
                    "rejected_hash": digest,
                }
            )
            return

        progress = self._store.get_progress(record.origin)
        self._tracker.observe_held(record.origin, progress.hi, progress.extra)
        if result.stored:
            self._emit(
                {"event": "record_stored", "ts_ms": ts_ms, "origin": record.origin, "seq": record.seq, "holder": neighbor}
            )

    def on_fetch_timeout(self, name: str, now: float) -> None:
        """A record-fetch Interest for `name` got no Data back in time.

        2026-10-22 fix: see on_record for why the neighbor is derived
        from the name's holder segment rather than taken as a separate
        argument.
        """
        try:
            parsed = names.parse(name)
        except names.MalformedNameError:
            return
        if not isinstance(parsed, names.RecordName):
            # CefpycoTransport also reports state-poll deadlines here. They are
            # left alone on purpose: _PollReplies judges state-poll timeouts from
            # the clock at decision time (see its docstring for why).
            return
        neighbor = parsed.holder
        # 2026-10-23 fix: ts_ms from `now`, same reasoning as on_record above.
        self._emit(
            {
                "event": "fetch_timeout",
                "ts_ms": round(now * 1000),
                "origin": parsed.origin,
                "seq": parsed.seq,
                "neighbor": neighbor,
            }
        )
        self._fail_fetch(parsed.origin, parsed.seq, neighbor, now, cause="timeout")

    def _fail_fetch(self, origin: str, seq: int, neighbor: str, now: float, *, cause: str = "rejected") -> None:
        """Resolve one failed fetch (timeout or rejected reply) against the Tracker and,
        only if it actually resolved a real in-flight fetch, log the matching
        `fetch_backoff` decision (map.md "判断の記録", kind 2).

        The state snapshot is taken *before* the Tracker call, which decrements its
        in-flight count for `neighbor` and, only when cause == "timeout", updates
        last_timeout_at -- the snapshot must show what was true going into this
        failure, not what the failure itself just changed. Tracker._fail returns
        None for a stale timeout/rejection (nothing was actually in flight any
        more); that is not a decision, so nothing is logged.
        """
        state = self._tracker.decision_state(neighbor)
        if cause == "timeout":
            wait = self._tracker.on_fetch_timeout(origin, seq, neighbor, now)
        else:
            wait = self._tracker.on_fetch_rejected(origin, seq, neighbor, now)
        if wait is not None:
            self._emit_decision(
                "fetch_backoff",
                state,
                {"neighbor": neighbor, "origin": origin, "seq": seq, "wait_s": wait, "cause": cause},
                now,
            )

    # -- holder-side serving --------------------------------------------------------

    def handle_interest(self, name: str) -> Response | None:
        """Produce this node's answer, as holder, to an incoming Interest for `name`.

        Returns None when there is nothing to answer: an unparsable
        name, a name addressed to a different holder, a record not
        held, or a fault.drop_rec injection withholding this reply. The
        caller should treat None as "let the Interest expire".
        """
        try:
            parsed = names.parse(name)
        except names.MalformedNameError:
            return None

        if isinstance(parsed, names.StateName):
            if parsed.holder != self._config.node:
                return None
            payload = holder_state.encode(self._store.all_progress())
            return Response(payload, self._config.state_data_expiry_s, self._config.state_data_cache_time_s)

        if parsed.holder != self._config.node:
            return None
        key = (parsed.origin, parsed.seq)
        remaining = self._config.fault_drop_rec.get(key, 0)
        if remaining > 0:
            self._config.fault_drop_rec[key] = remaining - 1
            return None
        payload = self._store.get_record(parsed.origin, parsed.seq)
        if payload is None:
            return None
        return Response(payload, self._config.record_data_expiry_s, self._config.record_data_cache_time_s)
