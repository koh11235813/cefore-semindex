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

from dataclasses import dataclass, field
from typing import Callable

from semindex import holder_state, names, schema
from semindex.store import Store
from semindex.tracker import Tracker
from semindex.transport import Response, Transport


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
    # backoff wait, fetch-plan target/count). True in every production config; a test
    # sets it False on an otherwise-identical engine to prove logging adds no
    # behavior of its own.
    log_decisions: bool = True


# How long past its Interest lifetime a state poll can still be answered:
# Cefore v0.12 keeps a PIT entry for interest_lifetime + CefC_Pit_CleaningTime
# (1 s; see tracker.DEFAULT_BACKOFF_INITIAL_S's comment), so Data for an older
# poll has nowhere to go. _PollReplies drops unanswered polls past this.
POLL_REPLY_GRACE_S = 1.0


@dataclass
class _Poll:
    """One state poll sent to a neighbor: its name's epoch_ms, when it was sent, and
    whether its own reply has arrived."""

    epoch_ms: int
    sent_at: float
    replied: bool = False


@dataclass
class _PollReplies:
    """Whether one neighbor has been answering this node's state polls (stage-2 map
    "判断の状態に足すもの") -- read-only bookkeeping for the `state_poll` decision state,
    never consulted by any scheduling decision.

    Each reply is matched to its own poll by the epoch_ms in the reply's name: with
    an interval shorter than the Interest lifetime several polls are outstanding at
    once, and a late reply to poll N must not mark poll N+1 answered.

    A poll's status is judged at decision time: "answered" once its reply has
    arrived (even late, as long as the poll is still held -- see `_expire`), else "timed_out" once send time + interest_lifetime_ms <=
    now -- CefpycoTransport's own deadline rule -- else "pending". It is judged here
    from the clock rather than from the transport's timeout callback because
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
    # Polls sent after the newest answered one that passed lifetime +
    # POLL_REPLY_GRACE_S unanswered and were dropped from `polls`. They count as
    # timed out for good, which keeps `polls` bounded by how many polls fit in
    # lifetime + grace, however long a neighbor stays silent.
    expired_timeouts: int = 0

    def sent(self, epoch_ms: int, now: float) -> None:
        """Record a new poll sent at `now`; it is pending until answered or expired."""
        self.polls.append(_Poll(epoch_ms, now))

    def replied(self, epoch_ms: int) -> None:
        """Mark the poll whose name carried `epoch_ms` answered and drop everything older.

        A reply to a poll no longer held (dropped as expired, or older than an
        answered one) changes nothing. Every expired poll is older than any poll
        still held, so a reply to a held poll also clears `expired_timeouts`.
        """
        for index, poll in enumerate(self.polls):
            if poll.epoch_ms == epoch_ms:
                poll.replied = True
                del self.polls[:index]
                self.expired_timeouts = 0
                return

    def _expire(self, now: float, lifetime_s: float) -> None:
        """Drop polls the reported values no longer need, oldest first; the newest
        poll is always kept, since last_poll_status is its status.

        An answered poll at the front is no longer needed once a newer poll exists.
        An unanswered one older than lifetime + POLL_REPLY_GRACE_S can no longer be
        answered in practice (Cefore has dropped its PIT entry), so it is folded
        into `expired_timeouts`.
        """
        while len(self.polls) > 1:
            oldest = self.polls[0]
            if oldest.replied:
                self.polls.pop(0)
            elif oldest.sent_at + lifetime_s + POLL_REPLY_GRACE_S <= now:
                self.polls.pop(0)
                self.expired_timeouts += 1
            else:
                return

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
        backoff_jitter: Callable[[], float] = lambda: 0.0,
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
        policies never share or perturb one stream. Its constant default
        0.0 always picks the first choice, deterministic like the jitter
        default.

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
        self._config = config
        self._store = store
        self._transport = transport
        self._poll_jitter_rng = poll_jitter_rng
        self._poll_choice_rng = poll_choice_rng
        self._tracker = Tracker(
            backoff_initial_s=config.fetch_backoff_initial_s,
            backoff_multiplier=config.fetch_backoff_multiplier,
            backoff_max_s=config.fetch_backoff_max_s,
            max_missing_per_origin=config.max_missing_per_origin,
            max_in_flight_per_neighbor=config.max_in_flight_per_neighbor,
            jitter=backoff_jitter,
        )
        self._next_state_poll: dict[str, float] = {}
        self._poll_replies: dict[str, _PollReplies] = {}
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
        picks its one deterministic action); only a state poll drawing from
        state_poll_interval_choices_s passes anything else.

        A no-op when config.log_decisions is False (test/analysis-only; every
        production config leaves it at its default True). Assumes a real decision
        has already been made -- it never re-checks that for itself, so it must only
        be called once something was actually chosen, not on a no-op.

        2026-10-23 fix: `ts_ms` is stamped here from `now` -- the processing
        time this decision was actually made at, i.e. whatever clock reading
        the caller (on_tick/on_state/_fail_fetch, all driven by the
        transport's own clock) was already holding -- rather than left for
        EventWriter.write to stamp at flush time. A decision made while this
        node's Agent is blocked inside a probe's synchronous content fetch
        (agent.py's _run_probe) is still queued, not flushed, until that probe
        returns, so the write-time stamp used to run however many seconds
        behind the decision itself the probe wait happened to take. The same
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
        # same record. Without this, a record created in the same step()
        # that then blocks inside a probe's synchronous fetch (drained only
        # after that probe returns) would log a ts_ms seconds later than it
        # was actually created.
        self._emit({"event": "record_created", "ts_ms": observed_at, "origin": record.origin, "seq": record.seq})
        return record

    # -- periodic housekeeping -----------------------------------------------------

    def on_tick(self, now: float) -> None:
        """Send due state requests and due record fetches to every neighbor."""
        for neighbor in self._config.neighbors:
            due_at = self._next_state_poll.get(neighbor, now)
            if now >= due_at:
                self._send_state_request(neighbor, now)
            self._send_fetch_plan(neighbor, now)

    def _send_state_request(self, neighbor: str, now: float) -> None:
        """Send one state-request Interest to `neighbor` and schedule its next poll."""
        epoch_ms = round(now * 1000)
        name = names.build_state_name(self._config.group, neighbor, self._config.node, epoch_ms)
        self._transport.send_interest(name, self._config.interest_lifetime_ms)
        interval, action_prob = self._choose_poll_interval()
        # The reply status is snapshotted before this poll itself is recorded as
        # sent, so the decision shows how the *previous* poll to this neighbor
        # fared, not the trivially-pending poll being sent right now.
        replies = self._poll_replies.setdefault(neighbor, _PollReplies())
        last_poll_status, consecutive_poll_timeouts = replies.snapshot(now, self._config.interest_lifetime_ms / 1000.0)
        state = {
            **self._tracker.decision_state(neighbor),
            "last_poll_status": last_poll_status,
            "consecutive_poll_timeouts": consecutive_poll_timeouts,
        }
        # map.md "判断の記録", kind 1: this *is* the decision -- which
        # neighbor, and the interval chosen for its next poll. Always made
        # (a neighbor due for a poll is always polled), so always logged
        # when log_decisions is on.
        self._emit_decision("state_poll", state, {"neighbor": neighbor, "interval_s": interval}, now, action_prob)
        replies.sent(epoch_ms, now)
        self._next_state_poll[neighbor] = now + interval

    def _choose_poll_interval(self) -> tuple[float, float]:
        """Draw the interval until a neighbor's next state poll, and the probability
        the policy gave it.

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
            return choices[index], 1.0 / len(choices)
        # Map the injected [0, 1) rng to +/- state_poll_jitter_frac around the
        # configured interval, so "2s +/-20%" is reproducible in tests yet real
        # under a real rng. action_prob stays 1.0, as in stage 1's data (stage-1
        # map's 2026-10-26 correction: those lines are not off-policy usable).
        offset = self._config.state_poll_jitter_frac * (2 * self._poll_jitter_rng() - 1)
        return self._config.state_poll_interval_s * (1 + offset), 1.0

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
        replies = self._poll_replies.get(neighbor)
        if replies is not None:
            replies.replied(poll_epoch_ms)
        try:
            decoded = holder_state.decode(payload)
        except holder_state.HolderStateError:
            self._emit({"event": "state_rx", "ts_ms": ts_ms, "neighbor": neighbor, "hi": {}})
            return  # malformed/oversize reply from a neighbor; drop silently
        self._emit(
            {"event": "state_rx", "ts_ms": ts_ms, "neighbor": neighbor, "hi": {o: e["hi"] for o, e in decoded.items()}}
        )
        for origin, entry in decoded.items():
            self._tracker.observe_state(origin, neighbor, entry["hi"], entry["extra"])
        self._send_fetch_plan(neighbor, now)

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
        # flushed. A record that lands while this node's Agent.step() is
        # blocked inside a probe's synchronous content fetch (agent.py's
        # _run_probe) is still stored and queued right here, but not
        # drained to events.jsonl until that probe returns; without a
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
