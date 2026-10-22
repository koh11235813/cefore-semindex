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


class SyncEngine:
    """The per-node Sync agent core: requester toward its neighbors, holder for itself."""

    def __init__(
        self,
        config: SyncConfig,
        store: Store,
        transport: Transport,
        *,
        poll_jitter_rng: Callable[[], float] = lambda: 0.5,
        backoff_jitter: Callable[[], float] = lambda: 0.0,
    ) -> None:
        """`poll_jitter_rng` must return a float in [0, 1); it is mapped to
        +/- state_poll_jitter_frac around state_poll_interval_s (map.md
        "状態確認の周期 2 秒 ±20%"). The default is a constant 0.5, which
        maps to zero offset, so engines are deterministic unless a real
        source of randomness is injected. `backoff_jitter` is passed
        straight through to the Tracker.

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
        self._tracker = Tracker(
            backoff_initial_s=config.fetch_backoff_initial_s,
            backoff_multiplier=config.fetch_backoff_multiplier,
            backoff_max_s=config.fetch_backoff_max_s,
            max_missing_per_origin=config.max_missing_per_origin,
            max_in_flight_per_neighbor=config.max_in_flight_per_neighbor,
            jitter=backoff_jitter,
        )
        self._next_state_poll: dict[str, float] = {}
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

    def _emit_decision(self, kind: str, state: dict, action: dict) -> None:
        """Queue one `decision` event (map.md "判断の記録"): the state visible at the
        moment this engine actually chose `action`, plus action_prob -- 1.0 here,
        since stage 1's policy always picks its one deterministic action.

        A no-op when config.log_decisions is False (test/analysis-only; every
        production config leaves it at its default True). Callers only reach here
        once they already know a real decision was made -- state_poll is only called
        when a poll is actually due, and fetch_plan/fetch_backoff's own call sites
        skip this entirely when there was nothing to send or nothing to fail -- so
        this never has to re-check that part.
        """
        if not self._config.log_decisions:
            return
        self._emit({"event": "decision", "kind": kind, "state": state, "action": action, "action_prob": 1.0})

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
        self._emit({"event": "record_created", "origin": record.origin, "seq": record.seq})
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
        # Map the injected [0, 1) rng to +/- state_poll_jitter_frac around
        # the configured interval, so "2s +/-20%" is reproducible in tests
        # yet real under a real rng. Drawn exactly once, whether or not
        # log_decisions is on: a `decision` event only reports this same
        # value below, it never draws its own.
        offset = self._config.state_poll_jitter_frac * (2 * self._poll_jitter_rng() - 1)
        interval = self._config.state_poll_interval_s * (1 + offset)
        # map.md "判断の記録", kind 1: this *is* the decision -- which
        # neighbor, and the interval chosen for its next poll. Always made
        # (a neighbor due for a poll is always polled in stage 1), so
        # always logged when log_decisions is on.
        self._emit_decision(
            "state_poll", self._tracker.decision_state(neighbor), {"neighbor": neighbor, "interval_s": interval}
        )
        self._next_state_poll[neighbor] = now + interval

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
                "fetch_plan", state, {"neighbor": neighbor, "fetched": [[origin, seq] for origin, seq in due]}
            )
        for origin, seq in due:
            name = names.build_record_name(self._config.group, neighbor, origin, seq)
            self._transport.send_interest(name, self._config.interest_lifetime_ms)

    # -- requester-side reply handling -----------------------------------------------

    def on_state(self, neighbor: str, payload: bytes, now: float) -> None:
        """A neighbor's reply to a state-request Interest arrived.

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
        try:
            decoded = holder_state.decode(payload)
        except holder_state.HolderStateError:
            self._emit({"event": "state_rx", "neighbor": neighbor, "hi": {}})
            return  # malformed/oversize reply from a neighbor; drop silently
        self._emit({"event": "state_rx", "neighbor": neighbor, "hi": {o: e["hi"] for o, e in decoded.items()}})
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
        if result.conflict:
            self._emit(
                {
                    "event": "conflict",
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
                {"event": "record_stored", "origin": record.origin, "seq": record.seq, "holder": neighbor}
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
            return
        neighbor = parsed.holder
        self._emit({"event": "fetch_timeout", "origin": parsed.origin, "seq": parsed.seq, "neighbor": neighbor})
        self._fail_fetch(parsed.origin, parsed.seq, neighbor, now, cause="timeout")

    def _fail_fetch(self, origin: str, seq: int, neighbor: str, now: float, *, cause: str = "rejected") -> None:
        """Resolve one failed fetch (timeout or rejected reply) against the Tracker and,
        only if it actually resolved a real in-flight fetch, log the matching
        `fetch_backoff` decision (map.md "判断の記録", kind 2).

        The state snapshot is taken *before* the Tracker call, which decrements its
        in-flight count for `neighbor` and sets its last_fail_at -- the snapshot must
        show what was true going into this failure, not what the failure itself just
        changed. Tracker._fail returns None for a stale timeout/rejection (nothing was
        actually in flight any more); that is not a decision, so nothing is logged.
        """
        state = self._tracker.decision_state(neighbor)
        if cause == "timeout":
            wait = self._tracker.on_fetch_timeout(origin, seq, neighbor, now)
        else:
            wait = self._tracker.on_fetch_rejected(origin, seq, neighbor, now)
        if wait is not None:
            self._emit_decision(
                "fetch_backoff", state, {"neighbor": neighbor, "origin": origin, "seq": seq, "wait_s": wait, "cause": cause}
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
