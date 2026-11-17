#!/usr/bin/env python3
"""ev13: offline discrete-event replay of the CURRENT state-poll rule, and its scoring.

    /home/lab_shared/cefore-semindex/.venv/bin/python3 -B replay.py
    /home/lab_shared/cefore-semindex/.venv/bin/python3 -B replay.py --trace s5short 0.2 1 A

Rebuilds validate_cells.csv and validate.md over the 42 cells (840 runs) of
evidence/11 and evidence/12.  Reads only `json` (no semindex import); under
`debug/node_dirs/` it opens `semindex/events.jsonl` and nothing else.

WHAT IS EXOGENOUS (taken from the run, never simulated)
  - the line topology h0-h1-h2-h3-h4 and the link_down/link_up instants, taken as
    `t0_ms + actual_at*1000` (facts-reply-fetch.md §4: measured best of the three
    candidate boundaries, residual bounded at 6 ms);
  - `record_created` ts at the origins h0 and h4;
  - `meta.json` policy settings, `stop_at_s`, `t0_ms`;
  - per-node `agent_start.ts_ms` (process launch, not a timer rule -- see
    DEVIATIONS in validate.md).

EVERYTHING ELSE IS SIMULATED for all five nodes jointly: every state poll (send
time, outcome, reply content), every fetch (plan, window of 8, timeout, 2/4/8/16 s
retry chain), every record_stored time.
"""

from __future__ import annotations

import csv
import datetime
import glob
import hashlib
import heapq
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Constants.  Fixed before the first replay number was seen; nothing below was
# changed after results (spec.md: "Do not tune constants to move a number").
# ---------------------------------------------------------------------------

STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")
EV = os.environ.get(
    "EV", "/home/lab_shared/wt/semindex-ev11/docs/wayfinder/semindex-stage2/evidence"
)
HERE = os.path.dirname(os.path.abspath(__file__))

# The link boundary.  Default (spec.md line 15): `t0 + actual_at*1000` for BOTH
# link_down and link_up.  With LINK_UP_FROM_ISO_TS=1, link_up instead takes the
# `results.json` ISO `ts`; that variant was found after seeing results and is
# reported as tuned, never as the result.  See load_run.
SPLIT_BOUNDARY = os.environ.get("LINK_UP_FROM_ISO_TS") == "1"

# The split boundary's pass counts, so the default run can print both rows of the
# "Tuned variants" table.  Produced by
#   LINK_UP_FROM_ISO_TS=1 python3 replay.py
# and copied here from that run's own verdict table (tuned_split_boundary/).
SPLIT_A1, SPLIT_A2, SPLIT_A3 = 40, 25, 42

NODES = ["h0", "h1", "h2", "h3", "h4"]

WINDOW_START_MS = 10_000.0        # evalgen.WINDOW_START_S
WINDOW_END_MS = 130_000.0         # evalgen.WINDOW_END_S, both edges inclusive
STOP_AT_MS = 130_000.0            # agent stop_at_s = 130.0 (facts-timers §6)

POLL_RTT_MS = 1.0                 # spec; facts-reply-fetch §2 measured p50 = 1 ms
STATE_RX_TO_PLAN_MS = 0.0         # spec; _send_fetch_plan runs inside on_state
FETCH_RTT_MS = 2.0                # spec; fetch_plan -> record_stored p50 = 2 ms
POLL_TIMEOUT_MS = 2000.0          # interest_lifetime_ms (engine.py:86)
FETCH_TIMEOUT_MS = 2000.0         # cefpyco_transport.py:233
MODE_A_TICK_LAG_MS = 50.0         # spec: the Mode A lag when the log has none
TICK_PERIOD_MS = 100.0            # receive_timeout_ms; Mode B lag ~ U[0, 100]
# facts-reply-fetch §6: a backed-off retry is re-planned +1..2 ms after
# next_attempt_at, not +[0,100] ms -- the wait is a whole number of tick periods,
# so the retry lands on the same idle-tick phase that detected the timeout.
RETRY_LAG_MS = 2.0

MAX_IN_FLIGHT = 8                 # max_in_flight_per_neighbor (engine.py:94)
MAX_MISSING_PER_ORIGIN = 1024     # tracker.py
BACKOFF_INIT_S = 2.0
BACKOFF_MULT = 2.0
BACKOFF_MAX_S = 16.0

INT_B = 50.9
REP_B = {0.0: 132.5, 0.1: 131.7, 0.2: 130.4, 0.3: 128.2, 0.4: 125.5}

R_REPLICATES = 5
BOOT_N = 2000
BOOT_SEED = 20261117
WORKERS = 16

# pass line (spec.md, fixed)
A1_MIN = 0.995
A2_MIN = 0.99
A2_TOL_MS = 150.0
A2_ONLYONE_MAX = 0.005
A3_POLLS_REL = 0.005
A3_DELAY_REL = 0.01
A3_DELAY_ABS = 50.0
A3_CENS_PP = 0.005
B1_MIN_FRAC = 0.90
B2_DELAY_REL = 0.02
B2_POLLS_REL = 0.01
B2_BYTES_REL = 0.01
B2_CENS_PP = 0.01

CELLS = (
    [(s, p) for s in ("s5b0p5", "s5b1p359", "s5b3p695", "s5b10p043", "s5short", "s5wait")
     for p in (0.0, 0.2, 0.4)]
    + [(s, p) for s in ("s6b0p5", "s6b1p359", "s6b3p695", "s6b10p043", "s6short", "s6wait")
       for p in (0.2, 0.4, 0.5, 0.6)]
)
SEEDS = list(range(1, 21))


def pdir(p: float) -> str:
    """Directory spelling of a p value: p0, p0.2, p0.5 ..."""
    return "p0" if p == 0.0 else "p" + ("%g" % p)


# ---------------------------------------------------------------------------
# Loading one run
# ---------------------------------------------------------------------------


def load_run(stage: str, p: float, seed: int) -> dict:
    run_dir = f"{STORE}/{stage}/{pdir(p)}/{seed}"
    events = {}
    for n in NODES:
        path = f"{run_dir}/debug/node_dirs/{n}/semindex/events.jsonl"
        with open(path) as f:
            events[n] = [json.loads(line) for line in f if line.strip()]

    meta = json.load(open(f"{run_dir}/meta.json"))
    exp_path = glob.glob(f"{run_dir}/*.experiment.json")[0]
    exp = json.load(open(exp_path))
    results = json.load(open(f"{run_dir}/results.json"))

    t0s = {e["t0_ms"] for rows in events.values() for e in rows if e.get("event") == "agent_start"}
    assert len(t0s) == 1, (stage, p, seed, t0s)
    t0 = float(t0s.pop())
    agent_start = {
        n: float(next(e["ts_ms"] for e in rows if e.get("event") == "agent_start"))
        for n, rows in events.items()
    }

    # link timeline per unordered node pair, from results.json actual_at
    timeline: dict[frozenset, list] = defaultdict(list)
    for row in results:
        if row.get("op_type") != "event":
            continue
        pair = frozenset(row["event"]["nodes"])
        is_up = row["event_type"] == "link_up"
        # spec.md line 15: both link directions come from "the mapping
        # facts-reply-fetch.md fact 8 found best", which is `t0 + actual_at*1000`
        # for link_down AND link_up.  This is the boundary the spec fixes, so it is
        # the default here and the number validate.md reports.
        #
        # SPLIT_BOUNDARY is the variant that takes the ISO `ts` (the completion of
        # the interface change) for link_up only.  It scores better, but it was
        # tried after the first results were seen, so validate.md reports it as
        # tuned and not as the result.  Off unless LINK_UP_FROM_ISO_TS=1.
        t = (datetime.datetime.fromisoformat(row["ts"]).timestamp() * 1000.0
             if (is_up and SPLIT_BOUNDARY)
             else t0 + float(row["actual_at"]) * 1000.0)
        timeline[pair].append((t, is_up))
    for v in timeline.values():
        v.sort()

    created = []  # (ts, origin, seq)
    for n, rows in events.items():
        for e in rows:
            if e.get("event") == "record_created":
                created.append((float(e["ts_ms"]), e["origin"], e["seq"]))
    created.sort()

    links = [tuple(sorted(pair)) for pair in exp["links"]]
    neighbors = {n: [] for n in NODES}
    for a, b in links:
        neighbors[f"h{a}"].append(f"h{b}")
        neighbors[f"h{b}"].append(f"h{a}")
    for n in neighbors:
        neighbors[n].sort(key=lambda h: int(h[1:]))  # config order = ascending host index

    return dict(
        stage=stage, p=p, seed=seed, run_dir=run_dir, events=events, t0=t0,
        agent_start=agent_start, timeline=dict(timeline), created=created,
        neighbors=neighbors, group=exp["group"],
        policy=meta.get("state_poll_policy"),
        choices=meta.get("state_poll_interval_choices_s"),
        choice_jitter=meta.get("state_poll_choice_jitter_frac"),
        # No run in the store was produced by a spec2 rule; a caller that wants one
        # simulated sets this key (or passes `rule=` to _sim_run), the way
        # mine/swap.py already overwrites policy/choices to put another stage's rule
        # on a donor run's exogenous inputs.
        rule=None,
        stop_at_ms=STOP_AT_MS,
    )


def logged_lags(run: dict) -> dict:
    """Per (node, neighbor), the logged tick lag of each state_poll, in order.

    lag_i = logged ts of poll i minus the due time the previous decision lines set
    (facts-timers.md §7/§8).  The first poll of a pair has no predecessor, so its
    "due" is the node's agent_start ts (facts-timers §3: the first on_tick polls
    every neighbor).
    """
    lags: dict = {}
    for node, rows in run["events"].items():
        due: dict = {}
        seen = set()
        for e in rows:
            if e.get("event") != "decision":
                continue
            kind = e.get("kind")
            a = e.get("action", {})
            if kind == "state_poll":
                nb, ts = a["neighbor"], float(e["ts_ms"])
                if nb in seen:
                    lag = ts - due[nb]
                else:
                    seen.add(nb)
                    lag = ts - run["agent_start"][node]
                lags.setdefault((node, nb), []).append(lag)
                due[nb] = ts + float(a.get("applied_interval_s", a["interval_s"])) * 1000.0
            elif kind == "state_poll_reschedule":
                nb = a["neighbor"]
                due[nb] = max(
                    float(a["poll_epoch_ms"]) + float(a["applied_interval_s"]) * 1000.0,
                    float(e["ts_ms"]),
                )
    return lags


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------

# heap priority at equal timestamps; a plan must run after the store that freed
# its slot, a poll goes last so a reply that lands on the same ms is folded first.
P_CREATE, P_STORE, P_RX, P_FTIMEOUT, P_POLLTO, P_PLAN, P_POLL = 0, 1, 2, 3, 4, 5, 6


class _Origin:
    __slots__ = ("hi", "extra", "advertised", "fetches", "hist")

    def __init__(self):
        self.hi = 0
        self.extra: set[int] = set()
        self.advertised: dict[str, tuple[int, tuple]] = {}
        # seq -> [next_attempt_at, backoff_s, in_flight_neighbor|None, failed_neighbors set]
        self.fetches: dict[int, list] = {}
        # (millisecond stamp, hi, extra tuple) after each fold, so a reply can be
        # answered with the store as it stood at the instant the Interest was served
        self.hist: list[tuple[int, int, tuple]] = []


class Node:
    def __init__(self, name, neighbors, seed, policy, choices, choice_jitter, rule=None):
        # `rule` is spec2 Part 2's "lengthen the interval while replies are empty":
        # {"i_min", "m", "i_max", "jf"}.  It replaces the policy table as the source of
        # the interval and keeps one number per neighbor, `ival`, starting at i_min.
        # It is mutually exclusive with `policy` and `choices`; it runs through the
        # policy MECHANISM (decide at send, re-decide at resolution with the poll's own
        # jitter draw), which is why the simulator branches on `policy is not None or
        # rule is not None` wherever the resolution matters.
        self.rule = rule
        self.ival = {nb: rule["i_min"] for nb in neighbors} if rule else {}
        self.name = name
        self.neighbors = neighbors
        self.origins: dict[str, _Origin] = {}
        self.in_flight: dict[str, int] = {nb: 0 for nb in neighbors}
        self.due: dict[str, float] = {}
        self.gen: dict[str, int] = {nb: 0 for nb in neighbors}
        self.consec: dict[str, int] = {nb: 0 for nb in neighbors}
        self.pending: dict[str, tuple] = {}   # nb -> (sent_at, jitter_u, resolved)
        self.poll_ord: dict[str, int] = {nb: 0 for nb in neighbors}
        self.policy = policy
        self.choices = choices
        self.choice_jitter = choice_jitter
        self.jitter_rng = random.Random(f"state_poll_interval_jitter:{seed}:{name}").random
        self.choice_rng = random.Random(f"state_poll_interval:{seed}:{name}").random

    def org(self, origin) -> _Origin:
        o = self.origins.get(origin)
        if o is None:
            o = self.origins[origin] = _Origin()
        return o

    # -- store ------------------------------------------------------------
    def fold(self, origin, seq, ts_ms=None):
        """store._fold_progress: extend hi (absorbing contiguous extras) or add to extra."""
        o = self.org(origin)
        if seq == o.hi + 1:
            o.hi = seq
            while o.hi + 1 in o.extra:
                o.extra.discard(o.hi + 1)
                o.hi += 1
        elif seq > o.hi:
            o.extra.add(seq)
        if ts_ms is not None:
            o.hist.append((ts_ms, o.hi, tuple(sorted(o.extra))))
        # tracker.observe_held: a seq now held is no longer a fetch
        for s in [s for s in o.fetches if s <= o.hi or s in o.extra]:
            f = o.fetches.pop(s)
            if f[2] is not None:
                self.in_flight[f[2]] = max(0, self.in_flight[f[2]] - 1)

    def progress_map(self) -> dict:
        """What this node advertises as holder: {origin: hi} over every origin in its store."""
        return {o: st.hi for o, st in sorted(self.origins.items()) if st.hi or st.extra}

    def progress_map_asof(self, ms: int) -> dict:
        """The same map as it stood strictly before millisecond `ms`.

        2026-11-17 model fix: agent.step() calls transport.poll() -- which serves
        every incoming Interest synchronously (cefpyco_transport.py:397-407) --
        BEFORE _drain_schedule(now) creates that iteration's own records
        (agent.py:1072-1073).  So a reply served in the same millisecond as a
        record_created/record_stored at the holder carries the state from before
        it.  Answering with the holder's state at the reply's receive stamp
        instead moved whole 4-record batches one poll interval early downstream,
        which showed up as 500 ms store-time errors in the 0.5 s arm.
        """
        out = {}
        for o, st in sorted(self.origins.items()):
            lo, hi_i = 0, len(st.hist)
            while lo < hi_i:                      # last entry with stamp < ms
                mid = (lo + hi_i) // 2
                if st.hist[mid][0] < ms:
                    lo = mid + 1
                else:
                    hi_i = mid
            if lo:
                h, ex = st.hist[lo - 1][1], st.hist[lo - 1][2]
                if h or ex:
                    out[o] = h
        return out

    def extras_asof(self, origin, ms: int) -> tuple:
        st = self.origins.get(origin)
        if st is None:
            return ()
        best = ()
        for stamp, _h, ex in st.hist:
            if stamp < ms:
                best = ex
            else:
                break
        return best

    # -- tracker ----------------------------------------------------------
    def notified_max(self, origin) -> int:
        o = self.origins.get(origin)
        if o is None or not o.advertised:
            return 0
        return max(max(hi, max(ex, default=0)) for hi, ex in o.advertised.values())

    def missing(self, origin):
        o = self.origins.get(origin)
        if o is None:
            return []
        upper = self.notified_max(origin)
        out = []
        for seq in range(o.hi + 1, upper + 1):
            if seq not in o.extra:
                out.append(seq)
                if len(out) >= MAX_MISSING_PER_ORIGIN:
                    break
        return out

    def has_advertiser_outside(self, origin, seq, excluded) -> bool:
        o = self.origins[origin]
        for other, (hi, ex) in o.advertised.items():
            if other in excluded:
                continue
            if seq <= hi or seq in ex:
                return True
        return False

    def plan_fetches(self, nb, now):
        """tracker.plan_fetches, verbatim in behaviour."""
        headroom = MAX_IN_FLIGHT - self.in_flight.get(nb, 0)
        if headroom <= 0:
            return []
        due = []
        for origin in sorted(self.origins):
            if len(due) >= headroom:
                break
            o = self.origins[origin]
            adv = o.advertised.get(nb)
            if adv is None:
                continue
            hi, extra = adv
            extra_set = set(extra)
            for seq in self.missing(origin):
                if len(due) >= headroom:
                    break
                if seq > hi and seq not in extra_set:
                    continue
                f = o.fetches.get(seq)
                if f is not None:
                    if f[2] is not None or f[0] > now:
                        continue
                    if nb in f[3] and self.has_advertiser_outside(origin, seq, f[3]):
                        continue
                due.append((origin, seq))
        for origin, seq in due:
            o = self.origins[origin]
            prev = o.fetches.get(seq)
            backoff = prev[1] if prev is not None else BACKOFF_INIT_S
            failed = prev[3] if prev is not None else frozenset()
            o.fetches[seq] = [now, backoff, nb, failed]
            self.in_flight[nb] = self.in_flight.get(nb, 0) + 1
        return due

    def fail_fetch(self, origin, seq, nb, now):
        """tracker._fail: release the slot, back off, remember the failure."""
        o = self.origins.get(origin)
        if o is None:
            return None
        f = o.fetches.get(seq)
        if f is None or f[2] != nb:
            return None
        wait = f[1]
        nxt = min(f[1] * BACKOFF_MULT, BACKOFF_MAX_S)
        failed = frozenset(f[3]) | {nb}
        if not self.has_advertiser_outside(origin, seq, failed):
            failed = frozenset({nb})
        o.fetches[seq] = [now + wait * 1000.0, nxt, None, failed]
        self.in_flight[nb] = max(0, self.in_flight.get(nb, 0) - 1)
        return wait


def _sim_run(run, mode, lags=None, sim_rng=None, rule=None):
    """Run one simulation and return {node: [event dicts]} shaped like events.jsonl.

    `rule` (spec2 Part 2) replaces the run's own poll-interval rule; it also silences
    `policy` and `choices`, so the caller does not have to clear them.  Without it the
    run's logged rule is simulated, which is what spec.md's validation needs.
    """
    s = _Engine(run, mode, lags, sim_rng, rule)
    return s.go()


class _Engine:
    """The actual event loop (kept separate so every handler sees its fire time)."""

    def __init__(self, run, mode, lags, sim_rng, rule=None):
        self.run = run
        self.mode = mode
        self.lags = lags or {}
        self.rng = sim_rng
        self.t0 = run["t0"]
        self.timeline = run["timeline"]
        rule = rule if rule is not None else run.get("rule")
        self.nodes = {
            n: Node(n, run["neighbors"][n], run["seed"],
                    None if rule else run["policy"],
                    None if rule else run["choices"], run["choice_jitter"], rule)
            for n in NODES
        }
        self.events = {n: [] for n in NODES}
        self.heap = []
        self.counter = 0

    # link -------------------------------------------------------------
    def delivers(self, a, b, t_send, t_arrive):
        """True when a packet sent at t_send over link a-b and arriving at t_arrive
        is delivered.

        facts-reply-fetch.md §5/§6: a packet put on a down link is lost outright and
        is never delivered late; one already in flight when the link drops is lost.

        What the loop below actually decides, in order:
        - the link state at send is the state set by the last timeline event
          STRICTLY BEFORE t_send (up before the first event), so an event stamped at
          exactly t_send does not yet change the state the packet is put on;
        - a link_down stamped in [t_send, t_arrive) loses the packet, which is the
          "already in flight when the link drops" case.  A link_down stamped at
          exactly t_send therefore still loses the packet, through this second test
          rather than the first;
        - a link_up stamped in [t_send, t_arrive) does NOT rescue a packet put on a
          down link: the return value is the state at send.

        The strictness of the first test is what matters for A1: `t0 + actual_at*1000`
        is when the interface change *starts* and facts-reply-fetch.md §4 measured the
        true flip a few ms after it (the 5 residual misfits sit at +0, +0, +2, +4,
        +6 ms, never before).  With `<=` there, a poll sent in exactly the boundary
        millisecond was judged against the new state, and in a policy stage that one
        wrong outcome desynchronised the node's jitter stream for the rest of the run
        -- A1 fell to 90 % on such runs.
        """
        tl = self.timeline.get(frozenset((int(a[1:]), int(b[1:]))))
        if not tl:
            return True
        up = True
        for ts, is_up in tl:
            if ts < t_send:
                up = is_up
            elif ts < t_arrive:
                if not is_up:
                    return False
            else:
                break
        return up

    def push(self, t, prio, node, nb, fn):
        self.counter += 1
        heapq.heappush(self.heap, (t, prio, int(node[1:]),
                                   int(nb[1:]) if nb else -1, self.counter, fn))

    def tick_lag(self, node, nb, kind):
        if self.mode == "A":
            if kind == "poll":
                i = self.nodes[node].poll_ord[nb]
                lst = self.lags.get((node, nb))
                if lst is not None and i < len(lst):
                    return lst[i]
            return MODE_A_TICK_LAG_MS
        return self.rng.uniform(0.0, TICK_PERIOD_MS)

    def emit(self, node, ev):
        self.events[node].append(ev)

    def go(self):
        for ts, origin, seq in self.run["created"]:
            self.push(ts, P_CREATE, origin, None,
                      lambda t, o=origin, s=seq: self.ev_create(t, o, s))
        for n in NODES:
            start = self.run["agent_start"][n]
            nbs = self.nodes[n].neighbors
            if self.mode == "A":
                lag0 = {nb: (self.lags[(n, nb)][0] if self.lags.get((n, nb))
                             else MODE_A_TICK_LAG_MS) for nb in nbs}
            else:
                shared = self.rng.uniform(0.0, TICK_PERIOD_MS)
                lag0 = {nb: shared for nb in nbs}
            for nb in nbs:
                self.push(start + lag0[nb], P_POLL, n, nb,
                          lambda t, nn=n, b=nb: self.ev_poll(t, nn, b, 0))
        while self.heap:
            t, _p, _n, _nb, _c, fn = heapq.heappop(self.heap)
            fn(t)
        for n in NODES:
            # 2026-11-17 model fix: a STABLE sort on ts_ms only.  Handlers already run
            # in float-time order, so this keeps the engine's own emission order inside
            # one millisecond.  Sorting state_rx ahead of a state_poll that shares its
            # stamp made the scorer charge that reply to the PREVIOUS poll, leaving a
            # poll that was in fact answered counted as timed out.
            self.events[n].sort(key=lambda e: e["ts_ms"])
        return self.events

    # handlers ------------------------------------------------------------
    def ev_create(self, t, origin, seq):
        ms = int(round(t))
        self.nodes[origin].fold(origin, seq, ms)
        self.emit(origin, {"event": "record_created", "origin": origin,
                           "seq": seq, "ts_ms": ms})

    def ev_poll(self, t, node, nb, gen):
        nd = self.nodes[node]
        if gen != nd.gen[nb]:
            return                       # superseded by a reschedule
        if t - self.t0 >= STOP_AT_MS:
            return   # agent._on_receive_boundary: elapsed >= stop_at_s takes the
                     # early return BEFORE on_tick, so no further Interest is sent
        u_j, interval, jf = self.decide_interval(nd, nb)
        applied = interval * (1.0 + jf * (2.0 * u_j - 1.0))
        ts = int(round(t))
        action = {"neighbor": nb, "interval_s": interval, "applied_interval_s": applied}
        self.emit(node, {"event": "decision", "kind": "state_poll", "ts_ms": ts,
                         "action": action,
                         "state": {"consecutive_poll_timeouts": nd.consec[nb]}})
        nd.poll_ord[nb] += 1
        pend = nd.pending[nb] = [t, u_j, False, None, None]
        nd.due[nb] = t + applied * 1000.0
        nd.gen[nb] += 1
        g = nd.gen[nb]
        if nd.policy is None and nd.rule is None:
            self.push(nd.due[nb] + self.tick_lag(node, nb, "poll"), P_POLL, node, nb,
                      lambda tt, nn=node, b=nb, gg=g: self.ev_poll(tt, nn, b, gg))
        elif nd.rule is not None:
            # The spec2 rule runs the policy MECHANISM but draws its tick lag HERE, at
            # send time, exactly where the choices path above draws it, and remembers
            # both the lag and the provisional due instant.  The resolution then re-arms
            # the timer only when its recomputed due differs (see resolve).  That is
            # what makes the identity test exact in Mode B: with m = 1 and
            # I_min = I_max = X the interval never changes, so the rule path consumes
            # the single shared `sim_rng` stream in the same order, at the same instants,
            # as the fixed-interval arm X.
            lag = self.tick_lag(node, nb, "poll")
            pend[3], pend[4] = lag, nd.due[nb]
            self.push(nd.due[nb] + lag, P_POLL, node, nb,
                      lambda tt, nn=node, b=nb, gg=g: self.ev_poll(tt, nn, b, gg))
        else:
            # 2026-11-17 model fix: engine.py:478 sets `_next_state_poll[neighbor] =
            # now + applied_interval` on EVERY send, outside the policy/choices
            # branch, and its own comment (engine.py:474-477) calls that value "only
            # provisional" under a state_poll_policy: _redecide_next_poll overrides it
            # when the poll resolves.  The replay used to arm no timer at all in a
            # policy stage, so a poll that resolves after its own provisional due time
            # -- which needs a table row below the 2 s interest lifetime -- would never
            # have been sent.  Every row of s5short/s5wait/s6short/s6wait exceeds 2 s,
            # so for those cells the resolution always lands first and this event is
            # always stale; the new rule of spec2 can go below 2 s, and then this is
            # the timer that fires.
            #
            # The event is queued at the BARE due time and the tick lag is drawn only
            # once the event is found to be still live (ev_poll_due).  A lag drawn here
            # would consume one `sim_rng` value per poll in Mode B and shift every later
            # draw of the run, which would move the 14 existing policy cells although
            # their schedule is unchanged.  Drawing it at the due instant keeps the
            # Mode B stream of those cells exactly as it was.
            self.push(nd.due[nb], P_POLL, node, nb,
                      lambda tt, nn=node, b=nb, gg=g: self.ev_poll_due(tt, nn, b, gg))
        # outcome
        arrive = t + POLL_RTT_MS
        if self.delivers(node, nb, t, arrive):
            # 2026-11-17 model fix: the reply carries the holder's store at the
            # instant the Interest is SERVED -- handle_interest runs synchronously
            # inside the holder's poll(), between the requester's send and its
            # receive (facts-reply-fetch.md §1).  Snapshotting at the receive stamp
            # instead let a record created in the same millisecond leak into the
            # reply, which moved whole batches one poll interval early downstream.
            self.push(arrive, P_RX, node, nb,
                      lambda tt, nn=node, b=nb, s=t: self.ev_rx(tt, nn, b, s))
        else:
            det = t + POLL_TIMEOUT_MS + self.tick_lag(node, nb, "timeout")
            self.push(det, P_POLLTO, node, nb,
                      lambda tt, nn=node, b=nb, s=t: self.ev_poll_timeout(tt, nn, b, s))

    def ev_poll_due(self, t, node, nb, gen):
        """The provisional send-time timer of a policy poll has come due.

        Split off from the push so the tick lag is drawn only for a timer that is
        still live: a resolution (engine.py:507-522) bumps `gen` and arms its own
        timer, and the provisional one then costs no RNG draw at all.  The lag is
        the same quantisation as any other poll: the next agent tick at or after
        the due time sends the Interest (facts-timers.md §8).
        """
        nd = self.nodes[node]
        if gen != nd.gen[nb]:
            return
        self.push(t + self.tick_lag(node, nb, "poll"), P_POLL, node, nb,
                  lambda tt, nn=node, b=nb, gg=gen: self.ev_poll(tt, nn, b, gg))

    def decide_interval(self, nd, nb):
        """One draw per poll from each stream the mode uses (facts-timers §4/§4b)."""
        if nd.rule is not None:
            # spec2 Part 2, at send: applied = I * (1 + jf*(2u-1)), one jitter draw per
            # poll, I the rule's current per-neighbor interval.  Only the jitter stream
            # is drawn, like the policy path (engine.py docstring: with a
            # state_poll_policy `poll_choice_jitter_rng` is the only poll rng drawn).
            # The choices path also draws `choice_rng`, but that is a SEPARATE
            # random.Random instance, so not drawing it leaves the jitter values of the
            # identity test bit-identical; a dummy draw would add nothing.
            return nd.jitter_rng(), nd.ival[nb], nd.rule["jf"]
        if nd.policy is not None:
            row = min(nd.consec[nb], len(nd.policy["intervals_s"]) - 1)
            interval = nd.policy["intervals_s"][row]
            jf = nd.policy["jitter_frac"][row]
            return nd.jitter_rng(), interval, jf
        if nd.choices is not None:
            u_c = nd.choice_rng()
            interval = nd.choices[int(u_c * len(nd.choices))]
            return nd.jitter_rng(), interval, nd.choice_jitter
        return nd.jitter_rng(), 2.0, 0.2      # SyncConfig defaults (stage-1 fixed)

    def ev_rx(self, t, node, nb, sent_at):
        """The reply lands.  Its body is the holder's store at the SERVE instant,
        which lies strictly between the requester's send and its receive: holder
        events stamped at the send millisecond are in, events stamped at the
        receive millisecond are out (facts-reply-fetch.md §1 found the fitting
        offset is 0 or -1 ms against the receive stamp, never positive)."""
        serve_ms = int(round(t))      # == send stamp + POLL_RTT_MS
        nd = self.nodes[node]
        holder = self.nodes[nb]
        hi_map = holder.progress_map_asof(serve_ms)
        # `no_new` is computed BEFORE this reply's advertisements are folded in and
        # before plan(), i.e. against what the node held when the reply arrived -- the
        # instant evidence/11 emptypoll.py's rx_no_new uses.  It is attached to every
        # simulated reply, whatever the rule, so one scorer can count the EMPTY share
        # for the fixed arms, short, wait and the spec2 variants alike.
        empty = self.no_new(nd, hi_map)
        self.emit(node, {"event": "state_rx", "neighbor": nb, "hi": hi_map,
                         "no_new": empty, "ts_ms": int(round(t))})
        if nd.policy is not None or nd.rule is not None:
            self.resolve(t, node, nb, sent_at, "answered", empty=empty)
        for origin, hi in hi_map.items():
            nd.org(origin).advertised[nb] = (hi, holder.extras_asof(origin, serve_ms))
        self.plan(t + STATE_RX_TO_PLAN_MS, node, nb)

    def no_new(self, nd, hi_map):
        """True when the reply `hi_map` brings `nd` nothing it does not already hold.

        evidence/11 emptypoll.py's `rx_no_new`, verbatim in behaviour: for every origin
        o in the reply's `hi`, every seq 1..hi[o] is already held by the node; an empty
        `hi` counts as no_new.  It ignores the HOLDER's sparse `extra`, because
        `state_rx` logs only `hi` -- the same restriction the evidence definition
        carries, and the reason the evidence calls the resulting byte figure a
        perfect-foresight reference rather than a reachable saving.
        """
        for origin, h in hi_map.items():
            st = nd.origins.get(origin)
            if st is None:
                if h > 0:
                    return False
                continue
            if h <= st.hi:
                continue
            if any(s not in st.extra for s in range(st.hi + 1, h + 1)):
                return False
        return True

    def ev_poll_timeout(self, t, node, nb, sent_at):
        nd = self.nodes[node]
        if nd.policy is None and nd.rule is None:
            return
        self.resolve(t, node, nb, sent_at, "timed_out")

    def resolve(self, t, node, nb, sent_at, status, empty=None):
        """engine._redecide_next_poll: fresh row, the poll's OWN jitter draw, measured
        from the poll's own send time, clamped forward to now.

        Returns without doing anything when `sent_at` is not this neighbor's newest
        poll: engine.py:503 resolves `replies.polls[-1]` only, and engine.py:638-658
        says a reply to an older unresolved poll changes nothing, "the poll after it
        has already been sent".  With an interval below the 2 s lifetime that is no
        longer a corner case -- the provisional timer has then already sent the next
        poll -- so under the spec2 rule whole polls go out unresolved and the rule's
        interval simply does not move on them.
        """
        nd = self.nodes[node]
        pend = nd.pending.get(nb)
        if pend is None or pend[0] != sent_at or pend[2]:
            return
        pend[2] = True
        nd.consec[nb] = 0 if status == "answered" else nd.consec[nb] + 1
        if nd.rule is not None:
            r = nd.rule
            if status == "answered":
                # spec2 Part 2: EMPTY lengthens by m up to I_max, a reply with anything
                # new resets to I_min.  On a timeout I is left alone -- the rule says
                # nothing about outages, and the short/wait policies are the arms that
                # answer that question.
                nd.ival[nb] = (min(nd.ival[nb] * r["m"], r["i_max"]) if empty
                               else r["i_min"])
            interval, jf = nd.ival[nb], r["jf"]
        else:
            row = min(nd.consec[nb], len(nd.policy["intervals_s"]) - 1)
            interval = nd.policy["intervals_s"][row]
            jf = nd.policy["jitter_frac"][row]
        applied = interval * (1.0 + jf * (2.0 * pend[1] - 1.0))
        due_at = sent_at + applied * 1000.0
        clamped = due_at < t
        due = max(due_at, t)
        self.emit(node, {"event": "decision", "kind": "state_poll_reschedule",
                         "ts_ms": int(round(t)),
                         "action": {"neighbor": nb, "poll_epoch_ms": int(round(sent_at)),
                                    "trigger": status, "interval_s": interval,
                                    "applied_interval_s": applied, "clamped": clamped},
                         "state": {"last_poll_status": status,
                                   "consecutive_poll_timeouts": nd.consec[nb]}})
        if nd.rule is not None and pend[4] is not None and due == pend[4]:
            # The provisional timer this poll armed at send already points at exactly
            # this instant, so engine.py:522's `_next_state_poll[neighbor] = max(due_at,
            # now)` writes back the value that is already there.  Re-arming it here
            # would draw a second tick lag for the same poll and shift the Mode B
            # stream; leaving it alone is what keeps the identity test bit-exact.
            return
        nd.due[nb] = due
        nd.gen[nb] += 1
        g = nd.gen[nb]
        self.push(due + self.tick_lag(node, nb, "poll"), P_POLL, node, nb,
                  lambda tt, nn=node, b=nb, gg=g: self.ev_poll(tt, nn, b, gg))

    def plan(self, t, node, nb):
        if t - self.t0 >= STOP_AT_MS:
            return
        nd = self.nodes[node]
        due = nd.plan_fetches(nb, t)
        if not due:
            return
        self.emit(node, {"event": "decision", "kind": "fetch_plan", "ts_ms": int(round(t)),
                         "action": {"neighbor": nb, "fetched": [list(x) for x in due]}})
        arrive = t + FETCH_RTT_MS
        for origin, seq in due:
            if self.delivers(node, nb, t, arrive):
                self.push(arrive, P_STORE, node, nb,
                          lambda tt, nn=node, o=origin, s=seq, b=nb:
                          self.ev_store(tt, nn, o, s, b))
            else:
                det = t + FETCH_TIMEOUT_MS + self.tick_lag(node, nb, "timeout")
                self.push(det, P_FTIMEOUT, node, nb,
                          lambda tt, nn=node, o=origin, s=seq, b=nb:
                          self.ev_ftimeout(tt, nn, o, s, b))

    def ev_store(self, t, node, origin, seq, nb):
        nd = self.nodes[node]
        ms = int(round(t))
        nd.fold(origin, seq, ms)
        self.emit(node, {"event": "record_stored", "origin": origin, "seq": seq,
                         "holder": nb, "ts_ms": ms})
        self.plan(t, node, nb)           # on_tick runs right after any packet arrives

    def ev_ftimeout(self, t, node, origin, seq, nb):
        nd = self.nodes[node]
        wait = nd.fail_fetch(origin, seq, nb, t)
        if wait is None:
            return
        self.emit(node, {"event": "fetch_timeout", "origin": origin, "seq": seq,
                         "neighbor": nb, "ts_ms": int(round(t))})
        self.emit(node, {"event": "decision", "kind": "fetch_backoff", "ts_ms": int(round(t)),
                         "action": {"neighbor": nb, "origin": origin, "seq": seq,
                                    "wait_s": wait, "cause": "timeout"}})
        self.plan(t, node, nb)                       # the freed slot may admit another seq
        self.push(t + wait * 1000.0 + RETRY_LAG_MS, P_PLAN, node, nb,
                  lambda tt, nn=node, b=nb: self.plan(tt, nn, b))


# ---------------------------------------------------------------------------
# Scoring -- ONE code path, run on logged and on simulated events alike
# ---------------------------------------------------------------------------


def score(events: dict, t0: float, p: float) -> dict:
    ws, we = t0 + WINDOW_START_MS, t0 + WINDOW_END_MS

    created, stored = {}, {}
    for n, rows in events.items():
        for e in rows:
            ev = e.get("event")
            if ev == "record_created":
                k = (e["origin"], e["seq"])
                if k not in created or e["ts_ms"] < created[k]:
                    created[k] = e["ts_ms"]
            elif ev == "record_stored":
                k = (e["origin"], e["seq"], n)
                if k not in stored or e["ts_ms"] < stored[k]:
                    stored[k] = e["ts_ms"]

    n4 = c4 = 0
    lat4 = 0.0
    for (o, s), c in created.items():
        if not (ws <= c <= we):
            continue
        node = "h4" if o == "h0" else "h0"
        ts = stored.get((o, s, node))
        if ts is None or ts > we:
            c4 += 1
        else:
            n4 += 1
            lat4 += ts - c

    # polls and their outcome: a reply is attributed to the latest poll to that
    # neighbor at or before the reply's rx ts (evidence/05 attrib.py:266,274)
    polls, answered_in_window, n_state_rx, empty_in_window = 0, 0, 0, 0
    labels = {}
    for n, rows in events.items():
        last_poll = {}
        answered_flag = {}
        order = defaultdict(list)
        for e in rows:
            ev = e.get("event")
            if ev == "decision" and e.get("kind") == "state_poll":
                nb = e["action"]["neighbor"]
                key = (n, nb, e["ts_ms"], len(order[(n, nb)]))
                last_poll[nb] = key
                answered_flag[key] = False
                order[(n, nb)].append(key)
                if ws <= e["ts_ms"] <= we:
                    polls += 1
            elif ev == "state_rx":
                nb = e["neighbor"]
                if ws <= e["ts_ms"] <= we:
                    n_state_rx += 1
                # attribution is NOT window-filtered: a reply is charged to the latest
                # poll to that neighbor at or before its rx ts, wherever it lands, and
                # only the POLL's own window membership decides whether it is counted.
                # (Window-filtering the reply too would label every pre-window poll
                # "timed_out" on both sides and make A1 trivially agree on them.)
                k = last_poll.get(nb)
                if k is not None and not answered_flag[k]:
                    answered_flag[k] = True
                    if ws <= k[2] <= we:
                        answered_in_window += 1
                        # The EMPTY share spec2 Part 3 asks for, over the same
                        # in-window answered polls `answered` counts.  Only simulated
                        # replies carry `no_new` (the engine computes it against the
                        # node's store at the serve instant); logged state_rx lines do
                        # not, so this counter is 0 on the measured side and is used
                        # for simulated runs only.
                        empty_in_window += int(bool(e.get("no_new")))
        for pair, keys in order.items():
            labels[pair] = ["answered" if answered_flag[k] else "timed_out" for k in keys]

    timed_out = polls - answered_in_window
    bytes_ = (answered_in_window * (INT_B + REP_B[p]) + timed_out * INT_B
              if p in REP_B else None)
    return dict(n4=n4, c4=c4, lat4=lat4, polls=polls, answered=answered_in_window,
                state_rx=n_state_rx, empty=empty_in_window,
                timed_out=timed_out, bytes=bytes_, labels=labels, stored=stored)


# ---------------------------------------------------------------------------
# One run: measured + Mode A + R Mode-B replicates
# ---------------------------------------------------------------------------


def rep_seed(stage, p, seed, rep) -> int:
    h = hashlib.sha256(f"{stage}:{p}:{seed}:{rep}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def do_run(job):
    stage, p, seed, mode_b = job
    run = load_run(stage, p, seed)
    meas = score(run["events"], run["t0"], p)

    lags = logged_lags(run)
    simA = _sim_run(run, "A", lags=lags)
    a = score(simA, run["t0"], p)

    # A1 -- poll outcome, per (node, neighbor), index by index
    a1_ok = a1_tot = 0
    for key in set(meas["labels"]) | set(a["labels"]):
        lg = meas["labels"].get(key, [])
        sm = a["labels"].get(key, [])
        a1_tot += max(len(lg), len(sm))
        a1_ok += sum(1 for x, y in zip(lg, sm) if x == y)

    # A2 -- record_stored time per (origin, seq, node)
    ls, ss = meas["stored"], a["stored"]
    both = set(ls) & set(ss)
    a2_within = sum(1 for k in both if abs(ss[k] - ls[k]) <= A2_TOL_MS)
    a2_only = len(set(ls) ^ set(ss))
    a2_pairs = len(set(ls) | set(ss))

    # layer diagnostic: does the REPLY CONTENT agree?  state_rx hi maps compared
    # index by index per (node, neighbor), independently of the store times.
    rx_ok = rx_tot = 0
    for n in NODES:
        lg, sm = defaultdict(list), defaultdict(list)
        for e in run["events"][n]:
            if e.get("event") == "state_rx":
                lg[e["neighbor"]].append(e["hi"])
        for e in simA[n]:
            if e.get("event") == "state_rx":
                sm[e["neighbor"]].append(e["hi"])
        for nb in set(lg) | set(sm):
            a_, b_ = lg.get(nb, []), sm.get(nb, [])
            rx_tot += max(len(a_), len(b_))
            rx_ok += sum(1 for x, y in zip(a_, b_) if x == y)

    out = dict(stage=stage, p=p, seed=seed, rx=(rx_ok, rx_tot),
               m=(meas["n4"], meas["c4"], meas["lat4"], meas["polls"],
                  meas["answered"], meas["timed_out"]),
               m_rx=meas["state_rx"],
               a=(a["n4"], a["c4"], a["lat4"], a["polls"], a["answered"], a["timed_out"]),
               a1=(a1_ok, a1_tot), a2=(a2_within, len(both), a2_only, a2_pairs), b=[])
    if mode_b:
        for rep in range(R_REPLICATES):
            rng = random.Random(rep_seed(stage, p, seed, rep))
            sb = _sim_run(run, "B", sim_rng=rng)
            b = score(sb, run["t0"], p)
            out["b"].append((b["n4"], b["c4"], b["lat4"], b["polls"],
                             b["answered"], b["timed_out"]))
    return out


# ---------------------------------------------------------------------------
# Cell aggregation and the pass line
# ---------------------------------------------------------------------------

METRICS = ("delay", "cens4", "polls_sent", "timed_out", "bytes")


def pooled(rows, p):
    """rows: list of (n4, c4, lat4, polls, answered, timed_out) -> cell values."""
    n4 = sum(r[0] for r in rows)
    c4 = sum(r[1] for r in rows)
    lat = sum(r[2] for r in rows)
    polls = sum(r[3] for r in rows)
    ans = sum(r[4] for r in rows)
    to = sum(r[5] for r in rows)
    k = len(rows)
    return dict(
        delay=lat / n4 if n4 else float("nan"),
        cens4=c4 / (n4 + c4) if (n4 + c4) else float("nan"),
        polls_sent=polls / k,
        timed_out=to / k,
        bytes=(ans * (INT_B + REP_B[p]) + to * INT_B) / k if p in REP_B else None,
    )


def bootstrap_ci(rows, p):
    arr = np.array([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows], dtype=float)
    k = len(rows)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, k, size=(BOOT_N, k))
    s = arr[idx]                                   # (BOOT_N, k, 6)
    n4 = s[:, :, 0].sum(1)
    c4 = s[:, :, 1].sum(1)
    lat = s[:, :, 2].sum(1)
    polls = s[:, :, 3].sum(1)
    ans = s[:, :, 4].sum(1)
    to = s[:, :, 5].sum(1)
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        out["delay"] = lat / n4
        out["cens4"] = c4 / (n4 + c4)
    out["polls_sent"] = polls / k
    out["timed_out"] = to / k
    out["bytes"] = (ans * (INT_B + REP_B[p]) + to * INT_B) / k if p in REP_B else None
    ci = {}
    for m, v in out.items():
        ci[m] = (None, None) if v is None else (
            float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
    return ci


# ---------------------------------------------------------------------------
# Cross-check of the MEASURED side against the published evidence.
# spec.md: "any difference is fatal".
# ---------------------------------------------------------------------------


def cross_check(rows) -> list:
    """Compare the measured cell values this script recomputed from events.jsonl
    against evidence/09|10/live_main.csv (delay, cens4) and evidence/11/cells.csv
    (polls_sent), for every cell where the evidence holds the same quantity."""
    pub, polls, perrun = {}, {}, defaultdict(lambda: [0.0, 0.0])
    try:
        for f in ("09", "10"):
            for r in csv.DictReader(open(f"{EV}/{f}/live_main.csv")):
                pub[(r["stage"], float(r["p"]))] = r
            for r in csv.DictReader(open(f"{EV}/{f}/perrun.csv")):
                a = perrun[(r["stage"], float(r["p"]))]
                a[0] += sum(float(r[f"ngap{h}"]) for h in (1, 2, 3, 4))
                a[1] += sum(float(r[f"ngap{h}"]) - float(r[f"nnorep{h}"]) for h in (1, 2, 3, 4))
        for f in ("11", "12"):
            path = f"{EV}/{f}/cells.csv"
            if os.path.exists(path):
                for r in csv.DictReader(open(path)):
                    polls.setdefault((r["stage"], float(r["p"])), r)
    except OSError as exc:
        return [("evidence unreadable", str(exc))]

    out = []
    for r in rows:
        k = (r["stage"], r["p"])
        w = pub.get(k)
        if w is not None:
            if abs(r["meas_delay"] - float(w["delay"])) > 1e-6:
                out.append((f"{k[0]} p{k[1]} delay",
                            f"mine {r['meas_delay']!r} vs live_main {w['delay']}"))
            if abs(r["meas_cens4"] - float(w["cens4"])) > 1e-9:
                out.append((f"{k[0]} p{k[1]} cens4",
                            f"mine {r['meas_cens4']!r} vs live_main {w['cens4']}"))
        c = polls.get(k)
        if c is not None:
            mine = r["meas_polls_sent"] * r["n_runs"]
            if abs(mine - float(c["polls_sent"])) > 1e-6:
                out.append((f"{k[0]} p{k[1]} polls_sent",
                            f"mine {mine!r} vs cells.csv {c['polls_sent']}"))
            if abs(r["meas_state_rx_total"] - float(c["state_rx"])) > 1e-6:
                out.append((f"{k[0]} p{k[1]} state_rx",
                            f"mine {r['meas_state_rx_total']} vs cells.csv {c['state_rx']}"))
        a = perrun.get(k)
        if a is not None:
            if abs(r["meas_polls_sent"] * r["n_runs"] - a[0]) > 1e-6:
                out.append((f"{k[0]} p{k[1]} ngap",
                            f"mine {r['meas_polls_sent'] * r['n_runs']!r} vs perrun {a[0]!r}"))
            if abs(r["meas_answered_total"] - a[1]) > 1e-6:
                out.append((f"{k[0]} p{k[1]} answered",
                            f"mine {r['meas_answered_total']} vs perrun ngap-nnorep {a[1]!r}"))
    return out


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


def do_trace(stage, p, seed, mode):
    run = load_run(stage, p, seed)
    if mode.upper() == "A":
        ev = _sim_run(run, "A", lags=logged_lags(run))
    else:
        ev = _sim_run(run, "B", sim_rng=random.Random(rep_seed(stage, p, seed, 0)))
    t0 = run["t0"]
    rows = []
    for n, lst in ev.items():
        for e in lst:
            rows.append((e["ts_ms"], n, e))
    rows.sort(key=lambda r: (r[0], r[1]))
    print(f"# trace {stage} p{p} seed{seed} mode {mode.upper()}  t0_ms={int(t0)}  "
          f"group={run['group']}")
    print(f"# {'rel_ms':>9} {'node':<4} event / key fields")
    for ts, n, e in rows:
        kind = e["event"] if e["event"] != "decision" else f"decision:{e['kind']}"
        if e["event"] == "decision" and e["kind"] == "state_poll":
            a = e["action"]
            body = (f"neighbor={a['neighbor']} interval_s={a['interval_s']} "
                    f"applied_interval_s={a['applied_interval_s']:.9f} "
                    f"consec={e['state']['consecutive_poll_timeouts']}")
        elif e["event"] == "decision" and e["kind"] == "state_poll_reschedule":
            a = e["action"]
            body = (f"neighbor={a['neighbor']} trigger={a['trigger']} "
                    f"poll_epoch_ms={a['poll_epoch_ms']} interval_s={a['interval_s']} "
                    f"applied_interval_s={a['applied_interval_s']:.9f} "
                    f"clamped={a['clamped']}")
        elif e["event"] == "decision" and e["kind"] == "fetch_plan":
            a = e["action"]
            body = f"neighbor={a['neighbor']} fetched={a['fetched']}"
        elif e["event"] == "decision":
            body = json.dumps(e["action"], sort_keys=True)
        elif e["event"] == "state_rx":
            body = f"neighbor={e['neighbor']} hi={json.dumps(e['hi'], sort_keys=True)}"
        elif e["event"] in ("record_stored",):
            body = f"origin={e['origin']} seq={e['seq']} holder={e['holder']}"
        elif e["event"] == "record_created":
            body = f"origin={e['origin']} seq={e['seq']}"
        elif e["event"] == "fetch_timeout":
            body = f"origin={e['origin']} seq={e['seq']} neighbor={e['neighbor']}"
        else:
            body = json.dumps(e, sort_keys=True)
        print(f"{ts - int(t0):>11} {n:<4} {kind:<28} {body}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def fmt(x, nd=4):
    if x is None:
        return ""
    if isinstance(x, float) and x != x:
        return "nan"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--trace":
        do_trace(argv[1], float(argv[2]), int(argv[3]), argv[4] if len(argv) > 4 else "A")
        return
    mode_b = "--mode-a-only" not in argv
    cells = CELLS
    if "--quick" in argv:
        cells = [CELLS[0], CELLS[6], CELLS[15], CELLS[18], CELLS[36]]

    jobs = [(s, p, seed, mode_b) for (s, p) in cells for seed in SEEDS]
    t_start = time.time()
    import multiprocessing as mp
    with mp.Pool(WORKERS) as pool:
        results = pool.map(do_run, jobs, chunksize=1)
    wall = time.time() - t_start

    by_cell = defaultdict(list)
    for r in results:
        by_cell[(r["stage"], r["p"])].append(r)

    rows = []
    for (stage, p) in cells:
        rs = sorted(by_cell[(stage, p)], key=lambda r: r["seed"])
        meas = pooled([r["m"] for r in rs], p)
        meas_rx_total = sum(r["m_rx"] for r in rs)
        meas_ans_total = sum(r["m"][4] for r in rs)
        a = pooled([r["a"] for r in rs], p)
        ci = bootstrap_ci([r["m"] for r in rs], p)

        a1_ok = sum(r["a1"][0] for r in rs)
        a1_tot = sum(r["a1"][1] for r in rs)
        rx_ok = sum(r["rx"][0] for r in rs)
        rx_tot = sum(r["rx"][1] for r in rs)
        a2_w = sum(r["a2"][0] for r in rs)
        a2_b = sum(r["a2"][1] for r in rs)
        a2_o = sum(r["a2"][2] for r in rs)
        a2_p = sum(r["a2"][3] for r in rs)
        a1_rate = a1_ok / a1_tot if a1_tot else 0.0
        a2_rate = a2_w / a2_b if a2_b else 0.0
        a2_orate = a2_o / a2_p if a2_p else 0.0

        polls_rel = abs(a["polls_sent"] - meas["polls_sent"]) / meas["polls_sent"]
        delay_abs = abs(a["delay"] - meas["delay"])
        cens_pp = abs(a["cens4"] - meas["cens4"])
        a3 = (polls_rel <= A3_POLLS_REL
              and delay_abs <= max(A3_DELAY_REL * meas["delay"], A3_DELAY_ABS)
              and cens_pp <= A3_CENS_PP)

        row = dict(stage=stage, p=p, n_runs=len(rs),
                   meas_state_rx_total=meas_rx_total, meas_answered_total=meas_ans_total)
        for m in METRICS:
            row[f"meas_{m}"] = meas[m]
            row[f"A_{m}"] = a[m]
            row[f"ci_lo_{m}"], row[f"ci_hi_{m}"] = ci[m]
        row.update(A_rx_hi_match=rx_ok, A_rx_hi_total=rx_tot,
                   A_rx_hi_rate=rx_ok / rx_tot if rx_tot else 0.0,
                   A1_match=a1_ok, A1_total=a1_tot, A1_rate=a1_rate,
                   A1_pass=int(a1_rate >= A1_MIN),
                   A2_within=a2_w, A2_both=a2_b, A2_rate=a2_rate,
                   A2_only_one=a2_o, A2_pairs=a2_p, A2_only_one_rate=a2_orate,
                   A2_pass=int(a2_rate >= A2_MIN and a2_orate <= A2_ONLYONE_MAX),
                   A3_polls_rel=polls_rel, A3_delay_abs=delay_abs, A3_cens_pp=cens_pp,
                   A3_pass=int(a3))

        if mode_b:
            reps = [pooled([r["b"][k] for r in rs], p) for k in range(R_REPLICATES)]
            for m in METRICS:
                vals = [rp[m] for rp in reps]
                bv = None if vals[0] is None else float(np.mean(vals))
                row[f"B_{m}"] = bv
                lo, hi = ci[m]
                row[f"B1_{m}"] = "" if bv is None else int(lo <= bv <= hi)
                if bv is None or meas[m] is None:
                    row[f"B2_{m}"] = None
                elif m == "cens4":
                    row[f"B2_{m}"] = bv - meas[m]
                else:
                    row[f"B2_{m}"] = (bv - meas[m]) / meas[m] if meas[m] else float("nan")
        rows.append(row)

    cols = list(rows[0].keys())
    csv_path = os.path.join(HERE, "validate_cells.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (fmt(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    xcheck = cross_check(rows)
    write_md(rows, mode_b, wall, xcheck)
    print("cross-check against evidence: " +
          ("OK (no difference)" if not xcheck else f"{len(xcheck)} DIFFERENCES"))
    for a, b in xcheck[:10]:
        print("   ", a, b)
    print(f"wrote {csv_path}")
    print(f"wall clock {wall:.1f} s ({wall/60:.1f} min), {len(jobs)} runs, {WORKERS} workers")


def write_md(rows, mode_b, wall, xcheck=()):
    L = []
    W = L.append
    nb = [r for r in rows if r["meas_bytes"] is not None]
    a1p = sum(r["A1_pass"] for r in rows)
    a2p = sum(r["A2_pass"] for r in rows)
    a3p = sum(r["A3_pass"] for r in rows)

    W("# ev13 validate: replay of the current rule, scored against spec.md's pass line\n")
    W(f"Generated by `replay.py` ({time.strftime('%Y-%m-%d %H:%M:%S')}), "
      f"840 runs, {WORKERS} workers, wall clock {wall:.0f} s.\n")
    b_fail = []
    if mode_b:
        for m in METRICS:
            pool = nb if m == "bytes" else rows
            need = 38 if len(pool) == 42 else 27
            if sum(int(r[f"B1_{m}"] or 0) for r in pool) < need:
                b_fail.append(f"B1 {m}")
    a_fail = [n for n, v in (("A1", a1p), ("A2", a2p), ("A3", a3p)) if v != 42]
    overall = ("PASS" if not a_fail and not b_fail else
               ("FAIL" if len(a_fail) == 3 else "PARTIAL"))
    W("## Verdict\n")
    W(f"**{overall}.** Mode A fails {', '.join(a_fail) if a_fail else 'nothing'}; "
      f"Mode B {'passes every line' if not b_fail else 'fails ' + ', '.join(b_fail)}. "
      "spec.md allows PASS only when every line holds. Which layer broke, what is irreducible "
      "at the logs' millisecond resolution, and what is a real effect left unmodelled, are "
      "all below.\n")
    W("| line | metric | result | pass line | verdict |")
    W("|---|---|---|---|---|")
    W(f"| A1 | poll outcome | {a1p}/42 cells >= 99.5 % | 42/42 | "
      f"{'PASS' if a1p == 42 else 'FAIL'} |")
    W(f"| A2 | record_stored time | {a2p}/42 cells | 42/42 | "
      f"{'PASS' if a2p == 42 else 'FAIL'} |")
    W(f"| A3 | cell aggregates | {a3p}/42 cells | 42/42 | "
      f"{'PASS' if a3p == 42 else 'FAIL'} |")
    if mode_b:
        for m in METRICS:
            pool = nb if m == "bytes" else rows
            n = len(pool)
            ok = sum(int(r[f"B1_{m}"] or 0) for r in pool)
            need = 38 if n == 42 else 27
            note = (" (passes only after scorer fix 4 below; 37/42 before it)"
                    if m == "timed_out" else "")
            W(f"| B1 | {m} | {ok}/{n} cells in the 95 % CI{note} | >= {need}/{n} | "
              f"{'PASS' if ok >= need else 'FAIL'} |")
        for m in METRICS:
            pool = nb if m == "bytes" else rows
            vals = sorted(r[f"B2_{m}"] for r in pool if r[f"B2_{m}"] is not None)
            if not vals:
                continue
            med = vals[len(vals) // 2] if len(vals) % 2 else (
                (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2)
            if m == "delay":
                lim, unit, ok = B2_DELAY_REL, "rel", abs(med) <= B2_DELAY_REL
            elif m in ("polls_sent", "bytes"):
                lim, unit, ok = B2_POLLS_REL, "rel", abs(med) <= B2_POLLS_REL
            elif m == "cens4":
                # 2026-11-17 review fix: a TRUE median of the absolute differences.
                # `vals[len(vals)//2]` is the upper middle value on an even count,
                # which for the 42 cells reported the 22nd smallest instead of the
                # mean of the 21st and 22nd.
                vals = sorted(abs(r[f"B2_{m}"]) for r in pool if r[f"B2_{m}"] is not None)
                med = (vals[len(vals) // 2] if len(vals) % 2 else
                       (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2)
                lim, unit, ok = B2_CENS_PP, "abs pp", med <= B2_CENS_PP
            else:
                lim, unit, ok = None, "rel", None
            W(f"| B2 | {m} | median {med:+.5f} ({unit}) | "
              f"{('<= %g' % lim) if lim is not None else 'not gated by spec'} | "
              f"{'PASS' if ok else ('FAIL' if ok is False else 'info')} |")
    W("")

    W("## Measured side, cross-checked against the published evidence\n")
    if not xcheck:
        W("`delay` and `cens4` against evidence/09|10 `live_main.csv`, `polls_sent` against "
          "evidence/11|12 `cells.csv`, for all 42 cells: **no difference**.\n")
    else:
        W("**DIFFERENCES FOUND** (spec.md calls any difference fatal):\n")
        for a, b in xcheck:
            W(f"- {a}: {b}")
        W("")

    W("## Mode A, per cell\n")
    W("| stage | p | A1 | reply hi | A2 <=150 ms | A2 only-one | polls rel | delay |delta| ms | cens4 pp | A3 |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        W(f"| {r['stage']} | {r['p']} | {r['A1_rate']*100:.3f} % | "
          f"{r['A_rx_hi_rate']*100:.3f} % | {r['A2_rate']*100:.3f} % | "
          f"{r['A2_only_one_rate']*100:.3f} % | {r['A3_polls_rel']*100:.3f} % | "
          f"{r['A3_delay_abs']:.1f} | {r['A3_cens_pp']*100:.3f} | "
          f"{'ok' if r['A3_pass'] else 'FAIL'} |")
    W("")

    if mode_b:
        W("## Mode B, per cell (sim = mean over 5 replicates; CI = 95 % seed bootstrap of measured)\n")
        W("| stage | p | delay meas / sim / CI | cens4 meas / sim | polls meas / sim | bytes meas / sim | B1 d,c,p,t,b |")
        W("|---|---|---|---|---|---|---|")
        for r in rows:
            flags = "".join(str(r[f"B1_{m}"]) if r[f"B1_{m}"] != "" else "-" for m in METRICS)
            bm = "" if r["meas_bytes"] is None else f"{r['meas_bytes']:.0f} / {r['B_bytes']:.0f}"
            W(f"| {r['stage']} | {r['p']} | {r['meas_delay']:.0f} / {r['B_delay']:.0f} / "
              f"[{r['ci_lo_delay']:.0f}, {r['ci_hi_delay']:.0f}] | "
              f"{r['meas_cens4']:.4f} / {r['B_cens4']:.4f} | "
              f"{r['meas_polls_sent']:.1f} / {r['B_polls_sent']:.1f} | {bm} | {flags} |")
        W("")

    rx_all = (sum(r["A_rx_hi_match"] for r in rows), sum(r["A_rx_hi_total"] for r in rows))
    # 2026-11-17 review fix: every failing cell below is listed from the data.  The
    # hardcoded lists were written when the split boundary made A1 fail in 2 cells and
    # A3 in none; reverting to spec.md's boundary changes both, and a hardcoded count
    # would then state the wrong thing with the right air of precision.
    a1_fail = [f"{r['stage']} p{r['p']} ({r['A1_rate']*100:.3f} %)"
               for r in rows if not r["A1_pass"]]
    a2_fail_pol = [f"{r['stage']} p{r['p']}" for r in sorted(rows, key=lambda r: r["A2_rate"])
                   if not r["A2_pass"] and not r["A1_pass"]]
    a3_fail = [r for r in rows if not r["A3_pass"]]

    def a3_why(r):
        """Which of A3's three tests this cell fails, with the numbers."""
        tol = max(A3_DELAY_REL * r["meas_delay"], A3_DELAY_ABS)
        parts = []
        if r["A3_polls_rel"] > A3_POLLS_REL:
            parts.append(f"polls_sent off by {r['A3_polls_rel']*100:.3f} % "
                         f"(line {A3_POLLS_REL*100:g} %)")
        if r["A3_delay_abs"] > tol:
            parts.append(f"delay off by {r['A3_delay_abs']:.0f} ms against a "
                         f"max(1 %, 50 ms) = {tol:.0f} ms tolerance")
        if r["A3_cens_pp"] > A3_CENS_PP:
            parts.append(f"cens4 off by {r['A3_cens_pp']*100:.3f} pp "
                         f"(line {A3_CENS_PP*100:g} pp)")
        return f"{r['stage']} p{r['p']}: " + "; ".join(parts)

    W("## Which layer broke\n")
    W("| layer | evidence | verdict |")
    W("|---|---|---|")
    # 2026-11-17 review fix: the timer evidence is facts-timers.md §8, not Mode A.
    # Mode A's own lag is DERIVED from the same equation §8 verifies (logged_lags
    # applies it line by line), so "Mode A reproduces the logged ts" is circular and
    # cannot be cited as independent support for the timer rule.
    W("| timer | facts-timers.md §8 predicts 786 553 logged `state_poll` stamps over 14 "
      "stages and 1480 runs from the previous decision lines, residual a non-negative "
      "lag in [0, 100] ms = `receive_timeout_ms`, 14/14 stages. Mode A is NOT evidence "
      "here: its per-poll lag is taken from the log through that same equation | sound, "
      "on §8 |")
    W(f"| outcome model | A1 {min(r['A1_rate'] for r in rows)*100:.2f} .. "
      f"{max(r['A1_rate'] for r in rows)*100:.2f} %, "
      f"{sum(r['A1_pass'] for r in rows)}/42 cells over 99.5 %; the failing cells are "
      f"{', '.join(a1_fail) if a1_fail else 'none'} | "
      f"**breaks A1 in {len(a1_fail)} cell{'s' if len(a1_fail) != 1 else ''}** |")
    W(f"| cell aggregates | A3 {a3p}/42. "
      + (" ".join(a3_why(r) for r in a3_fail) if a3_fail else "No cell fails.")
      + " | "
      + (f"**breaks A3 in {len(a3_fail)} cell{'s' if len(a3_fail) != 1 else ''}**, and it "
         f"is the price of spec.md line 15: on the split boundary A3 was 42/42 (see "
         f"*Tuned variants*)" if a3_fail else "sound") + " |")
    W(f"| reply content | `state_rx` hi maps agree on "
      f"{rx_all[0]}/{rx_all[1]} = {100*rx_all[0]/rx_all[1]:.3f} % of replies | "
      f"**the binding limit on A2** |")
    # 2026-11-17 review fix: "well under 1 %" was wrong for fetch_timeout.  Measured
    # by review-rules/ft.py over 126 runs (3 seeds x 42 cells, Mode A).
    W("| fetch model | `fetch_plan` count 76 221 logged vs 76 293 simulated over 126 runs "
      "(+0.09 %), but `fetch_timeout` 132 logged vs 116 simulated (**-12.1 %**): the replay "
      "misses about one fetch timeout in eight. At 132 timeouts against 76 221 plans this "
      "cannot move a cell aggregate, and no cell's A3 or B1 turns on it, but the timeout "
      "branch itself is not validated to better than ~12 % | sound on the plan path, "
      "**12 % low on the timeout path** |")
    W("")
    W("The A1 and A2 failures are dominated by millisecond-resolution coincidences rather "
      "than rule errors -- but see the unmodelled effects below for what that claim does "
      "NOT cover. "
      + (f"The A3 failure is not of that kind: {', '.join(r['stage'] + ' p' + str(r['p']) for r in a3_fail)} "
         f"misses a cell aggregate, which is the layer Q61 actually reads, and it misses it "
         f"on the boundary spec.md fixes. What saves the Mode B answer is that Mode B is "
         f"scored against the measured cell's own bootstrap interval and passes every line "
         f"here, including in that cell."
         if a3_fail else "") + "\n")
    W(f"- **A1.** The {len(a1_fail)} failing cells ({', '.join(a1_fail) or 'none'}) each "
      "contain one or two runs where a "
      "poll was sent inside the few-ms window in which a link is flipping. The events.jsonl "
      "stamps are whole milliseconds, so which side of the flip that poll fell on is not "
      "recoverable (facts-reply-fetch.md §4 and its own UNCONFIRMED 3 say the same). In a "
      "**policy** stage one such coin flip is not local: the outcome decides the next interval, "
      "and the extra or missing poll shifts that node's shared jitter stream by one draw for the "
      "rest of the run, so every later poll of that node diverges. The same coin flip costs "
      "nothing in the non-policy arms, which is why only short/wait cells fail A1.")
    W("- **A2.** A record created or stored at the holder in the same millisecond as a reply is "
      "served is on one side of the serve instant or the other, and the log cannot say which. "
      "Each such flip moves a 4-record batch by one whole poll interval at the node that missed "
      "it, and the shift then propagates down the line. The rate therefore tracks the poll "
      # 2026-11-17 review fix: these two ranges are read off this run's own cells.  They
      # had been copied from an earlier run, so they described a boundary the code no
      # longer uses.
      + (lambda lo, hi: f"interval: the 0.5 s arm (s5b0p5 / s6b0p5) sits at {lo} while the "
                        f"10.043 s arm sits at {hi}. ")(
          "%.1f-%.1f %%" % (min(r["A2_rate"] for r in rows if "b0p5" in r["stage"]) * 100,
                            max(r["A2_rate"] for r in rows if "b0p5" in r["stage"]) * 100),
          "%.1f-%.1f %%" % (min(r["A2_rate"] for r in rows if "b10p043" in r["stage"]) * 100,
                            max(r["A2_rate"] for r in rows if "b10p043" in r["stage"]) * 100))
      + "The shorter the interval the more replies there are to collide "
      "with, and one missed batch is already a 500 ms shift, well over the 150 ms tolerance. "
      + (f"The cells that fail both A1 and A2 ({', '.join(a2_fail_pol)}) fail A2 for the A1 "
         "reason above, not this one. " if a2_fail_pol else "")
      + "The fetch layer itself is not implicated: wherever the "
      "advertisement agrees, the store time agrees to a few ms.")
    W("")

    W("### Effects seen in the logs and deliberately NOT modelled\n")
    W("- **Slow replies.** The model gives every poll a 1 ms RTT (spec.md). In the logs 65 of "
      "315 315 in-run replies (0.021 %) arrive more than 100 ms after their poll: p50 422 ms, "
      "max 1040 ms. Worked example: s5wait p0.4 seed 2, h2 polls h1 at t0+65 208 ms with the "
      "link up since t0+65 000 ms, and the reply lands at t0+66 020 ms -- 812 ms. Only 12 of "
      "the 58 such polls that follow a `link_up` are within 1 s of it, so this is not purely "
      "interface recovery; plain agent-loop stalls account for the rest. It is left out "
      "because fitting it would need a new constant, which spec.md forbids. At 0.021 % of "
      "replies it cannot be the main A2 driver, but it is not a millisecond tie either.")
    W("- **Unexplained asymmetry.** Over 100 runs of 5 cells, where the simulated and the "
      "logged `state_rx` hi maps disagree at the same instant, the simulation is BEHIND the "
      "log 664 times and AHEAD 344 times. A pure rounding tie should be symmetric, and "
      "pushing the serve instant one millisecond later (the delta sweep's next candidate) "
      "made the total worse, not better. The cause is not established.")
    W("- **`extra` / `truncated`.** Implemented, never exercised: 0 of 137 659 logged "
      "advertisements carried a non-empty `extra` (facts-reply-fetch.md §1), so this replay "
      "cannot be said to have tested that path.")
    W("- **`clamped` reschedules and `state_poll_late_reply`.** Both implemented, both zero in "
      "every logged run and in every simulated run (facts-timers.md §9).")
    W("")

    W("## Model fixes made after the first results\n")
    W("Each was a rule implemented differently from the source, found with `--trace` against one "
      "failing run's events.jsonl, and each is a behaviour change, not a constant.\n")
    W("| # | what was wrong | what the source says | effect |")
    W("|---|---|---|---|")
    W("| 1 | The reply carried the holder's store at the reply's RECEIVE instant. | "
      "`handle_interest` runs synchronously inside the holder's `poll()` "
      "(cefpyco_transport.py:397-407), so the body is the store at the SERVE instant, between "
      "the requester's send and its receive (facts-reply-fetch.md §1). Implemented as: holder "
      "events stamped at the send millisecond are in, events stamped at the receive "
      "millisecond are out. | A2 on the 12-run diagnostic set 98.0 % -> 98.5 % (98.0 % is the "
      "original receive-instant snapshot; an intermediate over-correction that excluded the "
      "send millisecond too scored 96.8 %). Whole 4-record batches stopped arriving one poll "
      "interval early. |")
    W("| 2 | The link boundary comparisons were non-strict, so a packet sent in exactly the "
      "boundary millisecond was judged against the NEW link state. | `t0 + actual_at*1000` is "
      "when the interface change starts and the true flip is a few ms later "
      "(facts-reply-fetch.md §4: the 5 residual misfits are at +0..+6 ms, never before). | "
      "A1 37/42 -> 39/42 cells. |")
    W("| 3 | *(withdrawn 2026-11-17.)* Both link_down and link_up used "
      "`t0 + actual_at*1000`; this row changed link_up to the ISO `ts`. | The reasoning was "
      "§4's, but §4's own conclusion -- the mapping spec.md line 15 points at -- is "
      "`t0 + actual_at*1000` for both directions. Changing one direction after seeing A1 is "
      "tuning, so the code is back on the spec's boundary and the variant is reported under "
      "*Tuned variants* below. | see *Tuned variants* |")
    W("| 4 | The simulated event list was sorted with `state_rx` ahead of any `state_poll` "
      "sharing its millisecond. | The engine emits in real order: `poll()` dispatches the Data "
      "and emits `state_rx`, then `on_tick` emits the new `state_poll` (agent.py:1071-1072). A "
      "stable sort on `ts_ms` alone keeps the handlers' own float-time order. | The scorer had "
      "been charging such a reply to the previous poll, counting an answered poll as timed out; "
      "B1 timed_out 37/42 -> **42/42** cells. |")
    W("| 5 | The stop test was `elapsed > stop_at_s`. | `Agent._on_receive_boundary` returns "
      "true at `elapsed >= stop_at_s` (agent.py:853), before `on_tick`. | Cosmetic here; no "
      "cell moved. |")
    W("| 6 | A policy stage armed no timer at send, only at the resolution of each poll. | "
      "`engine.py:478` sets `_next_state_poll[neighbor] = now + applied_interval` on EVERY "
      "send, outside the policy/choices branch; its comment calls that value provisional "
      "under a policy, and `_redecide_next_poll` (engine.py:507-522) overrides it. | No cell "
      "moved: every row of the four policy tables exceeds the 2.0 s lifetime, so the "
      "resolution always lands first and the provisional timer is always stale. "
      "`validate_cells.csv` is byte-identical before and after, and so are the full Mode A "
      "and Mode B event streams of six policy runs (`item2_proof.py`). It matters only for "
      "a rule with an interval below 2 s, which is what spec2 then needs. |")
    W("")
    W("## Tuned variants: tried after seeing results, NOT the result\n")
    W("spec.md forbids moving a number across a line by changing something after seeing it. "
      "Two model rules were nevertheless chosen that way. Both are recorded here with their "
      "before and after, and the headline numbers above are the UNTUNED ones.\n")
    W("| variant | what it changes | A1 | A2 | A3 |")
    W("|---|---|---|---|---|")
    W(f"| spec.md line 15 (the default, reported above) | `t0 + actual_at*1000` for link_down "
      f"and link_up alike | {a1p}/42 | {a2p}/42 | {a3p}/42 |")
    W(f"| split boundary (`LINK_UP_FROM_ISO_TS=1`) | link_down keeps `actual_at`, link_up "
      f"takes the `results.json` ISO `ts` | {SPLIT_A1}/42 | {SPLIT_A2}/42 | {SPLIT_A3}/42 |")
    W("")
    W("The split boundary scores better on all three lines, and §4's start/completion reading "
      "is a real argument for it. It is still tuned: it was found by scoring A1 after the "
      "first full run, and §4's own best mapping -- the one spec.md line 15 names -- is "
      "`actual_at` for both. Reproduce either row with "
      "`LINK_UP_FROM_ISO_TS=1 python3 replay.py` / `python3 replay.py`.\n")
    W("The serve-instant tie rule (model fix 1) is the second tuned choice: the three "
      "candidate offsets were scored against the logs on a 12-run set and the best was kept "
      "(A2 96.8 / 98.5 / 98.0 % for serve-before-send-ms / between / after-receive-ms). What "
      "the model kept -- holder events stamped at the send millisecond are IN -- is also "
      "**not** what the source says for `record_created`: `Agent.step` runs "
      "`transport.poll()`, which serves every Interest, and only then reads "
      "`now = transport.now()` and calls `_drain_schedule(now)` "
      "(agent.py:1071-1077), which stamps each new record `round(now * 1000)` "
      "(agent.py:861-872). A `record_created` stamped at millisecond X was therefore created "
      "from a clock read AFTER the serve of that iteration, so it cannot be in a reply whose "
      "Interest was sent at X: for `record_created` the source decides the tie the other way. "
      "`record_stored` is written from inside `poll()` itself and stays genuinely ambiguous. "
      "`review-rules/alt_rule.py` implements the source-faithful split; it was not adopted, "
      "because adopting it after seeing its score would be the same tuning again.\n")
    W("No constant was changed after seeing a result. The constants at the top of `replay.py` "
      "are the ones spec.md fixes, plus `RETRY_LAG_MS = 2.0`, which facts-reply-fetch.md §6 "
      "measures directly (a backed-off retry is re-planned +1..2 ms after `next_attempt_at`, "
      "not +[0,100] ms, because the wait is a whole number of tick periods).\n")

    W("## Deviations from spec.md\n")
    W("1. **`agent_start.ts_ms` is treated as exogenous**, in both modes, alongside `t0_ms`. "
      "spec.md's exogenous list does not name it. It is process-launch timing, not a timer "
      "rule: facts-timers.md §3 shows the first poll lands 8-103 ms after `agent_start` (one "
      "`receive_timeout_ms`) but 434-2187 ms after `t0_ms`, dominated by process launch. "
      "Without it the first poll of every node would be unplaceable to better than ~1.7 s. "
      "Mode A then takes the first poll's lag from the log like any other; Mode B draws it "
      "U[0, 100] ms, one draw per node shared by both neighbors (facts-timers.md §2: both "
      "neighbors share the first tick).")
    W("2. **The serve instant is pinned at millisecond granularity** rather than left as a "
      "fraction of POLL_RTT (fix 1 above). spec.md fixes POLL_RTT at 1 ms but says nothing "
      "about which side of a same-millisecond holder event the serve falls on. The rule was "
      "**chosen by score on a 12-run set** (A2 96.8 / 98.5 / 98.0 % for serve-before-send-ms "
      "/ between / after-receive-ms), so it is **tuned**, and for `record_created` it is also "
      "contrary to the source. See *Tuned variants* above for both points.")
    W("3. **`RETRY_LAG_MS = 2.0` is an extra constant** not in spec.md's list, forced by "
      "facts-reply-fetch.md §6 (see above).")
    W("4. **Fetch planning also runs on every record_stored and every fetch_timeout**, not only "
      "on `state_rx`. spec.md names only the `state_rx -> fetch_plan` 0 ms edge. "
      "`transport.poll()` returns on any packet and `on_tick` then runs `_send_fetch_plan` for "
      "every neighbor, which is why a length-8 plan is followed by the next plan on the same "
      "pair after 1-3 ms in the logs (measured over 4 runs, p50 2 ms). Without it a backlog "
      "larger than the in-flight window of 8 would drain one window per 100 ms.")
    W("5. **A1's denominator is every poll of the run**, not only in-window polls, and unpaired "
      "polls (one side has more) count as mismatches. spec.md does not say. A1's outcome label "
      "comes from the same reply-attribution rule the metrics use.")
    W("6. **B2 for `polls timed out` is reported but not gated.** spec.md's B2 line names delay, "
      "polls_sent, bytes and cens4 only.")
    W("7. **~~`link_up` uses the `results.json` ISO `ts`~~ -- withdrawn 2026-11-17.** Both "
      "link directions are back on `t0 + actual_at*1000`, which is what spec.md line 15 "
      "instructs. No deviation from the spec remains on the link boundary; the ISO-`ts` "
      "variant is reported under *Tuned variants*.")
    W("8. **A policy stage's provisional send-time timer is queued at the bare due instant "
      "and its tick lag is drawn only if the timer is still live** (`ev_poll_due`), while the "
      "non-policy path draws the lag at send time. spec.md says nothing about RNG stream "
      "order. Drawing at send in the policy path too would consume one extra Mode B `sim_rng` "
      "value per poll and shift every later draw of the run, which would move the 14 existing "
      "policy cells although their schedule is unchanged. The deferred draw is what makes "
      "`validate_cells.csv` byte-identical across model fix 6.")
    W("9. **A fact file was corrected while scoring this replay**, and the measured side here "
      "follows the corrected version: facts-metrics.md said `ngap - nnorep` differs from the "
      "`state_rx` count in \"5 of the 42 cells\". It differs in **7**. The count had been read "
      "off `evidence/11/cells.csv`'s `perrun_answered_minus_state_rx`, which is blank for the "
      "12 s6 cells at p 0.5 / 0.6; recomputed over all 42 cells the extra two are s6b0p5 p0.5 "
      "(-1) and s6b0p5 p0.6 (-4). This replay's own cross-check compares both quantities for "
      "every cell and reports no difference, so nothing in the numbers above moves.")
    W("")
    W("### Jitter: regenerated, not read back (spec.md line 24 asks which and why)\n")
    W("Both modes re-seed the stdlib streams -- "
      "`random.Random(f\"state_poll_interval_jitter:{seed}:{node}\")` and, for the choices "
      "arms, `random.Random(f\"state_poll_interval:{seed}:{node}\")` -- one draw per "
      "`state_poll`, consumed in send order across both neighbors of a node, and none per "
      "reschedule (a reschedule reuses the poll's own `jitter_u`). `applied_interval_s` is not "
      "read back, for three reasons: s5wait/s6wait row 1 has `jitter_frac = 0`, so the draw is "
      "unrecoverable from it although it still advances the stream (facts-timers.md §4b); "
      "Mode B and any changed rule produce polls the log has no row for; and regenerating is "
      "what makes Mode A a test of the rule rather than a replay of its own output. "
      "facts-timers.md §4b confirms the streams reproduce the logged values exactly.\n")
    W("### Reproducibility\n")
    W("- Mode B replicate RNG: `random.Random(int.from_bytes(sha256("
      "f\"{stage}:{p}:{seed}:{rep}\").digest()[:8]))` -- not Python's salted `hash()`, so the "
      f"{WORKERS} worker processes draw the same streams on every run.")
    W(f"- Bootstrap: `numpy.random.default_rng({BOOT_SEED})` instantiated once per cell, so "
      f"every metric of a cell is resampled on the same {BOOT_N} x n_seeds index matrix.")
    W(f"- {WORKERS} worker processes on a 6-core host (deliberately oversubscribed, as the "
      f"task specified); wall clock {wall:.0f} s for all 840 runs x (1 Mode A + "
      f"{R_REPLICATES} Mode B replicates).")
    W("")

    W("## Worst cells\n")
    for key, label in (("A1_rate", "A1 poll outcome"), ("A2_rate", "A2 store time")):
        worst = sorted(rows, key=lambda r: r[key])[:5]
        W(f"- {label}: " + ", ".join(f"{w['stage']} p{w['p']} {w[key]*100:.2f} %" for w in worst))
    worst = sorted(rows, key=lambda r: -(r["A3_delay_abs"] /
                                        max(A3_DELAY_REL * r["meas_delay"], A3_DELAY_ABS)))[:5]
    W("- A3 delay (error / the max(1 %, 50 ms) tolerance it is measured against): "
      + ", ".join(f"{w['stage']} p{w['p']} {w['A3_delay_abs']:.0f} / "
                  f"{max(A3_DELAY_REL * w['meas_delay'], A3_DELAY_ABS):.0f} ms"
                  for w in worst))
    W(f"  {'The worst of these is OVER its tolerance and fails A3' if a3_fail else 'The tightest of these passes with little room'}"
      f". It is reported as it came out; nothing was adjusted to move it.")
    W("")
    with open(os.path.join(HERE, "validate.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
