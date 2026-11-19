#!/usr/bin/env python3
"""ev18 spec: how much gain foresight can take over the fixed-interval frontier.

    PYTHONDONTWRITEBYTECODE=1 /home/lab_shared/cefore-semindex/.venv/bin/python3 -B \
        bound.py <check1|check2|check3|check5|check7|run>

WHY THIS EXISTS (spec.md)
  evidence/13 to 16 compared named rules against the seven-arm fixed-interval frontier
  and found gains of a few points with intervals that span zero.  Before a three-way
  comparison of decision makers (a distilled LLM, an ML model, formula rules) is
  commissioned, this step asks on the replay alone how much gain foresight can take at
  all (V), how loose the structural ceiling is (C), and how much of the gain a formula
  already takes (F).

WHAT IT ADDS TO THE SIMULATOR
  One subclass of ../15/onoff.py's Engine, which is itself a subclass of
  ../13/replay.py's _Engine.  It changes three things and nothing else:
  a deep copy of the rule dict per node, a `resolve` that can place a poll at an
  arbitrary instant and that carries F's two new coordinates, and, under V only, a tick
  lag drawn from one cached generator per (replicate seed, node, neighbour).
  With the V flag off, `i_timeout` keep and `k_cap` unbounded the override writes
  nothing at all, which is what check 1 and check 2 measure.

WHAT IT CANNOT SHOW
  No number here has a testbed measurement behind it.  V reads the future, so V is not
  reachable without foresight, and the distance between V and the clairvoyant optimum is
  not measured.  C is budget free and is expected to be loose.  See spec.md.
"""

import bisect
import csv
import math
import os
import random
import sys
import time
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV13 = os.path.normpath(os.path.join(HERE, "..", "13"))
EV14 = os.path.normpath(os.path.join(HERE, "..", "14"))
EV15 = os.path.normpath(os.path.join(HERE, "..", "15"))
# EV14 goes in front of EV15 and EV13 because all three hold an `after.py` and the one
# this step reuses is evidence/14's early-record scorer.  `onoff` resolves to EV15 and
# `replay`/`trickle` to EV13, each being the only directory that holds them.
sys.path.insert(0, EV13)
sys.path.insert(0, EV15)
sys.path.insert(0, EV14)
import after as A  # noqa: E402  evidence/14's early-record scorer
import onoff as O  # noqa: E402  evidence/15's engine, workloads and tables
import replay as R  # noqa: E402
import trickle as T  # noqa: E402
import workload as W  # noqa: E402  evidence/14's generators, gap counter and tables

assert O.__file__.startswith(EV15), O.__file__
assert W.__file__.startswith(EV14), W.__file__
assert A.__file__.startswith(EV14), A.__file__

SEEDS = O.SEEDS                     # 1..20, as ev13 to ev15
REPS = O.REPS                       # 5 Mode B replicates
WORKERS = 12                        # the task's ceiling
BOOT_N = O.BOOT_N
BOOT_SEED = O.BOOT_SEED
CELLS = O.CELLS
NO_OUTAGE = O.NO_OUTAGE
WORKLOADS = O.WORKLOADS
CUT_MS = O.CUT_MS
HEADLINE = ("s6b10p043", 0.4, "W0")  # the only cell with a testbed measurement behind it
VKNOWS_ARM = "fix10.043"             # the arm the "What V knows" illustration sits beside
W0_CHECKED = (0, 0)                  # filled by the run, printed in bound.md

RULE_SPECS = O.RULE_SPECS            # the seven arms, short, wait, T37, T37r, T37b
NAMES = O.NAMES
FIXED = O.FIXED
ARM_INTERVALS = [0.5, 0.8, 1.359, 2.2, 3.695, 6.0, 10.043]
ARM_OF = dict(zip(FIXED, ARM_INTERVALS))

# spec.md "Reading rule": the half widths the step reads its intervals against, and the
# replay's own 6 point fidelity margin, which outcome 1 carries and outcomes 2 and 3 do
# not.  Fixed in spec.md before any number of this step was computed.
H = 6.5                              # points, a single gain
HD = 7.8                             # points, a difference of two gains
REPLAY_MARGIN = 6.0                  # points, the replay's own systematic error

FOLDS = {"A": [s for s in SEEDS if s <= 10], "B": [s for s in SEEDS if s > 10]}
# The lambda that V is scored with on one fold is bisected on five seeds of the OTHER
# fold, so V's budget is cross fitted exactly as F's member is (spec.md "Budget").
BISECT_SEEDS = {"A": [11, 12, 13, 14, 15], "B": [1, 2, 3, 4, 5]}

# ---------------------------------------------------------------------------
# V's placement mechanics, all of them measured in the scratchpad before the spec
# ---------------------------------------------------------------------------
V_INIT_S = 10.043        # the frontier's longest arm: the first poll of every pair
V_BIG_M = 1e9            # so that min(ival*m, i_max) is always i_max == the target gap
V_NO_MORE_S = 1e6        # the gap written when the DP has no further instant
TRANSIT_MS = R.POLL_RTT_MS + R.FETCH_RTT_MS      # 3 ms, the engine's own per-hop constant
TIMEOUT_FLOOR_MS = 2100.0                        # 2 s timeout plus a tick lag of 100 ms
PRE_DOWN_GUARD_MS = 100.0                        # the tick lag can push a poll this far
REACH_FRAC = 0.70        # an arm is a reachable V target below this share of candidates
BUDGET_TOL = 0.02        # the bisection must land within 2 % of the target's poll count
BUDGET_TOL_WIDE = 0.10   # between 2 % and 10 % V_at_F is reported with the gap printed
BISECT_STEPS = 8
LAM_LO_MS, LAM_HI_MS = 2.0, 200000.0             # the bisection bracket, fixed here

# The chain of every origin, as a list of (receiver, holder) directed pairs in hop
# order.  The topology of every logged run is the line h0-h1-h2-h3-h4, and a node's
# store is not changed by the polls of the node downstream of it, so hop h's schedule
# moves nothing at hops below h (spec.md "V, the clairvoyant point").
CHAINS = {
    "h0": [("h1", "h0"), ("h2", "h1"), ("h3", "h2"), ("h4", "h3")],
    "h4": [("h3", "h4"), ("h2", "h3"), ("h1", "h2"), ("h0", "h1")],
}
PAIRS = [pr for ch in CHAINS.values() for pr in ch]
N_HOPS = 4


# ---------------------------------------------------------------------------
# The simulator: one subclass, three changes
# ---------------------------------------------------------------------------

class Engine(O.Engine):
    """evidence/15's engine with a per-node rule copy, placeable polls and F's coordinates.

    Three changes, and nothing else (spec.md "Simulator").

    1. `_Engine.__init__` hands one and the same rule dict to all five nodes, and
       ../14/workload.py builds that dict once per (donor, p, seed) and reuses the
       object across workloads and replicates.  An in-place write would therefore reach
       another node and would survive the simulation, so the dict is copied per node
       here.

    2. `resolve` writes the fields the base method reads, just before calling it.
       Setting `nd.ival[nb]` alone does not place a poll after an answered reply,
       because the base method overwrites `ival` from the rule; writing the rule alone
       does not place one after a timeout, because the base method then leaves `ival`
       untouched.  Both are written on every live resolution of every pair, so that no
       pair runs on what another pair wrote.

    3. Under V only, every tick lag comes from one cached generator per
       (replicate seed, node, neighbour).  The base class draws every lag from one
       stream in event order, so changing hop h's schedule would shift the lags of hops
       below h and break the causal solve order V needs.  A fresh generator per call
       would hand the DP a constant lag it could count on, which is what check 5 looks
       for.

    The override is a pure pass-through when `plan` is None, `i_timeout` is "keep" and
    `k_cap` is None: it then writes no field at all and the base classes decide
    everything.  That is what makes checks 1 and 2 real checks.
    """

    def __init__(self, run, mode, lags, sim_rng, rule=None, plan=None, lag_seed=None):
        super().__init__(run, mode, lags, sim_rng, rule)
        for nd in self.nodes.values():
            if nd.rule is not None:
                nd.rule = dict(nd.rule)
                # The member's own `m`, kept beside the live one so that a resolution
                # that caps `m` at 1 can put the member's value back on the next one.
                nd.rule["m_member"] = nd.rule["m"]
        self.vplan = plan
        self.lag_seed = lag_seed
        self._lag_rng = {}
        self._empty_run = {}          # (node, nb) -> consecutive empty replies so far
        self._vi = {}                 # (node, nb) -> index into this pair's instants
        self.v_chosen = {}            # (node, nb) -> the instants the DP actually drove
        self.v_armed = []             # (node, nb, status, chosen, armed due)
        self.v_clamped = 0            # chosen instants the base method clamped to `now`
        self.v_dropped = 0            # resolutions the provisional timer had replaced
        self.v_exhausted = 0          # resolutions with no further instant to place

    # -- change 3 ---------------------------------------------------------
    def tick_lag(self, node, nb, kind):
        if self.vplan is None:
            return super().tick_lag(node, nb, kind)
        g = self._lag_rng.get((node, nb))
        if g is None:
            g = self._lag_rng[(node, nb)] = random.Random(
                f"ev18:vlag:{self.lag_seed}:{node}:{nb}")
        return g.uniform(0.0, R.TICK_PERIOD_MS)

    # -- change 2 ---------------------------------------------------------
    def _place(self, node, nb, sent_at):
        """The gap in seconds that puts this pair's next poll on the DP's next instant.

        The base method computes `due = max(sent_at + g*1000, t)`, so a gap measured
        from the send is what places the poll.  Instants at or before the send are
        already past and are skipped.  When the DP has no further instant the pair is
        parked beyond the end of the run rather than left on a stale interval.
        """
        lst = self.vplan.get((node, nb))
        if not lst:
            self.v_exhausted += 1
            return V_NO_MORE_S
        i = self._vi.get((node, nb), 0)
        while i < len(lst) and lst[i] <= sent_at:
            i += 1
        self._vi[(node, nb)] = i
        if i >= len(lst):
            self.v_exhausted += 1
            return V_NO_MORE_S
        self.v_chosen.setdefault((node, nb), []).append(lst[i])
        return (lst[i] - sent_at) / 1000.0

    def resolve(self, t, node, nb, sent_at, status, empty=None):
        nd = self.nodes[node]
        pend = nd.pending.get(nb)
        live = pend is not None and pend[0] == sent_at and not pend[2]
        chosen = None
        if live and nd.rule is not None:
            r = nd.rule
            if self.vplan is not None:
                g = self._place(node, nb, sent_at)
                chosen = sent_at + g * 1000.0
                if chosen < t:
                    self.v_clamped += 1
                r["i_min"] = r["i_max"] = g
                r["m"] = V_BIG_M
                r["jf"] = 0.0
                nd.ival[nb] = g
            else:
                cap = r.get("k_cap")
                tmo = r.get("i_timeout", "keep")
                if cap is not None or tmo != "keep":
                    key = (node, nb)
                    before = self._empty_run.get(key, 0)
                    if cap is not None:
                        # `m` stops applying once this pair has seen `k_cap` consecutive
                        # empty replies.  The base `consec` counter counts timeouts, so
                        # the empty counter has to be this subclass's own.
                        r["m"] = 1.0 if before >= cap else r["m_member"]
                    if status == "answered":
                        self._empty_run[key] = before + 1 if empty else 0
                    elif tmo != "keep":
                        # The base method leaves `ival` alone on a timeout, so the
                        # timeout coordinate has to be written here.  A timeout is not
                        # an empty reply, so it leaves the empty counter where it was.
                        nd.ival[nb] = tmo
        elif (self.vplan is not None and nd.rule is not None
              and pend is not None and pend[0] != sent_at):
            # The provisional timer replaced this poll before it resolved, so the
            # instant the DP chose for it is silently dropped.  Check 4 reads this.
            self.v_dropped += 1
        out = super().resolve(t, node, nb, sent_at, status, empty=empty)
        if chosen is not None:
            # What the base method actually armed, beside what the DP asked for.  The
            # two differ only when the clamp moved the instant forward, and the realised
            # send is this instant plus a tick lag below 100 ms, so this is the exact
            # form of check 4's first half.
            self.v_armed.append((node, nb, status, chosen, nd.due.get(nb)))
        return out


def sim(run, rep_rng, plan=None, lag_seed=None):
    """One Mode B simulation through this step's subclass."""
    return Engine(run, "B", None, rep_rng, None, plan, lag_seed).go()


# ---------------------------------------------------------------------------
# Runs and rules
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def run_of(stage, p, seed):
    """`../13/replay.py`'s run dict without the logged events, cached per worker.

    The events are the measured side and no number of this step reads them, while they
    are 90 times the size of everything the simulator needs, so dropping them is what
    lets every worker keep all its runs in memory.
    """
    r = R.load_run(stage, p, seed)
    r.pop("events")
    return r


def rep_rng_of(donor, p, seed, rep):
    """../13/trickle.py's tick-lag stream, literal prefix included.

    W0 only reproduces evidence/14 and evidence/15 if the stream is the same one, so
    the seed string is theirs and is not re-derived here.
    """
    return random.Random(R.rep_seed(f"trickle:{donor}", p, seed, rep))


def base_runs(donor, p, seed):
    """{rule name: a run dict ready to simulate} for the seven arms, short and wait.

    The settings of a "fixed" or "policy" arm are the target stage's own, put on the
    donor run's exogenous inputs, which is ../14/workload.py's move and the reason the
    four testbed arms reproduce evidence/13 bit for bit.
    """
    fam = donor[:2]
    out = {}
    for name, kind, spec in RULE_SPECS:
        if kind in ("fixed", "policy"):
            tgt = run_of(fam + spec, p, seed)
            out[name] = {k: tgt[k] for k in ("policy", "choices", "choice_jitter")}
    return out


def run_for(donor_run, created, name, kind, spec, tbl):
    """One run dict for one rule of RULE_SPECS, as ../15/onoff.py builds it."""
    run = dict(donor_run)
    run["created"] = created
    if kind in ("fixed", "policy"):
        run.update(tbl[name])
        run["rule"] = None
    else:
        run["policy"] = run["choices"] = None
        run["rule"] = (spec if kind == "rule"
                       else dict(i_min=spec, m=1.0, i_max=spec, jf=T.JF))
    return run


def rule_run(donor_run, created, rule):
    """One run dict for an F member or for a V schedule."""
    run = dict(donor_run)
    run["created"] = created
    run["policy"] = run["choices"] = None
    run["rule"] = dict(rule)
    return run


def created_of(wl, p, seed, donor_run):
    """The creation times of workload `wl`, through evidence/15's own generator."""
    return O.gen_created(wl, p, seed, donor_run["t0"], donor_run["created"])


# ---------------------------------------------------------------------------
# C, the structural ceiling
# ---------------------------------------------------------------------------

def link_events(run, a, b):
    """The timeline of the link between two node names, as (ts, is_up) in order."""
    return run["timeline"].get(frozenset((int(a[1:]), int(b[1:])))) or []


def up_at(tl, t):
    """The link state a packet put on the link at `t` is judged against.

    ../13/replay.py `delivers`: the state is set by the last timeline event STRICTLY
    before `t`, so an event stamped at exactly `t` has not taken effect yet and a poll
    sent in the exact millisecond of a link_up is lost.
    """
    up = True
    for ts, is_up in tl:
        if ts < t:
            up = is_up
        else:
            break
    return up


def next_up(tl, t):
    """The first instant at or after `t` at which the link is up.

    Used by C only, where being generous is the safe direction: C has to be at or below
    every realised delay, so it waits until the link_up stamp itself rather than until
    the first instant a poll sent then would survive (link_up + 1 ms).
    """
    if up_at(tl, t):
        return t
    for ts, is_up in tl:
        if ts >= t and is_up:
            return ts
    return float("inf")


def dmin_map(run, created):
    """{(origin, seq): D_min ms}, the outage-gated zero-wait floor of every record.

    spec.md "C, the structural ceiling": propagate the creation time along the record's
    chain with zero discovery wait and zero fetch wait, gated only by the link timeline,
    adding the engine's own per-hop transit constant at each hop.  A record cannot cross
    a link that is down and cannot be fetched before it is advertised, so no policy at
    any budget can beat this.
    """
    tls = {o: [link_events(run, a, b) for a, b in CHAINS[o]] for o in CHAINS}
    out = {}
    for ts, origin, seq in created:
        t = float(ts)
        for tl in tls[origin]:
            t = next_up(tl, t) + TRANSIT_MS
        out[(origin, seq)] = t - float(ts)
    return out


def c_record_set(run, created):
    """(the records C's table is read over, sum of their D_min, the records dropped).

    spec.md: every record created in the scorer's window and at or before t0+72 s whose
    floor reaches the receiver inside the window.  A record outside that set is censored
    under every policy including a clairvoyant one, so keeping it would let a policy
    that censors more score a faster subset than the arm it is read against.
    """
    t0 = run["t0"]
    ws, we = t0 + R.WINDOW_START_MS, t0 + R.WINDOW_END_MS
    dm = dmin_map(run, created)
    keep, drop, tot = {}, 0, 0
    for ts, origin, seq in created:
        if not (ws <= ts <= we) or ts > t0 + CUT_MS:
            continue
        tot += 1
        d = dm[(origin, seq)]
        if ts + d <= we:
            keep[(origin, seq)] = (float(ts), d)
        else:
            drop += 1
    return keep, drop, tot


def cens_score(ev, t0, keep):
    """(n, sum of delay ms, worst D_min violation) over C's own record set.

    Every policy's delay in C's table is recomputed over the same set, with a censored
    record counted at `we - created`, so that C's mean is a floor on every policy's mean
    over one shared set.  The violation column is spec.md's check 6: record by record,
    D_min has to be at or below what the simulation delivered.
    """
    we = t0 + R.WINDOW_END_MS
    stored = {}
    for node, rows in ev.items():
        for e in rows:
            if e.get("event") == "record_stored":
                k = (e["origin"], e["seq"], node)
                if k not in stored or e["ts_ms"] < stored[k]:
                    stored[k] = e["ts_ms"]
    n, tot, worst = 0, 0.0, -float("inf")
    for (origin, seq), (ts, d) in keep.items():
        node = "h4" if origin == "h0" else "h0"
        got = stored.get((origin, seq, node))
        real = we - ts if (got is None or got > we) else got - ts
        n += 1
        tot += real
        worst = max(worst, d - real)
    return n, tot, worst


# ---------------------------------------------------------------------------
# The seven arms, short, wait and the evidence/15 rules, through this subclass
# ---------------------------------------------------------------------------

def job_base(arg):
    """One (donor, p, seed): every workload, every rule of RULE_SPECS, every replicate.

    This is ../15/onoff.py's `job` with its own simulator swapped for this step's
    subclass and with C's record set added.  Check 1 is the statement that the swap
    changes nothing: the rows this produces on W0 have to reproduce
    ../14/workload_cells.csv and ../15/onoff_cells.csv to every printed digit.
    """
    donor, p, seed, workloads = arg
    donor_run = run_of(donor, p, seed)
    t0 = donor_run["t0"]
    tbl = base_runs(donor, p, seed)
    out = {}
    for wl in workloads:
        created = created_of(wl, p, seed, donor_run)
        W.check_created(created, t0, (wl, donor, p, seed))
        total = len(created)
        keep, drop, _tot = c_record_set(donor_run, created)
        out[("records", wl)] = total
        out[("cset", wl)] = (len(keep), drop, sum(d for _ts, d in keep.values()))
        for name, kind, spec in RULE_SPECS:
            run = run_for(donor_run, created, name, kind, spec, tbl)
            reps, cens, worst = [], [], -float("inf")
            for rep in range(REPS):
                ev = sim(run, rep_rng_of(donor, p, seed, rep))
                reps.append((T.tup(R.score(ev, t0, p)),
                             A.early_score(ev, t0),
                             W.gaps(ev, t0, total),
                             O.nonempty_applied(ev, t0)))
                n, tot_ms, bad = cens_score(ev, t0, keep)
                cens.append((n, tot_ms))
                worst = max(worst, bad)
            out[(wl, name)] = reps
            out[("cens", wl, name)] = (cens, worst)
    return donor, p, seed, out


def cell_row(wl, donor, p, name, kind, reps_all, rec, settings=None):
    """One row of bound_cells.csv, built exactly as ../15/onoff.py builds its own."""
    flat = [r for seed_reps in reps_all for r in seed_reps]
    v = T.pooled([m for m, _e, _g, _a in flat], p)
    v.update(W.pooled_gaps([g for _m, _e, g, _a in flat]))
    na = sum(a[0] for _m, _e, _g, a in flat)
    nn = sum(a[1] for _m, _e, _g, a in flat)
    em = T.seed_matrix([[tuple(e) + (0.0, 0.0, 0.0) for _m, e, _g, _a in seed_reps]
                        for seed_reps in reps_all])
    ev = O.early_vals(em)
    return dict(
        workload=wl, donor=donor, p=p, rule=name,
        kind=("fixed" if kind == "arm" else kind),
        settings=settings if settings is not None else O.interval_text(name),
        records=(rec[0] if min(rec) == max(rec) else float(sum(rec)) / len(rec)),
        delay=v["delay"], cens4=v["cens4"], polls_sent=v["polls_sent"],
        answered=v["answered"], timed_out=v["timed_out"], bytes=v["bytes"],
        empty_share=v["empty_share"], mean_gap_s=v["mean_gap_s"],
        lw_gap_s=v["lw_gap_s"], polls_after_complete=v["polls_after_complete"],
        records_min=min(rec), records_max=max(rec),
        early_delay=ev["delay"], early_cens4=ev["cens4"],
        early_polls_sent=ev["polls_sent"],
        nonempty_interval_s=(na / nn if nn else float("nan")),
        nonempty_answers=nn / len(flat))


def read_cells(path, key_cols=("workload", "donor", "p", "rule")):
    """A published cells csv, keyed by (workload, donor, p, rule)."""
    out = {}
    with open(path) as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            out[tuple(float(r[c]) if c == "p" else r[c] for c in key_cols)] = r
        cols = rdr.fieldnames
    return out, cols


def run_base(workloads, pool=None):
    """Every rule of RULE_SPECS on every cell and workload, in parallel over the seeds."""
    import multiprocessing as mp
    jobs = [(d, p, s, tuple(workloads)) for (d, p) in CELLS for s in SEEDS]
    if pool is None:
        with mp.Pool(WORKERS) as own:
            res = own.map(job_base, jobs, chunksize=1)
    else:
        res = pool.map(job_base, jobs, chunksize=1)
    per_seed, records, cset, cens, worst = {}, {}, {}, {}, 0.0
    for donor, p, seed, out in res:
        for key, v in out.items():
            if key[0] == "records":
                records[(key[1], donor, p, seed)] = v
            elif key[0] == "cset":
                cset[(key[1], donor, p, seed)] = v
            elif key[0] == "cens":
                cens[(key[1], donor, p, key[2], seed)] = v[0]
                worst = max(worst, v[1])
            else:
                per_seed[(key[0], donor, p, key[1], seed)] = v
    return per_seed, records, cset, cens, worst


def base_cell_rows(per_seed, records, workloads):
    rows = []
    for wl in workloads:
        for (donor, p) in CELLS:
            rec = [records[(wl, donor, p, s)] for s in SEEDS]
            for name, kind, _spec in RULE_SPECS:
                rows.append(cell_row(wl, donor, p, name, kind,
                                     [per_seed[(wl, donor, p, name, s)] for s in SEEDS],
                                     rec))
    return rows


# ---------------------------------------------------------------------------
# Check 1: W0 byte-exact reproduction.  Nothing else is read before it passes.
# ---------------------------------------------------------------------------

def check1():
    t = time.time()
    per_seed, records, _cset, _cens, _worst = run_base(["W0"])
    rows = base_cell_rows(per_seed, records, ["W0"])
    bad, checked = [], 0
    for path, only in ((os.path.join(EV15, "onoff_cells.csv"), None),
                       (os.path.join(EV14, "workload_cells.csv"), O.SHARED_RULES)):
        ref, cols = read_cells(path)
        for r in rows:
            if only is not None and r["rule"] not in only:
                continue
            want = ref[(r["workload"], r["donor"], r["p"], r["rule"])]
            diffs = {c: (O.csv_str(r[c]), want[c]) for c in cols
                     if c in r and O.csv_str(r[c]) != want[c]}
            checked += 1
            if diffs:
                bad.append((os.path.basename(path), r["donor"], r["p"], r["rule"], diffs))
    print(f"check 1 W0 byte-exact: {checked - len(bad)}/{checked} (cell x rule x file) "
          f"identical on every printed column", flush=True)
    for b in bad:
        print("  ", b)
    print(f"check 1 wall {time.time() - t:.1f} s")
    if bad:
        sys.exit("check 1 FAILED: W0 does not reproduce evidence/14 and evidence/15")
    return rows




# ---------------------------------------------------------------------------
# V: the candidate set, the per-hop DP, and the four-pass solve
# ---------------------------------------------------------------------------

def feasible_instant(tl, c, end):
    """The first instant at or after `c` at which V is allowed to place a poll.

    spec.md "The feasible instant set has three parts".  A poll put on a down link is
    lost, a poll sent in the exact millisecond of a link_up is judged against the down
    state and is lost, and a tick lag of up to 100 ms can push a poll placed just before
    a link_down into the outage, where the provisional timer would take the pair over
    and the chosen instant would be silently dropped.  So the down periods, the link_up
    instants themselves and the 100 ms before every link_down are all pushed forward.

    2026-11-19 verifier fix: the pre-down guard tests `ts >= c` and not `ts > c`.  A
    candidate stamped in the exact millisecond of a link_down is judged up by `up_at`,
    which reads the state from the last event strictly before `c`, while
    `replay.delivers` loses such a packet on its own second test, a link_down inside
    `[t_send, t_arrive)`.  That instant is infeasible, so it is pushed forward too.
    """
    for _ in range(2 * len(tl) + 2):
        if c > end:
            return None
        if not up_at(tl, c):
            nxt = next((ts for ts, is_up in tl if ts >= c and is_up), None)
            if nxt is None:
                return None
            c = nxt + 1.0
            continue
        down = next((ts for ts, is_up in tl if ts >= c and not is_up), None)
        if down is not None and down - c <= PRE_DOWN_GUARD_MS:
            c = down + 1.0
            continue
        return c
    return None


def candidate_set(run, pair, avail):
    """V's candidate poll instants for one directed pair.

    spec.md "The reachable target set": the availability instants at the holder plus
    the `link_up + 1 ms` instants of that pair, each pushed forward to the first
    feasible instant.  The count of this set is what caps V's budget.
    """
    recv, hold = pair
    tl = link_events(run, recv, hold)
    end = run["t0"] + R.STOP_AT_MS
    raw = {float(a) for a in avail}
    raw |= {float(ts) + 1.0 for ts, is_up in tl if is_up}
    out = set()
    for c in raw:
        f = feasible_instant(tl, c, end)
        if f is not None:
            out.add(f)
    return sorted(out)


def dp_solve(cands, avail, lam, tl, end):
    """Poll instants minimising discovery wait plus `lam` per poll, over `cands`.

    The objective is spec.md's: sum over availability instants of (first delivered poll
    at or after it) minus it, plus the price of every poll.  A record no poll reaches is
    charged until the last instant a poll could be sent, which is the one finite penalty
    that keeps the DP from dropping the tail of the run for free.  The state is the last
    chosen candidate, so the DP is optimal over this candidate set; the continuous
    optimum can be better, because the feasible set of a later poll depends on whether
    the previous poll was delivered.

    Returns (chosen instants, how often the post-timeout floor bound a transition).
    """
    A = sorted(float(a) for a in avail)
    K = len(cands)
    pref = [0.0]
    for a in A:
        pref.append(pref[-1] + a)
    inf = float("inf")
    dp = [inf] * (K + 1)
    pre = [-1] * (K + 1)
    dp[0] = 0.0
    # The candidate set already excludes the down periods, so a poll placed on one of
    # these instants is delivered and the 2.1 s floor should never bind.  It is still
    # asked for, and how often it bound is reported.
    lost = [not up_at(tl, c) for c in cands]
    nfloor = 0
    for i in range(1, K + 1):
        ci = cands[i - 1]
        hi = bisect.bisect_right(A, ci)
        for j in range(0, i):
            if dp[j] == inf:
                continue
            if j > 0:
                cj = cands[j - 1]
                if lost[j - 1] and ci - cj < TIMEOUT_FLOOR_MS:
                    nfloor += 1
                    continue
                lo = bisect.bisect_right(A, cj)
            else:
                lo = 0
            if lo > hi:
                continue
            v = dp[j] + lam + ((hi - lo) * ci - (pref[hi] - pref[lo]))
            if v < dp[i]:
                dp[i] = v
                pre[i] = j
    nA = len(A)
    best = nA * end - pref[nA]          # the schedule with no poll at all
    bi = 0
    for i in range(1, K + 1):
        if dp[i] == inf:
            continue
        hi = bisect.bisect_right(A, cands[i - 1])
        v = dp[i] + ((nA - hi) * end - (pref[nA] - pref[hi]))
        if v < best:
            best, bi = v, i
    out = []
    i = bi
    while i > 0:
        out.append(cands[i - 1])
        i = pre[i]
    return out[::-1], nfloor


def stored_at(ev, node, origin):
    """The instants at which `node` first held each record of `origin`, in order."""
    seen = {}
    for e in ev[node]:
        k = e.get("event")
        if k == "record_stored" and e["origin"] == origin:
            if e["seq"] not in seen or e["ts_ms"] < seen[e["seq"]]:
                seen[e["seq"]] = e["ts_ms"]
        elif k == "record_created" and e["origin"] == origin:
            seen.setdefault(e["seq"], e["ts_ms"])
    return sorted(float(v) for v in seen.values())


def v_rule():
    """V's rule dict.  `i_min` is the frontier's longest arm, which `Node.__init__`
    reads once for the first poll of each pair: a value below the 2 s interest lifetime
    would let a lost first poll hand the pair to the provisional timer.  `jf` is 0, so
    that `applied` equals the interval exactly and the per-node jitter stream, which is
    shared across a node's neighbours, cannot reorder under a schedule change."""
    return dict(i_min=V_INIT_S, m=1.0, i_max=V_INIT_S, jf=0.0)


def solve_v_run(donor, p, seed, wl, lam, rep):
    """One (seed, replicate) of V: four passes, four simulations, the last one scored.

    Hop 1's input is exogenous, so hop 1 is solved first, simulated, and the realised
    instants at which each record became available at the hop-1 receiver are read off.
    Hop 2 is solved against those, and so on.  A node's store is not changed by the
    polls of the node downstream of it, so hops below h stay put when hop h changes and
    the order terminates with no fixed point.
    """
    donor_run = run_of(donor, p, seed)
    created = created_of(wl, p, seed, donor_run)
    run = rule_run(donor_run, created, v_rule())
    end = donor_run["t0"] + R.STOP_AT_MS
    lag_seed = R.rep_seed(f"ev18:v:{donor}", p, seed, rep)
    avail = {o: [float(ts) for ts, origin, _s in created if origin == o]
             for o in CHAINS}
    plan, ncand, nfloor, ev = {}, 0, 0, None
    for h in range(N_HOPS):
        for origin, chain in CHAINS.items():
            pair = chain[h]
            if h:
                avail[origin] = stored_at(ev, chain[h - 1][0], origin)
            cands = candidate_set(donor_run, pair, avail[origin])
            ncand += len(cands)
            inst, nf = dp_solve(cands, avail[origin], lam,
                                link_events(donor_run, *pair), end)
            nfloor += nf
            plan[pair] = inst
        eng = Engine(run, "B", None, rep_rng_of(donor, p, seed, rep), None, plan,
                     lag_seed)
        ev = eng.go()
    return ev, eng, plan, ncand, nfloor


def job_v(arg):
    """One (cell, workload, seed, replicate) of V at one price, scored like any rule."""
    donor, p, seed, wl, lam, rep, keep_instants = arg
    donor_run = run_of(donor, p, seed)
    t0 = donor_run["t0"]
    created = created_of(wl, p, seed, donor_run)
    ev, eng, plan, ncand, nfloor = solve_v_run(donor, p, seed, wl, lam, rep)
    keep, _drop, _tot = c_record_set(donor_run, created)
    n, tot_ms, worst = cens_score(ev, t0, keep)
    sc = R.score(ev, t0, p)
    chosen = sum(len(v) for v in eng.v_chosen.values())
    taken = sum(len(v) for v in plan.values())
    place = v_placement(ev, eng, plan)
    return dict(
        key=(donor, p, wl, seed, rep),
        met=T.tup(sc), early=A.early_score(ev, t0),
        gaps=W.gaps(ev, t0, len(created)), nonempty=O.nonempty_applied(ev, t0),
        cens=(n, tot_ms), worst=worst, polls=sc["polls"],
        ncand=ncand, nfloor=nfloor, chosen=chosen, taken=taken,
        clamped=eng.v_clamped, dropped=eng.v_dropped, exhausted=eng.v_exhausted,
        place=place,
        instants=({pr: [round(x - t0, 1) for x in v] for pr, v in plan.items()}
                  if keep_instants else None))


def v_placement(ev, eng, plan):
    """Check 4's counters, measured on the instants themselves.

    Two statements, both alignment free.  First, per live resolution, the instant the
    base method armed against the instant the DP chose, separately for the resolutions
    that followed an answered poll and for those that followed a timed-out poll; the two
    differ only when the clamp moved the instant forward, and the realised send is the
    armed instant plus a tick lag below 100 ms.  Second, every realised poll of the run
    matched to the nearest instant the DP chose for that pair, which catches a poll the
    provisional timer sent somewhere the DP never asked for.

    An earlier form of this function zipped a pair's chosen instants with its realised
    sends in order.  That instrument misreads the run from the first poll the
    provisional timer takes over, because such a poll adds a send and consumes no chosen
    instant, and it reported five answered placements and one timed-out placement off
    target that the measurement below shows to be on target.
    """
    out = {"answered": [0, 0], "timed_out": [0, 0]}
    for _node, _nb, status, chosen, due in eng.v_armed:
        out[status][0] += 1
        out[status][1] += due is None or abs(due - chosen) > R.TICK_PERIOD_MS
    far = 0
    for node, rows in ev.items():
        for nb in eng.nodes[node].neighbors:
            want = plan.get((node, nb)) or []
            sends = [float(e["ts_ms"]) for e in rows
                     if e.get("event") == "decision" and e.get("kind") == "state_poll"
                     and e["action"]["neighbor"] == nb]
            for t in sends[1:]:          # the first poll of a pair is the forced one
                if not want:
                    far += 1
                    continue
                i = bisect.bisect_left(want, t)
                near = min((abs(t - want[j]) for j in (i - 1, i)
                            if 0 <= j < len(want)), default=float("inf"))
                far += near > R.TICK_PERIOD_MS
    out["far_from_any_chosen"] = far
    return out


def stored_seq(ev, node, origin):
    """{seq: the instant `node` first held that record of `origin`}."""
    seen = {}
    for e in ev[node]:
        if e.get("event") == "record_stored" and e["origin"] == origin:
            ts = float(e["ts_ms"])
            if e["seq"] not in seen or ts < seen[e["seq"]]:
                seen[e["seq"]] = ts
    return seen


def v_knows_illustration(donor, p, wl, lam, arm="fix10.043", origin="h0", seed=1,
                         rep=0):
    """One record's hop-by-hop fetch instants under V, beside the same record under one
    fixed arm.

    Added 2026-11-19 for the section "What V knows". The record is chosen mechanically:
    the lowest `seq` of `origin` that both runs carry to every hop of its chain, so that
    all eight instants exist. Nothing here enters a published number; it is one
    measured illustration of the information V's schedule is built from.
    """
    donor_run = run_of(donor, p, seed)
    t0 = donor_run["t0"]
    created = created_of(wl, p, seed, donor_run)
    ev_v, _eng, _plan, _nc, _nf = solve_v_run(donor, p, seed, wl, lam, rep)
    tbl = base_runs(donor, p, seed)
    name, kind, spec = next(r for r in RULE_SPECS if r[0] == arm)
    ev_a = sim(run_for(donor_run, created, name, kind, spec, tbl),
               rep_rng_of(donor, p, seed, rep))
    hops = [recv for recv, _hold in CHAINS[origin]]
    got = {"V": [stored_seq(ev_v, h, origin) for h in hops],
           "arm": [stored_seq(ev_a, h, origin) for h in hops]}
    common = sorted(set.intersection(*[set(d) for d in got["V"] + got["arm"]]))
    if not common:
        return None
    seq = common[0]
    made = next(float(ts) for ts, o, s in created if o == origin and s == seq)
    return dict(donor=donor, p=p, wl=wl, seed=seed, rep=rep, arm=arm, origin=origin,
                seq=seq, hops=hops, lam=lam, created=made - t0,
                v=[got["V"][i][seq] - t0 for i in range(len(hops))],
                a=[got["arm"][i][seq] - t0 for i in range(len(hops))])


# ---------------------------------------------------------------------------
# F: the family, its evaluation, and the coordinate ascent
# ---------------------------------------------------------------------------

M_VALUES = [1.2, 1.5, 2.0, 3.0]
K_CAPS = [1, 2, 3, None]
RESTRICTED_MIN_S = 2.2               # the interval below which the replay's 6 % fidelity
                                     # does not transfer and V is not allowed to go
START_MEMBER = (3.695, 2.0, 10.043, 3.695, None)     # T37r


def member_rule(mem):
    """The rule dict of one F member.  `jitter_frac` is 0.1 throughout, as every
    earlier step.  A member with `i_timeout` keep and `k_cap` None carries both keys and
    still runs through the base classes alone, which is what check 2 measures."""
    i_min, m, i_max, i_timeout, k_cap = mem
    return dict(i_min=i_min, m=m, i_max=i_max, jf=T.JF,
                i_timeout=i_timeout, k_cap=k_cap)


def member_text(mem):
    i_min, m, i_max, i_timeout, k_cap = mem
    return (f"I_min {i_min} m {m} I_max {i_max} jf {T.JF}, on a timeout "
            f"{'I unchanged' if i_timeout == 'keep' else f'I = {i_timeout}'}, "
            f"m stops after {'no cap' if k_cap is None else k_cap} empty replies")


def coord_values(mem, c, restricted):
    """The values coordinate `c` takes, in the order spec.md's table lists them."""
    lo = RESTRICTED_MIN_S if restricted else 0.0
    if c == 0:
        return [x for x in ARM_INTERVALS if x >= lo]
    if c == 1:
        return list(M_VALUES)
    if c == 2:
        return [x for x in ARM_INTERVALS if x >= mem[0]]
    if c == 3:
        return ["keep"] + [x for x in ARM_INTERVALS if x >= lo]
    return list(K_CAPS)


def job_member(arg):
    """One (cell, workload, seed) of one F member, all five replicates."""
    donor, p, seed, wl, mem = arg
    donor_run = run_of(donor, p, seed)
    t0 = donor_run["t0"]
    created = created_of(wl, p, seed, donor_run)
    keep, _drop, _tot = c_record_set(donor_run, created)
    run = rule_run(donor_run, created, member_rule(mem))
    reps, worst = [], -float("inf")
    for rep in range(REPS):
        ev = sim(run, rep_rng_of(donor, p, seed, rep))
        n, tot_ms, bad = cens_score(ev, t0, keep)
        worst = max(worst, bad)
        reps.append((T.tup(R.score(ev, t0, p)), A.early_score(ev, t0), (n, tot_ms),
                     W.gaps(ev, t0, len(created)), O.nonempty_applied(ev, t0)))
    return (donor, p, wl, seed, mem), reps, worst


# ---------------------------------------------------------------------------
# Pooling and the frontier, on the early-record scoring spec.md makes the headline
# ---------------------------------------------------------------------------

def seed_mat(per_seed_reps, idx, width):
    """A (seeds, 7) matrix of per-seed means, as ../13/trickle.py's bootstrap wants it.

    `idx` picks the tuple inside each replicate record and `width` says how long it is;
    the early-record tuple is four long and is zero padded, which is the shape
    ../15/onoff.py already feeds to the same code.
    """
    return T.seed_matrix([[tuple(r[idx]) + (0.0,) * (7 - width) for r in reps]
                          for reps in per_seed_reps])


def vals_of(mat):
    return O.early_vals(mat)


def boot_idx(n):
    return np.random.default_rng(BOOT_SEED).integers(0, n, size=(BOOT_N, n))


class Frontier:
    """The seven fixed arms of one (cell, workload) over one set of seeds.

    Everything this step reads a gain against is this object: the point gain comes from
    the arms' pooled values and the interval from the same resampled seeds, which is
    what makes a difference of two gains a paired statistic.
    """

    def __init__(self, arm_mats, p):
        self.p = p
        self.mats = arm_mats
        self.n = next(iter(arm_mats.values())).shape[0]
        self.idx = boot_idx(self.n)
        self.vals = {n: vals_of(arm_mats[n]) for n in FIXED}
        self.boot = {n: T.boot_values(arm_mats[n], self.idx, p) for n in FIXED}
        self.fxp = np.array([[self.vals[n]["polls_sent"] for n in FIXED]])
        self.fxd = np.array([[self.vals[n]["delay"] for n in FIXED]])
        self.bfp = np.stack([self.boot[n][1] for n in FIXED], axis=1)
        self.bfd = np.stack([self.boot[n][0] for n in FIXED], axis=1)
        order = sorted(FIXED, key=lambda n: self.vals[n]["polls_sent"])
        self.mono = all(self.vals[a]["delay"] > self.vals[b]["delay"]
                        for a, b in zip(order, order[1:]))

    def gain(self, mat):
        """(point gain or None, the bootstrap array, the pooled values of `mat`)."""
        v = vals_of(mat)
        g, _s, in_d, _in_p = T.gain_save(
            self.fxp, self.fxd, np.array([v["polls_sent"]]), np.array([v["delay"]]))
        d, pl, _c = T.boot_values(mat, self.idx, self.p)
        b, _bs, _i, _j = T.gain_save(self.bfp, self.bfd, pl, d)
        return (float(g[0]) if in_d[0] else None), b, v


def pct(x):
    return None if x is None else 100.0 * x


def ci_pts(arr):
    """A bootstrap interval in gain POINTS, with the resample count it rests on."""
    lo, hi, n = T.ci(arr)
    return (None if lo is None else 100.0 * lo,
            None if hi is None else 100.0 * hi, n)


# ---------------------------------------------------------------------------
# Checks 2, 3, 5, 7 and 8
# ---------------------------------------------------------------------------

IDENT_ARMS = [("s6b0p5", 0.5), ("s6b1p359", 1.359), ("s6b3p695", 3.695),
              ("s6b10p043", 10.043)]
IDENT_P = 0.4
IDENT_SEEDS = (1, 2, 3)


def check2():
    """An inert F member reproduces the fixed arm of its interval event for event.

    ../13/identity.py makes this claim for the base engine; here the rule path runs
    through this step's subclass, with `i_timeout` keep and `k_cap` unbounded, so a
    difference would mean the subclass changes something on a path spec.md says it does
    not touch.  The comparison is ../13/identity.py's own: the rule path's extra
    `state_poll_reschedule` lines are dropped, and the `state` dict is dropped because
    the choices path never maintains `consecutive_poll_timeouts`.
    """
    sys.path.insert(0, EV13)
    import identity as I  # noqa: PLC0415  only this check needs it
    t = time.time()
    bad, done = [], 0
    for stage, x in IDENT_ARMS:
        mem = (x, 1.0, x, "keep", None)
        for seed in IDENT_SEEDS:
            run = R.load_run(stage, IDENT_P, seed)
            lags = R.logged_lags(run)
            alt_run = dict(run)
            alt_run["policy"] = alt_run["choices"] = None
            alt_run["rule"] = member_rule(mem)
            for mode in ("A", "B"):
                if mode == "A":
                    base = R._sim_run(run, "A", lags=lags)
                    alt = Engine(alt_run, "A", lags, None).go()
                else:
                    base = R._sim_run(run, "B", sim_rng=random.Random(
                        R.rep_seed(stage, IDENT_P, seed, 0)))
                    alt = Engine(alt_run, "B", None, random.Random(
                        R.rep_seed(stage, IDENT_P, seed, 0))).go()
                same_ev = (I.comparable(base, with_state=False)
                           == I.comparable(alt, with_state=False))
                same_sc = (I.tup(R.score(base, run["t0"], IDENT_P))
                           == I.tup(R.score(alt, run["t0"], IDENT_P)))
                done += 1
                if not (same_ev and same_sc):
                    bad.append((stage, seed, mode, same_ev, same_sc))
    print(f"check 2 identity through the subclass: {done - len(bad)}/{done} "
          f"(arm x seed x mode) identical in events and in metrics", flush=True)
    for b in bad:
        print("  ", b)
    print(f"check 2 wall {time.time() - t:.1f} s")
    if bad:
        sys.exit("check 2 FAILED: the subclass changes an inert member")


class TraceEngine(Engine):
    """`Engine` with every live rule resolution written down, for check 3 only.

    It adds no behaviour: it reads `ival` before and after the base method and records
    which `m` the resolution ran on.  Nothing of the step's numbers goes through it.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.trace = []

    def resolve(self, t, node, nb, sent_at, status, empty=None):
        nd = self.nodes[node]
        pend = nd.pending.get(nb)
        live = pend is not None and pend[0] == sent_at and not pend[2]
        before = self._empty_run.get((node, nb), 0) if live else None
        ival0 = nd.ival.get(nb)
        out = super().resolve(t, node, nb, sent_at, status, empty=empty)
        if live and nd.rule is not None:
            self.trace.append(dict(t=t, node=node, nb=nb, status=status,
                                   empty=bool(empty), before=before,
                                   m_used=nd.rule["m"], ival0=ival0,
                                   ival1=nd.ival.get(nb)))
        return out


def check3():
    """`k_cap` is per pair, and a capped pair does not stop another pair's `m`.

    The only check that exercises a finite `k_cap`, since checks 1 and 2 both run with
    it unbounded.  Two statements are made: the whole trace obeys
    `m used == 1 exactly when this pair has already seen k_cap empty replies`, which is
    the locality statement itself, and at least one instant exists where one pair of a
    node is capped while another pair of the same node still multiplies by `m`.
    """
    t = time.time()
    donor, p, seed, wl, cap = "s6b10p043", 0.4, 1, "W0", 1
    mem = (2.2, 2.0, 10.043, "keep", cap)
    donor_run = run_of(donor, p, seed)
    run = rule_run(donor_run, created_of(wl, p, seed, donor_run), member_rule(mem))
    eng = TraceEngine(run, "B", None, rep_rng_of(donor, p, seed, 0))
    eng.go()
    tr = eng.trace
    wrong = [r for r in tr
             if r["m_used"] != (1.0 if r["before"] >= cap else mem[1])]
    grew = [r for r in tr if r["status"] == "answered" and r["empty"]
            and r["ival1"] > r["ival0"] + 1e-12]
    witness = None
    for r in grew:
        other = [q for q in tr if q["node"] == r["node"] and q["nb"] != r["nb"]
                 and q["t"] <= r["t"]]
        if other and other[-1]["before"] >= cap:
            witness = (r, other[-1])
            break
    byn = sorted({(r["node"], r["nb"]) for r in tr})
    print(f"check 3 k_cap locality: {len(tr)} live rule resolutions over {len(byn)} "
          f"pairs, {len(wrong)} of them ran on an `m` another pair's counter dictated",
          flush=True)
    if witness:
        a, b = witness
        print(f"  witness at t0+{(a['t'] - donor_run['t0']) / 1000:.3f} s: "
              f"{a['node']}->{a['nb']} had seen {a['before']} empty replies and still "
              f"multiplied {a['ival0']:.3f} s to {a['ival1']:.3f} s, while "
              f"{b['node']}->{b['nb']} of the same node had seen {b['before']} "
              f"(cap {cap}) and was running on m 1")
    print(f"check 3 wall {time.time() - t:.1f} s")
    if wrong or witness is None:
        sys.exit("check 3 FAILED: k_cap is not per pair, or no witness instant exists")


class LagEngine(Engine):
    """`Engine` with every tick lag written down, for check 5 only."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lags_seen = {}

    def tick_lag(self, node, nb, kind):
        v = super().tick_lag(node, nb, kind)
        self.lags_seen.setdefault((node, nb), []).append(v)
        return v


def check5():
    """V's tick lag is a cached generator per pair, not a constant and not shared.

    Two consecutive polls of one pair have to draw different lags, and a pair's lag
    sequence has to be unchanged when another pair's schedule changes.  A fresh
    `random.Random` per call would return the same value every call and would hand the
    DP a constant lag it could count on.
    """
    t = time.time()
    donor, p, seed, wl = "s6b10p043", 0.4, 1, "W0"
    donor_run = run_of(donor, p, seed)
    created = created_of(wl, p, seed, donor_run)
    run = rule_run(donor_run, created, v_rule())
    end = donor_run["t0"] + R.STOP_AT_MS
    probe, other = ("h1", "h0"), ("h3", "h4")
    base_plan = {}
    for origin, chain in CHAINS.items():
        pair = chain[0]
        av = [float(ts) for ts, o, _s in created if o == origin]
        cands = candidate_set(donor_run, pair, av)
        base_plan[pair] = dp_solve(cands, av, 3000.0, link_events(donor_run, *pair),
                                   end)[0]
    moved = dict(base_plan)
    moved[other] = [x for i, x in enumerate(base_plan[other]) if i % 2 == 0]
    seqs = []
    for plan in (base_plan, moved):
        eng = LagEngine(run, "B", None, rep_rng_of(donor, p, seed, 0), None, plan,
                        R.rep_seed(f"ev18:v:{donor}", p, seed, 0))
        eng.go()
        seqs.append(eng.lags_seen)
    a, b = seqs
    same_probe = a[probe] == b[probe]
    const = [k for k, v in a.items() if len(set(v)) <= 1 and len(v) > 1]
    pairs_eq = sum(1 for v in a.values() for x, y in zip(v, v[1:]) if x == y)
    print(f"check 5 V lag stream: {len(a)} pairs drew lags; "
          f"{pairs_eq} consecutive draws of a pair were equal; "
          f"{len(const)} pairs drew a constant; probe pair {probe[0]}->{probe[1]} "
          f"unchanged when {other[0]}->{other[1]} moved: {same_probe} "
          f"(the moved pair drew {len(a[other])} lags before and {len(b[other])} "
          f"after, from its own generator in poll order)", flush=True)
    print(f"  probe drew {len(a[probe])} lags, first three "
          f"{[round(x, 3) for x in a[probe][:3]]}")
    print(f"check 5 wall {time.time() - t:.1f} s")
    if const or pairs_eq or not same_probe:
        sys.exit("check 5 FAILED: V's lag stream is constant or is shared across pairs")


def check7(donor=HEADLINE[0], p=HEADLINE[1], wl=HEADLINE[2], seed=1):
    """Time the DP on one real (cell, workload, seed) before any cell list is started.

    The DP is charged in the cost table like everything else, and the point of timing it
    on a real cell is that the scratchpad sketch ran on synthetic instants.  Returns the
    seconds one run-solve of eight pairs takes.
    """
    donor_run = run_of(donor, p, seed)
    created = created_of(wl, p, seed, donor_run)
    end = donor_run["t0"] + R.STOP_AT_MS
    avail = {o: [float(ts) for ts, origin, _s in created if origin == o] for o in CHAINS}
    sets = []
    for origin, chain in CHAINS.items():
        for pair in chain:
            cands = candidate_set(donor_run, pair, avail[origin])
            sets.append((cands, avail[origin], link_events(donor_run, *pair)))
    n = 20
    t = time.time()
    for _ in range(n):
        for cands, av, tl in sets:
            dp_solve(cands, av, 3000.0, tl, end)
    per = (time.time() - t) / n
    sim_t = time.time()
    run = rule_run(donor_run, created, v_rule())
    for _ in range(5):
        sim(run, rep_rng_of(donor, p, seed, 0))
    sim_t = (time.time() - sim_t) / 5
    print(f"check 7 DP timing on {donor} p{p} {wl} seed {seed}: "
          f"{1000 * per:.2f} ms per run-solve of eight pairs "
          f"({sorted(len(c) for c, _a, _t in sets)} candidates per pair), against "
          f"{1000 * sim_t:.1f} ms for one replay simulation", flush=True)
    return per, sim_t


# ---------------------------------------------------------------------------
# The reachable target set
# ---------------------------------------------------------------------------

def candidate_count(donor, p, seed, wl):
    """The candidate count of one run, counted as the scratchpad counted it.

    The reachability rule of spec.md is stated against this count: per directed pair,
    the distinct creation instants of the origin whose chain the pair carries, plus the
    link_up instants of that pair.  The DP's own candidate set differs from it, because
    the DP uses the realised availability at the holder above hop 1 and pushes every
    instant forward to a feasible one, so this count is kept separate and is used for
    the reachability rule only.
    """
    donor_run = run_of(donor, p, seed)
    created = created_of(wl, p, seed, donor_run)
    per_origin = {}
    for ts, origin, _seq in created:
        per_origin.setdefault(origin, set()).add(int(round(ts)))
    tot = 0
    for origin, chain in CHAINS.items():
        for pair in chain:
            tl = link_events(donor_run, *pair)
            tot += len(per_origin.get(origin, ())) + sum(1 for _ts, u in tl if u)
    return tot


def reach_caps():
    """{(donor, p, wl): (min candidates over the 20 seeds, median, the 70 % cap)}."""
    out = {}
    for (donor, p) in CELLS:
        for wl in WORKLOADS:
            cs = sorted(candidate_count(donor, p, s, wl) for s in SEEDS)
            out[(donor, p, wl)] = (cs[0], cs[len(cs) // 2], REACH_FRAC * cs[0])
    return out


# ---------------------------------------------------------------------------
# V's budget: the price bisection
# ---------------------------------------------------------------------------

def v_eval(pool, donor, p, wl, seeds, lam, reps=(0,), keep=False):
    tasks = [(donor, p, s, wl, lam, rep, keep) for s in seeds for rep in reps]
    return pool.map(job_v, tasks, chunksize=1)


PRICE_LADDER = [4000.0, 6000.0, 7000.0, 7600.0, 7800.0, 7900.0, 8000.0, 8200.0,
                9000.0, 12000.0]


def price_ladder(pool, donor, p, wl):
    """V's realised poll count at a fixed ladder of poll prices, per bisection half.

    Added after the run showed that most budget targets were missed: it is a diagnostic
    of the step's own method and changes no number. The poll count is a staircase in the
    price, and the tread the target needs is narrower than eight bisection steps can
    resolve over a bracket spanning five orders of magnitude.
    """
    out = []
    for lam in PRICE_LADDER:
        row = {}
        for fold in sorted(FOLDS):
            res = v_eval(pool, donor, p, wl, BISECT_SEEDS[fold], lam)
            row[fold] = float(np.mean([r["polls"] for r in res]))
        out.append((lam, row))
    return out


def floor_price_polls(pool, donor, p, wl):
    """V's realised polls per run at the bracket floor price, per bisection half.

    2026-11-19 verifier fix: the poll count falls as the price rises, so the count at
    `LAM_LO_MS` is the most polls V can be made to send at any price in the bracket. A
    budget target above it is out of V's reach at every price, which is a different
    thing from a bisection that failed to resolve the staircase. The section "Why most
    budget targets were missed" classifies every missed target against this number. It
    is measured on the same five bisection seeds the bisection itself runs on, so the
    two counts are comparable, and it enters no published number.
    """
    out = {}
    for fold in sorted(FOLDS):
        res = v_eval(pool, donor, p, wl, BISECT_SEEDS[fold], LAM_LO_MS)
        out[fold] = float(np.mean([r["polls"] for r in res]))
    return out


def bisect_budget(pool, donor, p, wl, seeds, target):
    """The poll price whose realised poll count lands on `target`, cross fitted.

    The poll count falls as the price rises, so the price is bracketed once and then
    halved in log space.  The last two of the eight steps interpolate between the two
    bracket ends in (price, polls), which is what turns a 4 % band in the price into a
    band in the poll count that meets spec.md's 2 % tolerance.  The bracket ends are not
    evaluated: the count is monotone in the price, so an unevaluated end costs nothing
    and buys two more steps inside.  The step reports the realised count whether or not
    the target was reached.
    """
    seen = []

    def polls_at(lam):
        res = v_eval(pool, donor, p, wl, seeds, lam)
        v = float(np.mean([r["polls"] for r in res]))
        seen.append((lam, v))
        return v

    lo, hi = math.log(LAM_LO_MS), math.log(LAM_HI_MS)
    ylo = yhi = None
    for step in range(BISECT_STEPS):
        best = min(seen, key=lambda z: abs(z[1] - target)) if seen else None
        if best is not None and abs(best[1] - target) / target <= BUDGET_TOL:
            break
        if (step >= BISECT_STEPS - 2 and ylo is not None and yhi is not None
                and ylo > yhi):
            w = (ylo - target) / (ylo - yhi)
            x = lo + min(max(w, 0.05), 0.95) * (hi - lo)
        else:
            x = 0.5 * (lo + hi)
        y = polls_at(math.exp(x))
        if y > target:
            lo, ylo = x, y
        else:
            hi, yhi = x, y
    lam, got = min(seen, key=lambda z: abs(z[1] - target))
    return lam, got, abs(got - target) / target, len(seen)


def v_reps(res):
    """job_v results grouped into the per-replicate tuple shape the tables pool."""
    by = {}
    for r in res:
        _d, _p, _wl, seed, rep = r["key"]
        by.setdefault(seed, {})[rep] = r
    out = {}
    for seed, d in by.items():
        out[seed] = [(d[rep]["met"], d[rep]["early"], d[rep]["cens"], d[rep]["gaps"],
                      d[rep]["nonempty"]) for rep in sorted(d)]
    return out, by


def run_v_target(pool, donor, p, wl, target_by_fold, keep=False):
    """V at one budget: the price bisected per fold, then the 20 seeds scored.

    The price V is scored with on one fold is bisected on five seeds of the other fold,
    so there is no in-sample tuning of V's budget on any seed it is scored on.
    """
    reps, info, raw = {}, {}, []
    for fold, seeds in FOLDS.items():
        tgt = target_by_fold[fold]
        lam, got, rel, nev = bisect_budget(pool, donor, p, wl, BISECT_SEEDS[fold], tgt)
        res = v_eval(pool, donor, p, wl, seeds, lam, reps=tuple(range(REPS)), keep=keep)
        rp, by = v_reps(res)
        reps.update(rp)
        raw += res
        info[fold] = dict(lam=lam, target=tgt, bisect_polls=got, bisect_rel=rel,
                          evals=nev,
                          polls=float(np.mean([r["polls"] for r in res])))
    agg = dict(
        ncand=sum(r["ncand"] for r in raw), taken=sum(r["taken"] for r in raw),
        chosen=sum(r["chosen"] for r in raw), nfloor=sum(r["nfloor"] for r in raw),
        clamped=sum(r["clamped"] for r in raw), dropped=sum(r["dropped"] for r in raw),
        exhausted=sum(r["exhausted"] for r in raw),
        worst=max(r["worst"] for r in raw), runs=len(raw),
        far=sum(r["place"]["far_from_any_chosen"] for r in raw),
        place={k: [sum(r["place"][k][0] for r in raw), sum(r["place"][k][1] for r in raw)]
               for k in ("answered", "timed_out")})
    return reps, info, agg, raw


# ---------------------------------------------------------------------------
# F: the cross-fitted search
# ---------------------------------------------------------------------------

def search_f(pool, donor, p, wl, seeds, restricted, fr, cache):
    """Deterministic coordinate ascent from T37r, three sweeps, ties to the earliest.

    The ascent is over the selection half only, and its criterion is the early-record
    gain against the seven arms of the same half.  A member whose poll count falls
    outside the arms' range has no gain at all under evidence/14's rule, and is ranked
    below every member that has one.
    """
    def ev(mem):
        key = (donor, p, wl, tuple(seeds), mem)
        if key in cache:
            return cache[key]
        res = pool.map(job_member, [(donor, p, s, wl, mem) for s in seeds], chunksize=1)
        by = {r[0][3]: r[1] for r in res}
        mat = seed_mat([by[s] for s in seeds], 1, 4)
        g, _b, v = fr.gain(mat)
        cache[key] = dict(gain=g, polls=v["polls_sent"], delay=v["delay"], reps=by,
                          worst=max(r[2] for r in res))
        return cache[key]

    def rank(mem):
        g = ev(mem)["gain"]
        return -float("inf") if g is None else g

    cur = START_MEMBER
    rank(cur)
    order = [cur]
    for _sweep in range(3):
        for c in range(5):
            vals = coord_values(cur, c, restricted)
            cands = []
            for val in vals:
                mem = cur[:c] + (val,) + cur[c + 1:]
                if mem[2] < mem[0]:
                    continue        # i_max is a value at or above i_min, by the table
                cands.append(mem)
            best, bestr = None, None
            for mem in cands:
                r = rank(mem)
                if mem not in order:
                    order.append(mem)
                if best is None or r > bestr:
                    best, bestr = mem, r
            cur = best
    return cur, order


def f_stage(pool, donor, p, wl, arm_mats, cap, cache):
    """F_restricted, F_open and F_reach for one cell and workload, cross fitted.

    Fold A selects on seeds 1-10 and is scored on seeds 11-20, fold B the other way
    round, and the 20 out-of-sample per-seed values are pooled.  That pooled value is
    the only F the reading rule reads.
    """
    out = {}
    for variant in ("restricted", "open", "reach"):
        out[variant] = dict(sel={}, scored={}, reps={})
    sel_members, evaluated = {}, {}
    for fold, seeds in FOLDS.items():
        fr = Frontier({n: arm_mats[n][[SEEDS.index(s) for s in seeds]] for n in FIXED}, p)
        for variant, restricted in (("restricted", True), ("open", False)):
            mem, order = search_f(pool, donor, p, wl, seeds, restricted, fr, cache)
            sel_members[(variant, fold)] = mem
            out[variant]["sel"][fold] = dict(
                member=mem, gain=cache[(donor, p, wl, tuple(seeds), mem)]["gain"],
                polls=cache[(donor, p, wl, tuple(seeds), mem)]["polls"],
                evaluated=len(order))
            if variant == "restricted":
                evaluated[fold] = order
        # F_reach: the best restricted member whose poll count on the selection half is
        # at or below the reachability cap, taken from the same search.
        pool_members = [m for m in evaluated[fold]
                        if cap is not None
                        and cache[(donor, p, wl, tuple(seeds), m)]["polls"] <= cap
                        and cache[(donor, p, wl, tuple(seeds), m)]["gain"] is not None]
        if pool_members:
            mem = max(pool_members,
                      key=lambda m: cache[(donor, p, wl, tuple(seeds), m)]["gain"])
            sel_members[("reach", fold)] = mem
            out["reach"]["sel"][fold] = dict(
                member=mem, gain=cache[(donor, p, wl, tuple(seeds), mem)]["gain"],
                polls=cache[(donor, p, wl, tuple(seeds), mem)]["polls"],
                evaluated=len(pool_members))
        else:
            # 2026-11-19 verifier fix: record why this fold selected no F_reach member.
            # The members come from the restricted ascent's own path, so a fold whose
            # visited members all poll above the cap selects nothing and F_reach is then
            # missing for the whole cell, although V has a reachable target set there.
            seen_polls = [cache[(donor, p, wl, tuple(seeds), m)]["polls"]
                          for m in evaluated[fold]
                          if cache[(donor, p, wl, tuple(seeds), m)]["polls"] is not None]
            out["reach"].setdefault("empty", {})[fold] = dict(
                cap=cap, visited=len(evaluated[fold]),
                best=min(seen_polls) if seen_polls else None)
    # score each fold's member on the other fold
    for variant in ("restricted", "open", "reach"):
        for fold, seeds in FOLDS.items():
            other = "B" if fold == "A" else "A"
            mem = sel_members.get((variant, other))
            if mem is None:
                continue
            key = (donor, p, wl, tuple(seeds), mem)
            if key not in cache:
                res = pool.map(job_member, [(donor, p, s, wl, mem) for s in seeds],
                               chunksize=1)
                by = {r[0][3]: r[1] for r in res}
                cache[key] = dict(gain=None, polls=None, delay=None, reps=by,
                                  worst=max(r[2] for r in res))
            out[variant]["scored"][fold] = mem
            out[variant]["reps"].update(cache[key]["reps"])
    return out, sel_members


def strip(reps):
    """A member's or V's per-replicate records in the shape the cell tables pool."""
    return [(r[0], r[1], r[3], r[4]) for r in reps]


def pooled_cens(reps_by_seed, seeds):
    n = sum(r[2][0] for s in seeds for r in reps_by_seed[s])
    tot = sum(r[2][1] for s in seeds for r in reps_by_seed[s])
    return n, tot


# ---------------------------------------------------------------------------
# The reading rule, fixed in spec.md before any number of this step was computed
# ---------------------------------------------------------------------------

def read_outcome(vmax, vatf_diff, v_defined, vatf_exists):
    """The outcome word of one cell and workload, computed mechanically.

    vmax: (point, lo, hi) of V_max's interval in gain points, or None.
    vatf_diff: (point, lo, hi) of the paired `V_at_F - F` interval, or None.
    """
    reasons = []
    if not v_defined:
        return "undecided", ["V is not defined for that workload"]
    if vmax is None or vmax[1] is None:
        return "undecided", ["V_max has no interval"]
    _g, lo, hi = vmax
    if hi <= H - REPLAY_MARGIN:
        return "clearly none", []
    if lo > H:
        if not vatf_exists or vatf_diff is None or vatf_diff[1] is None:
            return "undecided", ["V_at_F does not exist because F's budget is outside "
                                 "V's reachable set"]
        _d, dlo, dhi = vatf_diff
        if dhi <= HD:
            return "the formula already takes it", []
        if dlo > HD:
            return "clearly large headroom", []
        return "undecided", ["`V_at_F - F`'s interval straddles Hd"]
    if lo <= H <= hi:
        reasons.append("V_max's interval straddles H")
    if lo <= H - REPLAY_MARGIN <= hi:
        reasons.append("V_max's interval straddles `H - 6`")
    if not reasons:
        reasons.append(f"V_max's interval lies between `H - 6` and H "
                       f"({lo:+.1f} .. {hi:+.1f} points)")
    return "undecided", reasons


def fnum(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def fpts(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{x:+.{nd}f}"


def ci_txt(lo, hi, n):
    if lo is None:
        return f"- ({n}/{BOOT_N} resamples inside the frontier)"
    tail = "" if n >= BOOT_N else f" (from {n}/{BOOT_N} resamples inside the frontier)"
    return f"{lo:+.1f} .. {hi:+.1f}{tail}"


# ---------------------------------------------------------------------------
# The run: C, V, F, the reading rule and the three files
# ---------------------------------------------------------------------------

def run_all():
    t_all = time.time()
    import multiprocessing as mp
    stage_t = {}
    with mp.Pool(WORKERS) as pool:
        t = time.time()
        per_seed, records, cset, cens_base, worst_base = run_base(WORKLOADS, pool)
        stage_t["arms and the evidence/15 rules"] = time.time() - t
        cell_rows = base_cell_rows(per_seed, records, WORKLOADS)
        w0 = [r for r in cell_rows if r["workload"] == "W0"]
        n_w0, bad = compare_published(w0)
        global W0_CHECKED
        W0_CHECKED = n_w0 - len(bad), n_w0
        print(f"check 1 W0 byte-exact: {n_w0 - len(bad)}/{n_w0} identical", flush=True)
        if bad:
            for b in bad:
                print("  ", b)
            sys.exit("check 1 FAILED inside the run")

        # 2026-11-19 verifier fix: check 8 used to run before `run_all` was entered, so
        # 63 member simulations were read before check 1 was evaluated.  spec.md's check
        # 1 says nothing else is read before it passes, so check 8 is evaluated here.
        check8()

        dp_per, sim_per = check7()
        caps = reach_caps()
        print("reachability caps (min candidates over 20 seeds, median, 70 % cap):",
              flush=True)
        for k in sorted(caps):
            print(f"  {k[0]} p{k[1]} {k[2]}: {caps[k][0]} {caps[k][1]} "
                  f"{caps[k][2]:.1f}")

        ladder = price_ladder(pool, HEADLINE[0], HEADLINE[1], HEADLINE[2])
        gain_rows, vrows, cinfo, outcomes, fsel, vinfo = [], [], {}, {}, {}, {}
        alt_outcomes, reach_empty, vknows = {}, {}, None
        cache = {}
        for wl in WORKLOADS:
            for (donor, p) in CELLS:
                key = (donor, p, wl)
                cap = caps[key][2]
                arm_mats = {n: seed_mat([per_seed[(wl, donor, p, n, s)] for s in SEEDS],
                                        1, 4) for n in FIXED}
                all_mats = {n: seed_mat([per_seed[(wl, donor, p, n, s)] for s in SEEDS],
                                        0, 7) for n in FIXED}
                fr = Frontier(arm_mats, p)
                fr_all = Frontier(all_mats, p)

                # ---- F ------------------------------------------------
                t = time.time()
                fout, _sel = f_stage(pool, donor, p, wl, arm_mats, cap, cache)
                stage_t.setdefault("F search and scoring", 0.0)
                stage_t["F search and scoring"] += time.time() - t
                fres = {}
                for variant in ("restricted", "open", "reach"):
                    if len(fout[variant]["scored"]) != 2:
                        continue
                    reps = fout[variant]["reps"]
                    mat = seed_mat([reps[s] for s in SEEDS], 1, 4)
                    mat_all = seed_mat([reps[s] for s in SEEDS], 0, 7)
                    g, b, v = fr.gain(mat)
                    ga, ba, va = fr_all.gain(mat_all)
                    per_fold_polls = {}
                    for fold, seeds in FOLDS.items():
                        m = seed_mat([reps[s] for s in seeds], 1, 4)
                        per_fold_polls[fold] = vals_of(m)["polls_sent"]
                    fres[variant] = dict(
                        gain=pct(g), boot=b, vals=v, all_gain=pct(ga), all_boot=ba,
                        all_vals=va, reps=reps, mat=mat, polls=v["polls_sent"],
                        fold_polls=per_fold_polls, sel=fout[variant]["sel"],
                        scored=fout[variant]["scored"],
                        cens=pooled_cens(reps, SEEDS))
                fsel[key] = fres
                reach_empty[key] = fout["reach"].get("empty")

                # ---- V ------------------------------------------------
                arm_polls = {n: float(arm_mats[n][:, 3].mean()) for n in FIXED}
                targets = []
                if wl != "WV":
                    for n in FIXED:
                        if arm_polls[n] <= cap:
                            targets.append((f"V@{n}", {f: arm_polls[n]
                                                       for f in FOLDS}, "arm"))
                f_for_vatf = None
                if wl != "WV" and "restricted" in fres:
                    rest = fres["restricted"]
                    reach_ok = all(v <= cap for v in rest["fold_polls"].values())
                    if reach_ok:
                        f_for_vatf = "restricted"
                    elif "reach" in fres:
                        f_for_vatf = "reach"
                    if f_for_vatf:
                        fp = fres[f_for_vatf]["fold_polls"]
                        targets.append((f"V@F_{f_for_vatf}", dict(fp), "F"))
                t = time.time()
                vres = {}
                floorp = floor_price_polls(pool, donor, p, wl) if targets else {}
                for tname, tgt, kind in targets:
                    reps, info, agg, raw = run_v_target(pool, donor, p, wl, tgt,
                                                        keep=True)
                    mat = seed_mat([reps[s] for s in SEEDS], 1, 4)
                    mat_all = seed_mat([reps[s] for s in SEEDS], 0, 7)
                    g, b, v = fr.gain(mat)
                    ga, ba, va = fr_all.gain(mat_all)
                    reached = max(info[f]["bisect_rel"] for f in FOLDS)
                    tmean = sum(tgt.values()) / len(tgt)
                    rel_real = abs(v["polls_sent"] - tmean) / tmean
                    vres[tname] = dict(gain=pct(g), boot=b, vals=v, all_gain=pct(ga),
                                       all_vals=va, all_boot=ba, reps=reps, info=info,
                                       agg=agg,
                                       kind=kind, target=tgt, rel=reached,
                                       rel_realised=rel_real, target_mean=tmean,
                                       floor_polls=dict(floorp),
                                       out_of_reach=[f for f in sorted(FOLDS)
                                                     if floorp[f] < tgt[f]],
                                       cens=pooled_cens(reps, SEEDS))
                    for r in raw:
                        _d, _p, _wl, seed, rep = r["key"]
                        if rep or r["instants"] is None:
                            continue
                        for pr, inst in sorted(r["instants"].items()):
                            vrows.append(dict(
                                workload=wl, donor=donor, p=p, target=tname,
                                seed=seed, rep=rep, receiver=pr[0], holder=pr[1],
                                polls=len(inst),
                                instants_ms_after_t0=" ".join(f"{x:.1f}"
                                                              for x in inst)))
                stage_t.setdefault("V bisection and final", 0.0)
                stage_t["V bisection and final"] += time.time() - t
                vinfo[key] = vres
                if key == HEADLINE and f"V@{VKNOWS_ARM}" in vres:
                    vknows = v_knows_illustration(
                        donor, p, wl, vres[f"V@{VKNOWS_ARM}"]["info"]["A"]["lam"],
                        arm=VKNOWS_ARM)

                # ---- C ------------------------------------------------
                nkeep = sum(cset[(wl, donor, p, s)][0] for s in SEEDS)
                ndrop = sum(cset[(wl, donor, p, s)][1] for s in SEEDS)
                sdmin = sum(cset[(wl, donor, p, s)][2] for s in SEEDS)
                cinfo[key] = dict(n=nkeep, drop=ndrop, dmin=sdmin / nkeep)

                # ---- the rows ----------------------------------------
                for name, kind, _spec in RULE_SPECS:
                    reps = {s: per_seed[(wl, donor, p, name, s)] for s in SEEDS}
                    mat = seed_mat([reps[s] for s in SEEDS], 1, 4)
                    mat_all = seed_mat([reps[s] for s in SEEDS], 0, 7)
                    g, b, v = fr.gain(mat)
                    ga, ba, va = fr_all.gain(mat_all)
                    n = sum(x[0] for s in SEEDS for x in cens_base[(wl, donor, p,
                                                                    name, s)])
                    tot = sum(x[1] for s in SEEDS for x in cens_base[(wl, donor, p,
                                                                      name, s)])
                    gain_rows.append(gain_row(wl, donor, p, name,
                                              O.interval_text(name), v, va,
                                              pct(g), b, pct(ga), ba,
                                              cinfo[key]["dmin"], tot / n if n else None,
                                              fr, kind == "arm" or kind == "fixed"))
                for tname, d in vres.items():
                    n, tot = d["cens"]
                    gain_rows.append(gain_row(
                        wl, donor, p, tname,
                        f"clairvoyant schedule, target {d['kind']}, "
                        f"price {d['info']['A']['lam']:.1f}/{d['info']['B']['lam']:.1f} "
                        f"ms per poll", d["vals"], d["all_vals"], d["gain"], d["boot"],
                        d["all_gain"], d["all_boot"], cinfo[key]["dmin"],
                        tot / n if n else None, fr, False))
                for variant, d in fres.items():
                    n, tot = d["cens"]
                    sel = "; ".join(f"{f}: {member_text(d['sel'][f]['member'])}"
                                    for f in sorted(d["sel"]))
                    gain_rows.append(gain_row(
                        wl, donor, p, f"F_{variant}", sel, d["vals"], d["all_vals"],
                        d["gain"], d["boot"], d["all_gain"], d["all_boot"],
                        cinfo[key]["dmin"], tot / n if n else None, fr, False))
                    cell_rows.append(cell_row(
                        wl, donor, p, f"F_{variant}", "F",
                        [strip(d["reps"][s]) for s in SEEDS],
                        [records[(wl, donor, p, s)] for s in SEEDS], settings=sel))
                for tname, d in vres.items():
                    cell_rows.append(cell_row(
                        wl, donor, p, tname, "V",
                        [strip(d["reps"][s]) for s in SEEDS],
                        [records[(wl, donor, p, s)] for s in SEEDS],
                        settings=(f"clairvoyant schedule, target {d['kind']}, price "
                                  f"{d['info']['A']['lam']:.1f}/"
                                  f"{d['info']['B']['lam']:.1f} ms per poll")))

                # ---- the reading rule --------------------------------
                outcomes[key] = read_cell(key, fres, vres, f_for_vatf, fr, cap)
                alt_outcomes[key] = read_cell(key, fres, vres, f_for_vatf, fr, cap,
                                              rel_key="rel_realised")
                o = outcomes[key]
                print(f"  {wl} {donor} p{p}: {o['outcome']}  "
                      f"V_max {fpts(o['vmax'][0]) if o['vmax'] else '-'} "
                      f"[{ci_txt(*(o['vmax'][1:] + (BOOT_N,))) if o['vmax'] else '-'}] "
                      f"F {fpts(fres['restricted']['gain']) if 'restricted' in fres else '-'}",
                      flush=True)

    gate_check4(vinfo)
    worst_c = max([worst_base]
                  + [c["worst"] for c in cache.values()]
                  + [x["agg"]["worst"] for d in vinfo.values() for x in d.values()])
    print(f"check 6 D_min at or below every record it bounds: the largest value of "
          f"D_min minus the realised delay, over every simulation of this step, is "
          f"{worst_c:+.3f} ms; a positive number would mean C is wrong", flush=True)
    if worst_c > 0:
        sys.exit("check 6 FAILED: D_min is above a record's realised delay")
    wall = time.time() - t_all
    write_outputs(cell_rows, gain_rows, vrows, cinfo, outcomes, fsel, vinfo, caps,
                  worst_c, dp_per, sim_per, stage_t, wall, ladder, alt_outcomes,
                  reach_empty, vknows)
    print(f"wall clock {wall:.1f} s")
    for k, v in stage_t.items():
        print(f"  {k}: {v:.1f} s")


def compare_published(w0_rows):
    """The W0 rows against ../14/workload_cells.csv and ../15/onoff_cells.csv."""
    bad, n = [], 0
    for path, only in ((os.path.join(EV15, "onoff_cells.csv"), None),
                       (os.path.join(EV14, "workload_cells.csv"), O.SHARED_RULES)):
        ref, cols = read_cells(path)
        for r in w0_rows:
            if only is not None and r["rule"] not in only:
                continue
            if (r["workload"], r["donor"], r["p"], r["rule"]) not in ref:
                continue
            want = ref[(r["workload"], r["donor"], r["p"], r["rule"])]
            diffs = {c: (O.csv_str(r[c]), want[c]) for c in cols
                     if c in r and O.csv_str(r[c]) != want[c]}
            n += 1
            if diffs:
                bad.append((os.path.basename(path), r["donor"], r["p"], r["rule"],
                            diffs))
    return n, bad


def gain_row(wl, donor, p, name, settings, v, va, gain, boot, all_gain, all_boot,
             dmin, dcens, fr, is_arm):
    lo, hi, nb = ci_pts(boot) if boot is not None else (None, None, 0)
    alo, ahi, anb = ci_pts(all_boot) if all_boot is not None else (None, None, 0)
    return dict(
        workload=wl, donor=donor, p=p, member=name, settings=settings,
        polls_sent=v["polls_sent"] if v else None,
        early_delay=v["delay"] if v else None,
        early_cens4=v["cens4"] if v else None,
        all_delay=va["delay"] if va else None,
        all_cens4=va["cens4"] if va else None,
        gain_pts=None if is_arm else gain, gain_lo=None if is_arm else lo,
        gain_hi=None if is_arm else hi, boot_n_gain=0 if is_arm else nb,
        all_gain_pts=None if is_arm else all_gain,
        all_gain_lo=None if is_arm else alo, all_gain_hi=None if is_arm else ahi,
        boot_n_all_gain=0 if is_arm else anb,
        c_dmin_ms=dmin, c_delay_cens_ms=dcens,
        c_gain_pts=None if not dcens else 100.0 * (1.0 - dmin / dcens),
        frontier_monotone=int(fr.mono))


OUT_OF_REACH_TEXT = ("the target is above V's poll count at the lowest price, so no "
                     "price reaches it")
STEPPED_OVER_TEXT = ("the bisection missed its budget by more than "
                     f"{100 * BUDGET_TOL:.0f} %, with the target inside V's reach at "
                     "the lowest price")


def miss_class_text(d):
    """Why one V target missed its budget, in the two classes the md reports.

    2026-11-19 verifier fix: the published text said "the bisection missed" for every
    target. Where V's poll count at the bracket floor price is already below the
    target, no price in the bracket reaches it and the bisection had nothing to find.
    `out_of_reach` holds the folds where that is so. This is wording only: which
    targets are dropped is the pre-registered rule and is untouched.
    """
    return OUT_OF_REACH_TEXT if d["out_of_reach"] else STEPPED_OVER_TEXT


def read_cell(key, fres, vres, f_for_vatf, fr, cap, rel_key="rel"):
    """Everything spec.md's reading rule reads for one cell and workload.

    `rel_key` picks which measurement of the budget miss decides whether a target is
    dropped.  "rel" is the bisection's own miss on its five bisection seeds, which is
    what spec.md's sentence about the bisection names and what the published reading
    uses.  "rel_realised" is the miss of V's realised polls per run over the 20 scored
    seeds, which is the other reading of the same sentence; it is computed so that a
    reader can see it without rerunning anything, and it enters no published outcome.
    """
    arm_targets = {k: d for k, d in vres.items() if d["kind"] == "arm"}
    dropped = [k for k, d in arm_targets.items() if d[rel_key] > BUDGET_TOL]
    usable = {k: d for k, d in arm_targets.items()
              if d[rel_key] <= BUDGET_TOL and d["gain"] is not None}
    v_defined = bool(arm_targets)
    vmax, vmax_at = None, None
    if usable:
        vmax_at = max(usable, key=lambda k: usable[k]["gain"])
        d = usable[vmax_at]
        lo, hi, _n = ci_pts(d["boot"])
        vmax = (d["gain"], lo, hi)
    vatf, diff, vatf_note = None, None, "none"
    tname = f"V@F_{f_for_vatf}" if f_for_vatf else None
    if tname and tname in vres and f_for_vatf in fres:
        d, fd = vres[tname], fres[f_for_vatf]
        lo, hi, _n = ci_pts(d["boot"])
        vatf = (d["gain"], lo, hi, d[rel_key])
        if d[rel_key] <= BUDGET_TOL_WIDE and d["gain"] is not None:
            b = d["boot"] - fd["boot"]
            dlo, dhi, _dn = ci_pts(b)
            diff = (d["gain"] - fd["gain"], dlo, dhi)
            vatf_note = f_for_vatf
        else:
            vatf_note = f"{f_for_vatf} (budget missed by {100 * d[rel_key]:.1f} %)"
    outcome, reasons = read_outcome(vmax, diff, v_defined, diff is not None)
    if vmax is None and dropped:
        reasons = ["every V target of this cell and workload was dropped: "
                   + "; ".join(f"{k}, where {miss_class_text(vres[k])}"
                               for k in sorted(dropped))]
    f = fres.get("restricted")
    flo, fhi, _fn = ci_pts(f["boot"]) if f else (None, None, 0)
    return dict(outcome=outcome, reasons=reasons, vmax=vmax, vmax_at=vmax_at,
                n_targets=len(usable), dropped=dropped, vatf=vatf, diff=diff,
                vatf_note=vatf_note, f_used=f_for_vatf, cap=cap,
                vatf_target=(vres[tname] if tname and tname in vres else None),
                f_vatf_gain=(fres[f_for_vatf]["gain"]
                             if f_for_vatf and f_for_vatf in fres else None),
                f_gain=f["gain"] if f else None, f_lo=flo, f_hi=fhi,
                f_detectable=bool(f and flo is not None and flo > 0
                                  and f["gain"] >= H),
                f_open=fres["open"]["gain"] if "open" in fres else None)


def write_csv(path, rows):
    with open(os.path.join(HERE, path), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: O.csv_str(v) for k, v in r.items()})


def write_outputs(cell_rows, gain_rows, vrows, cinfo, outcomes, fsel, vinfo, caps,
                  worst_c, dp_per, sim_per, stage_t, wall, ladder, alt_outcomes,
                  reach_empty, vknows):
    write_csv("bound_cells.csv", cell_rows)
    write_csv("bound_gain.csv", gain_rows)
    write_csv("bound_v.csv", vrows if vrows else [dict(note="no V target was reachable")])
    L = []
    A_ = L.append
    A_("# ev18 bound: the structural ceiling C, the clairvoyant point V, and the "
       "formula point F\n")
    A_(f"Generated by `bound.py run`, {WORKERS} workers, wall clock {wall:.0f} s "
       f"({', '.join(f'{k} {v:.0f} s' for k, v in stage_t.items())}).\n")
    A_("Everything below is **simulated against simulated**. No number of this step has "
       "a testbed measurement behind it. The schedules V produces lie far outside every "
       "schedule the replay was validated on, so evidence/16's 6 % fidelity does not "
       "transfer to V; it transfers to F only for members whose intervals stay at or "
       "above 2.2 s, which is why the headline F is F_restricted.\n")
    A_("C is an upper bound on the gain of any policy at any budget and is expected to "
       "be loose. V is a lower bound on what a policy with perfect foresight could "
       "reach and bounds nothing from above. F is a lower bound on what a formula can "
       "reach, inside one family and along one search path.\n")

    A_("## Checks\n")
    A_("| check | what it asserts | result |")
    A_("|---|---|---|")
    A_(f"| 1 | W0 reproduces ../14/workload_cells.csv and ../15/onoff_cells.csv "
       f"through this step's subclass | {W0_CHECKED[0]}/{W0_CHECKED[1]} "
       f"(cell x rule x file) identical on every printed column |")
    A_("| 2 | an inert F member reproduces the fixed arm of its interval event for "
       "event | see `check2` in RUN.txt |")
    A_("| 3 | `k_cap` is per pair | see `check3` in RUN.txt |")
    A_(f"| 4 | V's realised polls land within 100 ms of the DP's instants | "
       f"{place_text(vinfo)} |")
    A_("| 5 | V's tick lag is a cached generator per pair | see `check5` in RUN.txt |")
    A_(f"| 6 | `D_min` is at or below every record it bounds | the largest value of "
       f"D_min minus the realised delay, over every simulation of this step, is "
       f"{worst_c:+.3f} ms; a positive number would mean C is wrong |")
    A_(f"| 7 | the DP is timed before the cell list runs | {1000 * dp_per:.2f} ms per "
       f"run-solve of eight pairs, against {1000 * sim_per:.1f} ms for one replay "
       f"simulation |")
    A_("| 8 | the timeout coordinate is inert where no poll times out | "
       "see `check8` in the run log |")
    A_("| 9 | a rerun gives byte-identical csv | see RUN.txt |")
    A_("")
    A_(f"Check 4 split as spec.md asks, {place_split_text(vinfo)}.\n")
    A_("Two things about check 4 that the counts above do not show. The timed-out "
       f"column rests on {place_totals(vinfo)['tmo'][0]} resolutions in the whole "
       "step, so it asserts nothing about "
       "where V's polls land after a timeout. The write that half of the check was "
       "meant to protect is `nd.ival[nb] = g` under V, which is the only thing that "
       "places a poll after a timeout, because the base method leaves `ival` alone "
       "there. V's schedules do not exercise it: a verifier removed that line and "
       "check 4 still passed, because V places no poll where a poll times out.\n")

    A_("## The reachable target set\n")
    A_("An arm is a reachable target for V only when its polls per run is at or below "
       f"{100 * REACH_FRAC:.0f} % of the minimum candidate count over the 20 seeds of "
       "that cell and workload. The minimum is used because V has to be computable on "
       "every seed the bootstrap resamples.\n")
    A_("| cell | workload | min candidates | median | cap | DP candidates per run | "
       "reachable arms |")
    A_("|---|---|---|---|---|---|---|")
    for wl in WORKLOADS:
        for (donor, p) in CELLS:
            c = caps[(donor, p, wl)]
            d_all = vinfo.get((donor, p, wl), {})
            arms = [k for k, d in d_all.items() if d["kind"] == "arm"]
            dpc = [x["agg"]["ncand"] / x["agg"]["runs"] for x in d_all.values()]
            dpt = ("-" if not dpc else
                   (f"{min(dpc):.1f}" if max(dpc) - min(dpc) < 0.05
                    else f"{min(dpc):.1f} to {max(dpc):.1f}"))
            A_(f"| {donor} p{p} | {wl} | {c[0]} | {c[1]} | {c[2]:.1f} | {dpt} | "
               f"{', '.join(arms) if arms else 'none, so V is not computed'} |")
    A_("")
    A_("The cap rests on spec.md's own candidate count, which is the creation instants "
       "of the origin plus the link_up instants of the pair, summed over the eight "
       f"pairs ({100 * REACH_FRAC:.0f} % of its minimum over the 20 seeds). The DP's "
       "own candidate set is smaller than that count, because above hop 1 the DP uses "
       "the realised "
       "availability at the holder, which arrives in batches, and pushes every instant "
       "forward to a feasible one. The measured column above is that set, averaged "
       "over the runs of each V target of the cell and given as a range where the "
       "targets differ. A target this table lists as reachable can therefore still be "
       "out of V's reach at every price; the section \"Why most budget targets were "
       "missed\" says which ones are.\n")

    A_("## C, the structural ceiling\n")
    A_("`D_min` propagates each record's creation time along its chain with zero "
       "discovery wait and zero fetch wait, gated only by the link timeline, adding the "
       f"engine's own {TRANSIT_MS:.0f} ms per hop. C's record set is every record "
       "created in the scorer's window and at or before t0+72 s whose floor reaches the "
       "receiver inside the window, and every policy's delay in this table is "
       "recomputed over that same set with a censored record counted at `we - created`. "
       "The headline frontier that V and F are read against is unchanged and stays "
       "evidence/14's uncensored one.\n")
    A_("| cell | workload | records in C's set | records dropped | mean D_min ms | "
       "C against the 10.043 s arm | C against T37r | C against F_restricted |")
    A_("|---|---|---|---|---|---|---|---|")
    for wl in WORKLOADS:
        for (donor, p) in CELLS:
            ci = cinfo[(donor, p, wl)]
            def cg(name):
                hit = [r for r in gain_rows
                       if (r["workload"], r["donor"], r["p"], r["member"])
                       == (wl, donor, p, name)]
                return fpts(hit[0]["c_gain_pts"]) + " %" if hit else "-"
            A_(f"| {donor} p{p} | {wl} | {ci['n']} | {ci['drop']} | "
               f"{ci['dmin']:.0f} | {cg('fix10.043')} | {cg('T37r')} | "
               f"{cg('F_restricted')} |")
    A_("")
    A_("Every other policy's C gain is in `bound_gain.csv`, column `c_gain_pts`. C is "
       "budget free, so it is loose by exactly the amount a finite budget costs. It is "
       "here because it is nearly free and because it is the only strict bound in the "
       "step.\n")

    A_("## V, the clairvoyant point\n")
    A_("| cell | workload | target | budget per fold | V's polls per run | the "
       "bisection missed by | V's realised polls off target by | early delay ms | gain "
       "| 95 % | candidates taken by the DP | clamped | taken by the provisional timer "
       "| post-timeout floor bound |")
    A_("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for wl in WORKLOADS:
        for (donor, p) in CELLS:
            for tname, d in sorted(vinfo.get((donor, p, wl), {}).items()):
                lo, hi, nb = ci_pts(d["boot"])
                a = d["agg"]
                A_(f"| {donor} p{p} | {wl} | {tname} | "
                   f"{d['target']['A']:.1f} / {d['target']['B']:.1f} | "
                   f"{d['vals']['polls_sent']:.1f} | {100 * d['rel']:.1f} % | "
                   f"{100 * d['rel_realised']:.1f} % | "
                   f"{d['vals']['delay']:.0f} | {fpts(d['gain'])} | "
                   f"{ci_txt(lo, hi, nb)} | {a['taken']}/{a['ncand']} "
                   f"({100.0 * a['taken'] / a['ncand']:.1f} %), {a['chosen']} of them "
                   f"driven | {a['clamped']} | "
                   f"{a['dropped']} | {a['nfloor']} |")
    A_("")
    ntarget = sum(len(d) for d in vinfo.values())
    nbis = sum(1 for d in vinfo.values() for x in d.values()
               if x["rel"] <= BUDGET_TOL)
    nreal = sum(1 for d in vinfo.values() for x in d.values()
                if x["rel_realised"] <= BUDGET_TOL)
    A_(f"The bisection met spec.md's 2 % tolerance on {nbis} of the {ntarget} targets. "
       f"V's realised polls per run over the 20 scored seeds are within 2 % of the "
       f"target on {nreal} of them. The published reading uses the first number, "
       "because that is the one spec.md's sentence about the bisection names; the "
       "second reading is printed under the reading rule and enters no outcome.\n")
    A_("### Why most budget targets were missed\n")
    A_("Added after the run, as a diagnostic of the step's own method. It changes no "
       "number above and no target entered or left the reading rule because of it. "
       "Every missed target falls into one of two classes:\n")
    A_("- **(a) out of V's reach at any price.** V's poll count at the floor of the "
       f"price bracket, {LAM_LO_MS:.0f} ms per poll, is already below the target on at "
       "least one fold. The count falls as the price rises, so on that fold no price "
       "in the bracket reaches the target and the bisection had nothing to find. The "
       "price is bisected per fold, so one such fold is enough to miss the target. The "
       "class column names the folds where this happened when the other fold could "
       "reach the target.\n")
    A_("- **(b) inside V's reach, stepped over by the staircase.** The floor count is "
       "at or above the target, and the bisection landed outside the "
       f"{100 * BUDGET_TOL:.0f} % tolerance anyway, because the poll count is a "
       "staircase in the price whose tread around that target is narrower than eight "
       "bisection steps can resolve.\n")
    A_("The floor count below is measured on the same five bisection seeds of each "
       "half the bisection itself runs on.\n")
    A_("| cell | workload | target | budget A / B | polls at the floor price A / B | "
       "the bisection missed by | class |")
    A_("|---|---|---|---|---|---|---|")
    n_a = n_b = 0
    for wl in WORKLOADS:
        for (donor, p) in CELLS:
            for tname, d in sorted(vinfo.get((donor, p, wl), {}).items()):
                if d["rel"] <= BUDGET_TOL:
                    continue
                oor = d["out_of_reach"]
                cls = ("(b) stepped over" if not oor
                       else "(a) out of reach" if len(oor) == len(FOLDS)
                       else f"(a) out of reach on fold {' and '.join(oor)}")
                n_a += bool(d["out_of_reach"])
                n_b += not d["out_of_reach"]
                A_(f"| {donor} p{p} | {wl} | {tname} | "
                   f"{d['target']['A']:.1f} / {d['target']['B']:.1f} | "
                   f"{d['floor_polls']['A']:.1f} / {d['floor_polls']['B']:.1f} | "
                   f"{100 * d['rel']:.1f} % | {cls} |")
    A_("")
    A_(f"So {n_a} of the {n_a + n_b} missed targets are out of V's reach at any price "
       f"and {n_b} are inside it and were stepped over. A target in class (a) is a "
       "limit of what V's candidate set can carry, and the reachability table's cap "
       "does not catch it, because that cap counts spec.md's candidate set and the "
       "DP's own set is smaller.\n")
    A_("For class (b), the staircase. One price values every poll of all eight pairs "
       "at once, and the records arrive in batches, so nearly every candidate flips "
       "from worth taking to not worth taking inside the same narrow band. Eight "
       f"bisection steps over a bracket from {LAM_LO_MS:.0f} to {LAM_HI_MS:.0f} ms "
       f"resolve {11.5 / 2 ** BISECT_STEPS * 100:.1f} % in the price, which is wider "
       f"than that band. Measured on {HEADLINE[0]} p{HEADLINE[1]} {HEADLINE[2]}, on "
       "the five bisection seeds of each half:\n")
    A_("| price ms per poll | V's polls per run, bisection seeds of fold A | of fold B |")
    A_("|---|---|---|")
    for lam, row in ladder:
        A_(f"| {lam:.0f} | {row['A']:.2f} | {row['B']:.2f} |")
    A_("")
    A_("The seven arms sit at 96, 159, 256, 424, 683, 1128 and 1744 polls per run. "
       "This is the reason the reading rule has so little to read, and it is a limit "
       "of the construction rather than a result.\n")
    A_("### What V knows\n")
    A_("Stated from the code, so that no reader takes V for a policy. V's schedule is "
       "solved from three things a node running the protocol does not have:\n")
    A_("- **Every record's creation instant.** `solve_v_run` hands the DP of hop 1 the "
       "creation instants of the workload itself, so V knows at hop 1 exactly when "
       "each record came into existence at the origin.\n")
    A_("- **The whole link timeline, in advance.** `feasible_instant` and `candidate_"
       "set` read every link_down and every link_up of the pair before placing an "
       "instant, so V never puts a poll into an outage and knows the first "
       "millisecond after each outage ends.\n")
    A_("- **The availability instant of every record at every upstream hop.** Hop h's "
       "candidate set and objective are built from `stored_at` of the hop above, that "
       "is from the realised instants at which the upstream receiver actually held "
       "each record under V's own schedule. So the polls of successive hops are "
       "chained: hop h+1 is told when hop h will hold the record and places its poll "
       "just after that. A record's discovery wait is then paid about once along the "
       "path, where a node that polls on an interval pays a wait at every hop.\n")
    if vknows:
        v_t = vknows
        A_(f"One measured record, from the V run at {v_t['donor']} p{v_t['p']} "
           f"{v_t['wl']} seed {v_t['seed']} replicate {v_t['rep']}, target "
           f"V@{v_t['arm']} at {v_t['lam']:.1f} ms per poll, beside the same record of "
           f"the same seed and replicate under the {ARM_OF[v_t['arm']]} s fixed arm. "
           f"The record is {v_t['origin']} seq {v_t['seq']}, picked as the lowest seq "
           "of that origin that both runs carry to every hop. Instants are seconds "
           f"after t0; the record was created at t0+{v_t['created'] / 1000:.3f} s.\n")
        A_("| hop | the instant this hop held it under V | the same under the "
           f"{ARM_OF[v_t['arm']]} s arm |")
        A_("|---|---|---|")
        for i, h in enumerate(v_t["hops"]):
            A_(f"| {h} | t0+{v_t['v'][i] / 1000:.3f} s | "
               f"t0+{v_t['a'][i] / 1000:.3f} s |")
        A_("")
    A_("The DP is optimal per hop over its candidate set, given the realised upstream. "
       "It is not optimal for the four-hop objective, and the realised schedule differs "
       "from the solved one by the tick lag and by the clamp. **V is a lower bound on "
       "the clairvoyant optimum and bounds nothing from above.**\n")

    A_("## F, the formula point\n")
    A_("| cell | workload | variant | fold A selects | fold B selects | polls per run "
       "of the two selections | in-sample mean gain | pooled out-of-sample gain | 95 % | "
       "in-sample optimism | pooled polls per run |")
    A_("|---|---|---|---|---|---|---|---|---|---|---|")
    for wl in WORKLOADS:
        for (donor, p) in CELLS:
            for variant, d in sorted(fsel[(donor, p, wl)].items()):
                lo, hi, nb = ci_pts(d["boot"])
                ins = [d["sel"][f]["gain"] for f in sorted(d["sel"])
                       if d["sel"][f]["gain"] is not None]
                ins_mean = 100 * sum(ins) / len(ins) if ins else None
                A_(f"| {donor} p{p} | {wl} | F_{variant} | "
                   f"{member_text(d['sel']['A']['member'])} | "
                   f"{member_text(d['sel']['B']['member'])} | "
                   f"{d['sel']['A']['polls']:.1f} / {d['sel']['B']['polls']:.1f} | "
                   f"{fpts(ins_mean)} | "
                   f"{fpts(d['gain'])} | {ci_txt(lo, hi, nb)} | "
                   f"{fpts(None if ins_mean is None else ins_mean - d['gain'])} | "
                   f"{d['vals']['polls_sent']:.1f} |")
    A_("")
    A_("The in-sample optimism is the in-sample mean minus the pooled out-of-sample "
       "value, column by column above. Only 20 outage schedules exist per cell, because "
       "the link timeline comes from the stored runs' `results.json`, so every F "
       "interval here rests on 20 schedules and not on more. **F_open never enters the "
       "reading rule**; it is printed so that a reader can see what the 2.2 s "
       "restriction cost.\n")

    miss = []
    for wl in WORKLOADS:
        for (donor, p) in CELLS:
            key = (donor, p, wl)
            if "reach" in fsel[key] or not vinfo.get(key):
                continue
            em = reach_empty.get(key) or {}
            for fold in sorted(em):
                e = em[fold]
                best = "no scored member" if e["best"] is None else f"{e['best']:.1f}"
                miss.append(
                    f"{donor} p{p} {wl}, fold {fold}: of the {e['visited']} members "
                    f"the restricted ascent visited, the fewest polls per run on the "
                    f"selection half is {best}, against a cap of {e['cap']:.1f}")
    if miss:
        A_("F_reach is missing for " + "; ".join(miss) + ". F_reach is drawn from the "
           "members the restricted ascent already visited, so a fold whose visited "
           "members all poll above the reachability cap selects none, and then F_reach "
           "is not scored for that cell at all, although V has a reachable target set "
           "there. The consequence is that outcomes 2 and 3 of the reading rule cannot "
           "be reached for that cell. It is listed in the deviations.\n")

    same_open = sum(1 for k, v in fsel.items()
                    if "open" in v and "restricted" in v
                    and v["open"]["sel"]["A"]["member"] == v["restricted"]["sel"]["A"]["member"]
                    and v["open"]["sel"]["B"]["member"] == v["restricted"]["sel"]["B"]["member"])
    A_(f"F_open selected the same member as F_restricted on both folds in "
       f"{same_open} of the {len(fsel)} cell and workload pairs, so along this search "
       "path the restriction to intervals at or above 2.2 s cost nothing: the ascent "
       "never wanted a shorter `i_min` or a shorter `i_timeout`.\n")

    A_("## The all-record gain beside the headline\n")
    A_("The headline is the early-record gain, over records created at or before "
       f"t0+{CUT_MS / 1000:.0f} s. The all-record gain is printed here and enters no "
       "outcome. The residual bias runs in one direction: V and F censor close to "
       "nothing while the 10.043 s arm censors several percent at p 0.4, `delay` is the "
       "mean over uncensored records only, and evidence/15 measured that this works in "
       "favour of the low-censoring rule. So the all-record gain is the flattering "
       "reading for V and F, and where it is larger than the headline that is the "
       "direction to suspect.\n")
    A_("| cell | workload | F_restricted headline | F_restricted all records | V_max's "
       "target headline | that target, all records |")
    A_("|---|---|---|---|---|---|")
    for wl in WORKLOADS:
        for (donor, p) in CELLS:
            o = outcomes[(donor, p, wl)]
            f = fsel[(donor, p, wl)].get("restricted")
            vt = vinfo[(donor, p, wl)].get(o["vmax_at"]) if o["vmax_at"] else None
            A_(f"| {donor} p{p} | {wl} | {fpts(f['gain']) if f else '-'} | "
               f"{fpts(f['all_gain']) if f else '-'} | "
               f"{fpts(vt['gain']) if vt else '-'} | "
               f"{fpts(vt['all_gain']) if vt else '-'} |")
    A_("")

    A_("## The reading, under the rule spec.md fixed before the numbers\n")
    A_(f"H = {H} points for a single gain, Hd = {HD} points for a difference of two "
       f"gains, and the replay's own margin is {REPLAY_MARGIN:.0f} points, which "
       "outcome 1 carries and outcomes 2 and 3 do not. The headline cell is "
       f"{HEADLINE[0]} p {HEADLINE[1]} with {HEADLINE[2]}.\n")
    A_("The `F` column of this table is F_restricted, which is the one the \"reported "
       "in every case\" line uses. `V_at_F - F` is the difference against the F variant "
       "V_at_F's budget came from, which is F_reach in three rows, so that variant's "
       "own gain has a column of its own. A `V_at_F` whose budget the bisection missed "
       f"by more than {100 * BUDGET_TOL:.0f} % carries the miss beside it.\n")
    A_("| cell | workload | V_max | its 95 % | targets | V_at_F | F_restricted | the F "
       "`V_at_F - F` is against | V_at_F - F | its 95 % | **outcome** | what decided "
       "it |")
    A_("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for wl in WORKLOADS:
        for (donor, p) in CELLS:
            o = outcomes[(donor, p, wl)]
            vm = o["vmax"]
            df = o["diff"]
            vatf = "-"
            if o["vatf"]:
                vatf = fpts(o["vatf"][0])
                if o["vatf"][3] > BUDGET_TOL:
                    vatf += f" (budget missed by {100 * o['vatf'][3]:.1f} %)"
            fused = ("-" if o["f_vatf_gain"] is None
                     else f"{fpts(o['f_vatf_gain'])} (F_{o['f_used']})")
            A_(f"| {donor} p{p} | {wl} | "
               f"{fpts(vm[0]) if vm else '-'} | "
               f"{ci_txt(vm[1], vm[2], BOOT_N) if vm else '-'} | "
               f"{o['n_targets']} | {vatf} | "
               f"{fpts(o['f_gain'])} | {fused} | {fpts(df[0]) if df else '-'} | "
               f"{ci_txt(df[1], df[2], BOOT_N) if df else '-'} | "
               f"**{o['outcome']}** | {decided_by(o)} |")
    A_("")
    for wl in WORKLOADS:
        for (donor, p) in CELLS:
            o = outcomes[(donor, p, wl)]
            A_(f"- **{donor} p{p} {wl}: {o['outcome']}.** {outcome_sentence(o)}"
               + vatf_sentence(o)
               + (f" Targets dropped for missing their budget by more than "
                  f"{100 * BUDGET_TOL:.0f} %: {', '.join(o['dropped'])}."
                  if o["dropped"] else ""))
    A_("")
    A_("### The same rule read on V's realised poll count\n")
    A_("spec.md says the price is bisected \"so that V's realised polls per run land "
       "within 2 % of the target's realised polls per run\", and it drops a target the "
       "bisection fails to reach. The published table above drops a target on the "
       "bisection's own miss, measured on its five bisection seeds. The table below "
       "drops it on the miss of V's realised polls per run over the 20 scored seeds "
       "instead. **It is not the reading rule and no outcome of this step comes from "
       "it.** It is here because the two numbers differ by more than the tolerance, and "
       "a reader should not have to rerun anything to see what that costs.\n")
    A_("| cell | workload | V_max | its 95 % | targets | V_at_F - F | its 95 % | "
       "outcome under this reading | published outcome |")
    A_("|---|---|---|---|---|---|---|---|---|")
    for wl in WORKLOADS:
        for (donor, p) in CELLS:
            a = alt_outcomes[(donor, p, wl)]
            o = outcomes[(donor, p, wl)]
            vm, df = a["vmax"], a["diff"]
            A_(f"| {donor} p{p} | {wl} | {fpts(vm[0]) if vm else '-'} | "
               f"{ci_txt(vm[1], vm[2], BOOT_N) if vm else '-'} | {a['n_targets']} | "
               f"{fpts(df[0]) if df else '-'} | "
               f"{ci_txt(df[1], df[2], BOOT_N) if df else '-'} | {a['outcome']} | "
               f"{o['outcome']} |")
    A_("")

    A_("### Reported in every case\n")
    A_("| cell | workload | F_restricted | its 95 % | interval entirely above zero and "
       "point at or above H | F_open (enters no outcome) |")
    A_("|---|---|---|---|---|---|")
    for wl in WORKLOADS:
        for (donor, p) in CELLS:
            o = outcomes[(donor, p, wl)]
            A_(f"| {donor} p{p} | {wl} | {fpts(o['f_gain'])} | "
               f"{ci_txt(o['f_lo'], o['f_hi'], BOOT_N)} | "
               f"{'yes' if o['f_detectable'] else 'no'} | {fpts(o['f_open'])} |")
    A_("")

    A_("## Deviations from spec.md\n")
    for i, text in enumerate(DEVIATIONS, 1):
        A_(f"{i}. {text}")
    A_("")
    with open(os.path.join(HERE, "bound.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def place_split_text(vinfo):
    """Check 4's counts for the cells with outages and for the cell without.

    spec.md's check 4 asks for the placement counts separately for the cells with
    outages and for the cell without, and the pooled line alone hides that the whole
    timed-out column comes from one outage cell.
    """
    parts = []
    for label, keep in (("the cells with outages", lambda k: k[:2] != NO_OUTAGE),
                        ("the cell without an outage", lambda k: k[:2] == NO_OUTAGE)):
        t = place_totals(vinfo, keep)
        if not t["n"]:
            parts.append(f"{label}: no V schedule was computed")
            continue
        parts.append(
            f"{label}: {t['ans'][1]}/{t['ans'][0]} armed instants off the chosen one by "
            f"more than 100 ms after an answered poll and {t['tmo'][1]}/{t['tmo'][0]} "
            f"after a timed-out poll, {t['clamped']} clamped, {t['dropped']} taken by "
            f"the provisional timer, {t['far']} realised polls more than 100 ms from "
            f"every chosen instant")
    return "; ".join(parts)


def place_text(vinfo):
    t = place_totals(vinfo)
    if not t["n"]:
        return "no V schedule was computed"
    return (f"{t['ans'][1]}/{t['ans'][0]} armed instants off the chosen one by more than "
            f"100 ms after an answered poll and {t['tmo'][1]}/{t['tmo'][0]} after a "
            f"timed-out poll; {t['clamped']} clamped and {t['dropped']} taken by the "
            f"provisional timer, which is {100.0 * (t['clamped'] + t['dropped']) / t['n']:.4f} % "
            f"of {t['n']} placements; {t['far']} realised polls landed more than 100 ms "
            f"from every instant the DP chose")


def place_totals(vinfo, keep=None):
    """Check 4's counters summed over the cells `keep` admits, or over all of them."""
    tot = dict(ans=[0, 0], tmo=[0, 0], clamped=0, dropped=0, far=0, n=0)
    for key, d in vinfo.items():
        if keep is not None and not keep(key):
            continue
        for x in d.values():
            a = x["agg"]
            for k, src in (("ans", "answered"), ("tmo", "timed_out")):
                tot[k][0] += a["place"][src][0]
                tot[k][1] += a["place"][src][1]
            tot["clamped"] += a["clamped"]
            tot["dropped"] += a["dropped"]
            tot["far"] += a["far"]
    tot["n"] = tot["ans"][0] + tot["tmo"][0]
    return tot


def vatf_sentence(o):
    """How V_at_F was budgeted, and whether the reading rule could use it."""
    if not o["vatf"]:
        return ""
    rel = 100 * o["vatf"][3]
    at = f"F_{o['f_used']}"
    # The outcome sentence of "clearly large headroom" already carries spec.md's phrase,
    # so it is not repeated here; the other outcomes say nothing about budgets and carry
    # it here instead.
    reach = (" at the budgets V can reach"
             if o["f_used"] == "reach" and o["outcome"] != "clearly large headroom"
             else "")
    tail = ""
    if o["f_used"] == "reach":
        tail = (f" F_restricted's own gain is {fpts(o['f_gain'])} beside F_reach's "
                f"{fpts(o['f_vatf_gain'])}, which is what leaving the budget range V "
                f"can reach does to the formula's gain.")
    if o["diff"] is None:
        why = (OUT_OF_REACH_TEXT if (o["vatf_target"]
                                     and o["vatf_target"]["out_of_reach"])
               else f"the bisection missed that budget by {rel:.1f} %")
        return (f" V was also solved at {at}'s budget, but {why}, so `V_at_F` does not "
                f"exist for the reading rule.{tail}")
    return (f" V_at_F was computed at {at}'s budget{reach}, which the bisection met to "
            f"{rel:.1f} %.{tail}")


def decided_by(o):
    if o["outcome"] == "clearly none":
        return (f"the upper end of V_max's interval, {o['vmax'][2]:+.1f}, is at or "
                f"below {H - REPLAY_MARGIN:.1f}")
    if o["outcome"] == "the formula already takes it":
        return (f"V_max's lower end {o['vmax'][1]:+.1f} is above H and the upper end of "
                f"`V_at_F - F`, {o['diff'][2]:+.1f}, is at or below Hd")
    if o["outcome"] == "clearly large headroom":
        return (f"V_max's lower end {o['vmax'][1]:+.1f} is above H and the lower end of "
                f"`V_at_F - F`, {o['diff'][1]:+.1f}, is above Hd")
    return "; ".join(o["reasons"])


def outcome_sentence(o):
    if o["outcome"] == "clearly none":
        return ("The best clairvoyant schedule this step could construct shows no gain "
                "this testbed can detect in this cell. For the three-way comparison: do "
                "not commission it for this workload; change the workload or the cell "
                "first.")
    if o["outcome"] == "the formula already takes it":
        return ("A formula already takes the gain that exists at that budget, within "
                "what the testbed could resolve. For the three-way comparison: the "
                "distilled LLM and the ML model are predicted to tie with F within Hd, "
                "so running it is a test of that prediction and tuning the formula is "
                "the cheaper move.")
    if o["outcome"] == "clearly large headroom":
        # spec.md's addendum: where V_at_F was budgeted at F_reach, the prize is read
        # at the budgets V can reach and not over F_restricted's own budget.
        at = (" at the budgets V can reach" if o["f_used"] == "reach" else "")
        return ("Headroom exists at that budget that this formula family does not take, "
                "and V is its best case. For the three-way comparison: commission it, "
                "score every arm of it against F, and read `V_at_F - F` as the size of "
                f"the prize{at}.")
    return ("This step gives the three-way comparison no steer for this cell. Nothing "
            "is claimed about any of the three decision makers. The honest report is "
            "that the replay cannot separate them here, and that a denser testbed or a "
            "tighter clairvoyant construction is what would.")


DEVIATIONS = [
    "**The bootstrap does not redo the coordinate ascent inside each resample.** "
    "spec.md asks for the arms, the selected members and the pooled value to be "
    "recomputed per resample. The arms and the pooled value are. The selection is not: "
    "2000 coordinate ascents per cell and workload would be about 300 million "
    "simulations, against the 166000 the whole step costs. Each seed's value is still "
    "an out-of-sample one, so the interval carries the variance of the scoring half but "
    "not the variance of the selection itself.",
    "**Every gain of this step is pooled from per-seed means**, on the early-record "
    "scoring and on the all-record one alike, through ../13/trickle.py's `seed_matrix`. "
    "evidence/15 pools all 100 runs for its all-record point estimate. For sums the two "
    "orders agree exactly; for the ratios they differ in the last digits.",
    "**The DP charges a record no poll reaches at t0+130 s**, the last instant a poll "
    "could be sent. spec.md fixes the objective but not the tail, and without a finite "
    "tail penalty the DP drops the end of the run for free.",
    "**A timeout leaves the consecutive-empty counter where it was**, and a non-empty "
    "reply resets it. spec.md says `k_cap` counts consecutive empty replies and does "
    "not say what a timeout does to that count.",
    "**The bracket ends are not evaluated, and the last two of the eight steps "
    "interpolate.** The poll count is monotone in the price, so an unevaluated end costs "
    "nothing and buys two more steps inside the bracket. Even so the tolerance was met "
    "on few targets; the counts are under the V table, the price ladder says why, and "
    "the consequence is that the reading rule has almost nothing to read.",
    "**`V_at_F`'s budget is the selected member's own poll count on the fold it is "
    "scored on**, one per fold, rather than F's pooled 20-seed count. spec.md adds "
    "\"F's pooled realised polls per run\" to the bisection targets and also says that "
    "when the two folds select members with different poll counts both are tried. The "
    "per-fold form is what makes `V_at_F` pair with F seed by seed, because each seed "
    "is then scored against V at the budget of the member F used on that seed.",
    "**\"Every V target was dropped\" is not one of spec.md's four named reasons for "
    "`undecided`.** spec.md lists four: V_max straddles H, V_max straddles `H - 6`, "
    "`V_at_F` does not exist, and V is not defined under WV. A cell where every target "
    "of the reachable set missed its budget is a fifth case, and it is the one most "
    "cells of this run fall into. It is reported under `undecided` by name.",
    "**The post-timeout floor is reported per cell, workload and target**, not per pair. "
    "spec.md asks for it per pair. It bound no transition anywhere, so the per-pair "
    "breakdown would be a table of zeroes.",
    "**`bound_v.csv` holds replicate 0 only.** All five replicates are solved and "
    "scored; writing every one of them would be 16800 timetable rows, and one "
    "replicate is what a fixed-timetable testbed run would take.",
    "**The reachability cap is counted as the scratchpad counted it**: per directed "
    "pair, the distinct creation instants of the origin whose chain it carries plus the "
    "link_up instants of that pair. That is the count spec.md's own table is stated "
    "against, and this step reproduces that table exactly. The DP's own candidate set "
    "differs from it above hop 1, where it uses the realised availability at the holder "
    "and pushes every instant forward to a feasible one.",
    "**A V target is the 20-seed cell value of the target's realised polls per run**, "
    "while the bisection measures V's polls on its own five bisection seeds. spec.md "
    "names the target as a cell-level number and runs the bisection on five seeds, so "
    "the two are measured on different seed sets and part of the residual gap is seed "
    "variation rather than a failure to bisect.",
    "**`i_timeout` and `k_cap` live in the rule dict as keys.** A member with "
    "`i_timeout` keep and `k_cap` None carries both keys and still runs through the base "
    "classes alone, which is what checks 1 and 2 measure.",
    "**Members with `i_max` below `i_min` are skipped.** spec.md's table gives `i_max` "
    "the arms' intervals at or above `i_min`, so a sweep of `i_min` past the current "
    "`i_max` would leave the family; those candidates are not evaluated.",
    "**F_reach is taken from the members the restricted search already evaluated.** "
    "spec.md says it is computed in the same search; a search of its own over the "
    "reachable sub-family would be a second ascent and a second search path. One "
    "consequence is that F_reach can be missing where spec.md's addendum expects it: "
    "the members the restricted ascent visited on one fold can all sit above the "
    "reachability cap, and then that fold selects nothing and F_reach is not scored. "
    "The F table names the cell where this happened and by how much the fold's best "
    "visited member missed the cap.",
    "**Check 4's gate excuses a realised poll far from every chosen instant when the "
    "clamp or the provisional timer moved at least as many polls** (`far > clamped + "
    "dropped` is what fails the gate). spec.md states the within-100 ms property with "
    "no such clause, so a run in which the clamp or the timer moved polls could pass "
    "the gate with realised polls far from every chosen instant. This run's own counts "
    "are printed under the check table.",
]


def check8(pool=None):
    """Where no poll times out, members that differ only in `i_timeout` are one member.

    `s5b10p043` p 0 has no outage, so the branch that writes the timeout coordinate is
    never reached and the coordinate cannot change a number.  A difference would mean
    the subclass touches a path spec.md says it does not.
    """
    t = time.time()
    donor, p = NO_OUTAGE
    base = (2.2, 2.0, 10.043, "keep", None)
    mems = [base] + [(2.2, 2.0, 10.043, x, None) for x in ARM_INTERVALS]
    seeds = (1, 2, 3)
    bad, done = [], 0
    for wl in WORKLOADS:
        ref = {}
        for mem in mems:
            for s in seeds:
                key, reps, _w = job_member((donor, p, s, wl, mem))
                if mem == base:
                    ref[s] = reps
                else:
                    done += 1
                    if reps != ref[s]:
                        bad.append((wl, mem, s))
    print(f"check 8 timeout coordinate inert in {donor} p{p}: {done - len(bad)}/{done} "
          f"(workload x i_timeout x seed) identical to `keep` on every per-replicate "
          f"count", flush=True)
    print(f"check 8 wall {time.time() - t:.1f} s")
    if bad:
        sys.exit("check 8 FAILED: i_timeout changes a number where no poll times out")


def gate_check4(vinfo):
    """spec.md's check 4, as a gate: V is placed, or the step stops.

    The gate is spec.md's own number: at most 1 % of V's polls may be bound by the clamp
    or taken by the provisional timer.  Beside it, no realised poll may land more than
    100 ms from every instant the DP chose for that pair unless the clamp or the timer
    moved it, which is the statement that V's schedule is the DP's schedule.
    """
    t = place_totals(vinfo)
    if not t["n"]:
        print("check 4: no V schedule was computed, so there is nothing to place")
        return
    print(f"check 4 V placement: {place_text(vinfo)}", flush=True)
    for (donor, p, wl), d in sorted(vinfo.items()):
        for tname, x in sorted(d.items()):
            a = x["agg"]
            print(f"    {donor} p{p} {wl} {tname}: answered {a['place']['answered']}, "
                  f"timed out {a['place']['timed_out']}, clamped {a['clamped']}, "
                  f"timer {a['dropped']}, far {a['far']}")
    if (t["clamped"] + t["dropped"]) > 0.01 * t["n"] or t["far"] > t["clamped"] + t["dropped"]:
        sys.exit("check 4 FAILED: V is not placed")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "run"
    if what == "check1":
        check1()
    elif what == "check2":
        check2()
    elif what == "check3":
        check3()
    elif what == "check5":
        check5()
    elif what == "check7":
        check7()
    elif what == "check8":
        check8()
    elif what == "run":
        run_all()
    else:
        sys.exit(f"unknown command {what}")


if __name__ == "__main__":
    main()
