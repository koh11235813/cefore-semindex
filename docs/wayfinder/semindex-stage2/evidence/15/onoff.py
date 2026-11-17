#!/usr/bin/env python3
"""ev15 spec: on/off workloads, and rules that also move their interval on a timeout.

    /home/lab_shared/cefore-semindex/.venv/bin/python3 -B onoff.py

Writes onoff_cells.csv (one row per cell x workload x rule), onoff_gain.csv (one row
per cell x workload x tested rule) and onoff.md.

WHY THIS EXISTS (spec.md)
  evidence/14 left two inferences untested.  (1) The lengthening rule with I_min
  3.695 s (T37) was read as winning only where "a record that just came makes another
  one likely", a property the testbed workload has as one 60 s block; a workload that
  switches generation on and off several times should then favour T37 over a fixed
  interval wherever there is no outage.  (2) T37's gain vanished in the outage cells,
  and the suspected reason is that T37 leaves its interval alone when a poll times out;
  if that is the reason, a rule that moves its interval on a timeout should recover
  part of the gain there.

WHAT THIS CAN AND CANNOT SHOW
  The replay (../13/replay.py) was checked against measurements for the testbed
  workload only (../13/validate.md).  The replaced workloads have no measurement
  behind them, so every number here is a simulated value on an unvalidated input.  It
  says which testbed runs are worth doing, not what a testbed would measure.

THE ONLY INPUTS REPLACED are run["created"] (the list of (ts_ms, origin, seq)) and the
poll-interval rule.  The outage timeline, agent_start, t0 and the topology stay the
donor run's own, and ../13/replay.py and ../13/trickle.py are imported unchanged.  The
one behaviour the simulator does not have, moving the rule's interval when the newest
poll times out, is added here as a subclass that overrides a single method; the W0
check below is the proof that the subclass changes nothing for a rule without the new
key, because it reproduces ../14/workload_cells.csv for every rule the two steps share.
"""

import csv
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV13 = os.path.normpath(os.path.join(HERE, "..", "13"))
EV14 = os.path.normpath(os.path.join(HERE, "..", "14"))
# EV14 goes in front of EV13 on purpose: both directories hold an `after.py`, and the
# one this step reuses is evidence/14's early-record scorer.  Neither directory holds a
# `replay.py`/`trickle.py` other than evidence/13's, so those still resolve to EV13.
sys.path.insert(0, EV13)
sys.path.insert(0, EV14)
import after as A  # noqa: E402  evidence/14's early-record scorer
import replay as R  # noqa: E402
import trickle as T  # noqa: E402
import workload as W  # noqa: E402  evidence/14's generators, gap counter and tables

assert W.__file__.startswith(EV14), W.__file__
assert A.__file__.startswith(EV14), A.__file__

SEEDS = W.SEEDS                     # 1..20, as ev13 and ev14
REPS = W.REPS                       # 5 Mode B replicates
WORKERS = 12                        # the task's ceiling
BOOT_N = W.BOOT_N
BOOT_SEED = W.BOOT_SEED
ORIGINS = W.ORIGINS                 # h0, h4: the two origins of every logged run
CELLS = W.CELLS                     # s6 p 0.2, s6 p 0.4 (outages), s5 p 0 (none)
NO_OUTAGE = ("s5b10p043", 0.0)      # the cell where no poll can time out
ZERO = W.ZERO                       # a gain within 1e-9 of zero counts as zero
CUT_MS = A.CUT_MS                   # the headline scores records created by t0+72 s

ON_RATE_S = 2.0                     # while ON, a tick every 2 s (the testbed's rate)
ON_BATCH = 4                        # records per tick (the testbed's batch size)
FIRST_ON_LO, FIRST_ON_HI = 13.0, 33.0
WO_ON_S = WO_OFF_S = 20.0
WV_MEAN_S, WV_LO_S, WV_HI_S = 20.0, 4.0, 60.0
GEN_END_S = 110.0                   # no record is created after t0+110 s

WORKLOADS = ["W0", "WO", "WV"]
WORKLOAD_TEXT = {
    "W0": "control: the logged creation times, untouched",
    "WO": "on/off regular: first ON at U[13, 33] s, then ON 20 s / OFF 20 s until 110 s",
    "WV": "on/off variable: first ON at U[13, 33] s, then ON and OFF lengths from an "
          "exponential with mean 20 s clipped to [4, 60] s, until 110 s",
}

# The seven fixed arms are evidence/14's, unchanged, so that the W0 control can be
# compared with ../14/workload_cells.csv arm by arm.
FIXED_SPECS = W.FIXED_SPECS
# The tested rules: short and wait as reference points, and the three I_min 3.695 s
# rules that differ only in what a timeout does.  T37's dict is evidence/13's own entry,
# so its settings text and its behaviour are literally ev13's rule.
TESTED_SPECS = [
    ("short", "policy", "short"),
    ("wait", "policy", "wait"),
    ("T37", "rule", dict(T.RULE_OF["T37"])),
    ("T37r", "rule", dict(T.RULE_OF["T37"], on_timeout="reset")),
    ("T37b", "rule", dict(T.RULE_OF["T37"], on_timeout="backoff")),
]
RULE_SPECS = FIXED_SPECS + TESTED_SPECS
NAMES = [r[0] for r in RULE_SPECS]
FIXED = [r[0] for r in FIXED_SPECS]
TESTED = [r[0] for r in TESTED_SPECS]
VARIANTS = ["T37r", "T37b"]          # the two rules inference 2 is about
BASE = "T37"                         # what the paired difference is taken against

# The rules this step shares with evidence/14, and whose W0 numbers must therefore come
# out identical: every arm, short, wait and T37.  T37r and T37b are new.
SHARED_RULES = [n for n in NAMES if n not in ("T37r", "T37b")]
# Every column ../14/workload_cells.csv prints, compared as the strings both scripts
# write.  This is spec.md's first check and nothing else is read before it passes.
W0_COLS = ["workload", "donor", "p", "rule", "kind", "settings", "records", "delay",
           "cens4", "polls_sent", "answered", "timed_out", "bytes", "empty_share",
           "mean_gap_s", "lw_gap_s", "polls_after_complete"]


def interval_text(name):
    """What a rule does, for the tables.

    The shared rules take evidence/14's wording verbatim, because the W0 check compares
    the `settings` column as a string: a reworded arm would fail a check that is there
    to catch a changed behaviour.
    """
    if name in SHARED_RULES:
        return W.interval_text(name)
    for n, _kind, spec in TESTED_SPECS:
        if n == name:
            what = {"reset": "I = I_min", "backoff": "I = I_max"}[spec["on_timeout"]]
            return (f"I_min {spec['i_min']} m {spec['m']} I_max {spec['i_max']} "
                    f"jf {spec['jf']}, on a timeout {what}")
    raise KeyError(name)


# --------------------------------------------------------------------------
# The simulator: one overridden step
# --------------------------------------------------------------------------

class Engine(R._Engine):
    """The evidence/13 engine with one addition: a timeout may move the interval.

    evidence/13's rule leaves its interval alone when a poll times out, and
    evidence/14 could not tell whether that is why the rule's gain disappears in the
    outage cells.  A rule dict may now carry `on_timeout`: "reset" sets I to I_min when
    the newest outstanding poll times out, "backoff" sets it to I_max, and a dict
    without the key keeps evidence/13's behaviour exactly.

    Only the interval is set here.  Everything after it -- the jitter draw, `applied`,
    `max(sent + applied, now)`, the reschedule line and the re-arm -- is the base
    class's own code, which is what spec.md asks for ("the next poll time is then
    computed as for an answered poll").

    The guard is repeated from the base method on purpose: `resolve` ignores a
    resolution that does not belong to this neighbor's newest poll (engine.py:503
    resolves `replies.polls[-1]` only).  Without repeating it, a late timeout of a
    superseded poll would move the interval although the base class then returns
    without rescheduling anything, and the rule would react to a poll the source never
    resolves.
    """

    def resolve(self, t, node, nb, sent_at, status, empty=None):
        nd = self.nodes[node]
        pend = nd.pending.get(nb)
        live = pend is not None and pend[0] == sent_at and not pend[2]
        if live and status == "timed_out" and nd.rule is not None:
            act = nd.rule.get("on_timeout")
            if act == "reset":
                nd.ival[nb] = nd.rule["i_min"]
            elif act == "backoff":
                nd.ival[nb] = nd.rule["i_max"]
        return super().resolve(t, node, nb, sent_at, status, empty=empty)


def sim_run(run, sim_rng):
    """One Mode B simulation through the subclass.  Every rule of this step uses it."""
    return Engine(run, "B", None, sim_rng).go()


# --------------------------------------------------------------------------
# The workloads
# --------------------------------------------------------------------------

def _ticks(lengths, start):
    """Tick instants in seconds for alternating ON/OFF `lengths` starting at `start`.

    An ON block of length L is half-open [b, b+L): ticks at b, b+2, ... while below
    b+L.  With spec.md's 20 s ON that is 10 ticks, the last one 18 s into the block,
    and the instant where ON ends is the first OFF instant rather than a tick.  A tick
    after t0+110 s is dropped, as spec.md says ("a tick is emitted only if its time is
    <= 110 s"), so generation stops at the window the scorer leaves room in.
    """
    out = []
    t = start
    on = True
    for length in lengths:
        if t > GEN_END_S:
            break
        if on:
            k = 0
            while k * ON_RATE_S < length:
                ts = t + k * ON_RATE_S
                if ts > GEN_END_S:
                    return out
                out.append(ts)
                k += 1
        t += length
        on = not on
    return out


def gen_created(name, p, seed, t0, logged):
    """The (ts_ms, origin, seq) list of workload `name` for one run.

    spec.md: a workload is a function of (p, seed, origin) only, so every rule and both
    donors see the same creation times on one (p, seed), and the two origins are drawn
    independently.  The donor is deliberately not in the seed string: that is what makes
    the three cells differ in their outages and in nothing else.  The seed string is
    `ev15:...`, so no workload of this step repeats an evidence/14 draw.
    """
    if name == "W0":
        return logged
    out = []
    for origin in ORIGINS:
        rng = random.Random(f"ev15:{name}:{p}:{seed}:{origin}")
        start = rng.uniform(FIRST_ON_LO, FIRST_ON_HI)
        if name == "WO":
            # A fixed square wave: the rule can in principle learn both the period and
            # the phase, which is the property evidence/14 named as missing from every
            # workload it ran.
            lengths = [WO_ON_S if i % 2 == 0 else WO_OFF_S
                       for i in range(2 * int(GEN_END_S // (WO_ON_S + WO_OFF_S)) + 4)]
        elif name == "WV":
            # The same alternation with unpredictable lengths, so that "a record just
            # came" still raises the chance of another one but the next switch instant
            # is not derivable.  Clipping keeps a block from being shorter than two
            # ticks or longer than the whole window.
            lengths = [min(max(rng.expovariate(1.0 / WV_MEAN_S), WV_LO_S), WV_HI_S)
                       for _ in range(2 * int(GEN_END_S / WV_LO_S) + 4)]
        else:
            raise KeyError(name)
        secs = [ts for ts in _ticks(lengths, start) for _ in range(ON_BATCH)]
        out += [(float(round(t0 + s * 1000.0)), origin, i)
                for i, s in enumerate(secs, 1)]
    out.sort()                      # load_run sorts its own list the same way
    return out


# --------------------------------------------------------------------------
# The two measurements this step adds to evidence/14's
# --------------------------------------------------------------------------

def nonempty_applied(ev, t0):
    """(sum of applied intervals in seconds, count) over polls answered non-empty.

    spec.md's "interval in force when a non-empty reply arrived": the direct reading of
    whether a rule is asking fast while data is flowing.  The attribution is the
    scorer's (../13/replay.py score, from evidence/05 attrib.py): a reply is charged to
    the latest state poll to that neighbor at or before its receive stamp, and only the
    POLL's own window membership decides whether it counts, so this averages over
    exactly the polls `answered` counts, restricted to the ones that brought something.
    """
    ws, we = t0 + R.WINDOW_START_MS, t0 + R.WINDOW_END_MS
    total, n = 0.0, 0
    for rows in ev.values():
        last = {}
        for e in rows:
            if e.get("event") == "decision" and e.get("kind") == "state_poll":
                a = e["action"]
                last[a["neighbor"]] = [e["ts_ms"], a["applied_interval_s"], False]
            elif e.get("event") == "state_rx":
                k = last.get(e["neighbor"])
                if k is not None and not k[2]:
                    k[2] = True
                    if ws <= k[0] <= we and not e.get("no_new"):
                        total += k[1]
                        n += 1
    return total, n


# --------------------------------------------------------------------------
# One (donor, p, seed): every workload, every rule, every replicate
# --------------------------------------------------------------------------

def job(arg):
    donor, p, seed = arg
    fam = donor[:2]
    donor_run = R.load_run(donor, p, seed)
    # The rule settings, read once per (donor, p, seed) and reused for all workloads.
    # This is ../14/workload.py's (and mine/swap.py's) move: the target stage's rule
    # settings on the donor run's exogenous inputs, nothing else of the target run.
    tbl = {}
    for name, kind, spec in RULE_SPECS:
        if kind in ("fixed", "policy"):
            tgt = R.load_run(fam + spec, p, seed)
            tbl[name] = {k: tgt[k] for k in ("policy", "choices", "choice_jitter")}
    out = {}
    for wl in WORKLOADS:
        created = gen_created(wl, p, seed, donor_run["t0"], donor_run["created"])
        W.check_created(created, donor_run["t0"], (wl, donor, p, seed))
        total = len(created)
        out[("records", wl)] = total
        for name, kind, spec in RULE_SPECS:
            run = dict(donor_run)
            run["created"] = created
            if kind in ("fixed", "policy"):
                run.update(tbl[name])
                run["rule"] = None
            else:
                run["policy"] = run["choices"] = None
                run["rule"] = (spec if kind == "rule"
                               else dict(i_min=spec, m=1.0, i_max=spec, jf=T.JF))
            reps = []
            for rep in range(REPS):
                # The tick-lag stream seed is ../13/trickle.py's, literal prefix
                # included: W0 only reproduces ev13 and ev14 if the stream is the same
                # one.  The on/off workloads reuse it so that the lag stream is paired
                # across workloads as well as across rules.  Only the seed is shared
                # between rules; they draw different numbers of lags per poll, so the
                # values diverge from the first interval change on (a lag is at most
                # 100 ms against intervals of 500 ms and more).
                rng = random.Random(R.rep_seed(f"trickle:{donor}", p, seed, rep))
                ev = sim_run(run, rng)
                # Every measurement of this step comes from the same simulation: the
                # full-window score, the early-record score, the gap counts and the
                # applied interval of a non-empty answer.
                reps.append((T.tup(R.score(ev, donor_run["t0"], p)),
                             A.early_score(ev, donor_run["t0"]),
                             W.gaps(ev, donor_run["t0"], total),
                             nonempty_applied(ev, donor_run["t0"])))
            out[(wl, name)] = reps
    return donor, p, seed, out


# --------------------------------------------------------------------------
# Cell values
# --------------------------------------------------------------------------

def early_vals(mat):
    """Pooled early-record values from a (seeds, 7) matrix of n4, c4, lat, polls.

    Columns 4..6 are zero padding, so that ../13/trickle.py's `boot_values` and
    `seed_matrix` can be reused unchanged on the early-record numbers too.
    """
    n4, c4, lat, polls = mat[:, :4].sum(0)
    return dict(delay=lat / n4 if n4 else float("nan"),
                cens4=c4 / (n4 + c4) if (n4 + c4) else float("nan"),
                polls_sent=polls / mat.shape[0])


def read_ev14_cells():
    """../14/workload_cells.csv, keyed by (workload, donor, p, rule), for the W0 check."""
    out = {}
    with open(os.path.join(EV14, "workload_cells.csv")) as fh:
        for r in csv.DictReader(fh):
            out[(r["workload"], r["donor"], float(r["p"]), r["rule"])] = r
    return out


def w0_check(rows):
    """spec.md's first check: W0 reproduces ../14/workload_cells.csv for the shared rules.

    Compared as the strings both scripts write, over every column both files print, for
    the seven arms, short, wait and T37 in all three cells.  This is also the proof
    spec.md asks for that the Engine subclass changes nothing for a rule without the
    new key: every one of these rules runs through the subclass here and through the
    untouched engine there.

    ../14/workload_gain.csv is deliberately not compared: this step's headline gain is
    scored on the early records, and its all-record gain uses the same seven arms, so
    the gain columns are checked by the no-outage identity check and by the arms'
    numbers rather than against ev14's file.
    """
    ref = read_ev14_cells()
    checked, bad = [], []
    for r in rows:
        if r["workload"] != "W0" or r["rule"] not in SHARED_RULES:
            continue
        want = ref[(r["workload"], r["donor"], r["p"], r["rule"])]
        diffs = {c: (csv_str(r[c]), want[c]) for c in W0_COLS
                 if csv_str(r[c]) != want[c]}
        checked.append(r["rule"])
        if diffs:
            bad.append((r["donor"], r["p"], r["rule"], diffs))
    return checked, bad


def csv_str(v):
    """The string this script writes into a csv cell, as ../14/workload.py writes it."""
    return R.fmt(v, 6) if isinstance(v, float) else str(v) if v is not None else ""


IDENTITY_COLS = ["delay", "cens4", "polls_sent", "answered", "timed_out", "bytes",
                 "empty_share", "mean_gap_s", "lw_gap_s", "polls_after_complete",
                 "early_delay", "early_cens4", "early_polls_sent",
                 "nonempty_interval_s"]


def identity_check(rows):
    """spec.md: T37, T37r and T37b must be identical in the no-outage cell.

    No poll can time out where the links never go down, so the overridden step is never
    reached and the three rules are one rule.  A difference would mean the subclass
    changes something on a path spec.md says it does not touch.
    """
    donor, p = NO_OUTAGE
    out = []
    for wl in WORKLOADS:
        got = {r["rule"]: r for r in rows
               if r["workload"] == wl and r["donor"] == donor and r["p"] == p
               and r["rule"] in (BASE, *VARIANTS)}
        diffs = {}
        for name in VARIANTS:
            for c in IDENTITY_COLS:
                if csv_str(got[name][c]) != csv_str(got[BASE][c]):
                    diffs[(name, c)] = (csv_str(got[name][c]), csv_str(got[BASE][c]))
        out.append((wl, diffs))
    return out


# --------------------------------------------------------------------------
# The reading rule, fixed in spec.md before any of this was computed
# --------------------------------------------------------------------------

def read_inference1(gain_rows):
    """Inference 1: does an on/off workload favour T37 where there is no outage?

    spec.md: supported if, in the no-outage cell, T37's headline gain interval lies
    entirely above zero under WO and under WV; not supported if it spans zero or lies
    below in both; otherwise mixed.  The ZERO band is evidence/14's: a rule that is
    itself a frontier knot scores a rounding residue around zero, which a bare > 0
    would read as "above".
    """
    donor, p = NO_OUTAGE
    per = {}
    for wl in ("WO", "WV"):
        r = one(gain_rows, wl, donor, p, BASE)
        above = r["gain"] is not None and r["gain_lo"] is not None and r["gain_lo"] > ZERO
        below = (r["gain"] is not None and r["gain_hi"] is not None
                 and r["gain_hi"] < -ZERO)
        spans = (r["gain_lo"] is not None and not above and not below)
        per[wl] = ("above" if above else "below" if below
                   else "spans zero" if spans else "no value")
    vals = list(per.values())
    if vals == ["above", "above"]:
        verdict = "supported"
    elif all(v in ("spans zero", "below") for v in vals):
        verdict = "not supported"
    else:
        verdict = "mixed"
    return verdict, per


def read_inference2(gain_rows):
    """Inference 2: does reacting to a timeout recover the gain under outages?

    spec.md: per outage cell and workload, the paired difference gain(variant) - gain(T37)
    with its bootstrap interval over the same resamples.  Supported for a variant if the
    difference interval lies entirely above zero in at least 4 of the 6 cases, not
    supported if in at most 1, otherwise mixed.
    """
    out = {}
    for name in VARIANTS:
        cases = []
        for (donor, p) in CELLS:
            if (donor, p) == NO_OUTAGE:
                continue
            for wl in WORKLOADS:
                r = one(gain_rows, wl, donor, p, name)
                cases.append((wl, donor, p, r["diff"], r["diff_lo"], r["diff_hi"]))
        up = sum(1 for c in cases if c[4] is not None and c[4] > 0)
        verdict = ("supported" if up >= 4 else "not supported" if up <= 1 else "mixed")
        out[name] = (verdict, up, cases)
    return out


def one(rows, wl, donor, p, name):
    """The single gain row of one (workload, cell, rule)."""
    hits = [r for r in rows if (r["workload"], r["donor"], r["p"], r["rule"])
            == (wl, donor, p, name)]
    assert len(hits) == 1, (wl, donor, p, name, len(hits))
    return hits[0]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    t_start = time.time()
    import multiprocessing as mp
    jobs = [(d, p, s) for (d, p) in CELLS for s in SEEDS]
    with mp.Pool(WORKERS) as pool:
        res = pool.map(job, jobs, chunksize=1)

    per_seed, records = {}, {}
    for donor, p, seed, out in res:
        for key, v in out.items():
            if key[0] == "records":
                # Unlike evidence/14's workloads, an on/off workload creates a different
                # number of records in each seed (the first ON instant, and under WV
                # every block length, are drawn per seed), so the count is kept per seed
                # and reported as a mean with its range instead of being asserted equal.
                records[(key[1], donor, p, seed)] = v
            else:
                per_seed[(key[0], donor, p, key[1], seed)] = v

    cell_rows = []
    vals, arrs, evals, earrs, recs = {}, {}, {}, {}, {}
    for wl in WORKLOADS:
        for (donor, p) in CELLS:
            rec = [records[(wl, donor, p, s)] for s in SEEDS]
            recs[(wl, donor, p)] = rec
            for name, kind, spec in RULE_SPECS:
                reps = [per_seed[(wl, donor, p, name, s)] for s in SEEDS]
                flat = [r for seed_reps in reps for r in seed_reps]
                v = T.pooled([m for m, _e, _g, _a in flat], p)
                v.update(W.pooled_gaps([g for _m, _e, g, _a in flat]))
                na = sum(a[0] for _m, _e, _g, a in flat)
                nn = sum(a[1] for _m, _e, _g, a in flat)
                em = T.seed_matrix([[tuple(e) + (0.0, 0.0, 0.0)
                                     for _m, e, _g, _a in seed_reps] for seed_reps in reps])
                ev = early_vals(em)
                vals[(wl, donor, p, name)] = v
                arrs[(wl, donor, p, name)] = T.seed_matrix(
                    [[m for m, _e, _g, _a in seed_reps] for seed_reps in reps])
                evals[(wl, donor, p, name)] = ev
                earrs[(wl, donor, p, name)] = em
                cell_rows.append(dict(
                    workload=wl, donor=donor, p=p, rule=name,
                    kind=("fixed" if kind == "arm" else kind),
                    settings=interval_text(name),
                    records=(rec[0] if min(rec) == max(rec)
                             else float(sum(rec)) / len(rec)),
                    delay=v["delay"], cens4=v["cens4"], polls_sent=v["polls_sent"],
                    answered=v["answered"], timed_out=v["timed_out"], bytes=v["bytes"],
                    empty_share=v["empty_share"], mean_gap_s=v["mean_gap_s"],
                    lw_gap_s=v["lw_gap_s"],
                    polls_after_complete=v["polls_after_complete"],
                    records_min=min(rec), records_max=max(rec),
                    early_delay=ev["delay"], early_cens4=ev["cens4"],
                    early_polls_sent=ev["polls_sent"],
                    nonempty_interval_s=(na / nn if nn else float("nan")),
                    nonempty_answers=nn / len(flat)))

    # spec.md: checked before anything else is read
    checked, bad = w0_check(cell_rows)
    print(f"W0 check against ../14/workload_cells.csv: {len(checked) - len(bad)}/"
          f"{len(checked)} cell x rule identical on all {len(W0_COLS)} shared columns",
          flush=True)
    for donor, p, name, diffs in bad:
        print(f"  {donor} p{p} {name}: {diffs}")
    if bad:
        sys.exit("W0 does not reproduce evidence/14; refusing to write anything")

    # spec.md's second check, in its strongest form: where no poll can time out the
    # overridden step is never reached, so T37r and T37b must produce not merely equal
    # cell values but the same per-replicate counts as T37.  Done on the raw tuples,
    # because pooling could hide a difference that cancels.
    donor0, p0 = NO_OUTAGE
    exact = {}
    for wl in WORKLOADS:
        same = all(per_seed[(wl, donor0, p0, v, s)] == per_seed[(wl, donor0, p0, BASE, s)]
                   for v in VARIANTS for s in SEEDS)
        exact[wl] = same
        print(f"no-outage per-replicate identity {wl}: "
              f"{'T37 == T37r == T37b on every seed and replicate' if same else 'DIFFERS'}",
              flush=True)

    ident = identity_check(cell_rows)
    for wl, diffs in ident:
        print(f"no-outage identity {wl}: "
              f"{'T37 == T37r == T37b' if not diffs else diffs}", flush=True)
    if any(d for _wl, d in ident) or not all(exact.values()):
        sys.exit("T37r/T37b differ from T37 where no poll times out; refusing to write")

    gain_rows = []
    for wl in WORKLOADS:
        for (donor, p) in CELLS:
            key = (wl, donor, p)
            # The same resample matrix for every rule and both scorings, which is what
            # makes a paired difference of two gains a paired statistic.
            idx = np.random.default_rng(BOOT_SEED).integers(
                0, len(SEEDS), size=(BOOT_N, len(SEEDS)))
            boot = {n: T.boot_values(arrs[key + (n,)], idx, p) for n in NAMES}
            eboot = {n: T.boot_values(earrs[key + (n,)], idx, p) for n in NAMES}

            frontiers = {}
            for tag, vv, bb in (("all", vals, boot), ("early", evals, eboot)):
                fx = sorted(FIXED, key=lambda n: vv[key + (n,)]["polls_sent"])
                mono = all(vv[key + (fx[i],)]["delay"] > vv[key + (fx[i + 1],)]["delay"]
                           for i in range(len(fx) - 1))
                frontiers[tag] = dict(
                    mono=mono, order=fx,
                    fxp=np.array([[vv[key + (n,)]["polls_sent"] for n in FIXED]]),
                    fxd=np.array([[vv[key + (n,)]["delay"] for n in FIXED]]),
                    bfp=np.stack([bb[n][1] for n in FIXED], axis=1),
                    bfd=np.stack([bb[n][0] for n in FIXED], axis=1))

            bg = {}
            for name in TESTED:
                for tag, vv, bb in (("all", vals, boot), ("early", evals, eboot)):
                    fr = frontiers[tag]
                    v = vv[key + (name,)]
                    g, s, in_d, in_p = T.gain_save(
                        fr["fxp"], fr["fxd"], np.array([v["polls_sent"]]),
                        np.array([v["delay"]]))
                    b, bs, _, _ = T.gain_save(fr["bfp"], fr["bfd"],
                                              bb[name][1], bb[name][0])
                    bg[(name, tag)] = b
                    glo, ghi, gn = T.ci(b)
                    slo, shi, sn = T.ci(bs)
                    fr.setdefault("rule", {})[name] = dict(
                        gain=float(g[0]) if in_d[0] else None,
                        gain_lo=glo if in_d[0] else None,
                        gain_hi=ghi if in_d[0] else None,
                        save=float(s[0]) if in_p[0] else None,
                        save_lo=slo if in_p[0] else None,
                        save_hi=shi if in_p[0] else None,
                        boot_n_gain=gn if in_d[0] else 0,
                        boot_n_save=sn if in_p[0] else 0,
                        outside=("" if in_d[0] and in_p[0] else
                                 ("polls outside" if not in_d[0] else "") +
                                 ("delay outside" if not in_p[0] else "")))

            for name in TESTED:
                v, ev = vals[key + (name,)], evals[key + (name,)]
                cell = [r for r in cell_rows
                        if (r["workload"], r["donor"], r["p"], r["rule"])
                        == (wl, donor, p, name)][0]
                row = dict(workload=wl, donor=donor, p=p, rule=name,
                           settings=interval_text(name), records=cell["records"],
                           polls_sent=v["polls_sent"], delay=v["delay"],
                           cens4=v["cens4"], bytes=v["bytes"],
                           empty_share=v["empty_share"], mean_gap_s=v["mean_gap_s"],
                           lw_gap_s=v["lw_gap_s"],
                           polls_after_complete=v["polls_after_complete"],
                           nonempty_interval_s=cell["nonempty_interval_s"],
                           early_polls_sent=ev["polls_sent"], early_delay=ev["delay"],
                           early_cens4=ev["cens4"])
                for tag, vv, pre in (("early", evals, ""), ("all", vals, "all_")):
                    fr = frontiers[tag]
                    r = fr["rule"][name]
                    for k in ("gain", "gain_lo", "gain_hi", "save", "save_lo",
                              "save_hi", "boot_n_gain", "boot_n_save", "outside"):
                        row[pre + k] = r[k]
                    row[pre + "frontier_monotone"] = int(fr["mono"])
                    # cens4 of the two fixed arms bracketing this rule's polls_sent: a
                    # gain bought by losing records to the window is not a gain in speed.
                    xs = fr["order"]
                    px = [vv[key + (n,)]["polls_sent"] for n in xs]
                    k = min(max(sum(1 for q in px
                                    if q < vv[key + (name,)]["polls_sent"]) - 1, 0),
                            len(FIXED) - 2)
                    nb = [xs[k], xs[k + 1]]
                    nb_cens = max(vv[key + (n,)]["cens4"] for n in nb)
                    row[pre + "neighbours"] = "+".join(nb)
                    row[pre + "neighbour_cens4"] = nb_cens
                    row[pre + "cens_flag"] = int(
                        vv[key + (name,)]["cens4"] - nb_cens > T.CENS_FLAG_PP)
                # spec.md's inference 2: the paired difference against T37 over the same
                # resamples, on both scorings.  A resample in which either rule falls
                # outside the arms' range yields no difference, so the interval rests on
                # the resamples where both are inside.
                for tag, pre in (("early", ""), ("all", "all_")):
                    if name in VARIANTS:
                        d = bg[(name, tag)] - bg[(BASE, tag)]
                        dlo, dhi, dn = T.ci(d)
                        gv = frontiers[tag]["rule"][name]["gain"]
                        gb = frontiers[tag]["rule"][BASE]["gain"]
                        row[pre + "diff"] = (None if gv is None or gb is None
                                             else gv - gb)
                        row[pre + "diff_lo"], row[pre + "diff_hi"] = dlo, dhi
                        row[pre + "boot_n_diff"] = dn
                    else:
                        row[pre + "diff"] = row[pre + "diff_lo"] = None
                        row[pre + "diff_hi"] = None
                        row[pre + "boot_n_diff"] = 0
                gain_rows.append(row)
            print(f"  {wl} {donor} p{p}: records={min(recs[key])}"
                  f"{'' if min(recs[key]) == max(recs[key]) else '..'}"
                  f"{'' if min(recs[key]) == max(recs[key]) else max(recs[key])} "
                  f"frontier monotone early={frontiers['early']['mono']} "
                  f"all={frontiers['all']['mono']}", flush=True)

    wall = time.time() - t_start
    for rows, path in ((cell_rows, "onoff_cells.csv"), (gain_rows, "onoff_gain.csv")):
        with open(os.path.join(HERE, path), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: csv_str(v) for k, v in r.items()})
    write_md(cell_rows, gain_rows, checked, ident, recs, wall)
    print("wrote onoff_cells.csv, onoff_gain.csv, onoff.md")
    print(f"wall clock {wall:.1f} s "
          f"({len(WORKLOADS) * len(CELLS) * len(SEEDS) * len(NAMES) * REPS} simulations)")


# --------------------------------------------------------------------------
# onoff.md
# --------------------------------------------------------------------------

def ci_text(lo, hi, n):
    """A bootstrap interval, with the resample count whenever it is not all of them."""
    return T.ci_text(lo, hi, n)


def write_md(cell_rows, gain_rows, checked, ident, recs, wall):
    f = T.f
    L = []
    W_ = L.append
    W_("# ev15 onoff: on/off workloads, and rules that move their interval on a timeout\n")
    W_(f"Generated by `onoff.py`, "
       f"{len(WORKLOADS) * len(CELLS) * len(SEEDS) * len(NAMES) * REPS} simulations, "
       f"{WORKERS} workers, wall clock {wall:.0f} s.\n")
    W_("Everything below is **simulated against simulated**. The replay was checked "
       "against measurements for the logged workload only (../13/validate.md); for a "
       "replaced workload there is no measurement to compare with, so these numbers say "
       "which testbed runs are worth doing and nothing more.\n")

    W_("## The workloads\n")
    W_("| workload | records per run | what it is |")
    W_("|---|---|---|")
    for wl in WORKLOADS:
        counts = [x for (w, _d, _p), lst in recs.items() if w == wl for x in lst]
        allv = sorted(set(counts))
        txt = (str(allv[0]) if len(allv) == 1
               else f"{allv[0]}..{allv[-1]}, mean "
                    f"{sum(counts) / len(counts):.1f}")
        W_(f"| {wl} | {txt} | {WORKLOAD_TEXT[wl]} |")
    W_("")
    W_("Each workload is a function of (p, seed, origin) through "
       "`random.Random(f\"ev15:{name}:{p}:{seed}:{origin}\")`, so every rule sees the "
       "same creation times on one (p, seed) and the two origins are independent. While "
       f"ON an origin creates {ON_BATCH} records every {ON_RATE_S:.0f} s, the testbed's "
       "rate and batch size. All times lie in [t0+13 s, t0+110 s]. The outages, "
       "agent_start, t0 and the topology are the donor run's own and are not replaced.\n")

    W_("## The rules\n")
    W_("| name | kind | settings |")
    W_("|---|---|---|")
    for name, kind, spec in RULE_SPECS:
        W_(f"| {name} | {'fixed' if kind == 'arm' else kind} | {interval_text(name)} |")
    W_("")
    W_("T37 is evidence/13's rule: on a timeout its interval is unchanged. T37r and T37b "
       "differ from it in that one step only, and they are the reason this directory has "
       "its own engine subclass (`Engine.resolve` in `onoff.py`). The subclass sets the "
       "interval and leaves everything after it -- the jitter draw, `applied`, "
       "`max(sent + applied, now)` and the re-arm -- to ../13/replay.py, and it skips a "
       "resolution that does not belong to the neighbor's newest outstanding poll, which "
       "is where the simulator resolves a timeout today.\n")

    W_("## W0 check: does the control reproduce evidence/14?\n")
    W_(f"{len(checked)}/{len(checked)} of the (cell x rule) pairs the two steps share "
       f"reproduce `../14/workload_cells.csv` on all {len(W0_COLS)} columns both files "
       "print, compared as the strings both scripts write. The shared rules are the "
       "seven arms, short, wait and T37. Every one of them runs through this step's "
       "engine subclass, so this is also spec.md's required proof that the subclass "
       "changes nothing for a rule without the new key. It is the first thing the script "
       "checks and a failure exits without writing a file.\n")

    W_("## No-outage identity check\n")
    W_("Where the links never go down no poll times out, so the overridden step is never "
       "reached and T37, T37r and T37b must be one rule. Compared on "
       f"{len(IDENTITY_COLS)} columns ({', '.join(IDENTITY_COLS)}) in "
       f"{NO_OUTAGE[0]} p {NO_OUTAGE[1]}, and separately on the raw per-replicate "
       "counts of all 20 seeds and 5 replicates, which is the stronger statement "
       "because pooling could hide a difference that cancels:\n")
    for wl, diffs in ident:
        W_(f"- {wl}: " + ("T37 == T37r == T37b, on the pooled columns and on every "
                          "(seed, replicate) count" if not diffs else str(diffs)))
    W_("")

    v1, per1 = read_inference1(gain_rows)
    i2 = read_inference2(gain_rows)
    W_("## Reading, under spec.md's rule fixed before the numbers\n")
    W_("**Inference 1** (an on/off workload favours T37 where there is no outage): "
       "supported if T37's headline gain interval lies entirely above zero in the "
       "no-outage cell under WO and under WV; not supported if it spans zero or lies "
       "below in both; otherwise mixed.\n")
    donor, p = NO_OUTAGE
    W_(f"| workload | T37 headline gain in {donor} p {p} | 95 % | where the interval lies |")
    W_("|---|---|---|---|")
    for wl in ("WO", "WV"):
        r = one(gain_rows, wl, donor, p, BASE)
        W_(f"| {wl} | {f(r['gain'], 1, True)} | "
           f"{ci_text(r['gain_lo'], r['gain_hi'], r['boot_n_gain'])} | {per1[wl]} |")
    W_("")
    W_(f"**Inference 1: {v1}.**\n")
    W_("**Inference 2** (reacting to a timeout recovers the gain under outages): for each "
       "variant, the paired difference gain(variant) - gain(T37) per outage cell and "
       "workload, over the same resamples. Supported if the difference interval lies "
       "entirely above zero in at least 4 of the 6 cases, not supported if in at most 1, "
       "otherwise mixed.\n")
    W_("| variant | cases with the difference interval above zero | reading |")
    W_("|---|---|---|")
    for name in VARIANTS:
        verdict, up, _cases = i2[name]
        W_(f"| {name} | {up}/6 | **{verdict}** |")
    W_("")
    W_("| variant | workload | cell | gain(T37) | gain(variant) | difference | 95 % |")
    W_("|---|---|---|---|---|---|---|")
    for name in VARIANTS:
        for wl, d, pp, diff, dlo, dhi in i2[name][2]:
            base = one(gain_rows, wl, d, pp, BASE)
            r = one(gain_rows, wl, d, pp, name)
            W_(f"| {name} | {wl} | {d[:2]} p{pp} | {f(base['gain'], 1, True)} | "
               f"{f(r['gain'], 1, True)} | {f(diff, 1, True)} | "
               f"{ci_text(dlo, dhi, r['boot_n_diff'])} |")
    W_("")
    # How close the reading sits to its own threshold, and what the all-record scoring
    # would have read.  Stated because a verdict that rests on a lower bound of a few
    # parts per million is not a verdict a rerun with other seeds would repeat; the rule
    # is spec.md's and is applied as written, this only says how thin the margin is.
    for name in VARIANTS:
        verdict, up, cases = i2[name]
        pos = sorted(c[4] for c in cases if c[4] is not None and c[4] > 0)
        allup = sum(1 for (donor, p) in CELLS if (donor, p) != NO_OUTAGE
                    for wl in WORKLOADS
                    if (one(gain_rows, wl, donor, p, name)["all_diff_lo"] or 0) > 0)
        W_(f"- {name}: {up}/6 above zero on the headline scoring, {allup}/6 on the "
           "all-record scoring (which spec.md does not read). Smallest lower bound "
           "among the cases counted as above zero: "
           + (f"{f(pos[0], 4, True)}" if pos else "none") + ".")
    W_("")

    W_("## Headline gain: early records only\n")
    W_("The headline of this step is scored on the records created at or before "
       f"t0+{CUT_MS / 1000:.0f} s, as ../14/after.py does, because evidence/14 found that "
       "a record created late is censored more often under a sparse rule and that "
       "spec.md's cens flag cannot catch it (the flag only fires when the RULE is more "
       "censored than its bracketing arms). Polls are counted over the whole window. "
       "`gain = 1 - delay(rule) / delay_frontier(polls_sent(rule))`, the frontier being "
       "piecewise-linear log(delay) over log(polls_sent) through the seven simulated "
       "fixed arms of the same cell and workload. 95 % intervals: "
       f"{BOOT_N}-resample bootstrap over the {len(SEEDS)} seeds, "
       f"`default_rng({BOOT_SEED})`, frontier and rule recomputed from the same "
       "resampled seeds.\n")
    for wl in WORKLOADS:
        W_(f"### {wl}: {WORKLOAD_TEXT[wl]}\n")
        for (donor, p) in CELLS:
            rows = [r for r in gain_rows
                    if (r["workload"], r["donor"], r["p"]) == (wl, donor, p)]
            cells = [r for r in cell_rows
                     if (r["workload"], r["donor"], r["p"]) == (wl, donor, p)]
            W_(f"**donor {donor}, p {p}**\n")
            W_("| rule | settings | polls_sent | early delay ms | early cens4 | "
               "all-record delay ms | all-record cens4 | EMPTY share | mean gap s | "
               "gap at a random instant s | interval on a non-empty answer s | "
               "headline gain | headline 95 % | all-record gain | all-record 95 % | "
               "cens flag (early) |")
            W_("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in cells:
                if r["kind"] != "fixed":
                    continue
                W_(f"| *{r['rule']}* (frontier) | {r['settings']} | "
                   f"{f(r['early_polls_sent'])} | {f(r['early_delay'], 0)} | "
                   f"{f(r['early_cens4'], 4)} | {f(r['delay'], 0)} | "
                   f"{f(r['cens4'], 4)} | {f(r['empty_share'], 3)} | "
                   f"{f(r['mean_gap_s'], 2)} | {f(r['lw_gap_s'], 2)} | "
                   f"{f(r['nonempty_interval_s'], 2)} | - | - | - | - | - |")
            for r in rows:
                W_(f"| **{r['rule']}** | {r['settings']} | {f(r['early_polls_sent'])} | "
                   f"{f(r['early_delay'], 0)} | {f(r['early_cens4'], 4)} | "
                   f"{f(r['delay'], 0)} | {f(r['cens4'], 4)} | "
                   f"{f(r['empty_share'], 3)} | {f(r['mean_gap_s'], 2)} | "
                   f"{f(r['lw_gap_s'], 2)} | {f(r['nonempty_interval_s'], 2)} | "
                   f"{f(r['gain'], 1, True)} | "
                   f"{ci_text(r['gain_lo'], r['gain_hi'], r['boot_n_gain'])} | "
                   f"{f(r['all_gain'], 1, True)} | "
                   f"{ci_text(r['all_gain_lo'], r['all_gain_hi'], r['all_boot_n_gain'])} | "
                   f"{('YES, +' + f(100 * (r['early_cens4'] - r['neighbour_cens4']), 2) + ' pp over ' + r['neighbours']) if r['cens_flag'] else 'no'} |")
            flags = [r["rule"] + ": " + r["outside"] for r in rows if r["outside"]]
            if flags:
                W_("")
                W_("Outside the seven fixed arms' range on the early-record scoring, so "
                   "that reading has no value (`polls outside` suppresses `gain`, "
                   "`delay outside` suppresses `save`): " + "; ".join(flags) + ".")
            if not rows[0]["frontier_monotone"] or not rows[0]["all_frontier_monotone"]:
                W_("")
                W_("**The seven arms' frontier is not monotone in this cell** "
                   f"(early: {bool(rows[0]['frontier_monotone'])}, all records: "
                   f"{bool(rows[0]['all_frontier_monotone'])}): more polls do not buy "
                   "less delay across all seven arms, so an interpolation through it, "
                   "and `save` especially, is not trustworthy. The gains are printed as "
                   "they came out.")
            W_("")

    W_("## Every workload x cell x rule\n")
    W_("| workload | donor | p | rule | records | polls_sent | early delay ms | "
       "early cens4 | all-record delay ms | all-record cens4 | timed_out | bytes | "
       "EMPTY share | mean gap s | gap at a random instant s | polls after complete | "
       "interval on a non-empty answer s | non-empty answers per run |")
    W_("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in cell_rows:
        W_(f"| {r['workload']} | {r['donor']} | {r['p']} | {r['rule']} | "
           f"{r['records']} | {f(r['polls_sent'])} | {f(r['early_delay'], 0)} | "
           f"{f(r['early_cens4'], 4)} | {f(r['delay'], 0)} | {f(r['cens4'], 4)} | "
           f"{f(r['timed_out'])} | {f(r['bytes'], 0)} | {f(r['empty_share'], 3)} | "
           f"{f(r['mean_gap_s'], 2)} | {f(r['lw_gap_s'], 2)} | "
           f"{f(r['polls_after_complete'])} | {f(r['nonempty_interval_s'], 2)} | "
           f"{f(r['nonempty_answers'])} |")
    W_("")

    W_("## Assumptions and deviations from spec.md\n")
    W_("1. **An ON block is half-open.** spec.md says \"ON 20 s, OFF 20 s\" and \"while "
       "ON an origin creates 4 records every 2 s\" without saying whether the instant "
       "where ON ends still carries a tick. It is read as [b, b+L), so a 20 s ON block "
       "has 10 ticks, the last 18 s into the block, and the instant ON ends is the first "
       "OFF instant. With the other reading each WO block would hold 11 ticks.")
    W_("2. **Records per run vary by seed under WO and WV**, because the first ON instant "
       "and (under WV) every block length are drawn per seed. evidence/14's workloads all "
       "had one count per cell and its script asserted so; that assertion is dropped here "
       "and the count is reported as a mean with its range. The `records` column of the "
       "csv files holds the integer where every seed agrees (W0) and the mean otherwise, "
       "with `records_min` and `records_max` beside it. `polls after complete` uses each "
       "run's own record count, so it still means \"polls sent once the node held "
       "everything\".")
    W_("3. **WV's block lengths are drawn in order, ON first**, from "
       "`expovariate(1/20)` clipped to [4, 60] s, alternating ON and OFF until the "
       "generation window is past. The first ON start is drawn before them, from the same "
       "per-origin stream.")
    W_("4. **The headline is the early-record scoring**, as spec.md fixes it, and the "
       "all-record gain is printed beside it. Both use the same seven arms, the same "
       "resample matrix and the same code (../13/trickle.py's `gain_save`, `boot_values` "
       "and `ci` applied to a four-column early matrix padded to seven), rather than "
       "../14/after.py's separate scalar interpolation. ../14/after.py's `early_score` "
       "itself is imported unchanged, so the early numbers come from the scorer that was "
       "written independently of the main one.")
    W_("5. **The paired difference uses a plain `> 0` on its lower bound**, not the ZERO "
       "band spec.md's gain reading uses: in the no-outage cell the difference is exactly "
       "zero in every resample, so the two readings agree there, and spec.md's 1e-9 rule "
       "is written for the gain, not for the difference. One counted case does sit near "
       "the threshold (see the line under the inference 2 table), so the ZERO band would "
       "not have changed the count but a thinner threshold would. The ZERO band IS "
       "applied to inference 1's gain bound, as evidence/14 applies it.")
    W_("6. **A resample where either rule falls outside the arms' range yields no "
       "difference.** The interval then rests on the resamples where both are inside, and "
       "the `boot_n_diff` column says how many those were. spec.md does not say what such "
       "a resample does.")
    W_("7. **No gain is suppressed for a non-monotone frontier.** spec.md asks for no "
       "suppression in this step (evidence/14 suppressed one case, under WR only); the "
       "`frontier_monotone` and `all_frontier_monotone` columns carry the fact instead.")
    W_("8. **The new rule key is `on_timeout`** with the values \"reset\" and \"backoff\". "
       "A rule dict without the key is evidence/13's rule, bit for bit: that is what the "
       "W0 check measures.")
    W_("9. **The tick-lag stream seed is `trickle:{donor}` for every workload**, the "
       "literal string of ../13/trickle.py, which is what makes the W0 control reproduce "
       "evidence/14. Only the seed is shared between rules; they draw different numbers of "
       "lags per poll, so the values diverge from the first interval change on.")
    W_("10. **The bootstrap's unit is the seed, each seed's value being the mean over its "
        "5 replicates**, while the point estimates pool all 100 (seed, replicate) runs. "
        "This is ../13/trickle.py's deviation 2, inherited with its code: for sums the two "
        "orders agree exactly, for the ratios they differ in the last digits.")
    W_("11. **Bytes use the same `REP_B[p]` for every rule**, as evidence/13 and /14 do: "
        "the mean reply size is taken not to depend on the rule, and on a replaced "
        "workload not on the workload either, which is a second assumption nothing here "
        "checks. The headline is stated in polls_sent, not in bytes.")
    W_("12. **No tuning.** The workloads, arms and rules listed in spec.md are the only "
        "ones run, and every number is reported as it came out.")
    W_("13. **What a replaced workload cannot carry over is validation.** "
        "../13/validate.md measured the replay against the logged workload; an on/off "
        "workload may exercise the simulator where no measurement has ever checked it. "
        "The three I_min 3.695 s rules stay above the 2 s interest lifetime, so unlike "
        "evidence/13's sub-2 s variants they do not depend on source branches no logged "
        "run reaches; what is new and unmeasured here is the timeout step itself.")
    W_("14. **The csv files carry more than spec.md asks for**, so that a reader does not "
        "have to rerun the script to see the alternative: `save` (polls saved at equal "
        "delay) with its interval for both scorings, and the paired differences on the "
        "all-record scoring as well as on the headline one. The readings above use only "
        "what spec.md names: the headline gain, and the headline paired difference.")
    W_("15. **What the per-run assertions do and do not cover.** ../14/workload.py's "
        "`check_created` is reused unchanged and asserts spec.md's list constraints for "
        "every run: both origins are h0 and h4, every time is a whole millisecond in "
        "[t0+13 s, t0+110 s], and each origin's seq runs 1..N in non-decreasing time "
        "order. That the records come 4 to a tick, 2 s apart while ON, is enforced by the "
        "generator's construction and is not separately asserted.")
    with open(os.path.join(HERE, "onoff.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
