#!/usr/bin/env python3
"""ev14 spec: the ev13 replay again, with the record creation times replaced.

    /home/lab_shared/cefore-semindex/.venv/bin/python3 -B workload.py

Writes workload_cells.csv (one row per cell x workload x rule), workload_gain.csv
(one row per cell x workload x tested rule) and workload.md.

WHY THIS EXISTS (spec.md)
  evidence/13 found that a rule which lengthens the state-poll interval while replies
  are empty loses to a fixed interval, and inferred that it loses because the testbed
  workload never goes quiet while it runs (each origin creates 4 records every 2 s
  from t0+13 s to t0+71 s).  Without new testbed runs the only way to check that
  inference is to keep the outages of the logged runs and replace the creation times.
  If the inference is right, the lengthening rule should beat a fixed interval where
  records come rarely or in bursts.

WHAT THIS CAN AND CANNOT SHOW
  The replay was checked against measurements for the testbed workload only
  (evidence/13 validate.md).  For a replaced workload there is no measurement to
  compare with, so every number here is a simulated value on an unvalidated input.
  It says which testbed runs are worth doing, not what a testbed would measure.

THE ONLY INPUT REPLACED is run["created"], the list of (ts_ms, origin, seq).  The
outage timeline, agent_start, t0 and the topology stay the donor run's own, and the
simulator (../13/replay.py) and the rules (../13/trickle.py) are imported unchanged.
"""

import csv
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV13 = os.path.normpath(os.path.join(HERE, "..", "13"))
sys.path.insert(0, EV13)
import replay as R  # noqa: E402
import trickle as T  # noqa: E402

SEEDS = T.SEEDS                     # 1..20, as ev13
REPS = R.R_REPLICATES               # 5 Mode B replicates
WORKERS = 12
BOOT_N = R.BOOT_N
BOOT_SEED = R.BOOT_SEED
ORIGINS = ("h0", "h4")              # spec: the two origins of every logged run

# spec: donors and p.  s6 p 0.2 / 0.4 have an 8-12 s outage, s5 p 0 has none.
CELLS = [("s6b10p043", 0.2), ("s6b10p043", 0.4), ("s5b10p043", 0.0)]

WORKLOADS = ["W0", "WP", "WB", "WR"]
WORKLOAD_TEXT = {
    "W0": "control: the logged creation times, untouched",
    "WP": "periodic, period from (1, 2, 4) s, start U[13, 50] s, 60 s of generation, "
          "2*period records per tick (120 per origin)",
    "WB": "6 bursts of 20 records per origin, burst times U[13, 110] s",
    "WR": "12 single records per origin, times U[13, 110] s",
}

# The seven fixed arms of the frontier.  "fixed" takes the named stage's meta.json
# settings from the donor's own family, exactly as ../13/trickle.py does, so that the
# four testbed arms reproduce ev13 bit for bit.  "arm" is one of the three arms spec.md
# adds (0.8, 2.2, 6.0 s): the spec2 rule with m = 1 and I_min = I_max = X, which
# ../13/identity.py shows to produce the same events as a fixed arm X.  The three are
# there because ../13/after.md showed a four-arm chord flatters a rule whose poll count
# lands mid-segment.
FIXED_SPECS = [
    ("fix0.5", "fixed", "b0p5"),
    ("fix0.8", "arm", 0.8),
    ("fix1.359", "fixed", "b1p359"),
    ("fix2.2", "arm", 2.2),
    ("fix3.695", "fixed", "b3p695"),
    ("fix6.0", "arm", 6.0),
    ("fix10.043", "fixed", "b10p043"),
]
# short, wait and the five ../13/spec2.md variants, unchanged and in ev13's order.
TESTED_SPECS = [s for s in T.RULE_SPECS if s[0] in T.TESTED]
RULE_SPECS = FIXED_SPECS + TESTED_SPECS
NAMES = [r[0] for r in RULE_SPECS]
FIXED = [r[0] for r in FIXED_SPECS]
TESTED = [r[0] for r in TESTED_SPECS]

# The 11 (cell x rule) pairs this step shares with ev13: the W0 control must reproduce
# their trickle_cells.csv values to every printed digit before anything else is read.
SHARED_RULES = [n for n, k, s in RULE_SPECS if k != "arm"]
ZERO = 1e-9                         # spec: a gain within 1e-9 of zero counts as zero


def interval_text(name):
    """What a rule does, for the tables."""
    for n, kind, spec in RULE_SPECS:
        if n == name:
            if kind == "arm":
                return f"fixed {spec} s, jitter 0.1 (rule with m = 1, I_min = I_max)"
            return T.interval_text(name)
    raise KeyError(name)


# --------------------------------------------------------------------------
# The workloads
# --------------------------------------------------------------------------

def gen_created(name, p, seed, t0, logged):
    """The (ts_ms, origin, seq) list of workload `name` for one run.

    spec.md: a workload is a function of (p, seed, origin) only, so every rule and both
    donors see the same creation times on one (p, seed), and the two origins are drawn
    independently.  The donor is deliberately not in the seed string: that is what makes
    the three cells differ in their outages and in nothing else.
    """
    if name == "W0":
        return logged
    out = []
    for origin in ORIGINS:
        rng = random.Random(f"ev14:{name}:{p}:{seed}:{origin}")
        if name == "WP":
            # Neither the start nor the period is known to the rule in advance, which is
            # the one property evidence/13 named as missing from the testbed workload.
            # 2*period records per tick keeps the testbed's 120 records per origin and
            # its mean rate of 2 records per second whichever period is drawn.
            period = rng.choice((1, 2, 4))
            start = rng.uniform(13.0, 50.0)
            secs = [start + k * period
                    for k in range(60 // period) for _ in range(2 * period)]
        elif name == "WB":
            # Sorting is what enforces spec.md's "seq = 1..N in non-decreasing time
            # order"; the 20 records of a burst share one instant, as the 4 records of a
            # logged tick do.
            secs = [b for b in sorted(rng.uniform(13.0, 110.0) for _ in range(6))
                    for _ in range(20)]
        elif name == "WR":
            secs = sorted(rng.uniform(13.0, 110.0) for _ in range(12))
        else:
            raise KeyError(name)
        out += [(float(round(t0 + s * 1000.0)), origin, i)
                for i, s in enumerate(secs, 1)]
    out.sort()                      # load_run sorts its own list the same way
    return out


def check_created(created, t0, where):
    """spec.md's constraints on a replaced list, asserted for every run.

    The 20 s margin before the window ends is what keeps a record's delay from being
    cut short by the window rather than by the rule: the scorer censors a record that
    is not held 4 hops away by t0+130 s.
    """
    assert created, where
    per = defaultdict(list)
    for ts, origin, seq in created:
        assert origin in ORIGINS, (where, origin)
        assert float(ts).is_integer(), (where, ts)
        assert t0 + 13_000.0 <= ts <= t0 + 110_000.0, (where, ts - t0)
        per[origin].append((ts, seq))
    for origin, lst in per.items():
        assert [s for _, s in lst] == list(range(1, len(lst) + 1)), (where, origin)
        assert all(lst[i][0] <= lst[i + 1][0] for i in range(len(lst) - 1)), where


# --------------------------------------------------------------------------
# One (donor, p, seed): every workload, every rule, every replicate
# --------------------------------------------------------------------------

def gaps(events, t0, total):
    """[polls in window, polls after the node held everything, sum g, sum g^2, n gaps].

    g is the time between consecutive state polls to one neighbor, inside the window.
    This is ../13/after.py's measurement with its hardcoded 240 records replaced by
    `total`: under WR a run holds 24 records, so the imported version would report
    "polls after the node held everything" as 0 for every rule and every cell.  The
    reply-side columns of after.py are dropped because this step does not ask for them.

    Why both gap means: sum g / n is what the node does on average, sum g^2 / sum g is
    the gap a record created at an arbitrary instant falls into (a long gap catches
    records in proportion to its length).  They are equal for evenly spaced polls, and
    evidence/13 measured the second one far above the first under the lengthening rule.
    """
    ws, we = t0 + R.WINDOW_START_MS, t0 + R.WINDOW_END_MS
    polls = after = n_gap = 0
    sum_g = sum_g2 = 0.0
    for rows in events.values():
        held, prev, cur = set(), {}, {}
        for e in rows:
            ev = e.get("event")
            if ev in ("record_created", "record_stored"):
                held.add((e["origin"], e["seq"]))
            elif ev == "decision" and e.get("kind") == "state_poll":
                nb = e["action"]["neighbor"]
                prev[nb], cur[nb] = cur.get(nb), e["ts_ms"]
                if ws <= e["ts_ms"] <= we:
                    polls += 1
                    after += len(held) >= total
                    if prev[nb] is not None and prev[nb] >= ws:
                        g = cur[nb] - prev[nb]
                        sum_g += g
                        sum_g2 += g * g
                        n_gap += 1
    return [polls, after, sum_g, sum_g2, n_gap]


def job(arg):
    donor, p, seed = arg
    fam = donor[:2]
    donor_run = R.load_run(donor, p, seed)
    # The rule settings, read once per (donor, p, seed) and reused for all 4 workloads.
    # For a "fixed"/"policy" rule this is ../13/trickle.py's (and mine/swap.py's) move:
    # the target stage's rule settings on the donor run's exogenous inputs, nothing else
    # of the target run.
    tbl = {}
    for name, kind, spec in RULE_SPECS:
        if kind in ("fixed", "policy"):
            tgt = R.load_run(fam + spec, p, seed)
            tbl[name] = {k: tgt[k] for k in ("policy", "choices", "choice_jitter")}
    out = {}
    for wl in WORKLOADS:
        created = gen_created(wl, p, seed, donor_run["t0"], donor_run["created"])
        check_created(created, donor_run["t0"], (wl, donor, p, seed))
        total = len(created)
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
                # included: W0 only reproduces trickle_cells.csv if the stream is the
                # same one.  The other workloads reuse it so that the lag stream is
                # paired across workloads as well as across rules.  Only the seed is
                # shared between rules; they draw different numbers of lags per poll,
                # so the values diverge from the first interval change on (lag <= 100 ms
                # against intervals of 500 ms and more).
                rng = random.Random(R.rep_seed(f"trickle:{donor}", p, seed, rep))
                ev = R._sim_run(run, "B", sim_rng=rng)
                reps.append((T.tup(R.score(ev, donor_run["t0"], p)),
                             gaps(ev, donor_run["t0"], total)))
            out[(wl, name)] = reps
        out[("records", wl)] = total
    return donor, p, seed, out


# --------------------------------------------------------------------------
# Cell values
# --------------------------------------------------------------------------

def pooled_gaps(rows):
    """The gap columns, pooled over runs the way the metric columns are pooled."""
    polls, after, sum_g, sum_g2, n_gap = (sum(r[i] for r in rows) for i in range(5))
    k = len(rows)
    return dict(
        mean_gap_s=sum_g / n_gap / 1000.0 if n_gap else float("nan"),
        lw_gap_s=sum_g2 / sum_g / 1000.0 if sum_g else float("nan"),
        polls_after_complete=after / k,
    )


def read_ev13_cells():
    """../13/trickle_cells.csv, keyed by (donor, p, rule), for the W0 check."""
    out = {}
    with open(os.path.join(EV13, "trickle_cells.csv")) as fh:
        for r in csv.DictReader(fh):
            out[(r["donor"], float(r["p"]), r["rule"])] = r
    return out


W0_COLS = ["delay", "cens4", "polls_sent", "answered", "timed_out", "bytes",
           "empty_share"]


def w0_check(vals):
    """spec.md's first check: W0 reproduces ../13/trickle_cells.csv to every printed digit.

    Compared as the strings both scripts write (R.fmt(v, 6)) on the sim_* columns, over
    the cells and rules the two steps share.  trickle_gain.csv is deliberately NOT
    compared: this step scores against seven fixed arms instead of four, so every gain
    differs by construction and a difference there would be the spec working, not a fault.
    """
    ref = read_ev13_cells()
    rows, bad = [], 0
    for (donor, p) in CELLS:
        for name in SHARED_RULES:
            got = vals[("W0", donor, p, name)]
            want = ref[(donor, p, name)]
            diffs = [c for c in W0_COLS
                     if R.fmt(got[c], 6) != want["sim_" + c]]
            bad += bool(diffs)
            rows.append((donor, p, name, diffs,
                         {c: (R.fmt(got[c], 6), want["sim_" + c]) for c in diffs}))
    return rows, bad


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    t_start = time.time()
    import multiprocessing as mp
    jobs = [(d, p, s) for (d, p) in CELLS for s in SEEDS]
    with mp.Pool(WORKERS) as pool:
        res = pool.map(job, jobs, chunksize=1)
    per_seed = {}
    records = {}
    for donor, p, seed, out in res:
        for key, v in out.items():
            if key[0] == "records":
                k = (key[1], donor, p)
                # Every seed of a cell must create the same number of records, or the
                # "polls after the node held everything" column would mean a different
                # thing in each run of the same cell.
                assert records.setdefault(k, v) == v, (k, v)
            else:
                per_seed[(key[0], donor, p, key[1], seed)] = v

    vals, arrs = {}, {}
    for wl in WORKLOADS:
        for (donor, p) in CELLS:
            for name in NAMES:
                reps = [per_seed[(wl, donor, p, name, s)] for s in SEEDS]
                v = T.pooled([m for r in reps for m, _ in r], p)
                v.update(pooled_gaps([g for r in reps for _, g in r]))
                vals[(wl, donor, p, name)] = v
                arrs[(wl, donor, p, name)] = T.seed_matrix([[m for m, _ in r]
                                                            for r in reps])

    # spec.md: checked before anything else is read
    w0_rows, w0_bad = w0_check(vals)
    print(f"W0 check against ../13/trickle_cells.csv: "
          f"{len(w0_rows) - w0_bad}/{len(w0_rows)} cell x rule identical", flush=True)
    if w0_bad:
        for donor, p, name, diffs, detail in w0_rows:
            if diffs:
                print(f"  {donor} p{p} {name}: {detail}")
        sys.exit("W0 does not reproduce ev13; refusing to write anything")

    cell_rows, gain_rows = [], []
    for wl in WORKLOADS:
        for (donor, p) in CELLS:
            key = (wl, donor, p)
            rng = np.random.default_rng(BOOT_SEED)
            idx = rng.integers(0, len(SEEDS), size=(BOOT_N, len(SEEDS)))
            boot = {n: T.boot_values(arrs[key + (n,)], idx, p) for n in NAMES}

            for name, kind, spec in RULE_SPECS:
                v = vals[key + (name,)]
                cell_rows.append(dict(
                    workload=wl, donor=donor, p=p, rule=name,
                    kind=("fixed" if kind == "arm" else kind),
                    settings=interval_text(name), records=records[key],
                    delay=v["delay"], cens4=v["cens4"], polls_sent=v["polls_sent"],
                    answered=v["answered"], timed_out=v["timed_out"], bytes=v["bytes"],
                    empty_share=v["empty_share"], mean_gap_s=v["mean_gap_s"],
                    lw_gap_s=v["lw_gap_s"],
                    polls_after_complete=v["polls_after_complete"]))

            # The frontier's own monotonicity, on the point estimate: do more polls buy
            # less delay across all seven arms?
            fx = sorted(FIXED, key=lambda n: vals[key + (n,)]["polls_sent"])
            mono = all(vals[key + (fx[i],)]["delay"] > vals[key + (fx[i + 1],)]["delay"]
                       for i in range(len(fx) - 1))
            # spec.md fixes this only for WR, whose frontier it expects to be noisy:
            # no gain is computed there, and the raw delay and polls are shown instead.
            # For WP and WB a non-monotone frontier is reported with the flag and the
            # gain is still printed, as evidence/13 printed it.
            suppress = (wl == "WR" and not mono)
            fxp = np.array([[vals[key + (n,)]["polls_sent"] for n in FIXED]])
            fxd = np.array([[vals[key + (n,)]["delay"] for n in FIXED]])
            bfp = np.stack([boot[n][1] for n in FIXED], axis=1)
            bfd = np.stack([boot[n][0] for n in FIXED], axis=1)

            for name in TESTED:
                v = vals[key + (name,)]
                g, s, in_d, in_p = gain_save_1(fxp, fxd, v)
                bg, bs, _, _ = T.gain_save(bfp, bfd, boot[name][1], boot[name][0])
                glo, ghi, gn = T.ci(bg)
                slo, shi, sn = T.ci(bs)
                # cens4 of the two fixed arms bracketing this rule's polls_sent: a gain
                # bought by losing records to the window is not a gain in speed.
                xs = sorted(FIXED, key=lambda n: vals[key + (n,)]["polls_sent"])
                px = [vals[key + (n,)]["polls_sent"] for n in xs]
                k = min(max(sum(1 for q in px if q < v["polls_sent"]) - 1, 0),
                        len(FIXED) - 2)
                nb = [xs[k], xs[k + 1]]
                nb_cens = max(vals[key + (n,)]["cens4"] for n in nb)
                ok = in_d[0] and not suppress
                gain_rows.append(dict(
                    workload=wl, donor=donor, p=p, rule=name,
                    settings=interval_text(name), records=records[key],
                    polls_sent=v["polls_sent"], delay=v["delay"], cens4=v["cens4"],
                    bytes=v["bytes"], empty_share=v["empty_share"],
                    mean_gap_s=v["mean_gap_s"], lw_gap_s=v["lw_gap_s"],
                    polls_after_complete=v["polls_after_complete"],
                    gain=float(g[0]) if ok else None,
                    gain_lo=glo if ok else None, gain_hi=ghi if ok else None,
                    save=float(s[0]) if (in_p[0] and not suppress) else None,
                    save_lo=slo if (in_p[0] and not suppress) else None,
                    save_hi=shi if (in_p[0] and not suppress) else None,
                    outside=("" if in_d[0] and in_p[0] else
                             ("polls outside" if not in_d[0] else "") +
                             ("delay outside" if not in_p[0] else "")),
                    frontier_monotone=int(mono), gain_suppressed=int(suppress),
                    neighbours="+".join(nb), neighbour_cens4=nb_cens,
                    cens_flag=int(v["cens4"] - nb_cens > T.CENS_FLAG_PP),
                    boot_n_gain=gn if ok else 0, boot_n_save=sn if not suppress else 0))
            print(f"  {wl} {donor} p{p}: records={records[key]} "
                  f"frontier monotone={mono}", flush=True)

    wall = time.time() - t_start
    for rows, path in ((cell_rows, "workload_cells.csv"),
                       (gain_rows, "workload_gain.csv")):
        with open(os.path.join(HERE, path), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (R.fmt(v, 6) if isinstance(v, float) else v)
                            for k, v in r.items()})
    write_md(cell_rows, gain_rows, w0_rows, wall)
    print("wrote workload_cells.csv, workload_gain.csv, workload.md")
    print(f"wall clock {wall:.1f} s "
          f"({len(WORKLOADS)*len(CELLS)*len(SEEDS)*len(NAMES)*REPS} simulations)")


def gain_save_1(fxp, fxd, v):
    """../13/trickle.py's gain_save for one rule's point estimate."""
    return T.gain_save(fxp, fxd, np.array([v["polls_sent"]]), np.array([v["delay"]]))


# --------------------------------------------------------------------------
# The reading rule, fixed in spec.md before any of this was computed
# --------------------------------------------------------------------------

def reading(rows):
    """spec.md's reading rule for one (workload, rule), over the 3 cells.

    helps: the gain interval lies entirely above zero in at least 2 of the 3 cells
           without a cens flag.
    does not help: in all 3 cells the gain is at or below zero, or its interval spans zero.
    anything else: mixed.
    A cell with no gain (suppressed frontier, or the rule outside the arms' range)
    satisfies neither side, so it can only produce "mixed"; stated as an interpretation
    in the deviations below, because spec.md does not say what a missing cell does.
    """
    # The ZERO band on the lower end matters: short and wait at the cell without an
    # outage are the 10.043 s arm itself, so they sit on a frontier knot and their
    # interval is a rounding residue around zero, which a bare > 0 would read as "above".
    up = sum(1 for r in rows
             if r["gain_lo"] is not None and r["gain_lo"] > ZERO and not r["cens_flag"])
    down = sum(1 for r in rows
               if r["gain"] is not None
               and (r["gain"] <= ZERO
                    or (r["gain_lo"] is not None and r["gain_lo"] < 0 < r["gain_hi"])))
    if up >= 2:
        return "helps"
    if down == len(rows):
        return "does not help"
    return "mixed"


def ci_cell(r, which):
    """A bootstrap interval for a table cell.

    A reading suppressed by spec.md's non-monotone clause prints a plain dash: feeding
    its zero resample count to ../13/trickle.py's `ci_text` would print "0/2000
    resamples inside the frontier", which says the rule fell outside the arms' range
    and is not what happened.
    """
    if r["gain_suppressed"]:
        return "-"
    return T.ci_text(r[which + "_lo"], r[which + "_hi"], r["boot_n_" + which])


def write_md(cell_rows, gain_rows, w0_rows, wall):
    f = T.f
    L = []
    W = L.append
    W("# ev14 workload: the ev13 replay with the record creation times replaced\n")
    W(f"Generated by `workload.py` ({time.strftime('%Y-%m-%d %H:%M:%S')}), "
      f"{len(WORKLOADS)*len(CELLS)*len(SEEDS)*len(NAMES)*REPS} simulations, "
      f"{WORKERS} workers, wall clock {wall:.0f} s.\n")
    W("Everything below is **simulated against simulated**. The replay was checked "
      "against measurements for the logged workload only (../13/validate.md); for a "
      "replaced workload there is no measurement to compare with, so these numbers say "
      "which testbed runs are worth doing and nothing more.\n")

    W("## The workloads\n")
    W("| workload | records per run | what it is |")
    W("|---|---|---|")
    for wl in WORKLOADS:
        rec = sorted({r["records"] for r in cell_rows if r["workload"] == wl})
        W(f"| {wl} | {', '.join(str(x) for x in rec)} | {WORKLOAD_TEXT[wl]} |")
    W("")
    W("Each workload is a function of (p, seed, origin) through "
      "`random.Random(f\"ev14:{name}:{p}:{seed}:{origin}\")`, so every rule sees the same "
      "creation times on one (p, seed) and the two origins are independent. All times "
      "lie in [t0+13 s, t0+110 s], so every record has at least 20 s before the window "
      "ends at t0+130 s. The outages, agent_start, t0 and the topology are the donor "
      "run's own and are not replaced.\n")

    W("## W0 check: does the control reproduce evidence/13?\n")
    bad = [r for r in w0_rows if r[3]]
    W(f"{len(w0_rows) - len(bad)}/{len(w0_rows)} of the (cell x rule) pairs the two steps "
      "share reproduce `../13/trickle_cells.csv` to every printed digit, over the "
      "columns delay, cens4, polls_sent, answered, timed_out, bytes and EMPTY share. "
      "This is spec.md's first check and nothing else was read before it passed; a "
      "failure exits the script without writing a file.\n")
    if bad:
        W("Differences:\n")
        for donor, p, name, diffs, detail in bad:
            W(f"- {donor} p {p} {name}: {detail}")
        W("")
    W("`../13/trickle_gain.csv` is deliberately not compared. This step scores against "
      "seven fixed arms where evidence/13 used four, so every W0 gain differs from "
      "ev13's by construction. The cell values are what carries over.\n")

    W("## Reading, under spec.md's rule fixed before the numbers\n")
    W("\"Helps\" needs a gain interval entirely above zero in at least 2 of the 3 cells "
      "without a cens flag. \"Does not help\" needs the gain at or below zero, or an "
      "interval spanning zero, in all 3 cells. Anything else is mixed.\n")
    W("| workload | " + " | ".join(TESTED) + " |")
    W("|---" * (len(TESTED) + 1) + "|")
    for wl in WORKLOADS:
        cells = [reading([r for r in gain_rows
                          if r["workload"] == wl and r["rule"] == n]) for n in TESTED]
        W(f"| {wl} | " + " | ".join(cells) + " |")
    W("")

    W("## Gain per workload, cell and rule\n")
    W("`gain = 1 - delay(rule) / delay_frontier(polls_sent(rule))`, the frontier being "
      "piecewise-linear log(delay) over log(polls_sent) through the seven simulated "
      "fixed arms of the same cell and workload. Positive = faster than a fixed interval "
      "costing the same number of polls. `save` is the same statement read as polls "
      "saved at equal delay. 95 % intervals: 2000-resample bootstrap over the 20 seeds, "
      f"`default_rng({BOOT_SEED})`, frontier and rule recomputed from the same resampled "
      "seeds. A resample that puts the rule outside the arms' range yields no value, so "
      f"an interval resting on fewer than {BOOT_N} resamples says so.\n")
    for wl in WORKLOADS:
        W(f"### {wl}: {WORKLOAD_TEXT[wl]}\n")
        for (donor, p) in CELLS:
            rows = [r for r in gain_rows
                    if r["workload"] == wl and r["donor"] == donor and r["p"] == p]
            cells = [r for r in cell_rows
                     if r["workload"] == wl and r["donor"] == donor and r["p"] == p]
            W(f"**donor {donor}, p {p}**\n")
            W("| rule | settings | polls_sent | delay ms | cens4 | EMPTY share | "
              "mean gap s | gap at a random instant s | polls after complete | gain | "
              "gain 95 % | save | save 95 % | cens flag |")
            W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in cells:
                if r["kind"] != "fixed":
                    continue
                W(f"| *{r['rule']}* (frontier) | {r['settings']} | "
                  f"{f(r['polls_sent'])} | {f(r['delay'], 0)} | {f(r['cens4'], 4)} | "
                  f"{f(r['empty_share'], 3)} | {f(r['mean_gap_s'], 2)} | "
                  f"{f(r['lw_gap_s'], 2)} | {f(r['polls_after_complete'])} | - | - | - "
                  "| - | - |")
            for r in rows:
                W(f"| **{r['rule']}** | {r['settings']} | {f(r['polls_sent'])} | "
                  f"{f(r['delay'], 0)} | {f(r['cens4'], 4)} | {f(r['empty_share'], 3)} | "
                  f"{f(r['mean_gap_s'], 2)} | {f(r['lw_gap_s'], 2)} | "
                  f"{f(r['polls_after_complete'])} | {f(r['gain'], 1, True)} | "
                  f"{ci_cell(r, 'gain')} | {f(r['save'], 1, True)} | "
                  f"{ci_cell(r, 'save')} | "
                  f"{('YES, +' + f(100*(r['cens4']-r['neighbour_cens4']), 2) + ' pp over ' + r['neighbours']) if r['cens_flag'] else 'no'} |")
            if rows[0]["gain_suppressed"]:
                W("")
                W("**The seven arms' frontier is not monotone in this cell and this is "
                  "WR**, so spec.md computes no gain here and the raw delay and poll "
                  "counts above are the whole reading.")
            elif not rows[0]["frontier_monotone"]:
                W("")
                W("**The seven arms' frontier is not monotone in this cell**: more polls "
                  "do not buy less delay across all seven arms, so an interpolation "
                  "through it, and `save` especially, is not trustworthy. spec.md "
                  "suppresses a gain for this reason under WR only, so the gains above "
                  "are printed as they came out.")
            out = [r["rule"] + ": " + r["outside"] for r in rows if r["outside"]]
            if out:
                W("")
                W("Outside the seven fixed arms' range, so that reading has no value "
                  "(`polls outside` suppresses `gain`, `delay outside` suppresses "
                  "`save`): " + "; ".join(out) + ".")
            W("")

    W("## Every workload x cell x rule\n")
    W("| workload | donor | p | rule | polls_sent | delay ms | cens4 | timed_out | "
      "bytes | EMPTY share | mean gap s | gap at a random instant s | "
      "polls after complete |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in cell_rows:
        W(f"| {r['workload']} | {r['donor']} | {r['p']} | {r['rule']} | "
          f"{f(r['polls_sent'])} | {f(r['delay'], 0)} | {f(r['cens4'], 4)} | "
          f"{f(r['timed_out'])} | {f(r['bytes'], 0)} | {f(r['empty_share'], 3)} | "
          f"{f(r['mean_gap_s'], 2)} | {f(r['lw_gap_s'], 2)} | "
          f"{f(r['polls_after_complete'])} |")
    W("")

    W("## Assumptions and deviations from spec.md\n")
    W("1. **A cell with no gain can only produce \"mixed\".** spec.md's reading rule "
      "says what a gain above zero and a gain at or below zero mean, but not what a "
      "cell without a gain does. Such a cell satisfies neither side here, so it cannot "
      "contribute to \"does not help\" (which needs all 3 cells) and counts towards "
      "\"mixed\".")
    W("2. **The non-monotone rule is applied to WR only**, as spec.md words it. If the "
      "WP or WB frontier comes out non-monotone the gain is still printed, with the "
      "flag beside it, which is how evidence/13 reported the one non-monotone cell it "
      "had. The `frontier_monotone` and `gain_suppressed` columns of "
      "`workload_gain.csv` separate the two.")
    W("3. **The gap columns are ../13/after.py's measurement with the record count "
      "parameterised.** after.py tests `len(held) >= 240`; WR creates 24 records per "
      "run, so the imported version would report zero polls after completion "
      "everywhere. The reply-side gap columns of after.py (gap before an EMPTY reply "
      "against gap before a non-empty one) are not recomputed here, because this step "
      "does not ask for them. after.py pooled replicate 0 only; these columns pool all "
      "5 replicates. \"polls after complete\" counts the in-window state polls of all "
      "5 nodes of a run that were sent once that node already held every record of the "
      "run, divided by the 100 (seed, replicate) runs, so it is a per-run count over "
      "the whole topology, not a per-node one.")
    W("4. **The three added fixed arms (0.8, 2.2, 6.0 s) are the spec2 rule with m = 1 "
      "and I_min = I_max**, which ../13/identity.py shows produces the same events as a "
      "fixed arm. With m = 1 the due instant never moves at a resolution, so the rule "
      "path consumes the tick-lag stream in exactly the order a fixed arm does "
      "(../13/trickle.md deviation 7c, measured by identity.py in Mode B as well as "
      "Mode A): the seven arms are one kind of arm, not two. The four testbed arms keep "
      "their stage settings and the fixed code path, which is what makes the W0 check "
      "exact for them.")
    W("5. **The tick-lag stream seed is `trickle:{donor}` for every workload**, the "
      "literal string of ../13/trickle.py. W0 only reproduces ev13 with that stream, "
      "and reusing it for the other three workloads pairs the lag stream across "
      "workloads too. Only the seed is shared between rules: they draw different "
      "numbers of lags per poll, so the values diverge from the first interval change "
      "on. The lag is at most 100 ms against intervals of 500 ms and more.")
    W("6. **The bootstrap's unit is the seed, each seed's value being the mean over its "
      "5 replicates**, while the point estimates pool all 100 (seed, replicate) runs. "
      "This is ../13/trickle.py's deviation 2, inherited with its code: for sums the "
      "two orders agree exactly, for the ratios they differ in the last digits.")
    W("7. **Bytes use the same `REP_B[p]` for every rule**, as evidence/13 does: the "
      "mean reply size is taken not to depend on the rule, and on a replaced workload "
      "it is taken not to depend on the workload either, which is a second assumption "
      "nothing here checks. The headline is stated in polls_sent, not in bytes.")
    W("8. **Delay is the 4-hop uncensored mean**, so a rule that loses records to the "
      "window looks faster than it is. That is what the cens flag and the cens4 column "
      "are for.")
    W("9. **No tuning.** The workloads, arms and variants listed in spec.md are the only "
      "ones run, and every number is reported as it came out.")
    W("10. **What the replaced workload does not change**: the outage timeline, the "
      "record fetch machinery, the reply size model and the simulator itself. What it "
      "cannot carry over is validation: ../13/validate.md measured the replay against "
      "the logged workload, and a workload with a different shape may exercise the "
      "simulator where no measurement has ever checked it. The sub-2 s variants "
      "(T05, T136, T136s, T136c) additionally run source branches that no logged run "
      "reaches, as ../13/trickle.md deviation notes.")
    with open(os.path.join(HERE, "workload.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
