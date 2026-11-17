#!/usr/bin/env python3
"""ev13 spec2 Part 3: the "lengthen the interval while replies are empty" rule, replayed.

    /home/lab_shared/cefore-semindex/.venv/bin/python3 -B trickle.py

Writes trickle_cells.csv (one row per cell x rule), trickle_gain.csv (one row per
cell x variant-or-policy) and trickle.md.  Runs the identity test of spec2 Part 2
first, through identity.py, and refuses to go on if it fails -- a variant number is
only meaningful once the new code path is known to reproduce the old one.

WHAT IS SIMULATED AND WHAT IS NOT (spec2 Part 3)
  The exogenous inputs -- outage instants, record creation times, agent_start, t0,
  topology -- come from a DONOR arm's run: s6b10p043 for p 0.2/0.4/0.5/0.6 (the 8-12 s
  outage shape) and s5b10p043 for p 0/0.2/0.4 (the other shape).  On those same inputs
  11 poll-interval rules are simulated: the four fixed intervals, short, wait (each with
  the meta.json settings of the stage of the same family) and the five spec2 variants.
  This is mine/swap.py's construction -- a donor run's inputs with another stage's rule
  -- and it is what makes the comparison paired: for one (p, seed, replicate) every rule
  sees the same outages and the same records.

  ALL comparisons here are simulated against simulated.  The measured cell values of the
  six existing rules are printed beside them for reference only; nothing is scored
  against them.  Mode B, R = 5 replicates, 20 seeds.

NOT CLAIMED (spec2): anything about real bytes on the wire, about a rule's effect on
record-fetch traffic, or about workloads other than this one (120 records per origin
every 2 s from t0+13 s).
"""

import csv
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replay as R  # noqa: E402

PY = sys.executable
EV = R.EV

JF = 0.1              # spec2 Part 2; equals every b* arm's state_poll_choice_jitter_frac
SEEDS = list(range(1, 21))
REPS = R.R_REPLICATES
WORKERS = 16
BOOT_N = R.BOOT_N
BOOT_SEED = R.BOOT_SEED
CENS_FLAG_PP = 0.01   # spec2: flag a gain bought with >1 pp more censoring

# (donor stage, p).  The donor supplies the exogenous inputs only.
CELLS = ([("s6b10p043", p) for p in (0.2, 0.4, 0.5, 0.6)]
         + [("s5b10p043", p) for p in (0.0, 0.2, 0.4)])

# name, kind, spec.  "fixed"/"policy" take the named stage's meta.json settings from the
# donor's own family (s5 or s6), "rule" is a spec2 Part 2 rule.
RULE_SPECS = [
    ("fix0.5", "fixed", "b0p5"),
    ("fix1.359", "fixed", "b1p359"),
    ("fix3.695", "fixed", "b3p695"),
    ("fix10.043", "fixed", "b10p043"),
    ("short", "policy", "short"),
    ("wait", "policy", "wait"),
    ("T05", "rule", dict(i_min=0.5, m=2.0, i_max=10.043, jf=JF)),
    ("T136", "rule", dict(i_min=1.359, m=2.0, i_max=10.043, jf=JF)),
    ("T37", "rule", dict(i_min=3.695, m=2.0, i_max=10.043, jf=JF)),
    ("T136s", "rule", dict(i_min=1.359, m=1.5, i_max=10.043, jf=JF)),
    ("T136c", "rule", dict(i_min=1.359, m=2.0, i_max=20.086, jf=JF)),
]
NAMES = [r[0] for r in RULE_SPECS]
FIXED = NAMES[:4]
TESTED = NAMES[4:]                     # short, wait and the five variants
RULE_OF = {n: s for n, k, s in RULE_SPECS if k == "rule"}


def interval_text(name):
    """What the rule does, for the tables: the fixed value, the table rows, or I_min/m/I_max."""
    for n, kind, spec in RULE_SPECS:
        if n != name:
            continue
        if kind == "rule":
            return f"I_min {spec['i_min']} m {spec['m']} I_max {spec['i_max']} jf {spec['jf']}"
        if kind == "fixed":
            return f"fixed {name[3:]} s, jitter 0.1"
        return {"short": "table 10.043 -> 3.695 s on a timeout, jitter 0.1/0.1",
                "wait": "table 10.043 -> 5.0 s on a timeout, jitter 0.1/0.0"}[name]
    raise KeyError(name)


# --------------------------------------------------------------------------
# One (donor, p, seed): every rule, every replicate
# --------------------------------------------------------------------------

def tup(s):
    """The seven per-run counts every cell value is pooled from."""
    return (s["n4"], s["c4"], s["lat4"], s["polls"], s["answered"], s["timed_out"],
            s["empty"])


def job(arg):
    donor, p, seed = arg
    fam = donor[:2]
    donor_run = R.load_run(donor, p, seed)
    out = {}
    for name, kind, spec in RULE_SPECS:
        run = dict(donor_run)
        if kind == "rule":
            run["policy"] = run["choices"] = None
            run["rule"] = spec
        else:
            # mine/swap.py's move: the target stage's rule settings, nothing else of
            # the target run.  Its measured value is carried along for reference.
            tgt = R.load_run(fam + spec, p, seed)
            out[("meas", name)] = tup(R.score(tgt["events"], tgt["t0"], p))
            for k in ("policy", "choices", "choice_jitter"):
                run[k] = tgt[k]
            run["rule"] = None
        reps = []
        for rep in range(REPS):
            # The SAME tick-lag stream seed for every rule. Only the seed is shared: the
            # rules consume different numbers of draws per poll (measured 1.3 to 2.2), so
            # the lag VALUES differ from the first poll on. What is paired across rules is
            # the exogenous input (outages, creation times) and the jitter stream seed.
            rng = random.Random(R.rep_seed(f"trickle:{donor}", p, seed, rep))
            reps.append(tup(R.score(R._sim_run(run, "B", sim_rng=rng), run["t0"], p)))
        out[("sim", name)] = reps
    return donor, p, seed, out


# --------------------------------------------------------------------------
# Cell values
# --------------------------------------------------------------------------

def pooled(rows, p):
    """Cell values from a list of per-run count tuples, pooled as evidence/11 pools them."""
    n4, c4, lat, polls, ans, to, emp = (sum(r[i] for r in rows) for i in range(7))
    k = len(rows)
    return dict(
        delay=lat / n4 if n4 else float("nan"),
        cens4=c4 / (n4 + c4) if (n4 + c4) else float("nan"),
        polls_sent=polls / k,
        answered=ans / k,
        timed_out=to / k,
        bytes=(ans * (R.INT_B + R.REP_B[p]) + to * R.INT_B) / k if p in R.REP_B else None,
        empty_share=emp / ans if ans else float("nan"),
    )


def seed_matrix(per_seed):
    """(n_seeds, 7) array: each seed's mean over the replicates.

    spec.md's B1 takes the mean over replicates of the pooled 20-seed value.  For a
    bootstrap the unit has to be the seed, so the replicate mean is taken per seed
    first and the pooling then happens inside each resample.  For sums over seeds the
    two orders agree exactly; for the ratios (delay, cens4, empty share) they differ in
    the last digits, because a ratio of means is not a mean of ratios.  Stated in
    trickle.md as a deviation.
    """
    return np.array([[sum(r[i] for r in reps) / len(reps) for i in range(7)]
                     for reps in per_seed], dtype=float)


def boot_values(arr, idx, p):
    """Pooled delay / polls_sent / cens4 for every bootstrap resample. idx: (BOOT_N, k)."""
    s = arr[idx]
    n4 = s[:, :, 0].sum(1)
    c4 = s[:, :, 1].sum(1)
    lat = s[:, :, 2].sum(1)
    polls = s[:, :, 3].sum(1)
    k = arr.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return lat / n4, polls / k, c4 / (n4 + c4)


# --------------------------------------------------------------------------
# The frontier of the four fixed arms, and the two readings of it
# --------------------------------------------------------------------------

def _pw(xs, ys, xv):
    """Piecewise-linear y(xv) through the points (xs, ys), row by row.

    xs/ys: (N, 4), xv: (N,).  Returns (y, inside) with y = nan where xv is outside the
    range of that row's xs.  Everything is already in log space.
    """
    order = np.argsort(xs, axis=1)
    xs = np.take_along_axis(xs, order, 1)
    ys = np.take_along_axis(ys, order, 1)
    inside = (xv >= xs[:, 0]) & (xv <= xs[:, -1])
    k = (xs < xv[:, None]).sum(1)
    i = np.clip(k - 1, 0, xs.shape[1] - 2)
    x0 = np.take_along_axis(xs, i[:, None], 1)[:, 0]
    x1 = np.take_along_axis(xs, (i + 1)[:, None], 1)[:, 0]
    y0 = np.take_along_axis(ys, i[:, None], 1)[:, 0]
    y1 = np.take_along_axis(ys, (i + 1)[:, None], 1)[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(x1 > x0, (xv - x0) / (x1 - x0), 0.0)
    return np.where(inside, y0 + t * (y1 - y0), np.nan), inside


def gain_save(fx_polls, fx_delay, v_polls, v_delay):
    """spec2's two readings of the fixed arms' frontier, in log-log space.

    gain = 1 - delay(variant) / delay_frontier(polls(variant)):  positive means the
    variant is faster than a fixed interval costing the same number of polls.
    save = 1 - polls(variant) / polls_frontier(delay(variant)):  the same statement
    read as polls saved at equal delay.
    Arrays are (N, 4) and (N,); nan where the variant is outside the fixed arms' range.
    """
    lp, ld = np.log(fx_polls), np.log(fx_delay)
    fd, in_d = _pw(lp, ld, np.log(v_polls))
    fp, in_p = _pw(ld, lp, np.log(v_delay))
    return 1.0 - v_delay / np.exp(fd), 1.0 - v_polls / np.exp(fp), in_d, in_p


def ci(v):
    v = v[np.isfinite(v)]
    if v.size < BOOT_N * 0.5:
        return (None, None, v.size)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), v.size)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def published_empty_share():
    """evidence/11|12 cells.csv rx_no_new / state_rx, the measured EMPTY share.

    Reference only: the logged `state_rx` lines carry no `no_new` field, so the
    simulated EMPTY share cannot be recomputed from the logs by this script's own
    scorer; this is the published value for the same cell.
    """
    out = {}
    for f in ("11", "12"):
        path = f"{EV}/{f}/cells.csv"
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            k = (r["stage"], float(r["p"]))
            if k in out or not r.get("rx_no_new") or not r.get("state_rx"):
                continue
            try:
                out[k] = float(r["rx_no_new"]) / float(r["state_rx"])
            except (ValueError, ZeroDivisionError):
                pass
    return out


def main():
    t_start = time.time()
    print("identity test (spec2 Part 2) ...", flush=True)
    idn = subprocess.run([PY, "-B", os.path.join(HERE, "identity.py")],
                         capture_output=True, text=True)
    sys.stdout.write(idn.stdout)
    if "IDENTITY HOLDS" not in idn.stdout:
        sys.exit("identity test failed; refusing to report variant numbers")

    import multiprocessing as mp
    jobs = [(d, p, s) for (d, p) in CELLS for s in SEEDS]
    with mp.Pool(WORKERS) as pool:
        res = pool.map(job, jobs, chunksize=1)
    by_cell = defaultdict(dict)
    for donor, p, seed, out in res:
        by_cell[(donor, p)][seed] = out
    pub = published_empty_share()

    cell_rows, gain_rows = [], []
    for (donor, p) in CELLS:
        fam = donor[:2]
        per = by_cell[(donor, p)]
        arrs, vals = {}, {}
        for name in NAMES:
            arrs[name] = seed_matrix([per[s][("sim", name)] for s in SEEDS])
            vals[name] = pooled([r for s in SEEDS for r in per[s][("sim", name)]], p)
        rng = np.random.default_rng(BOOT_SEED)
        idx = rng.integers(0, len(SEEDS), size=(BOOT_N, len(SEEDS)))
        boot = {n: boot_values(arrs[n], idx, p) for n in NAMES}

        for name, kind, spec in RULE_SPECS:
            v = vals[name]
            meas = None
            if kind != "rule":
                meas = pooled([per[s][("meas", name)] for s in SEEDS], p)
            cell_rows.append(dict(
                donor=donor, p=p, rule=name, kind=kind, settings=interval_text(name),
                sim_delay=v["delay"], sim_cens4=v["cens4"], sim_polls_sent=v["polls_sent"],
                sim_answered=v["answered"], sim_timed_out=v["timed_out"],
                sim_bytes=v["bytes"], sim_empty_share=v["empty_share"],
                meas_stage=(fam + spec) if kind != "rule" else "",
                meas_delay=meas["delay"] if meas else None,
                meas_cens4=meas["cens4"] if meas else None,
                meas_polls_sent=meas["polls_sent"] if meas else None,
                meas_bytes=meas["bytes"] if meas else None,
                meas_empty_share=pub.get((fam + spec, p)) if kind != "rule" else None,
            ))

        # the frontier's own monotonicity, on the point estimate
        fx = sorted(FIXED, key=lambda n: vals[n]["polls_sent"])
        mono = all(vals[fx[i]]["delay"] > vals[fx[i + 1]]["delay"] for i in range(3))
        fxp = np.array([[vals[n]["polls_sent"] for n in FIXED]])
        fxd = np.array([[vals[n]["delay"] for n in FIXED]])
        bfp = np.stack([boot[n][1] for n in FIXED], axis=1)
        bfd = np.stack([boot[n][0] for n in FIXED], axis=1)

        for name in TESTED:
            g, s, in_d, in_p = gain_save(
                fxp, fxd, np.array([vals[name]["polls_sent"]]),
                np.array([vals[name]["delay"]]))
            bg, bs, _, _ = gain_save(bfp, bfd, boot[name][1], boot[name][0])
            glo, ghi, gn = ci(bg)
            slo, shi, sn = ci(bs)
            # cens4 of the two fixed arms bracketing this rule's polls_sent
            xs = sorted(FIXED, key=lambda n: vals[n]["polls_sent"])
            px = [vals[n]["polls_sent"] for n in xs]
            k = min(max(sum(1 for q in px if q < vals[name]["polls_sent"]) - 1, 0), 2)
            nb = [xs[k], xs[k + 1]]
            nb_cens = max(vals[n]["cens4"] for n in nb)
            gain_rows.append(dict(
                donor=donor, p=p, rule=name, settings=interval_text(name),
                polls_sent=vals[name]["polls_sent"], delay=vals[name]["delay"],
                cens4=vals[name]["cens4"], bytes=vals[name]["bytes"],
                empty_share=vals[name]["empty_share"],
                gain=float(g[0]) if in_d[0] else None,
                gain_lo=glo if in_d[0] else None, gain_hi=ghi if in_d[0] else None,
                save=float(s[0]) if in_p[0] else None,
                save_lo=slo if in_p[0] else None, save_hi=shi if in_p[0] else None,
                outside=("" if in_d[0] and in_p[0] else
                         ("polls outside" if not in_d[0] else "") +
                         ("delay outside" if not in_p[0] else "")),
                frontier_monotone=int(mono),
                neighbours="+".join(nb), neighbour_cens4=nb_cens,
                cens_flag=int(vals[name]["cens4"] - nb_cens > CENS_FLAG_PP),
                boot_n_gain=gn, boot_n_save=sn,
            ))
        print(f"  {donor} p{p}: frontier monotone={mono}", flush=True)

    wall = time.time() - t_start
    for rows, path in ((cell_rows, "trickle_cells.csv"), (gain_rows, "trickle_gain.csv")):
        with open(os.path.join(HERE, path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (R.fmt(v, 6) if isinstance(v, float) else v)
                            for k, v in r.items()})
    write_md(cell_rows, gain_rows, wall, idn.stdout)
    print("wrote trickle_cells.csv, trickle_gain.csv, trickle.md")
    print(f"wall clock {wall:.1f} s "
          f"({len(CELLS)*len(SEEDS)*len(NAMES)*REPS} simulations)")


def f(x, nd=1, pct=False):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{100*x:+.{nd}f} %" if pct else f"{x:.{nd}f}"


def ci_text(lo, hi, n):
    """A bootstrap interval, with the number of resamples it rests on whenever that is
    not all of them.

    A resample in which the rule falls outside the four fixed arms' range has no gain
    (spec2's own rule), so it contributes nothing.  When that happens often the interval
    is conditional on being inside, which is a different statement from the point
    estimate; the count is printed so the reader sees it, and below half the resamples
    no interval is printed at all.
    """
    tail = "" if n >= BOOT_N else f" (from {n}/{BOOT_N} resamples inside the frontier)"
    if lo is None:
        return f"- ({n}/{BOOT_N} resamples inside the frontier)"
    return f"{f(lo, 1, True)} .. {f(hi, 1, True)}{tail}"


def write_md(cell_rows, gain_rows, wall, identity_out):
    L = []
    W = L.append
    W("# ev13 trickle: the spec2 rule replayed against fixed intervals, short and wait\n")
    W(f"Generated by `trickle.py` ({time.strftime('%Y-%m-%d %H:%M:%S')}), "
      f"{len(CELLS)*len(SEEDS)*len(NAMES)*REPS} simulations, {WORKERS} workers, "
      f"wall clock {wall:.0f} s.\n")
    W("Everything below is **simulated against simulated**, paired by seed and replicate "
      "on one donor arm's exogenous inputs (spec2 Part 3). The measured columns are "
      "printed for reference and nothing is scored against them.\n")

    W("## The rules\n")
    W("| name | kind | settings |")
    W("|---|---|---|")
    for name, kind, spec in RULE_SPECS:
        W(f"| {name} | {kind} | {interval_text(name)} |")
    W("")
    W("The rule (spec2 Part 2): one interval I per (node, neighbor), I = I_min at start; "
      "at send `applied = I * (1 + jf*(2u-1))` and the next poll is provisionally "
      "`sent + applied`; at an answered resolution `I <- min(I*m, I_max)` if the reply was "
      "EMPTY and `I <- I_min` otherwise, and the next poll becomes "
      "`max(sent + I_new*(1 + jf*(2u-1)), now)` with the same u; at a timed-out resolution "
      "I is unchanged. EMPTY is evidence/11's `rx_no_new`: every seq 1..hi[o] of every "
      "origin o in the reply is already held by the polling node at that instant, and a "
      "reply with `hi == {}` counts as EMPTY.\n")

    W("## Identity test (spec2 Part 2), run before any variant number was read\n")
    W("```")
    for line in identity_out.strip().splitlines():
        W(line)
    W("```")
    W("")

    W("## What came out\n")
    W("The headline question spec2 fixed before any of this was computed: *at the same "
      "polls_sent, is the variant's delay lower than a fixed interval's?* Over the 7 cells:\n")
    W("| rule | gain > 0 | CI entirely above 0 | CI entirely below 0 | gain min .. max |")
    W("|---|---|---|---|---|")
    for name in TESTED:
        rs = [r for r in gain_rows if r["rule"] == name and r["gain"] is not None]
        pos = sum(1 for r in rs if r["gain"] > 1e-9)
        up = sum(1 for r in rs if r["gain_lo"] is not None and r["gain_lo"] > 0)
        dn = sum(1 for r in rs if r["gain_hi"] is not None and r["gain_hi"] < 0)
        W(f"| {name} | {pos}/{len(rs)} | {up}/{len(rs)} | {dn}/{len(rs)} | "
          f"{f(min(r['gain'] for r in rs), 1, True)} .. "
          f"{f(max(r['gain'] for r in rs), 1, True)} |")
    W("")
    W("A gain within 1e-9 of zero is counted as zero (a rule that is itself a frontier knot, "
      "as short and wait are at s5b10p043 p 0 where there is no outage, scores a rounding "
      "residue of 1e-16). What the table means, the bias of the chord frontier and the "
      "measurements added after these results were seen are in README.md and after.md; "
      "this file only carries what trickle.py computed.\n")
    W("")
    W("## Headline: gain at equal polls, and polls saved at equal delay\n")
    W("`gain = 1 - delay(rule) / delay_frontier(polls_sent(rule))`, the frontier being the "
      "piecewise-linear log(delay) over log(polls_sent) through the four SIMULATED fixed "
      "arms of the same cell. Positive = faster than a fixed interval that costs the same "
      "number of polls. `save = 1 - polls_sent(rule) / polls_frontier(delay(rule))` is the "
      "same statement read the other way. 95 % intervals: 2000-resample bootstrap over the "
      f"20 seeds, `default_rng({BOOT_SEED})`, the frontier and the rule recomputed from the "
      "same resampled seeds in every resample. A resample that puts the rule outside the "
      "four fixed arms' range yields no value, so an interval that rests on fewer than all "
      f"{BOOT_N} resamples says so, and below half of them no interval is printed.\n")
    for (donor, p) in CELLS:
        rows = [r for r in gain_rows if r["donor"] == donor and r["p"] == p]
        cells = [r for r in cell_rows if r["donor"] == donor and r["p"] == p]
        W(f"### donor {donor}, p {p}\n")
        W("| rule | settings | polls_sent | delay ms | cens4 | EMPTY share | gain | "
          "gain 95 % | save | save 95 % | cens flag |")
        W("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in cells:
            if r["kind"] != "fixed":
                continue
            W(f"| *{r['rule']}* (frontier) | {r['settings']} | {f(r['sim_polls_sent'])} | "
              f"{f(r['sim_delay'], 0)} | {f(r['sim_cens4'], 4)} | "
              f"{f(r['sim_empty_share'], 3)} | - | - | - | - | - |")
        for r in rows:
            W(f"| **{r['rule']}** | {r['settings']} | {f(r['polls_sent'])} | "
              f"{f(r['delay'], 0)} | {f(r['cens4'], 4)} | {f(r['empty_share'], 3)} | "
              f"{f(r['gain'], 1, True)} | {ci_text(r['gain_lo'], r['gain_hi'], r['boot_n_gain'])} | "
              f"{f(r['save'], 1, True)} | {ci_text(r['save_lo'], r['save_hi'], r['boot_n_save'])} | "
              f"{('YES, +' + f(100*(r['cens4']-r['neighbour_cens4']), 2) + ' pp over ' + r['neighbours']) if r['cens_flag'] else 'no'} |")
        out = [r["rule"] + ": " + r["outside"] for r in rows if r["outside"]]
        if out:
            W("")
            W("Outside the four fixed arms' range, so spec2 gives that reading no value "
              "(`polls outside` suppresses `gain`, `delay outside` suppresses `save`): "
              + "; ".join(out) + ". `delay outside` here means the rule is SLOWER than the "
              "slowest fixed arm, so there is no fixed interval whose delay it matches and "
              "the question \"how many polls does it save at equal delay\" has no answer; "
              "its `gain` is still defined, because its poll count does lie between two "
              "fixed arms.")
        if not rows[0]["frontier_monotone"]:
            W("")
            W("**The fixed arms' frontier is not monotone in this cell**: more polls do not "
              "buy less delay across all four arms, so `save` (the inverse reading) is "
              "interpolated through a non-monotone curve and is not trustworthy here. "
              "Reported as it came out rather than patched.")
        W("")

    W("## Every cell x rule\n")
    W("| donor | p | rule | polls_sent | delay ms | cens4 | timed_out | bytes | EMPTY share "
      "| measured stage | measured polls | measured delay | measured EMPTY share |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in cell_rows:
        W(f"| {r['donor']} | {r['p']} | {r['rule']} | {f(r['sim_polls_sent'])} | "
          f"{f(r['sim_delay'], 0)} | {f(r['sim_cens4'], 4)} | {f(r['sim_timed_out'])} | "
          f"{f(r['sim_bytes'], 0)} | {f(r['sim_empty_share'], 3)} | "
          f"{r['meas_stage'] or '-'} | {f(r['meas_polls_sent'])} | "
          f"{f(r['meas_delay'], 0)} | {f(r['meas_empty_share'], 3)} |")
    W("")

    W("## Assumptions and deviations from spec2\n")
    W("1. **Bytes use the same `REP_B[p]` for every rule**, as spec2 states: the mean reply "
      "size is taken not to depend on the rule. `REP_B` exists for p 0 / 0.2 / 0.4 only, so "
      "the p 0.5 / 0.6 cells have no byte column, which is why the headline is stated in "
      "polls_sent and not in bytes.")
    W("2. **The bootstrap's unit is the seed, and each seed's value is the mean over its 5 "
      "replicates.** spec.md's B1 averages the replicates after pooling the seeds; for sums "
      "the two orders agree exactly, for the ratios (delay, cens4, EMPTY share) they differ "
      "in the last digits. The point estimates in the tables are pooled over all 100 "
      "(seed, replicate) runs, as spec.md pools them; only the bootstrap uses the per-seed "
      "replicate mean, because a bootstrap needs one value per unit.")
    W("3. **The EMPTY share of a measured cell is the published `rx_no_new / state_rx`** of "
      "evidence/11|12 `cells.csv`, not something this script recomputed: the logged "
      "`state_rx` lines carry no emptiness field, and the simulated figure comes from the "
      "engine's own store at the serve instant. The two are the same definition but not the "
      "same code path, so the measured EMPTY column is weaker evidence than the other "
      "measured columns.")
    W("4. **The tick-lag stream seed is the same for all 11 rules** at one (p, seed, "
      "replicate). Only the seed is shared: the rule path draws a lag at send and again "
      "when a resolution moves the due time, so it consumes 1.7 to 2.2 draws per poll "
      "against 1.3 to 1.4 for the fixed and table paths (measured in review), and the lag "
      "values differ from the first interval change on. What is paired across rules is the "
      "exogenous input (outages, creation times, agent start) and the jitter stream seed. "
      "The lag is at most 100 ms against intervals of 500 ms and more.")
    W("5. **`short` and `wait` take the policy table of the donor's own family** "
      "(`s6short`/`s6wait` for the s6 donor, `s5short`/`s5wait` for the s5 donor), as spec2 "
      "says. The two families' tables happen to be identical.")
    W("6. **No tuning.** The five variants listed in spec2 are the only rules run. Their "
      "numbers are reported as they came out.")
    W("7. **The RNG stream usage of the two code paths** (spec2 Part 2 asks for this to be "
      "said out loud). (a) The choices path draws `state_poll_interval:{seed}:{node}` once "
      "per poll to pick its interval; the rule path does not, because the rule has no choice "
      "to make. That stream is a separate `random.Random` instance from the jitter stream, "
      "so leaving it undrawn changes no jitter value, and no dummy draw was added. (b) The "
      "rule path draws its tick lag at SEND, where the choices path draws it, and the "
      "resolution re-arms the timer only when the recomputed due instant differs from the "
      "provisional one; the policy-table path instead defers the draw to the due instant "
      "(validate.md deviation 8). The two paths differ on purpose: each mirrors the path it "
      "has to stay bit-identical to. (c) With m = 1 and I_min = I_max the rule's due instant "
      "never differs, so the rule path consumes the single shared tick-lag stream in exactly "
      "the order the fixed arm does -- which is what the identity test above measures, in "
      "Mode B as well as Mode A.")
    W("8. **The identity test is exact on every event field but one.** "
      "`state.consecutive_poll_timeouts` of a `state_poll` decision differs: the choices path "
      "never maintains the counter and logs 0, the rule path maintains it when the NEWEST "
      "poll resolves (the source also folds older unanswered polls into it after a 1 s "
      "grace, engine.py `_PollReplies._expire`; the replay does not, so under the two "
      "sub-2 s variants the logged value is lower than the source's would be). It is "
      "a logged diagnostic that no metric, pass line or scheduling decision reads; every poll "
      "time, interval, reply, store time and cell metric is identical. Reported rather than "
      "silently normalised away.")
    W("   What the identity test does and does not show: with m = 1 the due time never "
      "moves at a resolution, so the test covers the EMPTY computation, the resolution "
      "bookkeeping and the reuse of the send-time jitter draw, but not the grow / cap "
      "arithmetic or the re-arm branch every real variant takes. Those were checked in "
      "review by an independent model of one (node, neighbor) pair: 2134 polls and 1932 "
      "resolutions over the five variants, 0 differences (review2-rule/mysim.py, not part "
      "of this directory).")
    W("9. **\"Outside the frontier\" is applied per axis.** spec2 says a variant outside the "
      "range of the four fixed arms gets no gain, without naming an axis. `gain` interpolates "
      "over `polls_sent` and `save` over `delay`, so a rule can be inside one and outside the "
      "other; each reading is suppressed on its own axis. In this run only T136c is outside, "
      "and only on delay (it is slower than the slowest fixed arm).")
    W("10. **A bootstrap interval resting on fewer than half the resamples is not printed.** "
      "spec2 does not say what to do when a resample puts the rule outside the frontier. The "
      "count of resamples an interval does rest on is printed whenever it is not all "
      f"{BOOT_N} of them, so a conditional interval cannot be mistaken for an unconditional "
      "one.")
    W("")
    W("### What the new rule makes observable that the logs never showed\n")
    W("facts-timers.md §9 lists two source branches as UNCONFIRMED because no logged run "
      "could reach them: `clamped == true` reschedules (0 of 37 423) and the "
      "`POLL_REPLY_GRACE_S` / `expired_timeouts` bookkeeping. Both need a poll interval "
      "below the 2 s interest lifetime, which T05 and T136 have. Under those two variants "
      "the replay therefore runs source paths that no measurement in this store validates: "
      "the provisional send-time timer firing before its poll resolves (engine.py:478), a "
      "poll that is never resolved because a newer one has been sent (engine.py:503), and "
      "the clamp at engine.py:522. Their numbers rest on reading the source, not on a "
      "logged comparison.\n")
    with open(os.path.join(HERE, "trickle.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
