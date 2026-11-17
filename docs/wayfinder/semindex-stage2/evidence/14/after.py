"""A check added AFTER the results of workload.py were seen (not in spec.md).

Why: review found a confound that spec.md's cens flag cannot catch. The replaced workloads create records
up to t0+110 s while the window ends at t0+130 s, so a record created late is censored (not delivered
4 hops away in time) far more often under a sparse rule than under a dense one, and `delay` is the mean
over uncensored records only. The fixed 10.043 s arm loses 6 to 26 % of its records that way in the
outage cells, which makes the sparse end of the frontier look faster than it is and tilts the comparison
against a rule that lands there (T37, short, wait). The flag only fires when the RULE is more censored
than its bracketing arms, never the reverse.

What this does: the same simulations as workload.py (same generators, same seeds, same replicate streams),
scored on the records created at or before t0+72 s only. Every such record has at least 58 s to arrive,
the margin the testbed workload has (its last record is created at t0+71 s), so censoring is small for
every rule and the delay means are over nearly the same records. Polls are still counted over the whole
window, as in workload.py. Gain, frontier and bootstrap as in spec.md, written out here independently of
workload.py's and trickle.py's code (only the generators, the rule table and the simulator are imported).

Output: after.md. Usage: python3 after.py
"""
import math
import multiprocessing as mp
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "13"))
sys.path.insert(0, HERE)
import replay as R  # noqa: E402
import trickle as T  # noqa: E402
import workload as W  # noqa: E402

CUT_MS = 72_000.0
WE_MS = R.WINDOW_END_MS


def early_score(ev, t0):
    """(n4, c4, lat4 ms, polls): n4, c4, lat4 over records created at or before t0 + CUT_MS;
    polls over the whole window, as workload.py counts them."""
    created, stored, polls = {}, {}, 0
    for node, rows in ev.items():
        for e in rows:
            k = e.get("event")
            if k == "record_created":
                created[(e["origin"], e["seq"])] = e["ts_ms"]
            elif k == "record_stored":
                key = (e["origin"], e["seq"], node)
                if key not in stored or e["ts_ms"] < stored[key]:
                    stored[key] = e["ts_ms"]
            elif k == "decision" and e.get("kind") == "state_poll":
                polls += t0 + R.WINDOW_START_MS <= e["ts_ms"] <= t0 + WE_MS
    n4 = c4 = 0
    lat = 0.0
    for (o, s), c in created.items():
        if c > t0 + CUT_MS:
            continue
        ts = stored.get((o, s, "h4" if o == "h0" else "h0"))
        if ts is None or ts > t0 + WE_MS:
            c4 += 1
        else:
            n4 += 1
            lat += ts - c
    return n4, c4, lat, polls


def job(arg):
    donor, p, seed = arg
    d = R.load_run(donor, p, seed)
    fam = donor[:2]
    out = {}
    for wl in W.WORKLOADS:
        created = W.gen_created(wl, p, seed, d["t0"], d["created"])
        for name, kind, spec in W.RULE_SPECS:
            run = dict(d)
            run["created"] = created
            if kind in ("fixed", "policy"):
                tgt = R.load_run(fam + spec, p, seed)
                for k in ("policy", "choices", "choice_jitter"):
                    run[k] = tgt[k]
                run["rule"] = None
            else:
                run["policy"] = run["choices"] = None
                run["rule"] = spec if kind == "rule" else dict(i_min=spec, m=1.0, i_max=spec, jf=T.JF)
            acc = np.zeros(4)
            for rep in range(W.REPS):
                rng = random.Random(R.rep_seed(f"trickle:{donor}", p, seed, rep))
                acc += early_score(R._sim_run(run, "B", sim_rng=rng), d["t0"])
            out[(wl, name)] = acc / W.REPS  # the seed's value: mean over its replicates
    return donor, p, seed, out


def interp(xs, ys, x):
    """log-log piecewise-linear y(x) through the points, None outside their range."""
    pts = sorted(zip(xs, ys))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + w * (math.log(y1) - math.log(y0)))
    return None


def cell_vals(mat):
    """mat: (seeds, 4) of n4, c4, lat, polls -> delay ms, cens share, polls per run."""
    n4, c4, lat, polls = mat.sum(0)
    return lat / n4, c4 / (n4 + c4), polls / len(mat)


def main():
    jobs = [(d, p, s) for d, p in W.CELLS for s in W.SEEDS]
    with mp.Pool(12) as pool:
        res = pool.map(job, jobs)
    idx = np.random.default_rng(W.BOOT_SEED).integers(0, len(W.SEEDS), size=(W.BOOT_N, len(W.SEEDS)))
    L = ["# after.md: the comparison on early records only (see after.py; added after the results were seen)", "",
         f"Records created at or before t0+{CUT_MS / 1000:.0f} s. gain as in spec.md, against the seven fixed arms.",
         "cens = share of those records not delivered 4 hops away by t0+130 s; arms = the two fixed arms nearest",
         "to the rule's poll count (in log), with their cens. They are shown for the censoring only; the frontier",
         "uses all seven arms. spec.md's 1e-9 rule is not applied here, so a rule that is itself an arm shows",
         "+0.0 % or -0.0 %.", "",
         "| workload | cell | rule | polls | delay ms | cens | arms (cens) | gain | gain 95 % | gain in workload.md |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    nboot = []
    import csv
    pub = {(r["workload"], r["donor"], float(r["p"]), r["rule"]): r for r in
           csv.DictReader(open(os.path.join(HERE, "workload_gain.csv")))}
    for wl in W.WORKLOADS:
        for donor, p in W.CELLS:
            rr = sorted((r for r in res if (r[0], r[1]) == (donor, p)), key=lambda r: r[2])
            mats = {n: np.array([r[3][(wl, n)] for r in rr]) for n in W.NAMES}
            fx = {n: cell_vals(mats[n]) for n in W.FIXED}
            for name in W.TESTED:
                dly, cens, polls = cell_vals(mats[name])
                fr = interp([fx[n][2] for n in W.FIXED], [fx[n][0] for n in W.FIXED], polls)
                gain = None if fr is None else 1 - dly / fr
                near = sorted(W.FIXED, key=lambda n: abs(math.log(fx[n][2] / polls)))[:2]
                boots = []
                for row in idx:
                    f2 = {n: cell_vals(mats[n][row]) for n in W.FIXED}
                    d2, _, p2 = cell_vals(mats[name][row])
                    f = interp([f2[n][2] for n in W.FIXED], [f2[n][0] for n in W.FIXED], p2)
                    if f is not None:
                        boots.append(1 - d2 / f)
                nboot.append(len(boots))
                lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) >= W.BOOT_N / 2 else (None, None))
                pg = pub[(wl, donor, p, name)]["gain"]
                L.append(f"| {wl} | {donor[:2]} p{p} | {name} | {polls:.1f} | {dly:.0f} | {100 * cens:.1f} % | "
                         + ", ".join(f"{n} ({100 * fx[n][1]:.1f} %)" for n in near) + " | "
                         + ("-" if gain is None else f"{100 * gain:+.1f} %") + " | "
                         + ("-" if lo is None else f"{100 * lo:+.1f} .. {100 * hi:+.1f}") + " | "
                         + (f"{100 * float(pg):+.1f} %" if pg else "-") + " |")
    L += ["", f"Resamples each interval rests on: min {min(nboot)}, max {max(nboot)} of {W.BOOT_N}."]
    open(os.path.join(HERE, "after.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
