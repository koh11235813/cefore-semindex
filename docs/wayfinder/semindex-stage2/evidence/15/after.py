"""Checks added AFTER the results of onoff.py were seen (not in spec.md).

Why: spec.md's reading of inference 2 for T37r came out "supported" with 4 of 6 cases, and the fourth case
has a lower bound of +0.0003 %. A verdict that thin can be made or unmade by things spec.md did not fix: which
resamples the bootstrap happened to draw, how the percentile is interpolated, and where the early-record cut
is put. Review showed all three move it. This script measures how much, with scoring, frontier and bootstrap
written here independently of onoff.py. Imported from it: the generators, the engine, the rule table and the
shared constants (cells, seeds, replicates, bootstrap size and seed), so a wrong constant there would be
repeated here, not caught:

- per outage cell and workload: the headline gain of T37 and T37r, their paired difference, and how many of
  the 2000 resamples put the difference at or below zero (the count behind the lower bound);
- the same at an earlier cut (t0+60 s), where every rule's censoring is smaller;
- how many of 25 bootstrap seeds read T37r "supported" (lower bound above zero in at least 4 of 6 cases).

It also recounts the headline numbers of onoff_gain.csv at the spec's own cut and seed.
Output: after.md. Usage: python3 after.py
"""
import csv
import math
import multiprocessing as mp
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("13", "14"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
sys.path.insert(0, HERE)
import onoff as O  # noqa: E402
import replay as R  # noqa: E402

CUTS_MS = (72_000.0, 60_000.0)
RULES = list(O.FIXED) + ["T37", "T37r", "T37b"]
OUTAGE_CELLS = [c for c in O.CELLS if c != O.NO_OUTAGE]
SPEC = {n: (k, s) for n, k, s in O.RULE_SPECS}


def early(ev, t0, cut_ms):
    """(n4, c4, lat4 ms, polls): n4, c4, lat4 over records created at or before t0 + cut_ms,
    polls over the whole window."""
    we = t0 + R.WINDOW_END_MS
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
                polls += t0 + R.WINDOW_START_MS <= e["ts_ms"] <= we
    n4 = c4 = 0
    lat = 0.0
    for (o, s), c in created.items():
        if c > t0 + cut_ms:
            continue
        ts = stored.get((o, s, "h4" if o == "h0" else "h0"))
        if ts is None or ts > we:
            c4 += 1
        else:
            n4 += 1
            lat += ts - c
    return n4, c4, lat, polls


def job(arg):
    donor, p, seed = arg
    d = R.load_run(donor, p, seed)
    out = {}
    for wl in O.WORKLOADS:
        created = O.gen_created(wl, p, seed, d["t0"], d["created"])
        for name in RULES:
            kind, spec = SPEC[name]
            run = dict(d)
            run["created"] = created
            if kind in ("fixed", "policy"):
                tgt = R.load_run(donor[:2] + spec, p, seed)
                for k in ("policy", "choices", "choice_jitter"):
                    run[k] = tgt[k]
                run["rule"] = None
            else:
                run["policy"] = run["choices"] = None
                # kind "arm": one of the three added fixed arms, run as the rule with m = 1 (../14/workload.py)
                run["rule"] = spec if kind == "rule" else dict(i_min=spec, m=1.0, i_max=spec, jf=O.T.JF)
            acc = {c: np.zeros(4) for c in CUTS_MS}
            for rep in range(O.REPS):
                ev = O.sim_run(run, random.Random(R.rep_seed(f"trickle:{donor}", p, seed, rep)))
                for c in CUTS_MS:
                    acc[c] += early(ev, d["t0"], c)
            for c in CUTS_MS:
                out[(wl, name, c)] = acc[c] / O.REPS
    return donor, p, seed, out


def interp(xs, ys, x):
    pts = sorted(zip(xs, ys))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + w * (math.log(y1) - math.log(y0)))
    return None


def vals(mat):
    n4, c4, lat, polls = mat.sum(0)
    return lat / n4, polls / len(mat), c4 / (n4 + c4)


def gain(mats, name):
    fx = [vals(mats[n]) for n in O.FIXED]
    d, p, _ = vals(mats[name])
    f = interp([x[1] for x in fx], [x[0] for x in fx], p)
    return None if f is None else 1 - d / f


def main():
    with mp.Pool(12) as pool:
        res = pool.map(job, [(d, p, s) for d, p in O.CELLS for s in O.SEEDS])
    pub = {(r["workload"], r["donor"], float(r["p"]), r["rule"]): r
           for r in csv.DictReader(open(os.path.join(HERE, "onoff_gain.csv")))}
    n = len(O.SEEDS)
    idx0 = np.random.default_rng(O.BOOT_SEED).integers(0, n, size=(O.BOOT_N, n))
    L = ["# after.md: how firm the reading of T37r is (see after.py; added after the results were seen)", "",
         "gain and diff in %, diff = gain(T37r) - gain(T37), intervals from 2000 seed resamples "
         f"(default_rng({O.BOOT_SEED})). '<= 0' = resamples with diff at or below zero; the 2.5 % point sits",
         "between the 50th and 51st smallest, so 50 or more means the lower bound is not above zero under every",
         "percentile convention.", ""]
    stab = {c: [] for c in CUTS_MS}
    for cut in CUTS_MS:
        L += [f"## cut t0+{cut / 1000:.0f} s", "",
              "| workload | cell | cens T37 / T37r / fix10.043 | T37 gain | T37r gain | T37r gain 95 % | diff | diff 95 % | <= 0 | onoff_gain.csv diff |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        cases = []
        for wl in O.WORKLOADS:
            for donor, p in OUTAGE_CELLS:
                rr = sorted((r for r in res if (r[0], r[1]) == (donor, p)), key=lambda r: r[2])
                mats = {m: np.array([r[3][(wl, m, cut)] for r in rr]) for m in RULES}
                cases.append(mats)
                g0, g1 = gain(mats, "T37"), gain(mats, "T37r")
                bd, b1 = [], []
                for row in idx0:
                    sub = {m: mats[m][row] for m in RULES}
                    a, b = gain(sub, "T37"), gain(sub, "T37r")
                    if a is not None and b is not None:
                        bd.append(b - a)
                        b1.append(b)
                lo, hi = np.percentile(bd, [2.5, 97.5])
                l1, h1 = np.percentile(b1, [2.5, 97.5])
                pr = pub[(wl, donor, p, "T37r")]["diff"] if cut == CUTS_MS[0] else ""
                c = [100 * vals(mats[m])[2] for m in ("T37", "T37r", "fix10.043")]
                L.append(f"| {wl} | {donor[:2]} p{p} | {c[0]:.1f} / {c[1]:.1f} / {c[2]:.1f} % | {100 * g0:+.1f} | {100 * g1:+.1f} | "
                         f"{100 * l1:+.1f} .. {100 * h1:+.1f} | {100 * (g1 - g0):+.1f} | {100 * lo:+.4f} .. {100 * hi:+.4f} | "
                         f"{sum(x <= 0 for x in bd)}/{len(bd)} | " + (f"{100 * float(pr):+.1f}" if pr else "-") + " |")
        # seed sensitivity: the verdict under 25 other resample draws
        for bs in range(25):
            idx = np.random.default_rng(1000 + bs).integers(0, n, size=(O.BOOT_N, n))
            up = 0
            for mats in cases:
                bd = []
                for row in idx:
                    sub = {m: mats[m][row] for m in RULES}
                    a, b = gain(sub, "T37"), gain(sub, "T37r")
                    if a is not None and b is not None:
                        bd.append(b - a)
                up += np.percentile(bd, 2.5) > 0
            stab[cut].append(int(up))
        L += ["", "Cases (of 6) with the lower bound above zero under 25 other bootstrap seeds (default_rng(1000..1024)): "
              + ", ".join(f"{k} cases x{stab[cut].count(k)}" for k in sorted(set(stab[cut]))) + ".", ""]
    open(os.path.join(HERE, "after.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
