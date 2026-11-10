"""Replay predictions for the s5 policy runs (replay.py in this directory, a verbatim copy of taf-grant-2026 value-check).

(b) paired: the very (p, seed) schedules we ran, seeds 1-20, 10 jitter reps per seed, validate.py's seeding
    random.Random(hash((tag..., p, s, rep)) & 0xffffffff) with numeric tags only (number hashes are not salted).
(a') published re-run: policies.py's set-up (s4a schedules, seeds 1-100, 4 reps, its arithmetic seeding) for the
    rules we compare, to recover polls/run (not in policy_table.csv) and to check the published numbers reproduce.
Output: replay_perseed.csv (one row per set, policy, p, seed: means over reps of delay ms, B/rh, waste, polls, cnt).
"""
import csv
import multiprocessing as mp
import os
import random
import sys

import replay as R

OUT = os.path.dirname(os.path.abspath(__file__))
POL = {  # name -> (policy, numeric hash tag, live stage whose schedule the paired set uses)
    "shorten (10.043, 3.695)": (R.Table((10.043, 3.695)), (10.043, 3.695), "s5short"),
    "slotblind 10.043": (R.SlotBlind(10.043), (10.043, 5.0), "s5wait"),
    "fixed 0.5": (R.Fixed(0.5), (0.5,), "s5b0p5"),
    "fixed 1.359": (R.Fixed(1.359), (1.359,), "s5b1p359"),
    "fixed 3.695": (R.Fixed(3.695), (3.695,), "s5b3p695"),
    "fixed 10.043": (R.Fixed(10.043), (10.043,), "s5b10p043"),
    "fixed 5.0": (R.Fixed(5.0), (5.0,), "s4a"),
    "shorten-engine (10.043, 3.695)": (R.Table((10.043, 3.695), ideal=False), None, None),
}
PS = (0.0, 0.2, 0.4)


def task(t):
    """One (set, policy, p, seed): mean over reps."""
    st, name, p, s = t
    pol, tag, stage = POL[name]
    if st == "paired":
        links, rec = R.load_schedule(p, s, stage)
        reps = 10
        seeds = [random.Random(hash(tag + (p, s, rep)) & 0xffffffff) for rep in range(reps)]
    else:
        links, rec = R.load_schedule(p, s)
        reps = 4
        seeds = [random.Random(1000003 * s + 7919 * rep + int(p * 10)) for rep in range(reps)]
    links = [R.Link(iv) for iv in links]
    total = len(rec["0"]) + len(rec["4"])
    acc = dict(d=[], b=[], w=[], n=[], c=[])
    for rng in seeds:
        d, b, w, c, npoll = R.simulate(pol, links, rec, p, rng)
        if d == d:
            acc["d"].append(d * 1000)
        acc["b"].append(b); acc["w"].append(w); acc["n"].append(npoll); acc["c"].append(1 - c / total)
    m = lambda v: sum(v) / len(v) if v else float("nan")
    return dict(set=st, policy=name, p=p, seed=s, delay_ms=m(acc["d"]), bytes_rh=m(acc["b"]), waste=m(acc["w"]),
                polls=m(acc["n"]), cens=m(acc["c"]))


def main():
    tasks = []
    for name, (_pol, tag, stage) in POL.items():
        for p in PS:
            if tag is not None:
                tasks += [("paired", name, p, s) for s in range(1, 21)]
            tasks += [("pub", name, p, s) for s in range(1, 101)]
    with mp.get_context("fork").Pool(3) as pool:
        rows = pool.map(task, tasks, chunksize=8)
    with open(f"{OUT}/replay_perseed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("rows", len(rows), file=sys.stderr)


if __name__ == "__main__":
    main()
