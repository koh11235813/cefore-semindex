"""Out-of-sample check of replay.py, written by the main session (not by the implementer).

Why: Q61 will swap the poll-interval rule and trust the simulated bytes and delay. The store already holds
six rules run on the SAME outage schedule and record schedule per (p, seed) (checked: the link events of
s6b10p043, s6short and s6b0p5 at p0.4 seed 7 are identical). So: take the exogenous inputs of a DONOR stage's
run, put the TARGET stage's rule into the simulator, and compare with what the target stage measured.
Nothing of the target run enters the simulation except its meta.json rule settings.

Usage (in this directory): python3 swap.py replay.py > swap.csv
"""
import csv
import importlib.util
import multiprocessing as mp
import random
import sys

spec = importlib.util.spec_from_file_location("replay", sys.argv[1])
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

STAGES = ["s6b0p5", "s6b1p359", "s6b3p695", "s6b10p043", "s6short", "s6wait"]
DONORS = ["s6b10p043", "s6b0p5"]
PS = [0.2, 0.4, 0.5, 0.6]
SEEDS = range(1, 21)
REPS = 5


def tup(s):
    return (s["n4"], s["c4"], s["lat4"], s["polls"], s["answered"], s["timed_out"])


def job(a):
    p, seed = a
    runs = {st: R.load_run(st, p, seed) for st in STAGES}
    out = {("meas", st): [tup(R.score(runs[st]["events"], runs[st]["t0"], p))] for st in STAGES}
    for d in DONORS:
        for t in STAGES:
            run = dict(runs[d])
            for k in ("policy", "choices", "choice_jitter"):
                run[k] = runs[t][k]
            rows = []
            for rep in range(REPS):
                rng = random.Random(R.rep_seed(f"swap:{d}:{t}", p, seed, rep))
                rows.append(tup(R.score(R._sim_run(run, "B", sim_rng=rng), run["t0"], p)))
            out[(d, t)] = rows
    return p, seed, out


if __name__ == "__main__":
    with mp.Pool(6) as pool:
        res = pool.map(job, [(p, s) for p in PS for s in SEEDS])
    w = csv.writer(sys.stdout)
    w.writerow("p target donor metric measured ci_lo ci_hi sim rel inside donor_measured".split())
    for p in PS:
        cell = [r for r in res if r[0] == p]
        for t in STAGES:
            mrows = [r[2][("meas", t)][0] for r in cell]
            m, ci = R.pooled(mrows, p), R.bootstrap_ci(mrows, p)
            for d in DONORS:
                dm = R.pooled([r[2][("meas", d)][0] for r in cell], p)
                # mean over replicates of the 20-seed cell value, as spec.md B1
                reps = [R.pooled([r[2][(d, t)][i] for r in cell], p) for i in range(REPS)]
                for k in R.METRICS:
                    if m[k] is None:
                        continue
                    s = sum(x[k] for x in reps) / REPS
                    lo, hi = ci[k]
                    w.writerow([p, t, d, k, round(m[k], 4), round(lo, 4), round(hi, 4), round(s, 4),
                                round((s - m[k]) / m[k], 5) if m[k] else "", int(lo <= s <= hi), round(dm[k], 4)])
