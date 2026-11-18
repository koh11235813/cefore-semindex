#!/usr/bin/env python3
"""The prediction for the s6 p 0.4 testbed run, archived before that run is done.

    /home/lab_shared/cefore-semindex/.venv/bin/python3 -B predict.py > predict.txt

WHY THIS EXISTS
  The testbed can afford ONE run per seed, not the five replicates the simulated
  evidence used.  This step asks whether that thinner condition still separates T37r
  from T37 on the logged workload, and it fixes the numbers the testbed will be held
  to, so that the comparison afterwards cannot be a number chosen once the measurement
  was already in view.  The acceptance band is the predicted median plus or minus 6 %,
  printed here for delay and for polls per run.

WHAT IT COMPUTES
  For each replicate stream r (0..39) the 20 seeds' single simulations are treated as
  one "experiment": pooled delay, polls per run and censored share for T37, T37r and
  the three fixed arms, the paired T37r-minus-T37 gain difference against an
  interpolated fixed-interval frontier, and a seed bootstrap of that difference.  The
  engine, the rule table and the run store all come from the earlier evidence steps,
  imported unchanged; nothing here writes to the store.

WHAT IT CANNOT SHOW
  Every value is a simulated value.  The replay was checked against measurements for
  this workload (../13/validate.md), but a simulated band is still a prediction about
  the simulator's own agreement with the testbed, not a measurement.

The seed strings, the RNG constructors, the bootstrap size, the replicate count and
the worker count are held fixed on purpose: changing any of them would move the
predicted numbers, and a prediction that moves after it was archived is worthless.
"""
import math
import multiprocessing as mp
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Insert 13, 14, 15 in that order, so the front of sys.path ends up 15, 14, 13.  The
# order matters: evidence/13 and evidence/14 both hold an `after.py`, and evidence/15
# re-inserts its own two entries when it is imported.  `onoff` must resolve to
# evidence/15 and `replay` to evidence/13, which the asserts below check.
for _d in ("13", "14", "15"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", _d)))
import onoff as O  # noqa: E402  evidence/15: the engine, the rule table, sim_run
import replay as R  # noqa: E402  evidence/13: the run store, the scorer, the seeds

# realpath, not the raw __file__: evidence/15 inserts its own ".." entries, so replay
# arrives spelled ".../14/../13/replay.py" even though it is evidence/13's file.
assert os.path.realpath(O.__file__).startswith(os.path.realpath(os.path.join(HERE, "..", "15"))), O.__file__
assert os.path.realpath(R.__file__).startswith(os.path.realpath(os.path.join(HERE, "..", "13"))), R.__file__

DONOR = "s6b10p043"                 # the outage cell the testbed run will use
P = 0.4                             # its loss probability
SEEDS = range(1, 21)                # 20 seeds, as every earlier step
NREP = 40                           # replicate streams, i.e. 40 simulated experiments
WORKERS = 10                        # the task's ceiling
BOOT_N = 1000                       # seed bootstrap resamples per experiment
BOOT_SEED = 7                       # fixed, so the printed bounds are reproducible
BAND = 0.06                         # the acceptance line: within plus/minus 6 %

RULES = ["fix3.695", "fix6.0", "fix10.043", "T37", "T37r"]
ARMS3 = ["fix3.695", "fix6.0", "fix10.043"]
ARMS2 = ["fix3.695", "fix10.043"]
SPEC = {n: (k, s) for n, k, s in O.RULE_SPECS}


def build_run(donor_run, name, seed):
    """The simulator input for one rule, built from the donor run of this seed.

    A fixed or policy arm has no rule of its own: its poll schedule lives in another
    logged run, so that run's policy, choices and jitter are copied in and the rule is
    cleared.  A tested rule is the other way round: the logged schedule is cleared so
    that the rule alone decides when a poll goes out.
    """
    kind, spec = SPEC[name]
    run = dict(donor_run)
    if kind in ("fixed", "policy"):
        tgt = R.load_run(DONOR[:2] + spec, P, seed)
        for key in ("policy", "choices", "choice_jitter"):
            run[key] = tgt[key]
        run["rule"] = None
    else:
        run["policy"] = None
        run["choices"] = None
        run["rule"] = spec if kind == "rule" else dict(i_min=spec, m=1.0, i_max=spec, jf=O.T.JF)
    return run


def job(seed):
    """Simulate every rule of this seed NREP times, one scored row per replicate.

    The per-replicate RNG seed string is "power:<donor>", kept from the step that first
    produced these numbers: the seed string is part of what makes the prediction a
    fixed quantity, so it is not renamed to match this file.
    """
    donor_run = R.load_run(DONOR, P, seed)
    out = {}
    for name in RULES:
        run = build_run(donor_run, name, seed)
        rows = []
        for rep in range(NREP):
            sim_rng = random.Random(R.rep_seed(f"power:{DONOR}", P, seed, rep))
            ev = O.sim_run(run, sim_rng)
            s = R.score(ev, donor_run["t0"], P)
            rows.append((s["n4"], s["c4"], s["lat4"], s["polls"]))
        out[name] = rows
    return seed, out


def vals(m):
    """Pool one experiment's per-seed rows into (delay ms, polls per run, censored share).

    Delay is pooled over records, not averaged over seeds, so that a seed with more
    records weighs more; that is the same pooling the earlier steps' scorer uses.
    """
    n4, c4, lat, polls = m.sum(0)
    return lat / n4, polls / len(m), c4 / (n4 + c4)


def interp(pts, x):
    """Log-log interpolation of the fixed-interval frontier at polls rate `x`.

    Delay against polls is close to a straight line in logs over this range, and a
    linear reading would understate the frontier and so overstate every gain.  Returns
    None when `x` falls outside the measured arms, because extrapolating a frontier
    that has no arm beyond the point would invent the comparison.
    """
    pts = sorted(pts)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + w * (math.log(y1) - math.log(y0)))
    return None


def gain(m, name, arms):
    """How much delay the rule saves against the frontier at its own polls rate.

    Comparing delay alone would reward a rule for simply polling more often, so the
    rule is charged for its polls: the frontier is read at the rule's own polls rate
    and the gain is the fraction of that delay removed.
    """
    delay, polls, _ = vals(m[name])
    frontier = interp([(vals(m[a])[1], vals(m[a])[0]) for a in arms], polls)
    if frontier is None:
        return None
    return 1 - delay / frontier


def experiment(res, rep):
    """One experiment's per-rule array: 20 seeds' rows for replicate stream `rep`."""
    return {k: np.array([res[s][k][rep] for s in SEEDS], float) for k in RULES}


def print_separation(res, rng):
    """How often a single-run-per-seed experiment puts the T37r-minus-T37 gap above zero.

    The bootstrap resamples seeds, not records, because the testbed's unit of
    repetition is the seed: a bound from resampling records would describe a precision
    the testbed does not have.  Both frontiers are reported because the 6.0 s arm is
    interpolated rather than measured, and a conclusion that needs it is weaker.
    """
    for arms, label in ((ARMS3, "3 arms incl. 6.0 s"), (ARMS2, "2 measured arms only")):
        above_zero = 0
        diffs = []
        for rep in range(NREP):
            m = experiment(res, rep)
            diffs.append(gain(m, "T37r", arms) - gain(m, "T37", arms))
            idx = rng.integers(0, len(SEEDS), size=(BOOT_N, len(SEEDS)))
            boot = []
            for row in idx:
                sub = {k: m[k][row] for k in RULES}
                a = gain(sub, "T37", arms)
                b = gain(sub, "T37r", arms)
                if a is not None and b is not None:
                    boot.append(b - a)
            lo = np.percentile(boot, 2.5)
            above_zero += lo > 0
        print(label, ": diff mean %.1f sd %.1f min %.1f max %.1f pp; experiments (of %d) with lower bound > 0: %d"
              % (100 * np.mean(diffs), 100 * np.std(diffs), 100 * min(diffs), 100 * max(diffs), NREP, above_zero))


def print_paired(res):
    """The raw paired delay/polls/censoring move from T37 to T37r, per experiment.

    Ratios rather than differences, because the testbed will not reproduce the absolute
    delay exactly and a ratio is the part of the prediction that should survive.
    """
    dd, pp, c0, c1 = [], [], [], []
    for rep in range(NREP):
        m = experiment(res, rep)
        a = vals(m["T37"])
        b = vals(m["T37r"])
        dd.append(b[0] / a[0] - 1)
        pp.append(b[1] / a[1] - 1)
        c0.append(a[2])
        c1.append(b[2])
    print("T37r vs T37: delay ratio-1 mean %.3f [min %.3f max %.3f]; polls ratio-1 mean %.3f; cens T37 %.3f T37r %.3f"
          % (np.mean(dd), min(dd), max(dd), np.mean(pp), np.mean(c0), np.mean(c1)))


def print_pooled(res):
    """Each rule's value when all 40 replicates of a seed are averaged first.

    This is the all-replicates view, kept so that the thinner experiment-level numbers
    below can be read against it; it is NOT the quantity the testbed can match, because
    the testbed has one run per seed.
    """
    m = {k: np.array([np.mean(res[s][k], axis=0) for s in SEEDS]) for k in RULES}
    for k in RULES:
        print(k, "delay %.0f polls %.1f cens %.3f" % vals(m[k]))


def per_experiment(res, rule):
    """The 40 experiments' (delay ms, polls, censored share) rows for one rule."""
    return np.array([vals(np.array([res[s][rule][rep] for s in SEEDS], float)) for rep in range(NREP)])


def print_bands(res):
    """The spread across the 40 experiments, and the acceptance band for the testbed.

    The median is the predicted value rather than the mean, because it is the middle of
    what a single-run-per-seed experiment produces and is unmoved by one stream that
    happened to draw a bad outage.  The band is the median scaled by 1 plus/minus 6 %,
    which is the tolerance the testbed comparison is accepted within.  Censoring gets
    no band: its predicted value is at or near zero for T37r, and a 6 % band around
    zero would be a tolerance that nothing can fail.
    """
    print("experiment-level bands (40 single-run-per-seed experiments): rule, delay ms p2.5/p50/p97.5, polls, cens")
    medians = {}
    for k in ("T37", "T37r"):
        v = per_experiment(res, k)
        qs = [np.percentile(v[:, c], [2.5, 50, 97.5]) for c in range(3)]
        print(k, "delay", fmt_q(qs[0], 0), "| polls", fmt_q(qs[1], 1), "| cens", fmt_q(qs[2], 2),
              "| delay min %.0f max %.0f" % (v[:, 0].min(), v[:, 0].max()))
        medians[k] = (qs[0][1], qs[1][1], qs[2][1], v)
    print("acceptance bands (median of the 40 experiments, plus/minus %d %%):" % round(100 * BAND))
    for k in ("T37", "T37r"):
        d_med, p_med, c_med, v = medians[k]
        print("%s delay %.3f s band %.3f .. %.3f s | polls %.2f band %.2f .. %.2f | cens %.3f"
              % (k, d_med / 1000.0, d_med / 1000.0 * (1 - BAND), d_med / 1000.0 * (1 + BAND),
                 p_med, p_med * (1 - BAND), p_med * (1 + BAND), c_med))
        print("%s per-experiment min/max: delay %.3f .. %.3f s | polls %.2f .. %.2f | cens %.3f .. %.3f"
              % (k, v[:, 0].min() / 1000.0, v[:, 0].max() / 1000.0,
                 v[:, 1].min(), v[:, 1].max(), v[:, 2].min(), v[:, 2].max()))


def fmt_q(q, col):
    """Format one percentile triple with the digits that column is read at."""
    if col == 2:
        return " / ".join("%.3f" % x for x in q)
    if col == 0:
        return " / ".join("%.0f" % x for x in q)
    return " / ".join("%.1f" % x for x in q)


def main():
    """Run every seed in a worker pool, then print the prediction to stdout."""
    with mp.Pool(WORKERS) as pool:
        res = dict(pool.map(job, SEEDS))
    # One generator for the whole separation section, drawn in the order the arm sets
    # are printed: a second generator, or a different order, would change the bounds.
    rng = np.random.default_rng(BOOT_SEED)
    print_separation(res, rng)
    print_paired(res)
    print_pooled(res)
    print_bands(res)


if __name__ == "__main__":
    main()
