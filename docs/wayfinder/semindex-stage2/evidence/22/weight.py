#!/usr/bin/env python3
"""Acceptance 4: the one weight that puts the 3.695 s arm and the 10.043 s arm level.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B weight.py

Writes its whole printed output to weight.txt and the weight alone to weight.json.

WHAT IS COMPUTED
  score = delay + weight x polls, so the weight that makes two arms score equally is
      weight = (delay(10.043) - delay(3.695)) / (polls(3.695) - polls(10.043))
  in milliseconds of record delay per state poll.  `delay` is the headline delay of
  evidence/20 and 21 -- evidence/14's early-record scorer, pooled sum(lat4)/sum(n4) over
  the runs of one replicate stream -- and `polls` is that same scorer's windowed poll
  count per run.  Both are the median over the NREP streams, which is what evidence/21's
  own pass line reads.

THE CELL
  p 0.2, workload w0, the s6 outage shape (8 to 12 s), seeds 1 to 100, NREP replicate
  streams, built by evidence/21's own `run_dict`: a seed with a logged run of that shape
  keeps its realized outage instants, its logged creation stamps and its own clock, a
  seed without one takes the scheduled instants from this worktree's generator and
  borrows a clock.  The donor map is evidence/21's and is not re-derived here.
  p 0.4 is run for information and is not used for the weight.

HOW THE ARMS ARE RUN
  Through the decider port, as constant deciders over the three s11a arms, with the
  per-pair lag streams on -- that is, through the mechanism the labels will be made with,
  not through the old fixed-arm path.  The old path is run beside it, at p 0.2 only, so
  the reader can see what the port and the lag streams cost: `choices = [X]`, the
  re-decision off, no decider, and the single shared tick-lag stream evidence/18 draws
  with.

WHAT THIS CANNOT SHOW
  That the weight is right.  It is the exchange rate at which THIS cell's two outer arms
  are indifferent, measured in the replay; the testbed has never been asked the question
  and no run anywhere has been made with a decider.  The arms being level under it is an
  arithmetic identity, not evidence -- the weight was solved for it.  What does carry
  information is how far the per-stream weights spread, which is printed beside it, and
  whether the ordering at p 0.4 would want another value, which is printed and not acted
  on.
"""
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import port as PORT  # noqa: E402

R = PORT.R
B = PORT.B

# evidence/21's prediction script by path, the way it loads evidence/20's: it is named
# `predict.py` like evidence/20's, and its run dicts, donor map and replicate count are
# what "built exactly the way evidence/21 builds its runs" means.
_spec = importlib.util.spec_from_file_location(
    "predict21", os.path.join(os.path.normpath(os.path.join(HERE, "..", "21")), "predict.py"))
PR = importlib.util.module_from_spec(_spec)
sys.modules["predict21"] = PR
_spec.loader.exec_module(PR)

A = PR.A                                # evidence/14's early-record scorer
VALS = PR.P.vals                        # evidence/17's pooling, through evidence/20

assert PR.B is B, (PR.B.__file__, B.__file__)
assert PR.R is R, (PR.R.__file__, R.__file__)

PS = (0.2, 0.4)
MAIN_P = 0.2
SEEDS = PORT.check_seeds(PR.SEEDS, "weight.py")
NREP = PR.NREP
CHOICES = list(PR.CHOICES)
JF = PR.JF
STREAM_LABEL = PR.STREAM_LABEL
ARMS = (3.695, 10.043)                  # the two outer arms the weight levels
WORKERS = 5                             # the task's ceiling on a six-core machine

OUT_TXT = os.path.join(HERE, "weight.txt")
OUT_JSON = os.path.join(HERE, "weight.json")

RUNS: dict[tuple[float, int], dict] = {}        # filled before the pool is created

LINES = []


def out(text=""):
    """Print and keep, so weight.txt is exactly what the terminal showed."""
    print(text, flush=True)
    LINES.append(text)


# ---------------------------------------------------------------------------
# The runs
# ---------------------------------------------------------------------------

def port_run(run, arm, p, seed, rep):
    """One (arm, stream) through the port with the per-pair lag streams on.

    `sim_rng` is left None on purpose: the pair streams are meant to be the only source
    of a Mode B lag here, so a draw site that still reached the shared stream would raise
    instead of quietly mixing the two.  The lag seed depends on (p, seed, rep) and NOT on
    the arm, which is what makes the two arms see one set of lags per pair.
    """
    run = dict(run, policy=None, rule=None, choices=list(CHOICES), choice_jitter=JF,
               redecide=True, decider=PORT.const_decider(arm))
    return PORT.sim(run, None, R.rep_seed(f"pairlag:{STREAM_LABEL}", p, seed, rep))


def old_run(run, arm, p, seed, rep):
    """The same arm the way the replay ran a fixed arm before the port.

    `choices = [X]` with the re-decision off is the fixed-arm path of evidence/18's rows,
    and the tick lags come from the one shared stream evidence/18 draws with, so this is
    the comparison value for what the port plus the lag streams move.
    """
    run = dict(run, policy=None, rule=None, choices=[arm], choice_jitter=JF,
               redecide=False)
    run.pop("decider", None)
    return PORT.sim(run, B.rep_rng_of(STREAM_LABEL, p, seed, rep), None)


def job(key):
    """One (kind, p, seed): both arms, every stream, scored once each."""
    kind, p, seed = key
    run = RUNS[(p, seed)]
    maker = port_run if kind == "port" else old_run
    rows, fired = {}, 0
    for arm in ARMS:
        per_rep = []
        for rep in range(NREP):
            events, engine = maker(run, arm, p, seed, rep)
            per_rep.append(list(A.early_score(events, run["t0"])))
            fired += engine.provisional_fired
        rows[arm] = per_rep
    return key, rows, fired


# ---------------------------------------------------------------------------
# Reading the runs
# ---------------------------------------------------------------------------

def stream_vals(res, kind, p, arm):
    """Per stream, (delay ms, polls per run, censored share) pooled over the seeds."""
    return np.array([VALS(np.array([res[(kind, p, s)][arm][rep] for s in SEEDS], float))
                     for rep in range(NREP)])


def medians(res, kind, p, arm):
    v = stream_vals(res, kind, p, arm)
    return tuple(float(np.median(v[:, i])) for i in range(3)) + (v,)


def weight_of(short, long_):
    """(delay(long) - delay(short)) / (polls(short) - polls(long)), ms per poll."""
    return (long_[0] - short[0]) / (short[1] - long_[1])


def agreeing_digits(a, b):
    """How many leading significant digits of two numbers agree; None when equal."""
    if a == b:
        return None
    return int(math.floor(-math.log10(abs(a - b) / max(abs(a), abs(b)))))


def ticket_weight():
    """The ticket's 104 ms, recomputed from ../18/bound_cells.csv rather than copied."""
    ref, _cols = B.read_cells(os.path.join(PORT.EV18, "bound_cells.csv"))
    rows = {arm: ref[("W0", "s6b10p043", MAIN_P, "fix%g" % arm)] for arm in ARMS}
    short, long_ = rows[ARMS[0]], rows[ARMS[1]]
    return ((float(long_["delay"]) - float(short["delay"]))
            / (float(short["polls_sent"]) - float(long_["polls_sent"])),
            {arm: (float(r["delay"]), float(r["polls_sent"])) for arm, r in rows.items()})


def arm_line(label, m):
    return ("  %-34s delay %10.3f ms   polls %7.3f   censored %7.5f"
            % (label, m[0], m[1], m[2]))


def report(res, fired, wall):
    """Everything Acceptance 4 asks to be written down, and the comparisons beside it."""
    out("The weight that levels the two outer arms")
    out("=========================================")
    out(f"Cell: p {MAIN_P}, workload w0, the s6 outage shape, seeds {SEEDS[0]} to "
        f"{SEEDS[-1]}, {NREP} Mode B replicate streams, run dicts built by "
        "../21/predict.py's own run_dict.")
    out(f"Arms: {ARMS[0]} s and {ARMS[1]} s, as constant deciders over the choice set "
        f"{CHOICES} with the re-decision on, per-pair lag streams on.")
    out("Delay is evidence/14's early-record scorer pooled sum(lat4)/sum(n4) per stream, "
        "polls is that scorer's windowed poll count per run; both are the median over "
        "the streams.")
    out()

    main = {arm: medians(res, "port", MAIN_P, arm) for arm in ARMS}
    weight = weight_of(main[ARMS[0]], main[ARMS[1]])
    out("Main cell, through the port")
    for arm in ARMS:
        out(arm_line(f"fixed {arm} s", main[arm]))
    out("  weight = (%.6f - %.6f) / (%.6f - %.6f)"
        % (main[ARMS[1]][0], main[ARMS[0]][0], main[ARMS[0]][1], main[ARMS[1]][1]))
    out("  WEIGHT = %.17g ms per poll" % weight)
    out("         = %.6f ms per poll, to six decimals" % weight)
    out()

    scores = {arm: main[arm][0] + weight * main[arm][1] for arm in ARMS}
    digits = agreeing_digits(scores[ARMS[0]], scores[ARMS[1]])
    out("The two scores under that weight")
    for arm in ARMS:
        out("  score(%-6s) = %.17g" % (f"{arm} s", scores[arm]))
    out("  difference = %.17g (relative %.3e)"
        % (scores[ARMS[1]] - scores[ARMS[0]],
           abs(scores[ARMS[1]] - scores[ARMS[0]]) / max(scores.values())))
    out("  they agree to %s"
        % ("every digit of a double: the difference is exactly 0" if digits is None
           else f"{digits} significant digits"))
    out("  This agreement is an arithmetic identity: the weight was solved for it, so it "
        "is a statement about the arithmetic and not about the arms. What carries "
        "information is the spread below.")
    out()

    per_stream = np.array([
        weight_of(VALS(np.array([res[("port", MAIN_P, s)][ARMS[0]][rep] for s in SEEDS], float)),
                  VALS(np.array([res[("port", MAIN_P, s)][ARMS[1]][rep] for s in SEEDS], float)))
        for rep in range(NREP)])
    out("The same weight computed inside each stream (the spread the median hides)")
    out("  median %.3f   min %.3f   max %.3f   over %d streams"
        % (float(np.median(per_stream)), float(per_stream.min()),
           float(per_stream.max()), NREP))
    out("  The weight above is the one taken from the median delay and the median poll "
        "count, which is the definition; this line says how far one experiment's own "
        "answer can sit from it.")
    out()

    p04 = {arm: medians(res, "port", 0.4, arm) for arm in ARMS}
    w04 = weight_of(p04[ARMS[0]], p04[ARMS[1]])
    out("p 0.4, for information only; not used for the weight")
    for arm in ARMS:
        out(arm_line(f"fixed {arm} s", p04[arm]))
    out("  weight at p 0.4 = %.6f ms per poll, i.e. %.2f times the main cell's"
        % (w04, w04 / weight))
    out("  Whether a weight fixed on the main cell flips a label at p 0.4 is not "
        "answered here: that needs the labels themselves.")
    out()

    old = {arm: medians(res, "old", MAIN_P, arm) for arm in ARMS}
    w_old = weight_of(old[ARMS[0]], old[ARMS[1]])
    out("Main cell, the old fixed-arm path (no decider, no per-pair lag streams)")
    for arm in ARMS:
        out(arm_line(f"fixed {arm} s", old[arm]))
    out("  weight = %.6f ms per poll, %+.3f ms (%+.2f %%) against the port's"
        % (w_old, w_old - weight, 100.0 * (w_old - weight) / weight))
    out()

    tw, tr = ticket_weight()
    out("The ticket's estimate, recomputed from ../18/bound_cells.csv")
    for arm in ARMS:
        out("  %-34s delay %10.3f ms   polls %7.3f" % (f"fix{arm:g} row", *tr[arm]))
    out("  weight = %.6f ms per poll (the ticket writes it as 104 ms)" % tw)
    out("  It is a different set of runs: seeds 1 to 20 and 5 replicates, not "
        f"{SEEDS[0]} to {SEEDS[-1]} and {NREP}, and the arms there are the old "
        "fixed-arm path. The three values above are therefore three different "
        "questions, not three tries at one.")
    out()

    out("Provisional timers that fired before their own poll had resolved: %d, over the "
        "%d simulations of this script, of which the %d port simulations are the ones "
        "that can show any (the old fixed-arm path arms no provisional timer at all: "
        "its send-time timer is final)."
        % (fired, 3 * len(SEEDS) * len(ARMS) * NREP,
           2 * len(SEEDS) * len(ARMS) * NREP))
    out("  The resolution-time decision is the one that takes effect, so a fired "
        "provisional timer is a poll whose send-time interval decided the next send on "
        "its own, and a label on that decision would be a label on the wrong instant.")
    out()
    out("wall %.1f s on %d workers" % (wall, WORKERS))
    return weight


def main():
    t = time.time()
    for p in PS:
        for seed in SEEDS:
            RUNS[(p, seed)] = PR.run_dict(p, seed)
    keys = ([("port", p, s) for p in PS for s in SEEDS]
            + [("old", MAIN_P, s) for s in SEEDS])
    with mp.Pool(WORKERS) as pool:
        got = pool.map(job, keys, chunksize=1)
    res = {key: rows for key, rows, _f in got}
    fired = sum(f for _k, _r, f in got)
    weight = report(res, fired, time.time() - t)
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"weight_ms_per_poll": weight,
                   "cell": {"p": MAIN_P, "workload": "w0", "outage_shape": "s6 8-12 s",
                            "seeds": [SEEDS[0], SEEDS[-1]], "nrep": NREP},
                   "arms_s": list(ARMS),
                   "path": "the decider port, constant deciders, per-pair lag streams on"},
                  fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {OUT_TXT} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
