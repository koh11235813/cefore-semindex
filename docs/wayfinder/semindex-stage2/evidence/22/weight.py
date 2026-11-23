#!/usr/bin/env python3
"""Acceptance 4: the one weight that puts the 3.695 s arm and the 10.043 s arm level.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B weight.py

Writes its whole printed output to weight.txt and the weight alone to weight.json.

WHAT IS COMPUTED
  score = delay + weight x polls, so the weight that makes two arms score equally is
      weight = (delay(10.043) - delay(3.695)) / (polls(3.695) - polls(10.043))
  in milliseconds of record delay per state poll.

  2026-11-23, THE CORRECTION THIS FILE CARRIES.  The weight has to level the arms on the
  delay a LABEL is decided by, and until today it did not.  The old value was solved on
  `../14/after.py`'s `early_score`, whose delay is `sum(lat4) / sum(n4)` over the DELIVERED
  records alone, pooled over the seeds of one replicate stream and then taken as the median
  over the streams; `label.py`'s score enters a censored record at its lower bound and
  averages per run.  So the weight was the indifference point of a quantity no label was
  ever measured in.  `score.py` now holds the one definition and this file re-measures the
  weight under it.  The four steps below are printed one at a time, so that a reader sees
  what each change moved rather than one number replacing another:
    1. as committed: censored records left out, pooled per stream, median over streams,
       the node's own jitter stream.  It must reproduce 100.62168514446252 exactly;
    2. the same with the censored records at their lower bound;
    3. step 2 with the per-pair JITTER streams on, which is the labelling's own engine;
    4. the per-run mean of the label's own score under step 3's engine.  THE FIXED VALUE.
  Step 4 and not step 3 because a label is one run's score and the labels are compared run
  by run, so the weight that levels the arms has to be solved on the same per-run mean.

  2026-11-23, SECOND ADDITION.  Step 3 was the largest of the four moves and the file said
  only that it happened.  `print_part_a` measures why: the jackknife standard error of the
  step-2, step-3 and step-4 weights over the seeds, and the per-seed PAIRED move of the
  weight's numerator and denominator between steps 2 and 3, against a reading rule written
  before the numbers existed.  It reports and decides nothing: the fixed value and
  `weight.json` are untouched by it.

THE CELL
  p 0.2, workload w0, the s6 outage shape (8 to 12 s), seeds 1 to 100, NREP replicate
  streams, built by evidence/21's own `run_dict`: a seed with a logged run of that shape
  keeps its realized outage instants, its logged creation stamps and its own clock, a
  seed without one takes the scheduled instants from this worktree's generator and
  borrows a clock.  The donor map is evidence/21's and is not re-derived here.  Because
  that builder reads the generator for the seeds without a run, `PR.gate_generator()` is
  run first and this script stops when it disagrees with the stored descriptions.
  p 0.4 is run for information and is not used for the weight.

HOW THE ARMS ARE RUN
  Through the decider port, as constant deciders over the three s11a arms, with the
  per-pair lag streams on -- that is, through the mechanism the labels are made with, not
  through the old fixed-arm path.  Both per-pair seeds are the string
  `rep_seed("pairlag:<STREAM_LABEL>", p, seed, rep)` with STREAM_LABEL imported from
  evidence/21, which is named here because reproducing any number below depends on it:
  another stream family shifts the 40-stream median by a few tenths of a millisecond.
  Neither seed carries the arm, which is what makes the two arms see one set of lags per
  pair.  The old path is run beside it, at p 0.2 only, so the reader can see what the port
  and the lag streams cost: `choices = [X]`, the re-decision off, no decider, and the
  single shared tick-lag stream evidence/18 draws with.

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
import score as SCORE  # noqa: E402

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
# One module object for the scorer, or step 1's reproduction check would compare two of
# them rather than two definitions.
assert SCORE.A is A, (SCORE.A.__file__, A.__file__)

PS = (0.2, 0.4)
MAIN_P = 0.2
SEEDS = PORT.check_seeds(PR.SEEDS, "weight.py")
NREP = PR.NREP
CHOICES = list(PR.CHOICES)
JF = PR.JF
STREAM_LABEL = PR.STREAM_LABEL
LAG_PREFIX = f"pairlag:{STREAM_LABEL}"  # the seed-string prefix of every per-pair stream
ARMS = (3.695, 10.043)                  # the two outer arms the weight levels
WORKERS = 5                             # the task's ceiling on a six-core machine

# Five sweeps of (both arms x every stream x every seed): the port with the node's own
# jitter at both cells (steps 1 and 2), the port with the per-pair jitter at both cells
# (steps 3 and 4), and the old fixed-arm path at the main cell.  2026-11-23: counted from
# the sweeps rather than written as a literal -- the first half's output said 24 000 and
# 16 000, which was three sweeps, and adding the jitter-on pair would have left the two
# printed totals quietly wrong.
SWEEPS, PORT_SWEEPS = 5, 4
SIMS = SWEEPS * len(SEEDS) * len(ARMS) * NREP
PORT_SIMS = PORT_SWEEPS * len(SEEDS) * len(ARMS) * NREP

# The value step 1 has to reproduce: the weight as committed, before the definition was
# corrected.  Written here as a literal so that the reproduction is a check and not a
# comparison against whatever weight.json happens to hold after this script rewrote it.
COMMITTED = 100.62168514446252

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

def port_run(run, arm, p, seed, rep, jitter):
    """One (arm, stream) through the port with the per-pair lag streams on.

    `sim_rng` is left None on purpose: the pair streams are meant to be the only source
    of a Mode B lag here, so a draw site that still reached the shared stream would raise
    instead of quietly mixing the two.  The seed depends on (p, seed, rep) and NOT on the
    arm, which is what makes the two arms see one set of lags per pair.

    `jitter` is what separates steps 1 and 2 from steps 3 and 4: with it False the jitter
    of a poll comes from the node's own stream, which is what the committed weight was
    measured with, and with it True from the ordered pair's, which is what `label.py`
    runs.  The same seed string serves both streams; `port.Engine._seed` puts a different
    kind prefix on each, so they are two independent streams, and that is the pair
    `label.py` passes too.
    """
    run = dict(run, policy=None, rule=None, choices=list(CHOICES), choice_jitter=JF,
               redecide=True, decider=PORT.const_decider(arm))
    base = R.rep_seed(LAG_PREFIX, p, seed, rep)
    return PORT.sim(run, None, base, base if jitter else None)


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
    """One (kind, p, seed): both arms, every stream, scored once each.

    Every simulation is scored BOTH ways -- `score.terms` and `../14/after.py`'s own
    `early_score` -- and the four numbers they have in common are compared on the spot.
    That is what makes step 1 the committed quantity and not a rewrite of it that happens
    to agree on the three runs `labelcheck.py` samples; the disagreement count travels
    back and a non-zero one stops the script.
    """
    kind, p, seed = key
    run = RUNS[(p, seed)]
    rows, fired, calls, off = {}, 0, 0, 0
    for arm in ARMS:
        per_rep = []
        for rep in range(NREP):
            if kind == "old":
                events, engine = old_run(run, arm, p, seed, rep)
            else:
                events, engine = port_run(run, arm, p, seed, rep, kind == "portj")
            terms = SCORE.terms(events, run["t0"])
            if SCORE.early_four(terms) != tuple(A.early_score(events, run["t0"])):
                off += 1
            per_rep.append(terms)
            fired += engine.provisional_fired
            calls += sum(engine.decider_ord.values())
        rows[arm] = per_rep
    return key, rows, fired, calls, off


# ---------------------------------------------------------------------------
# Reading the runs: one function per way of pooling, because the steps differ in it
# ---------------------------------------------------------------------------

def per_stream_pooled(res, kind, p, arm, censored, seeds=None):
    """Per stream, (delay ms, polls per run, censored share), POOLED over the seeds.

    evidence/17's own pooling, through `vals`: the delay is `sum(lat) / sum(n4)` over the
    100 runs of one stream.  With `censored` True the censored lower-bound sum is added to
    the numerator and the censored records to the denominator, which `vals` cannot do, so
    that one line is computed here instead of handed to it.

    `seeds` defaults to all of them, which is what the four steps use.  It is a parameter
    only so that the jackknife below can leave one seed out without a second copy of this
    pooling; passing the default list must reproduce the step's own value exactly.
    """
    seeds = SEEDS if seeds is None else seeds
    arrays = [np.array([res[(kind, p, s)][arm][rep] for s in seeds], float)
              for rep in range(NREP)]
    if not censored:
        # Columns 0, 1, 2 and 4 are exactly what `early_score` returned, and advanced
        # indexing hands back a fresh C-contiguous array, so `vals` sums the same bytes in
        # the same order it summed before the censored column was added beside them.
        return np.array([VALS(m[:, [0, 1, 2, 4]]) for m in arrays])
    out_rows = []
    for m in arrays:
        n4, c4, lat, cens, polls = m.sum(0)
        out_rows.append(((lat + cens) / (n4 + c4), polls / len(m), c4 / (n4 + c4)))
    return np.array(out_rows)


def records_per_run(res, kind, ps):
    """The distinct values of `delivered + censored` over every run of the given cells.

    Printed beside step 4 because it says how much of the step 3 -> step 4 move is real.
    When this set holds ONE number, every run's denominator is the same, so step 3's
    pooled `(sum lat + sum censored) / (sum n4 + sum c4)` and the mean of the per-run
    delays coincide INSIDE a stream, and all that separates the two steps is taking the
    median over the 40 streams instead of the mean over the 4000 runs.  A second value in
    this set would mean the pooling itself weights the runs differently.
    """
    return sorted({t[0] + t[1] for p in ps for arm in ARMS for s in SEEDS
                   for t in res[(kind, p, s)][arm]})


def per_run_mean(res, kind, p, arm, seeds=None):
    """(mean D over the runs, mean polls over the runs, censored share) over ALL runs.

    A run is one (seed, stream) and there are len(SEEDS) x NREP of them.  The mean and not
    the median, and over runs and not over streams, because a label is one run's score:
    the weight that levels the arms for a label has to be solved on the quantity a label
    averages.

    `seeds` is the jackknife's leave-one-out list; see `per_stream_pooled`.
    """
    seeds = SEEDS if seeds is None else seeds
    terms = [res[(kind, p, s)][arm][rep] for s in seeds for rep in range(NREP)]
    delays = [SCORE.delay(t) for t in terms]
    polls = [float(t[4]) for t in terms]
    n4 = sum(t[0] for t in terms)
    c4 = sum(t[1] for t in terms)
    return (float(np.mean(delays)), float(np.mean(polls)), c4 / (n4 + c4))


def weight_of(short, long_):
    """(delay(long) - delay(short)) / (polls(short) - polls(long)), ms per poll."""
    return (long_[0] - short[0]) / (short[1] - long_[1])


def medians(rows):
    """(median delay, median polls, median censored share) over the streams."""
    return tuple(float(np.median(rows[:, i])) for i in range(3))


# ---------------------------------------------------------------------------
# How much of the step 2 -> step 3 move is seed-to-seed noise
# ---------------------------------------------------------------------------
# 2026-11-23.  Step 3 moved the weight by +3.0 ms, the largest of the four steps, and
# nothing said why.  The hypothesis this block measures, written down before any of its
# numbers existed: the node's own jitter stream is seeded
# `state_poll_interval_jitter:{seed}:{name}` (`../13/replay.py:329`) and carries NO
# replicate, so in steps 1 and 2 all NREP streams of a seed draw their jitter from ONE
# sequence per node and the jitter is averaged over len(SEEDS) independent draws; the
# per-pair stream of steps 3 and 4 is seeded per (seed, replicate) and averages over
# len(SEEDS) x NREP.  If that is the whole story, step 2's weight carries a jitter
# component it never averaged away and the +3.0 ms is sampling noise of the OLD
# measurement rather than a property of the keying.  A jitter draw happens in both
# configurations -- `port.Engine.decide_interval` rebinds `Node.jitter_rng` only when
# `pair_jitter_seed` is set, and leaves the node's own stream in place otherwise -- so
# this is a change of keying and not jitter being switched off.
#
# THE READING RULE, fixed here before the measurement and not moved afterwards.  Two
# sub-claims, each with its own threshold:
#   claim 1, the variance signature: the jackknife standard error of the step-2 weight is
#     at least SE_RATIO times the step-3 one.  Averaging the same jitter over 100 draws
#     instead of 4000 is what would make it so;
#   claim 2, the move is not distinguishable from zero: the per-seed PAIRED move of the
#     weight between the two steps has |mean| <= NOISE_SIGMA standard errors.
# Both hold -> SUPPORTED.  Claim 2 fails at MOVE_SIGMA standard errors or more -> NOT
# SUPPORTED, and the text then has to say what is left unexplained.  Anything else ->
# UNDECIDED.
SE_RATIO = 2.0
NOISE_SIGMA = 2.0
MOVE_SIGMA = 3.0


def jackknife_se(values):
    """The delete-one jackknife standard error from the n leave-one-out values.

    se = sqrt((n - 1) / n * sum (v_i - mean(v))^2).  The (n-1)/n factor and not 1/(n(n-1))
    is what makes this the jackknife rather than the standard error of the leave-one-out
    values themselves: each leave-one-out value differs from the full-sample one by about
    1/n of a seed's influence, so the spread has to be inflated back by (n-1).
    """
    v = np.asarray(values, float)
    n = len(v)
    return float(np.sqrt((n - 1) / n * float(np.sum((v - v.mean()) ** 2))))


def leave_one_out_weights(res, p, step):
    """The step's weight recomputed once per seed, with that seed left out.

    Steps 2 and 3 are re-pooled and re-median-taken over the remaining seeds, step 4
    re-averaged over the remaining runs, each through the same function the step itself
    used.  A caveat a reader has to carry for steps 2 and 3: the statistic is a MEDIAN over
    the NREP streams, so dropping one seed can move it by swapping which stream sits in the
    middle.  The jackknife is an approximation for a median and the leave-one-out values
    can sit at a few levels rather than spread smoothly; the paired comparison below does
    not have that problem and is the one the reading rule leans on.
    """
    vals = []
    for s in SEEDS:
        keep = [x for x in SEEDS if x != s]
        if step == 4:
            rows = {arm: per_run_mean(res, "portj", p, arm, keep) for arm in ARMS}
        else:
            kind = "port" if step == 2 else "portj"
            rows = {arm: medians(per_stream_pooled(res, kind, p, arm, True, keep))
                    for arm in ARMS}
        vals.append(weight_of(rows[ARMS[0]], rows[ARMS[1]]))
    return vals


def per_seed_pooled(res, kind, p, arm, seed):
    """(delay ms, polls per run) for ONE seed, pooled over the NREP streams.

    Steps 2 and 3 pool over the seeds inside a stream; this pools over the streams inside a
    seed, with the same `(sum lat + sum censored) / (sum n4 + sum c4)` delay.  It is the
    same quantity regrouped, so the step-2 and step-3 readings of one seed are paired: the
    lag stream of a (seed, replicate) is seeded identically in both, and the jitter keying
    is the only input that differs between them.
    """
    m = np.array([res[(kind, p, seed)][arm][rep] for rep in range(NREP)], float)
    n4, c4, lat, cens, polls = m.sum(0)
    return (lat + cens) / (n4 + c4), polls / NREP


def mean_se(values):
    """(mean, standard error of the mean) of a per-seed sample."""
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def paired_step23(res, p):
    """Per seed, what step 3 changed in the weight's numerator, denominator and ratio.

    Returns a dict of (mean, standard error) pairs over the seeds.  The numerator is the
    long arm's delay minus the short arm's, the denominator the short arm's polls minus the
    long arm's, and the ratio the weight those two make for that seed alone; the per-arm
    delay and poll moves are kept apart as well, because if the ratio does move the text has
    to be able to say which of the four terms moved it.
    """
    cols = {k: [] for k in ("num", "den", "w",
                            "delay_short", "delay_long", "polls_short", "polls_long")}
    for s in SEEDS:
        two = {kind: {arm: per_seed_pooled(res, kind, p, arm, s) for arm in ARMS}
               for kind in ("port", "portj")}
        d = {}
        for kind in ("port", "portj"):
            short, long_ = two[kind][ARMS[0]], two[kind][ARMS[1]]
            d[kind] = {"num": long_[0] - short[0], "den": short[1] - long_[1],
                       "w": weight_of(short, long_),
                       "delay_short": short[0], "delay_long": long_[0],
                       "polls_short": short[1], "polls_long": long_[1]}
        for k in cols:
            cols[k].append(d["portj"][k] - d["port"][k])
    return {k: mean_se(v) for k, v in cols.items()}


def print_part_a(res, p, weights):
    """The step 2 -> step 3 move set against the noise of the two measurements."""
    loo = {step: leave_one_out_weights(res, p, step) for step in (2, 3, 4)}
    se = {step: jackknife_se(loo[step]) for step in (2, 3, 4)}
    move = weights[3] - weights[2]
    paired = paired_step23(res, p)

    out("Why step 3 moved the weight, measured rather than asserted")
    out("  The question: the jitter of steps 1 and 2 comes from the node's own stream, "
        f"seeded with (seed, node name) and NOT with the replicate, so all {NREP} streams "
        "of a seed share one jitter sequence and the jitter is averaged over "
        f"{len(SEEDS)} independent draws; steps 3 and 4 key it per (seed, replicate) and "
        f"average over {len(SEEDS) * NREP}. If that is the whole story, the "
        # 2026-11-23: the sign comes out of the number and is not written into the
        # sentence.  On another cell or another seed set this move can be negative, and
        # the literal "+" that stood here would then have printed "+-"; the move is
        # positive on this cell, so no output of this script ever showed it.
        f"{move:+.6f} ms is sampling noise of the step-2 measurement.")
    out("  Jackknife standard error of the weight over the %d seeds, and the move set "
        "against it:" % len(SEEDS))
    for step in (2, 3):
        out("    step %d   weight %12.6f   jackknife se %8.6f ms"
            % (step, weights[step], se[step]))
    out("    the step 2 -> step 3 move is %+.6f ms = %.2f x step 2's own standard error "
        "and %.2f x step 3's" % (move, abs(move) / se[2], abs(move) / se[3]))
    out("    step 2's standard error is %.2f times step 3's; claim 1 asks for at least "
        "%.1f" % (se[2] / se[3], SE_RATIO))
    out("  The same seeds read twice, paired: per seed, both arms pooled over the "
        f"{NREP} streams, step 3's value minus step 2's. The lag stream of a "
        "(seed, replicate) is seeded identically in the two, so the jitter keying is the "
        "only input that differs.")
    for key, label, unit in (("delay_short", f"delay of the {ARMS[0]} s arm", "ms"),
                             ("delay_long", f"delay of the {ARMS[1]} s arm", "ms"),
                             ("polls_short", f"polls of the {ARMS[0]} s arm", "polls"),
                             ("polls_long", f"polls of the {ARMS[1]} s arm", "polls"),
                             ("num", "the weight's numerator", "ms"),
                             ("den", "the weight's denominator", "polls"),
                             ("w", "the weight itself", "ms per poll")):
        mean, sem = paired[key]
        out("    %-28s moved %+12.6f %-11s se %10.6f   %5.2f se"
            % (label, mean, unit, sem, abs(mean) / sem if sem else float("inf")))
    wm, wse = paired["w"]
    claim1 = se[2] >= SE_RATIO * se[3]
    claim2 = abs(wm) <= NOISE_SIGMA * wse
    broken = abs(wm) >= MOVE_SIGMA * wse
    out("  Against the rule fixed before the measurement (claim 1: step 2's standard "
        f"error is at least {SE_RATIO:.1f} times step 3's; claim 2: the paired move of "
        f"the weight is within {NOISE_SIGMA:.1f} standard errors of zero)")
    out("    claim 1 %s, claim 2 %s"
        % ("HOLDS" if claim1 else "DOES NOT HOLD", "HOLDS" if claim2 else "DOES NOT HOLD"))
    if claim1 and claim2:
        out("    THE HYPOTHESIS IS SUPPORTED: the per-pair keying averages the same "
            "jitter over more draws, and the move the four steps print is not "
            "distinguishable from zero once the seeds are paired. Step 1's and step 2's "
            "values carry a jitter component they never averaged away.")
    elif broken:
        out("    THE HYPOTHESIS IS NOT SUPPORTED: the paired move is %.2f standard errors "
            "from zero, which is %.1f or more, so the two keyings do not measure the same "
            "quantity up to noise and the move is a property of the keying. What is then "
            "unexplained is WHY the per-pair keying lands higher; the per-arm lines above "
            "say which of the four terms carries it, and nothing here says which of the "
            "two numbers a testbed would agree with."
            % (abs(wm) / wse, MOVE_SIGMA))
    else:
        out("    UNDECIDED: the paired move is %.2f standard errors from zero, inside the "
            "band the rule leaves open, so this measurement neither supports the "
            "hypothesis nor refutes it." % (abs(wm) / wse))
        out("    What is left unmeasured is how the seed-to-seed variance splits. Claim 1 "
            "asks the per-pair keying to cut the standard error by a factor, and it cuts "
            "it by less, which means the jitter is not the only thing varying from seed "
            "to seed: the outage instants and the record schedule vary too and are "
            "IDENTICAL in the two steps, so they sit in both standard errors and dilute "
            "the ratio. Splitting the two would need a sweep that holds the jitter draws "
            "fixed while the lags vary, and this script makes no such run.")
    # Printed whichever way the rule falls, because it is the question the four-step table
    # leaves open: a move smaller than the measurement's own standard error is a move no
    # reader may attribute to the thing that changed.
    # 2026-11-23 correction: this paragraph used to say that neither the step-3 move nor
    # the whole committed->fixed distance was "larger than the seed noise", which said more
    # than the numbers above.  1.16 standard errors IS larger than one standard error, and
    # the step 1 -> step 2 part of that distance is the same simulations rescored, so it
    # carries no seed-to-seed draw that a standard error could cover.  The noise comparison
    # belongs to the KEYING move alone, and the two are now printed apart.
    span = weights[4] - weights[1]
    # 2026-11-23: the band this sentence puts the paired move in is read off `claim2` and
    # not written into the sentence.  It used to assert "inside the 2.0 standard errors
    # claim 2 allows" in every branch, so on the NOT SUPPORTED branch -- where the move is
    # 3 standard errors or more -- the file would have printed a false statement in the
    # voice of a result.
    out("  Either way, and about the KEYING move alone: the step 2 -> step 3 move is %.2f "
        "times the step-2 weight's own standard error and the paired move is %.2f of its "
        "own, %s the %.1f standard errors claim 2 allows, so step 2's %.6f and step "
        "3's %.6f %s at these %d seeds and this cell %s the two jitter keyings apart."
        % (abs(move) / se[2], abs(wm) / wse,
           "inside" if claim2 else "OUTSIDE", NOISE_SIGMA,
           weights[2], weights[3],
           "do not separate" if claim2 else "DO separate", len(SEEDS),
           "cannot tell" if claim2 else "DOES tell"))
    out("  The step 1 -> step 2 move of %+.6f ms is a different kind of quantity and no "
        "standard error covers it: it is the SAME %d simulations of this cell scored "
        "twice, the censored records left out of the delay and then entered at their lower "
        "bound, so not one seed-to-seed draw enters the difference and its sign is settled "
        "in advance -- only the long arm has censored records, so leaving them out "
        "shortened its delay and the committed weight was biased DOWNWARD."
        % (weights[2] - weights[1], len(SEEDS) * len(ARMS) * NREP))
    out("  The whole distance from the committed value to the fixed one, %+.6f ms, is %.2f "
        "times step 2's standard error: larger than that standard error, and inside two of "
        "them. A reader who wants one sentence for it: the two values sit within two "
        "standard errors of each other, and the part of the distance that a rerun on other "
        "seeds could move is the keying move."
        % (span, abs(span) / se[2]))
    out("  The uncertainty a reader should carry on the FIXED weight: step 4 is "
        "%.6f ms per poll with a jackknife standard error of %.6f ms over the %d seeds."
        % (weights[4], se[4], len(SEEDS)))
    out("  The jackknife of steps 2 and 3 is taken on a MEDIAN over the %d streams, so it "
        "is an approximation there and its leave-one-out values can sit at a few levels "
        "rather than spread smoothly; step 4's is a mean and has no such caveat. On this "
        "data the risk did not materialise: the %d leave-one-out values are %d distinct for "
        "step 2, %d for step 3 and %d for step 4. The paired comparison does not depend on "
        "the median at all."
        % (NREP, len(SEEDS), len(set(loo[2])), len(set(loo[3])), len(set(loo[4]))))


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


# ---------------------------------------------------------------------------
# The four steps
# ---------------------------------------------------------------------------

def four_steps(res, p):
    """The four weights of one cell, each with the two arms' rows it was solved from."""
    steps = {}
    steps[1] = {arm: medians(per_stream_pooled(res, "port", p, arm, False))
                for arm in ARMS}
    steps[2] = {arm: medians(per_stream_pooled(res, "port", p, arm, True))
                for arm in ARMS}
    steps[3] = {arm: medians(per_stream_pooled(res, "portj", p, arm, True))
                for arm in ARMS}
    steps[4] = {arm: per_run_mean(res, "portj", p, arm) for arm in ARMS}
    return steps, {i: weight_of(steps[i][ARMS[0]], steps[i][ARMS[1]]) for i in steps}


STEP_TEXT = {
    1: ("as committed: the censored records LEFT OUT of the delay, pooled over the seeds "
        "per stream, median over the streams, the node's own jitter stream"),
    2: ("the same with every censored record at its LOWER BOUND, which is the label's "
        "delay; nothing else moved"),
    3: ("step 2 with the per-pair JITTER streams on as well, i.e. the engine the "
        "labelling runs"),
    4: ("the PER-RUN mean of the label's own score under step 3's engine: a run is one "
        "(seed, stream) and the mean is over all of them. Its censored column is the "
        "share over all those runs together, where steps 1 to 3 print the median of the "
        "40 per-stream shares"),
}


def print_steps(steps, weights):
    for i in (1, 2, 3, 4):
        out(f"  step {i} -- {STEP_TEXT[i]}")
        for arm in ARMS:
            out(arm_line(f"    fixed {arm} s", steps[i][arm]))
        out("    weight = (%.6f - %.6f) / (%.6f - %.6f) = %.17g ms per poll"
            % (steps[i][ARMS[1]][0], steps[i][ARMS[0]][0],
               steps[i][ARMS[0]][1], steps[i][ARMS[1]][1], weights[i]))
        if i > 1:
            out("    moved %+.6f ms (%+.2f %%) against step %d"
                % (weights[i] - weights[i - 1],
                   100.0 * (weights[i] - weights[i - 1]) / weights[i - 1], i - 1))


def report(res, fired, calls, off, wall, gate_ok):
    """Everything Acceptance 4 asks to be written down, and the comparisons beside it."""
    out("The weight that levels the two outer arms")
    out("=========================================")
    out(f"Cell: p {MAIN_P}, workload w0, the s6 outage shape, seeds {SEEDS[0]} to "
        f"{SEEDS[-1]}, {NREP} Mode B replicate streams, run dicts built by "
        "../21/predict.py's own run_dict.")
    out(f"Arms: {ARMS[0]} s and {ARMS[1]} s, as constant deciders over the choice set "
        f"{CHOICES} with the re-decision on, per-pair lag streams on throughout. The "
        "jitter keying is what steps 1 and 2 differ from steps 3 and 4 in: the node's "
        "own stream there, the ordered pair's own stream here, which is the labelling's.")
    out(f"Per-pair seed strings: rep_seed(\"{LAG_PREFIX}\", p, seed, rep), the same string "
        "for the lag and the jitter stream (port.Engine puts a different kind prefix on "
        "each). Reproducing any number here depends on that prefix; it is evidence/21's "
        "STREAM_LABEL and carries no arm.")
    out("The generator gate (../21/predict.py's own gate_generator): "
        + ("PASSED" if gate_ok else "FAILED")
        + " -- the outage events and record schedule of every seed that has a run are the "
          "stored ones, which is what the run dicts of the other seeds are built beside.")
    out(f"The scorer against ../14/after.py's early_score, on every one of the "
        f"{SIMS} simulations: {off} disagreed on (delivered, censored, delay sum, "
        "polls).")
    out()
    out("2026-11-23: the weight is re-measured because it and the label score used two "
        "different delays. The four steps below are printed one at a time so that each "
        "change is readable on its own; step 4 is the value weight.json carries.")
    out()

    main_steps, main_w = four_steps(res, MAIN_P)
    out("Main cell, through the port, the four steps")
    print_steps(main_steps, main_w)
    out()

    held = main_w[1] == COMMITTED
    out("Step 1 against the committed value")
    out("  step 1   = %.17g" % main_w[1])
    out("  committed= %.17g" % COMMITTED)
    out("  " + ("REPRODUCED exactly" if held else
                "DOES NOT REPRODUCE -- the difference is %.17g, so the step-1 pooling is "
                "not the one the committed value was measured with and nothing below may "
                "be read as a correction of it" % (main_w[1] - COMMITTED)))
    out()

    weight = main_w[4]
    per_run_counts = records_per_run(res, "portj", PS)
    out("THE FIXED VALUE, step 4")
    out("  records created at or before the cut, per run, over BOTH cells: %s. %s"
        % (per_run_counts,
           "One value, so every run's denominator is the same and step 3's pooled delay "
           "equals the mean of the per-run delays INSIDE a stream: what separates step 4 "
           "from step 3 here is only the mean over the 4000 runs against the median over "
           "the 40 streams." if len(per_run_counts) == 1 else
           "More than one value, so the pooled delay of step 3 weights the runs by their "
           "record count and the per-run mean of step 4 does not; the two steps differ by "
           "that as well as by mean against median."))
    out("  WEIGHT = %.17g ms per poll" % weight)
    out("         = %.6f ms per poll, to six decimals" % weight)
    scores = {arm: main_steps[4][arm][0] + weight * main_steps[4][arm][1]
              for arm in ARMS}
    digits = agreeing_digits(scores[ARMS[0]], scores[ARMS[1]])
    for arm in ARMS:
        out("  mean score(%-8s) = %.17g" % (f"{arm} s", scores[arm]))
    out("  difference = %.17g (relative %.3e)"
        % (scores[ARMS[1]] - scores[ARMS[0]],
           abs(scores[ARMS[1]] - scores[ARMS[0]]) / max(scores.values())))
    out("  they agree to %s"
        % ("every digit of a double: the difference is exactly 0" if digits is None
           else f"{digits} significant digits"))
    out("  This agreement is an arithmetic identity: the weight was solved for it, and "
        "the mean of a sum is the sum of the means, so it is a statement about the "
        "arithmetic and not about the arms. What carries information is the spread below.")
    out()

    per_stream = np.array([
        weight_of(
            (float(np.mean([SCORE.delay(res[("portj", MAIN_P, s)][ARMS[0]][rep])
                            for s in SEEDS])),
             float(np.mean([res[("portj", MAIN_P, s)][ARMS[0]][rep][4] for s in SEEDS]))),
            (float(np.mean([SCORE.delay(res[("portj", MAIN_P, s)][ARMS[1]][rep])
                            for s in SEEDS])),
             float(np.mean([res[("portj", MAIN_P, s)][ARMS[1]][rep][4] for s in SEEDS]))))
        for rep in range(NREP)])
    out("The step-4 weight computed inside each stream (the spread the mean hides)")
    out("  min %.3f   median %.3f   max %.3f   over %d streams"
        % (float(per_stream.min()), float(np.median(per_stream)),
           float(per_stream.max()), NREP))
    out("  The weight above is the one taken from the mean over all "
        f"{len(SEEDS) * NREP} runs, which is the definition; this line says how far one "
        "experiment's own answer can sit from it.")
    out()

    print_part_a(res, MAIN_P, main_w)
    out()

    p04_steps, p04_w = four_steps(res, 0.4)
    out("p 0.4, the same four steps, for information only; not used for the weight")
    print_steps(p04_steps, p04_w)
    out("  the step-4 weight at p 0.4 is %.2f times the main cell's"
        % (p04_w[4] / weight))
    out("  Whether a weight fixed on the main cell flips a label at p 0.4 is not "
        "answered here: that needs the labels themselves.")
    out()
    out("What the censored records do to the two cells, which is why step 2 exists: the "
        "censored share of the %s s arm is %.5f at p %s and %.5f at p 0.4, against "
        "%.5f and %.5f for the %s s arm. Leaving those records out shortens the long "
        "arm's delay more than the short arm's, so the committed weight was biased "
        "DOWNWARD, and more so at p 0.4."
        % (ARMS[1], main_steps[1][ARMS[1]][2], MAIN_P, p04_steps[1][ARMS[1]][2],
           main_steps[1][ARMS[0]][2], p04_steps[1][ARMS[0]][2], ARMS[0]))
    out()

    old_rows = {arm: medians(per_stream_pooled(res, "old", MAIN_P, arm, True))
                for arm in ARMS}
    w_old = weight_of(old_rows[ARMS[0]], old_rows[ARMS[1]])
    out("Main cell, the old fixed-arm path (no decider, no per-pair streams), step 2's "
        "pooling")
    for arm in ARMS:
        out(arm_line(f"fixed {arm} s", old_rows[arm]))
    out("  weight = %.6f ms per poll, %+.3f ms (%+.2f %%) against step 2's"
        % (w_old, w_old - main_w[2], 100.0 * (w_old - main_w[2]) / main_w[2]))
    out("  It is the port and the per-pair lag streams that this row prices, not the "
        "scoring: it is pooled and median-taken exactly as step 2 is.")
    out()

    tw, tr = ticket_weight()
    out("The ticket's estimate, recomputed from ../18/bound_cells.csv")
    for arm in ARMS:
        out("  %-34s delay %10.3f ms   polls %7.3f" % (f"fix{arm:g} row", *tr[arm]))
    out("  weight = %.6f ms per poll (the ticket writes it as 104 ms)" % tw)
    out("  It is a different set of runs: seeds 1 to 20 and 5 replicates, not "
        f"{SEEDS[0]} to {SEEDS[-1]} and {NREP}, the arms there are the old fixed-arm "
        "path, and its delay leaves the censored records out. It is a different "
        "question, not another try at this one.")
    out()

    out("Provisional timers that fired before their own poll had resolved: %d, over the "
        "%d simulations of this script, of which the %d port simulations are the ones "
        "that can show any (the old fixed-arm path arms no provisional timer at all: "
        "its send-time timer is final)."
        % (fired, SIMS, PORT_SIMS))
    out("  The resolution-time decision is the one that takes effect, so a fired "
        "provisional timer is a poll whose send-time interval decided the next send on "
        "its own, and a label on that decision would be a label on the wrong instant.")
    out(f"Decider calls over the port simulations of this script: {calls}. That is how "
        "many nested state copies a defensive deep copy of `obs[\"state\"]` would cost "
        "here, and the reason the contract is the guard instead.")
    out()
    out("wall %.1f s on %d workers" % (wall, WORKERS))
    return weight, held


def main():
    t = time.time()
    # The generator gate first: `run_dict` builds the seeds without a logged run from this
    # worktree's generator, so a generator that no longer agrees with the stored run
    # descriptions would make every number below a weight for instants the testbed never
    # laid down.  evidence/21's own script runs this before it prints anything; this one
    # did not, which left that dependency unchecked here.
    gate_ok = PR.gate_generator()
    if not gate_ok:
        print("STOP: the generator gate failed, so no weight is measured.")
        return 1
    for p in PS:
        for seed in SEEDS:
            RUNS[(p, seed)] = PR.run_dict(p, seed)
    keys = ([("port", p, s) for p in PS for s in SEEDS]
            + [("portj", p, s) for p in PS for s in SEEDS]
            + [("old", MAIN_P, s) for s in SEEDS])
    with mp.Pool(WORKERS) as pool:
        got = pool.map(job, keys, chunksize=1)
    res = {key: rows for key, rows, _f, _c, _o in got}
    fired = sum(f for _k, _r, f, _c, _o in got)
    calls = sum(c for key, _r, _f, c, _o in got if key[0] != "old")
    off = sum(o for _k, _r, _f, _c, o in got)
    weight, held = report(res, fired, calls, off, time.time() - t, gate_ok)
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"weight_ms_per_poll": weight,
                   "cell": {"p": MAIN_P, "workload": "w0", "outage_shape": "s6 8-12 s",
                            "seeds": [SEEDS[0], SEEDS[-1]], "nrep": NREP},
                   "arms_s": list(ARMS),
                   "delay": "per run, the censored records at their lower bound "
                            "(score.py); the mean over all seed x stream runs",
                   "pair_seed_prefix": LAG_PREFIX,
                   "path": "the decider port, constant deciders, per-pair lag streams "
                           "and per-pair jitter streams on",
                   "superseded": {
                       "weight_ms_per_poll": COMMITTED,
                       "why": "it levelled a delay that left the censored records out "
                              "(../14/after.py's early_score, pooled per stream and "
                              "median-taken over the streams), which is not the score a "
                              "label is decided by, so the two differed in their "
                              "definition of a delay",
                       "reproduced_as_step_1": held,
                   }},
                  fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {OUT_TXT} and {OUT_JSON}")
    # A step 1 that does not reproduce the committed value means the step-1 pooling is not
    # the one that produced it, and then nothing in this file is a correction of anything.
    # A scorer that disagrees with `early_score` means the record set moved.  Either is a
    # failure of this script, not a line in its output.
    if not held:
        print("FAIL step 1 did not reproduce the committed weight")
    if off:
        print(f"FAIL the scorer disagreed with early_score on {off} simulations")
    return 1 if (not held or off) else 0


if __name__ == "__main__":
    sys.exit(main())
