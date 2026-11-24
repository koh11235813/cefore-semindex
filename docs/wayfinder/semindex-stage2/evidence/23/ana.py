#!/usr/bin/env python3
"""The selecting score, the frontier gain, and the choice between candidate a and b.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src \
      <python> -B ana.py <eval.csv> [--dry]        > ana.txt

It reads that csv AND NOTHING ELSE -- no card, no model, no run store, no label file --
so a reader can re-run it on either machine in seconds and get the same bytes.  Nothing
below is a clock time or a path of the machine it ran on, for the same reason `parity.py`
prints none: the output is meant to be compared with `cmp` across the two hosts.

WHAT IT COMPUTES, AND WHICH NUMBER DECIDES WHAT
  1. THE SELECTING STATISTIC: the mean of `score.py`'s `delay + weight x polls` over a
     decider's runs, one run being one (seed, stream).  Lower is better.  This number,
     and nothing else, chooses between candidate a and candidate b.
  2. THE REPORTED FRONTIER GAIN, with a 95 % interval: `1 - delay(decider) /
     interp(frontier, polls(decider))` through `../20/predict.py`'s own `interp` and
     `gain`, whose delay is `lat / n4` with the censored records DROPPED.  Two different
     delays and two different directions of "better": the score selects, the gain is
     reported beside it.
  3. THE CHOICE: the lower mean score wins unless the two are CLOSE, which is
     `|mean(a) - mean(b)| < 297 ms`.  Close opens the relabelling question and carries b.
     A "separated" verdict whose gap lies inside the paired bootstrap's 95 % interval is
     reported as "separated, not resolved".

WHY THE TESTBED STAGE NAMES APPEAR IN THIS OUTPUT
  `../20/predict.py`'s `gain` reads its frontier from `ARM_STAGES = ["s10b3p695",
  "s10b6p0", "s10b10p043"]`, which name testbed stages this step never ran.  This step's
  OWN three re-run arms are stored under those keys so that `interp` and `gain` stay
  single-sourced instead of being written again here.  A reader who sees `s10b3p695`
  below must not read it as `../20`'s run: it is this step's `const3.695`, re-run on the
  replay port on this step's own per-pair streams.

WHAT A DRY RUN MEANS: NOTHING ABOUT WHICH CANDIDATE IS BETTER.  `--dry` reads a csv of
TUNING seeds that both candidates were fitted on, the bootstrap over two seeds is
degenerate, and every number below is then an exercise of the arithmetic and not a result.
"""
import os
import sys

# 2026-11-24: the thread caps before numpy is imported, as `parity.py` sets them and for
# the same reason: a BLAS library opens one thread per core at load time on the 96-core
# shared host, and this script does no float reduction a thread count could reorder.
for _name, _value in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
                      ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_name, _value)

import importlib.util  # noqa: E402
import json            # noqa: E402
import statistics      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
EV20 = os.path.normpath(os.path.join(HERE, "..", "20"))
sys.path.insert(0, EV22)
import port as PORT  # noqa: E402  the >100 seed guard, on this script's path as well

import numpy as np  # noqa: E402

# `../20/predict.py` by path and under its own module name, the way `../22/label.py` loads
# `../21/predict.py`: two files in this tree are called `predict.py`, and importing either
# by name would make which one is loaded depend on the order of sys.path.
_spec = importlib.util.spec_from_file_location("predict20",
                                               os.path.join(EV20, "predict.py"))
PR20 = importlib.util.module_from_spec(_spec)
sys.modules["predict20"] = PR20
_spec.loader.exec_module(PR20)

ARM_STAGES = PR20.ARM_STAGES         # the three keys `gain` reads its frontier from
BOOT_N = PR20.BOOT_N                 # 2000 resamples
BOOT_SEED = PR20.BOOT_SEED           # 20261122
COL = 0                              # this step has one column, not evidence/20's two

# The weight the scores in the csv were taken at, read from the file that holds it rather
# than imported: `../22/label.py` would pull in `port.py`'s whole engine and the store, and
# this script is meant to run with neither.  Every row of the csv carries the weight it was
# scored at and is checked against this value.
with open(os.path.join(EV22, "weight.json"), encoding="utf-8") as _fh:
    WEIGHT = json.load(_fh)["weight_ms_per_poll"]

# The band, from `../22/weight.txt`: the fixed weight's jackknife standard error over the
# 100 seeds, times the midpoint of the two outer arms' step-4 poll counts (256.455 and
# 95.985).  The RULE is the plan's own 297 ms; the product is printed beside it so that a
# reader sees the arithmetic and not a constant.
JACK_SE = 1.685586                   # ms per poll
POLL_MID = 176.220                   # polls per run
BAND_MS = 297.0                      # the rule, as the frozen plan states it

ARM_NAMES = ("const3.695", "const6.000", "const10.043")
MODEL_NAMES = ("untuned", "a", "b")
ORDER = ARM_NAMES + ("F",) + MODEL_NAMES
ARM_KEY = dict(zip(ARM_NAMES, ARM_STAGES))

OUT = []


def say(text=""):
    """Print and keep, so the whole report is one stream the caller redirects."""
    print(text, flush=True)
    OUT.append(text)


def head(title):
    """A part heading, in this directory's own shape."""
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


class Fail(Exception):
    """A check that did not hold.

    A type of its own, because part 5 feeds every check below a mutated input and requires
    it to raise: catching `Exception` there would also catch a typo in the mutation and
    print it as if the check had worked.
    """


# ---------------------------------------------------------------------------
# the checks, every one of them a function of plain values
# ---------------------------------------------------------------------------
def check_roles(roles, want, where):
    """Each seed role is inside its range and none of them meets 121 to 160.

    `decider.py` holds the same check and this script cannot import it: `decider.py`
    imports `prefill.py`, which resolves ROOT and the offline Hugging Face variables at
    its own import time and refuses to load without a ROOT directory, and the whole point
    of this script is that it needs no ROOT, no card and no model.  So the ranges are
    restated here, and the restatement is held to the data: the roles below are read off
    the csv's own seed column and not declared.
    """
    union = sorted({int(s) for seeds in roles.values() for s in seeds})
    met = sorted(set(union) & set(range(121, 161)))
    if met:
        raise Fail("%s: the seed set meets the measuring seeds 121-160: %s" % (where, met))
    for role, seeds in sorted(roles.items()):
        lo, hi = want[role]
        bad = sorted(int(s) for s in seeds if not lo <= int(s) <= hi)
        if bad:
            raise Fail("%s: the %s role carries %s, outside %d-%d"
                       % (where, role, bad[:4], lo, hi))
    return union


def check_one_digest(rows, column, where):
    """Every row of one decider carries the same script digest and the same model digest.

    `eval.py` refuses to resume a part file written under another digest; this is the same
    refusal read off the finished csv, so a csv assembled by hand out of two runs is
    refused here even though `eval.py` never saw it.
    """
    got = sorted({r[column] for r in rows})
    if len(got) != 1:
        raise Fail("%s: %d different %s values: %s" % (where, len(got), column, got[:3]))
    return got[0]


def check_weight(rows, want, where):
    """Every row was scored at the weight this script prices the comparison with.

    The selecting statistic IS a weighted sum, so a csv scored at another weight is a csv
    of another statistic, and the band below is built from this weight's own uncertainty.
    """
    bad = sorted({r["weight_ms_per_poll"] for r in rows if r["weight_ms_per_poll"] != want})
    if bad:
        raise Fail("%s: %d row(s) at another weight, e.g. %r against %r"
                   % (where, len(bad), bad[0], want))
    return want


def check_score(rows, weight, where):
    """The score column is `delay + weight x polls` of the same row's own terms.

    Recomputed and not trusted, because the csv carries both the terms and the score and
    the two have to be of one run: the floats are written through `repr`, so the
    recomputation is exact and a difference of any size is a defect and not rounding.
    """
    for r in rows:
        delay = (r["lat_ms"] + r["censored_ms"]) / (r["n4"] + r["c4"])
        want = delay + weight * r["polls"]
        if want != r["score_ms"] or delay != r["delay_ms"]:
            raise Fail("%s: %s seed %d stream %d: delay %r and score %r recompute to %r "
                       "and %r" % (where, r["decider"], r["seed"], r["stream"],
                                   r["delay_ms"], r["score_ms"], delay, want))
    return len(rows)


def check_grid(keys, seeds, streams, where):
    """The csv carries the full (seed, stream) grid the MODE declares, and no other.

    2026-11-24: `check_complete` below is handed `keys` read off the csv's own rows, so a
    csv that lost an entire seed or an entire stream from every decider agreed with
    itself and passed every check this script makes, and the verdict printed.  Reproduced
    on the dry csv with every stream-1 row removed: exit 0, "stream column min 0 max 0,
    1 distinct", "runs 2 per decider", and a verdict.  So the grid is asserted against
    the product the mode names and not against what was found.

    The product is eval.py's: 2 seeds x 2 streams in the dry mode, 20 x 40 in the real
    one.  It is restated here rather than imported because this script must import
    neither prefill nor decider -- it needs no ROOT, no card and no model, which is what
    lets a reader re-run it anywhere -- and a wrong restatement is a failure here and not
    a silent pass.
    """
    want = sorted((s, r) for s in seeds for r in streams)
    got = sorted(keys)
    if got != want:
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        raise Fail("%s: the csv carries %d of the %d (seed, stream) pairs this mode "
                   "declares; %d missing (e.g. %s) and %d unexpected (e.g. %s)"
                   % (where, len(got), len(want), len(missing), missing[:3],
                      len(extra), extra[:3]))
    return len(want)


def check_k(rows, model_names, where):
    """The three model deciders were scored at ONE worker count, and the other four at 1.

    2026-11-24: K was in no digest and in no row, so a real run started at --k 4, killed
    after some deciders, and restarted at --k 1 -- which is the plan's own fallback --
    produced one csv whose deciders were scored at two concurrency levels with nothing
    recording it.  The gate makes the arithmetic equal by construction, so no number
    would have been wrong; the record was incomplete and a reader could not tell.

    The four deciders that call no model are scored in the parent, one process, so their
    rows carry 1 whatever K was, and that is asserted rather than tolerated: a 4 there
    would mean a row had been produced somewhere this script does not know about.
    """
    model = sorted({r["k"] for r in rows if r["decider"] in model_names})
    other = sorted({r["k"] for r in rows if r["decider"] not in model_names})
    if len(model) != 1:
        raise Fail("%s: the model deciders' rows were produced at %d different worker "
                   "counts: %s" % (where, len(model), model))
    if other != [1]:
        raise Fail("%s: the deciders that make no forward pass carry the worker counts "
                   "%s, not [1]" % (where, other))
    return model[0]


def check_complete(rows, deciders, keys, where):
    """Exactly one row per (decider, seed, stream), no row twice and none missing.

    A mean over a csv that lost a worker's rows would be a mean over whichever runs
    finished, and nothing in the arithmetic below would notice.  What it does NOT do is
    say which rows SHOULD be there: `keys` is read off the csv, so `check_grid` above is
    the line of defence that compares the csv against the mode's declared product.
    """
    want = sorted((d, s, r) for d in deciders for s, r in keys)
    got = sorted((r["decider"], r["seed"], r["stream"]) for r in rows)
    if len(got) != len(set(got)):
        twice = sorted({k for k in got if got.count(k) > 1})
        raise Fail("%s: %d row(s) appear twice, e.g. %s" % (where, len(twice), twice[:3]))
    if got != want:
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        raise Fail("%s: %d of %d missing (e.g. %s) and %d unexpected (e.g. %s)"
                   % (where, len(missing), len(want), missing[:2], len(extra), extra[:2]))
    return len(got)


def check_row_order(row, n4, c4, lat, polls, where):
    """The per-seed row handed to `predict.gain` is `[n4, c4, lat, polls]` and nothing else.

    THE ONE DEFECT OF THIS ANALYSIS THAT RAISES NOTHING AND PRINTS A PLAUSIBLE NUMBER.
    `score.early_four` returns `(n4, c4, lat, polls)` and `predict.vals` unpacks exactly
    that, returning `lat / n4`.  An earlier draft of the frozen plan wrote the order
    `[lat, n, c, polls]`, which hands `vals` `lat` where `n4` belongs and produces a
    "delay" of `n4 / lat`, five orders of magnitude off, with `interp` then read at the
    wrong x.  So the order is not commented, it is checked: the row is compared against
    the four quantities by name AND `vals` is required to read the delay, the polls and
    the censored share back out of it.
    """
    want = [n4, c4, lat, polls]
    if list(row) != want:
        raise Fail("%s: the row is %r, not [n4, c4, lat, polls] = %r"
                   % (where, list(row), want))
    delay, per, cens = PR20.vals(np.array([row], float))
    if delay != lat / n4:
        raise Fail("%s: vals read the delay as %r, not lat / n4 = %r"
                   % (where, delay, lat / n4))
    if per != polls:
        raise Fail("%s: vals read the polls as %r, not %r" % (where, per, polls))
    if cens != c4 / (n4 + c4):
        raise Fail("%s: vals read the censored share as %r, not %r"
                   % (where, cens, c4 / (n4 + c4)))
    return want


def check_scale(m, scaled, name, where, tol=1e-9):
    """The gain does not move when every stage's rows are scaled by the same factor.

    The plan pools a seed's streams by the COLUMN-WISE SUM, so `vals`' `polls / len(m)` is
    a seed's poll count and not a run's -- 40 times the per-run figure in the real mode.
    `interp` is piecewise linear in log-log, so a factor common to the frontier's x and
    the decider's x cancels and the gain is the same number; the per-run poll counts are
    printed separately, for the reader.  That is an argument, so it is checked here
    against the same rows divided by the stream count rather than asserted in a comment.
    Scaling ONE stage alone DOES move the gain, which is the mutation in part 5 and is
    what keeps this from being a check that cannot fail.
    """
    here = PR20.gain(m, name, COL)
    there = PR20.gain(scaled, name, COL)
    if (here is None) != (there is None):
        raise Fail("%s: the gain is %r on the summed rows and %r on the scaled rows"
                   % (where, here, there))
    if here is not None and abs(here - there) > tol:
        raise Fail("%s: the gain moves from %.9f to %.9f when every stage is scaled by "
                   "the same factor" % (where, here, there))
    return here


def check_boot(idx, n, where):
    """The resample matrix is `(2000, n)` and every index is a position of this run.

    `boot_index(n)` changes both the draw range and the shape with `n`, so at 20 seeds no
    element of it is an element of `../20`'s 40-seed matrix: what is shared with those
    steps is the generator and the seed, and the matrix is this step's own.
    """
    if tuple(idx.shape) != (BOOT_N, n):
        raise Fail("%s: the resample matrix is %s, not (%d, %d)"
                   % (where, tuple(idx.shape), BOOT_N, n))
    if int(idx.min()) < 0 or int(idx.max()) >= n:
        raise Fail("%s: an index is outside 0 to %d: min %d max %d"
                   % (where, n - 1, int(idx.min()), int(idx.max())))
    return idx.shape


# ---------------------------------------------------------------------------
# reading the csv
# ---------------------------------------------------------------------------
INT_COLS = ("seed", "stream", "n4", "c4", "polls", "ord_poll", "ord_resched", "k",
            "reads", "sends", "ties", "n_short", "n_mid", "n_long")
# The grid each mode declares, which is eval.py's own DRY_SEEDS/DRY_REPS and
# REAL_SEEDS/REAL_REPS.  Restated and not imported, for the reason at `check_grid`.
DRY_GRID = (range(1, 3), range(0, 2))
REAL_GRID = (range(81, 101), range(0, 40))
FLOAT_COLS = ("lat_ms", "censored_ms", "delay_ms", "score_ms", "score_dropped_ms",
              "weight_ms_per_poll")


def read_csv(path):
    """Every row of `eval.py`'s csv, with the numeric columns typed.

    The four deciders that call no model leave `reads`, `sends`, `ties` and the three
    answer counts empty, which is why an empty cell is kept as None rather than read as a
    zero: a zero would say "this decider made no tie" where the truth is that the question
    does not apply to a constant arm.
    """
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    cols = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        row = dict(zip(cols, line.split(",")))
        for c in INT_COLS:
            row[c] = int(row[c]) if row[c] != "" else None
        for c in FLOAT_COLS:
            row[c] = float(row[c]) if row[c] != "" else None
        rows.append(row)
    return cols, rows


def per_seed(rows, seeds):
    """One `(len(seeds), 4)` array of `[n4, c4, lat, polls]`, summed over each seed's runs.

    The pooling the plan names: the column-wise sum, which `vals`' own `m.sum(0)` then
    re-sums over the seeds and which makes the pooled delay `sum(lat) / sum(n4)`.  It is
    NOT a mean of the per-stream gains -- `../20` computes a gain per stream and prints
    the median over the 40, a different estimator -- because the selecting statistic is a
    mean over the runs and a gain taken as a median over streams would not summarise the
    same runs.
    """
    by = {s: [0.0, 0.0, 0.0, 0.0] for s in seeds}
    for r in rows:
        acc = by[r["seed"]]
        acc[0] += r["n4"]
        acc[1] += r["c4"]
        acc[2] += r["lat_ms"]
        acc[3] += r["polls"]
    if len({len([r for r in rows if r["seed"] == s]) for s in seeds}) != 1:
        raise Fail("per_seed: the seeds do not all carry the same number of runs")
    return np.array([by[s] for s in seeds], float)


# ===========================================================================
# 0  WHAT THIS RUN IS
# ===========================================================================
ARGV = sys.argv[1:]
DRY = "--dry" in ARGV
PATHS = [a for a in ARGV if not a.startswith("--")]
if len(PATHS) != 1:
    raise SystemExit("usage: ana.py <eval.csv> [--dry]")
CSV = PATHS[0]

head("0  WHAT THIS IS, AND WHAT IT CANNOT BE")
if DRY:
    say("A DRY RUN.  The csv it reads is of TUNING seeds that both candidates were fitted")
    say("on, so every score, every gain and every verdict below is an exercise of the")
    say("arithmetic on training data and says NOTHING about which candidate is better.")
    say("The bootstrap is degenerate at two seeds: a resample of two positions has only")
    say("three distinct means, so the interval below is the arithmetic running and not an")
    say("interval.")
else:
    say("The two statistics of the comparison, on seeds 81 to 100 of the main cell.  The")
    say("mean selecting score chooses between candidate a and candidate b and nothing")
    say("else does; the frontier gain is reported beside it.")
say()
say("This script reads one csv and nothing else: no card, no model, no run store, no")
say("label file.  It prints no clock time and no path of the machine it ran on, so the")
say("two hosts' output can be compared with cmp.")

# ===========================================================================
# 1  THE CSV
# ===========================================================================
head("1  THE CSV, ITS DIGESTS AND ITS SEEDS")
COLS, ROWS = read_csv(CSV)
SEEDS = sorted({r["seed"] for r in ROWS})
STREAMS = sorted({r["stream"] for r in ROWS})
KEYS = sorted({(r["seed"], r["stream"]) for r in ROWS})
DECIDERS = sorted({r["decider"] for r in ROWS})
say("rows            %d" % len(ROWS))
say("columns         %s" % ",".join(COLS))
say("deciders        %s" % ", ".join(ORDER))
say("seed column     min %d max %d, %d distinct" % (SEEDS[0], SEEDS[-1], len(SEEDS)))
say("stream column   min %d max %d, %d distinct"
    % (STREAMS[0], STREAMS[-1], len(STREAMS)))
say("runs            %d per decider" % len(KEYS))
if sorted(DECIDERS) != sorted(ORDER):
    raise Fail("the csv carries the deciders %s, not the seven of the comparison %s"
               % (DECIDERS, sorted(ORDER)))
SCRIPT_SHA = check_one_digest(ROWS, "script_sha", "the csv")
say("script digest   %s, one value over every row" % SCRIPT_SHA)
say("                It is a digest over the files eval.py's own output names one by one,")
say("                beside the list of what it does NOT cover.  This script cannot")
say("                widen it or name its parts: it reads the csv and nothing else, which")
say("                is what lets a reader re-run it on any machine.")
for NAME in ORDER:
    rows = [r for r in ROWS if r["decider"] == NAME]
    say("  %-12s %4d rows, model digest %s"
        % (NAME, len(rows), check_one_digest(rows, "model_sha", "the rows of %s" % NAME)))
check_weight(ROWS, WEIGHT, "the csv")
say("weight          %r ms per poll, the value ../22/weight.json carries and every row"
    % WEIGHT)
say("                was scored at")
check_score(ROWS, WEIGHT, "the csv")
say("OK: delay + weight x polls recomputes the score column of all %d rows exactly"
    % len(ROWS))
GRID_SEEDS, GRID_STREAMS = DRY_GRID if DRY else REAL_GRID
check_grid(KEYS, GRID_SEEDS, GRID_STREAMS, "the csv")
say("grid            %d seeds x %d streams = %d runs per decider, which is the product"
    % (len(GRID_SEEDS), len(GRID_STREAMS), len(GRID_SEEDS) * len(GRID_STREAMS)))
say("                this mode DECLARES, not a count of what the csv happened to carry:")
say("                a csv that lost a whole seed or a whole stream from every decider")
say("                would agree with itself and is refused here")
check_complete(ROWS, ORDER, KEYS, "the csv")
say("OK: exactly one row per (decider, seed, stream), none twice and none missing")
say("K               %d worker process(es) read out every model decider's rows, one value"
    % check_k(ROWS, MODEL_NAMES, "the csv"))
say("                over the three of them; the four that make no forward pass are")
say("                scored in eval.py's parent and carry 1")

ROLES = {"tuning": list(range(1, 71)), "held out": list(range(71, 81)),
         "selection": list(range(81, 101)), "scored in this csv": SEEDS}
WANT = {"tuning": (1, 70), "held out": (71, 80), "selection": (81, 100),
        "scored in this csv": (1, 80) if DRY else (81, 100)}
UNION = check_roles(ROLES, WANT, "ana.py: the seed roles")
say()
for role in ("tuning", "held out", "selection", "scored in this csv"):
    seeds = ROLES[role]
    say("  %-19s %2d seeds, %d to %d" % (role, len(seeds), min(seeds), max(seeds)))
say("  the union of the four meets range(121, 161) in %d seeds"
    % len(set(UNION) & set(range(121, 161))))
PORT.check_seeds(UNION, "ana.py: the four seed roles")
say("  port.check_seeds (>100 refused) passed all four, and is shown refusing in part 5.")
say("  This script starts no simulation and reads no label row, so the first item of the")
say("  seeds-never-used proof is eval.py's and the fourth is read off the csv's own seed")
say("  column above.")

# ===========================================================================
# 2  THE SELECTING STATISTIC
# ===========================================================================
head("2  THE SELECTING STATISTIC: THE MEAN SCORE OVER THE RUNS")
say("`score.score_run(events, t0, %r)[0]` = delay + weight x polls, with every censored" % WEIGHT)
say("record entering the delay at its lower bound, meaned over the %d runs of a decider."
    % len(KEYS))
say("Lower is better.  THIS NUMBER, AND NOTHING ELSE, CHOOSES BETWEEN a AND b.")
say()
MEAN, PER_RUN_POLLS, PER_SEED = {}, {}, {}
for NAME in ORDER:
    rows = [r for r in ROWS if r["decider"] == NAME]
    MEAN[NAME] = statistics.fmean(r["score_ms"] for r in rows)
    PER_RUN_POLLS[NAME] = statistics.fmean(r["polls"] for r in rows)
    PER_SEED[NAME] = {s: statistics.fmean(r["score_ms"] for r in rows if r["seed"] == s)
                      for s in SEEDS}
say("  decider        mean score ms   mean polls/run   mean delay ms   ties / decisions")
for NAME in ORDER:
    rows = [r for r in ROWS if r["decider"] == NAME]
    ties = sum(r["ties"] for r in rows) if rows[0]["ties"] is not None else None
    reads = sum(r["reads"] for r in rows) if rows[0]["reads"] is not None else None
    say("  %-12s %14.3f %16.3f %15.3f   %s"
        % (NAME, MEAN[NAME], PER_RUN_POLLS[NAME],
           statistics.fmean(r["delay_ms"] for r in rows),
           "--" if ties is None else "%d / %d" % (ties, reads)))
say()
say("ranked, lower first: %s"
    % ", ".join("%s %.1f" % (n, MEAN[n]) for n in sorted(ORDER, key=lambda n: MEAN[n])))
say("The three fixed arms' own poll counts on these seeds, which the gain's \"no gain\"")
say("message quotes rather than ../22/weight.txt's 95.985 and 256.455 (those are step 4's")
say("means over seeds 1 to 100 and would name an interval this test did not use):")
ARM_POLLS = sorted(PER_RUN_POLLS[n] for n in ARM_NAMES)
say("  %.3f to %.3f polls per run" % (ARM_POLLS[0], ARM_POLLS[-1]))

# ===========================================================================
# 3  THE FRONTIER GAIN
# ===========================================================================
head("3  THE REPORTED FRONTIER GAIN, WITH ITS 95 % INTERVAL")
say("A DIFFERENT DELAY FROM THE ONE ABOVE: `predict.vals` is `lat / n4`, i.e. the censored")
say("records DROPPED, pooled over a seed's runs and then over the seeds.  The score above")
say("selects; this is reported beside it, and the two disagree about the censored records")
say("by construction.")
say()
M = {}
for NAME in ORDER:
    rows = [r for r in ROWS if r["decider"] == NAME]
    M[ARM_KEY.get(NAME, NAME)] = {COL: per_seed(rows, SEEDS)}
say("the three re-run arms are stored under ../20's own frontier keys, so that interp and")
say("gain stay single-sourced:")
for NAME in ARM_NAMES:
    say("  %-12s -> %s" % (NAME, ARM_KEY[NAME]))
say()
# The order check, on a row this script builds from the csv by name.  It is made on the
# first seed's row of the first arm, which is the same shape every row handed to `gain` has.
_first = [r for r in ROWS if r["decider"] == ARM_NAMES[0] and r["seed"] == SEEDS[0]]
check_row_order(M[ARM_STAGES[0]][COL][0],
                sum(r["n4"] for r in _first), sum(r["c4"] for r in _first),
                sum(r["lat_ms"] for r in _first), sum(r["polls"] for r in _first),
                "the per-seed row handed to predict.gain")
say("OK: the per-seed row handed to predict.gain is [n4, c4, lat, polls], checked against")
say("the four sums read out of the csv by column name, and predict.vals reads the delay,")
say("the polls and the censored share back out of it.  The order is a silent defect and")
say("not an error: the order [lat, n, c, polls] gives a \"delay\" of n4 / lat, five orders")
say("of magnitude off, and prints a plausible gain.  Shown firing in part 5.")
say()
_scaled = {k: {COL: v[COL] / len(STREAMS)} for k, v in M.items()}
say("the frontier, as interp reads it (pooled delay against a seed's poll count, which is")
say("%d times a run's; interp is piecewise linear in log-log, so a factor common to the" % len(STREAMS))
say("frontier's x and the decider's x cancels and the gain is the same number -- checked")
say("below against the same rows divided by %d):" % len(STREAMS))
for KEY in ARM_STAGES:
    delay, polls, cens = PR20.vals(M[KEY][COL])
    say("  %-14s delay %10.3f ms   polls %10.3f per seed (%8.3f per run)   censored %.4f"
        % (KEY, delay, polls, polls / len(STREAMS), cens))
_arms = [(PR20.vals(M[a][COL])[1], PR20.vals(M[a][COL])[0]) for a in ARM_STAGES]
_ordered = sorted(_arms)
MONOTONE = all(y0 > y1 for (_, y0), (_, y1) in zip(_ordered, _ordered[1:]))
say("  monotone (more polls, less delay): %s" % MONOTONE)
say()
IDX = PR20.boot_index(len(SEEDS))
check_boot(IDX, len(SEEDS), "the resample matrix")
say("the resample loop is this script's own and adopts ../20's drop-and-count rule by")
say("reference: %d resamples of the %d SEEDS from one integers(0, %d, size=(%d, %d))"
    % (BOOT_N, len(SEEDS), len(SEEDS), BOOT_N, len(SEEDS)))
say("matrix drawn once with default_rng(%d).  A resample whose arms are not monotone, or"
    % BOOT_SEED)
say("whose decider's polls leave the arms, YIELDS NO GAIN AND IS COUNTED AND DROPPED, not")
say("substituted, because substituting a value would hide how often the frontier is")
say("unreadable; the dropped count is printed beside every interval.")
say()
say("  decider        gain      95 % interval            half width   dropped   word")
GAINS = {}
for NAME in ("F",) + MODEL_NAMES:
    key = ARM_KEY.get(NAME, NAME)
    point = check_scale(M, _scaled, key, "the gain of %s" % NAME)
    if point is None:
        delay, polls, _c = PR20.vals(M[key][COL])
        if not MONOTONE:
            say("  %-12s no gain: the three arm points are not monotone" % NAME)
        else:
            lo = min(PR20.vals(M[a][COL])[1] for a in ARM_STAGES) / len(STREAMS)
            hi = max(PR20.vals(M[a][COL])[1] for a in ARM_STAGES) / len(STREAMS)
            say("  %-12s no gain: polls %.3f outside [%.3f, %.3f]"
                % (NAME, polls / len(STREAMS), lo, hi))
        GAINS[NAME] = None
        continue
    boot, dropped = [], 0
    for row in IDX:
        sub = {k: {COL: M[k][COL][row]} for k in M}
        g = PR20.gain(sub, key, COL)
        if g is None:
            dropped += 1
            continue
        boot.append(g)
    lo, hi, half = PR20.ci(boot) if boot else (float("nan"),) * 3
    GAINS[NAME] = (point, lo, hi, half, dropped)
    say("  %-12s %7.4f   %8.3f %% to %8.3f %%   %8.3f %%   %7d   %s"
        % (NAME, point, lo, hi, half, dropped, PR20.word(lo, hi)))
say()
say("The real mode's 95 % interval, at 20 seeds, is expected to be wider than ../20's at")
say("40 by roughly sqrt(2); that is an INFERENCE and not a measurement, and the printed")
say("half width is what counts.  This csv carries %d seeds." % len(SEEDS))
say("boot_gains is NOT called: its TESTED keys name ../20's two rules and not this step's")
say("four model-or-rule deciders.")

# ===========================================================================
# 4  THE CHOICE BETWEEN a AND b
# ===========================================================================
head("4  HOW THE CHOICE BETWEEN a AND b IS MADE")
GAP = abs(MEAN["a"] - MEAN["b"])
LOWER = "a" if MEAN["a"] < MEAN["b"] else "b"
say("mean score      a %12.3f ms,  b %12.3f ms" % (MEAN["a"], MEAN["b"]))
say("gap             %.3f ms, the lower being %s" % (GAP, LOWER))
say("the band        %.6f ms per poll x %.3f polls = %.4f ms, and the RULE the frozen"
    % (JACK_SE, POLL_MID, JACK_SE * POLL_MID))
say("                plan states is %.1f ms.  It is the jackknife standard error" % BAND_MS)
say("                ../22/weight.txt attaches to the fixed weight, times the midpoint of")
say("                the two outer arms' step-4 poll counts, i.e. the amount by which")
say("                re-solving the weight on another draw of seeds would move a typical")
say("                run's score.  IT IS NOT A CONFIDENCE INTERVAL ON THE DIFFERENCE and")
say("                it is not the seed-to-seed spread of a mean score, which the paired")
say("                bootstrap below carries.  The tighter of weight.txt's two readings")
say("                was taken (the per-stream weight range gives 3.676 x 176.220 = 647.8")
say("                ms), which makes \"separated\" the EASIER verdict and not the harder")
say("                one, so the separated branch is the weaker-evidence branch of the two.")
say()
say("the part of the band that does NOT cancel: a weight error shifts both candidates in")
say("the same direction and cancels to the extent their poll counts agree, so")
say("  %.6f x |polls(a) - polls(b)| = %.6f x |%.3f - %.3f| = %.3f ms"
    % (JACK_SE, JACK_SE, PER_RUN_POLLS["a"], PER_RUN_POLLS["b"],
       JACK_SE * abs(PER_RUN_POLLS["a"] - PER_RUN_POLLS["b"])))
say("is the part of the wobble that survives the cancellation.  It is computable only")
say("after the scores exist, so it is reported and is not the rule.")
say()
DIFF = np.array([PER_SEED["a"][s] - PER_SEED["b"][s] for s in SEEDS], float)
BOOT_D = np.array([DIFF[row].mean() for row in IDX], float)
D_LO, D_HI = (float(x) for x in np.percentile(BOOT_D, [2.5, 97.5]))
say("the paired bootstrap over the same %d seeds and the same index matrix, whose per-seed"
    % len(SEEDS))
say("value is the MEAN OF THAT SEED'S %d RUN SCORES -- the only aggregate under which"
    % len(STREAMS))
say("resampling %d seed indices reproduces the %d-run mean the rule compares:"
    % (len(SEEDS), len(ROWS) // len(ORDER)))
say("  score(a) - score(b): point %.3f ms, 95 %% interval %.3f to %.3f ms, contains 0: %s"
    % (float(DIFF.mean()), D_LO, D_HI, D_LO <= 0.0 <= D_HI))
say("  distinct resample values: %d of %d" % (len(set(BOOT_D.tolist())), BOOT_N))
if DRY:
    say("  AT TWO SEEDS THIS IS DEGENERATE: a resample of two positions has three distinct")
    say("  means, so the interval is the arithmetic running and not an interval.")
say()
CLOSE = GAP < BAND_MS
if CLOSE:
    say("VERDICT: CLOSE.  %.3f ms < %.1f ms, so the relabelling clause of the user's"
        % (GAP, BAND_MS))
    say("decision 6 opens, neither candidate is called better, and the candidate carried")
    say("to the next ticket is b -- a tie-break convention and not a measured preference:")
    say("b's filter is the one whose agreement across two independent labellings was")
    say("measured, and relabelling with more passes changes b's row set and not a's.")
    CARRY = "b"
elif D_LO <= 0.0 <= D_HI:
    say("VERDICT: SEPARATED, NOT RESOLVED.  The gap of %.3f ms clears the %.1f ms band,"
        % (GAP, BAND_MS))
    say("so the relabelling clause stays shut and the chosen candidate is the lower one,")
    say("%s; but the gap lies inside the paired bootstrap's 95 %% interval (%.3f to %.3f"
        % (LOWER, D_LO, D_HI))
    say("ms, which contains 0), so this report does not call the two distinguished.")
    CARRY = LOWER
else:
    say("VERDICT: SEPARATED.  The gap of %.3f ms clears the %.1f ms band and lies outside"
        % (GAP, BAND_MS))
    say("the paired bootstrap's 95 %% interval (%.3f to %.3f ms), so the chosen candidate"
        % (D_LO, D_HI))
    say("is the lower one, %s." % LOWER)
    CARRY = LOWER
say()
say("THE CANDIDATE THIS ANALYSIS CARRIES FORWARD: %s" % CARRY)
if DRY:
    say("...on TUNING seeds, so that name is the arithmetic's answer and not the step's.")
say()
say("The selection bias of choosing one of two candidates on the same seeds is not")
say("corrected anywhere: lower is better and the rule picks the lower of two, so the")
say("winner's figure is the MINIMUM OF TWO DRAWS and is biased in the favourable")
say("direction.  It reaches the gain as well, which is a different delay on the same runs,")
say("and it reaches the L-against-F and L-against-arm readings, each of which is then a")
say("best-of-two against a single comparator.")

# ===========================================================================
# 5  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT
# ===========================================================================
head("5  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT")
FIRED = []


def shows(name, call):
    """Run a mutated input through a check and print the failure it produced."""
    try:
        call()
    except (Fail, SystemExit, KeyError, ValueError) as exc:
        say("  %-26s FIRES: %s" % (name, exc))
        return True
    say("  %-26s DID NOT FIRE -- the check cannot catch its own mutation" % name)
    return False


FIRED.append(shows("port.check_seeds",
                   lambda: PORT.check_seeds([PORT.MAX_SEED + 1], "mutated")))
FIRED.append(shows("roles/range",
                   lambda: check_roles(dict(ROLES, tuning=[81]), WANT, "mutated")))
FIRED.append(shows("roles/measuring",
                   lambda: check_roles(dict(ROLES, selection=[121]), WANT, "mutated")))
FIRED.append(shows("one script digest", lambda: check_one_digest(
    ROWS[:1] + [dict(ROWS[1], script_sha="deadbeef")], "script_sha", "mutated")))
FIRED.append(shows("one model digest", lambda: check_one_digest(
    ROWS[:1] + [dict(ROWS[1], model_sha="deadbeef")], "model_sha", "mutated")))
FIRED.append(shows("weight", lambda: check_weight(
    [dict(ROWS[0], weight_ms_per_poll=100.0)], WEIGHT, "mutated")))
FIRED.append(shows("score/one lat moved", lambda: check_score(
    [dict(ROWS[0], lat_ms=ROWS[0]["lat_ms"] + 1.0)], WEIGHT, "mutated")))
FIRED.append(shows("score/one poll moved", lambda: check_score(
    [dict(ROWS[0], polls=ROWS[0]["polls"] + 1)], WEIGHT, "mutated")))
FIRED.append(shows("complete/one row missing",
                   lambda: check_complete(ROWS[1:], ORDER, KEYS, "mutated")))
FIRED.append(shows("complete/one decider twice", lambda: check_complete(
    ROWS + [dict(ROWS[0])], ORDER, KEYS, "mutated")))
# The csv with a whole stream gone from every decider, which `check_complete` cannot see
# because it is handed the csv's own keys.  This is the input that passed every check of
# this script before the grid was asserted against the mode's declared product.
FIRED.append(shows("grid/a whole stream gone", lambda: check_grid(
    [(s, r) for s, r in KEYS if r != GRID_STREAMS[0]], GRID_SEEDS, GRID_STREAMS,
    "mutated")))
FIRED.append(shows("grid/a whole seed gone", lambda: check_grid(
    [(s, r) for s, r in KEYS if s != GRID_SEEDS[0]], GRID_SEEDS, GRID_STREAMS,
    "mutated")))
_MROW = next(r for r in ROWS if r["decider"] in MODEL_NAMES)
FIRED.append(shows("K/two worker counts", lambda: check_k(
    ROWS + [dict(_MROW, k=_MROW["k"] + 1)], MODEL_NAMES, "mutated")))
FIRED.append(shows("K/a no-pass decider at K > 1", lambda: check_k(
    [dict(r, k=2) if r["decider"] in ARM_NAMES else r for r in ROWS],
    MODEL_NAMES, "mutated")))
# per_seed's own refusal, which had no mutation anywhere and was named by neither this
# script's "no mutation" list nor its counted ones.
FIRED.append(shows("per_seed/seeds of unequal length",
                   lambda: per_seed([r for r in ROWS
                                     if r["decider"] == ORDER[0]][1:], SEEDS)))
_R = M[ARM_STAGES[0]][COL][0]
_N4, _C4, _LAT, _POLLS = (float(x) for x in _R)
FIRED.append(shows("row order/[lat, n, c, polls]", lambda: check_row_order(
    [_LAT, _N4, _C4, _POLLS], _N4, _C4, _LAT, _POLLS, "mutated")))
FIRED.append(shows("row order/n4 and c4 swapped", lambda: check_row_order(
    [_C4, _N4, _LAT, _POLLS], _N4, _C4, _LAT, _POLLS, "mutated")))
FIRED.append(shows("boot/shape", lambda: check_boot(IDX, len(SEEDS) + 1, "mutated")))
FIRED.append(shows("boot/index out of range", lambda: check_boot(
    IDX + len(SEEDS), len(SEEDS), "mutated")))


def _one_stage_scaled():
    """Scale ONE stage's rows and require the gain to move.

    What makes the invariance argument above a statement about a COMMON factor: the gain is
    the same number when every stage is scaled together, and it is not when one is.
    """
    bad = {k: {COL: v[COL] * (2.0 if k == ARM_STAGES[0] else 1.0)} for k, v in M.items()}
    here = PR20.gain(M, "F", COL)
    there = PR20.gain(bad, "F", COL)
    if here is None or there is None:
        raise Fail("one stage scaled: no gain on one side, so the pair says nothing")
    if abs(here - there) <= 1e-9:
        raise Fail("one stage scaled: the gain did not move (%.6f against %.6f)"
                   % (here, there))
    raise Fail("one stage scaled: the gain moved from %.6f to %.6f, which is what makes "
               "the invariance a statement about a COMMON factor" % (here, there))


FIRED.append(shows("gain/one stage scaled", _one_stage_scaled))


def _not_monotone():
    """A frontier whose three arm points do not fall: `gain` yields None, by its own rule."""
    bad = dict(M)
    bad[ARM_STAGES[0]] = {COL: M[ARM_STAGES[2]][COL].copy()}
    if PR20.gain(bad, "F", COL) is not None:
        raise Fail("not monotone: gain returned a number on a frontier that does not fall")
    raise Fail("not monotone: gain returned None, so the line reads \"no gain: the three "
               "arm points are not monotone\"")


def _polls_outside():
    """A decider whose polls leave the arms' range: `gain` yields None, by its own rule."""
    bad = dict(M)
    bad["F"] = {COL: M["F"][COL].copy()}
    bad["F"][COL][:, 3] = bad["F"][COL][:, 3] * 10.0
    if PR20.gain(bad, "F", COL) is not None:
        raise Fail("polls outside: gain returned a number for polls outside the arms")
    raise Fail("polls outside: gain returned None, so the line reads \"no gain: polls X "
               "outside [MIN, MAX]\"")


FIRED.append(shows("gain/not monotone", _not_monotone))
FIRED.append(shows("gain/polls outside the arms", _polls_outside))
say()
say("The two gain branches above are `predict.gain`'s own refusals and not checks of this")
say("script: they are shown because the report has to print a sentence where a number")
say("would go, and a reader needs to know that sentence was reached by a real refusal.")
say()
say("WHAT HAS NO MUTATION, named and not counted: the closeness RULE itself, because a")
say("rule has no failing input.  Its two branches are reached by the data, and which one")
say("was reached is printed in part 4 with both numbers beside it.")
say("Nothing else of this script is in that list: per_seed's own refusal was in it by")
say("omission until 2026-11-24 and now carries the mutation above.")
say()
if all(FIRED):
    say("all %d mutations produced a failure" % len(FIRED))
else:
    say("FAILURE OF THIS SECTION: %d of %d mutations did not produce a failure"
        % (len(FIRED) - sum(FIRED), len(FIRED)))

# ===========================================================================
# 6  WHAT THIS DOES NOT SHOW
# ===========================================================================
head("6  WHAT THIS DOES NOT SHOW")
if DRY:
    say("- NOTHING about which candidate is better.  The csv is of tuning seeds that both")
    say("  candidates were fitted on; the verdict above is the arithmetic's and not the")
    say("  step's, and the bootstrap over two seeds is degenerate.")
say("- Nothing about L on the testbed.  evidence/13's per-decision agreement is on record")
say("  as failing, and Mode B's band is a per-stage band and not a bound on one decision.")
say("- Nothing that corrects the selection bias of choosing one of two candidates on the")
say("  same seeds.  The only clean correction is a set of seeds nobody has looked at.")
say("- Nothing about the difference between a decider's score and an arm's being the")
say("  decider's difference alone.  F and L draw one extra tick lag whenever they change")
say("  their mind between a send and a resolution and the three fixed arms never do, so")
say("  the comparator family consumes the pair streams differently by construction.")
say("- Nothing about the weight being right.  Its jackknife standard error is 1.685586 ms")
say("  per poll and the band above is built from exactly that uncertainty; the selecting")
say("  statistic moves with it.")
say("- Nothing about the agreement with the labels, which agree.py reports and which")
say("  predicts no score in either direction.")
say("- No pass or fail on any decider.  The verdict in part 4 is a choice between two")
say("  candidates and not a judgement of L against F or against the arms.")
say()
say("end of ana.py")
if not all(FIRED):
    raise SystemExit(1)
