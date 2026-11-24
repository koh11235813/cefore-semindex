#!/usr/bin/env python3
"""The two tuning runs' four adapters, scored on the same seeds and streams, side by side.

    PYTHONDONTWRITEBYTECODE=1 <python> -B spread.py [<eval.csv> [<eval_run1.csv>]]
                                                                    > spread.txt

THIS IS NOT IN THE FROZEN PLAN AND IT SELECTS NOTHING.  `spec.md` names this
measurement in one sentence, amendment A7's last: scoring the first run's two
adapters on the same seeds and streams is "not part of this plan and is run only if the
user asks for it".  The user asked (the ticket's decision 15), the run was made, and
this script is its reading.  It was written AFTER every score of both runs was seen.

WHAT IT CANNOT MOVE.  The candidates are the SECOND run's two adapters, by amendment
A3, which was written and committed before either run was scored; A3's own words are
"the second run's adapters are the candidates whatever its numbers are".  The first
run's b scores lower than the second run's b, and that is a number to report and not a
choice to make: choosing between the two runs' adapters on the seeds they were both
scored on is the one thing A3 fixed against.  `ana_run1.txt` prints "THE CANDIDATE THIS
ANALYSIS CARRIES FORWARD: b" because `ana.py` prints that line for any csv it is handed;
on the first run's csv it carries nothing forward.

WHY ITS INTERVALS SAY LESS THAN THEY LOOK LIKE SAYING.  n = 2 tuning runs is two points
and not a distribution: no spread below has a degree of freedom to spare, and the two
run-to-run rows are a difference of two draws and not an estimate of the recipe's
variance.  Every interval is a bootstrap over the 20 SEEDS with the two adapters HELD
FIXED, so it says how far the seeds let a difference between two particular weight files
move, and it says nothing about a third run of the recipe.

It reads TWO csvs and nothing else: no card, no model, no run store, no label file, no
ROOT.  It imports `../20/predict.py` for the resample matrix and `../22/port.py` for the
seed guard, so that neither is written a second time in this directory.  Neither `ana.py`
nor `versus.py` can be imported -- each does its whole analysis at import time, printing
as it goes -- so the csv reader, the per-seed mean, the paired bootstrap and the seed
check are restated here in `ana.py`'s own arithmetic (`statistics.fmean` per decider per
seed), which is also what makes the two a-against-b assertions below assertions about one
quantity rather than about two ways of summing it.  It prints no clock time and no path,
so two runs of it are byte-identical and the two hosts' output can be compared with cmp,
exactly as `ana.py` and `versus.py` are.
"""
import os
import sys

# The thread caps before numpy is imported, as `ana.py` and `versus.py` set them and for
# the same reason: a BLAS library opens one thread per core at load time on the 96-core
# shared host, and nothing here is a float reduction a thread count could reorder.
for _name, _value in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
                      ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_name, _value)

import importlib.util  # noqa: E402
import statistics      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
EV20 = os.path.normpath(os.path.join(HERE, "..", "20"))
sys.path.insert(0, EV22)
import port as PORT  # noqa: E402  the >100 seed guard, on this script's path as well

import numpy as np  # noqa: E402

# `../20/predict.py` by path and under its own module name, the way `ana.py` and
# `versus.py` load it: two files in this tree are called `predict.py`, and importing
# either by name would make which one is loaded depend on the order of sys.path.
_spec = importlib.util.spec_from_file_location("predict20",
                                               os.path.join(EV20, "predict.py"))
PR20 = importlib.util.module_from_spec(_spec)
sys.modules["predict20"] = PR20
_spec.loader.exec_module(PR20)

BOOT_N = PR20.BOOT_N                 # 2000 resamples
BOOT_SEED = PR20.BOOT_SEED           # 20261122

# The two deciders whose rows differ between the two csvs, and the two comparators every
# adapter is held against.  The comparators' rows are the REAL RUN's own bytes in both
# files, which part 1 asserts, so each adapter is compared against one row set and not
# against a per-file copy of it.
ADAPTED = ("a", "b")
COMPARATORS = ("F", "const6.000")
# The labels of this script's tables.  "run1" is the first tuning run's adapter and
# "run2" the second run's, i.e. the candidate.
ADAPTERS = ("a run1", "a run2", "b run1", "b run2")

# The twelve pairs, in the order they are printed.  The first name of a pair is the one
# the printed difference is "of", so a negative difference means the first scores lower.
# Rows 1 and 2 are the run-to-run rows, the only two that are the spread of the recipe;
# rows 3 and 4 are the within-run a-against-b differences `ana.py` reports for each csv.
PAIRS = (("a run1", "a run2"), ("b run1", "b run2"),
         ("a run1", "b run1"), ("a run2", "b run2"),
         ("a run1", "F"), ("a run2", "F"), ("b run1", "F"), ("b run2", "F"),
         ("a run1", "const6.000"), ("a run2", "const6.000"),
         ("b run1", "const6.000"), ("b run2", "const6.000"))

# The a-against-b lines of the two analyses (each one's part 4, the paired bootstrap
# line), read off those two archives and written here as literals so that this script
# still opens two csvs and nothing else.  The recomputations below have to reproduce all
# six digits at %.3f or the run stops: these are the digits the frozen plan's own
# statistic was reported at on each csv, and a second route to them that disagreed would
# mean one of the two routes is not the quantity it names.
ANA_AB = {"second": (617.840, -58.424, 1357.299),   # ana.txt, the candidates
          "first": (803.781, 414.651, 1257.029)}    # ana_run1.txt, the first run

# The seeds the measuring half holds, which nothing on the learning side may read.  Same
# literal range `decider.check_roles`, `ana.py` and `versus.py` carry, restated for the
# same reason they restate it: this script imports neither `decider.py` nor `prefill.py`,
# because both resolve ROOT at import time and this script needs no ROOT.
MEASURING = range(121, 161)

INT_COLS = ("seed", "stream", "n4", "c4", "polls", "ord_poll", "ord_resched", "k",
            "reads", "sends", "ties", "n_short", "n_mid", "n_long")
FLOAT_COLS = ("lat_ms", "censored_ms", "delay_ms", "score_ms", "score_dropped_ms",
              "weight_ms_per_poll")


class Fail(Exception):
    """A check that did not hold.

    A type of its own, because part 5 feeds every check below a mutated input and requires
    it to raise: catching `Exception` there would also catch a typo in the mutation and
    print it as if the check had worked.
    """


def say(text=""):
    print(text, flush=True)


def head(title):
    """A part heading, in this directory's own shape."""
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


# ---------------------------------------------------------------------------
# the checks, every one of them a function of plain values
# ---------------------------------------------------------------------------
def check_seed_roles(seeds, where):
    """The csv carries no measuring seed and nothing above 100.

    The measuring range is tested FIRST and `port.check_seeds` second, the order
    `decider.check_roles` uses and for its reason: 121 to 160 are above 100, so in the
    other order the replay's guard would refuse them first and the measuring branch could
    not fire on any input the guard admits.  A branch that cannot fail is not evidence.
    """
    got = sorted({int(s) for s in seeds})
    meets = sorted(set(got) & set(MEASURING))
    if meets:
        raise Fail("%s: the seed set meets the measuring seeds %d-%d: %s"
                   % (where, MEASURING[0], MEASURING[-1], meets))
    PORT.check_seeds(got, where)
    return got


def check_same_grid(first, second, where):
    """The two csvs carry the same deciders, the same seeds and the same streams.

    Every difference below is taken seed by seed between a row of one file and a row of
    the other, so a file that carried another seed set would still produce two means and
    the difference would be between two populations.  The two csvs are two runs of one
    script over one grid and this is the statement that they are.
    """
    for what, one, two in (("deciders", first[0], second[0]),
                           ("seeds", first[1], second[1]),
                           ("streams", first[2], second[2])):
        if sorted(one) != sorted(two):
            raise Fail("%s: the two csvs carry different %s: %s against %s"
                       % (where, what, sorted(one)[:4], sorted(two)[:4]))
    return len(first[0]), len(first[1]), len(first[2])


def check_grid(keys, deciders, seeds, streams, where):
    """Exactly one row per (decider, seed, stream), none twice and none missing.

    Every number below is a mean over a decider's runs or a difference of two such means,
    so a lost run would move a mean and a duplicated one would weight a seed twice, and
    neither shows up in the printed value.  The grid is the product this script DECLARES
    from the csv's own seed and stream columns; `ana.py` holds the stricter form of this
    check, which declares the 20 x 40 grid from the mode instead.
    """
    want = {(d, s, t) for d in deciders for s in seeds for t in streams}
    missing = sorted(want - set(keys))
    extra = sorted(set(keys) - want)
    if len(keys) != len(set(keys)):
        dup = sorted({k for k in keys if list(keys).count(k) > 1})
        raise Fail("%s: %d row(s) appear twice, e.g. %s" % (where, len(dup), dup[:3]))
    if missing or extra:
        raise Fail("%s: %d missing (e.g. %s) and %d unexpected (e.g. %s) of %d"
                   % (where, len(missing), missing[:3], len(extra), extra[:3], len(want)))
    return len(want)


def check_runs_per_seed(rows, deciders, seeds, where):
    """Every seed carries the same number of runs for every decider.

    The paired bootstrap's unit is the seed and its per-seed value is the MEAN of that
    seed's runs, so a seed carrying fewer runs would still produce a mean and would enter
    a resample with the same weight as a full one.
    """
    counts = {}
    for r in rows:
        counts.setdefault((r["decider"], r["seed"]), 0)
        counts[(r["decider"], r["seed"])] += 1
    sizes = sorted(set(counts.values()))
    if len(sizes) != 1:
        bad = sorted(k for k, v in counts.items() if v != sizes[-1])
        raise Fail("%s: the (decider, seed) groups carry %s runs, not one size; e.g. %s"
                   % (where, sizes, bad[:3]))
    if len(counts) != len(deciders) * len(seeds):
        raise Fail("%s: %d (decider, seed) group(s), not %d"
                   % (where, len(counts), len(deciders) * len(seeds)))
    return sizes[0]


def check_weight(rows, where):
    """Every row of BOTH csvs was scored at one weight.

    A difference of two mean scores is a difference of two `delay + weight x polls`
    values, so two rows at two weights would make the difference carry a weight change as
    well as an adapter change -- and here the two rows can come from two files, which is
    why the pooled rows are handed in rather than each file's own.  `ana.py` additionally
    holds this value against `../22/weight.json`; this script reads no file but the two
    csvs, so it checks the agreement of the rows with each other and says so rather than
    claiming the stronger check.
    """
    got = sorted({r["weight_ms_per_poll"] for r in rows})
    if len(got) != 1:
        raise Fail("%s: %d different weights in the csvs: %s" % (where, len(got), got[:3]))
    return got[0]


def check_shared_bytes(first, second, where):
    """The two csvs agree BYTE FOR BYTE on every line that is not an adapter's.

    The five deciders that carry no adapter were not re-run for the first run's adapters:
    the resume path of `eval.py` skipped them, so their 4000 rows and the header are the
    real run's own bytes in both files.  That is what lets F and the 6.0 s arm be ONE row
    set for all four adapters below instead of two.  Compared as raw text lines, header
    included, so that a re-serialisation that changed a float's last digit without
    changing its value would be caught as well.
    """
    if len(first) != len(second):
        raise Fail("%s: %d shared line(s) in the one csv and %d in the other"
                   % (where, len(first), len(second)))
    bad = [i for i, (x, y) in enumerate(zip(first, second)) if x != y]
    if bad:
        raise Fail("%s: %d of %d shared line(s) differ, the first at index %d: %r against"
                   " %r" % (where, len(bad), len(first), bad[0],
                            first[bad[0]][:48], second[bad[0]][:48]))
    return len(first)


def check_one_script_digest(rows, where):
    """Every row of BOTH csvs names one script digest.

    The two runs are the same bytes of code pointed at two adapter directories, which is
    the whole claim this comparison rests on: a difference between the two files that
    carried a code change as well as a weight change would not be the recipe's spread.
    """
    got = sorted({r["script_sha"] for r in rows})
    if len(got) != 1:
        raise Fail("%s: %d different script_sha values: %s" % (where, len(got), got))
    return got[0]


def check_model_digests(first, second, names, where):
    """Each adapted decider names ONE model digest per csv, and a DIFFERENT one in each.

    The two files are two adapters of one name, and nothing else in either file says so:
    if a digest matched across the files, the two "runs" would be one adapter read twice
    and every difference below would be a difference of nothing.
    """
    out = {}
    for name in names:
        one = sorted({r["model_sha"] for r in first if r["decider"] == name})
        two = sorted({r["model_sha"] for r in second if r["decider"] == name})
        if len(one) != 1 or len(two) != 1:
            raise Fail("%s: %s carries %s in the one csv and %s in the other, not one"
                       " digest each" % (where, name, one, two))
        if one[0] == two[0]:
            raise Fail("%s: %s carries the same model digest %s in both csvs, so the two"
                       " files hold one adapter and not two" % (where, name, one[0]))
        out[name] = (one[0], two[0])
    return out


def check_answers_sum(rows, name, where):
    """A model decider's three answer counts sum to its forward passes.

    The answer shares below are shares of that sum, and the tie share is a share of
    `reads`.  If the two totals ever parted, both would still print as plausible
    fractions, and the one that is the number of decisions would no longer be named.
    """
    words = sum(r["n_short"] + r["n_mid"] + r["n_long"] for r in rows)
    reads = sum(r["reads"] for r in rows)
    if words != reads:
        raise Fail("%s: %s answered %d times against %d forward pass(es)"
                   % (where, name, words, reads))
    return reads


def check_boot(idx, n, where):
    """The resample matrix is `(2000, n)` and every index is a position of this run.

    `boot_index(n)` changes both the draw range and the shape with `n`, so at 20 seeds no
    element of it is an element of `../20`'s 40-seed matrix: what is shared with that step
    is the generator and the seed 20261122, and the matrix is this step's own.  It is the
    same matrix `ana.py` draws, by the same call, on EACH of the two csvs, which is what
    makes both a-against-b intervals below comparable with their archive digit for digit.
    """
    if tuple(idx.shape) != (BOOT_N, n):
        raise Fail("%s: the resample matrix is %s, not (%d, %d)"
                   % (where, tuple(idx.shape), BOOT_N, n))
    if int(idx.min()) < 0 or int(idx.max()) >= n:
        raise Fail("%s: an index is outside 0 to %d: min %d max %d"
                   % (where, n - 1, int(idx.min()), int(idx.max())))
    return idx.shape


def check_ana_line(got, want, which, where):
    """An a-against-b difference reproduces its own analysis's three digits.

    `ana.py` is the script the frozen plan names and this one is post hoc, so the only
    honest way to print ten more paired differences beside the two it already carries is
    to show that the two shared ones are the same numbers.  Compared at the three decimals
    the archives print, because that is the precision they carry.
    """
    for label, g, w in zip(("point", "lower", "upper"), got, want):
        if "%.3f" % g != "%.3f" % w:
            raise Fail("%s: the %s of score(a) - score(b) on the %s run's csv is %.3f,"
                       " and its analysis prints %.3f" % (where, label, which, g, w))
    return tuple("%.3f" % g for g in got)


# ---------------------------------------------------------------------------
# reading the csvs and the two statistics this script prints
# ---------------------------------------------------------------------------
def read_csv(path):
    """(the column names, every row with its numeric columns typed, the raw text lines).

    `ana.py`'s own reader, restated because `ana.py` does its work at import, with the raw
    lines kept beside the parsed rows because the shared-bytes check above is a statement
    about text and not about values.  An empty cell stays None rather than becoming a
    zero: the four deciders that call no model leave `reads`, `sends`, `ties` and the three
    answer counts empty, and a zero would say "this decider made no tie" where the truth
    is that the question does not apply to a constant arm.
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
    return cols, rows, lines


def shared_lines(lines, names):
    """The raw lines of a csv that no adapter wrote, the header among them.

    Selected by the decider name at the start of the line, which is the csv's first
    column, and the same selection the main session's control made with
    `grep -v "^a,\\|^b,"` before this script existed.
    """
    heads = tuple("%s," % n for n in names)
    return [ln for ln in lines if not ln.startswith(heads)]


def per_seed_mean(rows, seeds):
    """{seed: the mean of that seed's run scores}, by `ana.py`'s own arithmetic.

    The aggregate `spec.md` names for the paired bootstrap, and the only one under which
    resampling 20 seed indices reproduces the 800-run mean: the mean of a seed's 40 run
    scores, then the mean over the resampled seeds.
    """
    return {s: statistics.fmean(r["score_ms"] for r in rows if r["seed"] == s)
            for s in seeds}


def paired(first, second, per_seed, seeds, idx):
    """(difference of means, lower, upper, seeds where the first is lower).

    The percentile interval is taken here and `predict.ci` is NOT reused: that function
    multiplies by 100 because `../20` reports gains in per cent, and these differences are
    milliseconds.  The resample matrix is shared with `ana.py` through `boot_index`, so
    each a-against-b row of this table and the matching `ana.py` line are one computation
    made twice.
    """
    diff = np.array([per_seed[first][s] - per_seed[second][s] for s in seeds], float)
    boot = np.array([diff[row].mean() for row in idx], float)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return float(diff.mean()), lo, hi, int((diff < 0.0).sum())


# ===========================================================================
# 0  WHAT THIS IS
# ===========================================================================
ARGV = [a for a in sys.argv[1:] if not a.startswith("--")]
CSV2 = ARGV[0] if ARGV else os.path.join(HERE, "eval.csv")
CSV1 = ARGV[1] if len(ARGV) > 1 else os.path.join(HERE, "eval_run1.csv")

head("0  WHAT THIS IS, AND WHAT IT IS NOT")
say("THIS IS NOT IN THE FROZEN PLAN.  spec.md carries this measurement in one sentence,")
say("amendment A7's last: scoring the first tuning run's two adapters on the same seeds")
say("and streams is \"not part of this plan and is run only if the user asks for it\".")
say("The user asked, the run was made, and this is its reading.")
say("IT WAS WRITTEN AFTER EVERY SCORE OF BOTH RUNS WAS SEEN.")
say("IT SELECTS NOTHING.  The candidates are the SECOND run's two adapters, by amendment")
say("A3, which was written and committed before either run was scored: \"the second run's")
say("adapters are the candidates whatever its numbers are\".  The first run's b scores")
say("lower than the second run's b below; that is a number to report and not a choice to")
say("make, because choosing between the two runs' adapters on the seeds they were both")
say("scored on is the one thing A3 fixed against.  ana_run1.txt's line \"THE CANDIDATE")
say("THIS ANALYSIS CARRIES FORWARD: b\" is a line ana.py prints for any csv it is handed;")
say("on the first run's csv it carries nothing forward.")
say("n = 2 TUNING RUNS IS TWO POINTS AND NOT A DISTRIBUTION.  The two run-to-run rows")
say("below are a difference of two draws of the recipe, not an estimate of its spread.")
say("EVERY INTERVAL IS OVER THE 20 SEEDS WITH THE TWO ADAPTERS HELD FIXED.  So it says")
say("how far the seeds let a difference between two particular weight files move, and it")
say("says nothing about a third run of the recipe.")
say()
say("It reads two csvs and nothing else: no card, no model, no run store, no label file,")
say("no ROOT.  It prints no clock time and no path, so two runs of it are byte-identical.")

# ===========================================================================
# 1  THE TWO CSVS
# ===========================================================================
head("1  THE TWO CSVS, AND WHAT THEY SHARE")
COLS2, ROWS2, LINES2 = read_csv(CSV2)
COLS1, ROWS1, LINES1 = read_csv(CSV1)
POOLED = ROWS2 + ROWS1
SEEDS = sorted({r["seed"] for r in ROWS2})
STREAMS = sorted({r["stream"] for r in ROWS2})
DECIDERS = sorted({r["decider"] for r in ROWS2})
say("the second tuning run's csv, whose a and b are the candidates:")
say("  rows          %d, %d decider(s)" % (len(ROWS2), len(DECIDERS)))
say("the first tuning run's csv, whose a and b are the adapters A3 set aside:")
say("  rows          %d, %d decider(s)" % (len(ROWS1), len({r["decider"] for r in ROWS1})))
check_same_grid((DECIDERS, SEEDS, STREAMS),
                (sorted({r["decider"] for r in ROWS1}),
                 sorted({r["seed"] for r in ROWS1}),
                 sorted({r["stream"] for r in ROWS1})),
                "the two csvs' grids")
say("seed column     min %d max %d, %d distinct, the same set in both"
    % (SEEDS[0], SEEDS[-1], len(SEEDS)))
say("stream column   min %d max %d, %d distinct, the same set in both"
    % (STREAMS[0], STREAMS[-1], len(STREAMS)))
check_seed_roles(SEEDS, "spread.py: the seeds the csvs carry")
say("port.check_seeds (>100 refused) passed them and the seed set meets the measuring")
say("seeds 121-160 in 0 seeds; both are shown refusing in part 5.  This script starts no")
say("simulation and reads no label row, so it adds nothing to the seeds-never-used proof")
say("beyond what the csvs' own seed columns carry.")
for _WHICH, _KEYS, _ROWS in (("second", [(r["decider"], r["seed"], r["stream"])
                                         for r in ROWS2], ROWS2),
                             ("first", [(r["decider"], r["seed"], r["stream"])
                                        for r in ROWS1], ROWS1)):
    check_grid(_KEYS, DECIDERS, SEEDS, STREAMS, "the %s run's grid" % _WHICH)
    check_runs_per_seed(_ROWS, DECIDERS, SEEDS, "the %s run's runs per (decider, seed)"
                        % _WHICH)
say("OK: in each csv, exactly one row per (decider, seed, stream), %d in all, and every"
    % (len(DECIDERS) * len(SEEDS) * len(STREAMS)))
say("    (decider, seed) group carries the same %d runs" % len(STREAMS))
WEIGHT = check_weight(POOLED, "the weight column of the two csvs together")
say("OK: every row of BOTH csvs was scored at one weight, %r ms per poll" % WEIGHT)
SCRIPT = check_one_script_digest(POOLED, "the script digest of the two csvs together")
say("OK: every row of BOTH csvs names one script digest,")
say("    %s," % SCRIPT)
say("    so the two runs are the same bytes of code pointed at two adapter directories")
SHARED2 = shared_lines(LINES2, ADAPTED)
SHARED1 = shared_lines(LINES1, ADAPTED)
NSHARED = check_shared_bytes(SHARED2, SHARED1, "the lines no adapter wrote")
say("OK: the %d line(s) no adapter wrote -- the header and the %d rows of the five"
    % (NSHARED, NSHARED - 1))
say("    deciders that carry no adapter -- are BYTE FOR BYTE the same in the two csvs.")
say("    eval.py's resume path skipped those five, so they are the real run's own bytes")
say("    in both files and only the %d rows of a and b are new.  Shown failing in part 5"
    % (len(ROWS2) - (NSHARED - 1)))
say("    on one mutated character.")
DIGESTS = check_model_digests(ROWS2, ROWS1, ADAPTED, "the model digests")
say("OK: a and b name a DIFFERENT model digest in the two csvs, one value per decider")
say("    per csv:")
for NAME in ADAPTED:
    say("  %-3s second run  %s" % (NAME, DIGESTS[NAME][0]))
    say("  %-3s first run   %s" % (NAME, DIGESTS[NAME][1]))

# The labelled row sets.  The two comparators are taken from the second run's csv alone,
# which the shared-bytes check above has just shown to be the first run's bytes as well.
BY = {}
for NAME in ADAPTED:
    BY["%s run2" % NAME] = [r for r in ROWS2 if r["decider"] == NAME]
    BY["%s run1" % NAME] = [r for r in ROWS1 if r["decider"] == NAME]
for NAME in COMPARATORS:
    BY[NAME] = [r for r in ROWS2 if r["decider"] == NAME]
MEAN = {n: statistics.fmean(r["score_ms"] for r in BY[n]) for n in BY}
PER_SEED = {n: per_seed_mean(BY[n], SEEDS) for n in BY}

# ===========================================================================
# 2  THE FOUR ADAPTERS
# ===========================================================================
head("2  WHAT THE FOUR ADAPTERS DID, RUN BY RUN")
say("The mean score is `delay + weight x polls` over a decider's %d runs, with every"
    % len(BY["F"]))
say("censored record entering the delay at its lower bound, and LOWER IS BETTER.  The")
say("answer shares are shares of a decider's own forward passes, one per resolution-time")
say("decision, and the tie share is the share of those decisions the tie rule settled")
say("(an exact tie at the maximum going to the first of (' short', ' mid', ' long')).")
say()
say("  adapter    mean score ms   polls/run   mean delay ms    3.695 s    6.0 s   10.043 s"
    "    ties")
for NAME in ADAPTERS:
    READS = check_answers_sum(BY[NAME], NAME, "the answer counts of %s" % NAME)
    SHORT = sum(r["n_short"] for r in BY[NAME])
    MID = sum(r["n_mid"] for r in BY[NAME])
    LONG = sum(r["n_long"] for r in BY[NAME])
    TIES = sum(r["ties"] for r in BY[NAME])
    say("  %-9s %13.3f %11.3f %15.3f     %.4f   %.4f     %.4f  %.4f"
        % (NAME, MEAN[NAME], statistics.fmean(r["polls"] for r in BY[NAME]),
           statistics.fmean(r["delay_ms"] for r in BY[NAME]),
           SHORT / READS, MID / READS, LONG / READS, TIES / READS))
say()
say("  adapter    decisions    short     mid    long    ties settled by the rule")
for NAME in ADAPTERS:
    say("  %-9s %9d %8d %7d %7d %7d"
        % (NAME, sum(r["reads"] for r in BY[NAME]),
           sum(r["n_short"] for r in BY[NAME]), sum(r["n_mid"] for r in BY[NAME]),
           sum(r["n_long"] for r in BY[NAME]), sum(r["ties"] for r in BY[NAME])))
say()
say("the two comparators, from the second run's csv and byte for byte the first run's:")
for NAME in COMPARATORS:
    say("  %-12s %12.3f ms   %8.3f polls/run   %12.3f ms delay"
        % (NAME, MEAN[NAME], statistics.fmean(r["polls"] for r in BY[NAME]),
           statistics.fmean(r["delay_ms"] for r in BY[NAME])))
say("They call no model, so they answer no word and resolve no tie.")

# ===========================================================================
# 3  THE PAIRED DIFFERENCES
# ===========================================================================
head("3  THE PAIRED DIFFERENCES OF MEAN SCORE, SEED BY SEED")
say("A difference is of the FIRST decider minus the second, so a negative value means the")
say("first scores lower.  The interval is the 95 % percentile interval of a paired")
say("bootstrap over the %d SEEDS: %d resamples from one integers(0, %d, size=(%d, %d))"
    % (len(SEEDS), BOOT_N, len(SEEDS), BOOT_N, len(SEEDS)))
say("matrix drawn once with default_rng(%d), i.e. predict.boot_index, the same call"
    % BOOT_SEED)
say("ana.py makes, and the per-seed value is THE MEAN OF THAT SEED'S %d RUN SCORES."
    % len(STREAMS))
say()
IDX = PR20.boot_index(len(SEEDS))
check_boot(IDX, len(SEEDS), "the resample matrix")
say("  first        second        difference ms      95 % interval ms    0 in   first lower")
TABLE = {}
for FIRST, SECOND in PAIRS:
    TABLE[(FIRST, SECOND)] = paired(FIRST, SECOND, PER_SEED, SEEDS, IDX)
    POINT, LO, HI, LOWER_ON = TABLE[(FIRST, SECOND)]
    say("  %-12s %-12s %13.3f  %9.3f to %9.3f   %-4s   %2d of %d"
        % (FIRST, SECOND, POINT, LO, HI, "yes" if LO <= 0.0 <= HI else "no",
           LOWER_ON, len(SEEDS)))
say()
say("THE FIRST TWO ROWS ARE THE SPREAD OF THE RECIPE AT n = 2, and they are the whole")
say("reason this run was made: one adapter name, one set of rows, one set of seeds, two")
say("draws of the same tuning procedure.  Two draws are two points.  The interval beside")
say("each of them is the seeds' own spread with both weight files held fixed, so it is")
say("NOT an interval on what a third run would give.")
say()
for WHICH, KEY, ARCHIVE in (("first", ("a run1", "b run1"), "ana_run1.txt"),
                            ("second", ("a run2", "b run2"), "ana.txt")):
    GOT = check_ana_line(TABLE[KEY][:3], ANA_AB[WHICH], WHICH, "the a-against-b line")
    say("THE a-AGAINST-b ROW OF THE %s RUN IS %s's OWN LINE, ASSERTED:"
        % (WHICH.upper(), ARCHIVE))
    say("  its part 4 prints point %s ms and the 95 %% interval %s to %s ms," % GOT)
    say("  and this run reproduces all three digits.")
say("Those two are the only rows of the twelve the frozen plan carries, one per csv; the")
say("other ten are post hoc.  Both are shown failing in part 5 on a moved score.")
say()
say("WHAT THE TWO WITHIN-RUN ROWS ARE AND ARE NOT: each is the difference its own csv's")
say("ana.py already reports, and ana.py's verdict on each stands as that run's.  The")
say("verdict on the first run's csv chooses nothing, because A3 chose the second run's")
say("adapters before either was scored.")

# ===========================================================================
# 4  THE FRONTIER GAIN
# ===========================================================================
head("4  THE FRONTIER GAIN, WHICH THIS SCRIPT DOES NOT RECOMPUTE")
say("The gains of the four adapters are in ana.txt's part 3 and ana_run1.txt's part 3,")
say("each with its 95 % interval, its half width, its dropped count and its word, and")
say("THEY ARE LEFT THERE.  Recomputing them here would need ana.py's per-seed")
say("[n4, c4, lat, polls] row builder, its arm-name-to-frontier-key mapping, its two")
say("row-order checks, its scale-invariance check and its drop-and-count resample loop")
say("around predict.gain -- not a few lines, and a second route to a number the archives")
say("already print would be a second place for that number to be wrong.  The two archives")
say("are ana.py's own output on the two csvs, so reading the four gains off them side by")
say("side needs no code at all.")

# ===========================================================================
# 5  EVERY CHECK, SHOWN FAILING
# ===========================================================================
head("5  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT")
say("The mutation is of the INPUT, inside this process.  The checks are functions of plain")
say("values for that reason; a per-check copy of the script would be many reads of the")
say("same two csvs, and a copy that silently matched nothing would read as a passing")
say("check.")
say()
FIRED = []


def shows(name, call):
    """Run a check on a mutated input and require it to refuse.

    `Fail` is this script's own checks and `SystemExit` is `port.check_seeds`, which is
    the replay's own guard and raises that.  Narrow on purpose: catching `Exception` here
    would also catch a typo in the mutation and print it as if the check had worked.
    """
    try:
        call()
    except (Fail, SystemExit) as exc:
        say("  %-30s FIRES: %s" % (name, exc))
        return True
    say("  %-30s DID NOT FIRE -- this check cannot fail on this input" % name)
    return False


FIRED.append(shows("port.check_seeds",
                   lambda: check_seed_roles(SEEDS + [PORT.MAX_SEED + 1], "mutated")))
FIRED.append(shows("roles/measuring",
                   lambda: check_seed_roles([MEASURING[0]], "mutated")))
FIRED.append(shows("same grid/a seed gone",
                   lambda: check_same_grid((DECIDERS, SEEDS, STREAMS),
                                           (DECIDERS, SEEDS[1:], STREAMS), "mutated")))
FIRED.append(shows("grid/one row missing",
                   lambda: check_grid([(r["decider"], r["seed"], r["stream"])
                                       for r in ROWS1[1:]],
                                      DECIDERS, SEEDS, STREAMS, "mutated")))
FIRED.append(shows("grid/one row twice",
                   lambda: check_grid([(r["decider"], r["seed"], r["stream"])
                                       for r in ROWS1 + ROWS1[:1]],
                                      DECIDERS, SEEDS, STREAMS, "mutated")))
FIRED.append(shows("runs per seed/sizes",
                   lambda: check_runs_per_seed(ROWS1[1:], DECIDERS, SEEDS, "mutated")))
# The second branch of the same check needs its own mutation: dropping one row leaves two
# group sizes and stops at the branch above, so a WHOLE (decider, seed) group goes instead
# and every surviving group still carries 40 runs.
_LOST = (DECIDERS[0], SEEDS[0])
FIRED.append(shows("runs per seed/groups",
                   lambda: check_runs_per_seed(
                       [r for r in ROWS1 if (r["decider"], r["seed"]) != _LOST],
                       DECIDERS, SEEDS, "mutated")))
FIRED.append(shows("weight/one row moved",
                   lambda: check_weight(POOLED + [dict(ROWS1[0],
                                                       weight_ms_per_poll=100.0)],
                                        "mutated")))
FIRED.append(shows("script digest/one row moved",
                   lambda: check_one_script_digest(
                       POOLED + [dict(ROWS1[0], script_sha="deadbeef")], "mutated")))
# The byte check of the 4000 shared rows, with ONE CHARACTER of one shared line moved.
# The mutation is of the text, which is what the check is about: a value-preserving
# re-serialisation would pass a comparison of parsed floats and fail this one.
_LINE = next(i for i, ln in enumerate(SHARED1) if ln.startswith("const3.695,"))
_BENT = list(SHARED1)
_BENT[_LINE] = _BENT[_LINE].replace("const3.695,", "const3.69X,", 1)
FIRED.append(shows("shared bytes/one character",
                   lambda: check_shared_bytes(SHARED2, _BENT, "mutated")))
FIRED.append(shows("shared bytes/a line gone",
                   lambda: check_shared_bytes(SHARED2, SHARED1[1:], "mutated")))
# The model digests, with the first run's a given the second run's digest: the two files
# would then hold one adapter under two names and every row of this script that carries
# "a run1" would be a difference of nothing.
_SAME = [dict(r, model_sha=DIGESTS["a"][0]) if r["decider"] == "a" else r
         for r in ROWS1]
FIRED.append(shows("model digests/the same adapter",
                   lambda: check_model_digests(ROWS2, _SAME, ADAPTED, "mutated")))
FIRED.append(shows("model digests/two in one csv",
                   lambda: check_model_digests(
                       ROWS2, [dict(ROWS1[0], decider="a", model_sha="deadbeef")] + ROWS1,
                       ADAPTED, "mutated")))
FIRED.append(shows("answers/one word lost",
                   lambda: check_answers_sum(
                       [dict(BY["a run1"][0],
                             n_long=BY["a run1"][0]["n_long"] - 1)] + BY["a run1"][1:],
                       "a run1", "mutated")))
FIRED.append(shows("boot/shape",
                   lambda: check_boot(IDX, len(SEEDS) + 1, "mutated")))
FIRED.append(shows("boot/index out of range",
                   lambda: check_boot(IDX + len(SEEDS), len(SEEDS), "mutated")))
# The two archive lines, each with one run's score moved by one millisecond: the mutation
# is of the csv rows the difference is computed from and not of the six literals, so what
# fires is the agreement of a recomputation with an archive and not a typo in a constant.
for _WHICH, _LABEL in (("first", "a run1"), ("second", "a run2")):
    _MOVED = [dict(r) for r in BY[_LABEL]]
    _MOVED[0]["score_ms"] = _MOVED[0]["score_ms"] + 1.0
    _MUT = dict(PER_SEED)
    _MUT[_LABEL] = per_seed_mean(_MOVED, SEEDS)
    FIRED.append(shows("ana line/%s run, one score moved" % _WHICH,
                       (lambda which=_WHICH, label=_LABEL, mut=_MUT:
                        check_ana_line(
                            paired(label, label.replace("a ", "b "), mut, SEEDS,
                                   IDX)[:3],
                            ANA_AB[which], which, "mutated"))))
say()
say("WHAT HAS NO MUTATION, named and not counted: the choice of the twelve pairs, because")
say("a choice has no failing input.  It is a post hoc choice and part 0 says so rather")
say("than leaving a reader to count the comparisons.  Also the fact that n = 2, which is")
say("a property of how many times the recipe was run and not of any input to this script.")
say()
say("all %d mutations produced a failure" % sum(FIRED))
if sum(FIRED) != len(FIRED):
    raise SystemExit("spread.py: %d of %d mutations did not fire"
                     % (len(FIRED) - sum(FIRED), len(FIRED)))

# ===========================================================================
# 6  WHAT THIS DOES NOT SHOW
# ===========================================================================
head("6  WHAT THIS DOES NOT SHOW")
say("- NO CHOICE OF ANY KIND.  The candidates are the second run's adapters by A3, fixed")
say("  before either run was scored, and the first run's lower b does not reopen that.")
say("  The rule A3 protects is exactly the one a reader of this table would break by")
say("  picking the better of two runs on the seeds both were scored on.")
say("- NOTHING ABOUT A THIRD RUN OF THE RECIPE.  Two runs are two points: there is no")
say("  spread of the recipe here with a degree of freedom to spare, and the intervals")
say("  above hold the two weight files fixed and resample only the seeds.")
say("- Nothing about WHERE the two runs parted.  Neither run asked torch for a")
say("  deterministic kernel, which is the recipe as frozen, and no log of either run")
say("  carries a digest of the untuned adapter, so the seeded lora_A draw is unverified")
say("  as well.  This script reads two csvs of scores and reaches none of that.")
say("- Nothing that corrects the selection bias.  The chosen candidate's figure is still")
say("  the minimum of two draws on these seeds, and the only clean correction is a set of")
say("  seeds nobody has looked at.  Four adapters on the same 20 seeds is more reads of")
say("  those seeds, not fewer.")
say("- No correction for multiplicity, and the pairs were chosen after every mean was on")
say("  the screen.  Twelve differences are printed here beside versus.txt's nine.")
say("- Nothing about a difference against a comparator being the decider's difference")
say("  alone.  F and L draw one extra tick lag whenever they change their mind between a")
say("  send and a resolution and the three fixed arms never do, so the comparator family")
say("  consumes the pair streams differently by construction, in an unknown direction.")
say("- Nothing about the label agreement or the pass time of the first run's adapters.")
say("  agree.py and latency.py were not run on them.")
say("- Nothing about the weight being right.  Its jackknife standard error is 1.685586 ms")
say("  per poll and every score above moves with it.")
say()
say("no check fired on real input; the failures in part 5 are the mutations")
say("end of spread.py")
