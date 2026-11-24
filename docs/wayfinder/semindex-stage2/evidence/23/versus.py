#!/usr/bin/env python3
"""Paired seed differences between the seven deciders' mean scores, read off eval.csv.

    PYTHONDONTWRITEBYTECODE=1 <python> -B versus.py [<eval.csv>]      > versus.txt

THIS IS NOT PART OF THE FROZEN PLAN AND IT SELECTS NOTHING.  `spec.md` fixes two
statistics -- the mean selecting score and the frontier gain -- and one rule, the 297 ms
band, and `ana.py` computes all three.  This script was written AFTER those scores were
seen, to put a number on comparisons the plan reports no interval for: L against the rule
F, L against the fixed 6.0 s arm, and the tuned candidates against the untuned model.
The choice between candidate a and candidate b was already made by `ana.py` on the mean
score alone and nothing here can reopen it; the one line this script shares with `ana.txt`
is the a-against-b difference, which it recomputes and asserts against the archive's own
digits so that a reader can see the two agree.

WHY ITS INTERVALS ARE DESCRIPTIVE AND NOT TESTS.  Nine comparisons are printed and no
correction for multiplicity is applied, the pairs were chosen after the means were on the
screen, and the chosen candidate's own figure is the minimum of two draws (`spec.md`
Threats 1).  So an interval below says how far the 20 seeds let a difference move, and it
does not say that a difference was predicted or that one decider is better.

It reads ONE csv and nothing else: no card, no model, no run store, no label file, no
ROOT.  It imports `../20/predict.py` for the resample matrix and `../22/port.py` for the
seed guard, so that neither is written a second time in this directory.  `ana.py` itself
cannot be imported -- it does its whole analysis at import time, as `rows.py` does -- so
the csv reader and the per-seed mean are restated here in `ana.py`'s own arithmetic
(`statistics.fmean` per decider per seed), which is what makes the a-against-b assertion
below an assertion about one quantity rather than about two ways of summing it.
It prints no clock time and no path, so two runs of it are byte-identical and the two
hosts' output can be compared with cmp, exactly as `ana.py` and `parity.py` are.
"""
import os
import sys

# The thread caps before numpy is imported, as `ana.py` and `parity.py` set them and for
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

# `../20/predict.py` by path and under its own module name, the way `ana.py` loads it: two
# files in this tree are called `predict.py`, and importing either by name would make
# which one is loaded depend on the order of sys.path.
_spec = importlib.util.spec_from_file_location("predict20",
                                               os.path.join(EV20, "predict.py"))
PR20 = importlib.util.module_from_spec(_spec)
sys.modules["predict20"] = PR20
_spec.loader.exec_module(PR20)

BOOT_N = PR20.BOOT_N                 # 2000 resamples
BOOT_SEED = PR20.BOOT_SEED           # 20261122

ARM_NAMES = ("const3.695", "const6.000", "const10.043")
MODEL_NAMES = ("untuned", "a", "b")
ORDER = ARM_NAMES + ("F",) + MODEL_NAMES

# The nine pairs, written out in the order they are printed.  Chosen after the means were
# seen and therefore not a pre-registered family; the first name of a pair is the one the
# printed difference is "of", so a negative difference means the first scores lower.
PAIRS = (("a", "b"),
         ("b", "F"), ("a", "F"),
         ("b", "const6.000"), ("a", "const6.000"), ("F", "const6.000"),
         ("b", "untuned"), ("a", "untuned"),
         ("untuned", "const3.695"))

# The a-against-b line of `ana.txt` (its part 4, the paired bootstrap line), read off that
# archive and written here as three literals so that this script still opens one csv and
# nothing else.  The recomputation below has to reproduce all three at %.3f or the run
# stops: these are the digits the frozen plan's own statistic was reported at, and a
# second route to them that disagreed would mean one of the two routes is not the quantity
# it names.
ANA_AB = (617.840, -58.424, 1357.299)

# The seeds the measuring half holds, which nothing on the learning side may read.  Same
# literal range `decider.check_roles` and `ana.py` carry, restated for the same reason
# `ana.py` restates it: this script imports neither `decider.py` nor `prefill.py`, because
# both resolve ROOT at import time and this script needs no ROOT.
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


def check_deciders(deciders, where):
    """The csv carries exactly the seven deciders this script names.

    Every mean, every pair and the censored table below are indexed by `ORDER`, so a csv
    that lost a decider would make this script read a name that is not in it, and a csv
    that carried an eighth would leave it out of every table without saying so.
    """
    if sorted(deciders) != sorted(ORDER):
        raise Fail("%s: the csv carries %s, not the seven deciders"
                   % (where, sorted(deciders)))
    return list(ORDER)


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


def check_runs_per_seed(rows, seeds, where):
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
    if len(counts) != len(ORDER) * len(seeds):
        raise Fail("%s: %d (decider, seed) group(s), not %d"
                   % (where, len(counts), len(ORDER) * len(seeds)))
    return sizes[0]


def check_weight(rows, where):
    """Every row was scored at one weight.

    A difference of two mean scores is a difference of two `delay + weight x polls`
    values, so two rows at two weights would make the difference carry a weight change as
    well as a decider change.  `ana.py` additionally holds this value against
    `../22/weight.json`; this script reads no file but the csv, so it checks the agreement
    of the rows with each other and says so rather than claiming the stronger check.
    """
    got = sorted({r["weight_ms_per_poll"] for r in rows})
    if len(got) != 1:
        raise Fail("%s: %d different weights in the csv: %s" % (where, len(got), got[:3]))
    return got[0]


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
    same matrix `ana.py` draws, by the same call, which is what makes the a-against-b
    interval below comparable with the archive's digit for digit.
    """
    if tuple(idx.shape) != (BOOT_N, n):
        raise Fail("%s: the resample matrix is %s, not (%d, %d)"
                   % (where, tuple(idx.shape), BOOT_N, n))
    if int(idx.min()) < 0 or int(idx.max()) >= n:
        raise Fail("%s: an index is outside 0 to %d: min %d max %d"
                   % (where, n - 1, int(idx.min()), int(idx.max())))
    return idx.shape


def check_ana_line(got, want, where):
    """The a-against-b difference reproduces `ana.txt`'s own three digits.

    `ana.py` is the script the frozen plan names and this one is post hoc, so the only
    honest way to print a second paired difference beside it is to show that the shared
    one is the same number.  Compared at the three decimals the archive prints, because
    that is the precision the archive carries.
    """
    for label, g, w in zip(("point", "lower", "upper"), got, want):
        if "%.3f" % g != "%.3f" % w:
            raise Fail("%s: the %s of score(a) - score(b) is %.3f, and ana.txt prints %.3f"
                       % (where, label, g, w))
    return tuple("%.3f" % g for g in got)


# ---------------------------------------------------------------------------
# reading the csv and the two statistics this script prints
# ---------------------------------------------------------------------------
def read_csv(path):
    """Every row of `eval.py`'s csv, with the numeric columns typed.

    `ana.py`'s own reader, restated because `ana.py` does its work at import.  An empty
    cell stays None rather than becoming a zero: the four deciders that call no model
    leave `reads`, `sends`, `ties` and the three answer counts empty, and a zero would say
    "this decider made no tie" where the truth is that the question does not apply to a
    constant arm.
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
    the a-against-b row of this table and `ana.txt`'s own line are one computation made
    twice.
    """
    diff = np.array([per_seed[first][s] - per_seed[second][s] for s in seeds], float)
    boot = np.array([diff[row].mean() for row in idx], float)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return float(diff.mean()), lo, hi, int((diff < 0.0).sum())


# ===========================================================================
# 0  WHAT THIS IS
# ===========================================================================
ARGV = [a for a in sys.argv[1:] if not a.startswith("--")]
CSV = ARGV[0] if ARGV else os.path.join(HERE, "eval.csv")

head("0  WHAT THIS IS, AND WHAT IT IS NOT")
say("THIS ANALYSIS IS NOT IN THE FROZEN PLAN.  spec.md fixes two statistics and one rule")
say("and ana.py computes all three; this script was written AFTER the scores were seen.")
say("IT SELECTS NOTHING.  The choice between candidate a and candidate b was made by")
say("ana.py on the mean selecting score alone, and no number below can reopen it.")
say("ITS INTERVALS ARE DESCRIPTIVE.  Nine comparisons are printed, no correction for")
say("multiplicity is applied, and the pairs were chosen after the means were on the")
say("screen, so an interval here says how far the 20 seeds let a difference move and NOT")
say("that a difference was predicted or that one decider is better than another.")
say()
say("It reads one csv and nothing else: no card, no model, no run store, no label file,")
say("no ROOT.  It prints no clock time and no path, so two runs of it are byte-identical.")

# ===========================================================================
# 1  THE CSV
# ===========================================================================
head("1  THE CSV, ITS GRID AND ITS SEEDS")
COLS, ROWS = read_csv(CSV)
SEEDS = sorted({r["seed"] for r in ROWS})
STREAMS = sorted({r["stream"] for r in ROWS})
DECIDERS = sorted({r["decider"] for r in ROWS})
KEYS = [(r["decider"], r["seed"], r["stream"]) for r in ROWS]
say("rows            %d" % len(ROWS))
say("deciders        %s" % ", ".join(ORDER))
say("seed column     min %d max %d, %d distinct" % (SEEDS[0], SEEDS[-1], len(SEEDS)))
say("stream column   min %d max %d, %d distinct" % (STREAMS[0], STREAMS[-1], len(STREAMS)))
check_seed_roles(SEEDS, "versus.py: the seeds the csv carries")
say("port.check_seeds (>100 refused) passed them and the seed set meets the measuring")
say("seeds 121-160 in 0 seeds; both are shown refusing in part 5.  This script starts no")
say("simulation and reads no label row, so it adds nothing to the seeds-never-used proof")
say("beyond what the csv's own seed column carries.")
check_deciders(DECIDERS, "versus.py: the deciders the csv carries")
GRID = check_grid(KEYS, ORDER, SEEDS, STREAMS, "the csv's grid")
PER_SEED_RUNS = check_runs_per_seed(ROWS, SEEDS, "the runs per (decider, seed)")
WEIGHT = check_weight(ROWS, "the weight column")
say("OK: exactly one row per (decider, seed, stream), %d in all, and every (decider,"
    % GRID)
say("    seed) group carries the same %d runs" % PER_SEED_RUNS)
say("OK: every row was scored at one weight, %r ms per poll" % WEIGHT)

BY = {n: [r for r in ROWS if r["decider"] == n] for n in ORDER}
MEAN = {n: statistics.fmean(r["score_ms"] for r in BY[n]) for n in ORDER}
PER_SEED = {n: per_seed_mean(BY[n], SEEDS) for n in ORDER}

# ===========================================================================
# 2  THE PAIRED DIFFERENCES
# ===========================================================================
head("2  THE PAIRED DIFFERENCES OF MEAN SCORE, SEED BY SEED")
say("The mean score is `delay + weight x polls` over a decider's %d runs, lower being"
    % len(BY["F"]))
say("better.  A difference is of the FIRST decider minus the second, so a negative value")
say("means the first scores lower.  The interval is the 95 % percentile interval of a")
say("paired bootstrap over the %d SEEDS: %d resamples from one integers(0, %d,"
    % (len(SEEDS), BOOT_N, len(SEEDS)))
say("size=(%d, %d)) matrix drawn once with default_rng(%d), i.e. predict.boot_index,"
    % (BOOT_N, len(SEEDS), BOOT_SEED))
say("the same call ana.py makes, and the per-seed value is THE MEAN OF THAT SEED'S %d"
    % len(STREAMS))
say("RUN SCORES.")
say()
say("the mean scores these differences are taken between, ranked, lower first:")
for NAME in sorted(ORDER, key=lambda n: MEAN[n]):
    say("  %-12s %12.3f ms" % (NAME, MEAN[NAME]))
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
AB = check_ana_line(TABLE[("a", "b")][:3], ANA_AB, "the a-against-b line")
say("THE FIRST ROW IS ana.txt's OWN LINE, RECOMPUTED AND ASSERTED.  ana.txt's part 4")
say("prints point %s ms and the 95 %% interval %s to %s ms, and this run" % AB)
say("reproduces all three digits.")
say("It is the one row of this table the frozen plan carries; the other eight are post")
say("hoc.  Shown failing in part 5 on a moved score.")

# ===========================================================================
# 3  WHAT THE MODEL DECIDERS ANSWERED
# ===========================================================================
head("3  THE ANSWER WORDS, AND HOW FAR THE TIE RULE REACHED")
say("One forward pass per resolution-time decision, the largest of the three candidate")
say("logits winning, and an exact tie going to the first of (' short', ' mid', ' long'),")
say("i.e. to the shortest interval.  The tie share is the share of a decider's own")
say("decisions the tie rule settled, which is the reach of a read-out convention and not")
say("a footnote: spec.md asks for it as a number for exactly that reason.")
say()
say("  decider   decisions    3.695 s    6.0 s   10.043 s    settled by the tie rule")
for NAME in MODEL_NAMES:
    READS = check_answers_sum(BY[NAME], NAME, "the answer counts")
    SHORT = sum(r["n_short"] for r in BY[NAME])
    MID = sum(r["n_mid"] for r in BY[NAME])
    LONG = sum(r["n_long"] for r in BY[NAME])
    TIES = sum(r["ties"] for r in BY[NAME])
    say("  %-8s %10d     %.4f   %.4f     %.4f    %6d  %.4f"
        % (NAME, READS, SHORT / READS, MID / READS, LONG / READS, TIES, TIES / READS))
say()
say("the same counts, not as shares:")
for NAME in MODEL_NAMES:
    say("  %-8s short %7d   mid %7d   long %7d   ties %6d of %d"
        % (NAME, sum(r["n_short"] for r in BY[NAME]), sum(r["n_mid"] for r in BY[NAME]),
           sum(r["n_long"] for r in BY[NAME]), sum(r["ties"] for r in BY[NAME]),
           sum(r["reads"] for r in BY[NAME])))
say()
say("The four deciders that call no model answer no word and resolve no tie: their rows")
say("carry those six columns empty, which is why they are absent from this part.")

# ===========================================================================
# 4  THE CENSORED RECORDS
# ===========================================================================
head("4  THE RUNS THAT CARRY A CENSORED RECORD")
say("A censored record is one whose discovery never completed inside the scoring window.")
say("The selecting score enters it at its LOWER BOUND and the frontier gain DROPS it, so")
say("the two statistics disagree about exactly these runs, and this is how many runs they")
say("disagree about.  Counted per decider over its %d runs." % len(BY["F"]))
say()
say("  decider        runs with a censored record    censored records in all")
for NAME in ORDER:
    HITS = sum(1 for r in BY[NAME] if r["c4"] > 0)
    say("  %-12s %22d %26d" % (NAME, HITS, sum(r["c4"] for r in BY[NAME])))
say()
say("Only the long arm and the two candidates that lean to it carry any, which is why the")
say("score and the gain can order the same two deciders differently.")

# ===========================================================================
# 5  EVERY CHECK, SHOWN FAILING
# ===========================================================================
head("5  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT")
say("The mutation is of the INPUT, inside this process.  The checks are functions of plain")
say("values for that reason; a per-check copy of the script would be seven reads of the")
say("same csv, and a copy that silently matched nothing would read as a passing check.")
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
        say("  %-26s FIRES: %s" % (name, exc))
        return True
    say("  %-26s DID NOT FIRE -- this check cannot fail on this input" % name)
    return False


FIRED.append(shows("port.check_seeds",
                   lambda: check_seed_roles(SEEDS + [PORT.MAX_SEED + 1], "mutated")))
FIRED.append(shows("roles/measuring",
                   lambda: check_seed_roles([MEASURING[0]], "mutated")))
FIRED.append(shows("deciders/one lost",
                   lambda: check_deciders(DECIDERS[1:], "mutated")))
FIRED.append(shows("grid/one row missing",
                   lambda: check_grid(KEYS[1:], ORDER, SEEDS, STREAMS, "mutated")))
FIRED.append(shows("grid/one row twice",
                   lambda: check_grid(KEYS + KEYS[:1], ORDER, SEEDS, STREAMS, "mutated")))
FIRED.append(shows("runs per seed/sizes",
                   lambda: check_runs_per_seed(ROWS[1:], SEEDS, "mutated")))
# The second branch of the same check needs its own mutation: dropping one row leaves two
# group sizes and stops at the branch above, so a WHOLE (decider, seed) group goes instead
# and every surviving group still carries 40 runs.
_LOST = (ORDER[0], SEEDS[0])
FIRED.append(shows("runs per seed/groups",
                   lambda: check_runs_per_seed(
                       [r for r in ROWS if (r["decider"], r["seed"]) != _LOST],
                       SEEDS, "mutated")))
FIRED.append(shows("weight/one row moved",
                   lambda: check_weight(ROWS + [dict(ROWS[0], weight_ms_per_poll=100.0)],
                                        "mutated")))
FIRED.append(shows("answers/one word lost",
                   lambda: check_answers_sum(
                       [dict(BY["a"][0], n_long=BY["a"][0]["n_long"] - 1)] + BY["a"][1:],
                       "a", "mutated")))
FIRED.append(shows("boot/shape",
                   lambda: check_boot(IDX, len(SEEDS) + 1, "mutated")))
FIRED.append(shows("boot/index out of range",
                   lambda: check_boot(IDX + len(SEEDS), len(SEEDS), "mutated")))
# The ana.txt line, with one run's score moved by one millisecond: the mutation is of the
# csv rows the difference is computed from and not of the three literals, so what fires is
# the agreement of a recomputation with the archive and not a typo in a constant.
_MOVED = [dict(r) for r in BY["a"]]
_MOVED[0]["score_ms"] = _MOVED[0]["score_ms"] + 1.0
_MUT_PER_SEED = dict(PER_SEED, a=per_seed_mean(_MOVED, SEEDS))
FIRED.append(shows("ana.txt line/one score moved",
                   lambda: check_ana_line(
                       paired("a", "b", _MUT_PER_SEED, SEEDS, IDX)[:3], ANA_AB,
                       "mutated")))
say()
say("WHAT HAS NO MUTATION, named and not counted: the choice of the nine pairs, because a")
say("choice has no failing input.  It is a post hoc choice and part 0 says so rather than")
say("leaving a reader to count the comparisons.")
say()
say("all %d mutations produced a failure" % sum(FIRED))
if sum(FIRED) != len(FIRED):
    raise SystemExit("versus.py: %d of %d mutations did not fire"
                     % (len(FIRED) - sum(FIRED), len(FIRED)))

# ===========================================================================
# 6  WHAT THIS DOES NOT SHOW
# ===========================================================================
head("6  WHAT THIS DOES NOT SHOW")
say("- No pass or fail on any decider, and no verdict the frozen plan does not carry.")
say("  The plan's own verdict is ana.txt's, on the a-against-b difference alone.")
say("- Nothing that corrects the selection bias.  The chosen candidate's figure is the")
say("  minimum of two draws, so every row above that carries a or b inherits that bias,")
say("  and the only clean correction is a set of seeds nobody has looked at.")
say("- Nothing about a difference against an arm being the decider's difference alone.  F")
say("  and L draw one extra tick lag whenever they change their mind between a send and a")
say("  resolution and the three fixed arms never do, so the comparator family consumes")
say("  the pair streams differently by construction, in an unknown direction.")
say("- Nothing about the weight being right.  Its jackknife standard error is 1.685586 ms")
say("  per poll and every score above moves with it; the band the plan tests against is")
say("  built from exactly that uncertainty.")
say("- Nothing about the tuning being reproducible.  Each candidate is one draw of a")
say("  recipe whose two runs wrote different weights, so a difference between a and b")
say("  carries one draw of tuning noise per candidate.")
say("- Nothing about L on the testbed, and nothing about the agreement with the labels.")
say("  No run of this step touches the emulator, and agree.py's numbers predict no score")
say("  in either direction.")
say()
say("no check fired on real input; the failures in part 5 are the mutations")
say("end of versus.py")
