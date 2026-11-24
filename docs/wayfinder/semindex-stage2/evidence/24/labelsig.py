#!/usr/bin/env python3
"""Do ticket 09's hindsight labels carry learnable signal at all.  THIS IS NOT M.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B labelsig.py

WHAT THIS IS, AND WHAT IT IS NOT.  The decision maker M learns from the s11a EMULATOR
logs and is scored in the replay.  This module touches neither: it fits a small classifier
to ticket 09's LABEL rows and reads its agreement with those labels on seeds 71 to 80.  A
label is the arm that is best IF the formula rule takes over afterwards, measured inside
the replay over 8 passes, so agreement with it is not a score of anything and the number
below is not M's.  It exists to separate one question from another: when the language
model's agreement stopped at its rows' own class prior, was that the text it had to read,
or is there no learnable signal in the labels to find?

THE ROWS.  `../23/rows.py` fixed the two row sets and counted them, and both counts are
reproduced here and asserted against its archived output:
  candidate a   every row of seeds 1 to 80 that carries a label;
  candidate b   the rows of a that also pass `t >= 3 or (sd = 0 and labelled)`, the
                selection `../22/stability.py` measured the agreement of.
`rows.py` itself is NOT imported: it runs its whole analysis at import and would print its
own output, machine paths included, into this one's.  Part 1 counts its top-level calls to
say so with a number.  What IS imported is `../22/label.py`'s `paired_t`, which is the
statistic candidate b is selected by, and `../22/stability.py`'s bin edges, which are
where the bar 3 comes from -- so neither is written a second time here.

THE CLASSIFIER.  Closed-form ridge regression from the feature vector to the three one-hot
targets, numpy only, features standardised on the fit rows, one fixed penalty named below.
The answer is the largest of the three outputs, with a tie going to the first in the order
3.695, 6.0, 10.043 -- the rule `../23/decider.py`'s `pick` uses for the language model's
three words, restated here rather than imported because that module reads a command-line
root at import.  Two more predictors stand beside it: the same ridge on the five scalar
cells alone, which is the information ADR 0004 line 5 permits a policy to read, and the
trivial answer that always names the fit rows' most frequent arm.

WHY AGREEMENT IS NOT A SCORE.  Two independent 8-pass labellings of the same decisions
disagreed on about a quarter of the ones they both labelled (`../22/stability.txt`, seeds
1 to 10, the number is read off that file in part 5).  A classifier that matched these
labels perfectly would match a teacher that disagrees with itself that often.  Nothing
here says which arm is better, in the replay or anywhere else.
"""
import ast
import collections
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
EV23 = os.path.normpath(os.path.join(HERE, "..", "23"))
sys.path.insert(0, HERE)
sys.path.insert(0, EV22)

import charges as CH     # noqa: E402  the five-cell names, one spelling
import features as FE    # noqa: E402  the vector, the label reader and the seed guards
import label as LBL      # noqa: E402  `paired_t`, `ARMS`: the statistic b is selected by
import port as PORT      # noqa: E402  the >100 seed guard
import stability as ST   # noqa: E402  the t bins, which is where the bar 3 comes from

ARMS = tuple(LBL.ARMS)
CELLS = CH.CELLS

# The split ticket 10 tuned under: seeds 1 to 70 fit, 71 to 80 read out, both inside the
# tuning half.  `features.MAX_LABEL_SEED` is the 80, and it is the bound `features.py`
# cuts the row list at while the file is being read, before anything is computed from a
# row.  The selection half (81 to 100) is not read here at all: a number computed from it
# would make the later choice between M's candidates circular.
HELD_FROM = 71
MAX_TUNE_SEED = FE.MAX_LABEL_SEED

# Candidate b's bar on the paired t, and the tolerance `../23/rows.py` reads it with.  The
# t of a row is a MASS POINT at 3 (seven passes one way and one the other give t = 3 in
# real arithmetic), so a bare `>=` in doubles splits that clump by the order the sums ran
# in.  Both numbers come from the modules that own them.
BAR = ST.T_BINS[-1][0]
BAR_TOL = 1e-6

# The ridge penalty, on STANDARDISED features, with the intercept unpenalised.  One fixed
# value, not tuned: a penalty chosen against the read-out rows would make them part of the
# fit and the agreement below would stop being a read-out number.
LAMBDA = 10.0

# The bootstrap over the ten read-out seeds: how many resamples, and the seed of the
# generator.  Seeds and not rows, because two decisions of one run are not independent.
RESAMPLES = 2000
BOOTSTRAP_SEED = 20261125

# The archived outputs whose numbers this module holds itself against.  Each is READ, not
# typed: part 1 and part 3 parse them and refuse if the line they need is absent.
ROWS_TXT = os.path.join(EV23, "rows.txt")
TUNE_TXT = os.path.join(EV23, "tune.txt")
STABILITY_TXT = os.path.join(EV22, "stability.txt")
ROWS_PY = os.path.join(EV23, "rows.py")


class Fail(Exception):
    """A check that did not hold.

    Its own class and not `AssertionError`, so that the mutation section can catch a check
    firing without also catching a typo in the mutation and printing that as if the check
    had worked.
    """


# ---------------------------------------------------------------------------
# reading the archived outputs, so that no number here is typed
# ---------------------------------------------------------------------------
def text_of(path):
    """One archived output's whole text.

    The read is separated from the parsing so that every refusal below can be shown firing
    on a mutated STRING, in this process: a mutated copy of an archived output on disk
    would be a file written outside this directory and a path in this output.
    """
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def found(pattern, text, where):
    """Every match of `pattern` in a text, refusing a text that holds none.

    A regex that matched nothing would silently become "no number to check against", which
    is the shape of error that lets a comparison pass by not happening.
    """
    hits = re.findall(pattern, text)
    if not hits:
        raise Fail("%s: nothing matches %r" % (where, pattern))
    return hits


def rows_txt_counts(text=None):
    """{candidate: (fit rows, read-out rows, the always-top-arm share)} from rows.txt.

    `rows.py`'s own part 4, parsed: the two row sets' fit and read-out counts and the
    baseline each one's read-out rows carry.  Read rather than typed, so a change in that
    script's counts breaks this script instead of being contradicted by it.
    """
    text = text_of(ROWS_TXT) if text is None else text
    split = found(r"candidate (\w)\s+tune (\d+) rows \(seeds 1-(\d+)\)\s+"
                  r"held out (\d+) rows \(seeds (\d+)-(\d+)\)", text,
                  "rows.txt part 4")
    base = found(r"always ([\d.]+) s on the held-out rows: ([\d.]+)", text,
                 "rows.txt part 4's baselines")
    if len(split) != len(base):
        raise Fail("rows.txt names %d candidate split line(s) and %d baseline line(s)"
                   % (len(split), len(base)))
    out = {}
    for (name, fit, upto, held, start, end), (arm, share) in zip(split, base):
        if (int(upto), int(start), int(end)) != (HELD_FROM - 1, HELD_FROM,
                                                 MAX_TUNE_SEED):
            raise Fail("rows.txt splits candidate %s at seeds 1-%s and %s-%s, this module"
                       " at 1-%d and %d-%d"
                       % (name, upto, start, end, HELD_FROM - 1, HELD_FROM,
                          MAX_TUNE_SEED))
        out[name] = (int(fit), int(held), float(arm), float(share))
    return out


def tune_txt_agreement(text=None):
    """{candidate: (kept epoch, agreed, of, share, baseline)} from ../23/tune.txt.

    The language model's own agreement on the same read-out rows, off the archive of the
    tuning run that produced the two kept adapters.  It is beside this module's numbers so
    that a reader is comparing two measurements of one quantity and not two quantities.
    """
    # The trailing newline is a LOOKAHEAD: the two summary lines sit one under the other,
    # so a pattern that consumed the newline after candidate a's line would eat the one
    # candidate b's match needs to start from, and only the first row would be found.
    text = text_of(TUNE_TXT) if text is None else text
    hits = found(r"\n  ([ab]) +(\d+) +([\d.]+) +(\d+)/(\d+) *= *([\d.]+) +([\d.]+)"
                 r" +(\d+) +(\d+)(?=\n)", text, "tune.txt part 4")
    out = {name: (int(epoch), int(agreed), int(of), float(share), float(base))
           for name, epoch, _ce, agreed, of, share, base, _rows, _ties in hits}
    if sorted(out) != ["a", "b"]:
        raise Fail("tune.txt part 4 names the candidates %s, not a and b" % sorted(out))
    return out


def stability_txt_residue():
    """(non-tie decisions, their agreement) of the two independent labellings.

    `../22/stability.txt`'s own J = 8 line for the main cell, which is the share of
    decisions that got a label from both 8-pass sets and got the SAME one.
    """
    hits = found(r"over the (\d+) non-tie decisions ([\d.]+)", text_of(STABILITY_TXT),
                 "stability.txt's J = 8 summary")
    return int(hits[0][0]), float(hits[0][1])


def top_level_calls(path):
    """The names a module calls at TOP LEVEL, i.e. the work `import` would do.

    Read off the source with `ast` and not by importing it, which is the point: a module
    whose analysis runs at import cannot be imported for one of its functions, and
    `../23/rows.py` is that module.  Counting its top-level calls says so with a number
    instead of a claim, and does it without running one line of it.
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            out.append(getattr(func, "id", None) or getattr(func, "attr", "?"))
    return out


def check_printing_at_import(path, where):
    """A module this one imports prints nothing while being imported.

    The two modules whose functions are reused here are held to it.  A module that printed
    at import would put its own lines inside this script's archived output, and an archived
    output that holds another script's output is not this script's output.
    """
    noisy = [name for name in top_level_calls(path) if name in ("say", "head", "print")]
    if noisy:
        raise Fail("%s: %s makes %d top-level call(s) that print, e.g. %s()"
                   % (where, os.path.basename(path), len(noisy), noisy[0]))
    return len(top_level_calls(path))


# ---------------------------------------------------------------------------
# the two row sets
# ---------------------------------------------------------------------------
def keeps_b(row, tol=BAR_TOL):
    """Candidate b's own rule on one row: `t >= 3 or (sd = 0 and labelled)`.

    Restated from `../23/rows.py` because that script cannot be imported (part 1), with
    its two reasons kept: `paired_t` hands back `t = None, sd = 0.0` when every pass
    separated the two best arms by the same amount, and such a row carries a label, which
    is the `sd = 0 and labelled` arm; and the bar is read with a tolerance because the t of
    a row is a mass point at 3 that a bare float comparison splits by summation order.  A
    row whose passes scored the two best arms EQUALLY is a tie, carries no label, and is
    refused by the first test before it reaches the sd branch.
    """
    if row["label"] is None:
        return False
    t, sd, _n = LBL.paired_t(row["pass_scores"], row["passes"])
    if sd == 0.0:
        return True
    return t is not None and t >= BAR - tol


def check_counts(got, want, where):
    """The row counts this module computes are the ones rows.txt archived.

    The whole point of recomputing them: if the filter were read differently here, every
    agreement below would be measured on a different row set than the one the language
    model was tuned and read out on, and the two would not be comparable.
    """
    if tuple(got) != tuple(want):
        raise Fail("%s: this module counts %s and rows.txt archives %s"
                   % (where, tuple(got), tuple(want)))
    return tuple(got)


def check_split(fit, held, where):
    """The fit rows and the read-out rows share no seed, and cover 1 to MAX_TUNE_SEED."""
    both = sorted({int(r["seed"]) for r in fit} & {int(r["seed"]) for r in held})
    if both:
        raise Fail("%s: %d seed(s) are in the fit and in the read-out, e.g. %s"
                   % (where, len(both), both[:4]))
    bad = sorted({int(r["seed"]) for r in list(fit) + list(held)}
                 - set(range(1, MAX_TUNE_SEED + 1)))
    if bad:
        raise Fail("%s: %d seed(s) outside 1 to %d, e.g. %s"
                   % (where, len(bad), MAX_TUNE_SEED, bad[:4]))
    return len(held)


def check_model_rows(of, got, name):
    """tune.txt read the language model out on the same rows this module reads out.

    If the two row counts differed, the agreement printed beside this module's would be
    one measured on another row set, and the whole point of the comparison is that it is
    not.
    """
    if of != got:
        raise Fail("tune.txt read candidate %s out on %d rows and this module on %d"
                   % (name, of, got))
    return of


def check_model_baseline(model_base, baseline, name, tol=5e-5):
    """tune.txt's own baseline for the candidate is the one computed here.

    The baseline is the always-top-arm share of the read-out rows.  Two different values
    would mean the two row sets differ in their labels even where they agree in count.
    """
    if abs(model_base - baseline) > tol:
        raise Fail("tune.txt's baseline for candidate %s is %.4f and this module's is"
                   " %.6f" % (name, model_base, baseline))
    return baseline


def check_resolved(rows, where):
    """Every label row carries a resolution, which is what makes its cell defined.

    The five cells are read off the resolution's trigger and its empty flag, so a row
    without one has no cell.  The label file is built from resolution-time decisions only,
    and this is that statement as a check rather than as an assumption: without it a
    send-time row would land in whichever cell the code happened to fall through to.
    """
    bad = [i for i, r in enumerate(rows) if r["resolved"] is None]
    if bad:
        raise Fail("%s: %d row(s) carry no resolution, the first at index %d"
                   % (where, len(bad), bad[0]))
    return len(rows)


# ---------------------------------------------------------------------------
# the classifier
# ---------------------------------------------------------------------------
def cell_of(record):
    """The five-cell scalar state of one label row's record, one of CELLS.

    The same bucketing `charges.cell` applies to a logged row, written here against the
    RECORD because a label row's resolution reaches this module through
    `observe.observation` and not as a raw logged action.  The two are held together by
    part 2, which counts the cells of both row sets and shows the five names exhaust them.
    """
    got = record["resolved"]
    if got is None:
        raise Fail("a record with no resolution has no cell")
    if got["trigger"] == "answered":
        return "ans_empty" if got["empty"] else "ans_content"
    return "to%d" % min(int(record["consecutive_poll_timeouts"]), 3)


def pick(three):
    """(index of the largest output, whether the largest was tied), first maximum wins.

    An explicit scan over the fixed arm order, so the shortest interval wins a tie.  It is
    the rule `../23/decider.py`'s `pick` applies to the language model's three logits, and
    it is restated and not imported because that module reads a command-line root at
    import.  The tie count is returned because a classifier whose outputs tie often is
    answering with the order of the arms and not with the data.
    """
    best = 0
    for i in (1, 2):
        if three[i] > three[best]:
            best = i
    return best, sum(1 for value in three if value == three[best]) > 1


def standardiser(matrix):
    """(mean, sd) of the fit rows, with a zero sd replaced by 1.

    A constant column has sd 0, and dividing by it would make every coefficient NaN and
    spread the NaN to all of them.  Replacing the sd by 1 leaves that column at exactly 0
    after centring, so the penalty holds its coefficient at 0 and the column contributes
    nothing instead of everything.
    """
    mean = matrix.mean(axis=0)
    sd = matrix.std(axis=0)
    return mean, np.where(sd > 0.0, sd, 1.0)


def design(matrix, mean, sd):
    """The standardised rows with a leading column of ones for the intercept."""
    return np.concatenate([np.ones((len(matrix), 1)), (matrix - mean) / sd], axis=1)


def ridge(mat, y, lam=LAMBDA):
    """The closed-form ridge solution, with the intercept left unpenalised.

    `numpy.linalg.solve` on the normal equations and no iteration anywhere, so the fit is
    the same every run.  The penalty skips the intercept because penalising it would pull
    every output towards 0 rather than towards the fit rows' own class shares.
    """
    penalty = np.eye(mat.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(mat.T @ mat + lam * penalty, mat.T @ y)


def answers_of(fit_columns, fit_labels, read_columns):
    """(answer per read row, answer per fit row, ties on the read rows, read outputs).

    One ridge to the three one-hot targets at once, which for a closed-form fit is three
    independent regressions sharing one factorisation, and then the largest output.  The
    fit rows' own answers come back too, because the gap between them and the read-out
    rows' is the only thing in this module that shows over-fitting.

    The three raw OUTPUTS of the read-out rows come back as well, and not only their
    argmax.  2026-11-25: the first version returned the argmax alone, and part 3 then read
    "the five cells agree exactly with the prior" as "the cells carry no label signal".
    Those are different statements -- an argmax can be blind to a predictor that moves its
    three outputs without ever reordering them -- and the outputs are what tells them
    apart.
    """
    mean, sd = standardiser(fit_columns)
    target = np.eye(len(ARMS))[[ARMS.index(arm) for arm in fit_labels]]
    w = ridge(design(fit_columns, mean, sd), target)
    read_out = design(read_columns, mean, sd) @ w
    on_fit = design(fit_columns, mean, sd) @ w
    picks = [pick(row) for row in read_out]
    return ([ARMS[i] for i, _tied in picks],
            [ARMS[pick(row)[0]] for row in on_fit],
            sum(1 for _i, tied in picks if tied),
            read_out)


def brier(outputs, labels):
    """The mean squared distance from a row's three outputs to its one-hot label.

    A CALIBRATED read-out, unlike agreement: it reads all three outputs of every row, so a
    predictor that moves the outputs towards the right arm scores better even when the
    largest one never changes.  Lower is better, 0 is a perfect prediction, and the
    always-the-prior predictor's value is the number the others are read against.  It is
    the Brier score and not a log loss because a ridge output is not a probability -- the
    three sum to one but can be negative -- and a log loss would need a clipping constant
    chosen here, which would be a number nobody computed.
    """
    want = np.eye(len(ARMS))[[ARMS.index(arm) for arm in labels]]
    return float(np.mean(np.sum((np.asarray(outputs) - want) ** 2, axis=1)))


def codes_of(cells, labels):
    """(cell index per row, arm index per row) as two integer arrays.

    The counting below is done on indices and not on names so that a permutation can be
    run two thousand times over ten thousand rows inside numpy instead of in a loop.
    """
    return (np.array([CELLS.index(cell) for cell in cells]),
            np.array([ARMS.index(arm) for arm in labels]))


def table_of_codes(cell_codes, arm_codes):
    """The counts of (five-cell state, label), as a CELLS-by-ARMS float array."""
    flat = np.bincount(cell_codes * len(ARMS) + arm_codes,
                       minlength=len(CELLS) * len(ARMS))
    return flat.reshape(len(CELLS), len(ARMS)).astype(float)


def cell_label_table(cells, labels):
    """The counts of (five-cell state, label) for rows named by cell and arm."""
    return table_of_codes(*codes_of(cells, labels))


def chi_square(table):
    """(the statistic, its degrees of freedom) for independence of rows and columns.

    The statistic is sum (observed - expected)^2 / expected over the cells the margins
    give an expectation to, and the degrees of freedom count only those rows and columns
    that hold something, so an empty row cannot inflate them.  No p-value is printed from
    it: this module has numpy and no distribution function, and a critical value typed
    from memory would be the one number here nobody computed.  What stands in for it is
    the permutation below, which is computed.
    """
    rows = table.sum(axis=1, keepdims=True)
    cols = table.sum(axis=0, keepdims=True)
    total = table.sum()
    if total <= 0:
        raise Fail("a chi-square over an empty table")
    expected = rows @ cols / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0.0, (table - expected) ** 2
                         / np.where(expected > 0.0, expected, 1.0), 0.0)
    df = (int((rows > 0).sum()) - 1) * (int((cols > 0).sum()) - 1)
    return float(terms.sum()), df


def permuted_p(cells, labels, statistic, draws=RESAMPLES, seed=BOOTSTRAP_SEED):
    """The share of label shuffles whose chi-square is at least the real one.

    The labels are permuted among the SAME rows, which keeps both margins and destroys
    only the pairing, so this is the independence test's own null drawn rather than looked
    up.  It is a share of `draws` and therefore cannot be smaller than 1 / draws; the
    output prints it as an upper bound when no draw reached the statistic.
    """
    rng = np.random.default_rng(seed)
    cell_codes, arm_codes = codes_of(cells, labels)
    hits = 0
    for _ in range(draws):
        shuffled = rng.permutation(arm_codes)
        if chi_square(table_of_codes(cell_codes, shuffled))[0] >= statistic:
            hits += 1
    return hits / draws


def majority(fit_labels):
    """The fit rows' most frequent arm, ties going to the first in ARMS order."""
    counts = collections.Counter(fit_labels)
    return max(ARMS, key=lambda arm: (counts[arm], -ARMS.index(arm)))


def agreement(answers, labels):
    """The share of rows where the answer is the label."""
    if not labels:
        raise Fail("agreement over no row")
    return sum(1 for a, b in zip(answers, labels) if a == b) / len(labels)


def bootstrap_se(answers, labels, seeds, baseline_arm):
    """(mean, SE) of `agreement - the always-baseline_arm agreement`, over the seeds.

    The resample is of the TEN read-out seeds with replacement, not of the rows: two
    decisions of one run share a record stream and a row resample would call that
    dependence independent and shrink the SE.  The two agreements are computed inside the
    SAME resample, so the difference is paired and the seeds' own class shares cancel.
    """
    by_seed = collections.defaultdict(list)
    for answer, want, seed in zip(answers, labels, seeds):
        by_seed[int(seed)].append((answer, want))
    names = sorted(by_seed)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(RESAMPLES):
        rows = [row for i in rng.integers(0, len(names), len(names))
                for row in by_seed[names[i]]]
        hit = sum(1 for answer, want in rows if answer == want)
        flat = sum(1 for _answer, want in rows if want == baseline_arm)
        draws.append((hit - flat) / len(rows))
    return float(np.mean(draws)), float(np.std(draws, ddof=1))


def per_seed_diff(answers, labels, seeds, baseline_arm):
    """(the per-seed paired differences, their mean, their SE, the t, seeds above 0).

    The same difference as the bootstrap's, one number per read-out seed, so the verdict
    can be read against a t on (seeds - 1) degrees of freedom instead of against a normal.
    2026-11-25: the first version printed the plain difference divided by the bootstrap SE
    as "N times the bootstrap SE" with no count beside it, which invites a reader to take
    N as a normal z; with ten clusters it is not one.
    """
    by_seed = collections.defaultdict(list)
    for answer, want, seed in zip(answers, labels, seeds):
        by_seed[int(seed)].append((answer, want))
    diffs = []
    for seed in sorted(by_seed):
        rows = by_seed[seed]
        hit = sum(1 for answer, want in rows if answer == want) / len(rows)
        flat = sum(1 for _answer, want in rows if want == baseline_arm) / len(rows)
        diffs.append(hit - flat)
    mean = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    return diffs, mean, se, (mean / se if se else 0.0), sum(1 for d in diffs if d > 0.0)


# ---------------------------------------------------------------------------
# the output
# ---------------------------------------------------------------------------
def say(line=""):
    """One line of the archived output."""
    print(line, flush=True)


def head(title):
    """One part's heading, so the archived output can be read part by part."""
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


def main():
    """Every table, and every check shown failing on a mutated input."""
    say("THIS IS NOT M.")
    say()
    say("M is the decision maker learned from the s11a EMULATOR logs and scored in the")
    say("replay.  This file touches neither.  It fits a small classifier to ticket 09's")
    say("LABEL rows and reports its agreement with those labels on seeds %d to %d."
        % (HELD_FROM, MAX_TUNE_SEED))
    say("A label is the arm that is best IF the formula rule takes over afterwards,")
    say("measured inside the replay over 8 passes, so agreement with it is a measure of")
    say("neither arm and of no policy.  Nothing below is M's number, nothing below is a")
    say("score, and no replay simulation and no emulator log is read here at all.")
    say("The question it answers is narrower: is there learnable signal in these labels?")

    head("0  THE FILE, AND THE SEEDS")
    path, want = FE.label_file()
    rows, digest = FE.label_rows(path, want)
    seeds_seen = sorted({int(r["seed"]) for r in rows})
    say("label cell       %s, the main cell; p 0.4's labels are not read" % FE.LABEL_CELL)
    say("sha256           %s" % digest)
    say("                 == ../22/labels.sha256, checked before any row was parsed")
    say("rows kept        %d, of seeds %d to %d (%d distinct)"
        % (len(rows), seeds_seen[0], seeds_seen[-1], len(seeds_seen)))
    say("the cut          features.label_rows parses each line of the file and drops the")
    say("                 row if its seed is above %d, so nothing of the selection half"
        % MAX_TUNE_SEED)
    say("                 (%d to 100) is counted, rendered, fitted or summarised anywhere"
        % (MAX_TUNE_SEED + 1))
    say("                 here.  The cut is before anything is computed from a row, not")
    say("                 before the line is parsed: the seed is a field of the row, and")
    say("                 the committed digest covers the whole file either way")
    say("seed guards      features.check_label_seeds refuses a label seed above %d and"
        % MAX_TUNE_SEED)
    say("                 port.check_seeds refuses one above %d; part 6 shows both firing"
        % PORT.MAX_SEED)
    say("the split        seeds 1 to %d fit, %d to %d read out, which is the split ticket"
        % (HELD_FROM - 1, HELD_FROM, MAX_TUNE_SEED))
    say("                 10 tuned the language model under")

    head("1  THE TWO ROW SETS, AND WHY rows.py IS NOT IMPORTED")
    say("../23/rows.py fixed the two candidates and counted them.  It is not imported:")
    noisy = [name for name in top_level_calls(ROWS_PY)
             if name in ("say", "head", "print")]
    say("it makes %d call(s) at top level, %d of which print, so `import rows` would run"
        % (len(top_level_calls(ROWS_PY)), len(noisy)))
    say("its whole analysis and write its own output -- machine paths included -- into")
    say("this one's.  Counted off its source with `ast`, running none of it.")
    say()
    say("What IS imported, so that no definition is written twice:")
    for name, where in (("../22/label.py", os.path.join(EV22, "label.py")),
                        ("../22/stability.py", os.path.join(EV22, "stability.py"))):
        say("  %-20s %d top-level call(s), none of which print"
            % (name, check_printing_at_import(where, "the modules this file imports")))
    say("  label.paired_t       the statistic candidate b is selected by")
    say("  stability.T_BINS     the last bin's lower edge, which is the bar %.1f" % BAR)
    say()
    say("Candidate a   every kept row that carries a label.")
    say("Candidate b   the rows of a with t >= %.1f - %g, or sd = 0 and labelled."
        % (BAR, BAR_TOL))
    say("              The tolerance is there because the t of a row is a mass point at")
    say("              %.1f and a bare float comparison splits that clump by the order"
        % BAR)
    say("              the sums ran in.")
    say()
    archived = rows_txt_counts()
    sets = {"a": [r for r in rows if r["label"] is not None],
            "b": [r for r in rows if keeps_b(r)]}
    say("  %-12s %9s %10s %15s %14s  %s"
        % ("candidate", "fit rows", "read-out", "rows.txt fit", "rows.txt out",
           "equal"))
    split = {}
    for name in ("a", "b"):
        fit = [r for r in sets[name] if int(r["seed"]) < HELD_FROM]
        held = [r for r in sets[name] if int(r["seed"]) >= HELD_FROM]
        check_split(fit, held, "candidate %s" % name)
        check_resolved([FE.record_from_label_row(r) for r in held],
                       "candidate %s's read-out rows" % name)
        want_fit, want_held, _arm, _share = archived[name]
        check_counts((len(fit), len(held)), (want_fit, want_held),
                     "candidate %s" % name)
        split[name] = (fit, held)
        say("  %-12s %9d %10d %15d %14d  %s"
            % (name, len(fit), len(held), want_fit, want_held, True))
    say()
    say("  the five scalar cells of the read-out rows, to show the five names exhaust")
    say("  them and that the cell-only classifier has something to read")
    say("  %-12s %9s %11s %7s %7s %7s %8s"
        % ("candidate", *CELLS, "all"))
    for name in ("a", "b"):
        counts = collections.Counter(cell_of(FE.record_from_label_row(r))
                                     for r in split[name][1])
        say("  %-12s %9d %11d %7d %7d %7d %8d"
            % (name, *[counts[one] for one in CELLS], sum(counts.values())))
    say()
    say("  the label each read-out row carries, per arm")
    say("  %-12s %10s %10s %10s %14s" % ("candidate", *["%.3f s" % a for a in ARMS],
                                         "always-top arm"))
    for name in ("a", "b"):
        counts = collections.Counter(r["label"] for r in split[name][1])
        top = majority([r["label"] for r in split[name][1]])
        say("  %-12s %10d %10d %10d %8.3f s %.4f"
            % (name, *[counts[a] for a in ARMS], top, counts[top] / len(split[name][1])))

    head("2  THE THREE PREDICTORS")
    say("All three are fitted on seeds 1 to %d of their own row set and read out on %d to"
        % (HELD_FROM - 1, HELD_FROM))
    say("%d.  No read-out row's seed is in any fit (asserted in part 1)." % MAX_TUNE_SEED)
    say()
    say("  full vector  closed-form ridge from the %d-cell feature vector to the three"
        % FE.LENGTH)
    say("               one-hot targets, penalty %.1f on features standardised on the fit"
        % LAMBDA)
    say("               rows, the answer being the largest of the three outputs")
    say("  five cells   the same arithmetic on the five scalar-cell indicators alone,")
    say("               which is what ADR 0004 line 5 permits a policy to read")
    say("  majority     always the fit rows' most frequent arm, which reads no input")
    say()
    say("The feature vector is built from each row's own observation through")
    say("features.record_from_label_row, which is observe.observation on the row's")
    say("`state`, `resolved`, `history`, `ts_ms` and `neighbor` -- the same five fields,")
    say("in the same call, that the language model's text was rendered from.  A tie among")
    say("the three outputs goes to the first arm in the order")
    say("%s, which is the rule the language model's three words" % list(ARMS))
    say("were picked by.")

    head("3  THE READ-OUT, SEEDS %d TO %d" % (HELD_FROM, MAX_TUNE_SEED))
    say("Agreement is the share of read-out rows whose answer is the row's label.  The")
    say("baseline is the always-10.043 s answer on THOSE rows, which is also the most")
    say("frequent arm there.  The difference is paired inside a bootstrap of %d resamples"
        % RESAMPLES)
    say("of the TEN read-out seeds with replacement (numpy.random.default_rng(%d)),"
        % BOOTSTRAP_SEED)
    say("so the SE carries the between-run spread and not a between-row one.  The `diff`")
    say("column is that bootstrap's own mean of the paired difference, which sits a few")
    say("ten-thousandths off the plain difference because a resample reweights the seeds;")
    say("the plain difference is the one the verdict below the table divides by the SE.")
    say()
    say("The language model's own agreement on the SAME rows is the last column, read off")
    say("../23/tune.txt's own part 4: its kept epoch's held-out agreement.  It is the")
    say("number this module exists to stand beside.")
    say()
    language = tune_txt_agreement()
    say("  %-5s %-12s %7s %10s %10s %11s %8s %7s %12s"
        % ("set", "predictor", "rows", "agreed", "agreement", "baseline", "diff", "SE",
           "the model"))
    scored = {}
    for name in ("a", "b"):
        fit, held = split[name]
        fit_labels = [r["label"] for r in fit]
        held_labels = [r["label"] for r in held]
        held_seeds = [int(r["seed"]) for r in held]
        fit_records = [FE.record_from_label_row(r) for r in fit]
        held_records = [FE.record_from_label_row(r) for r in held]
        base_arm = majority(fit_labels)
        baseline = agreement([base_arm] * len(held_labels), held_labels)
        got = {}
        for label, columns in (
                ("full vector", lambda rs: np.stack([FE.vector(r) for r in rs])),
                ("five cells",
                 lambda rs: np.eye(len(CELLS))[[CELLS.index(cell_of(r)) for r in rs]])):
            got[label] = answers_of(columns(fit_records), fit_labels,
                                    columns(held_records))
        # The prior as a PREDICTOR of all three arms, and not only as an argmax: the fit
        # rows' own class shares, the same for every read-out row.  It is what the two
        # fitted predictors' Brier scores below are read against.
        shares = np.array([sum(1 for arm in fit_labels if arm == a) / len(fit_labels)
                           for a in ARMS])
        got["majority"] = ([base_arm] * len(held_labels),
                           [base_arm] * len(fit_labels), 0,
                           np.tile(shares, (len(held_labels), 1)))
        held_cells = [cell_of(r) for r in held_records]
        fit_cells = [cell_of(r) for r in fit_records]
        scored[name] = (got, held_labels, held_seeds, fit_labels, baseline, base_arm,
                        held_cells, fit_cells)
        epoch, agreed, of, share, model_base = language[name]
        check_model_rows(of, len(held_labels), name)
        check_model_baseline(model_base, baseline, name)
        for label in ("full vector", "five cells", "majority"):
            read_out = got[label][0]
            rate = agreement(read_out, held_labels)
            mean, se = bootstrap_se(read_out, held_labels, held_seeds, base_arm)
            say("  %-5s %-12s %7d %10d %10.4f %11.4f %8.4f %7.4f %12s"
                % (name, label, len(held_labels),
                   sum(1 for x, y in zip(read_out, held_labels) if x == y), rate,
                   baseline, mean, se,
                   "%d/%d" % (agreed, of) if label == "full vector" else ""))
        say("  %-5s %-12s %7s %10s %10.4f %11.4f %8.4f %7s %12s"
            % (name, "the model", of, agreed, share, model_base, share - model_base, "",
               "epoch %d" % epoch))
    say()
    say("  The model's own row carries tune.txt's single archived number and no SE: it")
    say("  is one read-out of one adapter, so the two `diff` columns are not two draws of")
    say("  one quantity and their difference is not tested anywhere.")
    say()
    say("  THE ANSWER TO THE QUESTION, in the numbers above:")
    for name in ("a", "b"):
        (got, held_labels, held_seeds, _fit_labels, baseline, base_arm,
         _hc, _fc) = scored[name]
        rate = agreement(got["full vector"][0], held_labels)
        mean, se = bootstrap_se(got["full vector"][0], held_labels, held_seeds, base_arm)
        diffs, pmean, pse, t, up = per_seed_diff(got["full vector"][0], held_labels,
                                                 held_seeds, base_arm)
        flat = agreement(got["five cells"][0], held_labels)
        say("  candidate %s: the full vector agrees %.4f against the always-%.3f baseline"
            % (name, rate, base_arm))
        say("    %.4f, which is +%.4f, or %.1f times the bootstrap SE %.4f of %d read-out"
            % (baseline, rate - baseline,
               abs(rate - baseline) / se if se else 0.0, se, len(diffs)))
        say("    SEED clusters.  Read per seed instead: the paired difference is +%.4f"
            % pmean)
        say("    with SE %.4f, t %.2f on %d degrees of freedom, and %d of the %d seeds"
            % (pse, t, len(diffs) - 1, up, len(diffs)))
        say("    positive.  With ten clusters that ratio is not a normal z, which is why")
        say("    the count is beside it.")
        say("    The five cells' ARGMAX agrees %.4f, the baseline to %.4f: it never leaves"
            % (flat, abs(flat - baseline)))
        say("    the majority arm %.3f, so an argmax read-out of them is the prior by"
            % base_arm)
        say("    construction.  What the cells do carry is the next two tables, not this")
        say("    one.")
    say()
    say("  the per-seed paired differences of the full vector, so the ten clusters the SE")
    say("  is taken over are visible and not only their spread")
    for name in ("a", "b"):
        (got, held_labels, held_seeds, _fit_labels, _base, base_arm,
         _hc, _fc) = scored[name]
        diffs, _m, _s, _t, _up = per_seed_diff(got["full vector"][0], held_labels,
                                               held_seeds, base_arm)
        say("  candidate %s   %s"
            % (name, " ".join("%+.3f" % d for d in diffs)))
    say()
    say("  ties among the three outputs, which would mean the answer came from the arm")
    say("  order and not from the data")
    for name in ("a", "b"):
        got = scored[name][0]
        say("  candidate %s   full vector %d, five cells %d, of %d read-out rows"
            % (name, got["full vector"][2], got["five cells"][2],
               len(scored[name][1])))

    say()
    say("  A CALIBRATED READ-OUT OF THE SAME THREE PREDICTORS, because agreement is an")
    say("  argmax and an argmax cannot tell a predictor that carries nothing from one")
    say("  that carries something it never reorders.  The Brier score is the mean squared")
    say("  distance from a row's three outputs to its one-hot label, lower being better,")
    say("  and the prior row is the fit rows' own class shares handed to every read-out")
    say("  row.  The SE column is the same seed bootstrap as above, on the paired")
    say("  difference from the prior, so a negative `diff` is a predictor that beats the")
    say("  prior on all three outputs and not only at its argmax.")
    say()
    say("  %-5s %-12s %10s %10s %10s %8s"
        % ("set", "predictor", "Brier", "prior", "diff", "SE"))
    for name in ("a", "b"):
        (got, held_labels, held_seeds, _fit_labels, _base, _arm,
         _hc, _fc) = scored[name]
        flat = brier(got["majority"][3], held_labels)
        for label in ("full vector", "five cells", "majority"):
            mine = brier(got[label][3], held_labels)
            per_seed = collections.defaultdict(list)
            for i, seed in enumerate(held_seeds):
                per_seed[int(seed)].append(i)
            draws = []
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            names_of = sorted(per_seed)
            for _ in range(RESAMPLES):
                idx = [i for j in rng.integers(0, len(names_of), len(names_of))
                       for i in per_seed[names_of[j]]]
                labels_here = [held_labels[i] for i in idx]
                draws.append(brier([got[label][3][i] for i in idx], labels_here)
                             - brier([got["majority"][3][i] for i in idx], labels_here))
            say("  %-5s %-12s %10.4f %10.4f %10.4f %8.4f"
                % (name, label, mine, flat, mine - flat,
                   float(np.std(draws, ddof=1))))
    say()
    say()
    say("  THE CORRECTED READING OF THE FIVE CELLS, in those two columns: their Brier")
    say("  score is BELOW the prior's on both row sets, by a few times its own seed-")
    say("  clustered SE, so they do carry label information -- what they never do is move")
    say("  the largest of the three outputs off %.3f, which is why the argmax table above"
        % ARMS[-1])
    say("  reads as the prior exactly.  The full vector's Brier gain is an order of")
    say("  magnitude larger than theirs on both sets.  2026-11-25: the first version of")
    say("  this part read the equality of the argmax numbers as `the cells add nothing`,")
    say("  which the two tables here contradict.")
    say()
    say("  WHAT THE FIVE CELLS CARRY, as the label against the cell on the SAME rows.")
    say("  The counts are the read-out rows; the share columns are P(label | cell).  If")
    say("  the cells carried no label information these five rows would be one")
    say("  distribution.  The statistic is the chi-square of independence with its degrees")
    say("  of freedom, and the p is the share of %d label PERMUTATIONS of the same rows"
        % RESAMPLES)
    say("  whose statistic reached the real one (a permutation and not a table lookup, so")
    say("  it is a number this module computed).  Note what it does NOT overturn: 10.043")
    say("  is the majority label inside every one of the five cells, which is why the")
    say("  argmax read-out above equals the prior exactly.")
    for which, title in ((6, "read-out rows, seeds %d to %d" % (HELD_FROM,
                                                                MAX_TUNE_SEED)),
                         (7, "fit rows, seeds 1 to %d" % (HELD_FROM - 1))):
        for name in ("a", "b"):
            (_got, held_labels, _seeds, fit_labels, _base, _arm,
             held_cells, fit_cells) = scored[name]
            cells_here = held_cells if which == 6 else fit_cells
            labels_here = held_labels if which == 6 else fit_labels
            table = cell_label_table(cells_here, labels_here)
            stat, df = chi_square(table)
            share = permuted_p(cells_here, labels_here, stat)
            say()
            say("  candidate %s, %s" % (name, title))
            say("    %-12s %7s %10s %10s %10s %10s"
                % ("cell", "rows", *["P(%.3f)" % a for a in ARMS], "majority"))
            for i, one in enumerate(CELLS):
                total = table[i].sum()
                if not total:
                    say("    %-12s %7d %10s %10s %10s %10s"
                        % (one, 0, "--", "--", "--", "--"))
                    continue
                say("    %-12s %7d %10.4f %10.4f %10.4f %10.3f"
                    % (one, int(total), *[table[i][j] / total
                                          for j in range(len(ARMS))],
                       ARMS[int(np.argmax(table[i]))]))
            margin = table.sum(axis=0)
            say("    %-12s %7d %10.4f %10.4f %10.4f %10.3f"
                % ("every row", int(margin.sum()),
                   *[margin[j] / margin.sum() for j in range(len(ARMS))],
                   ARMS[int(np.argmax(margin))]))
            say("    chi-square %.1f on %d degrees of freedom, permutation p %s"
                % (stat, df, "%.4f" % share if share else "< %.4f" % (1.0 / RESAMPLES)))

    head("4  THE 3 x 3 TABLE OF ANSWERS AGAINST LABELS")
    say("Rows are the label the file carries, columns the answer.  A predictor whose")
    say("argmax never leaves the majority arm fills one column: that is what the five-cell")
    say("classifier and the majority both do below, and it is NOT the same statement as")
    say("`the cells carry no label information` -- the cross-tab above shows they do.  One")
    say("column means the information never reorders the three outputs, nothing more.")
    for name in ("a", "b"):
        got, held_labels = scored[name][0], scored[name][1]
        for label in ("full vector", "five cells", "majority"):
            answers = got[label][0]
            say()
            say("  candidate %s, %s" % (name, label))
            say("    %-12s %10s %10s %10s %10s"
                % ("label", *["answer %.3f" % a for a in ARMS], "rows"))
            table = collections.Counter(zip(held_labels, answers))
            for arm in ARMS:
                row = [table[(arm, other)] for other in ARMS]
                say("    %-12.3f %10d %10d %10d %10d"
                    % (arm, *row, sum(row)))
            column = [sum(table[(arm, other)] for arm in ARMS) for other in ARMS]
            say("    %-12s %10d %10d %10d %10d"
                % ("answers", *column, sum(column)))
            check_table(table, held_labels, answers, "candidate %s, %s" % (name, label))

    head("5  THE FIT SIDE, SO OVER-FITTING IS VISIBLE")
    say("The same three predictors' agreement on the rows they were fitted on, seeds 1 to")
    say("%d.  A full-vector fit that sat far above its read-out number would be reading"
        % (HELD_FROM - 1))
    say("its own fit rows; the gap below is what says how much of that there is.")
    say()
    say("  %-5s %-12s %8s %10s %10s %10s"
        % ("set", "predictor", "fit rows", "on the fit", "read out", "gap"))
    for name in ("a", "b"):
        got, held_labels, fit_labels = (scored[name][0], scored[name][1],
                                        scored[name][3])
        for label in ("full vector", "five cells", "majority"):
            on_fit = agreement(got[label][1], fit_labels)
            read_out = agreement(got[label][0], held_labels)
            say("  %-5s %-12s %8d %10.4f %10.4f %10.4f"
                % (name, label, len(fit_labels), on_fit, read_out, on_fit - read_out))

    head("6  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT")
    say("The mutation is of the INPUT, inside this process: a row list, a count, a set of")
    say("outputs or a file name made wrong on purpose.")
    say()
    fired = []

    def shows(name, call):
        """Run a check on a mutated input and require it to refuse."""
        try:
            call()
        except (Fail, SystemExit, FE.Refuse) as exc:
            say("  %-26s FIRES: %s" % (name, exc))
            return True
        say("  %-26s DID NOT FIRE -- this check cannot fail on this input" % name)
        return False

    fit_a, held_a = split["a"]
    fired.append(shows("label file digest",
                       lambda: FE.label_rows(path, "deadbeef")))
    fired.append(shows("label seeds > %d" % MAX_TUNE_SEED,
                       lambda: FE.check_label_seeds([MAX_TUNE_SEED + 1], "mutated")))
    fired.append(shows("port.check_seeds > %d" % PORT.MAX_SEED,
                       lambda: PORT.check_seeds([PORT.MAX_SEED + 1], "mutated")))
    fired.append(shows("counts against rows.txt",
                       lambda: check_counts((len(fit_a), len(held_a) + 1),
                                            archived["a"][:2], "one row too many")))
    fired.append(shows("fit and read-out split",
                       lambda: check_split(fit_a, fit_a, "the fit read out on itself")))
    fired.append(shows("seed outside 1 to %d" % MAX_TUNE_SEED,
                       lambda: check_split(
                           fit_a, held_a + [dict(held_a[0], seed=MAX_TUNE_SEED + 1)],
                           "a selection-half row")))
    fired.append(shows("every row resolved", lambda: check_resolved(
        [dict(FE.record_from_label_row(held_a[0]), resolved=None)], "a send-time row")))
    fired.append(shows("cell without a resolution",
                       lambda: cell_of(dict(FE.record_from_label_row(held_a[0]),
                                            resolved=None))))
    fired.append(shows("the 3 x 3 table closes", lambda: check_table(
        collections.Counter(), [ARMS[0]], [ARMS[0]], "an empty table")))
    fired.append(shows("agreement over no row", lambda: agreement([], [])))
    fired.append(shows("a regex that finds nothing",
                       lambda: found(r"no such line anywhere", text_of(ROWS_TXT),
                                     "mutated")))
    fired.append(shows("a module prints at import",
                       lambda: check_printing_at_import(ROWS_PY, "mutated")))

    # The five refusals that parse the two archived outputs, each on a mutated TEXT of
    # that output.  2026-11-25: all five were reachable and in no mutation row, while the
    # heading of this part claimed every check; the mutation is the string, so no copy of
    # an archive is written and no path reaches this output.
    rows_text = text_of(ROWS_TXT)
    tune_text = text_of(TUNE_TXT)
    one_more = ("candidate c tune 10 rows (seeds 1-%d) held out 5 rows (seeds %d-%d)"
                % (HELD_FROM - 1, HELD_FROM, MAX_TUNE_SEED))
    fired.append(shows("rows.txt, a split unpaired",
                       lambda: rows_txt_counts(rows_text + "\n" + one_more + "\n")))
    fired.append(shows("rows.txt, another split", lambda: rows_txt_counts(
        rows_text + "\ncandidate c tune 10 rows (seeds 1-60) held out 5 rows (seeds"
        " 61-%d)\nalways 10.043 s on the held-out rows: 0.5000\n" % MAX_TUNE_SEED)))
    fired.append(shows("tune.txt, one candidate", lambda: tune_txt_agreement(
        re.sub(r"\n  a +\d+ +[\d.]+ +\d+/\d+ *= *[\d.]+ +[\d.]+ +\d+ +\d+(?=\n)", "",
               tune_text))))
    fired.append(shows("tune.txt, other rows",
                       lambda: check_model_rows(len(held_a) + 1, len(held_a), "a")))
    fired.append(shows("tune.txt, another baseline",
                       lambda: check_model_baseline(0.5, 0.6, "a")))
    fired.append(shows("a chi-square of nothing",
                       lambda: chi_square(np.zeros((len(CELLS), len(ARMS))))))
    say()
    say("%d of %d mutations produced a failure" % (sum(fired), len(fired)))
    say()
    say("Two checks no input mutation can fail, named rather than left out:")
    say("- that a tie goes to the first arm.  `pick` scans for a STRICTLY greater output,")
    say("  so the first maximum is kept by construction and no input can change that.")
    say("  What can be shown is the rule's answer on a hand-made tie, which is next.")
    three = [1.0, 1.0, 1.0]
    index, tied = pick(three)
    say("  pick(%s) returns index %d (%.3f s) and reports the tie: %s"
        % (three, index, ARMS[index], tied))
    say("- that the feature vector carries what the rendered text carries.  That is")
    say("  features.py's own round trip, shown failing on mutated records in its part 6,")
    say("  and nothing here can make it fail again.")

    head("7  WHAT THIS RESULT CAN AND CANNOT MEAN")
    non_tie, same = stability_txt_residue()
    say("- A LABEL IS NOT A VERDICT ON AN ARM.  It is the arm that is best IF the formula")
    say("  rule takes over afterwards, measured inside the replay over 8 passes.  So a")
    say("  high agreement means a classifier matched that quantity, not that it would")
    say("  lower a score, and a low one does not say the arm was wrong.")
    say("- THE LABELS DISAGREE WITH THEMSELVES.  Two independent 8-pass labellings of the")
    say("  same decisions agreed on %.4f of the %d decisions they both labelled"
        % (same, non_tie))
    say("  (../22/stability.txt, seeds 1 to 10 of this cell, read off that file and not")
    say("  typed here).  So %.4f of them got a DIFFERENT label, and a classifier that"
        % (1.0 - same))
    say("  matched these labels perfectly would be matching a teacher that disagrees with")
    say("  itself about a quarter of the time.")
    say("  That share is an upper bound on the labels that are pass noise, not an")
    say("  estimate of it: one disagreement needs only one of the two sides to be wrong.")
    say("- AN ARGMAX AND A SIGNAL ARE NOT THE SAME QUESTION.  The five cells' argmax equals")
    say("  the prior exactly, and their Brier score and the cross-tab both say they carry")
    say("  label information.  So nothing here says the ADR 0004 scalars carry no signal;")
    say("  what it says is that a three-arm argmax read-out of them cannot express the")
    say("  signal they have, because 10.043 is the majority inside all five cells.  How")
    say("  much of the full vector's own gain lies OUTSIDE the cells is not decomposed")
    say("  here: its Brier gain is the larger of the two on both row sets, and no number")
    say("  in this file splits it into a part the cells reach and a part they do not.")
    say("- AGREEMENT IS NOT A SCORE.  No replay simulation is run here and no score is")
    say("  computed.  The choice between any two deciders is made on replay scores of the")
    say("  selection seeds, which this file does not read.")
    say("- THE ROW SETS ARE NOT A SAMPLE OF DECISIONS.  Candidate a drops the ties and")
    say("  candidate b keeps about four tenths of a, selected by the paired t of the very")
    say("  passes that produced the label, so b's rows are the ones the labelling was")
    say("  most sure of and its baseline is higher for that reason.")
    say("- NOTHING HERE IS ABOUT M.  M reads emulator logs, not label rows, and the rows")
    say("  it learns from carry no label at all: the labels were attached to decisions")
    say("  inside a replay roll-in, so there is no row-level join between the two.")
    say("- Nothing about the p 0.4 label cell, which is not read, and nothing about seeds")
    say("  %d to 100, which are the selection half." % (MAX_TUNE_SEED + 1))
    say("- The penalty %.1f is fixed and not tuned; nothing here says it is the best one,"
        % LAMBDA)
    say("  and a penalty chosen against the read-out rows would make them part of the fit.")
    say()
    say("no check fired on real input; the failures in part 6 are the mutations")
    say("end of labelsig.py")
    return 0


def check_table(table, labels, answers, where):
    """The 3 x 3 table holds every read-out row exactly once.

    Printed as row and column sums above; asserted here, because a table whose margins do
    not add to the row count would mean an answer or a label outside the three arms had
    been dropped silently, and the shares read off it would be over the wrong denominator.
    """
    got = sum(table[(arm, other)] for arm in ARMS for other in ARMS)
    if got != len(labels) or len(labels) != len(answers):
        raise Fail("%s: the table holds %d of the %d read-out row(s), with %d answer(s)"
                   % (where, got, len(labels), len(answers)))
    return got


if __name__ == "__main__":
    sys.exit(main())
