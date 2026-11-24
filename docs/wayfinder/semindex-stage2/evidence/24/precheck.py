#!/usr/bin/env python3
"""What the s11a logs of seeds 1 to 20 can carry, p 0.2 and p 0.4 separately.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B precheck.py

NO MODEL IS CHOSEN HERE.  This module measures what a fit could stand on: how many owned
decision periods the logs hold, what the logging policy's own cost per second is, how much
of what follows a decision the inputs explain on seeds the fit never saw, whether the
ORDER of the three intervals is readable from that, and what one decision's input path
costs in time.  Nothing below is a policy, nothing below is a score, and no replay
simulation of any kind is run.

THE TABLE, in one paragraph, so another agent can reimplement every number below.  One row
per OWNED decision period of `charges.py`: the period from one `state_poll_reschedule`
decision of a pair to that pair's next one.  The row carries the feature vector
`features.py` builds from the decision's own record, the interval the decision chose, the
period length tau in seconds, the period cost c in ms of score (`charges.py`'s
`cost_ms / N`), the five-cell scalar state, and the position of the NEXT decision of the
same pair in that pair's chain.  A pair's LAST decision owns no period (`charges.py` calls
its period the tail), so it is not a row: 8 pairs x 40 runs = 320 decisions are left out
and the count is printed.

THE TARGETS.  `rho` is the logging policy's own average cost per second, sum(c) / sum(tau)
over a cell's owned periods.  The one-period target is `y = c - rho * tau`: the cost of
the period less what the same seconds cost on average, so a long interval is not penalised
for owning more seconds.  Its sum over a cell's rows is exactly 0 by construction, which
is asserted.  The longer target sums `c - rho * tau` over the next K periods of the SAME
pair, counting this one, truncated at the end of the pair's chain.

WHAT THE HELD-OUT NUMBERS ARE, AND WHAT THEY ARE NOT.  The folds are seeds, five folds of
four, so no decision of a run contributes to both the fit and the read-out.  A held-out R
squared says how much of the variance of what FOLLOWED a decision the inputs explain.  It
does not say that acting on the prediction would lower a score: the rows are the logging
policy's, every continuation in them is the logging policy's, and the cost a decision
causes at the DOWNSTREAM pair is charged to that pair and is in nobody's target here.

THE LAST PART IS TIME ON THIS MACHINE and is the one part that does not reproduce byte for
byte.  It is last so that a second run of this script differs from the first only below
the marker line that opens it.
"""
import collections
import hashlib
import os
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
sys.path.insert(0, HERE)
sys.path.insert(0, EV22)

import charges as CH    # noqa: E402  the runs, the periods, the cost, the five cells
import features as FE   # noqa: E402  the vector and the one road from a logged row
import observe as OBS   # noqa: E402  `from_logged_rows`, `observation`: timed in part 9
import port as PORT     # noqa: E402  the >100 seed guard

P_CELLS = ("0.2", "0.4")
SEEDS = tuple(range(1, CH.MAX_LOG_SEED + 1))
ARMS = CH.ARMS
CELLS = CH.CELLS

# Five folds of four seeds, assigned by seed number alone: fold f holds seeds 4f+1 to
# 4f+4.  Fixed rather than drawn, because a drawn assignment would make every held-out
# number depend on an rng seed a reader cannot check.  Seeds and not decisions, because
# two decisions of one run share a record stream: splitting decisions would put the same
# stream on both sides of the split and the read-out would not be held out at all.
FOLDS = 5
FOLD_SIZE = 4

# The ridge penalty, on STANDARDISED features, with the intercept unpenalised.  One fixed
# value, not tuned: tuning it against the held-out folds would make those folds part of
# the fit and the R squared below would stop being held out.  10.0 is small beside the
# roughly 1000 rows each per-interval fit has, so it holds the 150 columns features.py
# names dead at 0 rather than shrinking the live ones much.
LAMBDA = 10.0

# The horizons of the longer target, in periods of the same pair, counting this one.
KS = (4, 8)

# The permutation seed of part 5's control.  Fixed, so that part reproduces.
PERMUTE_SEED = 20261125

# The number of calls each timing in part 9 is taken over, and the runs the decisions come
# from.  At least 1000 DISTINCT decisions, so a median and a p99 both rest on more than a
# handful of calls and neither rests on one decision timed a thousand times.
TIMING_CALLS = 1200
TIMING_SEEDS = tuple(range(1, 11))

# The one line part 9 opens with.  A reader comparing two runs of this script can diff
# them and find every differing line below this one.
TIME_MARKER = ">>> TIMINGS BELOW THIS LINE DO NOT REPRODUCE BYTE FOR BYTE <<<"


class Fail(Exception):
    """A check that did not hold.

    Its own class and not `AssertionError`, so that the mutation section can catch a check
    firing without also catching a typo in the mutation and printing that as if the check
    had worked.
    """


# ---------------------------------------------------------------------------
# the decision table
# ---------------------------------------------------------------------------
def records_of_run(events, source=None):
    """{(node, neighbour, ts_ms): (row, record)} for every reschedule row of one run.

    The key is what identifies an owned period: `charges.periods` cuts a period at its
    decision's own `ts_ms` and `charges.check_chain` refuses a pair whose decisions are not
    ordered by it, so (pair, start) names one decision and one only.

    The records come from `features.records_from_logged_rows`, which is
    `observe.from_logged_rows` -- the one shared writing of a decision's input.  That
    generator yields one record per reschedule row of the node in file order and does not
    hand the row back, so the rows are re-filtered here with the same two tests and zipped
    with it.

    `source` is that generator, as an argument.  It is an argument for ONE reason: the zip
    rests on two assumptions about it -- that it yields exactly one record per reschedule
    row, in file order, and that the neighbour it yields is the row's own -- and neither
    can be made to fail by any LOG, because both filters read the same two fields of the
    same rows.  The two checks below are therefore shown failing on a mutated SOURCE, which
    is what would really break them: a change to `observe.from_logged_rows`'s own filter.
    """
    source = FE.records_from_logged_rows if source is None else source
    out = {}
    for node in sorted(events):
        rows = [e for e in events[node] if e.get("event") == "decision"
                and e.get("kind") == "state_poll_reschedule"]
        made = list(source(events[node]))
        if len(rows) != len(made):
            raise Fail("%s: %d reschedule rows against %d records from the record source"
                       % (node, len(rows), len(made)))
        for row, (neighbor, record) in zip(rows, made):
            if row["action"]["neighbor"] != neighbor:
                raise Fail("%s: a record of pair %s was zipped onto a row of pair %s"
                           % (node, neighbor, row["action"]["neighbor"]))
            out[(node, neighbor, float(row["ts_ms"]))] = (row, record)
    return out


def matched_row(found, pair, per, t0, where):
    """(row, record) of the decision one owned period belongs to, matched three ways.

    The period and the row have to agree about the instant the decision was taken, the
    interval it chose and the cell it was taken in.  Each test is a separate refusal, and
    each would fire on a different kind of drift: a period cut at an instant no reschedule
    row carries, a period whose `interval_s` came from another row, and a period whose cell
    was bucketed from another row's state.

    `found` is the record dict, as an argument, so that part 6 can hand in a mutated one.
    2026-11-25: all three arms were reachable and in no mutation row of the first version,
    and the output claimed the cell arm could not be made to fail by any input -- true of
    any LOG, because both sides call `charges.cell`, and not true of a mutated record
    source, which is what this argument makes testable.
    """
    key = (pair[0], pair[1], per["start"])
    if key not in found:
        raise Fail("%s: the period of %s->%s starting %.3f s into the run has no"
                   " reschedule row at its own start instant"
                   % (where, pair[0], pair[1], (per["start"] - t0) / 1000.0))
    row, record = found[key]
    if float(row["action"]["interval_s"]) != per["interval_s"]:
        raise Fail("%s: %s->%s chose %.3f s and its period says %.3f s"
                   % (where, pair[0], pair[1], float(row["action"]["interval_s"]),
                      per["interval_s"]))
    if CH.cell(row) != per["cell"]:
        raise Fail("%s: %s->%s is in cell %s by the row and %s by the period"
                   % (where, pair[0], pair[1], CH.cell(row), per["cell"]))
    return row, record


def table_of_run(p, seed):
    """([row dict, ...], the charged run) for one logged run's owned decision periods.

    A row dict carries `vec` (the feature vector at the decision), `arm` (the interval it
    chose), `tau` (the period length in seconds), `c` (the period cost in ms of score),
    `cell` (the five-cell scalar state), `pos` (the decision's position in its pair's own
    chain of owned periods) and `chain` (that list), which is how the next decision of the
    pair is reached.  `next_pos` is `pos + 1` and is None on the pair's last OWNED period,
    whose next decision is the pair's last and owns no period of its own.
    """
    events, t0 = CH.load_run(p, seed)
    terms = CH.SCORE.terms(events, t0)
    tracks = CH.record_tracks(events, t0)
    run = CH.charge_run(events, t0, terms=terms, tracks=tracks)
    CH.check_identity(run, "p%s seed %d" % (p, seed))
    CH.check_placed(run, "p%s seed %d" % (p, seed))
    found = records_of_run(events)
    out = []
    for pair, chain in sorted(run["periods"].items()):
        seq = []
        for per in chain:
            if per["kind"] != "owned":
                continue
            row, record = matched_row(found, pair, per, run["t0"],
                                      "p%s seed %d" % (p, seed))
            seq.append(dict(p=p, seed=seed, pair=pair,
                            vec=FE.vector(record), arm=per["interval_s"],
                            tau=per["length_s"], c=per["cost_ms"] / run["n"],
                            cell=per["cell"]))
        for i, got in enumerate(seq):
            got["pos"] = i
            got["chain"] = seq
            got["next_pos"] = i + 1 if i + 1 < len(seq) else None
        out.extend(seq)
    return out, run


def check_owned_count(rows, runs, where):
    """The table holds every owned period of every run and no other decision.

    `charge_run` counts the owned periods itself, so this is the one check that says the
    table was built from the periods rather than from whatever rows happened to match.
    """
    want = sum(run["n_owned"] for run in runs)
    if len(rows) != want:
        raise Fail("%s: the table has %d rows and the runs hold %d owned periods"
                   % (where, len(rows), want))
    return want


def check_cost_closes(rows, runs, where, tol=1e-6):
    """The owned rows' cost plus the head and tail cost is the scorer's own cost.

    `charges.py` asserts this per run before any of its own tables; it is re-asserted here
    over the TABLE, because the table is a second object and a row lost between the periods
    and the table would otherwise be invisible -- the identity in `charges.txt` would still
    hold and every rate here would be computed over fewer rows.
    """
    got = sum(r["c"] for r in rows) + sum(run["cost_edge"] / run["n"] for run in runs)
    want = sum(run["total_cost"] / run["n"] for run in runs)
    if abs(got - want) > tol:
        raise Fail("%s: the table holds %.6f score ms and the scorer charges %.6f"
                   " (difference %.6f)" % (where, got, want, got - want))
    return got


# ---------------------------------------------------------------------------
# the targets
# ---------------------------------------------------------------------------
def rho_of(rows):
    """sum(c) / sum(tau): the cost per second of the periods handed in, in score ms/s.

    A pooled ratio and not a mean of per-period rates, so one short period cannot dominate
    it.  This is the logging policy's OWN rate over whatever subset is handed in; on a
    subset selected by the chosen interval it is not the effect of that interval, because
    the three subsets are three different mixtures of the five cells and of clock position.
    """
    secs = sum(r["tau"] for r in rows)
    if secs <= 0.0:
        raise Fail("rho over %d row(s) whose seconds sum to %.6f" % (len(rows), secs))
    return sum(r["c"] for r in rows) / secs


def target(row, k, rho):
    """The time-fair cost of this period and the next k-1 of the same pair.

    `c - rho * tau` per period, summed over the pair's own chain from this decision
    forward, truncated at the chain's end.  Counting THIS period means k = 1 is the
    one-period target, so the three horizons share one function and cannot drift.
    """
    chain = row["chain"]
    return sum(got["c"] - rho * got["tau"]
               for got in chain[row["pos"]:row["pos"] + k])


def targets(rows, k, rho):
    """The K-period target of every row handed in, as one array."""
    return np.array([target(r, k, rho) for r in rows])


def check_y_sums_to_zero(rows, rho, where, tol=1e-6):
    """The one-period target sums to 0 over the rows rho was computed from.

    By construction: sum(c) - rho * sum(tau) = 0 when rho is sum(c)/sum(tau).  It is
    asserted rather than trusted because it is the one line that says the target really is
    centred on the logging policy's own rate; a target centred somewhere else would make
    every R squared below partly a statement about the offset.
    """
    got = float(targets(rows, 1, rho).sum())
    scale = sum(abs(r["c"]) for r in rows) + 1.0
    if abs(got) > tol * scale:
        raise Fail("%s: the one-period targets sum to %.6f score ms, not 0" % (where, got))
    return got


# ---------------------------------------------------------------------------
# the three predictors
# ---------------------------------------------------------------------------
def folds_of(seeds):
    """[[seed, ...], ...]: five folds of four, assigned by seed number alone."""
    out = [list(seeds[f * FOLD_SIZE:(f + 1) * FOLD_SIZE]) for f in range(FOLDS)]
    check_folds(out, seeds, "folds_of")
    return out


def check_folds(groups, seeds, where):
    """The folds partition the seeds: every seed in exactly one, none invented."""
    flat = [s for group in groups for s in group]
    if sorted(flat) != sorted(seeds) or len(set(flat)) != len(flat):
        missing = sorted(set(seeds) - set(flat))
        extra = sorted(set(flat) - set(seeds))
        raise Fail("%s: %d fold(s) covering %d seed(s) of %d, %d missing (%s), %d not a"
                   " seed (%s), %d duplicated"
                   % (where, len(groups), len(set(flat)), len(seeds), len(missing),
                      missing[:4], len(extra), extra[:4], len(flat) - len(set(flat))))
    return len(groups)


def check_held_out(train, held, where):
    """No seed of the read-out fold is in the fit, which is what "leave seeds out" means.

    The one check that says the R squared below is held out at all.  Without it a fold
    assignment off by one would fit the rows it reads out and every number would be high
    for that reason alone.
    """
    both = sorted({r["seed"] for r in train} & {r["seed"] for r in held})
    if both:
        raise Fail("%s: %d seed(s) are in the fit and in the read-out, e.g. %s"
                   % (where, len(both), both[:4]))
    return len(held)


def standardiser(matrix):
    """(mean, sd) of the fit rows, with a zero sd replaced by 1.

    A constant column -- `features.py` names 150 of them dead on this data -- has sd 0, and
    dividing by it would make every coefficient NaN and spread the NaN to all of them.
    Replacing the sd by 1 leaves that column at exactly 0 after centring, so the penalty
    holds its coefficient at 0 and the column contributes nothing instead of everything.
    """
    mean = matrix.mean(axis=0)
    sd = matrix.std(axis=0)
    return mean, np.where(sd > 0.0, sd, 1.0)


def design(matrix, mean, sd):
    """The standardised rows with a leading column of ones for the intercept."""
    return np.concatenate([np.ones((len(matrix), 1)), (matrix - mean) / sd], axis=1)


def ridge(mat, y, lam=LAMBDA):
    """The closed-form ridge solution, with the intercept left unpenalised.

    `numpy.linalg.solve` on the normal equations and no iteration anywhere, so two fits of
    the same rows in the same order give the same bytes; part 7 is that check.  The penalty
    skips the intercept because penalising it would pull the prediction towards 0 rather
    than towards the fit rows' own mean, and 0 is not a meaningful cost.
    """
    penalty = np.eye(mat.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(mat.T @ mat + lam * penalty, mat.T @ y)


def full_columns(rows):
    """The whole feature vector of each row, as a matrix."""
    return np.stack([r["vec"] for r in rows])


def cell_columns(rows):
    """One-hot columns over the five scalar cells, in CELLS order.

    The information ADR 0004 line 5 permits a policy to read, as a design matrix, so that
    the third predictor is the same arithmetic as the second on fewer columns and the two
    differ in their inputs alone.
    """
    return np.eye(len(CELLS))[[CELLS.index(r["cell"]) for r in rows]]


def per_arm_fit(train, held, columns_of, y_train, lam=LAMBDA):
    """(prediction per held row, {arm: coefficients}, (mean, sd)) one fit per interval.

    A separate fit per interval is what lets the three intervals have different slopes on
    the same input, which is the whole point of predicting a cost per interval: one pooled
    fit with the interval as a column could only shift the level.  The standardisation is
    taken over ALL the fit rows, not per interval, so the three coefficient sets are in one
    scale and comparable column by column.
    """
    mean, sd = standardiser(columns_of(train))
    out = np.zeros(len(held))
    coefficients = {}
    for arm in ARMS:
        mine = [i for i, r in enumerate(train) if r["arm"] == arm]
        theirs = [i for i, r in enumerate(held) if r["arm"] == arm]
        if not mine:
            raise Fail("no fit row chose %.3f s, so its coefficients cannot exist" % arm)
        mat = design(columns_of([train[i] for i in mine]), mean, sd)
        coefficients[arm] = ridge(mat, y_train[mine], lam)
        if theirs:
            got = design(columns_of([held[i] for i in theirs]), mean, sd) \
                @ coefficients[arm]
            for slot, value in zip(theirs, got):
                out[slot] = value
    return out, coefficients, (mean, sd)


def cell_arm_mean(train, held, y_train):
    """(prediction per held row, how many held rows fell on an unseen (cell, arm)).

    The mean of the target over the fit rows of the same (five-cell state, interval), which
    is the strongest thing the ADR 0004 scalars plus the interval can say: 15 numbers.  A
    held row whose combination no fit row carries gets the fit rows' grand mean, and the
    count of those rows is returned rather than hidden, because a predictor that fell back
    often would be being read as something it is not.
    """
    sums = collections.defaultdict(float)
    counts = collections.Counter()
    for r, value in zip(train, y_train):
        sums[(r["cell"], r["arm"])] += value
        counts[(r["cell"], r["arm"])] += 1
    grand = float(y_train.mean())
    out, missing = np.zeros(len(held)), 0
    for i, r in enumerate(held):
        key = (r["cell"], r["arm"])
        if counts[key]:
            out[i] = sums[key] / counts[key]
        else:
            out[i] = grand
            missing += 1
    return out, missing


def r_squared(y, prediction):
    """1 - SSres/SStot against the READ-OUT rows' own mean.

    Their own mean and not the fit rows': the question is how much of the variation AMONG
    the held-out decisions the inputs explain, and a predictor that only got the level
    right would otherwise score for that.  A negative value is possible and is printed as
    it comes -- it means the predictor is worse on these rows than their own mean.
    """
    centred = y - y.mean()
    total = float(centred @ centred)
    if total <= 0.0:
        raise Fail("the %d read-out targets have no variance" % len(y))
    residual = y - prediction
    return 1.0 - float(residual @ residual) / total


def check_permuted(got, where, bar=0.05):
    """A fit on permuted targets explains nothing held out.

    WHAT THIS DOES AND DOES NOT SHOW, stated here because the obvious reading is wrong.
    It shows that part 3's R squared is not an artefact of the ARITHMETIC: the only thing
    the permutation removes is the link between a fit row's inputs and its own target, and
    once that is gone the prediction explains nothing, so no part of the number came from
    the read-out targets reaching the fit through the code.  It is NOT a check against a
    leaked future value: a cell that carried the target itself would also lose its link
    under the permutation, so a legitimate input and a leaked one collapse alike.  And a
    read-out row accidentally left in the fit would have carried a PERMUTED target there,
    so it could not have been memorised either; this bar cannot detect that, which is what
    `check_held_out` is for.
    What rules out a leaked future value is structural and not statistical:
    `observe.observation` builds the record from the state, the resolution and the history
    as they stand at the decision's own `ts_ms` and from nothing else, and the coefficient
    table in part 3 names the cells the fit leans on so a reader can judge them.
    The size of the negative value is a second reading: it is how much a ridge on all of
    these features over-fits four folds of these rows when there is nothing to find.
    """
    if got > bar:
        raise Fail("%s: a fit on permuted targets explains %.4f of the held-out variance,"
                   " over the bar %.2f, so part 3's R squared is partly an artefact of the"
                   " arithmetic and not of the inputs" % (where, got, bar))
    return got


def mean_se(values):
    """(mean, standard error) over a handful of numbers, 0.0 when there is only one."""
    if len(values) < 2:
        return (float(values[0]) if values else 0.0), 0.0
    return statistics.fmean(values), statistics.stdev(values) / len(values) ** 0.5


def ranking_of(mine, rho):
    """(logged arm, predicted cheapest arm, the logged arm's predicted RANK, y, seed).

    One tuple per read-out row, pooled.  Every row appears once, in the fold whose seeds
    hold it, so the prediction on it was made by a fit that never saw its run.  Pooled
    rather than per fold because part 4's standard error is clustered on RUNS and a run
    lies inside one fold.

    The rank is 0 when the logged interval is the one the ridge calls cheapest for that
    row, 1 for the middle and 2 for the dearest.  It is here because the rank-0 rows alone
    cannot separate "this interval was the right one" from "this ROW was going to be
    cheap": rank 0 against rank 2 is the comparison that reads the ordering.
    """
    out = []
    for group in folds_of(SEEDS):
        held = [r for r in mine if r["seed"] in group]
        train = [r for r in mine if r["seed"] not in group]
        check_held_out(train, held, "ranking fold %s" % group[0])
        y_train = targets(train, 1, rho)
        mean, sd = standardiser(full_columns(train))
        stacked = np.stack(
            [ridge(design(full_columns([r for r in train if r["arm"] == arm]), mean, sd),
                   y_train[[i for i, r in enumerate(train) if r["arm"] == arm]])
             for arm in ARMS], axis=1)
        guess = design(full_columns(held), mean, sd) @ stacked
        order = guess.argsort(axis=1)
        for r, row, y in zip(held, order, targets(held, 1, rho)):
            ranked = [ARMS[int(i)] for i in row]
            out.append((r["arm"], ranked[0], ranked.index(r["arm"]), float(y),
                        r["seed"]))
    return out


# ---------------------------------------------------------------------------
# the coefficient digest, which is also this script's own second process
# ---------------------------------------------------------------------------
def digest_rows(rows):
    """{p: (sha256 of the three coefficient sets, their byte count)} for a full-data fit.

    The fit is the one-period target over every owned period of the cell, with no folds, so
    the digest covers a fit a reader can re-run in one command.  The bytes hashed are the
    coefficient arrays' own, concatenated in ARMS order, which is the thing that has to be
    identical for a closed-form fit to be called reproducible.
    """
    out = {}
    for p in P_CELLS:
        mine = [r for r in rows if r["p"] == p]
        rho = rho_of(mine)
        _pred, coefficients, _scale = per_arm_fit(mine, mine[:1], full_columns,
                                                  targets(mine, 1, rho))
        raw = b"".join(coefficients[arm].tobytes() for arm in ARMS)
        out[p] = (hashlib.sha256(raw).hexdigest(), len(raw))
    return out


def digest_main():
    """`--digest`: print the coefficient digests and nothing else, for the second process.

    A separate process and not a second call in this one, because what part 7 has to show
    is that the fit does not depend on anything this process happens to hold: the same
    files, read again, solved again, in a fresh interpreter.
    """
    rows = []
    for p in P_CELLS:
        for seed in SEEDS:
            mine, _run = table_of_run(p, seed)
            rows.extend(mine)
    for p, (sha, size) in sorted(digest_rows(rows).items()):
        print("p%s %s %d" % (p, sha, size), flush=True)
    return 0


def second_process():
    """The `--digest` output of a fresh interpreter, as a list of lines.

    The absolute path of this file is used because a subprocess inherits none of this
    process's assumptions about a working directory, and byte compilation is turned off in
    the child as well as the parent so that no cache directory appears beside this script.
    """
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), "--digest"],
                          capture_output=True, text=True, env=env, check=False)
    if done.returncode != 0:
        raise Fail("the second process exited %d: %s"
                   % (done.returncode, done.stderr.strip().splitlines()[-1:]))
    return [line for line in done.stdout.splitlines() if line.strip()]


def check_same_digests(left, right, where):
    """Two processes produced the same coefficient bytes.

    Named line by line rather than as one boolean, so that a difference says WHICH cell's
    fit moved; a closed-form solve that differed between two processes of one machine would
    mean the solver itself is not deterministic here, and every number in this file would
    then be one draw.
    """
    if len(left) != len(right):
        raise Fail("%s: %d digest line(s) against %d" % (where, len(left), len(right)))
    bad = [(a, b) for a, b in zip(left, right) if a != b]
    if bad:
        raise Fail("%s: %d of %d lines differ, first %r against %r"
                   % (where, len(bad), len(left), bad[0][0], bad[0][1]))
    return len(left)


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


def quant(values, q):
    """One order statistic by position, with no interpolation between two samples."""
    ordered = sorted(values)
    return ordered[min(int(q * (len(ordered) - 1) + 0.5), len(ordered) - 1)]


def part0():
    """Part 0, and the table every later part reads: (rows, {(p, seed): run})."""
    head("0  WHAT IS READ, AND WHAT ONE ROW IS")
    say("stage            %s, cells p %s" % (CH.STAGE, " and p ".join(P_CELLS)))
    say("seeds            %d to %d, %d runs; this workflow reads no other emulator seed"
        % (SEEDS[0], SEEDS[-1], len(SEEDS) * len(P_CELLS)))
    say("seed guards      charges.load_run refuses an emulator seed above %d and"
        % CH.MAX_LOG_SEED)
    say("                 port.check_seeds refuses one above %d; part 6 shows both firing"
        % PORT.MAX_SEED)
    say("files per run    debug/node_dirs/h*/semindex/events.jsonl, and nothing else")
    say("not read here    no label row, no run meta.json or results.json, and no replay")
    say("                 simulation of any kind")
    say()
    say("one row          = one OWNED decision period of charges.py")
    say("  vec             the feature vector of the decision's own record, %d cells,"
        % FE.LENGTH)
    say("                  name list sha256 %s" % FE.NAMES_DIGEST)
    say("  arm             action.interval_s, one of %s" % list(ARMS))
    say("  tau             the period length in seconds")
    say("  c               the period cost in ms of SCORE, charges.py's cost_ms / N")
    say("  cell            the five-cell scalar state, charges.cell: %s" % list(CELLS))
    say("  next            the next decision of the same pair, as its position in that")
    say("                  pair's chain of owned periods; the pair's last owned period")
    say("                  has none, because the decision after it is the pair's last and")
    say("                  owns no period itself")
    rows, runs = [], {}
    for p in P_CELLS:
        for seed in SEEDS:
            mine, run = table_of_run(p, seed)
            rows.extend(mine)
            runs[(p, seed)] = run
    check_owned_count(rows, list(runs.values()), "the table against the runs")
    check_cost_closes(rows, list(runs.values()), "the table's cost against the scorer's")
    say()
    say("every run's identity and placement asserted by charges.py before the table was")
    say("built, and the table's own cost re-closed against the scorer afterwards")
    return rows, runs


def part1(by_p, runs):
    """Part 1: the decision table's counts."""
    head("1  THE DATA AS A SEMI-MARKOV DECISION TABLE")
    say("Every owned period of every run, with the decision's input at its left edge.  The")
    say("decisions NOT in the table are the 8 pairs' last decisions of each run, one per")
    say("pair per run: they own no period, so there is no tau and no c to put on them.")
    say()
    say("  p     decisions  owned rows  not a row  seconds   cost score ms")
    for p in P_CELLS:
        mine = by_p[p]
        tails = sum(1 for seed in SEEDS for per in runs[(p, seed)]["flat"]
                    if per["kind"] == "tail")
        say("  %-5s %10d %11d %10d %8.0f %14.0f"
            % (p, len(mine) + tails, len(mine), tails,
               sum(r["tau"] for r in mine), sum(r["c"] for r in mine)))
    say()
    say("  rows per interval and per cell")
    say("  %-5s %-12s %9s %11s %7s %7s %7s %8s"
        % ("p", "interval", *CELLS, "all"))
    for p in P_CELLS:
        mine = by_p[p]
        for arm in ARMS:
            got = [r for r in mine if r["arm"] == arm]
            counts = collections.Counter(r["cell"] for r in got)
            say("  %-5s %12.3f %9d %11d %7d %7d %7d %8d"
                % (p, arm, *[counts[one] for one in CELLS], len(got)))
        counts = collections.Counter(r["cell"] for r in mine)
        say("  %-5s %-12s %9d %11d %7d %7d %7d %8d"
            % (p, "all", *[counts[one] for one in CELLS], len(mine)))
    say()
    say("  the next decision of the pair, by the cell the decision itself was taken in.")
    say("  A row whose next decision is the pair's LAST is counted under `last`: it has a")
    say("  next decision, but that decision owns no period, so a multi-period target")
    say("  stops there.")
    say("  %-5s %-12s %9s %11s %7s %7s %7s %8s"
        % ("p", "own cell", *CELLS, "last"))
    for p in P_CELLS:
        for one in CELLS:
            got = [r for r in by_p[p] if r["cell"] == one]
            nxt = collections.Counter(
                r["chain"][r["next_pos"]]["cell"] if r["next_pos"] is not None else "last"
                for r in got)
            say("  %-5s %-12s %9d %11d %7d %7d %7d %8d"
                % (p, one, *[nxt[two] for two in CELLS], nxt["last"]))
    say()
    say("  how many of the K periods a target would sum really exist, per K")
    say("  %-5s %3s %10s %10s %10s" % ("p", "K", "rows", "full", "truncated"))
    for p in P_CELLS:
        for k in (1,) + KS:
            mine = by_p[p]
            full = sum(1 for r in mine if r["pos"] + k <= len(r["chain"]))
            say("  %-5s %3d %10d %10d %10d"
                % (p, k, len(mine), full, len(mine) - full))


def part2(by_p):
    """Part 2: rho, pooled and per interval.  Returns {p: rho}."""
    head("2  THE LOGGING POLICY'S OWN COST PER SECOND")
    say("rho = sum(c) / sum(tau) over the owned periods, in score ms per second: what one")
    say("second of this stage cost under the policy that drew the three intervals")
    say("uniformly.  The pooled ratio is the quantity; the mean of the %d per-run ratios"
        % len(SEEDS))
    say("and its standard error are beside it, so the spread between runs is visible.")
    say()
    say("  p     pooled rho   mean of per-run rho   SE over %d runs      min      max"
        % len(SEEDS))
    rho_of_p = {}
    for p in P_CELLS:
        mine = by_p[p]
        rho_of_p[p] = rho_of(mine)
        per_run = [rho_of([r for r in mine if r["seed"] == s]) for s in SEEDS]
        mean, se = mean_se(per_run)
        say("  %-5s %11.4f %21.4f %16.4f %8.4f %8.4f"
            % (p, rho_of_p[p], mean, se, min(per_run), max(per_run)))
        check_y_sums_to_zero(mine, rho_of_p[p], "p%s" % p)
    say()
    say("  the one-period target sums to 0 over each cell's own rows, asserted to 1e-6 of")
    say("  the summed absolute cost, which is what says the target is centred on rho")
    say()
    say("Within each chosen interval.  THIS IS NOT A CAUSAL EFFECT OF THE INTERVAL.  The")
    say("logging policy drew the interval before anything about the period was known, so")
    say("the three row sets are three different mixtures of the five cells and of clock")
    say("position, and the difference between their rates mixes the cost of a cell with")
    say("the cost of an interval.  charges.py part 6 carries the same three pooled numbers")
    say("and says the same thing; they are recomputed here, not copied.")
    say()
    say("  p     interval   rows   pooled rho   mean per-run rho   SE over runs")
    for p in P_CELLS:
        for arm in ARMS:
            got = [r for r in by_p[p] if r["arm"] == arm]
            per_run = [rho_of([r for r in got if r["seed"] == s]) for s in SEEDS]
            mean, se = mean_se(per_run)
            say("  %-5s %8.3f %6d %12.4f %18.4f %14.4f"
                % (p, arm, len(got), rho_of(got), mean, se))
    return rho_of_p


def part3(by_p, rho_of_p):
    """Part 3: the held-out R squared of the three predictors.  Returns the table."""
    head("3  HOW MUCH OF WHAT FOLLOWS A DECISION THE INPUTS EXPLAIN")
    say("Leave-seeds-out: %d folds of %d seeds, fold f holding seeds 4f+1 to 4f+4, so no"
        % (FOLDS, FOLD_SIZE))
    say("decision of a run is ever in the fit and the read-out at once.  Targets, all in")
    say("score ms: y(K) = the sum of c - rho * tau over this period and the next K-1 of")
    say("the same pair, truncated at the end of the pair's chain.  K = 1 is the one-period")
    say("target.  rho is the cell's pooled rho from part 2.")
    say()
    say("Three predictors, each fitted on the four fit folds and read out on the fifth:")
    say("  cell mean    the mean of y over the fit rows of the same (cell, interval): 15")
    say("               numbers, the most the ADR 0004 scalars plus the interval can say")
    say("  ridge full   closed-form ridge, penalty %.1f on standardised features, a"
        % LAMBDA)
    say("               separate coefficient set per interval, on all %d cells" % FE.LENGTH)
    say("  ridge cells  the same arithmetic on the five cell indicators alone")
    say("Held-out R squared is 1 - SSres/SStot against the read-out rows' own mean, and")
    say("the SE is over the %d folds.  `fallbacks` counts read-out rows whose (cell," % FOLDS)
    say("interval) no fit fold carried, which take the fit rows' grand mean instead.")
    say()
    say("  %-5s %2s %-12s %10s %10s %6s %10s"
        % ("p", "K", "predictor", "R2 mean", "SE", "folds", "fallbacks"))
    table = {}
    coefficients_of = {}
    for p in P_CELLS:
        mine = by_p[p]
        rho = rho_of_p[p]
        for k in (1,) + KS:
            scores = {"cell mean": [], "ridge full": [], "ridge cells": []}
            fallbacks = 0
            for group in folds_of(SEEDS):
                held = [r for r in mine if r["seed"] in group]
                train = [r for r in mine if r["seed"] not in group]
                check_held_out(train, held, "p%s K%d fold %s" % (p, k, group[0]))
                y, y_train = targets(held, k, rho), targets(train, k, rho)
                flat, missing = cell_arm_mean(train, held, y_train)
                fallbacks += missing
                scores["cell mean"].append(r_squared(y, flat))
                full, coefficients, scale = per_arm_fit(train, held, full_columns, y_train)
                scores["ridge full"].append(r_squared(y, full))
                cells, _c, _s = per_arm_fit(train, held, cell_columns, y_train)
                scores["ridge cells"].append(r_squared(y, cells))
                if k == 1 and group[0] == 1:
                    coefficients_of[p] = (coefficients, scale)
            for name in ("cell mean", "ridge full", "ridge cells"):
                mean, se = mean_se(scores[name])
                table[(p, k, name)] = (mean, se, scores[name])
                say("  %-5s %2d %-12s %10.4f %10.4f %6d %10s"
                    % (p, k, name, mean, se, len(scores[name]),
                       "%d" % fallbacks if name == "cell mean" else ""))
    say()
    say("The comparison a plan rests on, stated plainly: the full vector against the five")
    say("cells, as the PAIRED difference of the two held-out R squared values inside each")
    say("fold, so the fold-to-fold level cancels.  A positive difference means the full")
    say("vector explains more.")
    say()
    say("  %-5s %2s %12s %14s %13s %9s %s"
        % ("p", "K", "R2 full", "R2 five cells", "paired diff", "SE", "diff > 2 SE"))
    for p in P_CELLS:
        for k in (1,) + KS:
            full = table[(p, k, "ridge full")]
            cells = table[(p, k, "ridge cells")]
            diffs = [a - b for a, b in zip(full[2], cells[2])]
            mean, se = mean_se(diffs)
            say("  %-5s %2d %12.4f %14.4f %13.4f %9.4f %s"
                % (p, k, full[0], cells[0], mean, se,
                   "yes" if abs(mean) > 2.0 * se else "no"))
    say()
    say("THE ONE SCALAR OF OUT-OF-FOLD INFORMATION IN THE TARGET, AS A NUMBER.  rho is the")
    say("cell's pooled rate over all %d seeds, which is the target's own definition, so a"
        % len(SEEDS))
    say("read-out row's target carries a rate estimated partly on its own seed.  The two")
    say("ridges below are refitted with rho recomputed on each fold's FIT rows alone, which")
    say("is what that reading costs rather than part 2's spread standing in for it.")
    say()
    say("  %-5s %2s %-12s %13s %15s %9s"
        % ("p", "K", "predictor", "pooled rho", "fit-fold rho", "move"))
    for p in P_CELLS:
        mine = by_p[p]
        for k in (1,) + KS:
            got = {"ridge full": [], "ridge cells": []}
            for group in folds_of(SEEDS):
                held = [r for r in mine if r["seed"] in group]
                train = [r for r in mine if r["seed"] not in group]
                # rho from the FIT rows of this fold only, so no read-out seed's cost or
                # seconds reach the rate its own target is netted against.
                own = rho_of(train)
                y, y_train = targets(held, k, own), targets(train, k, own)
                for name, columns in (("ridge full", full_columns),
                                      ("ridge cells", cell_columns)):
                    pred, _c, _s = per_arm_fit(train, held, columns, y_train)
                    got[name].append(r_squared(y, pred))
            for name in ("ridge full", "ridge cells"):
                mean, _se = mean_se(got[name])
                table[(p, k, name + ", fit-fold rho")] = (mean, _se, got[name])
                say("  %-5s %2d %-12s %13.4f %15.4f %9.4f"
                    % (p, k, name, table[(p, k, name)][0], mean,
                       mean - table[(p, k, name)][0]))
    say()
    say("WHAT THAT DIFFERENCE IS NOT.  Three things it mixes in, named because the size of")
    say("the jump invites the opposite reading:")
    say("- The vector carries the backlog (`o*_hi`, `o*_missing`), the age of the oldest")
    say("  exchange it remembers and the pair's own exchange count, all of which move")
    say("  through a run, so part of what it explains is WHERE IN THE RUN the decision")
    say("  sits and how much work is outstanding, not what to do.  The coefficient table")
    say("  below says which cells it really leans on.  The table before it repeats the two")
    say("  ridges over the read-out rows whose whole")
    say("  K-period continuation exists, which removes the share that comes from")
    say("  predicting the truncation.")
    say("- Explaining what followed is not the effect of the interval.  Every continuation")
    say("  in these rows is the logging policy's.")
    say("- The cost a decision causes at the DOWNSTREAM pair is charged to that pair and is")
    say("  in no target here (charges.py part 9).")
    say()
    say("  R squared over the read-out rows whose whole K-period continuation exists")
    say("  %-5s %2s %-12s %8s %10s %10s" % ("p", "K", "predictor", "rows", "R2", "SE"))
    for p in P_CELLS:
        mine = by_p[p]
        rho = rho_of_p[p]
        for k in KS:
            got = {"ridge full": [], "ridge cells": []}
            kept = 0
            for group in folds_of(SEEDS):
                train = [r for r in mine if r["seed"] not in group]
                held = [r for r in mine if r["seed"] in group
                        and r["pos"] + k <= len(r["chain"])]
                kept += len(held)
                y, y_train = targets(held, k, rho), targets(train, k, rho)
                for name, columns in (("ridge full", full_columns),
                                      ("ridge cells", cell_columns)):
                    pred, _c, _s = per_arm_fit(train, held, columns, y_train)
                    got[name].append(r_squared(y, pred))
            for name in ("ridge full", "ridge cells"):
                mean, se = mean_se(got[name])
                say("  %-5s %2d %-12s %8d %10.4f %10.4f" % (p, k, name, kept, mean, se))
    say()
    say("The cells the full fit leans on: the six largest standardised coefficients of the")
    say("one-period fit of the fold that holds out seeds 1 to %d, per interval.  A"
        % FOLD_SIZE)
    say("standardised coefficient is the score ms the prediction moves per one fit-fold")
    say("standard deviation of that cell, so the names say WHICH inputs carry the")
    say("difference above and a reader can judge whether they are state or clock.")
    for p in P_CELLS:
        coefficients, (_mean, _sd) = coefficients_of[p]
        for arm in ARMS:
            w = coefficients[arm][1:]
            order = np.argsort(-np.abs(w))[:6]
            say("  p %s %8.3f s  %s"
                % (p, arm, ", ".join("%s %+.1f" % (FE.NAMES[i], w[i]) for i in order)))
    return table


def part4(by_p, rho_of_p):
    """Part 4: the ranking reading, with a run-clustered standard error."""
    head("4  THE SAME QUESTION AS A RANKING, WHICH IS ALL A DECIDER NEEDS")
    say("A decider does not need the cost of an interval, only the ORDER of the three.  So:")
    say("on the read-out rows, give the ridge all three intervals' predictions of the")
    say("one-period target and take its argmin.  The logged interval happens to equal that")
    say("argmin on some rows; compare the realised y of THOSE rows with the realised y of")
    say("all read-out rows.  y is a cost, so a negative difference is the direction a")
    say("decider would want.")
    say()
    say("WHAT THIS IS.  A one-period, logged-policy-continuation reading.  It says what")
    say("followed the subset of logged decisions the predictor would have agreed with,")
    say("while every decision after them was still drawn uniformly.  It says NOTHING about")
    say("the replay score of a policy that took the argmin at every decision: that")
    say("policy's own continuation is itself, and no replay is run here.  The standard")
    say("error is clustered on runs -- the %d per-run differences, SE = sd / sqrt(%d) --"
        % (len(SEEDS), len(SEEDS)))
    say("because two decisions of one run share a record stream and are not independent.")
    say()
    say("The mean of y over ALL read-out rows of a cell is 0 to rounding, because rho is")
    say("that cell's own rate and y is centred on it (part 2), so the `all` column below")
    say("is a computed 0 and not a typed one.")
    say()
    say("  p     matched rows   share   mean y matched   mean y all   difference      SE")
    ranked_of = {p: ranking_of(by_p[p], rho_of_p[p]) for p in P_CELLS}
    for p in P_CELLS:
        got = ranked_of[p]
        realised = np.array([y for _lg, _pk, _rk, y, _sd in got])
        matched = np.array([lg == pk for lg, pk, _rk, _y, _sd in got])
        seeds = np.array([seed for _lg, _pk, _rk, _y, seed in got])
        per_run = []
        for s in SEEDS:
            here = seeds == s
            mine = here & matched
            if not mine.any():
                continue
            per_run.append(float(realised[mine].mean() - realised[here].mean()))
        mean, se = mean_se(per_run)
        say("  %-5s %13d %7.4f %16.2f %12.2f %12.2f %7.2f"
            % (p, int(matched.sum()), float(matched.mean()),
               float(realised[matched].mean()), float(realised.mean()), mean, se))
    say()
    say("The control that says whether the ORDER carries anything: the same realised y by")
    say("the RANK the ridge gave the interval the log happened to draw.  Rank 0 is the")
    say("subset above.")
    say()
    say("WHY THIS SEPARATES THE INTERVAL FROM THE ROW.  The logging policy drew the")
    say("interval uniformly and independently of the state, so a row falls into rank 0,")
    say("1 or 2 with probability one third whatever its features are: the three rank")
    say("groups are three samples of the SAME state distribution, which is why their")
    say("counts come out near equal below.  A difference in realised y between rank 0 and")
    say("rank 2 is therefore the interval and not the row.  Had the predictor only been")
    say("finding cheap ROWS, all three ranks would hold the same mixture of cheap and dear")
    say("rows and the three means would agree.")
    say()
    say("  p     rank of the logged interval   rows   mean y   per-run diff vs all    SE")
    for p in P_CELLS:
        got = ranked_of[p]
        realised = np.array([y for _lg, _pk, _rk, y, _sd in got])
        ranks = np.array([rank for _lg, _pk, rank, _y, _sd in got])
        seeds = np.array([seed for _lg, _pk, _rk, _y, seed in got])
        for rank, name in ((0, "0 (the predicted cheapest)"), (1, "1 (the middle)"),
                           (2, "2 (the predicted dearest)")):
            pick = ranks == rank
            per_run = [float(realised[(seeds == s) & pick].mean()
                             - realised[seeds == s].mean())
                       for s in SEEDS if ((seeds == s) & pick).any()]
            mean, se = mean_se(per_run)
            say("  %-5s %-29s %6d %8.2f %21.2f %5.2f"
                % (p, name, int(pick.sum()), float(realised[pick].mean()), mean, se))
    say()
    say("  which interval the ridge calls cheapest, and which the log drew")
    say("  %-5s %-10s %11s %10s %10s"
        % ("p", "interval", "predicted", "logged", "matched"))
    for p in P_CELLS:
        got = ranked_of[p]
        for arm in ARMS:
            say("  %-5s %10.3f %11d %10d %10d"
                % (p, arm, sum(1 for _lg, pk, _rk, _y, _sd in got if pk == arm),
                   sum(1 for lg, _pk, _rk, _y, _sd in got if lg == arm),
                   sum(1 for lg, pk, _rk, _y, _sd in got if lg == arm and pk == arm)))
    say()
    say("  The argmin returns one of the three intervals on every row above and nothing")
    say("  else, which the `predicted` column adding to the row count is what shows.")


def part5(by_p, rho_of_p):
    """Part 5: the permuted-target control.  Returns {(p, predictor): R squared}."""
    head("5  THE PERMUTED-TARGET CONTROL")
    say("The same leave-seeds-out fit with the FIT folds' targets permuted among their own")
    say("rows, so every input keeps its value and loses its target.  The permutation uses")
    say("numpy.random.default_rng(%d), a fixed seed, so this part reproduces."
        % PERMUTE_SEED)
    say()
    say("WHAT IT SHOWS: that part 3's R squared is not an artefact of the arithmetic.  The")
    say("only thing the permutation removes is the link between a fit row's inputs and its")
    say("own target, and the prediction then explains nothing held out, so no part of part")
    say("3's number came from the read-out targets reaching the fit through the code.  The")
    say("negative value is also a reading of how much a %d-feature ridge over-fits four"
        % FE.LENGTH)
    say("folds of these rows when there is nothing to find.")
    say()
    say("WHAT IT DOES NOT SHOW: that no cell of the vector carries a leaked future value.")
    say("Such a cell would lose its link under the permutation too, so a legitimate input")
    say("and a leaked one collapse alike here.  What rules a leak out is structural:")
    say("observe.observation builds the record from the state, the resolution and the")
    say("history as they stand at the decision's own ts_ms and from nothing else, and part")
    say("3's coefficient table names the cells the fit leans on.  Nor does it detect a")
    say("read-out row left in the fit: that row would carry a PERMUTED target there and")
    say("could not be memorised.  check_held_out is what detects that, and part 6 shows it")
    say("firing.")
    say()
    say("  %-5s %-12s %10s %10s %12s" % ("p", "predictor", "R2 mean", "SE", "bar"))
    out = {}
    for p in P_CELLS:
        mine = by_p[p]
        rho = rho_of_p[p]
        got = {"ridge full": [], "ridge cells": []}
        rng = np.random.default_rng(PERMUTE_SEED)
        for group in folds_of(SEEDS):
            held = [r for r in mine if r["seed"] in group]
            train = [r for r in mine if r["seed"] not in group]
            shuffled = targets(train, 1, rho)
            rng.shuffle(shuffled)
            y = targets(held, 1, rho)
            for name, columns in (("ridge full", full_columns),
                                  ("ridge cells", cell_columns)):
                pred, _c, _s = per_arm_fit(train, held, columns, shuffled)
                got[name].append(r_squared(y, pred))
        for name in ("ridge full", "ridge cells"):
            mean, se = mean_se(got[name])
            out[(p, name)] = mean
            check_permuted(mean, "p%s %s, permuted targets" % (p, name))
            say("  %-5s %-12s %10.4f %10.4f %12.2f" % (p, name, mean, se, 0.05))
    return out


def part6(rows, runs, by_p, rho_of_p, table):
    """Part 6: every check, shown failing on a mutated input."""
    head("6  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT")
    say("The mutation is of the INPUT, inside this process: a run, a record source, a")
    say("table, a fold list or an R squared made wrong on purpose.  A check that cannot be")
    say("made to fail by any input is named as such at the end.")
    say()
    fired = []

    def shows(name, call):
        """Run a check on a mutated input and require it to refuse."""
        try:
            call()
        except (Fail, SystemExit, CH.Fail, FE.Refuse) as exc:
            say("  %-26s FIRES: %s" % (name, exc))
            return True
        say("  %-26s DID NOT FIRE -- this check cannot fail on this input" % name)
        return False

    base_events, base_t0 = CH.load_run(P_CELLS[0], 1)
    sample = by_p[P_CELLS[0]]
    all_runs = list(runs.values())
    base_found = records_of_run(base_events)
    base_chains = CH.periods(base_events, base_t0)
    base_pair = sorted(base_chains)[0]
    base_per = next(per for per in base_chains[base_pair] if per["kind"] == "owned")
    base_key = (base_pair[0], base_pair[1], base_per["start"])

    def with_found(changed):
        """The record dict with one key changed or removed, as `matched_row`'s input."""
        out = dict(base_found)
        if changed is None:
            del out[base_key]
        else:
            out[base_key] = changed
        return matched_row(out, base_pair, base_per, base_t0, "a mutated record source")

    def crossed_cell():
        """One period's row re-bucketed into a cell that is not the period's.

        The row's own resolution is rewritten, which is what a record source handing back
        another row's record would look like from here.  The replacement is chosen so that
        `charges.cell` of it differs from the period's cell whichever cell that is.
        """
        row, record = base_found[base_key]
        for action, timeouts in ((dict(row["action"], trigger="answered", empty=True), 0),
                                 (dict(row["action"], trigger="timed_out"), 2)):
            bent = dict(row, action=action,
                        state=dict(row["state"], consecutive_poll_timeouts=timeouts))
            if CH.cell(bent) != base_per["cell"]:
                return with_found((bent, record))
        raise Fail("no rewritten resolution lands outside the period's own cell")

    def short_source(node_rows):
        """A record source that drops its last record: one record fewer than rows."""
        made = list(FE.records_from_logged_rows(node_rows))
        return iter(made[:-1])

    def crossed_source(node_rows):
        """A record source that puts each record on the next pair's name.

        What a filter change inside `observe.from_logged_rows` would look like from here:
        the records still come one per row, in order, but the name beside them no longer
        belongs to the row the zip pairs them with.
        """
        made = list(FE.records_from_logged_rows(node_rows))
        names = sorted({nb for nb, _record in made})
        return iter([(names[(names.index(nb) + 1) % len(names)], record)
                     for nb, record in made])

    fired.append(shows("records/source too short",
                       lambda: records_of_run(base_events, short_source)))
    fired.append(shows("records/name crossed",
                       lambda: records_of_run(base_events, crossed_source)))
    fired.append(shows("records/no row at the start", lambda: with_found(None)))
    fired.append(shows("records/interval crossed", lambda: with_found(
        (dict(base_found[base_key][0],
              action=dict(base_found[base_key][0]["action"], interval_s=1.234)),
         base_found[base_key][1]))))
    fired.append(shows("records/cell crossed", crossed_cell))
    fired.append(shows("owned count",
                       lambda: check_owned_count(rows[1:], all_runs, "one row short")))
    fired.append(shows("cost closes", lambda: check_cost_closes(
        [dict(r, c=r["c"] + 1.0) if i == 0 else r for i, r in enumerate(rows)],
        all_runs, "one period's cost moved")))
    fired.append(shows("y sums to zero", lambda: check_y_sums_to_zero(
        sample, rho_of_p[P_CELLS[0]] * 1.001, "a rho 0.1 % off")))
    fired.append(shows("folds partition", lambda: check_folds(
        [[1, 2], [2, 3]], SEEDS, "two folds sharing a seed")))
    fired.append(shows("held out", lambda: check_held_out(sample, sample,
                                                          "the fit read out on itself")))
    fired.append(shows("permuted control", lambda: check_permuted(
        table[(P_CELLS[0], 1, "ridge full")][0],
        "the real R squared handed to the permuted control")))
    fired.append(shows("rho over no seconds",
                       lambda: rho_of([dict(r, tau=0.0) for r in sample[:3]])))
    fired.append(shows("R2 of a constant target",
                       lambda: r_squared(np.zeros(5), np.zeros(5))))
    fired.append(shows("per-arm fit, no fit row",
                       lambda: per_arm_fit([r for r in sample if r["arm"] == ARMS[0]],
                                           sample[:1], cell_columns,
                                           np.zeros(sum(1 for r in sample
                                                        if r["arm"] == ARMS[0])))))
    fired.append(shows("digest lines differ", lambda: check_same_digests(
        ["p0.2 aaa 1"], ["p0.2 bbb 1"], "two fits that disagree")))
    fired.append(shows("log seeds > %d" % CH.MAX_LOG_SEED,
                       lambda: CH.load_run("0.2", CH.MAX_LOG_SEED + 1)))
    fired.append(shows("port.check_seeds > %d" % PORT.MAX_SEED,
                       lambda: PORT.check_seeds([PORT.MAX_SEED + 1], "mutated")))
    say()
    say("%d of %d mutations produced a failure" % (sum(fired), len(fired)))
    say()
    say("What two of those rows do and do not cover, said rather than left to the count:")
    say("- `digest lines differ` hands check_same_digests two literal strings.  It tests")
    say("  the comparator, not that part 7's fit reproduces; what says that is part 7's")
    say("  own line, where two child processes fit the real data.")
    say("- `permuted control` hands check_permuted the REAL R squared against the bar.  It")
    say("  tests the bar, not that a permuted fit collapses; what says that is part 5's")
    say("  own line, where the permuted target is fitted.")
    say()
    say("One check no LOG can fail, named rather than left out:")
    say("- that the record source yields one record per reschedule row and the row's own")
    say("  neighbour.  Both filters read the same two fields of the same rows, so no log")
    say("  can separate them; what they guard against is a change to")
    say("  observe.from_logged_rows, which is why the source is an argument and the two")
    say("  mutations above are sources and not logs.")
    say("  The same holds for the five-cell agreement between a row and its period: both")
    say("  sides call `charges.cell`, so no LOG can make them differ.  2026-11-25: that")
    say("  was first written as `no input can make them differ`, which is wrong -- a")
    say("  mutated RECORD SOURCE can, and `records/cell crossed` above is it firing.")


def part7(rows):
    """Part 7: the closed-form fit, twice, in two separate processes."""
    head("7  REPRODUCIBILITY OF THE CLOSED-FORM FIT, IN TWO PROCESSES")
    say("The fit is the one-period ridge on the full vector, one coefficient set per")
    say("interval, over every owned period of the cell and with no folds, so one command")
    say("reproduces it.  The bytes hashed are the three coefficient arrays' own, in the")
    say("order %s.  Two fresh interpreters are started, each reads the" % list(ARMS))
    say("%d runs again and solves again, and their lines are compared line by line."
        % (len(SEEDS) * len(P_CELLS)))
    say()
    first, second = second_process(), second_process()
    for line in first:
        say("  process 1    %s" % line)
    for line in second:
        say("  process 2    %s" % line)
    same = check_same_digests(first, second, "two processes")
    say("  %d of %d lines identical" % (same, len(first)))
    inside = digest_rows(rows)
    want = sorted("p%s %s %d" % (p, inside[p][0], inside[p][1]) for p in P_CELLS)
    for line in want:
        say("  this process %s" % line)
    say("  all three processes agree: %s"
        % (sorted(first) == sorted(second) == want))
    say("  (a difference here would say the solver is not deterministic on this machine,")
    say("  and every number in this file would then be one draw of it)")


def biggest_rho_move(table):
    """The largest absolute move of a held-out R squared under a fit-fold rho.

    Part 8 declares that rho is estimated on all the cell's seeds; this is the number that
    says what the declaration is worth, read off part 3's own table rather than typed.
    """
    return max(abs(table[(p, k, name + ", fit-fold rho")][0] - table[(p, k, name)][0])
               for p in P_CELLS for k in (1,) + KS
               for name in ("ridge full", "ridge cells"))


def part8(table):
    """Part 8: what this does not show."""
    head("8  WHAT THIS DOES NOT SHOW")
    say("- No policy and no score.  Nothing here is a decider, no replay simulation is")
    say("  run, and the per-decision agreement between the emulator and the replay is")
    say("  recorded as failed in evidence/13.  Every number is a property of the logs.")
    say("- Part 3's R squared is explanation, not effect.  The continuation after every")
    say("  row is the logging policy's, and a decision's cost at the downstream pair is")
    say("  charged to that pair and appears in no target here.")
    say("- rho is estimated on all %d seeds of its cell and then used in the target of the"
        % len(SEEDS))
    say("  read-out rows too, so one scalar of out-of-fold information is in those")
    say("  targets.  Part 3's fit-fold-rho table is that bound as a measurement: refitting")
    say("  with rho taken from each fold's fit rows alone moves no held-out R squared in")
    say("  the table by more than %.4f." % biggest_rho_move(table))
    say("- The penalty %.1f is fixed and not tuned.  A penalty chosen against these folds"
        % LAMBDA)
    say("  would make them part of the fit; nothing here says it is the best penalty.")
    say("- Part 4 is one period deep and reads the logged policy's own continuation.  It")
    say("  is not a replay score and it is not any policy's value.")
    say("- Nothing about an interval finer than the three, which the logs do not carry,")
    say("  and nothing about any seed above %d or about the label rows." % CH.MAX_LOG_SEED)
    say("- The %d decisions that own no period (one per pair per run) are in no table"
        % (8 * len(SEEDS) * len(P_CELLS)))
    say("  above; their cost is inside part 1's closing identity and nowhere else.")
    say("- No standard error on any pooled rate: part 2's SE is over runs and part 4's is")
    say("  clustered on runs, and neither is a confidence statement about a policy.")
    say()
    say("no check fired on real input; the failures in part 6 are the mutations")


def part9(by_p, rho_of_p):
    """Part 9: the wall time of one decision's input path on this machine."""
    head("9  TIME ON THIS MACHINE")
    say(TIME_MARKER)
    say("Acceptance 6's input path, in three parts timed separately, each over %d DISTINCT"
        % TIMING_CALLS)
    say("decisions of the logs, named below.  Medians and p99 in ms, printed beside the")
    say("100 ms tick the node's synchronous round runs on.  NO VERDICT: this is")
    say("an input to the step that puts a decider in the emulator, and the three parts are")
    say("timed apart because a sum quoted as one of them is not comparable with anything")
    say("on record.")
    say()
    say("This part does not reproduce byte for byte, and every line of it is below the")
    say("marker above, so a second run of this script differs from the first only here.")
    say()
    inputs, opened = [], []
    for seed in TIMING_SEEDS:
        events, _t0 = CH.load_run(P_CELLS[0], seed)
        opened.append(seed)
        for node in sorted(events):
            inputs.extend(OBS.from_logged_rows(events[node]))
        # Stop at the first seed that supplies enough: the later seeds of TIMING_SEEDS are
        # a reserve against a run with fewer decisions than usual and are not opened when
        # the earlier ones already cover the count.  `opened` is printed rather than the
        # range, so the output names the runs it really read.
        if len(inputs) >= TIMING_CALLS:
            break
    if len(inputs) < TIMING_CALLS:
        raise Fail("%d logged decisions, fewer than the %d a timing needs"
                   % (len(inputs), TIMING_CALLS))
    inputs = inputs[:TIMING_CALLS]
    say("  the %d decisions come from p %s, seed(s) %s, in the order the logs hold them"
        % (TIMING_CALLS, P_CELLS[0], opened))
    say()
    records = [OBS.observation(state, resolved, history, now, nb)
               for nb, state, resolved, history, now in inputs]
    vectors = [FE.vector(r) for r in records]
    mine = by_p[P_CELLS[0]]
    y_train = targets(mine, 1, rho_of_p[P_CELLS[0]])
    mean, sd = standardiser(full_columns(mine))
    stacked = np.stack(
        [ridge(design(full_columns([r for r in mine if r["arm"] == arm]), mean, sd),
               y_train[[i for i, r in enumerate(mine) if r["arm"] == arm]])
         for arm in ARMS], axis=1)
    answers = set()

    def one_value(vec):
        """Three linear value functions on one vector, and the cheapest interval."""
        row = np.concatenate([[1.0], (vec - mean) / sd])
        return ARMS[int(np.argmin(row @ stacked))]

    def time_calls(call, items):
        """[ms per call] over the items, one timing per call, nothing batched."""
        out = []
        for item in items:
            start = time.perf_counter()
            call(item)
            out.append((time.perf_counter() - start) * 1000.0)
        return out

    parts = (("observe.observation",
              lambda got: OBS.observation(got[1], got[2], got[3], got[4], got[0]),
              inputs),
             ("features.vector", FE.vector, records),
             ("3 values + argmin", lambda vec: answers.add(one_value(vec)), vectors))
    say("  %-22s %7s %10s %10s %10s %10s"
        % ("part", "calls", "median ms", "p99 ms", "max ms", "the tick"))
    for name, call, items in parts:
        got = time_calls(call, items)
        say("  %-22s %7d %10.4f %10.4f %10.4f %10.1f"
            % (name, len(got), quant(got, 0.5), quant(got, 0.99), max(got), 100.0))
    say()
    say("  the argmin returned %d distinct value(s) over the %d calls, every one of them"
        % (len(answers), TIMING_CALLS))
    say("  inside %s: %s" % (list(ARMS), sorted(answers) == sorted(answers & set(ARMS))))
    say()
    say("end of precheck.py")


def main():
    """Every table, and every check shown failing on a mutated input."""
    rows, runs = part0()
    by_p = {p: [r for r in rows if r["p"] == p] for p in P_CELLS}
    part1(by_p, runs)
    rho_of_p = part2(by_p)
    table = part3(by_p, rho_of_p)
    part4(by_p, rho_of_p)
    part5(by_p, rho_of_p)
    part6(rows, runs, by_p, rho_of_p, table)
    part7(rows)
    part8(table)
    part9(by_p, rho_of_p)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--digest":
        sys.exit(digest_main())
    sys.exit(main())
