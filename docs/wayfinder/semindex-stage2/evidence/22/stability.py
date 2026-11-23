#!/usr/bin/env python3
"""Acceptance 6: how far a label moves when the Mode B passes behind it are replaced.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B stability.py

Writes its whole printed output to stability.txt.  **It carries no pass line**: the
ticket asks for the size of the jitter residue to be seen as a number for the first time,
not for a threshold on it.  The one decision it does make is which J the full labelling
runs with, by a rule fixed before any number existed (below).

WHAT IS MEASURED
  On seeds 1 to 10 of each cell, every resolution-time decision is labelled twice, from
  two DISJOINT sets of 8 passes: set A is passes 0 to 7 and set B is passes 8 to 15.  A
  pass is a (lag, jitter) stream keyed by the run, the probed decision and the pass index,
  live from the probed decision on, so two passes differ in nothing but the randomness
  after that decision.  For J in 1, 2, 4 and 8 the two labels are recomputed from the
  FIRST J passes of each set and compared.

  Three populations, because one share would hide where the disagreement sits:
  * ALL decisions.  A decision that ties on one side and has a label on the other counts
    as a disagreement, which is why this share is the lowest of the three.
  * NON-TIE decisions: both sides have a label.  This is the population the J rule reads.
  * The LARGER-MARGIN HALF: the decisions of that cell whose set-A margin at that J is in
    the upper half, ranked descending with (seed, k) as the tie-break so the split is
    reproducible.  A label with a wide margin should survive a change of passes, and a
    cell where it does not has a problem the pass count cannot fix.
    The half is taken over ALL decisions, which is how the measurement was defined, so it
    is over a WIDER population than the non-tie column beside it: a member of the half
    that ties on set B only carries no label there and is a forced disagreement.
    2026-11-23: a fourth column therefore restricts the half to the members that carry a
    label on BOTH sides, because the half share and the non-tie share are otherwise two
    numbers over two populations printed side by side.  The two populations differ only at
    the small J (the margin at J = 1 is one pass's spread, so a decision can be high in
    set A's ranking and tied in set B), and the column's own (n) is where to read that.

TWO BREAKDOWNS OF THE J = 8 AGREEMENT (2026-11-23), bins fixed before the numbers
  The table above shows the larger-margin half agreeing LESS than the whole at every J and
  both cells, so the margin is not a confidence and the obvious move -- keep the
  wide-margin labels, drop the rest -- does not follow from it.  These two breakdowns ask
  whether anything else about a decision predicts whether its label survives a change of
  passes:
  * set A's own PAIRED T over its 8 passes, `mean(d) / (sd(d) / sqrt(8))` with `d` the
    per-pass gap between the two best arms, in bins [0,1), [1,2), [2,3), [3,inf) plus a
    bucket for `sd = 0`, which has no t.  The t is the margin divided by its own spread,
    which is the thing the margin alone cannot distinguish;
  * the decision's POSITION in its run, as equal-count quartiles of `ts_rel_ms`, with the
    median margin printed beside each one.  A decision near the end of the window has
    fewer polls left in which the arms can separate and less randomness left to average,
    and the table above cannot say which of the two dominates.
  Both are over ALL decisions of the cell, with agreement defined as `set A's label ==
  set B's label`, so two ties agree and a one-sided tie disagrees.  When a breakdown
  separates nothing, the script says so in those words.

THE RULE THAT PICKS J, fixed here before any number existed
  The smallest J in (1, 2, 4, 8) whose agreement over the NON-TIE decisions of the main
  cell (p 0.2) is at least 0.90, and 8 if none of them reaches it.  The agreement at the
  chosen J is printed plainly, including when it is poor: a J chosen because nothing
  reached the bar is a J with a stated weakness, not a J that passed.

WHAT THIS CANNOT SHOW
  That the labels are right at the chosen J.  Two labellings that agree are two labellings
  of the same construction; they agree about the construction, not about the testbed.  And
  agreement measured on seeds 1 to 10 is a statement about those seeds: the full labelling
  runs 1 to 100 and is not re-checked.
"""
import multiprocessing as mp
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label as L  # noqa: E402
import port as PORT  # noqa: E402

PS = (0.2, 0.4)
MAIN_P = 0.2
SEEDS = PORT.check_seeds(range(1, 11), "stability.py")
SET_A = list(range(0, 8))
SET_B = list(range(8, 16))
JS = (1, 2, 4, 8)
BAR = 0.90                              # the J rule's bar, on the main cell's non-ties
WORKERS = 5

OUT_TXT = os.path.join(HERE, "stability.txt")

LINES = []


def out(text=""):
    print(text, flush=True)
    LINES.append(text)


def job(key):
    """One (p, seed) labelled over both pass sets at once: (p, seed, rows, fired, wall)."""
    p, seed = key
    _p, _s, rows, fired, wall, _passes = L.label_run((p, seed, SET_A + SET_B))
    # Only what the comparison needs travels back through the pool: the full rows carry
    # the state, the history and every per-pass score, which is megabytes per run and
    # nothing this script reads.
    # 2026-11-23: `ts_rel_ms` travels too, because the position breakdown bins on it.
    thin = [{"seed": seed, "k": r["k"], "ts_rel_ms": r["ts_rel_ms"],
             "pass_scores": r["pass_scores"]} for r in rows]
    return p, seed, thin, fired, wall


def label_from(row, offset, j):
    """(label, margin, tie) from `j` consecutive passes, by `label.pick`'s own rule.

    The per-pass scores are in the order the run was asked for, `SET_A + SET_B`, so
    offset 0 is set A's first pass and offset 8 is set B's.  `label.pick` and not a second
    copy of the rule, so that a label computed here is a label computed the way the file
    on disk computes it.
    """
    means = {arm: statistics.fmean(row["pass_scores"][repr(arm)][offset:offset + j])
             for arm in L.ARMS}
    return L.pick(means)


def agreement(rows, j):
    """The three shares at one J, the half restricted to both-labelled, and their sizes."""
    both = []
    for row in rows:
        a = label_from(row, 0, j)
        b = label_from(row, len(SET_A), j)
        both.append((row, a, b))
    allshare = statistics.fmean([float(a[0] == b[0]) for _r, a, b in both])
    nontie = [(r, a, b) for r, a, b in both if not a[2] and not b[2]]
    ntshare = (statistics.fmean([float(a[0] == b[0]) for _r, a, b in nontie])
               if nontie else float("nan"))
    ranked = sorted(both, key=lambda t: (-t[1][1], t[0]["seed"], t[0]["k"]))
    half = ranked[:len(ranked) // 2]
    halfshare = (statistics.fmean([float(a[0] == b[0]) for _r, a, b in half])
                 if half else float("nan"))
    # The half is ranked over `both`, so it holds decisions that tie on ONE side, which
    # the non-tie share above excludes; restricting it here is what makes the two
    # comparable.  2026-11-23: added because the two shares were printed side by side
    # with no statement that they are over different populations.
    halfnt = [(r, a, b) for r, a, b in half if not a[2] and not b[2]]
    halfntshare = (statistics.fmean([float(a[0] == b[0]) for _r, a, b in halfnt])
                   if halfnt else float("nan"))
    return (allshare, ntshare, len(nontie), halfshare, len(half),
            halfntshare, len(halfnt), both)


# 2026-11-23, fixed here before the numbers and not moved after seeing them.  Two
# breakdowns of the same agreement, because `stability.txt` showed the larger-margin half
# agreeing LESS than the non-ties as a whole, which is the opposite of what a margin is
# supposed to mean -- so the margin is not a confidence, and the question becomes whether
# anything else about a decision predicts whether its label survives a change of passes.
T_BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, float("inf")))
QUARTILES = 4


def breakdown_by_t(rows, j):
    """({t bin: [agreed]}, [agreed where sd = 0], [(agreed, t) in no bin]) at one J.

    The t is set A's OWN paired t over its first `j` passes (`label.paired_t`): the two
    best arms by mean score, the per-pass difference between them, and
    `mean(d) / (sd(d) / sqrt(j))`.  It is a different quantity from the margin the half is
    ranked by -- the margin is `mean(d)` alone, so a wide margin with wildly spread passes
    and a narrow one with eight identical passes get the same rank there and very
    different t's here.

    The population is ALL decisions of the cell, and agreement is `A's label == B's
    label`, so a decision that ties on both sides agrees (both labels are null) and one
    that ties on one side only disagrees.  That is the same rule the table's `all` column
    uses.  A decision that TIES on set A has its two best arms' pass means equal, so its
    `mean(d)` is zero to the last bits of a double -- not exactly zero, because
    `fmean(a) == fmean(b)` does not make `fmean(a_j - b_j)` exactly 0.  It therefore lands
    in the first bin, in the `sd = 0` bucket, or, when that rounding comes out a hair
    BELOW zero, in no bin at all.

    `sd` exactly 0 gets its own bucket and no t at all: every pass separated the two arms
    by the same amount, so the divisor is 0, and folding that into the open top bin would
    mix "the passes agree perfectly" with "the passes agree strongly".

    2026-11-23: the third return value exists because the bin loop used to `break` with no
    `else`, so a decision matching no bin was dropped without a word and the table covered
    fewer decisions than the cell it named -- the t table of p 0.4 printed 1621 of 1623.
    The bins are not widened and no t is clamped, because the bins were fixed before these
    numbers were seen; what matched nothing is reported instead, with its t, so the reader
    can see it is a tie's rounding and not a decision the bins fail to cover.
    """
    bins = {b: [] for b in T_BINS}
    no_sd = []
    no_bin = []
    passes = range(0, j)
    for row in rows:
        agreed = float(label_from(row, 0, j)[0] == label_from(row, len(SET_A), j)[0])
        t, _sd, _n = L.paired_t(row["pass_scores"], passes)
        if t is None:
            no_sd.append(agreed)
            continue
        for lo, hi in T_BINS:
            if lo <= t < hi:
                bins[(lo, hi)].append(agreed)
                break
        else:
            no_bin.append((agreed, t))
    return bins, no_sd, no_bin


def breakdown_by_position(rows, j):
    """Per quartile of `ts_rel_ms` over the cell: (first, last, n, agreement, median margin).

    The quartiles are equal-COUNT quartiles of the decision's own stamp relative to its
    run's `t0`, so the four buckets are comparable in size whatever the shape of the
    distribution, and each one prints the stamp range it covers.  The margin is set A's at
    this J, the same number the larger-margin half is ranked by, so the two breakdowns can
    be read against each other.

    Why position at all: a decision late in the window has fewer polls left before the
    window ends, so there is less room for the three arms to separate AND less randomness
    left to average -- which of the two wins is not something the table above can say.
    """
    scored = sorted(
        (row["ts_rel_ms"],
         float(label_from(row, 0, j)[0] == label_from(row, len(SET_A), j)[0]),
         label_from(row, 0, j)[1])
        for row in rows)
    n = len(scored)
    out = []
    for q in range(QUARTILES):
        chunk = scored[n * q // QUARTILES:n * (q + 1) // QUARTILES]
        if not chunk:
            continue
        out.append((chunk[0][0], chunk[-1][0], len(chunk),
                    statistics.fmean(c[1] for c in chunk),
                    statistics.median([c[2] for c in chunk])))
    return out


def spread_line(shares, what):
    """One line saying whether a breakdown separated anything, including when it did not."""
    if len(shares) < 2:
        return f"  {what}: one bucket only, so this breakdown separates nothing."
    lo, hi = min(shares), max(shares)
    if hi - lo < 0.02:
        return (f"  {what}: NOTHING -- the agreement runs from {lo:.4f} to {hi:.4f} "
                f"across the buckets, a spread of {hi - lo:.4f}, so this breakdown does "
                "not separate the labels that survive a change of passes from the ones "
                "that do not.")
    return (f"  {what}: the agreement runs from {lo:.4f} to {hi:.4f} across the buckets, "
            f"a spread of {hi - lo:.4f}.")


def pick_j(per_j):
    """The rule of the docstring, applied to the main cell's non-tie shares."""
    for j in JS:
        if per_j[j][1] >= BAR:
            return j, True
    return JS[-1], False


def main():
    started = time.time()
    keys = [(p, seed) for p in PS for seed in SEEDS]
    cells, fired, per_run = {}, 0, {}
    with mp.Pool(WORKERS) as pool:
        for p, seed, thin, got, wall in pool.imap_unordered(job, keys, chunksize=1):
            cells.setdefault(p, []).extend(thin)
            fired += got
            per_run[(p, seed)] = (len(thin), wall)
            print(f"  p {p} seed {seed}: {len(thin)} decisions, {wall:.1f} s", flush=True)
    for p in cells:
        cells[p].sort(key=lambda r: (r["seed"], r["k"]))

    out("The stability of a label under a change of Mode B passes")
    out("========================================================")
    out(f"Seeds {SEEDS[0]} to {SEEDS[-1]} of p {PS[0]} and p {PS[1]}, every "
        "resolution-time decision, two disjoint sets of 8 passes "
        f"(A = {SET_A[0]} to {SET_A[-1]}, B = {SET_B[0]} to {SET_B[-1]}).")
    out("No pass line: this is the first measurement of the residue, not a threshold on "
        "it.")
    out()
    tables = {}
    for p in PS:
        rows = cells[p]
        out(f"p {p}: {len(rows)} decisions over {len(SEEDS)} runs")
        out("   J   agreement all     non-tie (n)          larger-margin half (n)"
            "   that half, both sides labelled (n)")
        tables[p] = {}
        for j in JS:
            (allshare, ntshare, nnt, halfshare, nhalf,
             halfntshare, nhalfnt) = agreement(rows, j)[:7]
            tables[p][j] = (allshare, ntshare, nnt, halfshare, nhalf,
                            halfntshare, nhalfnt)
            out("  %2d   %7.4f           %7.4f (%5d)      %7.4f (%5d)   %7.4f (%5d)"
                % (j, allshare, ntshare, nnt, halfshare, nhalf,
                   halfntshare, nhalfnt))
        out()
    out("The larger-margin half is ranked over ALL decisions, so it holds decisions that "
        "tie on one side and carry a label on the other; each of those is a forced "
        "disagreement. The last column is that same half with those members removed, and "
        "its (n) against the half's (n) is how many they are.")
    out()
    chosen, reached = pick_j(tables[MAIN_P])
    out("The J the full labelling runs with")
    out(f"  rule: the smallest J in {list(JS)} whose non-tie agreement on the main cell "
        f"(p {MAIN_P}) is at least {BAR:.2f}, and {JS[-1]} if none reaches it.")
    out(f"  CHOSEN J = {chosen}"
        + ("" if reached else
           f" -- and NOT because it reached the bar: no J in {list(JS)} did, so the rule's "
           "fallback took it."))
    row = tables[MAIN_P][chosen]
    out(f"  at J = {chosen} on the main cell: agreement over all decisions "
        f"{row[0]:.4f}, over the {row[2]} non-tie decisions {row[1]:.4f}, over the "
        f"larger-margin half {row[3]:.4f} ({row[5]:.4f} over the {row[6]} of that half "
        "that carry a label on both sides)")
    other = [p for p in PS if p != MAIN_P][0]
    orow = tables[other][chosen]
    out(f"  at J = {chosen} on p {other}: {orow[0]:.4f} / {orow[1]:.4f} / {orow[3]:.4f} "
        "(all / non-tie / larger-margin half), which the rule does not read")
    out(f"  Read the non-tie share as what it is: {1 - row[1]:.4f} of the decisions that "
        f"got a label from both sets got a DIFFERENT label, on seeds {SEEDS[0]} to "
        f"{SEEDS[-1]}. That is the disagreement between two INDEPENDENT {len(SET_A)}-pass "
        "labellings, so it BOUNDS FROM ABOVE the share of the labels on disk that are "
        "pass noise: a disagreement needs only ONE of the two sides to be wrong, so the "
        "share of one side's labels that are an artefact of which passes were drawn is "
        "between half of this number and all of it.")
    out()
    out("Two breakdowns of the same agreement, at J = %d, over ALL decisions of each "
        "cell" % chosen)
    out("  Why: the larger-margin half above agrees LESS than the whole, so the margin is "
        "not a confidence. These two ask whether anything else about a decision says "
        "whether its label survives a change of passes. The bins were fixed before these "
        "numbers existed. Agreement is `set A's label == set B's label`, so two ties "
        "agree and a one-sided tie disagrees, as in the `all` column.")
    out()
    for p in PS:
        bins, no_sd, no_bin = breakdown_by_t(cells[p], chosen)
        out(f"  p {p}, by set A's own paired t over its {chosen} passes "
            "(t = mean(d) / (sd(d) / sqrt(J)), d = the per-pass gap between the two best "
            "arms; a decision that TIES on set A has the two arms' pass means equal, so "
            "its mean(d) is zero to the last bits of a double and it falls in the first "
            "bin, in the sd = 0 bucket, or -- if that rounding comes out below zero -- in "
            "no bin, which the line under the table counts)")
        out("     t bin         n   agreement")
        shares = []
        for lo, hi in T_BINS:
            got = bins[(lo, hi)]
            label = f"[{lo:.0f},{'inf' if hi == float('inf') else '%.0f' % hi})"
            if not got:
                out(f"    {label:<10} {0:6d}   --")
                continue
            shares.append(statistics.fmean(got))
            out(f"    {label:<10} {len(got):6d}   {shares[-1]:7.4f}")
        if no_sd:
            share = statistics.fmean(no_sd)
            shares.append(share)
            out(f"    {'sd = 0':<10} {len(no_sd):6d}   {share:7.4f}   (every pass "
                "separated the two arms by the same amount, so there is no t)")
        else:
            out(f"    {'sd = 0':<10} {0:6d}   --")
        binned = sum(len(bins[b]) for b in T_BINS) + len(no_sd)
        # 2026-11-23: printed even when it is 0, because the count is the claim that the
        # table covers the whole cell.  Silence here used to mean "nothing fell out" and
        # "two decisions fell out" alike.
        out(f"  the table covers {binned} of the cell's {len(cells[p])} decisions; "
            f"matching no bin: {len(no_bin)}"
            + (f" (t {', '.join('%.3g' % t for _a, t in no_bin)}, a tie's rounding "
               "below zero)" if no_bin else ""))
        # The two top buckets together, printed here so that no reader has to do this
        # arithmetic over two printed rows and get the rounding wrong.
        top = bins[T_BINS[-1]] + no_sd
        if top:
            out(f"  t >= 3 or sd = 0 together: {len(top)} decisions "
                f"({len(top) / len(cells[p]):.4f} of the cell) at "
                f"{statistics.fmean(top):.4f} agreement, against "
                f"{tables[p][chosen][0]:.4f} over everything")
        out(spread_line(shares, "what the paired t says"))
        out()
    for p in PS:
        quarts = breakdown_by_position(cells[p], chosen)
        out(f"  p {p}, by the decision's position in its run (equal-count quartiles of "
            "ts_rel_ms)")
        out("     quartile   ts_rel_ms range            n   agreement   median margin ms")
        for i, (first, last, n, share, margin) in enumerate(quarts, 1):
            out("    Q%d         %9.0f to %9.0f %6d   %9.4f   %16.4f"
                % (i, first, last, n, share, margin))
        out(spread_line([q[3] for q in quarts], "what the position says"))
        out()
    ties = {p: 1.0 - tables[p][chosen][2] / len(cells[p]) for p in PS}
    out("Ties, at the chosen J (a decision where one side or the other scored two arms "
        "exactly equally)")
    for p in PS:
        out(f"  p {p}: {ties[p]:.4f} of {len(cells[p])} decisions")
    out()
    out("Provisional timers that fired before their own poll had resolved, over every "
        f"simulation of this script: {fired}")
    out(f"wall {time.time() - started:.1f} s on {WORKERS} workers, "
        f"{sum(n for n, _w in per_run.values())} decisions x 3 arms x "
        f"{len(SET_A + SET_B)} passes")
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT_TXT}")
    return 1 if fired else 0


if __name__ == "__main__":
    sys.exit(main())
