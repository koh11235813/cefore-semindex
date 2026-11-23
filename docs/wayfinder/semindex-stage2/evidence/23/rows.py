#!/usr/bin/env python3
"""The rows candidate a and candidate b tune on, counted from the label file's bytes.

Runs on the LOCAL machine and reads two things: `evidence/22/labels.sha256` and the label
file it names.  It opens no run store, starts no simulation and loads no model, so it can
be re-run by a reader in under a minute.

2026-11-24.  Why it exists: `spec.md` has to name the training sets before anything is
tuned, and the only numbers it may name are numbers a generated file carries.  The plan
that preceded this step quoted 4 600 rows for candidate b from an earlier estimate and
`precheck.py` used that figure as given without recomputing the filter, so candidate b's
row count has never been counted on seeds 1 to 80.  This script counts it, and the
per-arm breakdown beside it, because the class shares decide what the "always the most
frequent arm" baseline is and that baseline is what any agreement number is read against.

The two candidates, as the user fixed them (their own decision 4):
  a  every row of seeds 1-80 that carries a label;
  b  the rows of a that also pass `t >= 3 or (sd = 0 and labelled)`, the selection
     `evidence/22/stability.py` measured the agreement of, with `label.paired_t` as the
     statistic and every row whose t is exactly 3 in real arithmetic KEPT.

The t of a row is a MASS POINT at the bar: seven passes at +a and one at -a give
mean 6a/8 against a sample sd of a/sqrt(2), i.e. t = 3 exactly, and per-pass differences
in whole multiples of one poll's weight land there too.  In doubles those rows sit within
1e-13 of 3 on either side and a bare `>=` splits them by the order the sums ran in
(`evidence/22/specnums.txt` prints 39 such rows over seeds 1-100 and the float keeping 19
of them).  So the bar here is `t >= 3 - BAR_TOL` with the same 1e-6 tolerance
`specnums.py` measures the clump against, and the strict-float count is printed beside it
so the difference the tolerance makes is a number and not a claim.
"""
import gzip
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
sys.path.insert(0, EV22)

import label as L        # noqa: E402  `paired_t`, `ARMS`: one definition, imported
import port as PORT      # noqa: E402  the >100 guard
import stability as ST    # noqa: E402  the bar the agreement was binned on

SHA_FILE = os.path.join(EV22, "labels.sha256")
CELL = "labels_p0.2.jsonl.gz"          # the main cell; p 0.4 is not used anywhere here

MAX_TUNE_SEED = 80                     # seeds 1-80 tune, 81-100 select (user, 決まったこと 5)
HELD_FROM = 71                         # the held-out slice of the tuning half, see spec.md
BAR = ST.T_BINS[-1][0]                 # 3.0, the last t bin's lower edge
BAR_TOL = 1e-6                         # specnums.py's own tolerance, for the same clump
STOP_AT_MS = 130000                    # the instant a live agent stops (replay's STOP_AT_MS)

ARMS = L.ARMS                          # (3.695, 6.0, 10.043), the exact floats


class Fail(Exception):
    """A check that did not hold.

    Its own class and not `AssertionError`, so that the mutation part below can catch a
    check firing without also catching a typo in the mutation and printing that as if the
    check had worked.
    """


def say(line=""):
    print(line, flush=True)


def head(title):
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


# ---------------------------------------------------------------------------
# the checks, as functions, so each one can be fed a mutated input below
# ---------------------------------------------------------------------------
def check_digest(path, want, got):
    """The label file is the one `evidence/22/labels.sha256` committed.

    Counted numbers are only as good as the bytes they came from, and the label file
    lives outside the repository, so nothing in git pins it.
    """
    if got != want:
        raise Fail("digest of %s is %s, committed %s" % (os.path.basename(path), got, want))
    return got


def check_seeds_tuning(seeds, where):
    """Every seed counted here is in the tuning half, 1-80.

    `port.check_seeds` refuses the measuring seeds (>100) and lets 81-100 through, so it
    cannot stand in for this one: 81-100 are the selection half, and a count that quietly
    included them would make the later choice between a and b circular.
    """
    bad = sorted({int(s) for s in seeds if int(s) > MAX_TUNE_SEED})
    if bad:
        raise Fail("%s: seeds above %d: %s" % (where, MAX_TUNE_SEED, bad[:8]))
    return sorted({int(s) for s in seeds})


def check_post_stop_all_ties(rows, where):
    """Every decision at or after the stop instant carries no label.

    A poll scheduled past `t0 + 130 000 ms` falls outside the scoring window, so all three
    arms score identically, the margin is exactly 0.0 and the row is a tie.  This is
    checked rather than quoted because `spec.md` leans on it to say that filtering on
    `ts_rel_ms` would drop no labelled row -- and if a labelled row ever appeared there,
    the filter the spec declines to write would start costing something.
    """
    bad = [r for r in rows if r["ts_rel_ms"] >= STOP_AT_MS and r["label"] is not None]
    if bad:
        raise Fail("%s: %d row(s) at or after ts_rel_ms %d carry a label, e.g. seed %s k %s"
                   % (where, len(bad), STOP_AT_MS, bad[0]["seed"], bad[0]["k"]))
    return len([r for r in rows if r["ts_rel_ms"] >= STOP_AT_MS])


def check_subset(small, big, where):
    """Candidate b is a subset of candidate a.

    b adds a condition to a's and removes none, so a b row outside a would mean the
    filter had been read the wrong way round -- for instance by keeping an `sd = 0` row
    that carries no label, which is the row `stability.txt` says a learner can learn
    nothing from.
    """
    keys_small = {(r["seed"], r["k"]) for r in small}
    keys_big = {(r["seed"], r["k"]) for r in big}
    extra = sorted(keys_small - keys_big)
    if extra:
        raise Fail("%s: %d row(s) of b are not in a, e.g. %s" % (where, len(extra), extra[:3]))
    return len(keys_small)


def check_counts_sum(counts, total, where):
    """The per-arm counts add up to the candidate's own row count.

    Without it a per-arm breakdown and a headline count could disagree and both be
    printed, which is the shape of error that makes a share wrong in the last digit and
    unexplainable a month later.
    """
    got = sum(counts.values())
    if got != total:
        raise Fail("%s: per-arm counts sum to %d, not %d" % (where, got, total))
    return total


def check_bar_isolated(on_bar, nearest_outside, where):
    """The 1e-6 tolerance decides nothing: there is room on both sides of the clump.

    `on_bar` are the |t - 3| of the rows inside the tolerance and `nearest_outside` is the
    smallest |t - 3| outside it.  The clump must sit a hundredfold INSIDE the tolerance
    and the nearest t off it a hundredfold OUTSIDE, so that a tolerance a hundred times
    wider or narrower would select the same rows.  If that room is absent the tolerance is
    choosing the count, and a count a tolerance chose is not a property of the data.
    """
    inner = max(on_bar) if on_bar else 0.0
    if inner * 100.0 > BAR_TOL:
        raise Fail("%s: the clump reaches %g, within a hundredth of the tolerance %g"
                   % (where, inner, BAR_TOL))
    if nearest_outside is not None and nearest_outside < 100.0 * BAR_TOL:
        raise Fail("%s: the nearest t off the clump is %g, inside a hundred times the"
                   " tolerance %g" % (where, nearest_outside, BAR_TOL))
    return inner, nearest_outside


# ---------------------------------------------------------------------------
# reading and classifying
# ---------------------------------------------------------------------------
def digests_wanted():
    """{basename: sha256} from the committed digest file.

    The committed file names ABSOLUTE local paths; only the path column is reduced to a
    basename, so the hashes compared are the committed ones.
    """
    want = {}
    with open(SHA_FILE, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                h, path = line.split()
                want[os.path.basename(path)] = h
    return want


def label_path():
    """The label file's absolute path, taken from the committed digest file.

    Read out of that file rather than written here, so that the path and the hash cannot
    drift apart: a file whose location moved takes its hash line with it.
    """
    with open(SHA_FILE, encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and os.path.basename(line.split()[1]) == CELL:
                return line.split()[1]
    raise Fail("%s names no %s" % (SHA_FILE, CELL))


def read_rows(path):
    """Every label row of the file, and the sha256 of the bytes they came out of.

    Hashed from the gzip stream, not from the decompressed JSON, because that is what the
    committed digest covers.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    got = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(line) for line in gzip.decompress(raw).splitlines() if line.strip()]
    return rows, got


def t_of(row):
    """`(t, sd)` of a row, by `label.paired_t` over the row's own passes.

    Imported and not re-implemented: `stability.py` bins its agreement on this statistic,
    and a second copy here would be a second definition of the thing candidate b is
    selected by.
    """
    t, sd, _n = L.paired_t(row["pass_scores"], row["passes"])
    return t, sd


def keeps_b(row, tol=BAR_TOL):
    """Candidate b's own rule on one row: `t >= 3 or (sd = 0 and labelled)`.

    `paired_t` hands back `t = None, sd = 0.0` when every pass separated the two best arms
    by the same amount -- there is no t to compare, and such a row carries a label, which
    is the `sd = 0 and labelled` arm of the rule.  A row whose passes all scored the two
    best arms EQUALLY is a tie, carries no label, and is refused by the `label is None`
    test first, so it never reaches the sd branch.
    """
    if row["label"] is None:
        return False
    t, sd = t_of(row)
    if sd == 0.0:
        return True
    return t is not None and t >= BAR - tol


def arm_counts(rows):
    """{arm: how many of these rows carry that label}, over the three arms."""
    counts = {arm: 0 for arm in ARMS}
    for r in rows:
        counts[r["label"]] += 1
    return counts


def table(name, rows, total_rows):
    """One candidate's per-arm counts, shares and most-frequent-arm baseline."""
    counts = arm_counts(rows)
    n = len(rows)
    check_counts_sum(counts, n, "candidate %s" % name)
    say("  candidate %s   %d rows of %d (%.4f of the seed range's rows)"
        % (name, n, total_rows, n / total_rows))
    for arm in ARMS:
        say("    %-8s %6d  %.4f" % ("%.3f s" % arm, counts[arm], counts[arm] / n))
    top = max(ARMS, key=lambda a: (counts[a], -ARMS.index(a)))
    say("    always %.3f s (the most frequent arm) agrees with %.4f of these rows"
        % (top, counts[top] / n))
    return counts


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
head("0  THE FILE")
PATH = label_path()
WANT = digests_wanted()[CELL]
ROWS, GOT = read_rows(PATH)
check_digest(PATH, WANT, GOT)
say("label file      %s" % PATH)
say("sha256          %s  == evidence/22/labels.sha256" % GOT)
say("rows            %d" % len(ROWS))
SEEDS_IN_FILE = sorted({int(r["seed"]) for r in ROWS})
say("seeds in file   min %d max %d, %d distinct" % (SEEDS_IN_FILE[0], SEEDS_IN_FILE[-1],
                                                    len(SEEDS_IN_FILE)))
PORT.check_seeds(SEEDS_IN_FILE, "rows.py: the seeds the label file carries")
say("port.check_seeds passed them, so nothing above 100 is in this file")

head("1  THE TUNING HALF, SEEDS 1 TO %d" % MAX_TUNE_SEED)
TUNE = [r for r in ROWS if int(r["seed"]) <= MAX_TUNE_SEED]
TUNE_SEEDS = check_seeds_tuning([r["seed"] for r in TUNE], "the rows this script counts")
say("rows            %d of %d, over %d seeds (%d to %d)"
    % (len(TUNE), len(ROWS), len(TUNE_SEEDS), TUNE_SEEDS[0], TUNE_SEEDS[-1]))
say("seeds 81-100 (selection) and 121-160 (measuring) appear nowhere below: the row list")
say("is cut at seed %d before anything is counted" % MAX_TUNE_SEED)
TIES = [r for r in TUNE if r["label"] is None]
say("ties (no label) %d of %d (%.4f)" % (len(TIES), len(TUNE), len(TIES) / len(TUNE)))
POST = check_post_stop_all_ties(TUNE, "the tuning half")
say("rows at or after ts_rel_ms %d: %d, and every one of them is a tie, so a consumer"
    % (STOP_AT_MS, POST))
say("that filters them out loses no labelled row")

head("2  THE TWO CANDIDATES ON SEEDS 1 TO %d" % MAX_TUNE_SEED)
A = [r for r in TUNE if r["label"] is not None]
B = [r for r in TUNE if keeps_b(r)]
check_subset(B, A, "b against a")
COUNTS_A = table("a", A, len(TUNE))
say()
COUNTS_B = table("b", B, len(TUNE))
say("    b keeps %.4f of candidate a's rows" % (len(B) / len(A)))

head("3  WHAT THE TOLERANCE AT THE BAR DOES")
ON_BAR, OUTSIDE, STRICT = [], None, 0
for r in A:
    t, sd = t_of(r)
    if t is None:
        continue
    dist = abs(t - BAR)
    if dist <= BAR_TOL:
        ON_BAR.append(dist)
    elif OUTSIDE is None or dist < OUTSIDE:
        OUTSIDE = dist
    if t >= BAR:
        STRICT += 1
STRICT += sum(1 for r in A if t_of(r)[1] == 0.0)
INNER, NEAR = check_bar_isolated(ON_BAR, OUTSIDE, "the bar")
say("rows with a t within %g of the bar %g: %d" % (BAR_TOL, BAR, len(ON_BAR)))
say("the largest |t - %g| among them: %g, which is %.3g of the tolerance"
    % (BAR, INNER, INNER / BAR_TOL))
say("the nearest |t - %g| outside the tolerance: %g, which is %.3g times it"
    % (BAR, NEAR, NEAR / BAR_TOL))
say("so the clump is isolated and a tolerance a hundred times wider or narrower would")
say("catch the same rows")
say("candidate b with the tolerance (every row ON the bar kept): %d rows" % len(B))
say("candidate b with a bare float `t >= %g`:                    %d rows" % (BAR, STRICT))
say("the tolerance keeps %d row(s) the bare comparison drops" % (len(B) - STRICT))

head("4  THE SPLIT THE TUNING USES: SEEDS 1-%d TUNE, %d-%d HELD OUT"
     % (HELD_FROM - 1, HELD_FROM, MAX_TUNE_SEED))
say("The held-out slice is carved out of the TUNING half, so no number from seeds 81-100")
say("reaches the tuning.  It does not tune: the counts below are what each candidate")
say("actually optimises on, and they are smaller than part 2's for that reason.")
for name, rows in (("a", A), ("b", B)):
    fit = [r for r in rows if int(r["seed"]) < HELD_FROM]
    held = [r for r in rows if int(r["seed"]) >= HELD_FROM]
    cf, ch = arm_counts(fit), arm_counts(held)
    check_counts_sum(cf, len(fit), "candidate %s, seeds 1-%d" % (name, HELD_FROM - 1))
    check_counts_sum(ch, len(held), "candidate %s, seeds %d-%d" % (name, HELD_FROM,
                                                                   MAX_TUNE_SEED))
    say()
    say("  candidate %s   tune %d rows (seeds 1-%d)   held out %d rows (seeds %d-%d)"
        % (name, len(fit), HELD_FROM - 1, len(held), HELD_FROM, MAX_TUNE_SEED))
    say("    tune     " + "  ".join("%.3f s %d (%.4f)" % (arm, cf[arm], cf[arm] / len(fit))
                                    for arm in ARMS))
    say("    held out " + "  ".join("%.3f s %d (%.4f)" % (arm, ch[arm], ch[arm] / len(held))
                                    for arm in ARMS))
    top = max(ARMS, key=lambda a: (ch[a], -ARMS.index(a)))
    say("    always %.3f s on the held-out rows: %.4f" % (top, ch[top] / len(held)))

head("5  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT")
say("The mutation is of the INPUT, inside this process.  A per-check copy of the script")
say("would be six copies of a 60-second read of the same file, and a copy that silently")
say("matched nothing would read as a passing check.")
say()
FIRED = []


def shows(name, call):
    """Run a check on a mutated input and require it to refuse.

    `Fail` is this script's own checks; `SystemExit` is `port.check_seeds`, which is the
    replay's own guard and raises that.  Narrow on purpose: catching `Exception` here
    would also catch a typo in the mutation and print it as if the check had worked.
    """
    try:
        call()
    except (Fail, SystemExit) as exc:
        say("  %-22s FIRES: %s" % (name, exc))
        return True
    say("  %-22s DID NOT FIRE -- this check cannot fail on this input" % name)
    return False


BAD_ROW = dict(A[0], ts_rel_ms=float(STOP_AT_MS), label=ARMS[0])
FIRED.append(shows("digest", lambda: check_digest(PATH, "deadbeef", GOT)))
FIRED.append(shows("seeds_tuning",
                   lambda: check_seeds_tuning(TUNE_SEEDS + [MAX_TUNE_SEED + 1], "mutated")))
FIRED.append(shows("port.check_seeds",
                   lambda: PORT.check_seeds([PORT.MAX_SEED + 1], "mutated")))
FIRED.append(shows("post_stop_ties",
                   lambda: check_post_stop_all_ties(TUNE + [BAD_ROW], "mutated")))
# The row removed from `a` is one that candidate b KEEPS, because removing an arbitrary
# `a` row need not break the subset relation -- most `a` rows are not in `b` at all, and a
# mutation that removes one of those leaves the check passing and says nothing.
A_MINUS_B0 = [r for r in A if (r["seed"], r["k"]) != (B[0]["seed"], B[0]["k"])]
FIRED.append(shows("subset", lambda: check_subset(B, A_MINUS_B0, "mutated")))
FIRED.append(shows("counts_sum",
                   lambda: check_counts_sum(COUNTS_A, len(A) + 1, "mutated")))
FIRED.append(shows("bar_isolated/inner",
                   lambda: check_bar_isolated([BAR_TOL / 2.0], OUTSIDE, "mutated")))
FIRED.append(shows("bar_isolated/outside",
                   lambda: check_bar_isolated(ON_BAR, BAR_TOL * 2.0, "mutated")))
say()
say("%d of %d mutations produced a failure" % (sum(FIRED), len(FIRED)))

head("6  WHAT THIS SCRIPT DOES NOT SAY")
say("- Nothing about which candidate is better.  It counts rows; the choice between a")
say("  and b is made on seeds 81-100 by the replay score, and not here and not by any")
say("  agreement with the labels.")
say("- Nothing about label QUALITY.  The retention of candidate b's filter is not an")
say("  agreement: the agreement of that selection was measured on seeds 1-10 against a")
say("  second labelling (evidence/22/stability.txt, 0.9777 there against 0.7623 overall),")
say("  and the file on disk is ONE labelling, so no agreement can be computed from it.")
say("- The most-frequent-arm shares above are a baseline, not a result.  A tuned model")
say("  that only matched them would have learnt the class prior and nothing else.")
say()
say("no check fired on real input; the failures in part 5 are the mutations")
say("end of rows.py")
