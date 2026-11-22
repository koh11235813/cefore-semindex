"""The four pass lines of spec.md for evidence/20, from perrun.csv here and the stored runs.

Writes results.csv (one row per reported quantity) and results.md (the same numbers as tables), and
prints results.md on stdout. Every verdict word in both is the mechanical one spec.md defines;
nothing here reads more into a number than that.

The five s10 stages and the arm each one is, in spec.md's own order:

    B3    `s10b3p695`    fixed 3.695 s, jitter 0.1
    B6    `s10b6p0`      fixed 6.0 s, jitter 0.1
    B10   `s10b10p043`   fixed 10.043 s, jitter 0.1
    T37r  `s10t37r`      rule, i_min 3.695, factor 2, i_max 10.043, reset
    F     `s10f`         the pinned F, i_min 3.695, factor 1.5, i_max 10.043, reset

One cell: p 0.2, workload w0, seeds 121 to 160, 40 runs per stage. Unlike evidence/19 there is no
gate stage and no second load, and unlike evidence/19's T37/T37r pair the two rules compared here
differ in `factor` alone.

The two scorings of spec.md's "Quantities":

    headline      the records whose scheduled creation time is at or before t0+72.0 s, delay the
                  mean over the uncensored ones 4 hops away (`hlat4 / hn4`), censored share
                  `hc4 / (hn4 + hc4)`
    all records   every record of the run, the quantity every earlier step reported
                  (`lat4 / n4`, `c4 / (n4 + c4)`)

spec.md says that under `w0` every record is due at or before t0+71 s, so the two scorings select
the same 240 records and give one number. That is reported as a measured equality per stage and not
assumed: the headline scoring carries each verdict word and the all-record value stands beside it
everywhere, as in evidence/19, so a stage where the equality failed would be visible.

Polls per run is the mean of `polls_sent` over the window [t0+10 s, t0+130 s] under both scorings,
because spec.md counts polls over the whole window whichever set of records the delay is taken on.

What each pass line needs:

line 1  All 200 runs DONE and unflagged, per stage, with the flag breakdown by stage and by flag
        name printed even when it is empty; every `interval_s` in every decision log inside the set
        its stage's setting allows, the set built by the arithmetic `_advance_rule_interval` uses so
        that the comparison is on exact doubles; and the pinned F's five values restated against
        main10.sh's launched FPIN string and fpin.txt's winner line, both read here at run time.
line 2  The gain of T37r and of the pinned F against the three-arm log-log frontier, both scorings,
        with the 95 % seed-bootstrap interval, the half width, the verdict word, the censored share
        beside every delay, the monotone check in the full sample, and the resamples left out split
        by the reason they were left out.
line 3  F against T37r directly: the delay ratio and the polls ratio from the same resamples, with
        intervals and measured half widths beside the predicted ones. No verdict.
line 4  The delay and the polls per run of all five stages against the plus or minus 6 % bands of
        predict.txt, ten values, with the three-way word, the breakdown for the seven informative
        values, and every value's deviation in percent with sign whatever the count inside is.

Seed sets (spec.md pass line 1, taken from evidence/17's amendment 1 by way of evidence/19): a
line's seeds are the seeds that are DONE and unflagged in all five stages, because spec.md's
bootstrap recomputes all five stages from the same resampled seeds. The number of seeds is printed
beside every verdict, and a line computed on fewer than 40 seeds reads `incomplete` in place of its
verdict. The interval check of line 1 is the one thing read over every DONE run rather than over the
intersection, because spec.md asks it of every decision log and a flagged run still logged one.

Bootstrap: one index matrix `default_rng(20261122).integers(0, n, size=(2000, n))`, drawn once and
reused for every quantity and both scorings, which is how predict.py drew its own (`boot_index`).
The unit is the seed number: one index row selects the seeds and all five stages are recomputed from
that one selection, so the stages stay paired inside a resample.

The predicted constants below are predict.txt's, and `check_prediction` parses predict.txt at run
time and stops the script if any of them has drifted from the digits that frozen file prints. Both
files are needed because the constants carry the structure (which number is a band end, which a half
width) that a parse alone would not fix, and the parse is what keeps the two from parting.

Reads only `<run_dir>/meta.json` and `<run_dir>/debug/node_dirs/h*/semindex/events.jsonl` under the
store, and writes nothing there. Nothing else under `node_dirs` is opened: the rest of that tree
holds the hosts' Cefore key material.

Environment: STORE (default /home/lab_shared/semindex-eval). Needs no PYTHONPATH of its own; it
reads perrun.csv, spec.md's two frozen companions main10.sh and fpin.txt, predict.txt and numpy.
"""
import csv
import glob
import json
import math
import os
import re
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")

P = 0.2
BOOT_REPS = 2000
BOOT_SEED = 20261122        # spec.md's "Interval" entry, the date spec.md was written
SEEDS_WANTED = 40
SEED_FIRST, SEED_LAST = 121, 160
# spec.md's one cell. perrun.csv carries the load off each run's own meta.json, and this is what
# that column has to read for the rows to be the cell spec.md named.
WORKLOAD = "w0"
RUNS_WANTED = 200
BAND_PCT = 6.0              # spec.md pass line 4; the band ends themselves come from predict.txt

ARMS = ("B3", "B6", "B10")
RULES = ("T37r", "F")
ARM_ORDER = ARMS + RULES
SUFFIX = {"B3": "b3p695", "B6": "b6p0", "B10": "b10p043", "T37r": "t37r", "F": "f"}
ARM_LABEL = {"B3": "fixed 3.695 s, jitter 0.1", "B6": "fixed 6.0 s, jitter 0.1",
             "B10": "fixed 10.043 s, jitter 0.1",
             "T37r": "rule T37r: i_min 3.695, factor 2, i_max 10.043, reset",
             "F": "the pinned F: i_min 3.695, factor 1.5, i_max 10.043, reset"}
# The interval each fixed arm is, for naming an arm in a sentence rather than by its label.
ARM_SECONDS = {"B3": "3.695 s", "B6": "6.0 s", "B10": "10.043 s"}
# spec.md "The pinned F": the five values the winner table fixes. They are restated against
# main10.sh and fpin.txt rather than trusted, which is what pass line 1 asks for.
PINNED_F = {"i_min_s": 3.695, "factor": 1.5, "i_max_s": 10.043, "jitter_frac": 0.1,
            "on_timeout": "reset"}
# spec.md "Why the floor is 3.695 s": no coordinate of F may leave the 3.695 .. 10.043 s band, so
# the floor is checked against the launched string too.
F_FLOOR = 3.695

# The scorings, and which perrun.csv columns each one sums. Order matters: the headline scoring is
# the one spec.md attaches a verdict word to, so it is first everywhere.
SCORINGS = (("headline", ("hlat4", "hn4", "hc4", "polls_sent")),
            ("all records", ("lat4", "n4", "c4", "polls_sent")))

WORDS = ("seen", "not seen", "reversed")

# The two rules for leaving a resample out, in the order they are reported. `joint` is first and
# carries every verdict: it is predict.py's `boot_gains`, so a measured interval drawn under it and
# a predicted interval from predict.txt are the same statistic. `per quantity` is evidence/19's and
# is reported beside it, never instead of it.
DROP_RULES = ("joint", "per quantity")
# The quantities the joint rule binds together, which is predict.py's `TESTED` list and the
# difference built from it. The ratios of pass line 3 are not among them, because predict.py's
# `boot_ratios` leaves no resample out of a ratio.
JOINT_KEYS = ("gain_T37r", "gain_F", "diff")

# ---- predict.txt, copied as printed there, and checked against it at run time -------------------
# Per stage: delay s, its band ends, polls per run, its band ends, the censored share. The band ends
# are predict.txt's own printed digits rather than the median times 1 -+ 6 %, which differ in the
# third decimal; the file is what spec.md pass line 4 names.
PRED_STAGE = {
    "s10b3p695": (13.908, 13.074, 14.742, 256.68, 241.27, 272.08, 0.000),
    "s10b6p0": (20.386, 19.162, 21.609, 159.75, 150.16, 169.34, 0.000),
    "s10b10p043": (30.437, 28.611, 32.264, 96.08, 90.31, 101.84, 0.000),
    "s10t37r": (18.517, 17.406, 19.628, 165.01, 155.11, 174.91, 0.000),
    "s10f": (17.047, 16.024, 18.070, 180.95, 170.09, 191.81, 0.000),
}
# rule -> (gain %, interval lo, interval hi, half width pp, (streams seen, not seen, reversed),
# predicted word).
PRED_GAIN = {
    "T37r": (+7.17, +2.28, +12.00, 5.01, (36, 4, 0), "seen"),
    "F": (+7.80, +3.95, +11.76, 3.92, (40, 0, 0), "seen"),
}
# F minus T37r on the same frontier: pp, lo, hi, half width pp, stream tally.
PRED_DIFF = (+0.72, -3.91, +4.83, 4.52, (0, 40, 0))
# quantity -> (percent change of F against T37r, lo, hi, half width pp).
PRED_RATIO = {"delay": (-7.91, -12.21, -3.32, 4.54), "polls": (+9.81, +8.95, +10.58, 0.80)}
# The three near-tautological values of pass line 4, named by spec.md: a fixed arm's poll count
# follows from its interval and the window length, so the band says nothing about the replay there.
TAUTOLOGICAL = tuple(f"s10{SUFFIX[a]}" for a in ARMS)

# spec.md's own text for each reading of a pass line 2 row, keyed by the word. The text of the
# predicted word is carried too, because spec.md attaches a sentence to it as well and that sentence
# is what keeps a "seen" from being read as more than it is.
SPEC_TEXT2 = {
    # Keyed by the word alone, so this text is what any rule reading `seen` gets. The sentence that
    # belongs to the pinned F and to no other rule stands under the table instead, in F_SEEN_NOTE:
    # printed inside this cell it would appear on T37r's row too and read as a claim about T37r.
    "seen": "spec.md: seen is the reading when the whole interval of the gain lies above zero, and "
            "seen is the prediction for both rules.",
    "not seen": "spec.md: \"Not seen would not show that the rule adds nothing; it would say only "
                "that the interval is wider than the effect.\"",
    "reversed": "spec.md: reversed is the opposite sign from the prediction, and the reason is "
                "investigated before any other line's conclusion is written.",
    "no line": "spec.md: no line stops the plan at this step. Ticket 12 does not start its "
               "comparison on these seeds before the user has decided what the measuring stick is.",
    "incomplete": "spec.md pass line 1's parenthesis puts `incomplete` in place of a verdict "
                  "computed on fewer seeds than the line names, so no reading is attached to this "
                  "row yet and it is not compared against the predicted word.",
}
# The sentence that belongs to the pinned F alone. It is printed once under the spec.md text table
# so that it cannot be read as a statement about T37r, whose row carries the same verdict word.
F_SEEN_NOTE = ("The pinned F's row only: spec.md's prediction for F is a three-arm gain on seeds F "
               "has never seen. fpin.txt's +11.1 is a different quantity, a seven-arm gain in "
               "sample on the 20 seeds that chose the member, and this row is not read against it.")
# The `why` text of a verdict slot that reads `no line` although the rule's own full-sample gain is
# defined. See `verdict_word`: the joint rule drops a resample whenever either rule loses its
# frontier, so the other rule's losses can empty this rule's draw and leave no bound to read.
JOINT_EMPTY_WHY = ("the joint drop rule left no resample for this rule: the other rule lost its "
                   "frontier in every resample, and the joint rule drops both gains together")
# spec.md's text for each of the three readings of pass line 4, and for the case spec.md's
# parenthesis covers instead.
TEXT4 = {
    "incomplete": "spec.md pass line 1's parenthesis puts `incomplete` in place of this verdict "
                  "until every seed the line names is DONE and unflagged in all five stages, so "
                  "the count above is not yet a reading.",
    # The quotation is spec.md's sentence word for word, T37 included. The clause after it is this
    # script's own: T37 is named in spec.md's sentence but is not a stage of this step, so three of
    # the four things that sentence names are what this line gives a testbed number to.
    "predicted": "spec.md: \"The 6.0 s arm, T37, T37r and F have no testbed number in this cell at "
                 "all. Pass line 4 reads this.\" T37 is not a stage of this step, so the testbed "
                 "numbers this line reads cover the 6.0 s arm, T37r and F, and T37 still has none "
                 "in this cell.",
    "mixed": "spec.md: every value outside the band is reported with how far outside and in which "
             "direction, whatever the count is, which is the table below.",
    "missed": "spec.md: missed triggers evidence/16's procedure, the decision sequences around "
              "timeouts compared between the testbed log and the replay, and where the two part is "
              "reported before any other line's conclusion. Tickets 09 to 12 screen candidates by "
              "replay scoring, so a missed here puts the screening itself in question.",
}


# ---------------------------------------------------------------------------
# The frozen companions: predict.txt, main10.sh, fpin.txt
# ---------------------------------------------------------------------------
def check_prediction(path):
    """Stop unless every predicted constant above equals the digits predict.txt prints.

    The constants are retyped from a frozen file, and a retyped number can drift from the file it
    came from without anything failing: the analysis would then compare the measurement against a
    prediction nobody wrote. So predict.txt is parsed here and every field is compared, and a
    mismatch stops the script by name before a single measured number is computed.

    Returns the list of (field, value) pairs it matched, so that results.md can say how many of
    them were checked rather than only that the check passed.
    """
    with open(path) as f:
        text = f.read()
    got, matched = {}, []
    # The prediction table. The substitution table above it names the same stages, so the `[band]`
    # bracket is what tells the two apart: only the prediction rows carry one.
    row = re.compile(r"^\s+(s10\w+)\s*\|\s*([\d.]+)\s*\[\s*([\d.]+)\s*\.\.\s*([\d.]+)\]"
                     r"\s*[\d.]+\s*\.\.\s*[\d.]+\s*\|\s*([\d.]+)\s*\[\s*([\d.]+)\s*\.\.\s*([\d.]+)\]"
                     r"\s*[\d.]+\s*\.\.\s*[\d.]+\s*\|\s*([\d.]+)\s", re.M)
    for m in row.finditer(text):
        got[("stage", m.group(1))] = tuple(float(x) for x in m.groups()[1:])
    gain = re.compile(r"^\s+(s10t37r|s10f)\s+gain\s+([+-][\d.]+) % \[\s*([+-][\d.]+)\s*\.\.\s*"
                      r"([+-][\d.]+)\] half width\s+([\d.]+) pp \| streams seen (\d+) / not seen "
                      r"(\d+) / reversed (\d+) \| predicted word (\S+)", re.M)
    for m in gain.finditer(text):
        got[("gain", m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)),
                                     float(m.group(5)),
                                     (int(m.group(6)), int(m.group(7)), int(m.group(8))),
                                     m.group(9))
    diff = re.search(r"^\s+F minus T37r\s+([+-][\d.]+) pp \[\s*([+-][\d.]+)\s*\.\.\s*([+-][\d.]+)\]"
                     r" half width\s+([\d.]+) pp \| streams seen (\d+) / not seen (\d+) / reversed "
                     r"(\d+)", text, re.M)
    if diff:
        got[("diff", "")] = (float(diff.group(1)), float(diff.group(2)), float(diff.group(3)),
                             float(diff.group(4)),
                             (int(diff.group(5)), int(diff.group(6)), int(diff.group(7))))
    ratio = re.compile(r"^\s+F vs T37r (delay|polls) \([^)]*\)\s+([+-][\d.]+) % \[\s*([+-][\d.]+)"
                       r"\s*\.\.\s*([+-][\d.]+)\] half width\s+([\d.]+) pp", re.M)
    for m in ratio.finditer(text):
        got[("ratio", m.group(1))] = tuple(float(x) for x in m.groups()[1:])

    want = {}
    for stage, v in PRED_STAGE.items():
        want[("stage", stage)] = v
    for rule, v in PRED_GAIN.items():
        want[("gain", f"s10{SUFFIX[rule]}")] = v
    want[("diff", "")] = PRED_DIFF
    for q, v in PRED_RATIO.items():
        want[("ratio", q)] = v
    bad = []
    for key, expect in sorted(want.items()):
        if key not in got:
            bad.append(f"{key[0]} {key[1]}: no such row found in {os.path.basename(path)}")
            continue
        if got[key] != expect:
            bad.append(f"{key[0]} {key[1]}: ana.py carries {expect}, {os.path.basename(path)} "
                       f"prints {got[key]}")
        else:
            matched.append((f"{key[0]} {key[1]}", expect))
    if bad:
        raise SystemExit("the predicted constants in ana.py no longer match predict.txt:\n  "
                         + "\n  ".join(bad))
    return matched


def launched_rules(path):
    """{shell variable name: parsed --poll-rule dict} and {stage: --interval-choices list}.

    The allowed interval set of every stage is built from what main10.sh launched, not from a value
    retyped here, because what the decision logs can hold is decided by the string the sweep passed
    and by nothing else. A stage whose launcher line changed shape would show up as a missing stage
    rather than as a silently empty allowed set.
    """
    with open(path) as f:
        text = f.read()
    rules = {name: json.loads(body)
             for name, body in re.findall(r"^(\w+)='(\{.*\})'\s*$", text, re.M)}
    choices, stage_rule = {}, {}
    for line in text.splitlines():
        m = re.match(r"\s*run --stage (\S+)\s", line)
        if not m:
            continue
        stage = m.group(1)
        c = re.search(r"--interval-choices ((?:[\d.]+\s*)+)", line)
        if c:
            choices[stage] = [float(x) for x in c.group(1).split()]
        r = re.search(r'--poll-rule "\$(\w+)"', line)
        if r:
            stage_rule[stage] = r.group(1)
    return rules, choices, stage_rule


def winner_rule(path):
    """fpin.txt's winner line, parsed: the --poll-rule JSON printed under "Winner (the pinned F)".

    spec.md pass line 1 asks for the pinned F's five values to be restated against this file as well
    as against main10.sh, so that a value that moved in one of the two is visible here. The line is
    taken after the winner heading rather than by searching the whole file, because fpin.txt prints
    the same JSON again in its top-five list.
    """
    with open(path) as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Winner (the pinned F):"):
            for later in lines[i + 1:i + 4]:
                m = re.search(r"--poll-rule '(\{.*\})'", later)
                if m:
                    return json.loads(m.group(1))
            break
    raise SystemExit(f"{os.path.basename(path)}: no --poll-rule line under the winner heading")


def rule_interval_set(rule):
    """Every value a rule's interval can stand at, by `engine._advance_rule_interval`'s arithmetic.

    The walk is `i_min_s`, then `min(interval * factor, i_max_s)` repeated until it stops moving,
    which is the only way the engine ever changes the interval: an answer with nothing new backs it
    off by `factor` up to the ceiling, an answer with something new and a reset timeout put it back
    at `i_min_s`, and a keep timeout leaves it where it is. Doing the multiplication here rather
    than writing the products down is what makes the comparison against a logged `interval_s` an
    exact comparison of IEEE doubles: F's second value is 5.5424999999999995, which no rounded
    literal equals.
    """
    seen = [float(rule["i_min_s"])]
    while True:
        nxt = min(seen[-1] * float(rule["factor"]), float(rule["i_max_s"]))
        if nxt in seen:
            return tuple(seen)
        seen.append(nxt)


# ---------------------------------------------------------------------------
# perrun.csv, the store, the quantities
# ---------------------------------------------------------------------------
def load_perrun(path):
    """{(stage, seed): row} for the p 0.2 rows of perrun.csv, numeric columns as floats."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if float(r["p"]) != P:
                continue
            d = {k: (v if k in ("stage", "workload") else float(v)) for k, v in r.items()}
            out[(r["stage"], int(float(d["seed"])))] = d
    return out


def stage_name(arm):
    return f"s10{SUFFIX[arm]}"


def run_dir(stage, seed):
    return os.path.join(STORE, stage, f"p{P}", str(seed))


def validity(stage, seed):
    """(valid, flags) of one run from its meta.json `validity` block."""
    with open(os.path.join(run_dir(stage, seed), "meta.json")) as f:
        v = json.load(f)["validity"]
    return bool(v["valid"]), list(v["flags"])


def provenance(stage, seed):
    """(semindex commit, dirty flag) of one run, from its own meta.json.

    spec.md argues that this step needs no gate stage because the code did not move since
    evidence/19. That argument is about which commit the runs were made with, and every run records
    it, so reading it back here is what makes the argument checkable from this output instead of
    taken on the spec's word. A stage holding two commits, or any run made from a dirty tree, means
    the five stages were not all run from one state of the code.
    """
    with open(os.path.join(run_dir(stage, seed), "meta.json")) as f:
        m = json.load(f)
    return m["semindex_git_commit"], bool(m["semindex_git_dirty"])


def pooled(m):
    """(delay ms, polls per run, censored share) of a stage, pooled over records.

    `m` is the per-seed array of [lat, n, c, polls] in the columns of one scoring. An empty record
    set gives nan, so a stage that scored nothing is still reported instead of stopping the script:
    on a partial store that is the normal state of a stage whose runs have not started.
    """
    lat, n, c, polls = m.sum(0)
    delay = lat / n if n else float("nan")
    cens = c / (n + c) if (n + c) else float("nan")
    return delay, polls / len(m) if len(m) else float("nan"), cens


def front_delay(arm_pts, x):
    """(delay of the three-arm frontier at `x`, reason it is undefined).

    The frontier is piecewise linear in log-log through the three arm points, so it is read on the
    one segment that contains `x`. Monotone means delay strictly falls as polls per run rise. A
    sample whose arms are not monotone has no frontier at all, and spec.md asks for such a sample to
    be counted and left out, so the reason comes back with the value.
    """
    pts = sorted(arm_pts)
    ys = [y for _x, y in pts]
    if not all(y == y for y in ys) or not all(x0 == x0 for x0, _y in pts):
        return None, "stage scored nothing"
    if not (ys[0] > ys[1] > ys[2]):
        return None, "not monotone"
    if x != x:
        return None, "stage scored nothing"
    if not pts[0][0] <= x <= pts[-1][0]:
        return None, "polls outside arms"
    lo, hi = (pts[0], pts[1]) if x <= pts[1][0] else (pts[1], pts[2])
    (x0, y0), (x1, y1) = lo, hi
    w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y0) + w * (math.log(y1) - math.log(y0))), ""


def front_segment(pts_by_arm, x):
    """Which two arms the frontier segment at `x` runs between, as (lower arm, upper arm).

    `pts_by_arm` is {arm: (polls per run, delay)} for the three fixed arms, and `x` is a rule's own
    polls per run. `front_delay` reads a gain off one segment and throws the segment away; this
    returns it by arm name so that the output can say which arms a gain rests on and which arm
    never enters one. None when the frontier is undefined for `x`, under the same conditions
    `front_delay` tests.
    """
    order = sorted(ARMS, key=lambda a: pts_by_arm[a][0])
    xs = [pts_by_arm[a][0] for a in order]
    ys = [pts_by_arm[a][1] for a in order]
    if not all(v == v for v in xs + ys + [x]):
        return None
    if not (ys[0] > ys[1] > ys[2]):
        return None
    if not xs[0] <= x <= xs[-1]:
        return None
    return (order[0], order[1]) if x <= xs[1] else (order[1], order[2])


def quantities(mats):
    """Every bootstrapped quantity of one sample, as {name: value} and {name: reason undefined}.

    `mats` maps each of the five arms to its per-seed array of [lat, n, c, polls] in one scoring's
    columns. All of the quantities come from the one sample, so the intervals of one scoring share
    their resamples exactly. A None value means this sample leaves that quantity undefined, and
    `why` says which of spec.md's reasons it was.
    """
    base = {a: pooled(m) for a, m in mats.items()}
    pt = {a: (base[a][1], base[a][0]) for a in base}
    three = [pt[a] for a in ARMS]
    val, why = {}, {}
    for r in RULES:
        d, pol, _c = base[r]
        fr, reason = front_delay(three, pol)
        val[f"gain_{r}"] = None if fr is None else 1 - d / fr
        why[f"gain_{r}"] = reason
    a, b = val["gain_T37r"], val["gain_F"]
    val["diff"] = None if a is None or b is None else b - a
    why["diff"] = why["gain_T37r"] or why["gain_F"]
    # The direct comparison needs no frontier, so it is defined in every resample that scored
    # anything at all; spec.md attaches no drop rule to it.
    val["ratio_delay"] = base["F"][0] / base["T37r"][0] - 1
    val["ratio_polls"] = base["F"][1] / base["T37r"][1] - 1
    why["ratio_delay"] = why["ratio_polls"] = ""
    return val, why


def fmt(x, nd=3):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def pct(x, nd=2):
    return "-" if x is None or x != x else f"{100 * x:+.{nd}f}"


def signed(x, nd=2):
    """`x` with an explicit sign, or a dash when it is undefined.

    A deviation has to carry its direction, and spec.md asks for it on every value whatever the
    count inside the band is, so an undefined one is printed as a dash rather than skipped.
    """
    return "-" if x is None or x != x else f"{x:+.{nd}f}"


def text_key(word):
    """The key a verdict word looks its spec.md text up under.

    `incomplete` arrives carrying its seed count, so it cannot be matched whole; every other word
    is already the key.
    """
    return "incomplete" if word.startswith("incomplete") else word


def interval_word(lo, hi):
    """spec.md's three words for where an interval sits around zero.

    `lo` and `hi` must both be numbers. The caller `verdict_word` turns a bound that no resample
    was left to compute into `no line` before reaching here, so there is no fourth word.
    """
    if lo > 0:
        return WORDS[0]
    if hi < 0:
        return WORDS[2]
    return WORDS[1]


def verdict_word(n_seeds, full_value, lo, hi):
    """The verdict slot of a pass line 2 row: incomplete, no line, or one of spec.md's three words.

    spec.md pass line 1's parenthesis puts `incomplete` in place of any verdict computed on fewer
    seeds than the line names, so it is tested first and wins over everything below it. "No line" is
    read off the full sample, not off the resamples: spec.md's "no line" is the case where the arms
    of the measurement do not fall, or the rule's own polls fall outside them, and a resample that
    loses the frontier is counted and left out instead.

    The joint drop rule can still leave no resample at all behind for a rule whose own full-sample
    gain is defined: it drops a resample whenever EITHER rule loses its frontier, so the other
    rule's losses empty this rule's draw as well. There is then no interval to read a word off.
    spec.md has no word for that, and inventing one would put a reading in a verdict slot that
    spec.md never defined, so the slot reads `no line` as well and the `why` cell says which rule's
    frontier was lost. On this step's data it cannot arise: 0 of 2000 resamples are left out.
    """
    if n_seeds < SEEDS_WANTED:
        return f"incomplete ({n_seeds} of {SEEDS_WANTED} seeds)"
    if full_value is None:
        return "no line"
    if lo != lo or hi != hi:
        return "no line"
    return interval_word(lo, hi)


def decision_rows(stage, seed):
    """Every `decision` event of kind state_poll / state_poll_reschedule of one run, and late replies.

    Returns (rows, late_replies). The host logs are read in sorted order so that two runs of this
    script see the same rows in the same order. Only events.jsonl is opened under node_dirs: the
    rest of that tree holds the hosts' Cefore key material.
    """
    rows, late = [], 0
    pattern = os.path.join(run_dir(stage, seed), "debug", "node_dirs", "h*", "semindex", "events.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for ln in f:
                e = json.loads(ln)
                if e.get("event") == "state_poll_late_reply":
                    late += 1
                elif e.get("event") == "decision" and e.get("kind") in ("state_poll", "state_poll_reschedule"):
                    rows.append(e)
    return rows, late


def rule_extras(stage, seeds):
    """The no-pass-line decision-row counters of one stage, summed over `seeds`, and its histogram.

    The histogram is what pass line 1's interval check reads, and its keys are the logged doubles
    untouched: the engine writes `interval_s` as the float it computed, so a key compares exactly
    against the set `rule_interval_set` builds.
    """
    x = dict(rows=0, polls=0, resched=0, answered=0, timed_out=0, empty_true=0, empty_false=0,
             empty_null_answered=0, clamped=0, late=0)
    hist = Counter()
    for s in seeds:
        rows, late = decision_rows(stage, s)
        x["late"] += late
        for e in rows:
            a = e["action"]
            x["rows"] += 1
            hist[a["interval_s"]] += 1
            if e["kind"] == "state_poll":
                x["polls"] += 1
                continue
            x["resched"] += 1
            x["clamped"] += bool(a["clamped"])
            if a["trigger"] == "answered":
                x["answered"] += 1
                if a.get("empty") is True:
                    x["empty_true"] += 1
                elif a.get("empty") is False:
                    x["empty_false"] += 1
                else:
                    x["empty_null_answered"] += 1
            else:
                x["timed_out"] += 1
    return x, hist


class Boot:
    """The one index matrix of spec.md, and the bootstrapped quantities of one scoring.

    spec.md fixes the interval as 2000 resamples of seeds 121 to 160 with `default_rng(20261122)`,
    one index draw reused for every quantity, which is how predict.py drew its own matrix
    (`boot_index`). The unit is the seed number: one index row selects the seeds and all five stages
    are recomputed from that selection, so the stages stay paired inside a resample. A seed set
    short of a seed is drawn at its own size from a generator with the same seed, and the sizes
    drawn are recorded in `log`, which is how a partial store announces that its bounds are not the
    bounds 40 seeds would give.
    """

    def __init__(self, rows, seeds):
        self.rows = rows
        self.seeds = seeds
        self.idx = {}
        self.log = []
        self.cache = {}

    def index(self, n):
        if n not in self.idx:
            self.idx[n] = np.random.default_rng(BOOT_SEED).integers(0, n, size=(BOOT_REPS, n))
            self.log.append(n)
        return self.idx[n]

    def matrices(self, cols):
        return {a: np.array([[self.rows[(stage_name(a), s)][k] for k in cols] for s in self.seeds],
                            float).reshape(len(self.seeds), len(cols))
                for a in ARM_ORDER}

    def get(self, scoring, cols):
        """(full-sample values, reasons, ci, dropped), the last two keyed by drop rule then quantity.

        Both drop rules of DROP_RULES are computed from the one index matrix in the same pass, so
        the two sets of bounds rest on the same resampled positions and differ only in which
        resamples each rule keeps.

        `joint` is predict.py's `boot_gains`: a resample in which either rule's gain is undefined is
        left out of both gains and of their difference. That is the rule the frozen prediction's own
        intervals were drawn under, so it is the rule the verdicts read; a measured interval drawn
        the other way would be a different statistic from the one predict.txt prints.

        `per quantity` leaves a resample out of each quantity it leaves undefined and no other,
        which is evidence/19's rule. It is reported beside the joint one so that what the choice
        costs is readable rather than argued.

        The ratios of pass line 3 need no frontier: predict.py's `boot_ratios` drops nothing, so
        both rules keep every resample for them. Whether the two rules did in fact keep the same
        resamples for every quantity is the computed comparison `same_resample_set` reports.
        """
        if scoring in self.cache:
            return self.cache[scoring]
        mats = self.matrices(cols)
        full, full_why = quantities(mats)
        draws = {rule: {k: [] for k in full} for rule in DROP_RULES}
        dropped = {rule: {k: Counter() for k in full} for rule in DROP_RULES}
        if self.seeds:
            for row in self.index(len(self.seeds)):
                sub = {a: m[row] for a, m in mats.items()}
                v, w = quantities(sub)
                # predict.py's condition, over its TESTED list: the two rules' gains only. A
                # resample that loses one rule's frontier loses the other rule's gain with it.
                frontier_lost = any(v[f"gain_{r}"] is None or v[f"gain_{r}"] != v[f"gain_{r}"]
                                    for r in RULES)
                joint_reason = next((w[f"gain_{r}"] for r in RULES if w[f"gain_{r}"]), "undefined")
                for k, x in v.items():
                    undefined = x is None or x != x
                    for rule in DROP_RULES:
                        out = undefined if rule == "per quantity" else (
                            frontier_lost if k in JOINT_KEYS else undefined)
                        if out:
                            reason = (w[k] or "undefined") if rule == "per quantity" else (
                                joint_reason if k in JOINT_KEYS else (w[k] or "undefined"))
                            dropped[rule][k][reason] += 1
                        else:
                            draws[rule][k].append(x)
        ci = {}
        for rule in DROP_RULES:
            ci[rule] = {}
            for k, xs in draws[rule].items():
                ci[rule][k] = ((float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5)))
                               if xs else (float("nan"), float("nan")))
        self.cache[scoring] = (full, full_why, ci, dropped)
        return self.cache[scoring]


def main():
    matched = check_prediction(f"{HERE}/predict.txt")
    rules, choices, stage_rule = launched_rules(f"{HERE}/main10.sh")
    fpin_winner = winner_rule(f"{HERE}/fpin.txt")
    rows = load_perrun(f"{HERE}/perrun.csv")

    out = []          # results.csv rows
    L = []            # results.md lines

    def rec(line, stage, metric, value, note=""):
        out.append(dict(line=line, stage=stage, metric=metric, value=value, note=note))

    # ---- DONE and unflagged seeds of every stage ------------------------------------------------
    # DONE means the run has a row in perrun.csv: perrun.py puts only runs with a DONE marker there.
    # Unflagged is meta.json validity.
    stage_seeds, stage_flagged = {}, {}
    for arm in ARM_ORDER:
        stage = stage_name(arm)
        got = sorted(s for (st, s) in rows if st == stage)
        ok, flagged = [], {}
        for s in got:
            valid, flags = validity(stage, s)
            if valid and not flags:
                ok.append(s)
            else:
                flagged[s] = flags or ["valid=false"]
        stage_seeds[stage] = ok
        stage_flagged[stage] = flagged
    seeds = sorted(set.intersection(*[set(stage_seeds[stage_name(a)]) for a in ARM_ORDER]))
    boot = Boot(rows, seeds)
    n = len(seeds)

    L += ["# evidence/20 results", "",
          "Computed by ana.py from perrun.csv here, the frozen predict.txt, main10.sh and fpin.txt "
          "of this directory, and the stored runs' meta.json and event logs. The pass lines, the "
          "bands, the predictions and the verdict words are spec.md's; the words are read off the "
          "numbers mechanically and nothing here adds to them. Every line names the number of seeds "
          "it used, and a line computed on fewer than "
          f"{SEEDS_WANTED} seeds gets the word `incomplete` in place of its verdict.", "",
          "One cell: p 0.2, workload w0, the s6 outage shape, seeds "
          f"{SEED_FIRST} to {SEED_LAST}, {RUNS_WANTED} runs over five stages. There is no gate "
          "stage: spec.md records `git diff --stat 03f5ecc d009cfb -- src tests` and its empty "
          "output as the evidence that the code did not move since evidence/19. The semindex "
          "commit and the dirty flag each run recorded for itself are printed per stage under line "
          "1, so that argument can be read off the runs here.", "",
          "Two scorings, as spec.md fixes them. The headline scoring takes the records whose "
          "scheduled creation time is at or before t0+72.0 s, selected by the experiment "
          "description's `schedule[host][seq-1][\"at_s\"] <= 72.0` read against discovery.csv's "
          "`(origin, seq)`, and its delay is the mean over the uncensored ones 4 hops away. The "
          "all-record scoring takes every record of the run, which is the quantity every earlier "
          "step reported. Polls per run is counted over the whole window [t0+10 s, t0+130 s] under "
          "both scorings. spec.md says the two scorings select the same 240 records under `w0`; the "
          "headline scoring carries each verdict word, the all-record value stands beside it, and "
          "whether the two are equal is a measured column below and not an assumption here.", "",
          "| arm | stage | setting |", "|---|---|---|"]
    for a in ARM_ORDER:
        L.append(f"| {a} | `{stage_name(a)}` | {ARM_LABEL[a]} |")
    L += ["",
          "The predicted constants carried in ana.py were checked against predict.txt at run time: "
          f"{len(matched)} rows of predicted numbers, every number in them equal to the digits that "
          "file prints. A mismatch stops the script, so the analysis cannot drift from the frozen "
          "prediction.", ""]
    rec(0, "", "predict_txt_rows_checked", len(matched))

    # ---- line 1: all 200 runs DONE and unflagged -------------------------------------------------
    L += ["## Line 1: all 200 runs DONE and unflagged", "",
          "Validity is `<run_dir>/meta.json` -> `validity` (`valid`, `flags`), written by "
          "semindex.evalrun from semindex.evalvalid. The flag names carry the node and the sequence "
          "number (`record_off_schedule:h0:seq5`), so the breakdown below counts flags by the first "
          "word of the name and lists the names seen. It is printed whether or not any flag "
          "appeared.", "",
          "| stage | runs expected | runs DONE | valid and unflagged | flagged | flag names seen |",
          "|---|---|---|---|---|---|"]
    done_total = ok_total = 0
    flag_counts = Counter()
    for a in ARM_ORDER:
        stage = stage_name(a)
        got = sorted(s for (st, s) in rows if st == stage)
        ok, fl = stage_seeds[stage], stage_flagged[stage]
        done_total += len(got)
        ok_total += len(ok)
        flat = [f for flags in fl.values() for f in flags]
        for f in flat:
            flag_counts[(stage, f.split(":")[0])] += 1
        names = Counter(f.split(":")[0] for f in flat)
        L.append(f"| `{stage}` | {SEEDS_WANTED} | {len(got)} | {len(ok)} | {len(fl)} | "
                 + ("; ".join(f"{k} {v}" for k, v in sorted(names.items())) or "none") + " |")
        rec(1, stage, "runs_done", len(got))
        rec(1, stage, "runs_valid", len(ok))
        rec(1, stage, "runs_flagged", len(fl))
        rec(1, stage, "flag_names", ";".join(f"{k}={v}" for k, v in sorted(names.items())) or "none")
    if flag_counts:
        L += ["", "Flagged runs by stage and by flag name, and the seeds they are:", "",
              "| stage | flag | runs | seeds |", "|---|---|---|---|"]
        for (stage, name), count in sorted(flag_counts.items()):
            seeds_f = sorted(s for s, flags in stage_flagged[stage].items()
                             if any(f.split(":")[0] == name for f in flags))
            L.append(f"| `{stage}` | {name} | {count} | "
                     + ",".join(str(s) for s in seeds_f) + " |")
            rec(1, stage, f"flagged_seeds[{name}]", ",".join(str(s) for s in seeds_f))
        L += [""]
    else:
        L += ["", "No run carries a validity flag, so there is no breakdown by stage and by flag "
                  "to print.", ""]

    L += [f"{done_total} of {RUNS_WANTED} runs DONE, {ok_total} of them valid and unflagged.", "",
          "Seeds every other line is read on, the intersection of the DONE and unflagged seeds over "
          f"the five stages: {n} of {SEEDS_WANTED}"
          + (f", {min(seeds)} to {max(seeds)}" if n and n == max(seeds) - min(seeds) + 1
             else (", " + ",".join(str(s) for s in seeds) if seeds else ", none")) + ".", ""]
    rec(1, "", "runs_done_total", done_total)
    rec(1, "", "runs_valid_total", ok_total)
    rec(1, "", "seeds_used", n)

    # The interval check runs over every DONE run of every stage, not over the intersection: spec.md
    # asks it of every decision log, and a flagged run logged one too.
    L += ["Every `interval_s` in every decision log against the set its stage's setting allows. The "
          "decision log is every `decision` event of kind `state_poll` or "
          "`state_poll_reschedule` on every host, which is the whole log of the arm's or the rule's "
          "own arithmetic. The allowed set is built here from what main10.sh launches: the "
          "`--interval-choices` value for a fixed arm, and for a rule the walk `i_min_s`, "
          "`min(interval * factor, i_max_s)` repeated until it stops moving, which is the only way "
          "`engine._advance_rule_interval` ever changes the interval. The products are computed and "
          "not written down, so the comparison is on exact IEEE doubles; F's second value is "
          "5.5424999999999995 and no rounded literal equals it. This check reads every DONE run, "
          "including a flagged one. A stage whose decision logs yielded no row at all reads `no` "
          "here: the empty set is a subset of every allowed set, so a subset test on its own would "
          "pass a stage whose logs were never read and hide that the check measured nothing.", "",
          "| stage | runs read | allowed values | values seen (count) | every value allowed |",
          "|---|---|---|---|---|"]
    allowed_sets, hists, interval_ok = {}, {}, []
    for a in ARM_ORDER:
        stage = stage_name(a)
        if a in ARMS:
            allowed = tuple(choices[stage])
        else:
            allowed = rule_interval_set(rules[stage_rule[stage]])
        allowed_sets[stage] = allowed
        read = sorted(s for (st, s) in rows if st == stage)
        _x, hist = rule_extras(stage, read)
        hists[stage] = hist
        # `bool(hist)` first: a stage with zero decision rows must fail this check, because the
        # subset test alone cannot tell "every logged value is allowed" from "no value was read".
        ok = bool(hist) and set(hist) <= set(allowed)
        interval_ok.append(ok)
        L.append(f"| `{stage}` | {len(read)} | " + ", ".join(repr(v) for v in allowed) + " | "
                 + (", ".join(f"{v!r} ({hist[v]})" for v in sorted(hist)) or "none")
                 + f" | {'yes' if ok else 'no'} |")
        rec(1, stage, "interval_s_allowed", ";".join(repr(v) for v in allowed))
        rec(1, stage, "interval_s_seen", ";".join(f"{v!r}={hist[v]}" for v in sorted(hist)))
        rec(1, stage, "interval_s_exact", "yes" if ok else "no",
            f"{len(read)} runs read, {sum(hist.values())} decision rows")
        if not ok:
            rec(1, stage, "interval_s_outside_allowed",
                ";".join(f"{v!r}={hist[v]}" for v in sorted(set(hist) - set(allowed))))
    L += [""]

    # Which rows perrun.csv actually holds, and which code made them. Everything above reads
    # perrun.csv for whatever it happens to carry; these three columns are what says the rows are
    # the cell spec.md named and that one state of the code produced all five stages.
    L += ["Which seeds and which workload the rows are, and which semindex commit the runs record. "
          "The seeds must be exactly "
          f"{SEED_FIRST} to {SEED_LAST} in every stage and every row's `workload` must be "
          f"`{WORKLOAD}`, because every other line here reads perrun.csv for whatever rows it holds "
          "and would otherwise score a different cell without saying so. The commit and the dirty "
          "flag come from each run's own `meta.json`, and they are what makes spec.md's \"no gate "
          "stage\" argument checkable here: that argument says the code did not move since "
          "evidence/19, and one clean commit across all five stages is the run-by-run form of it. "
          "A stage with more than one commit, or with any run made from a dirty tree, fails this "
          "line.", "",
          "| stage | seeds DONE | seeds are "
          f"{SEED_FIRST} to {SEED_LAST} | workloads seen | workload is `{WORKLOAD}` | semindex "
          "commit | runs from a dirty tree | one clean commit |",
          "|---|---|---|---|---|---|---|---|"]
    want_seeds = list(range(SEED_FIRST, SEED_LAST + 1))
    seeds_ok, workload_ok, commit_ok = [], [], []
    for a in ARM_ORDER:
        stage = stage_name(a)
        got = sorted(s for (st, s) in rows if st == stage)
        s_ok = got == want_seeds
        loads = sorted({rows[(stage, s)]["workload"] for s in got})
        w_ok = loads == [WORKLOAD]
        prov = [provenance(stage, s) for s in got]
        commits = sorted({c for c, _d in prov})
        dirty = [s for s, (_c, d) in zip(got, prov) if d]
        c_ok = len(commits) == 1 and not dirty
        seeds_ok.append(s_ok)
        workload_ok.append(w_ok)
        commit_ok.append(c_ok)
        L.append(f"| `{stage}` | {len(got)} | {'yes' if s_ok else 'no'} | "
                 + (", ".join(f"`{x}`" for x in loads) or "none")
                 + f" | {'yes' if w_ok else 'no'} | "
                 + (", ".join(f"`{c}`" for c in commits) or "none")
                 + " | " + (",".join(str(s) for s in dirty) or "0")
                 + f" | {'yes' if c_ok else 'no'} |")
        rec(1, stage, "seeds_are_wanted", "yes" if s_ok else "no",
            f"{len(got)} DONE, wanted {SEED_FIRST} to {SEED_LAST}")
        rec(1, stage, "workloads_seen", ";".join(loads) or "none")
        rec(1, stage, "workload_is_wanted", "yes" if w_ok else "no", WORKLOAD)
        rec(1, stage, "semindex_git_commit", ";".join(commits) or "none")
        rec(1, stage, "semindex_git_dirty_runs", ",".join(str(s) for s in dirty) or "0")
        rec(1, stage, "one_clean_commit", "yes" if c_ok else "no")
    L += [""]

    # The pinned F restated against the two frozen files that carry it.
    L += ["The pinned F's five values, restated against main10.sh's launched `FPIN` string and "
          "fpin.txt's winner line, both read from those files at run time. spec.md's \"The pinned "
          "F\" table is the third column. All three must agree, and `i_min_s` must be at or above "
          f"the floor {F_FLOOR} s that spec.md and the ticket both set.", "",
          "| field | main10.sh FPIN | fpin.txt winner line | spec.md's table | agree |",
          "|---|---|---|---|---|"]
    launched_f = rules[stage_rule[stage_name("F")]]
    f_agree = []
    for field, want in PINNED_F.items():
        a_, b_ = launched_f.get(field), fpin_winner.get(field)
        same = a_ == want and b_ == want
        f_agree.append(same)
        L.append(f"| `{field}` | {a_!r} | {b_!r} | {want!r} | {'yes' if same else 'no'} |")
        rec(1, "s10f", f"pinned_F[{field}]", repr(a_),
            f"fpin.txt {b_!r}, spec.md {want!r}, agree {'yes' if same else 'no'}")
    floor_ok = float(launched_f["i_min_s"]) >= F_FLOOR
    extra_keys = sorted(set(launched_f) - set(PINNED_F))
    L += ["", f"`i_min_s` at or above the floor {F_FLOOR} s: {'yes' if floor_ok else 'no'}. Keys in "
              "the launched string beyond the five above: "
              + (", ".join(f"`{k}`" for k in extra_keys) if extra_keys else "none")
              + ". spec.md names `k_cap` as a key the family has no member with, and the launched "
                "string carries no such key.", ""]
    rec(1, "s10f", "pinned_F_floor_ok", "yes" if floor_ok else "no", f"floor {F_FLOOR} s")
    rec(1, "s10f", "pinned_F_extra_keys", ",".join(extra_keys) or "none")
    L += ["spec.md also reports that both validators on the `--poll-rule` path accept this string "
          "and T37r, with the bounds they enforce. That check was made by calling them when the "
          "member was pinned and its result stands in spec.md; it is not re-run here, and this "
          "script imports no semindex module.", ""]

    if done_total < RUNS_WANTED:
        verdict1 = f"incomplete ({done_total} of {RUNS_WANTED} runs DONE)"
    elif (ok_total < done_total or not all(interval_ok) or not all(f_agree) or not floor_ok
          or not all(seeds_ok) or not all(workload_ok) or not all(commit_ok)):
        verdict1 = "fail"
    else:
        verdict1 = "pass"
    L += [f"Verdict on {done_total} runs and {n} seeds: **{verdict1}**.", ""]
    rec(1, "", "verdict", verdict1)

    # ---- line 2: gain against the three-arm frontier ---------------------------------------------
    L += ["## Line 2: gain of T37r and of the pinned F against the three-arm fixed frontier", "",
          "The line is the three fixed arms of the same cell (3.695, 6.0, 10.043 s), polls per run "
          "on the horizontal axis and delay on the vertical, joined piecewise in log-log. Gain is "
          "`1 - delay(rule) / the line's delay at the rule's own polls per run`. The arms must "
          "form a falling line, which here means delay falls strictly as polls per run rise; a "
          "resample whose arms are not monotone, and a resample whose rule polls fall outside the "
          "outer arms, is counted and left out rather than replaced. If the full sample's arms do "
          "not fall, or a rule's own polls fall outside them, the verdict slot reads `no line`.", "",
          "The reading is attached to each rule separately and the two are not required to agree.",
          "", f"### Measured values, {n} of {SEEDS_WANTED} seeds", "",
          "| arm | stage | scoring | delay s | censored share | polls per run |",
          "|---|---|---|---|---|---|"]
    for a in ARM_ORDER:
        for scoring, cols in SCORINGS:
            d, pol, c = pooled(boot.matrices(cols)[a])
            L.append(f"| {a} | `{stage_name(a)}` | {scoring} | {fmt(d / 1000.0)} | {fmt(c, 5)} | "
                     f"{fmt(pol, 2)} |")
            rec(2, stage_name(a), f"delay_s[{scoring}]", f"{d / 1000.0:.6f}")
            rec(2, stage_name(a), f"censored_share[{scoring}]", f"{c:.6f}")
            rec(2, stage_name(a), f"polls_per_run[{scoring}]", f"{pol:.6f}")

    # ---- where the censoring is, and which arms a gain actually rests on ------------------------
    # The table above gives each stage's censored share pooled over its seeds. A pooled share says
    # nothing about whether the censoring is spread over the seeds or sits in one or two of them,
    # and that is what decides whether the stage's seeds carry equal weight in its delay. Every
    # number below is computed from perrun.csv and from the same bootstrap index matrix; none of
    # it is typed in.
    L += ["", "Where the censored 4-hop records are, seed by seed. A record is censored when it "
              "was not stored 4 hops away inside the measurement window, and the delay is the mean "
              "over the uncensored ones, so a censored record is left out of the delay instead of "
              "lengthening it. The replay predicted a share for each stage and those stand in the "
              "last column.", "",
          "| stage | scoring | seeds carrying a censored record | censored records | records | "
          "measured share | replay's predicted share |",
          "|---|---|---|---|---|---|---|"]
    censored_arms = set()
    for a in ARM_ORDER:
        stage = stage_name(a)
        for scoring, cols in SCORINGS:
            n_col, c_col = cols[1], cols[2]
            per = [(s, rows[(stage, s)][c_col]) for s in seeds if rows[(stage, s)][c_col]]
            n_rec = sum(rows[(stage, s)][n_col] + rows[(stage, s)][c_col] for s in seeds)
            n_cen = sum(rows[(stage, s)][c_col] for s in seeds)
            if n_cen and a in ARMS:
                censored_arms.add(a)
            share = n_cen / n_rec if n_rec else float("nan")
            L.append(f"| `{stage}` | {scoring} | "
                     + ("; ".join(f"{s} ({c:.0f})" for s, c in per) or "none")
                     + f" | {n_cen:.0f} | {n_rec:.0f} | {fmt(share, 5)} | "
                     f"{fmt(PRED_STAGE[stage][6], 3)} |")
            rec(2, stage, f"censored_records[{scoring}]", f"{n_cen:.0f}", f"of {n_rec:.0f} records")
            rec(2, stage, f"censored_seeds[{scoring}]",
                ";".join(f"{s}={c:.0f}" for s, c in per) or "none")

    # spec.md says that under w0 pooling over records and averaging over seeds coincide. They do
    # when every seed contributes the same number of uncensored records; censoring is what can
    # make the counts unequal, so whether the sentence holds is measured here per stage.
    uneven = []
    for a in ARM_ORDER:
        stage = stage_name(a)
        counts = {s: rows[(stage, s)][SCORINGS[0][1][1]] for s in seeds}
        if len(set(counts.values())) > 1:
            common = Counter(counts.values()).most_common(1)[0][0]
            odd = sorted((s, c) for s, c in counts.items() if c != common)
            uneven.append((stage, common, odd))
    if uneven:
        L += ["", "spec.md says that under `w0` pooling over records and averaging over seeds "
                  "coincide. That holds while every seed contributes the same number of uncensored "
                  "records, and censoring makes the counts unequal. Where they are unequal, the "
                  "pooled delay of that stage weights its seeds by how many records each one "
                  "delivered, so the sentence is not exact for it:", ""]
        for stage, common, odd in uneven:
            L.append(f"- `{stage}`: most seeds contribute {common:.0f} uncensored 4-hop records, "
                     + ", ".join(f"seed {s} contributes {c:.0f}" for s, c in odd) + ".")
            rec(2, stage, "uneven_uncensored_counts",
                ";".join(f"{s}={c:.0f}" for s, c in odd), f"other seeds {common:.0f}")
        L += [""]
    else:
        L += ["", "spec.md says that under `w0` pooling over records and averaging over seeds "
                  "coincide. Every seed of every stage contributes the same number of uncensored "
                  "4-hop records here, so the two do coincide.", ""]

    # Which segment of the frontier each rule's gain is read off, in the full sample and in the
    # resamples. A gain named against three arms is only against three arms if the segments read
    # between them cover all three; this is what says whether they do.
    cols_h = SCORINGS[0][1]
    mats_h = boot.matrices(cols_h)
    base_h = {a: pooled(m) for a, m in mats_h.items()}
    pts_full = {a: (base_h[a][1], base_h[a][0]) for a in ARMS}
    seg_full = {r: front_segment(pts_full, base_h[r][1]) for r in RULES}
    seg_count = {r: Counter() for r in RULES}
    if seeds:
        for row in boot.index(len(seeds)):
            sub = {a: pooled(m[row]) for a, m in mats_h.items()}
            p_sub = {a: (sub[a][1], sub[a][0]) for a in ARMS}
            for r in RULES:
                seg_count[r][front_segment(p_sub, sub[r][1])] += 1

    def seg_name(seg):
        return (" to ".join(ARM_SECONDS[a] for a in seg) + " arms") if seg else "no frontier"

    L += ["Which segment of the frontier each rule's gain is read off. The frontier is two segments "
          "joined at the middle arm, and a gain is read off the one segment that contains the "
          "rule's own polls per run, so an arm outside that segment does not enter the gain. The "
          f"count is over the same {BOOT_REPS} resamples the intervals are drawn from, headline "
          "scoring.", "",
          "| rule | segment in the full sample | segments over the resamples |",
          "|---|---|---|"]
    for r in RULES:
        L.append(f"| {r} | {seg_name(seg_full[r])} | "
                 + "; ".join(f"{seg_name(s)} {c} of {BOOT_REPS}"
                             for s, c in sorted(seg_count[r].items(), key=lambda kv: -kv[1]))
                 + " |")
        rec(2, "", f"frontier_segment[{r}]", seg_name(seg_full[r]))
        rec(2, "", f"frontier_segment_resamples[{r}]",
            ";".join(f"{seg_name(s)}={c}" for s, c in sorted(seg_count[r].items(),
                                                             key=lambda kv: -kv[1])))
    used = {a for r in RULES for seg in list(seg_count[r]) + [seg_full[r]] if seg for a in seg}
    unused = [a for a in ARMS if a not in used]
    if unused and len(used) == 2:
        pair = sorted(used, key=lambda a: base_h[a][1])
        L += ["", "Consequence: the "
                  + ", ".join(ARM_SECONDS[a] for a in unused)
                  + " arm enters no gain at all, in the full sample or in any resample, because no "
                    "segment either rule's gain was read off touches it"
                  + (". That arm is the only one carrying censored records, so the censoring above "
                     "reaches no gain on this cell"
                     if censored_arms and set(unused) == censored_arms else "")
                  + ". The gain called a three-arm gain is therefore in effect a gain against the "
                    "line between two arms, "
                  + " and ".join(ARM_SECONDS[a] for a in pair) + "."]
        rec(2, "", "frontier_arm_never_read", ",".join(unused),
            "no gain, full sample or resample, is read off a segment touching it")

    for scoring, cols in SCORINGS:
        full, full_why, ci, drop = boot.get(scoring, cols)
        mats = boot.matrices(cols)
        pts = sorted((pooled(mats[a])[1], pooled(mats[a])[0]) for a in ARMS)
        # An arm that scored nothing leaves no point to order, so the falling-line condition is
        # reported as not measured rather than as a line that failed to fall.
        if any(v != v for p_ in pts for v in p_):
            mono = "not measured"
        else:
            mono = "yes" if all(y0 > y1 for (_x0, y0), (_x1, y1) in zip(pts, pts[1:])) else "no"
        L += ["", f"Arm points of the full sample, {scoring} scoring, sorted by polls per run: "
                  + ", ".join(f"({fmt(x, 2)} polls, {fmt(y / 1000.0)} s)" for x, y in pts)
                  + f". The three arms form a falling line: {mono}.", ""]
        rec(2, "", f"arms_monotone[{scoring}]", mono)
        L += ["Two rules for leaving a resample out, both computed from the one index matrix. The "
              "`joint` columns are predict.py's `boot_gains`: a resample in which either rule's "
              "gain is undefined is left out of both gains and of their difference. That is the "
              "rule the frozen prediction's own intervals were drawn under, so **the joint columns "
              "carry the verdict** and an interval drawn the other way would not be the statistic "
              "predict.txt prints. The `per quantity` columns leave a resample out of each quantity "
              "it leaves undefined and no other, which is evidence/19's rule; they are printed "
              "beside the joint ones, never instead of them.", ""]
        L += ["| rule | scoring | seeds | gain % | 95 % interval % (joint) | half width pp (joint) "
              "| verdict | why no gain | predicted gain % and interval | predicted half width pp | "
              "predicted word | left out (joint) | 95 % interval % (per quantity) | half width pp "
              "(per quantity) | left out (per quantity) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in RULES:
            v = full[f"gain_{r}"]
            key = f"gain_{r}"
            lo, hi = ci["joint"][key]
            word = verdict_word(n, v, lo, hi)
            pg = PRED_GAIN[r]
            hw = 100 * (hi - lo) / 2 if hi == hi else float("nan")
            lo_q, hi_q = ci["per quantity"][key]
            hw_q = 100 * (hi_q - lo_q) / 2 if hi_q == hi_q else float("nan")
            # `full_why` is empty when the full sample has a frontier. A `no line` under that
            # condition is the joint rule's doing, and saying so is the only way the reader can
            # tell the two causes of the same word apart.
            why = full_why[key] or (JOINT_EMPTY_WHY if word == "no line" else "")
            L.append(f"| {r} | {scoring} | {n} | {pct(v)} | [{pct(lo)}, {pct(hi)}] | {fmt(hw, 2)} | "
                     f"**{word}** | {why or '-'} | {pg[0]:+.2f} [{pg[1]:+.2f}, "
                     f"{pg[2]:+.2f}] | {pg[3]:.2f} | {pg[5]} | "
                     f"{sum(drop['joint'][key].values())} | [{pct(lo_q)}, {pct(hi_q)}] | "
                     f"{fmt(hw_q, 2)} | {sum(drop['per quantity'][key].values())} |")
            tag = f"{r}:{scoring}"
            rec(2, "", f"gain[{tag}]", "" if v is None else f"{v:.6f}", why)
            rec(2, "", f"word[{tag}]", word, "joint drop rule, predict.py's own")
            rec(2, "", f"predicted_word[{tag}]", pg[5])
            for rule in DROP_RULES:
                lo_r, hi_r = ci[rule][key]
                hwr = 100 * (hi_r - lo_r) / 2 if hi_r == hi_r else float("nan")
                dn = drop[rule][key]
                rec(2, "", f"gain_lo[{tag}:{rule}]", f"{lo_r:.6f}")
                rec(2, "", f"gain_hi[{tag}:{rule}]", f"{hi_r:.6f}")
                rec(2, "", f"gain_half_width_pp[{tag}:{rule}]", f"{hwr:.4f}")
                rec(2, "", f"boot_dropped[{tag}:{rule}]", sum(dn.values()),
                    "; ".join(f"{k} {c}" for k, c in sorted(dn.items())) or "none")
        vd = full["diff"]
        lod, hid = ci["joint"]["diff"]
        wd = verdict_word(n, vd, lod, hid)
        lod_q, hid_q = ci["per quantity"]["diff"]
        pdq = PRED_DIFF
        L += ["", f"F minus T37r on the same frontier, {scoring} scoring, on {n} seeds: {pct(vd)} pp "
                  f"[{pct(lod)}, {pct(hid)}], half width {fmt(100 * (hid - lod) / 2, 2)} pp, "
                  f"reading **{wd}**, resamples left out {sum(drop['joint']['diff'].values())}, all "
                  "under the joint drop rule. Under the per quantity rule the same difference reads "
                  f"[{pct(lod_q)}, {pct(hid_q)}], half width "
                  f"{fmt(100 * (hid_q - lod_q) / 2, 2)} pp, resamples left out "
                  f"{sum(drop['per quantity']['diff'].values())}. The replay predicted "
                  f"{pdq[0]:+.2f} pp [{pdq[1]:+.2f}, {pdq[2]:+.2f}], half width {pdq[3]:.2f} pp, "
                  f"with {pdq[4][0]} of 40 streams entirely above zero. spec.md attaches no pass "
                  "line to this difference.", ""]
        rec(2, "", f"diff[{scoring}]", "" if vd is None else f"{vd:.6f}")
        rec(2, "", f"diff_word[{scoring}]", wd, "joint drop rule, predict.py's own")
        for rule in DROP_RULES:
            lo_r, hi_r = ci[rule]["diff"]
            rec(2, "", f"diff_lo[{scoring}:{rule}]", f"{lo_r:.6f}")
            rec(2, "", f"diff_hi[{scoring}:{rule}]", f"{hi_r:.6f}")
            rec(2, "", f"diff_dropped[{scoring}:{rule}]", sum(drop[rule]["diff"].values()))
    L += ["The verdict of a rule is the headline scoring's; the all-record word stands beside it in "
          "the tables above. What spec.md says about each reading:", "",
          "| rule | seeds | headline verdict | all-record verdict | predicted word | spec.md's text "
          "for the headline reading |", "|---|---|---|---|---|---|"]
    unexpected = []
    for r in RULES:
        got_words = {}
        for scoring, cols in SCORINGS:
            full, _w, ci, _d = boot.get(scoring, cols)
            # The verdict reads the joint drop rule, predict.py's own.
            lo, hi = ci["joint"][f"gain_{r}"]
            got_words[scoring] = verdict_word(n, full[f"gain_{r}"], lo, hi)
        pw = PRED_GAIN[r][5]
        head = got_words["headline"]
        text = SPEC_TEXT2.get(text_key(head), "No text in spec.md for this reading.")
        # An `incomplete` row is not a reading that differs from the prediction, so it is not sent
        # to the list spec.md asks to be investigated; it is a row with no reading yet.
        if text_key(head) != "incomplete" and head != pw:
            unexpected.append((r, head, pw))
        L.append(f"| {r} | {n} | **{head}** | {got_words['all records']} | {pw} | {text} |")
        rec(2, "", f"spec_text[{r}]", text)
    L += ["", F_SEEN_NOTE, ""]
    rec(2, "s10f", "spec_text_F_only", F_SEEN_NOTE)
    if unexpected:
        L += ["Rows whose verdict is not the predicted word, which is what spec.md sends to be "
              "investigated or, for `no line`, what stops the plan at this step:", ""]
        for r, head, pw in unexpected:
            L.append(f"- {r}: read **{head}**, predicted {pw}. "
                     f"{SPEC_TEXT2.get(text_key(head), '')}")
            rec(2, "", f"unexpected_word[{r}]", head, f"predicted {pw}")
        L += [""]
    elif n < SEEDS_WANTED:
        L += ["Neither row has a reading yet, so neither can differ from the predicted word.", ""]
    else:
        L += ["Both rules read the word spec.md predicted, so neither sends the step to investigate "
              "a reason.", ""]

    # ---- line 3: F against T37r directly ---------------------------------------------------------
    L += ["## Line 3: the pinned F against T37r directly", "",
          "The delay ratio and the polls ratio, from the same resamples, printed as the percent "
          "change of F against T37r so that they sit beside the replay's numbers. The censored "
          "share of each rule stands beside the delay below. spec.md attaches no pass or fail to "
          "this line: the two rules differ by one coordinate (factor 1.5 against 2), so this is the "
          "measurement of a size and not a test of a direction. The reading is whether each ratio "
          "spans zero. The reading is the joint drop rule's, as pass line 2's is, so that the two "
          "lines rest on one resample set; the per quantity interval stands beside it.", "",
          "| quantity | scoring | seeds | measured % | 95 % interval % (joint) | measured half "
          "width pp (joint) | predicted % and interval | predicted half width pp | spans zero | "
          "95 % interval % (per quantity) | half width pp (per quantity) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, label in (("ratio_delay", "delay"), ("ratio_polls", "polls")):
        for scoring, cols in SCORINGS:
            full, _why, ci, drop = boot.get(scoring, cols)
            v = full[key]
            lo, hi = ci["joint"][key]
            hw = 100 * (hi - lo) / 2
            lo_q, hi_q = ci["per quantity"][key]
            hw_q = 100 * (hi_q - lo_q) / 2
            pr = PRED_RATIO[label]
            if n < SEEDS_WANTED:
                spans = f"incomplete ({n} of {SEEDS_WANTED} seeds)"
            elif lo != lo or hi != hi:
                spans = "not computed"
            else:
                spans = "yes" if lo <= 0 <= hi else "no"
            L.append(f"| {label} | {scoring} | {n} | {pct(v)} | [{pct(lo)}, {pct(hi)}] | "
                     f"{fmt(hw, 2)} | {pr[0]:+.2f} [{pr[1]:+.2f}, {pr[2]:+.2f}] | {pr[3]:.2f} | "
                     f"{spans} | [{pct(lo_q)}, {pct(hi_q)}] | {fmt(hw_q, 2)} |")
            tag = f"{label}:{scoring}"
            rec(3, "", f"ratio_pct[{tag}]", f"{100 * v:.6f}")
            rec(3, "", f"spans_zero[{tag}]", spans, "joint drop rule, predict.py's own")
            for rule in DROP_RULES:
                lo_r, hi_r = ci[rule][key]
                rec(3, "", f"ratio_lo_pct[{tag}:{rule}]", f"{100 * lo_r:.6f}")
                rec(3, "", f"ratio_hi_pct[{tag}:{rule}]", f"{100 * hi_r:.6f}")
                rec(3, "", f"ratio_half_width_pp[{tag}:{rule}]", f"{100 * (hi_r - lo_r) / 2:.4f}")
                rec(3, "", f"ratio_dropped[{tag}:{rule}]", sum(drop[rule][key].values()))
    L += ["", "predict.py's `boot_ratios` leaves no resample out of a ratio, so neither drop rule "
              "drops one here either and the two pairs of columns above are expected to be equal "
              "number for number. Whether they are is a computed comparison in the resample table "
              "near the end of this file, not an assumption here.", "",
          "The polls ratio is the same number under both scorings, because polls are counted "
              "over the whole window whichever records the delay is taken on. predict.txt prints it "
              "once for that reason and the same predicted numbers stand under both rows here.", "",
          "The censored share of each rule, beside the delays the ratio is built from:", "",
          "| stage | scoring | delay s | censored share |", "|---|---|---|---|"]
    for r in RULES:
        for scoring, cols in SCORINGS:
            d, _pol, c = pooled(boot.matrices(cols)[r])
            L.append(f"| `{stage_name(r)}` | {scoring} | {fmt(d / 1000.0)} | {fmt(c, 5)} |")
    L += [""]

    # ---- line 4: did the replay predict the measurement? -----------------------------------------
    L += ["## Line 4: did the replay predict the measurement?", "",
          "The delay and the polls per run over the whole window of all five stages, ten values, "
          "each against the plus or minus "
          f"{BAND_PCT:.0f} % band of predict.txt, copied from that file as printed and checked "
          "against it at run time. 9 or more of the 10 inside reads `predicted`, exactly 8 reads "
          "`mixed`, 7 or fewer reads `missed`. spec.md records that the band comes from the largest "
          "deviation the replay showed on the default load, 5.9 % (evidence/13), measured on four "
          "arms and two policies that do not include the 6.0 s arm, T37r or F, so every value "
          "outside it is reported with how far outside and in which direction whatever the count "
          "is. The band is a Mode B band on per-stage aggregates and says nothing about agreement "
          "decision by decision.", "",
          "| stage | quantity | informative | measured | prediction | band | inside or outside | "
          "deviation % |", "|---|---|---|---|---|---|---|---|"]
    inside = inside_info = n_info = 0
    outside_list = []
    cols_head = SCORINGS[0][1]
    for a in ARM_ORDER:
        stage = stage_name(a)
        d, pol, _c = pooled(boot.matrices(cols_head)[a])
        p = PRED_STAGE[stage]
        for metric, value, mid, lo_b, hi_b, nd, info in (
            ("delay s", d / 1000.0, p[0], p[1], p[2], 3, True),
            ("polls per run", pol, p[3], p[4], p[5], 2, stage not in TAUTOLOGICAL),
        ):
            # A stage that scored nothing has no measured value, and calling that "outside" would
            # read as a measurement that missed the band. It is counted as neither.
            word = ("not measured" if value != value
                    else "inside" if lo_b <= value <= hi_b else "outside")
            inside += word == "inside"
            n_info += info
            inside_info += info and word == "inside"
            dev = 100 * (value / mid - 1) if mid else float("nan")
            if word == "outside":
                outside_list.append((stage, metric, value, mid, dev))
            L.append(f"| `{stage}` | {metric} | {'yes' if info else 'no'} | {fmt(value, nd)} | "
                     f"{fmt(mid, nd)} | {fmt(lo_b, nd)} .. {fmt(hi_b, nd)} | {word} | "
                     f"{signed(dev)} |")
            rec(4, stage, f"{metric}_measured", f"{value:.6f}")
            rec(4, stage, f"{metric}_band", word, f"{lo_b} .. {hi_b}")
            rec(4, stage, f"{metric}_deviation_pct", f"{dev:.4f}")
            rec(4, stage, f"{metric}_informative", "yes" if info else "no")
    if n < SEEDS_WANTED:
        verdict4 = f"incomplete ({n} of {SEEDS_WANTED} seeds)"
    elif inside >= 9:
        verdict4 = "predicted"
    elif inside == 8:
        verdict4 = "mixed"
    else:
        verdict4 = "missed"
    L += ["", f"{inside} of 10 values inside their band. Verdict on {n} seeds: **{verdict4}**.", "",
          TEXT4.get(text_key(verdict4), ""), "",
          "spec.md names three of the ten values near-tautological: a fixed arm's poll count "
          "follows from its interval and the window length, so the three fixed arms' polls are "
          "expected inside the band whatever the replay does. The seven values that can inform are "
          f"the five delays and the two rules' poll counts, and {inside_info} of {n_info} of them "
          "are inside their band. The verdict word above is spec.md's, read on the ten; this "
          "breakdown is reported beside it and carries no word of its own.", ""]
    rec(4, "", "values_inside", inside)
    rec(4, "", "values_informative_inside", inside_info)
    rec(4, "", "values_informative", n_info)
    rec(4, "", "seeds_used", n)
    rec(4, "", "verdict", verdict4)
    L += ["Every value outside its band, one by one, with how far outside and in which direction. "
          "spec.md asks for this whatever the count is, so the table is printed even when it is "
          "empty. A value outside the band is also read against the numbers that bound the "
          "substitution: on seeds 1 to 20 the generated schedules moved the delay by at most 0.3 % "
          "and a borrowed clock by at most 1.1 % (predict.txt), so a miss much larger than their "
          "sum is the replay's and not the borrowing's.", ""]
    if outside_list:
        for stage, metric, value, mid, dev in outside_list:
            far = abs(dev) - BAND_PCT
            L.append(f"- `{stage}` {metric}: measured {value:.3f}, predicted {mid:.3f}, "
                     f"{dev:+.2f} % against the prediction, which is {far:.2f} points "
                     f"{'above' if dev > 0 else 'below'} the {BAND_PCT:.0f} % band.")
            rec(4, stage, f"outside[{metric}]", f"{dev:.4f}", f"predicted {mid}")
        L += [""]
    else:
        L += ["- no value fell outside its band.", ""]
    L += ["The censored share beside every delay, which spec.md asks for on all five stages, is in "
          "line 2's measured table. The comparison of decision sequences that spec.md asks for when "
          "this line reads `mixed` or `missed` is not computed here: it is a reading of two logs "
          "against each other, which this script does not do. The word this line read is printed "
          "above, so the reader can see whether spec.md asks for that comparison at all.", ""]

    # ---- also reported, without a pass line ------------------------------------------------------
    L += ["## Also reported, without a pass line", "",
          "Every stage's measured values beside the replay's, both scorings together, and whether "
          "the two scorings came out equal. spec.md says they must under `w0` because every record "
          "is due at or before t0+71 s; the last two columns are the measurement of that.", "",
          "| stage | seeds | headline delay s | headline censored | all-record delay s | "
          "all-record censored | polls per run | replay delay s | replay polls | replay censored | "
          "headline equals all records |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARM_ORDER:
        stage = stage_name(a)
        dh, pol, ch = pooled(boot.matrices(SCORINGS[0][1])[a])
        da, _p2, ca = pooled(boot.matrices(SCORINGS[1][1])[a])
        p = PRED_STAGE[stage]
        same = (round(dh, 6), round(ch, 9)) == (round(da, 6), round(ca, 9))
        L.append(f"| `{stage}` | {n} | {fmt(dh / 1000.0)} | {fmt(ch, 5)} | {fmt(da / 1000.0)} | "
                 f"{fmt(ca, 5)} | {fmt(pol, 2)} | {fmt(p[0])} | {fmt(p[3], 2)} | {fmt(p[6])} | "
                 f"{'yes' if same else 'no'} |")
        rec(0, stage, "headline_delay_s", f"{dh / 1000.0:.6f}")
        rec(0, stage, "all_delay_s", f"{da / 1000.0:.6f}")
        rec(0, stage, "headline_censored_share", f"{ch:.6f}")
        rec(0, stage, "all_censored_share", f"{ca:.6f}")
        rec(0, stage, "polls_per_run", f"{pol:.6f}")
        rec(0, stage, "headline_equals_all_records", "yes" if same else "no")

    # The three perrun.py counters that say the measurement and the prediction rest on the same
    # records and the same censoring rule. They are totals over every DONE run, flagged or not.
    L += ["", "Three counters from perrun.csv, over every DONE run of the five stages. "
              "`sched_regen_diff` counts a run's record origins whose instants the committed "
              "generator does not draw again for that run's own (host, seed, k, workload); no run "
              "of these seeds existed when predict.py drew the prediction, so predict.py built the "
              "record schedule from that same generator, and this counter is what says run by run "
              "that the testbed ran the instants the prediction scored. The two `missing_disc` "
              "counters count records the schedule names that never appeared in the aggregation; a "
              "nonzero count would deflate a run's sums silently. All three are expected to be "
              "zero.", "",
          "| counter | total over all DONE runs | runs with a nonzero count |", "|---|---|---|"]
    for c in ("sched_regen_diff", "head_missing_disc", "all_missing_disc"):
        bad = sorted(k for k, r in rows.items() if r[c])
        L.append(f"| `{c}` | {sum(r[c] for r in rows.values()):.0f} | {len(bad)} of {len(rows)}"
                 + ("" if not bad else ": " + ", ".join(f"`{st}` seed {s}" for st, s in bad)) + " |")
        rec(0, "", f"{c}_total", f"{sum(r[c] for r in rows.values()):.0f}",
            f"{len(bad)} of {len(rows)} runs nonzero")
    L += ["", "Records the schedule lays down per run, from the schedule each run was handed. "
              "predict.txt puts the held-out schedules at 240.0 [240 .. 240] records per run and "
              "the same for the headline cut, so the headline cut taking fewer than all of them "
              "would be visible here. All five stages of one seed are handed the same schedule, so "
              "one row per seed is the whole of it, and the `stages agree` column says that every "
              "stage of that seed carried the same two counts.", "",
          "| seed | all records | headline records | stages agree | instants are the generator's "
          "draw |", "|---|---|---|---|---|"]
    for s in seeds:
        counts = {(rows[(stage_name(a), s)]["sched_all"], rows[(stage_name(a), s)]["sched_head"])
                  for a in ARM_ORDER}
        one = sorted(counts)[0]
        bad = sum(rows[(stage_name(a), s)]["sched_regen_diff"] for a in ARM_ORDER)
        L.append(f"| {s} | {one[0]:.0f} | {one[1]:.0f} | {'yes' if len(counts) == 1 else 'no'} | "
                 f"{'yes' if not bad else 'no'} |")
        rec(0, "", f"sched_all[seed {s}]", f"{one[0]:.0f}")
        rec(0, "", f"sched_head[seed {s}]", f"{one[1]:.0f}")

    L += ["", "Whole-run counts over every decision row of all five stages, not cut to the "
              "measurement window, because what they describe is the rule's own arithmetic over "
              "every poll the run sent. The three fixed arms have no re-decision at all: a choices "
              "arm picks its interval once per poll and never re-decides, so their rows are all "
              "zero past the poll count and that is what says the arms were run as fixed arms. An "
              "answered re-decision carries `empty`: true when the "
              "reply held nothing new, false when it held something new, null when the reply could "
              "not be decoded. spec.md's \"One difference between the replay and the testbed\" is "
              "about exactly that null case, so its count is printed whether or not it is zero. A "
              "timed-out re-decision carries `empty` null by construction and is not counted among "
              "the answered ones. These counters are read over every DONE run, as the interval "
              "check above is.", "",
          "| stage | runs | polls | re-decisions | answered | timed out | empty true share of "
          "answered | empty false | empty null and answered | clamped | late replies |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARM_ORDER:
        stage = stage_name(a)
        read = sorted(s for (st, s) in rows if st == stage)
        x, _hist = rule_extras(stage, read)
        share = x["empty_true"] / x["answered"] if x["answered"] else float("nan")
        L.append(f"| `{stage}` | {len(read)} | {x['polls']} | {x['resched']} | {x['answered']} | "
                 f"{x['timed_out']} | {fmt(share, 4)} | {x['empty_false']} | "
                 f"{x['empty_null_answered']} | {x['clamped']} | {x['late']} |")
        for k, v in sorted(x.items()):
            rec(0, stage, f"extra_{k}", v)
        rec(0, stage, "extra_empty_true_share_of_answered", f"{share:.6f}")

    # ---- how the bootstrap was drawn -------------------------------------------------------------
    L += ["", "## How the bootstrap was drawn", "",
          f"One index matrix, `numpy.random.default_rng({BOOT_SEED}).integers(0, n, size="
          f"({BOOT_REPS}, n))`, drawn once at the seed count `n` the intersection has and reused for "
          "every quantity and both scorings. That is how predict.py drew its own matrix, so the "
          "measured bounds and the predicted bounds rest on the same resampled positions. The unit "
          "is the seed number: one index row selects the seeds, and all five stages are recomputed "
          "from that one selection, so a resample holds the same seeds in every stage at once and "
          "the stages stay paired. A seed set short of a seed is drawn at its own size from a "
          f"generator with the same seed, which is why a run on fewer than {SEEDS_WANTED} seeds "
          "gives bounds that are not the bounds predict.txt was drawn against; every size drawn is "
          "listed here.", "",
          "| index matrix | seeds per resample | resamples |", "|---|---|---|"]
    for i, size in enumerate(boot.log, 1):
        L.append(f"| {i} | {size} | {BOOT_REPS} |")
        rec(0, "bootstrap", f"draw_{i}", size, f"{BOOT_REPS} resamples")
    if not boot.log:
        L.append("| - | no seed is DONE and unflagged in all five stages, so no matrix was drawn "
                 "| - |")
    L += ["", "Resamples left out of every interval, under both drop rules, with the reason each one "
              "was left out. A gain is undefined in a resample whose three arms do not fall and in "
              "one whose rule polls fall outside the arms. Under the `joint` rule a resample in "
              "which either rule's gain is undefined is left out of both gains and of their "
              "difference, which is predict.py's `boot_gains` and the rule every verdict above "
              "reads. Under `per quantity` it is left out of each quantity it leaves undefined and "
              "no other. spec.md counts such a resample and leaves it out; no value is substituted "
              "for it.", "",
          "| scoring | quantity | drop rule | resamples left out | of | reasons |",
          "|---|---|---|---|---|---|"]
    for scoring, cols in SCORINGS:
        _f, _w, _ci, drop = boot.get(scoring, cols)
        for k in sorted(drop["joint"]):
            for rule in DROP_RULES:
                why = "; ".join(f"{r} {c}" for r, c in sorted(drop[rule][k].items())) or "none"
                L.append(f"| {scoring} | {k} | {rule} | {sum(drop[rule][k].values())} | "
                         f"{BOOT_REPS if boot.log else 0} | {why} |")
                rec(0, "", f"dropped[{scoring}:{k}:{rule}]", sum(drop[rule][k].values()), why)
    # Whether the two rules kept the same resamples, computed rather than assumed. A resample the
    # joint rule drops is one in which some rule's gain is undefined, so the per quantity rule drops
    # it from at least that quantity: the per quantity kept set of a quantity always contains the
    # joint one. Equal counts therefore mean equal sets, and that is what is compared here.
    L += ["", "Whether the two rules kept the same resamples, per scoring and per quantity. The "
              "joint rule drops a resample only when some rule's gain is undefined, and the per "
              "quantity rule then drops that same resample from at least that quantity, so each "
              "quantity's per quantity kept set contains its joint one. Equal counts therefore mean "
              "equal sets, and the counts are compared here rather than assumed equal.", "",
          "| scoring | quantity | joint left out | per quantity left out | same resample set |",
          "|---|---|---|---|---|"]
    for scoring, cols in SCORINGS:
        _f, _w, _ci, drop = boot.get(scoring, cols)
        for k in sorted(drop["joint"]):
            nj = sum(drop["joint"][k].values())
            nq = sum(drop["per quantity"][k].values())
            L.append(f"| {scoring} | {k} | {nj} | {nq} | {'yes' if nj == nq else 'no'} |")
            rec(0, "", f"same_resample_set[{scoring}:{k}]", "yes" if nj == nq else "no",
                f"joint {nj}, per quantity {nq}")

    # Whether the joint rule left any resample out at all, over both scorings and the three
    # quantities it binds. The bullet below says which of its two cases this run is in, so that
    # sentence is read off the data rather than typed into the script.
    joint_dropped = sum(sum(boot.get(sc, co)[3]["joint"][k].values())
                        for sc, co in SCORINGS for k in JOINT_KEYS)
    L += ["", "## Where spec.md left a choice", "",
          "These are the places spec.md leaves a choice open and what this script took in each. "
          "Most of them are the plainest reading of spec.md's own text. The drop rule below is not: "
          "spec.md's text reads one quantity at a time and says nothing about binding the "
          "quantities together, and the rule taken there is the frozen predict.py's. None of them "
          "moves a pass line, a band or a verdict word.", "",
          "- The verdict of a pass line 2 or line 3 row is the headline scoring's. spec.md's "
          "Quantities makes the headline the scoring the pass lines read and says the all-record "
          "value is reported anyway, so the all-record word is printed beside the headline word in "
          "every table. Under `w0` the two select the same records, and the `headline equals all "
          "records` column above is the measurement of that.", "",
          "- The seed set is the intersection of the DONE and unflagged seeds over the five stages, "
          "because spec.md's bootstrap recomputes all five stages from the same resampled seeds. "
          "Line 1's own interval check and the decision-row counters are read over every DONE run "
          "instead, flagged or not, because spec.md asks the interval check of every decision log.",
          "",
          "- Which resamples a quantity's interval is drawn from. spec.md asks for a resample that "
          "loses the frontier to be counted and left out, which reads one quantity at a time; it "
          "leaves open whether a resample that one rule loses is also left out of the other rule's "
          "gain. The rule taken here is the frozen predict.py's `boot_gains`, which leaves a "
          "resample out of both rules' gains and of their difference whenever either rule's gain is "
          "undefined, while evidence/19's ana.py leaves it out of each quantity separately. It is "
          "taken because the frozen prediction's own intervals were drawn under it, so the measured "
          "interval and the predicted interval are the same statistic; an interval drawn per "
          "quantity would be a different one from what predict.txt prints. The joint rule therefore "
          "carries every verdict, every printed interval that a "
          "verdict reads and every half width of pass lines 2 and 3, and the per quantity interval "
          "and drop count are printed beside it in the same tables. predict.py's `boot_ratios` "
          "drops no resample from a ratio, so the two rules are expected to agree on pass line 3; "
          "the `same resample set` table above compares their counts instead of assuming it.", "",
          "- `no line` is read off the full sample, not off the resamples: spec.md's `no line` is "
          "the case where the arms of the measurement do not fall or a rule's own polls fall "
          "outside them, while a resample that loses the frontier is counted and left out. "
          "`incomplete` is tested before `no line`, because spec.md's parenthesis puts it in place "
          f"of any verdict computed on fewer than {SEEDS_WANTED} seeds.", "",

          "- A rule whose own full-sample gain is defined but whose every resample the joint drop "
          "rule left out, because the OTHER rule lost its frontier in all of them. There is then no "
          "interval to read a word off, and spec.md has no word for that case. The slot reads `no "
          "line` as well, with a `why` cell saying the joint rule left no resample because the "
          "other rule lost its frontier; a new word would be a reading spec.md never defined. The "
          "same applies to the F minus T37r sentence, which has no `why` cell of its own. "
          + ("On this step's data it did not arise: the joint rule left "
             f"{joint_dropped} resamples out of {BOOT_REPS}, over both scorings and all three "
             "quantities it binds together."
             if not joint_dropped else
             "On this step's data the joint rule did leave resamples out: "
             f"{joint_dropped} of {BOOT_REPS}, over both scorings and all three quantities it "
             "binds together, so the tables above are where to read whether any slot hit it."),
          "",
          "- The bands of line 4 are predict.txt's own printed band ends, not the printed median "
          "times 1 -+ 6 %. The two differ in the third decimal because predict.txt rounds the "
          "median after forming the band. spec.md pass line 4 says \"the prediction tabled above\", "
          "and predict.txt is the file that table was printed from.", "",
          "- Line 4 reads the delay of the headline scoring. predict.txt prints one delay per stage "
          "and says the two scorings give the same digits under `w0`, so there is one predicted "
          "number to compare against and the all-record delay stands beside it in the table above.",
          "",
          "- The allowed interval set of a stage is built from the string main10.sh launches, not "
          "from the values spec.md lists. spec.md lists the same sets, and building them from the "
          "launcher is what makes the check a check of what the sweep ran. The fixed arms' sets "
          "come from `--interval-choices`, which is logged as the nominal value because the jitter "
          "of 0.1 is applied after the choice is logged.", "",
          "- spec.md reports that both validators on the `--poll-rule` path accept the pinned F and "
          "T37r, with the bounds they enforce. That was checked by calling them when the member was "
          "pinned; this script restates the five values against main10.sh and fpin.txt and imports "
          "no semindex module, so the validators are not called again here.", "",
          "- spec.md gives no reading for the case where a stage has no scored record at all. On a "
          "partial store that happens, and such a stage is reported as `-` rather than as a number, "
          "with the resamples that lose the frontier counted under `stage scored nothing`. In line "
          "4 such a value reads `not measured` and counts as neither inside nor outside its band, "
          "because calling it outside would read as a measurement that missed.", ""]

    with open(f"{HERE}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["line", "stage", "metric", "value", "note"])
        w.writeheader()
        w.writerows(out)
    with open(f"{HERE}/results.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
