"""The five pass lines of spec.md for evidence/19, from perrun.csv here and the stored runs.

Writes results.csv (one row per reported quantity) and results.md (the same numbers as tables), and
prints results.md on stdout. Every verdict word in both is the mechanical one spec.md defines;
nothing here reads more into a number than that.

The thirteen s9 stages and the logical arm each one is:

    gate   `s9w0_b3p695`, `s9w0_b10p043`, `s9w0_t37r`   the default load, seed 1 (pass line 1)
    wo     `s9wo_b3p695`, `s9wo_b6p0`, `s9wo_b10p043`, `s9wo_t37`, `s9wo_t37r`   seeds 1 to 20
    wv     `s9wv_b3p695`, `s9wv_b6p0`, `s9wv_b10p043`, `s9wv_t37`, `s9wv_t37r`   seeds 1 to 20

The two scorings of spec.md:

    headline      the records whose scheduled creation time is at or before t0+72.0 s, delay the
                  mean over the uncensored ones 4 hops away (`hlat4 / hn4`), censored share
                  `hc4 / (hn4 + hc4)`
    all records   every record of the run, the quantity every earlier step reported
                  (`lat4 / n4`, `c4 / (n4 + c4)`)

Polls per run is the mean of `polls_sent` over the window [t0+10 s, t0+130 s] under both scorings,
because spec.md counts polls over the whole window whichever set of records the delay is taken on.
Delay is pooled over records, which is the pooling spec.md names and the one predict.py scored with,
so a seed with more records weighs more. Under an on/off load that weighting matters more than it
did before, since the number of records per run moves by more than a factor of two across seeds.

What each pass line needs:

line 1  The three gate runs: every `interval_s` in the decision log inside the set the setting
        allows, the two fixed arms inside spec.md's polls and delay bands, the censored share
        beside each delay, and `s9w0_t37r` reported beside the original `s7t37r` seed-1 run of
        ../16/perrun.csv.
line 2  All 200 sweep runs DONE and valid, per stage, with the count of `record_off_schedule` and
        `record_count` flags per stage printed even when it is zero.
line 3  The gain of each rule against the three-arm log-log frontier of its own load, per (load,
        rule) row, both scorings, with the 95 % seed-bootstrap interval, the verdict word, the
        prediction beside it, and spec.md's own instruction text when the word is not the predicted
        one.
line 4  T37r against T37 directly, per load: the delay ratio and the polls ratio with intervals and
        measured half widths beside the predicted ones, and the censored shares.
line 5  The headline delay and the polls per run of all ten sweep stages against the plus or minus
        6 % bands of spec.md, 20 values, with the three-way word. Beside the word, the count of
        runs whose scheduled instants are the generator's own draw (perrun.csv's
        `sched_regen_diff`), which is what carries spec.md's note that the testbed and the replay
        ran the same instants.

Seed sets (spec.md pass line 2, taken from evidence/17's amendment 1): a line's seeds are the seeds
that are DONE and unflagged in every stage that line names. For a load that is the intersection
over its five stages, because spec.md's bootstrap recomputes all five stages of a load from the same
resampled seeds. The number of seeds is printed beside every verdict, and a line computed on fewer
seeds than it names reads `incomplete` in place of its verdict.

Bootstrap: one index matrix `default_rng(20261117).integers(0, n, size=(2000, n))`, drawn once and
reused for both loads and both scorings, which is how predict.py drew its own (`boot_index`). The
unit is the seed number: one index row selects the seeds and all five stages of the load are
recomputed from that one selection, so the stages stay paired inside a resample.

Reads only `<run_dir>/meta.json` and `<run_dir>/debug/node_dirs/h*/semindex/events.jsonl` under the
store, and writes nothing there.

Environment: STORE (default /home/lab_shared/semindex-eval). Needs no PYTHONPATH of its own; it
reads perrun.csv here, ../16/perrun.csv and numpy only.
"""
import csv
import glob
import json
import math
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV16 = os.path.join(HERE, os.pardir, "16")
STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")

P = 0.4
BOOT_REPS = 2000
BOOT_SEED = 20261117        # spec.md's "Interval" entry; predict.py's BOOT_SEED
SEEDS_WANTED = 20

ARMS = ("B3", "B6", "B10")
RULES = ("T37", "T37r")
ARM_ORDER = ARMS + RULES
SUFFIX = {"B3": "b3p695", "B6": "b6p0", "B10": "b10p043", "T37": "t37", "T37r": "t37r"}
ARM_LABEL = {"B3": "fixed 3.695 s", "B6": "fixed 6.0 s", "B10": "fixed 10.043 s",
             "T37": "rule, on_timeout keep", "T37r": "rule, on_timeout reset"}
LOADS = ("wo", "wv")
LOAD_LABEL = {"wo": "ON 20 s / OFF 20 s alternating", "wv": "ON and OFF lengths exponential, mean 20 s"}

# The three gate stages, the set of `interval_s` values each one's setting allows, and spec.md's
# bands. `s9w0_t37r` has no band: spec.md reports it beside the original run and keeps it out of the
# gate, because whether a poll is answered or times out is a race on the testbed.
GATE = (
    ("s9w0_b3p695", (3.695,), "s7b3p695", (242.25, 267.75), (15.771, 23.656)),
    ("s9w0_b10p043", (10.043,), "s7b10p043", (91.200, 100.800), (29.035, 43.553)),
    ("s9w0_t37r", (3.695, 7.39, 10.043), "s7t37r", None, None),
)
# The three intervals the rule (i_min 3.695, factor 2, i_max 10.043) can stand at: the floor, one
# back-off, and the ceiling (a second back-off would reach 14.78 and is clamped to i_max).
RULE_INTERVALS = (3.695, 7.39, 10.043)

# The scorings, and which perrun.csv columns each one sums. Order matters: the headline scoring is
# the one spec.md attaches a row's verdict word to, so it is first everywhere.
SCORINGS = (("headline", ("hlat4", "hn4", "hc4", "polls_sent")),
            ("all records", ("lat4", "n4", "c4", "polls_sent")))

WORDS = ("seen", "not seen", "reversed")
SPANS = ("entirely above zero", "spanning zero", "entirely below zero")

# ---- predict.txt, copied as printed there -------------------------------------------------------
# Per stage: headline delay s and its plus or minus 6 % band, polls per run and its band, the
# headline censored share, the all-record delay s and the all-record censored share.
PRED_STAGE = {
    "s9wo_b3p695": (21.549, 20.256, 22.842, 256.15, 240.78, 271.52, 0.000, 20.945, 0.047),
    "s9wo_b6p0": (29.537, 27.764, 31.309, 159.40, 149.84, 168.96, 0.005, 28.176, 0.107),
    "s9wo_b10p043": (41.218, 38.745, 43.691, 95.90, 90.15, 101.65, 0.023, 38.862, 0.241),
    "s9wo_t37": (38.452, 36.145, 40.759, 137.00, 128.78, 145.22, 0.031, 35.013, 0.165),
    "s9wo_t37r": (26.596, 25.000, 28.191, 179.88, 169.08, 190.67, 0.000, 25.627, 0.075),
    "s9wv_b3p695": (21.471, 20.183, 22.759, 256.15, 240.78, 271.52, 0.000, 20.747, 0.043),
    "s9wv_b6p0": (29.252, 27.497, 31.007, 159.40, 149.84, 168.96, 0.000, 28.023, 0.120),
    "s9wv_b10p043": (42.501, 39.951, 45.051, 95.90, 90.15, 101.65, 0.022, 39.771, 0.266),
    "s9wv_t37": (37.415, 35.170, 39.660, 141.07, 132.61, 149.54, 0.044, 33.503, 0.169),
    "s9wv_t37r": (26.452, 24.864, 28.039, 180.40, 169.58, 191.22, 0.000, 25.003, 0.073),
}
# (load, rule, scoring) -> (gain %, interval lo, interval hi, half width pp, predicted word,
# experiments seen / not seen / reversed).
PRED_GAIN = {
    ("wo", "T37", "headline"): (-17.73, -26.03, -10.01, 7.80, "reversed", (0, 0, 40)),
    ("wo", "T37r", "headline"): (+1.64, -8.03, +9.91, 8.81, "not seen", (2, 38, 0)),
    ("wo", "T37", "all records"): (-13.16, -20.42, -6.29, 6.83, "reversed", (0, 1, 39)),
    ("wo", "T37r", "all records"): (+1.17, -7.10, +8.73, 7.73, "not seen", (1, 39, 0)),
    ("wv", "T37", "headline"): (-16.51, -27.46, -7.06, 10.13, "reversed", (0, 1, 39)),
    ("wv", "T37r", "headline"): (+2.51, -7.90, +11.84, 9.71, "not seen", (0, 40, 0)),
    ("wv", "T37", "all records"): (-10.35, -18.65, -2.75, 7.41, "reversed", (0, 7, 33)),
    ("wv", "T37r", "all records"): (+2.60, -4.88, +10.89, 7.81, "not seen", (0, 40, 0)),
}
# (load, scoring) -> the T37r minus T37 difference on the frontier: pp, lo, hi, half width, tally.
PRED_DIFF = {
    ("wo", "headline"): (+19.95, +8.80, +29.69, 10.30, (40, 0, 0)),
    ("wo", "all records"): (+14.55, +4.64, +22.81, 8.91, (40, 0, 0)),
    ("wv", "headline"): (+19.03, +5.08, +32.03, 13.90, (37, 3, 0)),
    ("wv", "all records"): (+12.97, +2.05, +23.44, 10.90, (34, 6, 0)),
}
# (load, quantity, scoring) -> (percent change, lo, hi, half width pp) for pass line 4. The polls
# ratio is printed once per load in predict.txt, because polls are counted over the whole window
# under both scorings, so the same numbers stand under both here.
PRED_RATIO = {
    ("wo", "delay", "headline"): (-30.31, -37.11, -22.77, 6.84),
    ("wo", "delay", "all records"): (-26.50, -32.87, -19.42, 6.44),
    ("wo", "polls", "headline"): (+31.27, +27.75, +35.07, 3.65),
    ("wo", "polls", "all records"): (+31.27, +27.75, +35.07, 3.65),
    ("wv", "delay", "headline"): (-29.39, -38.54, -18.05, 10.09),
    ("wv", "delay", "all records"): (-24.89, -33.13, -15.29, 8.89),
    ("wv", "polls", "headline"): (+27.89, +23.37, +32.45, 4.49),
    ("wv", "polls", "all records"): (+27.89, +23.37, +32.45, 4.49),
}
# The record counts predict.py printed for these schedules: all records and headline records per
# run, mean [min .. max], and the thinnest headline seed.
PRED_COUNTS = {"wo": (190.6, 160, 224, 117.6, 80, 152, 19, 80),
               "wv": (214.6, 120, 316, 127.8, 48, 208, 15, 48)}

# spec.md's own instruction text for each reading of a pass line 3 row, keyed by (rule, word). The
# text of the word that was predicted is printed too, because spec.md attaches a sentence to the
# predicted reading as well and both T37r readings are explicitly not a claim about the rule.
SPEC_TEXT = {
    ("T37", "reversed"): "Reversed is the prediction, and it says that T37 loses to the fixed "
                         "interval on that load.",
    ("T37", "not seen"): "Not seen has the shape the default load's measurement had (evidence/16, "
                         "-6.4 %, interval -14.4 .. +1.8) and says nothing at all about the rule.",
    ("T37", "seen"): "Seen is the opposite sign from the prediction, so the reason is investigated "
                     "before any other line's conclusion is written.",
    ("T37r", "not seen"): "Not seen is the prediction, and even then nothing is claimed about the "
                          "rule. It is not a demonstration that the rule adds nothing. The "
                          "interval is wider than the effect, so the only claim is that neither "
                          "direction can be told.",
    ("T37r", "seen"): "Seen means a larger effect than the replay predicts and the reason is "
                      "investigated.",
    ("T37r", "reversed"): "Reversed is the opposite sign from the prediction and is likewise "
                          "investigated.",
}
# spec.md's instruction text for each of the three readings of pass line 5.
TEXT5 = {
    "predicted": "If this line reads predicted, the on/off numbers of evidence/14 and 15 have "
                 "support for the first time. That is the largest thing this step buys.",
    "mixed": "Mixed means: every value outside the band is reported one by one, naming the stage "
             "and the quantity, and the decision sequences of those stages are compared between "
             "the testbed log and the replay. Whether the replay can be used on this load is "
             "written only after that.",
    "missed": "Missed triggers evidence/16's procedure: the decision sequences around timeouts and "
              "around the ON/OFF switches are compared between the testbed log and the replay, and "
              "where the two part is reported before any other line's conclusion.",
}


def load_perrun(path):
    """{(stage, seed): row} for the p 0.4 rows of a perrun.csv, numeric columns as floats."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if float(r["p"]) != P:
                continue
            d = {k: (v if k in ("stage", "workload") else float(v)) for k, v in r.items()}
            out[(r["stage"], int(float(d["seed"])))] = d
    return out


def stage_name(load, arm):
    return f"s9{load}_{SUFFIX[arm]}"


def run_dir(stage, seed):
    return os.path.join(STORE, stage, f"p{P}", str(seed))


def validity(stage, seed):
    """(valid, flags) of one run from its meta.json `validity` block."""
    with open(os.path.join(run_dir(stage, seed), "meta.json")) as f:
        v = json.load(f)["validity"]
    return bool(v["valid"]), list(v["flags"])


def pooled(m):
    """(delay ms, polls per run, censored share) of a stage, pooled over records.

    `m` is the per-seed array of [lat, n, c, polls] in the columns of one scoring. An empty record
    set gives nan, so a stage that scored nothing is still reported.
    """
    lat, n, c, polls = m.sum(0)
    delay = lat / n if n else float("nan")
    cens = c / (n + c) if (n + c) else float("nan")
    return delay, polls / len(m), cens


def front_delay(arm_pts, x):
    """(delay of the three-arm frontier at `x`, reason it is undefined).

    The frontier is piecewise linear in log-log through the three arm points, so it is read on the
    one segment that contains `x`. Monotone means delay strictly falls as polls per run rise. A
    sample whose arms are not monotone has no frontier at all, and spec.md asks for such a sample to
    be counted and left out, so the reason comes back with the value.
    """
    pts = sorted(arm_pts)
    ys = [y for _x, y in pts]
    if not (ys[0] > ys[1] > ys[2]):
        return None, "not monotone"
    if not pts[0][0] <= x <= pts[-1][0]:
        return None, "polls outside arms"
    lo, hi = (pts[0], pts[1]) if x <= pts[1][0] else (pts[1], pts[2])
    (x0, y0), (x1, y1) = lo, hi
    w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y0) + w * (math.log(y1) - math.log(y0))), ""


def line_delay(arm_pts, x):
    """The log-log line through two arm points read at polls per run `x`, None when `x` is outside.

    Beyond the measured arms there is no line to read, so the answer there is None: a value from an
    extrapolated point would not be the comparison spec.md asks for.
    """
    (x0, y0), (x1, y1) = sorted(arm_pts)
    if not x0 <= x <= x1:
        return None
    w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y0) + w * (math.log(y1) - math.log(y0)))


def quantities(mats):
    """Every bootstrapped quantity of one sample of one load, as {name: value} and {name: reason}.

    `mats` maps each of the five arms to its per-seed array of [lat, n, c, polls] in one scoring's
    columns. All of the quantities come from the one sample, so the intervals of one load and one
    scoring share their resamples exactly. A None value means this sample leaves that quantity
    undefined, and `why` says which of spec.md's two reasons it was.
    """
    base = {a: pooled(m) for a, m in mats.items()}
    pt = {a: (base[a][1], base[a][0]) for a in base}
    three = [pt[a] for a in ARMS]
    two = [pt["B3"], pt["B10"]]
    val, why = {}, {}
    for r in RULES:
        d, pol, _c = base[r]
        fr, reason = front_delay(three, pol)
        val[f"gain_{r}"] = None if fr is None else 1 - d / fr
        why[f"gain_{r}"] = reason
    a, b = val["gain_T37"], val["gain_T37r"]
    val["diff"] = None if a is None or b is None else b - a
    why["diff"] = why["gain_T37"] or why["gain_T37r"]
    d6, pol6, _c6 = base["B6"]
    ref6 = line_delay(two, pol6)
    val["bend_B6"] = None if ref6 is None else 1 - d6 / ref6
    why["bend_B6"] = "" if ref6 is not None else "polls outside arms"
    val["ratio_delay"] = base["T37r"][0] / base["T37"][0] - 1
    val["ratio_polls"] = base["T37r"][1] / base["T37"][1] - 1
    why["ratio_delay"] = why["ratio_polls"] = ""
    return val, why


def fmt(x, nd=3):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def pct(x, nd=2):
    return "-" if x is None or x != x else f"{100 * x:+.{nd}f}"


def interval_word(lo, hi):
    """spec.md's three words: entirely above zero, spanning zero, entirely below zero."""
    if lo != lo or hi != hi:
        return "not computed"
    if lo > 0:
        return WORDS[0]
    if hi < 0:
        return WORDS[2]
    return WORDS[1]


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
    """The no-pass-line decision-row quantities of one stage, summed over `seeds`, and its histogram."""
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
    """The one index matrix of spec.md, and the bootstrapped quantities of one (load, scoring).

    spec.md fixes the interval as 2000 resamples of seeds 1 to 20 with `default_rng(20261117)`, one
    index draw reused for every stage and both loads, which is how predict.py drew its own matrix
    (`boot_index`). The unit is the seed number: one index row selects the seeds and all five stages
    of the load are recomputed from that selection, so the stages stay paired inside a resample. A
    seed set short of a seed is drawn at its own size from a generator with the same seed, and the
    sizes drawn are recorded in `log`.
    """

    def __init__(self, rows, seeds_by_load):
        self.rows = rows
        self.seeds_by_load = seeds_by_load
        self.idx = {}
        self.log = []
        self.cache = {}

    def index(self, n):
        if n not in self.idx:
            self.idx[n] = np.random.default_rng(BOOT_SEED).integers(0, n, size=(BOOT_REPS, n))
            self.log.append(n)
        return self.idx[n]

    def matrices(self, load, cols):
        seeds = self.seeds_by_load[load]
        return {a: np.array([[self.rows[(stage_name(load, a), s)][k] for k in cols] for s in seeds], float)
                for a in ARM_ORDER}

    def get(self, load, scoring, cols):
        """(full-sample values, reasons, {name: (lo, hi)}, {name: Counter of drop reasons})."""
        key = (load, scoring)
        if key in self.cache:
            return self.cache[key]
        mats = self.matrices(load, cols)
        full, full_why = quantities(mats)
        seeds = self.seeds_by_load[load]
        draws = {k: [] for k in full}
        dropped = {k: Counter() for k in full}
        for row in self.index(len(seeds)):
            sub = {a: m[row] for a, m in mats.items()}
            v, w = quantities(sub)
            for k, x in v.items():
                if x is None or x != x:
                    dropped[k][w[k] or "undefined"] += 1
                else:
                    draws[k].append(x)
        ci = {}
        for k, xs in draws.items():
            ci[k] = ((float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5)))
                     if xs else (float("nan"), float("nan")))
        self.cache[key] = (full, full_why, ci, dropped)
        return self.cache[key]


def main():
    rows = load_perrun(f"{HERE}/perrun.csv")
    ev16 = load_perrun(f"{EV16}/perrun.csv")

    out = []          # results.csv rows
    L = []            # results.md lines

    def rec(line, stage, metric, value, note=""):
        out.append(dict(line=line, stage=stage, metric=metric, value=value, note=note))

    # ---- DONE and unflagged seeds of every s9 stage ------------------------------------------
    # DONE means the run has a row in perrun.csv: perrun.py puts only runs with a DONE marker
    # there. Unflagged is meta.json validity.
    stage_seeds, stage_flagged = {}, {}
    all_stages = [g[0] for g in GATE] + [stage_name(ld, a) for ld in LOADS for a in ARM_ORDER]
    for stage in all_stages:
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

    seeds_by_load = {}
    for load in LOADS:
        sets = [set(stage_seeds[stage_name(load, a)]) for a in ARM_ORDER]
        seeds_by_load[load] = sorted(set.intersection(*sets))
    boot = Boot(rows, seeds_by_load)

    L += ["# evidence/19 results", "",
          "Computed by ana.py from perrun.csv here, ../16/perrun.csv and the stored runs' meta.json "
          "and event logs. The pass lines, the bands, the predictions and the verdict words are "
          "spec.md's; the words are read off the numbers mechanically and nothing here adds to "
          "them. Every line names the number of seeds it used, and a line computed on fewer seeds "
          "than it names gets the word `incomplete` in place of its verdict.", "",
          "Two scorings, as spec.md fixes them. The headline scoring takes the records whose "
          "scheduled creation time is at or before t0+72.0 s, selected by the experiment "
          "description's `schedule[host][seq-1][\"at_s\"] <= 72.0` read against discovery.csv's "
          "`(origin, seq)`, and its delay is the mean over the uncensored ones 4 hops away. The "
          "all-record scoring takes every record of the run, which is the quantity every earlier "
          "step reported. Polls per run is counted over the whole window [t0+10 s, t0+130 s] under "
          "both scorings. The headline scoring carries each row's verdict word and the all-record "
          "value stands beside it.", "",
          "| load | shape | stages |", "|---|---|---|"]
    for load in LOADS:
        L.append(f"| `{load}` | {LOAD_LABEL[load]} | "
                 + ", ".join(f"`{stage_name(load, a)}`" for a in ARM_ORDER) + " |")
    L += ["", "| arm | setting |", "|---|---|"]
    for a in ARM_ORDER:
        L.append(f"| {a} | {ARM_LABEL[a]} |")
    L += [""]

    # ---- line 1: the gate ---------------------------------------------------------------------
    L += ["## Line 1: the comparison across code versions holds (the gate)", "",
          "The three `s9w0_*` runs, seed 1, the default load. The first part of spec.md's line 1, "
          "that a generated run with `--workload w0` is byte-identical to one generated without it, "
          "is a test checked at merge and is not re-read here.", "",
          "Every `interval_s` in the decision log of a gate run must be a value the setting allows. "
          "The decision log is every `decision` event of kind `state_poll` or "
          "`state_poll_reschedule` on every host, which is the whole log of the rule's own "
          "arithmetic.", "",
          "| gate stage | allowed values | values seen (count) | exact |",
          "|---|---|---|---|"]
    exact_ok = []
    for stage, allowed, _ref, _pb, _db in GATE:
        _x, hist = rule_extras(stage, stage_seeds[stage])
        seen = sorted(hist)
        ok = set(seen) <= set(allowed)
        exact_ok.append(ok)
        L.append(f"| `{stage}` | " + ", ".join(str(v) for v in allowed) + " | "
                 + ", ".join(f"{v} ({hist[v]})" for v in seen) + f" | {'yes' if ok else 'no'} |")
        rec(1, stage, "interval_s_values_seen", ";".join(str(v) for v in seen))
        rec(1, stage, "interval_s_exact", "yes" if ok else "no",
            "allowed " + ",".join(str(v) for v in allowed))

    L += ["", "The two fixed arms against spec.md's bands, and `s9w0_t37r` beside the original "
              "`s7t37r` seed-1 run of ../16/perrun.csv with no band on it. On the default load "
              "every record is scheduled at or before t0+71 s, so the headline set is the whole "
              "set of records and the two scorings give the same delay here; the equality is "
              "checked in the last column.", "",
          "| gate stage | polls | polls band | delay s | delay band | censored share | ../16 seed-1 "
          "run: polls / delay s / censored share | headline equals all records |",
          "|---|---|---|---|---|---|---|---|"]
    band_ok = []
    for stage, _allowed, ref, pb, db in GATE:
        r = rows[(stage, 1)]
        m_all = np.array([[r[k] for k in ("lat4", "n4", "c4", "polls_sent")]], float)
        m_head = np.array([[r[k] for k in ("hlat4", "hn4", "hc4", "polls_sent")]], float)
        d, pol, c = pooled(m_all)
        dh, _polh, ch = pooled(m_head)
        same = (round(d, 6), round(c, 9)) == (round(dh, 6), round(ch, 9))
        q = ev16[(ref, 1)]
        rd, rpol, rc = pooled(np.array([[q[k] for k in ("lat4", "n4", "c4", "polls_sent")]], float))
        if pb is None:
            cells = ["reported only", "reported only"]
        else:
            ok = pb[0] <= pol <= pb[1] and db[0] <= d / 1000.0 <= db[1]
            band_ok.append(ok)
            cells = [f"{fmt(pb[0], 2)} .. {fmt(pb[1], 2)}", f"{fmt(db[0])} .. {fmt(db[1])}"]
        L.append(f"| `{stage}` | {fmt(pol, 2)} | {cells[0]} | {fmt(d / 1000.0)} | {cells[1]} | "
                 f"{fmt(c, 4)} | `{ref}` {fmt(rpol, 2)} / {fmt(rd / 1000.0)} / {fmt(rc, 4)} | "
                 f"{'yes' if same else 'no'} |")
        rec(1, stage, "polls_per_run", f"{pol:.6f}")
        rec(1, stage, "delay_s", f"{d / 1000.0:.6f}")
        rec(1, stage, "censored_share", f"{c:.6f}")
        rec(1, stage, "headline_delay_s", f"{dh / 1000.0:.6f}")
        rec(1, stage, "headline_equals_all_records", "yes" if same else "no")
        rec(1, stage, "ev16_reference", ref, f"polls {rpol:.2f}, delay {rd / 1000.0:.3f} s, "
                                             f"censored {rc:.4f}")
        if pb is not None:
            rec(1, stage, "inside_bands", "yes" if ok else "no",
                f"polls {pb[0]} .. {pb[1]}, delay {db[0]} .. {db[1]} s")
    n_gate = sum(len(stage_seeds[g[0]]) for g in GATE)
    if n_gate < 3:
        verdict1 = f"incomplete ({n_gate} of 3 gate runs DONE and unflagged)"
    elif all(exact_ok) and all(band_ok):
        verdict1 = "pass"
    else:
        verdict1 = "fail"
    L += ["", f"Gate runs DONE and unflagged: {n_gate} of 3.",
          f"Verdict on {n_gate} runs: **{verdict1}**.", ""]
    rec(1, "", "gate_runs_valid", n_gate)
    rec(1, "", "verdict", verdict1)

    # ---- line 2: all 200 sweep runs DONE and valid ---------------------------------------------
    L += ["## Line 2: all 200 sweep runs DONE and valid", "",
          "Validity is `<run_dir>/meta.json` -> `validity` (`valid`, `flags`), written by "
          "semindex.evalrun from semindex.evalvalid. Of the flags, `record_off_schedule` and "
          "`record_count` say directly whether the on/off schedule was kept by the testbed, so the "
          "count of each is printed per stage even when it is zero. The flag names carry the node "
          "and the sequence number (`record_off_schedule:h0:seq5`), so what is counted is the "
          "flags whose name starts with those two words.", "",
          "| stage | load | runs expected | runs DONE | valid and unflagged | flagged | "
          "record_off_schedule | record_count | other flags seen |",
          "|---|---|---|---|---|---|---|---|---|"]
    sweep_done = sweep_ok = 0
    for load in LOADS:
        for a in ARM_ORDER:
            stage = stage_name(load, a)
            got = sorted(s for (st, s) in rows if st == stage)
            ok, fl = stage_seeds[stage], stage_flagged[stage]
            sweep_done += len(got)
            sweep_ok += len(ok)
            flat = [f for flags in fl.values() for f in flags]
            n_off = sum(1 for f in flat if f.startswith("record_off_schedule"))
            n_cnt = sum(1 for f in flat if f.startswith("record_count"))
            other = sorted({f for f in flat if not f.startswith(("record_off_schedule", "record_count"))})
            L.append(f"| `{stage}` | `{load}` | {SEEDS_WANTED} | {len(got)} | {len(ok)} | {len(fl)} "
                     f"| {n_off} | {n_cnt} | {'; '.join(other) or 'none'} |")
            rec(2, stage, "runs_done", len(got))
            rec(2, stage, "runs_valid", len(ok))
            rec(2, stage, "runs_flagged", len(fl))
            rec(2, stage, "flags_record_off_schedule", n_off)
            rec(2, stage, "flags_record_count", n_cnt)
    want2 = SEEDS_WANTED * len(LOADS) * len(ARM_ORDER)
    if sweep_done < want2:
        verdict2 = f"incomplete ({sweep_done} of {want2} DONE)"
    elif sweep_ok < sweep_done:
        verdict2 = "fail"
    else:
        verdict2 = "pass"
    L += ["", f"{sweep_done} of {want2} sweep runs DONE, {sweep_ok} of them valid and unflagged.",
          f"Verdict on {sweep_done} runs: **{verdict2}**.", "",
          "Seeds each load's lines are read on, the intersection over that load's five stages:", "",
          "| load | seeds used | seeds |", "|---|---|---|"]
    for load in LOADS:
        sd = seeds_by_load[load]
        L.append(f"| `{load}` | {len(sd)} of {SEEDS_WANTED} | "
                 + (f"{min(sd)} to {max(sd)}" if len(sd) == max(sd) - min(sd) + 1
                    else ",".join(str(s) for s in sd)) + " |")
        rec(2, load, "seeds_used", len(sd))
    L += [""]
    rec(2, "", "runs_done_total", sweep_done)
    rec(2, "", "runs_valid_total", sweep_ok)
    rec(2, "", "verdict", verdict2)

    # ---- line 3: gain against the fixed-interval line ------------------------------------------
    L += ["## Line 3: gain against the fixed-interval line", "",
          "The line is the three arms of the same load and p (3.695, 6.0, 10.043 s), polls per run "
          "on the horizontal axis and delay on the vertical, joined piecewise in log-log. Gain is "
          "`1 - delay(rule) / the line's delay at the rule's own polls per run`. The arms must be "
          "monotone, which here means delay falls strictly as polls per run rise; a resample whose "
          "arms are not monotone, and a resample whose rule polls fall outside the outer arms, is "
          "counted and left out.", "",
          "The reading is attached to each of the four (load, rule) rows separately and the four "
          "rows are not required to agree.", ""]
    for load in LOADS:
        sd = seeds_by_load[load]
        L += [f"### Load `{load}`, {len(sd)} of {SEEDS_WANTED} seeds", "",
              "| arm | stage | scoring | delay s | censored share | polls per run |",
              "|---|---|---|---|---|---|"]
        for a in ARM_ORDER:
            for scoring, cols in SCORINGS:
                m = boot.matrices(load, cols)[a]
                d, pol, c = pooled(m)
                L.append(f"| {a} | `{stage_name(load, a)}` | {scoring} | {fmt(d / 1000.0)} | "
                         f"{fmt(c, 5)} | {fmt(pol, 2)} |")
                tag = f"{load}:{scoring}"
                rec(3, stage_name(load, a), f"delay_s[{tag}]", f"{d / 1000.0:.6f}")
                rec(3, stage_name(load, a), f"censored_share[{tag}]", f"{c:.6f}")
                rec(3, stage_name(load, a), f"polls_per_run[{tag}]", f"{pol:.6f}")
        for scoring, cols in SCORINGS:
            full, _why, ci, drop = boot.get(load, scoring, cols)
            mats = boot.matrices(load, cols)
            pts = sorted((pooled(mats[a])[1], pooled(mats[a])[0]) for a in ARMS)
            mono = pts[0][1] > pts[1][1] > pts[2][1]
            L += ["", f"Arm points of the full sample, {scoring} scoring, sorted by polls per run: "
                      + ", ".join(f"({fmt(x, 2)} polls, {fmt(y / 1000.0)} s)" for x, y in pts)
                      + f". Monotone: {'yes' if mono else 'no'}.", ""]
            rec(3, load, f"arms_monotone[{scoring}]", "yes" if mono else "no")
            L += ["| rule | scoring | gain % | 95 % interval % | half width pp | word | predicted "
                  "gain % and interval | predicted word | resamples left out |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for r in RULES:
                v = full[f"gain_{r}"]
                lo, hi = ci[f"gain_{r}"]
                word = (f"incomplete ({len(sd)} of {SEEDS_WANTED} seeds)" if len(sd) < SEEDS_WANTED
                        else interval_word(lo, hi))
                pg = PRED_GAIN[(load, r, scoring)]
                hw = 100 * (hi - lo) / 2 if hi == hi else float("nan")
                dn = drop[f"gain_{r}"]
                L.append(f"| {r} | {scoring} | {pct(v)} | [{pct(lo)}, {pct(hi)}] | {fmt(hw, 2)} | "
                         f"**{word}** | {pg[0]:+.2f} [{pg[1]:+.2f}, {pg[2]:+.2f}], half width "
                         f"{pg[3]:.2f} pp | {pg[4]} | {sum(dn.values())} |")
                tag = f"{load}:{r}:{scoring}"
                rec(3, load, f"gain[{tag}]", "" if v is None else f"{v:.6f}")
                rec(3, load, f"gain_lo[{tag}]", f"{lo:.6f}")
                rec(3, load, f"gain_hi[{tag}]", f"{hi:.6f}")
                rec(3, load, f"gain_half_width_pp[{tag}]", f"{hw:.4f}")
                rec(3, load, f"word[{tag}]", word)
                rec(3, load, f"predicted_word[{tag}]", pg[4])
                rec(3, load, f"boot_dropped[{tag}]", sum(dn.values()),
                    "; ".join(f"{k} {c}" for k, c in sorted(dn.items())) or "none")
            vd = full["diff"]
            lod, hid = ci["diff"]
            pd = PRED_DIFF[(load, scoring)]
            wd = (f"incomplete ({len(sd)} of {SEEDS_WANTED} seeds)" if len(sd) < SEEDS_WANTED
                  else interval_word(lod, hid))
            L += ["", f"T37r minus T37 on the same frontier, {scoring} scoring: {pct(vd)} pp "
                      f"[{pct(lod)}, {pct(hid)}], half width {fmt(100 * (hid - lod) / 2, 2)} pp, "
                      f"word **{wd}**. The replay predicted {pd[0]:+.2f} pp [{pd[1]:+.2f}, "
                      f"{pd[2]:+.2f}], half width {pd[3]:.2f} pp, with {pd[4][0]} of 40 experiments "
                      f"entirely above zero. spec.md attaches no pass line to this difference.", ""]
            rec(3, load, f"diff[{scoring}]", "" if vd is None else f"{vd:.6f}")
            rec(3, load, f"diff_lo[{scoring}]", f"{lod:.6f}")
            rec(3, load, f"diff_hi[{scoring}]", f"{hid:.6f}")
            rec(3, load, f"diff_word[{scoring}]", wd)
    L += ["What spec.md says about each reading. The word of the headline scoring is the row's "
          "reading; the all-record word is printed beside it.", "",
          "| load | rule | headline word | all-record word | predicted word | spec.md's text for "
          "the headline word |", "|---|---|---|---|---|---|"]
    unexpected = []
    for load in LOADS:
        for r in RULES:
            got = {}
            for scoring, cols in SCORINGS:
                _f, _w, ci, _d = boot.get(load, scoring, cols)
                lo, hi = ci[f"gain_{r}"]
                got[scoring] = (interval_word(lo, hi) if len(seeds_by_load[load]) >= SEEDS_WANTED
                                else f"incomplete ({len(seeds_by_load[load])} of {SEEDS_WANTED} seeds)")
            pw = PRED_GAIN[(load, r, "headline")][4]
            head = got["headline"]
            text = SPEC_TEXT.get((r, head), "No text in spec.md for this reading.")
            if head != pw:
                unexpected.append((load, r, head, pw))
            L.append(f"| `{load}` | {r} | **{head}** | {got['all records']} | {pw} | {text} |")
            rec(3, load, f"spec_text[{r}]", text)
    L += [""]
    if unexpected:
        L += ["Rows whose word is not the predicted one, which is what spec.md sends to be "
              "investigated:", ""]
        for load, r, head, pw in unexpected:
            L.append(f"- `{load}` {r}: read **{head}**, predicted {pw}. {SPEC_TEXT.get((r, head), '')}")
            rec(3, load, f"unexpected_word[{r}]", head, f"predicted {pw}")
        L += [""]
    else:
        L += ["Every one of the four rows read the word spec.md predicted, so none of them sends "
              "the step to investigate a reason.", ""]
    L += ["For `wv` T37 spec.md asks what each scoring read, because the replay's two scorings "
          "already differ in how wide the interval is there. Both words are in the table above and "
          "the two intervals are in the per-load tables.", ""]

    # ---- line 4: T37r against T37 directly ------------------------------------------------------
    L += ["## Line 4: T37r against T37 directly, per load", "",
          "The delay ratio and the polls ratio, from the same resamples, printed as the percent "
          "change of T37r against T37 so that they sit beside the replay's numbers. The censored "
          "share of each rule stands beside the delay. spec.md attaches no pass or fail to this "
          "line: the direction is known from the replay and this line is the measurement of the "
          "size. The default load measured -23.0 % in delay and +26.8 % in polls, neither interval "
          "spanning zero (../16/after.md).", "",
          "| load | seeds | quantity | scoring | measured % | 95 % interval % | measured half width "
          "pp | predicted % and interval | predicted half width pp | spans zero |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for load in LOADS:
        sd = seeds_by_load[load]
        for key, label in (("ratio_delay", "delay"), ("ratio_polls", "polls")):
            for scoring, cols in SCORINGS:
                full, _why, ci, _drop = boot.get(load, scoring, cols)
                v = full[key]
                lo, hi = ci[key]
                hw = 100 * (hi - lo) / 2
                pr = PRED_RATIO[(load, label, scoring)]
                spans = "yes" if lo <= 0 <= hi else "no"
                if len(sd) < SEEDS_WANTED:
                    spans = f"incomplete ({len(sd)} of {SEEDS_WANTED} seeds)"
                L.append(f"| `{load}` | {len(sd)} | {label} | {scoring} | {pct(v)} | [{pct(lo)}, "
                         f"{pct(hi)}] | {fmt(hw, 2)} | {pr[0]:+.2f} [{pr[1]:+.2f}, {pr[2]:+.2f}] | "
                         f"{pr[3]:.2f} | {spans} |")
                tag = f"{load}:{label}:{scoring}"
                rec(4, load, f"ratio_pct[{tag}]", f"{100 * v:.6f}")
                rec(4, load, f"ratio_lo_pct[{tag}]", f"{100 * lo:.6f}")
                rec(4, load, f"ratio_hi_pct[{tag}]", f"{100 * hi:.6f}")
                rec(4, load, f"ratio_half_width_pp[{tag}]", f"{hw:.4f}")
                rec(4, load, f"spans_zero[{tag}]", spans)
    L += ["", "The polls ratio is the same number under both scorings, because polls are counted "
              "over the whole window whichever records the delay is taken on. predict.txt prints it "
              "once per load for that reason and the same predicted numbers stand under both rows "
              "here.", "",
          "Censored shares of the two rule stages, beside the delays above:", "",
          "| load | stage | scoring | delay s | censored share |", "|---|---|---|---|---|"]
    for load in LOADS:
        for r in RULES:
            for scoring, cols in SCORINGS:
                d, _pol, c = pooled(boot.matrices(load, cols)[r])
                L.append(f"| `{load}` | `{stage_name(load, r)}` | {scoring} | {fmt(d / 1000.0)} | "
                         f"{fmt(c, 5)} |")
    L += [""]

    # ---- line 5: did the replay predict the measurement? ----------------------------------------
    L += ["## Line 5: did the replay predict the measurement?", "",
          "The headline delay and the polls per run over the whole window of all ten sweep stages, "
          "20 values, each against the plus or minus 6 % band of spec.md, copied from it as "
          "printed. 18 or more of the 20 inside reads `predicted`, 15 to 17 reads `mixed`, 14 or "
          "fewer reads `missed`. The 6 % band comes from the largest deviation the replay showed on "
          "the default load, 5.9 % (evidence/13); that band was measured on the default load and "
          "nothing says it is the right band for an on/off load, so every value outside it is "
          "reported with how far outside and in which direction whatever the count is.", "",
          "| stage | quantity | measured | prediction | band | inside or outside | deviation % |",
          "|---|---|---|---|---|---|---|"]
    inside = 0
    outside_list = []
    n5 = min(len(seeds_by_load[ld]) for ld in LOADS)
    for load in LOADS:
        cols = SCORINGS[0][1]
        for a in ARM_ORDER:
            stage = stage_name(load, a)
            d, pol, _c = pooled(boot.matrices(load, cols)[a])
            p = PRED_STAGE[stage]
            for metric, value, mid, lo_b, hi_b, nd in (
                ("headline delay s", d / 1000.0, p[0], p[1], p[2], 3),
                ("polls per run", pol, p[3], p[4], p[5], 2),
            ):
                word = "inside" if lo_b <= value <= hi_b else "outside"
                inside += word == "inside"
                dev = 100 * (value / mid - 1)
                if word == "outside":
                    outside_list.append((stage, metric, value, mid, dev))
                L.append(f"| `{stage}` | {metric} | {fmt(value, nd)} | {fmt(mid, nd)} | "
                         f"{fmt(lo_b, nd)} .. {fmt(hi_b, nd)} | {word} | {dev:+.2f} |")
                rec(5, stage, f"{metric}_measured", f"{value:.6f}")
                rec(5, stage, f"{metric}_band", word, f"{lo_b} .. {hi_b}")
                rec(5, stage, f"{metric}_deviation_pct", f"{dev:.4f}")
    if n5 < SEEDS_WANTED:
        verdict5 = f"incomplete ({n5} of {SEEDS_WANTED} seeds in at least one load)"
    elif inside >= 18:
        verdict5 = "predicted"
    elif inside >= 15:
        verdict5 = "mixed"
    else:
        verdict5 = "missed"
    L += ["", f"{inside} of 20 values inside their band. Verdict on {n5} seeds per load: "
              f"**{verdict5}**.", "", TEXT5.get(verdict5, ""), ""]
    rec(5, "", "values_inside", inside)
    rec(5, "", "seeds_used", n5)
    rec(5, "", "verdict", verdict5)
    if outside_list:
        L += ["Every value outside its band, one by one, with how far outside and in which "
              "direction:", ""]
        for stage, metric, value, mid, dev in outside_list:
            far = abs(dev) - 6.0
            L.append(f"- `{stage}` {metric}: measured {value:.3f}, predicted {mid:.3f}, "
                     f"{dev:+.2f} % against the prediction, which is {far:.2f} points "
                     f"{'above' if dev > 0 else 'below'} the 6 % band.")
            rec(5, stage, f"outside[{metric}]", f"{dev:.4f}", f"predicted {mid}")
        L += [""]
    else:
        L += ["No value fell outside its band.", ""]
    regen_bad = sorted(k for k, r in rows.items() if r["sched_regen_diff"])
    L += ["This line does not show that the creation times are what WO and WV are defined to be. "
          "The prediction reads the creation times out of the testbed's own experiment "
          "descriptions, so a mistake in the tick-laying code would be in the prediction too. That "
          "those instants have the defined shape is shown by `tests/test_evalgen.py`.", "",
          "That the testbed and the replay ran the same instants is what this line's comparison "
          "states, and it is checked run by run. perrun.py reads the instants out of each run's "
          "own experiment description and draws them again from the committed generator for that "
          "run's own (host, seed, k, workload), which is the builder every description's schedule "
          "block is written from and the one the replay's instants come from. Runs whose two "
          f"origins both agree, instant by instant and in order: {len(rows) - len(regen_bad)} of "
          f"{len(rows)}." + ("" if not regen_bad else " The runs that disagree: "
          + ", ".join(f"`{st}` seed {s}" for st, s in regen_bad) + "."), ""]
    rec(5, "", "schedule_runs_checked", len(rows))
    rec(5, "", "schedule_runs_matching_generator", len(rows) - len(regen_bad))
    L += ["The comparison of decision sequences that spec.md asks for when this line reads `mixed` "
          "or `missed` is not computed here. It is a reading of two logs against each other, which "
          "this script does not do. The word this line read is printed above, so the reader can "
          "see whether spec.md asks for that comparison at all.", ""]

    # ---- also reported, without a pass line ------------------------------------------------------
    L += ["## Also reported, without a pass line", "",
          "Every stage's measured values beside the replay's, both scorings together. The 6.0 s arm "
          "appears here and in line 3's tables: spec.md runs it to measure the bend of the "
          "trade-off on these loads and attaches no pass line to it.", "",
          "| stage | seeds | headline delay s | headline censored | all-record delay s | "
          "all-record censored | polls per run | replay headline delay s | replay all-record delay "
          "s | replay headline censored | replay all-record censored |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for load in LOADS:
        for a in ARM_ORDER:
            stage = stage_name(load, a)
            dh, pol, ch = pooled(boot.matrices(load, SCORINGS[0][1])[a])
            da, _p2, ca = pooled(boot.matrices(load, SCORINGS[1][1])[a])
            p = PRED_STAGE[stage]
            L.append(f"| `{stage}` | {len(seeds_by_load[load])} | {fmt(dh / 1000.0)} | "
                     f"{fmt(ch, 5)} | {fmt(da / 1000.0)} | {fmt(ca, 5)} | {fmt(pol, 2)} | "
                     f"{fmt(p[0])} | {fmt(p[7])} | {fmt(p[6])} | {fmt(p[8])} |")
            rec(0, stage, "headline_delay_s", f"{dh / 1000.0:.6f}")
            rec(0, stage, "headline_censored_share", f"{ch:.6f}")
            rec(0, stage, "all_delay_s", f"{da / 1000.0:.6f}")
            rec(0, stage, "all_censored_share", f"{ca:.6f}")
            rec(0, stage, "polls_per_run", f"{pol:.6f}")

    L += ["", "The 6.0 s arm against the straight line between the 3.695 s and 10.043 s arms of the "
              "same load, computed as 1 minus the measured delay over the delay the straight line "
              "reads at the arm's own polls. A positive value means the measured 6.0 s arm is "
              "faster than the straight line, so the trade-off bends below it, and that is the "
              "bend spec.md runs the third arm to measure. A negative value means the arm is "
              "slower than the line. No pass line is attached.", "",
          "| load | scoring | bend % | 95 % interval % | resamples left out |",
          "|---|---|---|---|---|"]
    for load in LOADS:
        for scoring, cols in SCORINGS:
            full, _w, ci, drop = boot.get(load, scoring, cols)
            lo, hi = ci["bend_B6"]
            L.append(f"| `{load}` | {scoring} | {pct(full['bend_B6'])} | [{pct(lo)}, {pct(hi)}] | "
                     f"{sum(drop['bend_B6'].values())} |")
            rec(0, load, f"bend_B6[{scoring}]", f"{full['bend_B6']:.6f}")
            rec(0, load, f"bend_B6_lo[{scoring}]", f"{lo:.6f}")
            rec(0, load, f"bend_B6_hi[{scoring}]", f"{hi:.6f}")

    L += ["", "Record counts per seed, from the schedule each run was handed. predict.py printed "
              "the mean and the range of these counts over the seeds and no value per seed, so "
              "what stands per seed beside the counts is the `instants match` column: it reads "
              "`yes` when every stage of that cell had both origins' instants equal to the "
              "generator's own draw, which is the check line 5 reports the total of. All five "
              "stages of one (load, seed) are handed the same schedule, so one row per (load, "
              "seed) is the whole of it; the `stages agree` column says that every stage of that "
              "cell carried the same two counts.", "",
          "| load | seed | all records | headline records | stages agree | instants match |",
          "|---|---|---|---|---|---|"]
    for load in LOADS:
        for s in seeds_by_load[load]:
            counts = {(rows[(stage_name(load, a), s)]["sched_all"],
                       rows[(stage_name(load, a), s)]["sched_head"]) for a in ARM_ORDER}
            one = sorted(counts)[0]
            bad = sum(rows[(stage_name(load, a), s)]["sched_regen_diff"] for a in ARM_ORDER)
            L.append(f"| `{load}` | {s} | {one[0]:.0f} | {one[1]:.0f} | "
                     f"{'yes' if len(counts) == 1 else 'no'} | {'yes' if not bad else 'no'} |")
            rec(0, load, f"sched_all[seed {s}]", f"{one[0]:.0f}")
            rec(0, load, f"sched_head[seed {s}]", f"{one[1]:.0f}")
    L += ["", "| load | all records: mean [min .. max] | headline: mean [min .. max] | replay all "
              "records | replay headline | thinnest headline seed |", "|---|---|---|---|---|---|"]
    for load in LOADS:
        sd = seeds_by_load[load]
        allc = np.array([rows[(stage_name(load, "B3"), s)]["sched_all"] for s in sd], float)
        head = np.array([rows[(stage_name(load, "B3"), s)]["sched_head"] for s in sd], float)
        pc = PRED_COUNTS[load]
        thin = sd[int(head.argmin())]
        L.append(f"| `{load}` | {allc.mean():.1f} [{allc.min():.0f} .. {allc.max():.0f}] | "
                 f"{head.mean():.1f} [{head.min():.0f} .. {head.max():.0f}] | {pc[0]:.1f} "
                 f"[{pc[1]} .. {pc[2]}] | {pc[3]:.1f} [{pc[4]} .. {pc[5]}] | seed {thin} with "
                 f"{head.min():.0f} records, the replay's seed {pc[6]} with {pc[7]} |")
        rec(0, load, "sched_all_mean", f"{allc.mean():.4f}")
        rec(0, load, "sched_head_mean", f"{head.mean():.4f}")
        rec(0, load, "sched_head_min", f"{head.min():.0f}", f"seed {thin}")

    L += ["", "Whole-run counts over every decision row of the five rule stages, not cut to the "
              "measurement window, because what they describe is the rule's own arithmetic over "
              "every poll the run sent. An answered re-decision carries `empty`: true when the "
              "reply held nothing new, false when it held something new, null when the reply could "
              "not be decoded. A timed-out re-decision carries `empty` null by construction and is "
              "not counted among the answered ones.", "",
          "| stage | runs | polls | re-decisions | answered | timed out | empty true share of "
          "answered | empty false | empty null and answered | clamped | late replies |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    rule_stages = [g[0] for g in GATE if "t37" in g[0]] + [stage_name(ld, r) for ld in LOADS for r in RULES]
    hists = {}
    for stage in rule_stages:
        sd = stage_seeds[stage]
        x, hist = rule_extras(stage, sd)
        hists[stage] = hist
        share = x["empty_true"] / x["answered"] if x["answered"] else float("nan")
        L.append(f"| `{stage}` | {len(sd)} | {x['polls']} | {x['resched']} | {x['answered']} | "
                 f"{x['timed_out']} | {fmt(share, 4)} | {x['empty_false']} | "
                 f"{x['empty_null_answered']} | {x['clamped']} | {x['late']} |")
        for k, v in sorted(x.items()):
            rec(0, stage, f"extra_{k}", v)
        rec(0, stage, "extra_empty_true_share_of_answered", f"{share:.6f}")
    L += ["", "Every `interval_s` the rule stages logged, against the three values the rule can "
              "stand at (i_min 3.695, one back-off 7.39, the ceiling 10.043). Any other value would "
              "fill the `other` column.", "",
          "| stage | " + " | ".join(str(v) for v in RULE_INTERVALS) + " | other | all rows |",
          "|---|" + "---|" * (len(RULE_INTERVALS) + 2)]
    for stage, hist in hists.items():
        other = {k: v for k, v in hist.items() if k not in RULE_INTERVALS}
        L.append(f"| `{stage}` | " + " | ".join(str(hist[v]) for v in RULE_INTERVALS) + " | "
                 + (json.dumps(other, sort_keys=True) if other else "0") + f" | {sum(hist.values())} |")
        for v in RULE_INTERVALS:
            rec(0, stage, f"interval_s_{v}", hist[v])
        rec(0, stage, "interval_s_other", sum(other.values()), json.dumps(other, sort_keys=True))
    L += ["", "The three fixed-arm stages of each load log one `interval_s` each, the arm's own "
              "nominal value; the jitter of 0.1 is applied after the choice is logged, so the "
              "logged value is the nominal one. Those stages are left out of the table above, "
              "which covers the stages whose interval is the rule's own output.", "",
          "| stage | " + " | ".join(str(v) for v in (3.695, 6.0, 10.043)) + " | other | all rows |",
          "|---|---|---|---|---|---|"]
    for load in LOADS:
        for a in ARMS:
            stage = stage_name(load, a)
            _x, hist = rule_extras(stage, stage_seeds[stage])
            other = {k: v for k, v in hist.items() if k not in (3.695, 6.0, 10.043)}
            L.append(f"| `{stage}` | " + " | ".join(str(hist[v]) for v in (3.695, 6.0, 10.043))
                     + " | " + (json.dumps(other, sort_keys=True) if other else "0")
                     + f" | {sum(hist.values())} |")
            for v in (3.695, 6.0, 10.043):
                rec(0, stage, f"interval_s_{v}", hist[v])
            rec(0, stage, "interval_s_other", sum(other.values()), json.dumps(other, sort_keys=True))

    # ---- how the bootstrap was drawn -------------------------------------------------------------
    L += ["", "## How the bootstrap was drawn", "",
          f"One index matrix, `numpy.random.default_rng({BOOT_SEED}).integers(0, n, size="
          f"({BOOT_REPS}, n))`, drawn once at the seed count `n` the loads have and reused for both "
          "loads and both scorings. That is how predict.py drew its own matrix, so the measured "
          "bounds and the predicted bounds rest on the same resampled positions. The unit is the "
          "seed number: one index row selects the seeds, and all five stages of the load are "
          "recomputed from that one selection, so a resample holds the same seeds in every stage at "
          "once and the stages stay paired. A load short of a seed would be drawn at its own size "
          "from a generator with the same seed, and every size drawn is listed here.", "",
          "| index matrix | seeds per resample | resamples |", "|---|---|---|"]
    for i, n in enumerate(boot.log, 1):
        L.append(f"| {i} | {n} | {BOOT_REPS} |")
        rec(0, "bootstrap", f"draw_{i}", n, f"{BOOT_REPS} resamples")
    L += ["", "Resamples left out of every interval. A gain is left out of a resample when that "
              "resample's rule polls per run fall outside the arms, and also when that resample's "
              "three arms are not monotone. The ratios of line 4 are defined in every resample, so "
              "none is left out of them. spec.md counts such a resample and leaves it out; no "
              "value is substituted for it.", "",
          "| load | scoring | quantity | resamples left out | reasons |", "|---|---|---|---|---|"]
    for load in LOADS:
        for scoring, cols in SCORINGS:
            _f, _w, _ci, drop = boot.get(load, scoring, cols)
            for k in sorted(drop):
                why = "; ".join(f"{r} {c}" for r, c in sorted(drop[k].items())) or "none"
                L.append(f"| `{load}` | {scoring} | {k} | {sum(drop[k].values())} | {why} |")
                rec(0, load, f"dropped[{scoring}:{k}]", sum(drop[k].values()), why)

    L += ["", "## Where spec.md left a choice", "",
          "Each of these is the plainest reading of spec.md's own text; none of them moves a pass "
          "line, a band or a verdict word.", "",
          "- The word of a (load, rule) row of line 3 is the headline scoring's word. spec.md's "
          "Quantities makes the headline the scoring the pass lines read and says the all-record "
          "value is reported beside it everywhere, so the all-record word is printed beside the "
          "headline word in every table.", "",
          "- A load's seed set is the intersection over that load's five stages, because spec.md's "
          "bootstrap recomputes all five stages of that load from the same resampled seeds. The "
          "gate is counted separately, on its own three runs.", "",
          "- A resample is left out of each quantity it leaves undefined, one quantity at a time. "
          "predict.py drops a resample from both rules at once when either rule has no readable "
          "frontier. Both counts are zero or near zero in the replay, so the two readings cannot "
          "part by much; the drop counts are printed above so the reader can see which case "
          "applied.", "",
          "- The decision log of line 1 is every `decision` event of kind `state_poll` or "
          "`state_poll_reschedule`, which is every row in which the rule or the arm names an "
          "`interval_s`.", "",
          "- Line 4's ratios are printed as percent changes of T37r against T37, which is the form "
          "predict.txt printed them in, and the raw ratio minus one is what results.csv carries.", "",
          "- The headline sums are built from the discovery rows of the records the schedule puts "
          "at or before t0+72.0 s, while the all-record sums keep attrib's own censoring chain, "
          "which is where every earlier step's numbers come from. perrun.py computes the "
          "all-record sums the second way as well and reports how many runs the two ways disagree "
          "on; the count was 0 over all 203 runs, so the two scorings rest on one censoring rule.", "",
          "- Line 5 reads `incomplete` on the smaller of the two loads' seed counts, since it is "
          "one word over the ten stages of both loads together.", "",
          "- On the default load every record is scheduled at or before t0+71 s, so the gate's "
          "headline set is the whole set of records. Line 1's table checks that equality and "
          "prints the result, so it is a measurement here and not an assumption.", "",
          "- spec.md's note under line 5 calls the check of the instants a file-against-file "
          "comparison. What is computed is each run's own experiment description against a fresh "
          "draw of the generator's schedule builder, which is the function every such file's "
          "schedule block is written from, so the second file is not written again to be read "
          "back. The plainest reading of the note is that the instants the testbed ran and the "
          "instants the replay scored are the same instants, and that is what this comparison "
          "answers.", ""]

    with open(f"{HERE}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["line", "stage", "metric", "value", "note"])
        w.writeheader()
        w.writerows(out)
    with open(f"{HERE}/results.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
