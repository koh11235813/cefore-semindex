"""The five pass lines of spec.md and its amendments, from the three perrun.csv files and the stored runs.

Writes results.csv (one row per reported quantity) and results.md (the same numbers as tables), and
prints results.md on stdout. Every verdict word in both is the mechanical one spec.md defines; nothing
here reads more into a number than that.

The five logical arms and where each seed's run comes from:

    B3    fixed 3.695 s   seeds 1-20 `s6b3p695`  (../10/perrun.csv)   seeds 21-40 `s8b3p695`
    B10   fixed 10.043 s  seeds 1-20 `s6b10p043` (../10/perrun.csv)   seeds 21-40 `s8b10p043`
    B6    fixed 6.0 s     seeds 1-40 `s8b6p0`
    T37   rule, keep      seeds 1-20 `s7t37`     (../16/perrun.csv)   seeds 21-40 `s8t37`
    T37r  rule, reset     seeds 1-20 `s7t37r`    (../16/perrun.csv)   seeds 21-40 `s8t37r`

What each pass line needs:

line 1  Every s8 run's DONE status and validity. Validity is `<run_dir>/meta.json` -> `validity`
        (`valid`, `flags`), which `semindex.evalrun` writes from `semindex.evalvalid`. That is the
        field `evidence/05/attrib.py` gates on, so a flagged run would already be dropped from the
        pair sums perrun.py reports. The seed set of every other line is the set of seeds that are
        DONE and unflagged in all of the stages that line names (amendment 1), and the validity of
        the seed-1-to-20 runs of `s6*` and `s7*` is read from the store here as well.

line 2  Seeds 21 to 40, the four s8 stages. Delay is pooled over records (`sum(lat4) / sum(n4)`, ms),
        censored share is `sum(c4) / sum(n4 + c4)`, polls per run is the mean of `polls_sent`. The
        two fixed arms are joined by a straight line in log-log; gain is
        `1 - delay(rule) / line(polls(rule))`, undefined when the rule's polls per run fall outside
        the two arms. This reading carries the verdict (amendment 2).

line 3  The same quantity on seeds 1 to 40, arms `s6` plus `s8` and rules `s7` plus `s8`. It contains
        the 20 seeds evidence/16 already read and it mixes two code versions, so it is the estimate
        of the size and never an independent test.

line 4  The three-arm frontier: piecewise linear in log-log through the 3.695 s, 6.0 s and 10.043 s
        arms, read at the rule's polls per run. The arms must be monotone (more polls, less delay);
        a resample in which they are not is counted and left out, and a full sample in which they are
        not gets raw values and no gain. The verdict words are read on seeds 21 to 40, where all
        three arms and both rules are one code version and no seed was read before (amendment 5). The
        seeds 1 to 40 numbers are printed beside them with no verdict word.

line 5  `s8b6p0` on seeds 1 to 20 against the replay bands of spec.md. Lines 2 to 4 use measured arms
        only and are read whatever this line says (amendment 7).

Bootstrap: every seed set gets its own index draw `rng.integers(0, n, size=(2000, n))` from a fresh
`numpy.random.default_rng(20261119)`, taken the first time a quantity on that set is wanted. Every
interval on that set is computed from that one draw, so one index row applies to all stages at once
and the intervals on one seed set share their resamples. A draw depends on the number of seeds in
the set and on nothing else, so the intervals of one line cannot move because another line's seed
set changed.

Reads only `<run_dir>/meta.json` and `<run_dir>/debug/node_dirs/h*/semindex/events.jsonl` under the
store, and writes nothing there.

Environment: STORE (default /home/lab_shared/semindex-eval). Needs no PYTHONPATH of its own; it reads
the three perrun.csv files and numpy only.
"""
import csv
import glob
import json
import math
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV10 = os.path.join(HERE, os.pardir, "10")
EV16 = os.path.join(HERE, os.pardir, "16")
STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")

P = 0.4
BOOT_REPS = 2000
BOOT_SEED = 20261119

# The five s8 stages of this step and how many runs each must have for line 1.
S8_EXPECTED = {"s8t37": 20, "s8t37r": 20, "s8b3p695": 20, "s8b10p043": 20, "s8b6p0": 40}
# Logical arm -> (stage for seeds 1-20, stage for seeds 21-40).
ARM_SOURCE = {
    "B3": ("s6b3p695", "s8b3p695"),
    "B10": ("s6b10p043", "s8b10p043"),
    "B6": ("s8b6p0", "s8b6p0"),
    "T37": ("s7t37", "s8t37"),
    "T37r": ("s7t37r", "s8t37r"),
}
ARM_LABEL = {"B3": "fixed 3.695 s", "B10": "fixed 10.043 s", "B6": "fixed 6.0 s",
             "T37": "rule, on_timeout keep", "T37r": "rule, on_timeout reset"}
RULES = ("T37", "T37r")
OLD = 1, 20        # the seed range whose arms and rules were measured for evidence/10 and 16
NEW = 21, 40       # the seed range this step added

# spec.md pass line 5: the replay's pooled prediction for the 6.0 s arm and the plus or minus 6 %
# band around it, copied from spec.md as written there.
BAND6 = dict(delay_s=29.744, delay_lo=27.959, delay_hi=31.529, polls=159.4, polls_lo=149.84, polls_hi=168.96)
# The three intervals the rule (i_min 3.695, factor 2, i_max 10.043) can stand at: the floor, one
# back-off, and the ceiling (a second back-off would reach 14.78 and is clamped to i_max).
RULE_INTERVALS = (3.695, 7.39, 10.043)

WORDS_DIFF = ("seen", "not seen", "reversed")
WORDS_FRONT = ("above the fixed frontier", "not distinguishable from the fixed frontier",
               "below the fixed frontier")
WORDS_B6 = ("the trade-off bends below the straight line",
            "the two-arm line and the three-arm frontier are not distinguishable",
            "the trade-off bends above the straight line")


def load_perrun(path):
    """{(stage, seed): row} for the p 0.4 rows of a perrun.csv, numeric columns as floats."""
    out = {}
    for r in csv.DictReader(open(path)):
        if float(r["p"]) != P:
            continue
        d = {k: float(v) for k, v in r.items() if k != "stage"}
        out[(r["stage"], int(d["seed"]))] = d
    return out


def run_dir(stage, seed):
    return os.path.join(STORE, stage, f"p{P}", str(seed))


def validity(stage, seed):
    """(valid, flags) of one run from its meta.json `validity` block."""
    with open(os.path.join(run_dir(stage, seed), "meta.json")) as f:
        v = json.load(f)["validity"]
    return bool(v["valid"]), list(v["flags"])


def pooled(m):
    """(delay ms, polls per run, censored share) of a stage from its per-seed rows, pooled over records.

    Pooled over records because that is the `delay` spec.md names (`lat4 / n4` summed over the runs of
    a stage), which is also the pooling predict.py scores with.
    """
    lat, n, c, polls = m.sum(0)
    return lat / n, polls / len(m), c / (n + c)


def line_delay(arm_pts, x):
    """The log-log line through two arm points read at polls per run `x`, None when `x` is outside them.

    Beyond the measured arms there is no line to read, so the answer there is None: a gain computed off
    an extrapolated point would not be the comparison spec.md asks for.
    """
    (x0, y0), (x1, y1) = sorted(arm_pts)
    if not x0 <= x <= x1:
        return None
    w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y0) + w * (math.log(y1) - math.log(y0)))


def front_delay(arm_pts, x):
    """(delay of the three-arm frontier at `x`, reason it is undefined).

    The frontier is piecewise linear in log-log through the three arm points, so it is read on the one
    segment that contains `x`. Monotone means delay strictly falls as polls per run rise. A sample
    whose arms are not monotone has no frontier at all, and spec.md asks for such a sample to be
    counted and left out, so the reason is returned with the value.
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


def quantities(mats):
    """Every bootstrapped quantity of one sample, as {name: value or None} and {name: reason}.

    `mats` maps each logical arm to its per-seed array of [lat4, n4, c4, polls_sent]. All of the
    quantities are computed from the one sample, so an interval on this seed set and any other
    interval on the same seed set share their resamples exactly. A None value means the sample
    leaves that quantity undefined. A missing key means an arm the quantity needs is not in `mats`.
    Every line reads its quantities on its own seed set, which holds every arm that line names, so a
    missing key is never read.
    """
    base = {a: pooled(m) for a, m in mats.items()}
    pt = {a: (base[a][1], base[a][0]) for a in base}
    have = set(pt)
    two = [pt["B3"], pt["B10"]] if {"B3", "B10"} <= have else None
    three = [pt["B3"], pt["B6"], pt["B10"]] if {"B3", "B6", "B10"} <= have else None
    val, why = {}, {}
    for r in RULES:
        if r not in have:
            continue
        d, pol, _c = base[r]
        if two is not None:
            ref = line_delay(two, pol)
            val[f"gain_two_{r}"] = None if ref is None else 1 - d / ref
            why[f"gain_two_{r}"] = "" if ref is not None else "polls outside arms"
        if three is not None:
            fr, reason = front_delay(three, pol)
            val[f"gain_front_{r}"] = None if fr is None else 1 - d / fr
            why[f"gain_front_{r}"] = reason
    for kind in ("two", "front"):
        ka, kb = f"gain_{kind}_T37", f"gain_{kind}_T37r"
        if ka not in val or kb not in val:
            continue
        a, b = val[ka], val[kb]
        val[f"diff_{kind}"] = None if a is None or b is None else b - a
        why[f"diff_{kind}"] = why[ka] or why[kb]
    if two is not None and "B6" in have:
        d6, pol6, _c6 = base["B6"]
        ref6 = line_delay(two, pol6)
        val["gain_two_B6"] = None if ref6 is None else 1 - d6 / ref6
        why["gain_two_B6"] = "" if ref6 is not None else "polls outside arms"
    if set(RULES) <= have:
        val["ratio_delay"] = base["T37r"][0] / base["T37"][0]
        val["ratio_polls"] = base["T37r"][1] / base["T37"][1]
        why["ratio_delay"] = why["ratio_polls"] = ""
    return val, why


def fmt(x, nd=3):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def pct(x, nd=2):
    return "-" if x is None or x != x else f"{100 * x:.{nd}f}"


def interval_word(lo, hi, words):
    """`words` is (entirely above zero, spanning zero, entirely below zero)."""
    if lo != lo or hi != hi:
        return "not computed"
    if lo > 0:
        return words[0]
    if hi < 0:
        return words[2]
    return words[1]


def decision_rows(stage, seed):
    """Every `decision` event of kind state_poll / state_poll_reschedule of one run, and its late replies.

    Returns (rows, late_replies). The host logs are read in sorted order so that two runs of this
    script see the same rows in the same order. Only events.jsonl is opened under node_dirs: the rest
    of that tree holds the hosts' Cefore key material.
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
    """The no-pass-line decision-row quantities of one rule stage, summed over `seeds`."""
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
    """The index draw of every seed set that is bootstrapped.

    A seed set is drawn once, on first use, from a fresh `default_rng(BOOT_SEED)`, and the draw is
    kept, so every interval on that seed set comes from the same 2000 index rows. The draw is a
    function of the number of seeds alone, so what one seed set reports does not depend on which
    other seed sets the script was asked for. `log` records the seed sets in the order they were
    first wanted.
    """

    def __init__(self, rows_by_arm):
        self.rows = rows_by_arm
        self.cache = {}
        self.log = []

    def matrices(self, seeds):
        """{arm: array of [lat4, n4, c4, polls_sent] per seed}, seeds in the order given.

        An arm that does not hold every one of `seeds` is left out. A line's seed set is the
        intersection over the stages that line names (amendment 1), so an arm the line does not name
        can be short of a seed. Leaving that arm out makes the quantities that need it undefined on
        this seed set, and the lines that do not need it are still computed.

        2026-11-19 review: building every arm for every seed set raised KeyError on the first
        missing or flagged run of an arm, and the whole analysis then wrote no output at all, so
        line 1 could not even report the flag that caused it.
        """
        return {a: np.array([[self.rows[a][s][k] for k in ("lat4", "n4", "c4", "polls_sent")] for s in seeds],
                            float)
                for a in ARM_SOURCE if all(s in self.rows[a] for s in seeds)}

    def get(self, seeds, label):
        """(full-sample values, full-sample reasons, {name: (lo, hi)}, {name: Counter of drop reasons})."""
        key = tuple(seeds)
        if key in self.cache:
            return self.cache[key]
        mats = self.matrices(seeds)
        full, full_why = quantities(mats)
        n = len(seeds)
        # 2026-11-19 review: a single continuing stream for the whole script made the second seed
        # set's draw depend on the first, which is not what spec.md pre-registers and would have
        # moved the seeds 1 to 40 intervals whenever a seed dropped out of another line's set.
        idx = np.random.default_rng(BOOT_SEED).integers(0, n, size=(BOOT_REPS, n))
        self.log.append((label, n, min(seeds), max(seeds)))
        draws = {k: [] for k in full}
        dropped = {k: Counter() for k in full}
        for row in idx:
            sub = {a: m[row] for a, m in mats.items()}
            v, w = quantities(sub)
            for k, x in v.items():
                if x is None:
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
    here = load_perrun(f"{HERE}/perrun.csv")
    rows = dict(here)
    rows.update(load_perrun(f"{EV10}/perrun.csv"))
    rows.update(load_perrun(f"{EV16}/perrun.csv"))

    out = []          # results.csv rows
    L = []            # results.md lines

    def rec(line, stage, metric, value, note=""):
        out.append(dict(line=line, stage=stage, metric=metric, value=value, note=note))

    # ---- DONE and unflagged seeds of every stage in use ---------------------------------
    # DONE means the run has a row in a perrun.csv: perrun.py puts only runs with a DONE marker
    # there. Unflagged is meta.json validity. Both are read for the s6 and s7 stages as well, so
    # that amendment 1's seed rule is applied to the whole of a pooled line and not only to its
    # new half.
    stage_seeds = {}
    stage_flagged = {}
    for a, (old_stage, new_stage) in ARM_SOURCE.items():
        for stage in {old_stage, new_stage}:
            if stage in stage_seeds:
                continue
            got = sorted(s for (st, s) in rows if st == stage)
            flagged = {}
            ok = []
            for s in got:
                valid, flags = validity(stage, s)
                if valid and not flags:
                    ok.append(s)
                else:
                    flagged[s] = flags or ["valid=false"]
            stage_seeds[stage] = ok
            stage_flagged[stage] = flagged

    def arm_stage(arm, seed):
        return ARM_SOURCE[arm][0 if seed <= OLD[1] else 1]

    def arm_seeds(arm, lo, hi):
        """The seeds of `arm` in [lo, hi] that are DONE and unflagged in the stage that holds them."""
        return [s for s in range(lo, hi + 1) if s in stage_seeds[arm_stage(arm, s)]]

    def common(arms, lo, hi):
        sets = [set(arm_seeds(a, lo, hi)) for a in arms]
        return sorted(set.intersection(*sets))

    # One row per (arm, seed) for the bootstrap, taken from the stage that holds that seed.
    rows_by_arm = {a: {s: rows[(arm_stage(a, s), s)] for s in arm_seeds(a, 1, 40)} for a in ARM_SOURCE}
    boot = Boot(rows_by_arm)

    L += ["# evidence/17 results", "",
          "Computed by ana.py from perrun.csv here, ../10/perrun.csv, ../16/perrun.csv and the stored "
          "runs' meta.json and event logs. The pass lines, the bands and the verdict words are "
          "spec.md's and its amendments of 2026-11-19; the words are read off the numbers mechanically "
          "and nothing here adds to them. Every line names the number of seeds it used, and a line "
          "computed on fewer seeds than it names gets the word `incomplete` in place of its verdict "
          "(amendment 1).", "",
          "Logical arms and the stage each seed's run comes from:", "",
          "| arm | setting | seeds 1 to 20 | seeds 21 to 40 |", "|---|---|---|---|"]
    for a, (o, n_) in ARM_SOURCE.items():
        L.append(f"| {a} | {ARM_LABEL[a]} | `{o}` | `{n_}` |")
    L += [""]

    # ---- line 1 --------------------------------------------------------------------------
    L += ["## Line 1: all 120 runs DONE and valid", "",
          "DONE is a run with a DONE marker, which is what perrun.py puts in perrun.csv. Validity is "
          "`<run_dir>/meta.json` -> `validity` (`valid`, `flags`), written by semindex.evalrun from "
          "semindex.evalvalid. `results.json` carries the run's operations and no verdict, so it is "
          "not read.", "",
          "| stage | runs expected | runs DONE | valid and unflagged | flagged | flags seen |",
          "|---|---|---|---|---|---|"]
    total_done = 0
    total_ok = 0
    for stage, want in S8_EXPECTED.items():
        got = sorted(s for (st, s) in rows if st == stage)
        ok = stage_seeds[stage]
        fl = stage_flagged[stage]
        total_done += len(got)
        total_ok += len(ok)
        seen = "; ".join(f"seed {s}: {','.join(f)}" for s, f in sorted(fl.items())) or "none"
        L.append(f"| {stage} | {want} | {len(got)} | {len(ok)} | {len(fl)} | {seen} |")
        rec(1, stage, "runs_expected", want)
        rec(1, stage, "runs_done", len(got))
        rec(1, stage, "runs_valid", len(ok))
        rec(1, stage, "runs_flagged", len(fl))
    want_total = sum(S8_EXPECTED.values())
    if total_done < want_total:
        verdict1 = f"incomplete ({total_done} of {want_total} DONE)"
    elif total_ok < total_done:
        verdict1 = "fail"
    else:
        verdict1 = "pass"
    L += ["", f"{total_done} of {want_total} runs DONE, {total_ok} of them valid and unflagged.",
          f"Verdict: **{verdict1}**.", ""]
    rec(1, "", "runs_done_total", total_done)
    rec(1, "", "runs_valid_total", total_ok)
    rec(1, "", "verdict", verdict1)

    # The seed sets. Every one of them is the intersection over the stages its line names, so a
    # flagged run drops its seed from that line and the line reports the smaller count.
    seeds_rep = common(("B3", "B10") + RULES, *NEW)
    seeds_pool = common(("B3", "B10") + RULES, OLD[0], NEW[1])
    seeds_front_rep = common(("B3", "B6", "B10") + RULES, *NEW)
    seeds_front_pool = common(("B3", "B6", "B10") + RULES, OLD[0], NEW[1])
    seeds_ratio_rep = common(RULES, *NEW)
    seeds_ratio_pool = common(RULES, OLD[0], NEW[1])

    def block(seeds, label):
        return boot.get(seeds, label)

    def gain_table(seeds, mats_label, arms):
        """The measured delay, censored share and polls per run of `arms` on `seeds`."""
        lines = ["| arm | stage | runs | delay ms | delay s | censored share | polls per run |",
                 "|---|---|---|---|---|---|---|"]
        for a in arms:
            m = boot.matrices(seeds)[a]
            d, pol, c = pooled(m)
            src = "/".join(sorted({arm_stage(a, s) for s in seeds}))
            lines.append(f"| {a} | `{src}` | {len(seeds)} | {fmt(d, 1)} | {fmt(d / 1000.0)} | "
                         f"{fmt(c, 5)} | {fmt(pol, 2)} |")
            rec(mats_label, a, "delay_ms", f"{d:.6f}")
            rec(mats_label, a, "delay_s", f"{d / 1000.0:.6f}")
            rec(mats_label, a, "censored_share", f"{c:.6f}")
            rec(mats_label, a, "polls_per_run", f"{pol:.6f}")
        return lines

    def gain_rows(full, ci, keys, labels, line_no, tag):
        lines = ["| quantity | value | points | 95 % interval | interval in points |",
                 "|---|---|---|---|---|"]
        for k, lab in zip(keys, labels):
            v = full[k]
            lo, hi = ci[k]
            lines.append(f"| {lab} | {fmt(v, 4) if v is not None else 'not computed'} | "
                         f"{pct(v) if v is not None else '-'} | [{fmt(lo, 4)}, {fmt(hi, 4)}] | "
                         f"[{pct(lo)}, {pct(hi)}] |")
            rec(line_no, tag, k, "" if v is None else f"{v:.6f}")
            rec(line_no, tag, f"{k}_lo", f"{lo:.6f}")
            rec(line_no, tag, f"{k}_hi", f"{hi:.6f}")
        return lines

    # ---- line 2: the replication ---------------------------------------------------------
    full_r, why_r, ci_r, drop_r = block(seeds_rep, "line 2, seeds 21 to 40, four s8 stages")
    n_rep = len(seeds_rep)
    L += ["## Line 2: the replication on seeds 21 to 40", "",
          "Seeds 21 to 40 only, all four stages from this step's s8 runs. This reading is the answer "
          "to the question whether the difference is seen (amendment 2). It is independent of "
          "evidence/16: none of these seeds was read before.", "",
          f"Seeds used: {n_rep} of 20, DONE and unflagged in `s8b3p695`, `s8b10p043`, `s8t37` and "
          "`s8t37r`.", ""]
    L += gain_table(seeds_rep, 2, ("B3", "B10") + RULES)
    mats = boot.matrices(seeds_rep)
    (x0, y0), (x1, y1) = sorted((pooled(mats[a])[1], pooled(mats[a])[0]) for a in ("B3", "B10"))
    slope = (math.log(y1) - math.log(y0)) / (math.log(x1) - math.log(x0))
    L += ["", f"Log-log line through the two fixed arms: through ({fmt(x0, 2)} polls, {fmt(y0, 1)} ms) "
              f"and ({fmt(x1, 2)} polls, {fmt(y1, 1)} ms), slope d log delay / d log polls = "
              f"{fmt(slope)}.", ""]
    rec(2, "", "line_slope_loglog", f"{slope:.6f}")
    L += gain_rows(full_r, ci_r, ["gain_two_T37", "gain_two_T37r", "diff_two"],
                   ["gain of T37 against the two-arm line", "gain of T37r against the two-arm line",
                    "difference T37r minus T37"], 2, "")
    if n_rep < 20:
        verdict2 = f"incomplete ({n_rep} of 20 seeds DONE and unflagged in all four stages)"
    else:
        verdict2 = interval_word(*ci_r["diff_two"], WORDS_DIFF)
    L += ["", f"Interval from {BOOT_REPS} resamples of the {n_rep} seeds with "
              f"`default_rng({BOOT_SEED})`, the unit being the seed number and every stage recomputed "
              f"from the same resampled seeds. Resamples left out of the difference: "
              f"{sum(drop_r['diff_two'].values())}.", "",
          f"Verdict on {n_rep} seeds: **{verdict2}**.", ""]
    rec(2, "", "seeds_used", n_rep)
    rec(2, "", "boot_dropped", sum(drop_r["diff_two"].values()))
    rec(2, "", "verdict", verdict2)

    # ---- line 3: the pooled estimate -----------------------------------------------------
    full_p, why_p, ci_p, drop_p = block(seeds_pool, "line 3, seeds 1 to 40, s6 plus s8 arms and s7 plus s8 rules")
    n_pool = len(seeds_pool)
    L += ["## Line 3: the pooled estimate on seeds 1 to 40", "",
          "Arms `s6b3p695` and `s6b10p043` on seeds 1 to 20 and `s8b3p695` and `s8b10p043` on seeds 21 "
          "to 40; rules `s7t37` and `s7t37r` on seeds 1 to 20 and `s8t37` and `s8t37r` on seeds 21 to "
          "40.", "",
          "This is not an independent test and it mixes two code versions: it contains the 20 seeds "
          "evidence/16 already read, and its seed-1-to-20 half was measured with the code before the "
          "rule was added and without `PYTHONDONTWRITEBYTECODE`. It is the best estimate of the size "
          "of the difference, and its word is never quoted without this sentence.", "",
          f"Seeds used: {n_pool} of 40.", ""]
    L += gain_table(seeds_pool, 3, ("B3", "B10") + RULES)
    L += [""]
    L += gain_rows(full_p, ci_p, ["gain_two_T37", "gain_two_T37r", "diff_two"],
                   ["gain of T37 against the two-arm line", "gain of T37r against the two-arm line",
                    "difference T37r minus T37"], 3, "")
    if n_pool < 40:
        verdict3 = f"incomplete ({n_pool} of 40 seeds DONE and unflagged in all of the stages named)"
    else:
        verdict3 = interval_word(*ci_p["diff_two"], WORDS_DIFF)
    L += ["", f"Interval from {BOOT_REPS} resamples of the {n_pool} seeds with "
              f"`default_rng({BOOT_SEED})`, unit the seed number. Resamples left out of the difference: "
              f"{sum(drop_p['diff_two'].values())}.", "",
          f"Verdict word on {n_pool} seeds: **{verdict3}**. This line is the estimate of the size of "
          "the difference. It is not an independent test.", ""]
    rec(3, "", "seeds_used", n_pool)
    rec(3, "", "boot_dropped", sum(drop_p["diff_two"].values()))
    rec(3, "", "verdict", verdict3)
    if verdict2 == "not seen" and verdict3 == "seen":
        d = full_p["diff_two"]
        lo, hi = ci_p["diff_two"]
        L += [f"Lines 2 and 3 split, which amendment 3 names as the single most likely outcome at this "
              f"size. Reported as: not confirmed by the independent replication; pooled estimate "
              f"{pct(d)} points, 95 % interval [{pct(lo)}, {pct(hi)}] points.", ""]
        rec(3, "", "split_reading", f"not confirmed by the independent replication; pooled estimate "
                                    f"{pct(d)} points [{pct(lo)}, {pct(hi)}]")

    # ---- line 4: the three-arm frontier --------------------------------------------------
    full_fr, why_fr, ci_fr, drop_fr = block(seeds_front_rep, "line 4, seeds 21 to 40, five s8 stages")
    full_fp, why_fp, ci_fp, drop_fp = block(seeds_front_pool, "line 4, seeds 1 to 40, five arms")
    n_fr, n_fp = len(seeds_front_rep), len(seeds_front_pool)
    L += ["## Line 4: the three-arm frontier", "",
          "The frontier is piecewise linear in log-log through the 3.695 s, 6.0 s and 10.043 s arms, "
          "read at the rule's polls per run. The arms must be monotone, which here means delay falls "
          "strictly as polls per run rise. A resample whose arms are not monotone, and a resample "
          "whose rule polls fall outside the outer arms, is counted and left out.", "",
          "The verdict words are read on seeds 21 to 40, where all three arms and both rules are one "
          "code version and no seed was read before (amendment 5). The seeds 1 to 40 numbers stand "
          "beside them with no verdict word.", "",
          f"Seeds used: {n_fr} of 20 for the words, {n_fp} of 40 for the numbers beside them.", ""]
    L += gain_table(seeds_front_rep, 4, ("B3", "B6", "B10") + RULES)
    def frontier_block(full, ci, drop, seeds, n, want, with_words):
        tag = f"seeds_{min(seeds)}_{max(seeds)}"
        mats_f = boot.matrices(seeds)
        pts = sorted((pooled(mats_f[a])[1], pooled(mats_f[a])[0]) for a in ("B3", "B6", "B10"))
        mono = pts[0][1] > pts[1][1] > pts[2][1]
        lines = ["Full-sample arm points, sorted by polls per run: "
                 + ", ".join(f"({fmt(x, 2)} polls, {fmt(y, 1)} ms)" for x, y in pts)
                 + f". Monotone: {'yes' if mono else 'no'}.", ""]
        rec(4, tag, "arms_monotone", "yes" if mono else "no")
        if not mono:
            lines += ["The three arms of this full sample are not monotone, so no frontier gain is "
                      "computed on this seed set and the raw arm points above are what it reports.", ""]
            return lines
        keys = ["gain_front_T37", "gain_front_T37r", "diff_front", "gain_two_B6"]
        labels = ["gain of T37 against the three-arm frontier",
                  "gain of T37r against the three-arm frontier",
                  "difference T37r minus T37 against the frontier",
                  "gain of the 6.0 s arm against the two-arm line"]
        lines += gain_rows(full, ci, keys, labels, 4, tag)
        lines += ["", "| quantity | resamples left out | reasons |", "|---|---|---|"]
        for k, lab in zip(keys, labels):
            d = drop[k]
            why = "; ".join(f"{r} {c}" for r, c in sorted(d.items())) or "none"
            lines.append(f"| {lab} | {sum(d.values())} | {why} |")
            rec(4, tag, f"{k}_dropped", sum(d.values()), why)
        if not with_words:
            return lines
        lines += [""]
        if n < want:
            short = f"incomplete ({n} of {want} seeds DONE and unflagged in all five stages)"
            for lab in labels:
                lines.append(f"- {lab}: **{short}**")
            rec(4, tag, "verdict", short)
            return lines
        words = [interval_word(*ci["gain_front_T37"], WORDS_FRONT),
                 interval_word(*ci["gain_front_T37r"], WORDS_FRONT),
                 interval_word(*ci["diff_front"], WORDS_DIFF),
                 interval_word(*ci["gain_two_B6"], WORDS_B6)]
        for lab, w in zip(labels, words):
            lines.append(f"- {lab}: **{w}**")
        for k, w in zip(keys, words):
            rec(4, tag, f"{k}_verdict", w)
        return lines

    L += ["", f"Read on seeds 21 to 40 ({n_fr} seeds), with the verdict words.", ""]
    L += frontier_block(full_fr, ci_fr, drop_fr, seeds_front_rep, n_fr, 20, True)
    L += ["", f"The same quantities on seeds 1 to 40 ({n_fp} seeds), with no verdict word. The 6.0 s "
              "arm is all new code on every seed while the other two arms are the old code on seeds 1 "
              "to 20, which is why no word is attached here.", ""]
    L += frontier_block(full_fp, ci_fp, drop_fp, seeds_front_pool, n_fp, 40, False)
    rec(4, "", "seeds_used_21_40", n_fr)
    rec(4, "", "seeds_used_1_40", n_fp)

    # ---- line 5: the replay check of the new arm -----------------------------------------
    s6p0 = arm_seeds("B6", *OLD)
    L += ["", "## Line 5: the replay check of the 6.0 s arm", "",
          "`s8b6p0` on seeds 1 to 20 against the plus or minus 6 % bands of spec.md, copied from it as "
          "written. The centres are the replay's pooled prediction, the mean over 40 replicate "
          "streams; spec.md's amendment 7 records that a one-run-per-seed testbed is not expected to "
          "match a pooled replay line as closely as the per-experiment medians the evidence/16 bands "
          "used, and that how much of the 6 % is single-run noise is unknown. Lines 2 to 4 use "
          "measured arms only and are read whatever this line says.", "",
          f"Seeds used: {len(s6p0)} of 20.", "",
          "| quantity | measured | prediction | band | inside or outside | deviation % |",
          "|---|---|---|---|---|---|"]
    words5 = []
    if s6p0:
        m = np.array([[rows[("s8b6p0", s)][k] for k in ("lat4", "n4", "c4", "polls_sent")] for s in s6p0], float)
        d6, pol6, _c6 = pooled(m)
        for metric, value, mid, lo_b, hi_b, nd, key in (
            ("delay s", d6 / 1000.0, BAND6["delay_s"], BAND6["delay_lo"], BAND6["delay_hi"], 3, "delay_s"),
            ("polls per run", pol6, BAND6["polls"], BAND6["polls_lo"], BAND6["polls_hi"], 2, "polls_per_run"),
        ):
            word = "inside" if lo_b <= value <= hi_b else "outside"
            words5.append(word)
            dev = 100 * (value / mid - 1)
            L.append(f"| {metric} | {fmt(value, nd)} | {fmt(mid, nd)} | {fmt(lo_b, nd)} .. {fmt(hi_b, nd)} "
                     f"| {word} | {dev:+.2f} |")
            rec(5, "s8b6p0", f"{key}_measured", f"{value:.6f}")
            rec(5, "s8b6p0", f"{key}_band", word, f"{lo_b} .. {hi_b}")
            rec(5, "s8b6p0", f"{key}_deviation_pct", f"{dev:.4f}")
    if len(s6p0) < 20:
        verdict5 = f"incomplete ({len(s6p0)} of 20 seeds DONE and unflagged)"
    elif set(words5) == {"inside"}:
        verdict5 = "pass"
    else:
        verdict5 = "the replay's 6.0 s arm missed"
    L += ["", f"Verdict on {len(s6p0)} seeds: **{verdict5}**.", ""]
    rec(5, "s8b6p0", "seeds_used", len(s6p0))
    rec(5, "s8b6p0", "verdict", verdict5)

    # ---- also reported, without a pass line ----------------------------------------------
    L += ["## Also reported, without a pass line", "",
          "Every arm's delay, censored share and polls per run on each seed range, each on that arm's "
          "own DONE and unflagged seeds in the range. These counts are per arm, so they are not cut to "
          "the intersections the pass lines use.", "",
          "| arm | seed range | stage | runs | delay ms | delay s | censored share | polls per run |",
          "|---|---|---|---|---|---|---|---|"]
    for a in ARM_SOURCE:
        for lo, hi in (OLD, NEW, (OLD[0], NEW[1])):
            sd = arm_seeds(a, lo, hi)
            if not sd:
                L.append(f"| {a} | {lo} to {hi} | | 0 | | | | |")
                continue
            m = np.array([[rows[(arm_stage(a, s), s)][k] for k in ("lat4", "n4", "c4", "polls_sent")]
                          for s in sd], float)
            d, pol, c = pooled(m)
            src = "/".join(sorted({arm_stage(a, s) for s in sd}))
            L.append(f"| {a} | {lo} to {hi} | `{src}` | {len(sd)} | {fmt(d, 1)} | {fmt(d / 1000.0)} | "
                     f"{fmt(c, 5)} | {fmt(pol, 2)} |")
            rec(0, a, f"delay_ms_{lo}_{hi}", f"{d:.6f}")
            rec(0, a, f"censored_share_{lo}_{hi}", f"{c:.6f}")
            rec(0, a, f"polls_per_run_{lo}_{hi}", f"{pol:.6f}")
            rec(0, a, f"runs_{lo}_{hi}", len(sd))

    full_rr, _w, ci_rr, drop_rr = block(seeds_ratio_rep, "ratios, seeds 21 to 40, the two rule stages")
    full_rp, _w2, ci_rp, drop_rp = block(seeds_ratio_pool, "ratios, seeds 1 to 40, the two rule stages")
    L += ["", "The plain ratio of T37r to T37, with no line and no frontier in it. A ratio is defined "
              "in every resample, so none is left out.", "",
          "| seed range | seeds | quantity | ratio | 95 % interval | resamples left out |",
          "|---|---|---|---|---|---|"]
    for lo, hi, sd, full_x, ci_x, drop_x in ((NEW[0], NEW[1], seeds_ratio_rep, full_rr, ci_rr, drop_rr),
                                             (OLD[0], NEW[1], seeds_ratio_pool, full_rp, ci_rp, drop_rp)):
        for k, lab in (("ratio_delay", "delay T37r / T37"), ("ratio_polls", "polls T37r / T37")):
            lo_c, hi_c = ci_x[k]
            L.append(f"| {lo} to {hi} | {len(sd)} | {lab} | {fmt(full_x[k], 4)} | [{fmt(lo_c, 4)}, "
                     f"{fmt(hi_c, 4)}] | {sum(drop_x[k].values())} |")
            rec(0, f"seeds_{lo}_{hi}", k, f"{full_x[k]:.6f}")
            rec(0, f"seeds_{lo}_{hi}", f"{k}_lo", f"{lo_c:.6f}")
            rec(0, f"seeds_{lo}_{hi}", f"{k}_hi", f"{hi_c:.6f}")

    L += ["", "Whole-run counts over every decision row of `s8t37` and `s8t37r`, not cut to the "
              "measurement window, because what they describe is the rule's own arithmetic over every "
              "poll the run sent. An answered re-decision carries `empty`: true when the reply held "
              "nothing new, false when it held something new, null when the reply could not be "
              "decoded. A timed-out re-decision carries `empty` null by construction and is not "
              "counted among the answered ones.", "",
          "| stage | runs | polls | re-decisions | answered | timed out | empty true share of answered "
          "| empty false | empty null and answered | clamped | late replies |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    hists = {}
    for stage in ("s8t37", "s8t37r"):
        sd = stage_seeds[stage]
        x, hist = rule_extras(stage, sd)
        hists[stage] = hist
        share = x["empty_true"] / x["answered"] if x["answered"] else float("nan")
        L.append(f"| {stage} | {len(sd)} | {x['polls']} | {x['resched']} | {x['answered']} | "
                 f"{x['timed_out']} | {fmt(share, 4)} | {x['empty_false']} | {x['empty_null_answered']} | "
                 f"{x['clamped']} | {x['late']} |")
        for k, v in sorted(x.items()):
            rec(0, stage, f"extra_{k}", v)
        rec(0, stage, "extra_empty_true_share_of_answered", f"{share:.6f}")
    L += ["", "Every `interval_s` the two rule stages logged, against the three values the rule can "
              "stand at (i_min 3.695, one back-off 7.39, the ceiling 10.043). Any other value would "
              "fill the `other` column.", "",
          "| stage | " + " | ".join(str(v) for v in RULE_INTERVALS) + " | other | all rows |",
          "|---|" + "---|" * (len(RULE_INTERVALS) + 2)]
    for stage, hist in hists.items():
        other = {k: v for k, v in hist.items() if k not in RULE_INTERVALS}
        L.append(f"| {stage} | " + " | ".join(str(hist[v]) for v in RULE_INTERVALS) + " | "
                 + (json.dumps(other, sort_keys=True) if other else "0") + f" | {sum(hist.values())} |")
        for v in RULE_INTERVALS:
            rec(0, stage, f"interval_s_{v}", hist[v])
        rec(0, stage, "interval_s_other", sum(other.values()), json.dumps(other, sort_keys=True))

    # ---- how each seed set was drawn ------------------------------------------------------
    L += ["", "## How the bootstrap was drawn", "",
          f"Every seed set is resampled with a fresh `numpy.random.default_rng({BOOT_SEED})` of its "
          "own. The unit is the seed number: one index row of a draw selects the seeds, and every arm "
          "in use is recomputed from that one selection, so a resample holds the same seeds in all "
          "stages at once. A seed set is drawn once, as `rng.integers(0, n, size="
          f"({BOOT_REPS}, n))`, and every interval on that seed set comes from that draw, so the "
          "intervals on one seed set share their resamples. A seed set that a later line asks for "
          "again reuses the draw it already has. A draw is fixed by the number of seeds in the set, "
          "so the intervals one line reports cannot move because another line's seed set changed.", "",
          "| first wanted by | seeds | seed range |", "|---|---|---|"]
    for i, (label, n, lo, hi) in enumerate(boot.log, 1):
        L.append(f"| {label} | {n} | {lo} to {hi} |")
        rec(0, "bootstrap", f"draw_{i}", label, f"n={n}, seeds {lo} to {hi}")
    L += ["", "Resamples left out of every interval, over all lines. A gain is left out of a resample "
              "when that resample's rule polls per run fall outside the arms, and a frontier gain also "
              "when that resample's three arms are not monotone. A quantity that a seed set cannot "
              "define at all, because the seed set is short of a seed in an arm that quantity needs, "
              "has no row here.", "",
          "| seed set | quantity | resamples left out | reasons |", "|---|---|---|---|"]
    seen_sets = []
    for seeds in (seeds_rep, seeds_pool, seeds_front_rep, seeds_front_pool,
                  seeds_ratio_rep, seeds_ratio_pool):
        key = tuple(seeds)
        if key in seen_sets:
            continue
        seen_sets.append(key)
        label = f"seeds {min(seeds)} to {max(seeds)}, {len(seeds)} seeds"
        drop = boot.cache[key][3]
        for k in sorted(drop):
            why = "; ".join(f"{r} {c}" for r, c in sorted(drop[k].items())) or "none"
            L.append(f"| {label} | {k} | {sum(drop[k].values())} | {why} |")
            rec(0, f"seeds_{min(seeds)}_{max(seeds)}", f"dropped_{k}", sum(drop[k].values()), why)

    with open(f"{HERE}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["line", "stage", "metric", "value", "note"])
        w.writeheader()
        w.writerows(out)
    open(f"{HERE}/results.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
