"""The four pass lines of spec.md, from perrun.csv here, evidence/10 perrun.csv and the stored event logs.

Writes results.csv (one row per reported quantity) and results.md (the same numbers as tables). Every
verdict word in both is the mechanical one spec.md defines; nothing here reads more into a number than
that.

What each pass line needs, and where it comes from:

line 1  Each rule stage's run count and each run's validity. Validity is read from
        `<run_dir>/meta.json` -> `validity` (`valid`, `flags`), which `semindex.evalrun` writes from
        `semindex.evalvalid` at the end of the run. That is the same field `evidence/05/attrib.py`
        gates on, so a flagged run would already be dropped from the pair sums perrun.py reports.
        `results.json` holds the run's operation list and carries no verdict, so it is not used.
        A stage with fewer than 20 DONE runs is reported as incomplete; its pass word is withheld
        rather than being read off the runs that happen to exist.

line 2  Delay (pooled `sum(lat4) / sum(n4)` over the stage's runs, ms), censored share
        (`sum(c4) / sum(n4 + c4)`) and polls per run (mean of `polls_sent`) for the two fixed arms
        `s6b3p695` and `s6b10p043` at p 0.4 (per-seed rows of ../10/perrun.csv) and for the two rule
        stages. The two arms are joined by a straight line in log-log; gain is
        `1 - delay(rule) / line(polls(rule))`, undefined when a rule's polls per run fall outside the
        two arms. The reported interval is 2000 seed resamples with `default_rng(20261117)`, drawn as
        `rng.integers(0, n, size=(2000, n))` so that this bootstrap resamples seed indices exactly as
        predict.py's does; all four stages are recomputed from the same resampled seeds inside one
        resample. A resample whose rule polls land outside the arms has no gain difference and is
        counted instead of being folded in.

line 3  The measured delay and polls per run of each rule stage against the plus or minus 6 % bands
        spec.md prints. The band endpoints are taken from spec.md literally (the spec rounded them
        from predict.py's unrounded medians, so recomputing them from the rounded medians would move
        the last digit), and the deviation is against the prediction medians spec.md prints.

line 4  Each rerun against its original (../10/perrun.csv, p 0.4, seed 1): polls per run within 5 %
        and delay within 20 %, plus the exact check that every `interval_s` of every `state_poll` and
        `state_poll_reschedule` decision row is a value that run's setting allows.

Also reported without a pass line, per rule stage: the share of answered re-decisions whose reply was
empty, the number with `empty` null and trigger answered (an undecodable reply), `state_poll_late_reply`
events, clamped re-decisions, the trigger split, and the histogram of `interval_s` over the three values
the rule can produce.

The decision-row quantities (line 4 and the extras) are whole-run counts over every decision row in the
log. They are not cut to the measurement window, because what they check is the rule's own arithmetic,
which holds for every poll the run ever sent. Delay and polls per run stay windowed: those are the
measurement, and perrun.py's columns are already the windowed ones.

Reads only `<run_dir>/meta.json` and `<run_dir>/debug/node_dirs/h*/semindex/events.jsonl` under the
store, and writes nothing there.

Environment: STORE (default /home/lab_shared/semindex-eval). Needs no PYTHONPATH of its own; it reads
the two perrun.csv files and numpy only.
"""
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV10 = os.path.join(HERE, os.pardir, "10")
STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")

P = 0.4
SEEDS_EXPECTED = 20
ARMS = ("s6b3p695", "s6b10p043")
RULES = ("s7t37", "s7t37r")
# (rerun stage, the stage of the original run it repeats); both at p 0.4, seed 1.
RERUNS = (("s7short", "s6short"), ("s7b3p695", "s6b3p695"), ("s7b10p043", "s6b10p043"))
BOOT_REPS = 2000
BOOT_SEED = 20261117

# spec.md's prediction medians and the plus or minus 6 % bands it prints, copied as written there.
BANDS = {
    "s7t37": dict(delay_s=34.627, delay_lo=32.549, delay_hi=36.704, polls=142.50, polls_lo=133.95, polls_hi=151.05),
    "s7t37r": dict(delay_s=25.160, delay_lo=23.650, delay_hi=26.669, polls=180.15, polls_lo=169.34, polls_hi=190.96),
}
# The three intervals the rule (i_min 3.695, factor 2, i_max 10.043) can stand at: the floor, one
# back-off, and the ceiling (a second back-off would reach 14.78 and is clamped to i_max).
RULE_INTERVALS = (3.695, 7.39, 10.043)
RERUN_INTERVAL_RULE = {
    # s7short polls under a two-row policy table, so which value is allowed depends on the row the
    # decision read: consecutive_poll_timeouts 0 takes the first row, anything above it the second.
    "s7short": lambda row: 10.043 if row["state"].get("consecutive_poll_timeouts", 0) == 0 else 3.695,
    "s7b3p695": lambda row: 3.695,
    "s7b10p043": lambda row: 10.043,
}


def load_perrun(path):
    """{(stage, seed): row} for the p 0.4 rows of a perrun.csv, numeric columns as floats."""
    out = {}
    for r in csv.DictReader(open(path)):
        if float(r["p"]) != P:
            continue
        d = {k: float(v) for k, v in r.items() if k != "stage"}
        out[(r["stage"], int(d["seed"]))] = d
    return out


def matrix(rows, seeds):
    """The per-seed array [lat4, n4, c4, polls_sent] for `seeds`, in the order given."""
    return np.array([[rows[s]["lat4"], rows[s]["n4"], rows[s]["c4"], rows[s]["polls_sent"]] for s in seeds], float)


def pooled(m):
    """(delay ms, polls per run, censored share) of a stage from its per-seed rows, pooled over records.

    Pooled over records rather than averaged over seeds because that is the `delay` spec.md names
    (`lat4 / n4` summed over the runs of a stage) and the pooling predict.py scores with.
    """
    lat, n, c, polls = m.sum(0)
    return lat / n, polls / len(m), c / (n + c)


def line_delay(arm_pts, x):
    """The log-log line through `arm_pts` read at polls per run `x`, or None when `x` is outside them.

    None rather than an extrapolation: beyond the measured arms there is no line to read, and a gain
    computed off an invented point would not be the comparison spec.md asks for.
    """
    (x0, y0), (x1, y1) = sorted(arm_pts)
    if not x0 <= x <= x1:
        return None
    w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y0) + w * (math.log(y1) - math.log(y0)))


def gain_of(rule_m, arm_ms):
    """1 - delay(rule) / line(polls(rule)), or None when the rule's polls fall outside the arms."""
    delay, polls, _ = pooled(rule_m)
    pts = [(pooled(m)[1], pooled(m)[0]) for m in arm_ms]
    ref = line_delay(pts, polls)
    return None if ref is None else 1 - delay / ref


def run_dir(stage, seed):
    return os.path.join(STORE, stage, f"p{P}", str(seed))


def validity(stage, seed):
    """(valid, flags) of one run from its meta.json `validity` block."""
    with open(os.path.join(run_dir(stage, seed), "meta.json")) as f:
        v = json.load(f)["validity"]
    return bool(v["valid"]), list(v["flags"])


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


def rerun_intervals(stage, seed):
    """(rows seen, rows whose interval_s is not what the setting allows, per-kind row counts)."""
    rows, _late = decision_rows(stage, seed)
    allowed = RERUN_INTERVAL_RULE[stage]
    bad = [e for e in rows if e["action"]["interval_s"] != allowed(e)]
    return len(rows), bad, Counter(e["kind"] for e in rows)


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


def fmt(x, nd=3):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def pct(x, nd=2):
    return "-" if x is None or x != x else f"{100 * x:.{nd}f}"


def band_word(value, lo, hi):
    return "inside" if lo <= value <= hi else "outside"


def main():
    here = load_perrun(f"{HERE}/perrun.csv")
    ten = load_perrun(f"{EV10}/perrun.csv")
    out = []          # results.csv rows
    L = []            # results.md lines

    def rec(line, stage, metric, value, note=""):
        out.append(dict(line=line, stage=stage, metric=metric, value=value, note=note))

    seeds_of = defaultdict(list)
    for stage, seed in sorted(here) + sorted(ten):
        seeds_of[stage].append(seed)
    for v in seeds_of.values():
        v.sort()

    L += ["# evidence/16 results", "",
          "Computed by ana.py from perrun.csv here, ../10/perrun.csv and the stored runs' meta.json and "
          "event logs. The pass lines, the bands and the verdict words are spec.md's; the words are "
          "read off the numbers mechanically and nothing here adds to them.", ""]

    # ---- line 1: runs present and valid -------------------------------------------------
    L += ["## Line 1: all 40 runs valid", "",
          "Validity source: `<run_dir>/meta.json` -> `validity` (`valid`, `flags`), written by "
          "semindex.evalrun from semindex.evalvalid. `results.json` carries the run's operations and no "
          "verdict, so it is not read. A stage with fewer than 20 DONE runs is incomplete: its pass word "
          "is withheld.", "",
          "| stage | runs DONE | valid | flagged | flags seen | verdict |", "|---|---|---|---|---|---|"]
    for stage in RULES:
        seeds = seeds_of[stage]
        vals = {s: validity(stage, s) for s in seeds}
        n_valid = sum(1 for ok, _ in vals.values() if ok)
        flagged = {s: fl for s, (ok, fl) in vals.items() if fl or not ok}
        if len(seeds) < SEEDS_EXPECTED:
            word = f"incomplete ({len(seeds)} of {SEEDS_EXPECTED} DONE)"
        else:
            word = "pass" if not flagged else "fail"
        seen = "; ".join(f"seed {s}: {','.join(fl) or 'valid=false'}" for s, fl in sorted(flagged.items())) or "none"
        L.append(f"| {stage} | {len(seeds)} | {n_valid} | {len(flagged)} | {seen} | {word} |")
        rec(1, stage, "runs_done", len(seeds))
        rec(1, stage, "runs_valid", n_valid)
        rec(1, stage, "runs_flagged", len(flagged))
        rec(1, stage, "verdict", word, seen)

    # ---- line 2: gain difference ---------------------------------------------------------
    stages4 = ARMS + RULES
    common = sorted(set.intersection(*[set(seeds_of[s]) for s in stages4])) if all(seeds_of[s] for s in stages4) else []
    L += ["", "## Line 2: is the difference seen?", ""]
    if not common:
        L += ["No seed is DONE in all four of " + ", ".join(stages4) + ", so nothing on this line is computed.", ""]
        rec(2, "", "verdict", "not computed", "no seed DONE in all four stages")
    else:
        src = {s: (ten if s in ARMS else here) for s in stages4}
        mats = {s: matrix({k[1]: v for k, v in src[s].items() if k[0] == s}, common) for s in stages4}
        base = {s: pooled(mats[s]) for s in stages4}
        arm_pts = sorted((base[a][1], base[a][0]) for a in ARMS)
        L += ["Delay is pooled over records (`sum(lat4) / sum(n4)`, ms); polls per run is the mean of "
              "`polls_sent`; censored share is `sum(c4) / sum(n4 + c4)`. The arms come from "
              f"../10/perrun.csv at p {P}. Seeds used: the {len(common)} DONE in all four stages.", "",
              "| stage | runs | delay ms | delay s | censored share | polls per run |", "|---|---|---|---|---|---|"]
        for s in stages4:
            d, pol, c = base[s]
            L.append(f"| {s} | {len(common)} | {fmt(d, 1)} | {fmt(d / 1000.0)} | {fmt(c, 5)} | {fmt(pol, 2)} |")
            rec(2, s, "delay_ms", f"{d:.6f}")
            rec(2, s, "delay_s", f"{d / 1000.0:.6f}")
            rec(2, s, "censored_share", f"{c:.6f}")
            rec(2, s, "polls_per_run", f"{pol:.6f}")
        (x0, y0), (x1, y1) = arm_pts
        slope = (math.log(y1) - math.log(y0)) / (math.log(x1) - math.log(x0))
        L += ["", f"Log-log line through the two arms: through ({fmt(x0, 2)} polls, {fmt(y0, 1)} ms) and "
                  f"({fmt(x1, 2)} polls, {fmt(y1, 1)} ms), slope d log delay / d log polls = {fmt(slope)}.", "",
              "| rule | polls per run | line delay ms at those polls | measured delay ms | gain | polls inside arms |",
              "|---|---|---|---|---|---|"]
        rec(2, "", "line_slope_loglog", f"{slope:.6f}")
        gains = {}
        for s in RULES:
            d, pol, _c = base[s]
            ref = line_delay(arm_pts, pol)
            gains[s] = None if ref is None else 1 - d / ref
            inside = "yes" if ref is not None else f"no (arms span {fmt(x0, 2)} .. {fmt(x1, 2)})"
            L.append(f"| {s} | {fmt(pol, 2)} | {fmt(ref, 1) if ref is not None else '-'} | {fmt(d, 1)} | "
                     f"{fmt(gains[s], 4) if gains[s] is not None else 'not computed'} | {inside} |")
            rec(2, s, "line_delay_ms", "" if ref is None else f"{ref:.6f}")
            rec(2, s, "gain", "" if gains[s] is None else f"{gains[s]:.6f}")
            rec(2, s, "polls_inside_arms", "yes" if ref is not None else "no")
        diff = None if None in gains.values() else gains["s7t37r"] - gains["s7t37"]
        rec(2, "", "gain_diff", "" if diff is None else f"{diff:.6f}", "s7t37r minus s7t37")

        rng = np.random.default_rng(BOOT_SEED)
        n = len(common)
        idx = rng.integers(0, n, size=(BOOT_REPS, n))
        boot, outside = [], 0
        for row in idx:
            sub = {s: mats[s][row] for s in stages4}
            a = gain_of(sub["s7t37"], [sub[x] for x in ARMS])
            b = gain_of(sub["s7t37r"], [sub[x] for x in ARMS])
            if a is None or b is None:
                outside += 1
            else:
                boot.append(b - a)
        lo = float(np.percentile(boot, 2.5)) if boot else float("nan")
        hi = float(np.percentile(boot, 97.5)) if boot else float("nan")
        if n < SEEDS_EXPECTED:
            verdict = f"incomplete ({n} of {SEEDS_EXPECTED} seeds in all four stages)"
        elif diff is None or not boot:
            verdict = "not computed (a rule's polls fall outside the arms)"
        elif lo > 0:
            verdict = "seen"
        elif hi < 0:
            verdict = "reversed"
        else:
            verdict = "not seen"
        L += ["", f"Gain difference s7t37r minus s7t37: {fmt(diff, 4) if diff is not None else 'not computed'} "
                  f"({pct(diff) if diff is not None else '-'} points). Interval from {BOOT_REPS} resamples of the "
                  f"{n} seeds with `default_rng({BOOT_SEED})`, all four stages recomputed from the same resampled "
                  f"seeds: [{fmt(lo, 4)}, {fmt(hi, 4)}] ([{pct(lo)}, {pct(hi)}] points) over {len(boot)} "
                  f"resamples with a defined gain; {outside} resamples put a rule's polls per run outside the arms "
                  "and were left out.", "",
              f"Verdict: **{verdict}**.", ""]
        rec(2, "", "boot_lo", f"{lo:.6f}")
        rec(2, "", "boot_hi", f"{hi:.6f}")
        rec(2, "", "boot_reps_used", len(boot))
        rec(2, "", "boot_reps_outside_arms", outside)
        rec(2, "", "seeds_used", n)
        rec(2, "", "verdict", verdict)

    # ---- line 3: against the prediction bands -------------------------------------------
    # On each rule stage's own DONE seeds, not on the seed set line 2 shares with the arms: this line
    # compares a rule against the replay and never touches an arm. With all 20 runs of both stages
    # DONE the two seed sets are the same 1 to 20.
    L += ["## Line 3: did the replay predict it?", "",
          "Measured on each rule stage's own DONE runs. Bands are spec.md's plus or minus 6 % bands, "
          "copied from it; the deviation is against the prediction medians it prints.", "",
          "| stage | runs | quantity | measured | prediction | band | inside or outside | deviation % |",
          "|---|---|---|---|---|---|---|---|"]
    words3 = []
    short3 = []
    for s in RULES:
        seeds = seeds_of[s]
        if not seeds:
            L.append(f"| {s} | 0 | | no run DONE | | | | |")
            short3.append(s)
            continue
        d, pol, _c = pooled(matrix({k[1]: v for k, v in here.items() if k[0] == s}, seeds))
        b = BANDS[s]
        if len(seeds) < SEEDS_EXPECTED:
            short3.append(s)
        for key, metric, value, mid, lo_b, hi_b, nd in (
            ("delay_s", "delay s", d / 1000.0, b["delay_s"], b["delay_lo"], b["delay_hi"], 3),
            ("polls_per_run", "polls per run", pol, b["polls"], b["polls_lo"], b["polls_hi"], 2),
        ):
            word = band_word(value, lo_b, hi_b)
            words3.append(word)
            dev = 100 * (value / mid - 1)
            L.append(f"| {s} | {len(seeds)} | {metric} | {fmt(value, nd)} | {fmt(mid, nd)} | {fmt(lo_b, nd)} .. "
                     f"{fmt(hi_b, nd)} | {word} | {dev:+.2f} |")
            rec(3, s, f"{key}_measured", f"{value:.6f}")
            rec(3, s, f"{key}_band", word, f"{lo_b} .. {hi_b}")
            rec(3, s, f"{key}_deviation_pct", f"{dev:.4f}")
        rec(3, s, "runs_used", len(seeds))
    if short3:
        counts = ", ".join(f"{s} {len(seeds_of[s])}" for s in short3)
        verdict3 = f"incomplete ({counts} of {SEEDS_EXPECTED} DONE)"
        note3 = ("The rows above are measured on an incomplete stage, so this line gets no pass word: "
                 f"runs DONE {counts}.")
    elif set(words3) == {"inside"}:
        verdict3 = "pass"
        note3 = "All four values are inside their band."
    else:
        verdict3 = "fail"
        note3 = ("A value falls outside its band. spec.md then asks that the decision sequence around "
                 "timeouts be compared between the testbed log and the replay and the point of "
                 "disagreement reported, before line 2 is read.")
    L += ["", f"Verdict: **{verdict3}**. {note3}", ""]
    rec(3, "", "verdict", verdict3)

    # ---- line 4: the three reruns --------------------------------------------------------
    L += ["## Line 4: existing policies unchanged (the three reruns)", "",
          "Polls per run and delay are the windowed per-run values of the single seed-1 run on each side. "
          "The interval check is over every `state_poll` and `state_poll_reschedule` decision row of the "
          "rerun, whole run, not cut to the measurement window.", "",
          "| rerun | original | polls rerun | polls original | polls band +-5 % | delay rerun s | delay original s "
          "| delay band +-20 % | decision rows | rows by kind | bad intervals |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for new, old in RERUNS:
        a = here.get((new, 1))
        b = ten.get((old, 1))
        if a is None or b is None:
            L.append(f"| {new} | {old} | run missing | | | | | | | | |")
            rec(4, new, "status", "run missing")
            continue
        pol_n, pol_o = a["polls_sent"], b["polls_sent"]
        dly_n, dly_o = a["lat4"] / a["n4"] / 1000.0, b["lat4"] / b["n4"] / 1000.0
        pw = band_word(pol_n, 0.95 * pol_o, 1.05 * pol_o)
        dw = band_word(dly_n, 0.80 * dly_o, 1.20 * dly_o)
        total, bad, kinds = rerun_intervals(new, 1)
        kind_txt = " + ".join(f"{k} {kinds[k]}" for k in ("state_poll", "state_poll_reschedule") if kinds[k])
        L.append(f"| {new} | {old} | {int(pol_n)} | {int(pol_o)} | {pw} | {fmt(dly_n)} | {fmt(dly_o)} | {dw} | "
                 f"{total} | {kind_txt} | {len(bad)} |")
        rec(4, new, "polls_rerun", int(pol_n))
        rec(4, new, "polls_original", int(pol_o))
        rec(4, new, "polls_band", pw, f"{0.95 * pol_o:.2f} .. {1.05 * pol_o:.2f}")
        rec(4, new, "delay_rerun_s", f"{dly_n:.6f}")
        rec(4, new, "delay_original_s", f"{dly_o:.6f}")
        rec(4, new, "delay_band", dw, f"{0.80 * dly_o:.3f} .. {1.20 * dly_o:.3f}")
        rec(4, new, "decision_rows", total, kind_txt)
        rec(4, new, "bad_intervals", len(bad))
        for e in bad[:5]:
            rec(4, new, "bad_interval_row", json.dumps(e, sort_keys=True))
    L += ["", "The two exact checks of this line are the interval check above and the policy tests at merge; "
              "the two bands are the coarse guide spec.md calls them, since answered against timed out is a race "
              "on the testbed.", ""]

    # ---- also reported -------------------------------------------------------------------
    L += ["## Also reported, without a pass line", "",
          "Whole-run counts over every decision row of the rule stages' DONE runs. An answered re-decision "
          "carries `empty`: true when the reply held nothing new, false when it held something new, null when "
          "the reply could not be decoded. A timed-out re-decision carries `empty` null by construction and is "
          "not counted among the answered ones.", "",
          "| stage | runs | polls | re-decisions | answered | timed out | empty true share of answered | empty false "
          "| empty null and answered | clamped | late replies |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    hists = {}
    for s in RULES:
        seeds = seeds_of[s]
        if not seeds:
            L.append(f"| {s} | 0 | | | | | | | | | |")
            continue
        x, hist = rule_extras(s, seeds)
        hists[s] = hist
        share = x["empty_true"] / x["answered"] if x["answered"] else float("nan")
        L.append(f"| {s} | {len(seeds)} | {x['polls']} | {x['resched']} | {x['answered']} | {x['timed_out']} | "
                 f"{fmt(share, 4)} | {x['empty_false']} | {x['empty_null_answered']} | {x['clamped']} | {x['late']} |")
        for k, v in sorted(x.items()):
            rec(0, s, f"extra_{k}", v)
        rec(0, s, "extra_empty_true_share_of_answered", f"{share:.6f}")
    if hists:
        L += ["", "Every `interval_s` the rule stages logged, against the three values the rule can stand at "
                  "(i_min 3.695, one back-off 7.39, the ceiling 10.043). Any other value would be a fourth "
                  "column here.", "",
              "| stage | " + " | ".join(str(v) for v in RULE_INTERVALS) + " | other | all rows |",
              "|---|" + "---|" * (len(RULE_INTERVALS) + 2)]
        for s, hist in hists.items():
            other = {k: v for k, v in hist.items() if k not in RULE_INTERVALS}
            L.append(f"| {s} | " + " | ".join(str(hist[v]) for v in RULE_INTERVALS) + " | "
                     + (json.dumps(other, sort_keys=True) if other else "0") + f" | {sum(hist.values())} |")
            for v in RULE_INTERVALS:
                rec(0, s, f"interval_s_{v}", hist[v])
            rec(0, s, "interval_s_other", sum(other.values()), json.dumps(other, sort_keys=True))

    with open(f"{HERE}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["line", "stage", "metric", "value", "note"])
        w.writeheader()
        w.writerows(out)
    open(f"{HERE}/results.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
