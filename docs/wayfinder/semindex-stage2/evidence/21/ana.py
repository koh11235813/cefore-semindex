"""The three pass lines of spec.md for evidence/21, from perrun.csv and nothing else.

    OUT=<dir> /home/lab_shared/cefore-semindex/.venv/bin/python3 -B ana.py

Writes `<OUT>/results.md` (the reading, for a human), `<OUT>/results.csv` (every number of
results.md in long form) and prints results.md on stdout. Every verdict word is the
mechanical one spec.md defines and nothing here reads more into a number than that.

One stage, two cells: `s11a` p 0.2 and p 0.4, workload w0, seeds 1 to 100 each, 200 runs,
the choices mode re-deciding its interval at every resolution. perrun.py has already read
the store and the node logs, so this script opens perrun.csv, the frozen predict.txt and
the two smoke runs' meta.json, and nothing else.

WHAT EACH PASS LINE NEEDS

line 3  All 200 runs DONE and valid with no evalvalid flag, the flagged ones broken down
        by cell and by flag name. Beside it, what the runs themselves record: one clean
        semindex commit, workload w0, the re-decision knob on, the three arms, and the
        seed set the step asked for. A run recording settings other than the step's stops
        this script before any measured number is printed, because every count below would
        then be counted on a stage nobody specified. A run that is missing or not DONE does
        not stop anything: it puts `incomplete` in place of the verdict word of every line,
        which is evidence/20's behaviour, and the numbers are still printed.

line 4  Per cell, over the cell's valid runs: the decision rows per run (send rows and
        reschedule rows counted separately, over the whole run), the share of the
        reschedule rows that fell to each of the three arms with the tolerance
        `1/3 +- 3*sqrt((2/9)/N)` read at the measured N, the timeout share, and the
        truncated rows on each kind of row. The pass or fail word of the line is whether
        all three shares are inside the tolerance. Reported with them, with no pass line
        behind them: the polls whose successor left outside the smoke gate's 0.5 s, the
        answered reschedule rows whose `empty` is null, the clamped share, the
        `state_poll_late_reply` count, the reschedule rows per send row, and one integrity
        count of this analysis's own, the reschedule rows that are not a uniform draw over
        the three arms.

line 5  The headline delay `sum(hlat4)/sum(hn4)` and the polls per run (the mean of
        evalagg's in-window count over the 8 directed pairs) at both p, four values, each
        against the plus or minus 6 % band of predict.txt. All 4 inside reads `predicted`,
        1 or 2 outside `mixed`, 3 or more outside `missed`. A value a cell produced no
        valid run for is neither inside nor outside, and the line then reads `not
        measured` rather than one of spec.md's three words.

The predicted constants below are retyped from predict.txt, and `check_prediction` parses
that file at run time and stops the script if any of them has drifted from the digits it
prints. A retyped number can part from its source without anything failing, and the
analysis would then hold the measurement against a prediction nobody wrote.

ENVIRONMENT

  STORE   default /home/lab_shared/semindex-eval, read only for the two smoke runs.
  STAGE   default s11a.
  SEEDS   default 1-100, the seed set pass line 3 requires per cell.
  OUT     default this directory, where perrun.csv is read and the two results files are
          written.

Whenever STORE, STAGE, SEEDS or OUT differs from its default, the first line of stdout, of
results.md and of results.csv says that the output is a test of the script and not the
reading of stage s11a. Every test of these scripts runs on s11smoke or on a fake store, and
an unlabelled number out of such a run would be read as the step's result.
"""
import csv
import json
import math
import os
import re
import statistics
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")
STAGE = os.environ.get("STAGE", "s11a")
SEEDS = os.environ.get("SEEDS", "1-100")
OUT = os.environ.get("OUT", HERE)
OVERRIDDEN = (STORE != "/home/lab_shared/semindex-eval" or STAGE != "s11a"
              or SEEDS != "1-100" or os.path.abspath(OUT) != HERE)
# No comma anywhere in the banner: it is written as the first line of results.csv, and a
# reader that does not skip `#` lines (csv.DictReader, pandas.read_csv) would otherwise
# split it into fields and take it for the header row, losing every column name.
BANNER = (f"test input: stage {STAGE}; seeds {SEEDS}; store {STORE}; output {OUT}. "
          "These numbers are a test of the script and not the reading of stage s11a.")

PS = (0.2, 0.4)


def parse_seeds(text):
    """The seed set an environment value such as `1-100` or `101` or `1-5,9` names."""
    out = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# How many runs a complete cell holds, read off SEEDS rather than written as 100. On the
# stage the two are the same; on a test input with fewer seeds they are not, and a hard
# 100 would make the complete-stage branch of every pass line below unreachable in a test,
# so the pass and fail words would never be exercised before the stage is read.
RUNS_PER_CELL = len(parse_seeds(SEEDS))
RUNS_WANTED = RUNS_PER_CELL * len(PS)
BAND_PCT = 6.0
ARMS = (3.695, 6.0, 10.043)
ARM_COLS = ("arm_3695", "arm_6000", "arm_10043")
UNIFORM = 1.0 / len(ARMS)
# spec.md "Runs": the settings every run of this step was made with. A run recording
# anything else is not a run of this step, whatever stage directory it sits in.
WANT_COMMIT = "a8add2cff4275317ddceea0a6068e5a1dd18e3da"
WANT_WORKLOAD = "w0"
WANT_CHOICES = ";".join(repr(a) for a in ARMS)
NODES_PER_RUN = 5

# ---- predict.txt, copied as printed there and checked against it at run time ------------
# p -> (delay s, band lo, band hi, polls per run, band lo, band hi, censored share). The
# band ends are predict.txt's own printed digits and not the median times 1 -+ 6 %: the two
# differ in the third decimal because the file rounds the median after forming the band,
# and spec.md pass line 5 reads "the prediction tabled above", which is this file's table.
PRED5 = {
    0.2: (24.225, 22.771, 25.678, 145.51, 136.78, 154.24, 0.000),
    0.4: (34.930, 32.835, 37.026, 145.70, 136.96, 154.45, 0.021),
}
# p -> (send median, send min, send max, resched median, resched min, resched max,
#       reschedule rows per send, share of reschedule rows written past the stop instant).
PRED_ROWS = {
    0.2: (160.0, 148.0, 176.0, 160.0, 148.0, 176.0, 1.000, 0.004),
    0.4: (161.0, 146.0, 174.0, 161.0, 146.0, 174.0, 1.000, 0.006),
}
# p -> (timeout share, its min, its max, clamped share, the three arm shares, N, 1/3,
#       the tolerance at that N). N and the tolerance beside it are for information: the
#       pass line evaluates the same formula at the N counted after the run.
PRED_RES = {
    0.2: (0.191, 0.189, 0.192, 0.000, (0.329, 0.337, 0.333), 16078, 0.3333, 0.0112),
    0.4: (0.368, 0.366, 0.369, 0.000, (0.331, 0.336, 0.333), 16099, 0.3333, 0.0111),
}

# spec.md's own sentences that limit what each line may be read as. They are printed beside
# the numbers rather than at the end, so that a reader meets them where the number is.
TEXT5 = {
    "predicted": "All four values lie inside the band spec.md fixed before the runs. That "
                 "is a statement about the replay on these four values and about no "
                 "decision maker: this stage draws uniformly and runs no policy.",
    "mixed": "spec.md: every value outside the band is reported with how far outside and "
             "in which direction, whatever the count is, which is the list below.",
    "missed": "spec.md: on `missed` the replay's copy of the re-decision branch is the "
              "first suspect, because it is the one piece of this step that has never "
              "been held against a measurement. The comparison to run then is "
              "evidence/16's: the decision sequences around timeouts in a testbed log "
              "against the replay's, reported before any other conclusion is written. "
              "What a miss costs is the later screening, since tickets 09 to 12 pick "
              "candidates by replay scoring.",
    "not measured": "spec.md's three words presume four measured values. A value with no "
                    "valid run behind it is neither inside its band nor outside it, so "
                    "none of the three words is read: calling the line `predicted` would "
                    "say the replay reproduced a value this stage never measured.",
    "incomplete": "spec.md pass line 3 is not met, so no word is read off these four "
                  "values yet. The numbers above are printed on the runs that are DONE "
                  "and valid, and they are not the stage's reading.",
}


# ---------------------------------------------------------------------------
# predict.txt
# ---------------------------------------------------------------------------
def check_prediction(path):
    """Stop unless every predicted constant above equals the digits predict.txt prints.

    Returns the list of (name, value) pairs matched, so results.md can say how many
    numbers were checked rather than only that the check passed. Each of the three tables
    has a row per p; they are told apart by their shape, the `[band]` bracket of the pass
    line 5 table and the `+-` of the resolutions table.
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    num = r"([\d.]+)"
    got = {}
    five = re.compile(rf"^\s*(0\.\d)\s*\|\s*{num}\s*\[\s*{num}\s*\.\.\s*{num}\s*\]\s*"
                      rf"[\d.]+\s*\.\.\s*[\d.]+\s*\|\s*{num}\s*\[\s*{num}\s*\.\.\s*{num}"
                      rf"\s*\]\s*[\d.]+\s*\.\.\s*[\d.]+\s*\|\s*{num}\s*$", re.M)
    for m in five.finditer(text):
        got[("line5", float(m.group(1)))] = tuple(float(x) for x in m.groups()[1:])
    rows = re.compile(rf"^\s*(0\.\d)\s*\|\s*{num}\s+{num}\s*\.\.\s*{num}\s*\|\s*{num}\s+"
                      rf"{num}\s*\.\.\s*{num}\s*\|\s*{num}\s*\|\s*{num} of the reschedule "
                      r"rows\s*$", re.M)
    for m in rows.finditer(text):
        got[("rows", float(m.group(1)))] = tuple(float(x) for x in m.groups()[1:])
    res = re.compile(rf"^\s*(0\.\d)\s*\|\s*{num}\s+{num}\s*\.\.\s*{num}\s*\|\s*{num}\s*\|"
                     rf"\s*{num}\s*/\s*{num}\s*/\s*{num}\s*\|\s*(\d+)\s*\|\s*{num}\s*\+-"
                     rf"\s*{num}\s*$", re.M)
    for m in res.finditer(text):
        g = m.groups()
        got[("res", float(g[0]))] = (float(g[1]), float(g[2]), float(g[3]), float(g[4]),
                                     (float(g[5]), float(g[6]), float(g[7])), int(g[8]),
                                     float(g[9]), float(g[10]))
    want = {}
    for p in PS:
        want[("line5", p)] = PRED5[p]
        want[("rows", p)] = PRED_ROWS[p]
        want[("res", p)] = PRED_RES[p]
    bad, matched = [], []
    for key, expect in sorted(want.items()):
        if key not in got:
            bad.append(f"{key[0]} p {key[1]}: no such row found in {os.path.basename(path)}")
        elif got[key] != expect:
            bad.append(f"{key[0]} p {key[1]}: ana.py carries {expect}, "
                       f"{os.path.basename(path)} prints {got[key]}")
        else:
            matched.append((f"{key[0]} p {key[1]}", expect))
    if bad:
        raise SystemExit("the predicted constants in ana.py no longer match predict.txt:\n  "
                         + "\n  ".join(bad))
    # The tolerance formula itself, held against the value predict.txt prints beside its
    # own N. PRED_RES carries those digits for display only and the pass word of line 4 is
    # read off `tolerance()` at the measured N, so without this a mistyped formula would
    # pass every comparison above and still move every inside-or-outside word.
    for p in PS:
        n, printed = PRED_RES[p][5], PRED_RES[p][7]
        if round(tolerance(n), 4) != printed:
            raise SystemExit(f"tolerance({n}) is {tolerance(n):.6f} in ana.py, "
                             f"predict.txt prints {printed} at p {p}")
    return matched


# ---------------------------------------------------------------------------
# perrun.csv and small formatting helpers
# ---------------------------------------------------------------------------
def load_perrun(path):
    """perrun.py's rows, numbers as floats, or an empty list when the file is not there.

    A missing file is the state of a store that holds no DONE run of the stage yet, and
    pass line 3 reads that as `incomplete` rather than as a failure of this script. The
    `#` line perrun.py writes when it was given a test input is skipped here and reported
    by this script's own banner.
    """
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(ln for ln in fh if not ln.startswith("#")))
    text_cols = ("stage", "workload", "flags", "semindex_git_commit", "choices", "valid",
                 "semindex_git_dirty", "redecide")
    return [{k: (v if k in text_cols else float(v)) for k, v in r.items()} for r in rows]


def fmt(x, nd=3):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def signed(x, nd=2):
    """`x` with its sign, or a dash when it is undefined.

    A deviation has to carry its direction, which spec.md pass line 5 asks for on every
    value whatever the count inside the band is, so an undefined one is printed and not
    skipped.
    """
    return "-" if x is None or x != x else f"{x:+.{nd}f}"


def total(runs, col):
    return sum(r[col] for r in runs)


def spread(runs, col):
    """(median, min, max) of one column over a cell's runs, or dashes on no run."""
    if not runs:
        return float("nan"), float("nan"), float("nan")
    xs = [r[col] for r in runs]
    return statistics.median(xs), min(xs), max(xs)


def tolerance(n):
    """`3*sqrt((2/9)/N)`, the half width spec.md fixes on a share per arm, at N rows.

    Three standard deviations of a binomial share at 1/3. The formula is the ticket's and
    is fixed before the run; what moves is only the N it is read at, which is the number of
    reschedule rows the cell actually produced.
    """
    return 3 * math.sqrt((2.0 / 9.0) / n) if n else float("nan")


def smoke_runs():
    """[(p, what s11smoke's meta.json records)], for the separate report of pass line 3.

    spec.md keeps the two pilot runs out of the 200 and reports them on their own. They are
    read here from the store rather than from perrun.csv, because perrun.csv holds one
    stage and that stage is the sweep's.
    """
    out = []
    for p in PS:
        path = os.path.join(STORE, "s11smoke", f"p{p:g}", "101", "meta.json")
        if not os.path.exists(path):
            out.append((p, None))
            continue
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        v = m.get("validity") or {}
        out.append((p, {"valid": bool(v.get("valid")),
                        "flags": ";".join(v.get("flags") or []) or "none",
                        "commit": m.get("semindex_git_commit"),
                        "redecide": m.get("state_poll_choice_redecide"),
                        "choices": m.get("state_poll_interval_choices_s")}))
    return out


# ---------------------------------------------------------------------------
def settings_check(rows, wanted):
    """(table rows, mismatches) of what the runs record about how they were made.

    The mismatches are the cases spec.md's own settings fix: a commit other than the one
    the step ran, a dirty tree, another workload, the knob off, other arms, or a seed
    outside the set the step asked for. Any of them means the rows are not this step's
    runs, and the caller stops on them before a measured number is printed. Fewer runs than
    the step asked for is not among them: a partial sweep is read as `incomplete`.
    """
    table, bad = [], []
    # The stage is the one identity of a row that does not come off the row itself
    # anywhere else: STAGE names the directory perrun.py read, and a perrun.csv of another
    # stage left behind in OUT would be read under this run's name and pass every check
    # below.
    stages = sorted({r["stage"] for r in rows})
    if stages and stages != [STAGE]:
        bad.append(f"perrun.csv holds stage(s) {stages}, this run reads {STAGE}")
    # The same for p: every table below selects rows by `r["p"] == p` over PS, so a row at
    # a third p is counted in no table, trips no check and is not even in the run totals.
    # Without this stop it would be reported as a run that is missing instead.
    # Only a p outside PS is a mismatch. A cell of PS that holds no run yet is a sweep that
    # has not reached it, and that reads `incomplete` like any other missing run.
    stray = sorted({r["p"] for r in rows} - set(PS))
    if stray:
        bad.append(f"perrun.csv holds p value(s) {stray} beside the {sorted(PS)} this "
                   "analysis reads")
    for p in PS:
        runs = [r for r in rows if r["p"] == p]
        commits = sorted({r["semindex_git_commit"] for r in runs})
        dirty = sorted(int(r["seed"]) for r in runs if r["semindex_git_dirty"] == "True")
        loads = sorted({r["workload"] for r in runs})
        knobs = sorted({r["redecide"] for r in runs})
        choices = sorted({r["choices"] for r in runs})
        seeds = sorted(int(r["seed"]) for r in runs)
        extra = [s for s in seeds if s not in wanted]
        table.append((p, len(runs), commits, dirty, loads, knobs, choices, seeds, extra))
        if commits and commits != [WANT_COMMIT]:
            bad.append(f"p {p}: the runs record semindex commit(s) {commits}, "
                       f"the step ran {WANT_COMMIT}")
        if dirty:
            bad.append(f"p {p}: runs made from a dirty tree, seeds {dirty}")
        if loads and loads != [WANT_WORKLOAD]:
            bad.append(f"p {p}: workload(s) {loads}, the step ran {WANT_WORKLOAD}")
        if knobs and knobs != ["True"]:
            bad.append(f"p {p}: state_poll_choice_redecide {knobs}, the step ran it on")
        if choices and choices != [WANT_CHOICES]:
            bad.append(f"p {p}: state_poll_interval_choices_s {choices}, the step ran "
                       f"{WANT_CHOICES}")
        if extra:
            bad.append(f"p {p}: seeds outside the set this run was asked for: {extra}")
    return table, bad


def main():
    # On stderr as well as in every written output: the settings stop below prints only to
    # stderr and writes no file, so without this line that path carries no label at all.
    if OVERRIDDEN:
        print(BANNER, file=sys.stderr)
    wanted = parse_seeds(SEEDS)
    matched = check_prediction(os.path.join(HERE, "predict.txt"))
    rows = load_perrun(os.path.join(OUT, "perrun.csv"))
    table, bad = settings_check(rows, wanted)
    if bad:
        for line in bad:
            print(line, file=sys.stderr)
        raise SystemExit("the runs in perrun.csv are not the runs spec.md describes; "
                         "nothing was written")

    out, L = [], []

    def rec(line, p, metric, value, note=""):
        out.append(dict(line=line, p=p, metric=metric, value=value, note=note))

    if OVERRIDDEN:
        L += [f"**{BANNER}**", ""]
    L += ["# evidence/21 results", "",
          "Computed by ana.py from perrun.csv and the frozen predict.txt of this "
          "directory. The pass lines, the bands, the tolerance formula and the verdict "
          "words are spec.md's; the words are read off the numbers mechanically and "
          "nothing here adds to them.", "",
          f"One stage at two p: `{STAGE}` p 0.2 and p 0.4, workload `{WANT_WORKLOAD}`, "
          f"seeds {SEEDS} in each cell, {RUNS_WANTED} runs, the choices mode "
          "(3.695 / 6.0 / 10.043 s, jitter 0.1) re-deciding its interval at every "
          "resolution. Decision rows are counted over the whole run, the polls per run "
          "over the window [t0+10 s, t0+130 s], and the delay over the records created at "
          "or before t0+72 s, which is what spec.md \"Quantities\" fixes for each.", "",
          f"The predicted constants carried in ana.py were checked against predict.txt at "
          f"run time: {len(matched)} rows of predicted numbers, every number in them equal "
          "to the digits that file prints. A mismatch stops the script, so the analysis "
          "cannot drift from the frozen prediction.", ""]
    rec(0, "", "predict_txt_rows_checked", len(matched))
    rec(0, "", "test_input", "yes" if OVERRIDDEN else "no", BANNER if OVERRIDDEN else "")

    # ---- line 3: all 200 runs valid ---------------------------------------------------
    L += [f"## Line 3: all {RUNS_WANTED} runs DONE and valid", "",
          "DONE means the run has a row in perrun.csv: perrun.py writes only runs that "
          "carry a DONE marker. Valid is `meta.json` -> `validity`, written by "
          "semindex.evalrun out of semindex.evalvalid. The flag names carry the node and "
          "the sequence number (`record_off_schedule:h0:seq5`), so the breakdown counts "
          "them by the first word of the name. It is printed whether or not any flag "
          "appeared.", "",
          "| p | runs expected | runs DONE | valid and unflagged | flagged | flag names |",
          "|---|---|---|---|---|---|"]
    cells, done_total, ok_total = {}, 0, 0
    flagged_seeds = {}
    for p in PS:
        runs = [r for r in rows if r["p"] == p]
        clean = [r["valid"] == "True" and r["flags"] == "none" for r in runs]
        ok = [r for r, c in zip(runs, clean) if c]
        fl = [r for r, c in zip(runs, clean) if not c]
        cells[p] = ok
        done_total += len(runs)
        ok_total += len(ok)
        # By the first word of the flag name, because the rest of it is the node and the
        # sequence number the flag was raised on and every run would make its own name.
        names = Counter(f.split(":")[0] for r in fl for f in r["flags"].split(";")
                        if r["flags"] != "none")
        flagged_seeds[p] = {}
        for r in fl:
            flagged_seeds[p].setdefault(r["flags"], []).append(int(r["seed"]))
        L.append(f"| {p} | {RUNS_PER_CELL} | {len(runs)} | {len(ok)} | {len(fl)} | "
                 + ("; ".join(f"{k} {v}" for k, v in sorted(names.items())) or "none")
                 + " |")
        rec(3, p, "runs_done", len(runs))
        rec(3, p, "runs_valid", len(ok))
        rec(3, p, "runs_flagged", len(fl))
        rec(3, p, "flag_names", ";".join(f"{k}={v}" for k, v in sorted(names.items()))
            or "none")
    if any(flagged_seeds[p] for p in PS):
        L += ["", "Every flagged run, by cell and by the flags it carries:", "",
              "| p | flags | runs | seeds |", "|---|---|---|---|"]
        for p in PS:
            for flags, seeds_f in sorted(flagged_seeds[p].items()):
                L.append(f"| {p} | {flags} | {len(seeds_f)} | "
                         + ",".join(str(s) for s in seeds_f) + " |")
                rec(3, p, f"flagged_seeds[{flags}]", ",".join(str(s) for s in seeds_f))
        L += [""]
    else:
        L += ["", "No run carries a validity flag, so there is no breakdown by cell and "
                  "by flag to print.", ""]
    L += [f"{done_total} of {RUNS_WANTED} runs DONE, {ok_total} of them valid and "
          "unflagged. Lines 4 and 5 below are counted on the valid and unflagged runs "
          "only: a flagged run is one evalvalid says did not run as specified, and "
          "counting its decisions among the rest would put rows from a run nobody can read "
          "into the share per arm. Leaving a run out also lowers N, which widens the "
          "tolerance on the share per arm, so a cell read on fewer runs is checked less "
          "tightly than spec.md's 100 runs would check it.", ""]
    rec(3, "", "runs_done_total", done_total)
    rec(3, "", "runs_valid_total", ok_total)

    L += ["What the runs themselves record about how they were made. These are spec.md's "
          "own settings, and a run recording anything else is not a run of this step: such "
          "a row stops this script before a single measured number is printed, because "
          "every count below would then be counted on a stage nobody specified. Fewer runs "
          "than the cell asks for does not stop anything and reads `incomplete` instead.",
          "",
          "| p | runs | semindex commit | dirty runs | workload | knob | arms | seeds | "
          "node logs per run |", "|---|---|---|---|---|---|---|---|---|"]
    logs_ok = True
    for (p, n_runs, commits, dirty, loads, knobs, choices, seeds, _extra) in table:
        runs = [r for r in rows if r["p"] == p]
        logs = sorted({int(r["node_logs"]) for r in runs})
        logs_ok = logs_ok and logs in ([], [NODES_PER_RUN])
        span = (f"{min(seeds)} to {max(seeds)}"
                if seeds and len(seeds) == max(seeds) - min(seeds) + 1
                else (",".join(str(s) for s in seeds) or "none"))
        L.append(f"| {p} | {n_runs} | " + (", ".join(f"`{c}`" for c in commits) or "none")
                 + f" | {len(dirty)} | " + (", ".join(f"`{x}`" for x in loads) or "none")
                 + " | " + (", ".join(knobs) or "none") + " | "
                 + (", ".join(f"`{c}`" for c in choices) or "none")
                 + f" | {span} | " + (", ".join(str(x) for x in logs) or "none") + " |")
        rec(3, p, "semindex_git_commit", ";".join(commits) or "none")
        rec(3, p, "workload", ";".join(loads) or "none")
        rec(3, p, "redecide", ";".join(knobs) or "none")
        rec(3, p, "interval_choices_s", ";".join(choices) or "none")
        rec(3, p, "seeds", span)
        rec(3, p, "node_logs_per_run", ";".join(str(x) for x in logs) or "none")
    L += ["", f"Every run must have written {NODES_PER_RUN} node logs, one per host: a run "
              "with fewer logged its decisions somewhere this analysis did not read, and "
              "its decision counts would be short with nothing else saying so. "
              f"{'Every run has ' + str(NODES_PER_RUN) + '.' if logs_ok else 'Not every run has.'}",
          ""]
    rec(3, "", "node_logs_ok", "yes" if logs_ok else "no")

    L += ["The two smoke runs of `s11smoke` seed 101, reported separately and never "
          "counted among the 200. spec.md gates the sweep on them, and seed 101 lies "
          "outside both this step's seeds and evidence/20's held-out seeds, so neither set "
          "is spent on it.", ""]
    if STAGE == "s11smoke":
        L += ["This run of the analysis reads `s11smoke` as its own stage, so the two "
              "smoke runs are already the table above and are not repeated here.", ""]
    else:
        L += ["| p | seed | valid | flags | knob | arms | semindex commit |",
              "|---|---|---|---|---|---|---|"]
        for p, m in smoke_runs():
            if m is None:
                L.append(f"| {p} | 101 | not present in the store | - | - | - | - |")
                rec(3, p, "smoke_run", "not present")
                continue
            L.append(f"| {p} | 101 | {m['valid']} | {m['flags']} | {m['redecide']} | "
                     f"`{m['choices']}` | `{m['commit']}` |")
            rec(3, p, "smoke_run_valid", str(m["valid"]), f"flags {m['flags']}")
        L += [""]

    # Completeness is read per cell and not off the total, because 101 runs in one cell and
    # 99 in the other sum to the 200 the step asked for while neither cell holds the runs
    # it asked for, and lines 4 and 5 would pool the surplus run into one cell's shares.
    done_per_cell = {p: len([r for r in rows if r["p"] == p]) for p in PS}
    complete = all(n == RUNS_PER_CELL for n in done_per_cell.values())
    word_incomplete = ("incomplete ("
                       + ", ".join(f"p {p} {done_per_cell[p]} of {RUNS_PER_CELL}"
                                   for p in PS) + " runs DONE)")
    if not complete:
        verdict3 = word_incomplete
    else:
        # The word names what produced it: a reader meeting `fail` beside a table showing
        # no flagged run would otherwise have to find the node-log sentence above to learn
        # that the node logs, and not evalvalid, are what failed.
        reasons = []
        if ok_total < done_total:
            reasons.append("runs carrying an evalvalid flag")
        if not logs_ok:
            reasons.append(f"runs with fewer than {NODES_PER_RUN} node logs")
        verdict3 = "fail (" + ", ".join(reasons) + ")" if reasons else "pass"
    # A run that is missing or not DONE is what withholds the word of lines 4 and 5: their
    # numbers are then counted on a stage that is still being written. A flagged run does
    # not withhold it. The flagged run is left out of the counts and lines 4 and 5 say on
    # how many runs each cell's numbers stand, which is the whole of what a flag costs
    # them; calling the lines `incomplete` over a finished stage would say instead that
    # the sweep had not ended.
    L += [f"Verdict on {done_total} runs: **{verdict3}**.", ""]
    if not complete:
        L += [f"A cell does not hold the {RUNS_PER_CELL} runs the step asked for, so lines "
              "4 and 5 below carry no pass or fail word. Their numbers are printed on the "
              "runs that are there, and they are not the stage's reading.", ""]
    elif verdict3 != "pass":
        L += ["Pass line 3 is not met. Every run is DONE, so lines 4 and 5 below do carry "
              "their word, counted on the valid and unflagged runs alone; the `runs` "
              "column of the first table of each line says how many runs that is.", ""]
    rec(3, "", "verdict", verdict3)

    # ---- line 4: the decision log a learner reads -------------------------------------
    L += ["## Line 4: the decision log is material a learner can read", "",
          "Counted over each cell's valid runs, over the whole run rather than over the "
          "measurement window: spec.md \"Quantities\" gives the window to the delay and "
          "the polls per run, and says that what this line asks is what a run decided. A "
          "send row is one `state_poll` decision, a reschedule row one "
          "`state_poll_reschedule`.", "",
          "### Decision rows per run", "",
          "| p | runs | send rows median (min..max) | predicted | reschedule rows median "
          "(min..max) | predicted |", "|---|---|---|---|---|---|"]
    for p in PS:
        runs = cells[p]
        sm, slo, shi = spread(runs, "send_rows")
        rm, rlo, rhi = spread(runs, "resched_rows")
        q = PRED_ROWS[p]
        L.append(f"| {p} | {len(runs)} | {fmt(sm, 1)} ({fmt(slo, 0)}..{fmt(shi, 0)}) | "
                 f"{fmt(q[0], 1)} ({fmt(q[1], 0)}..{fmt(q[2], 0)}) | "
                 f"{fmt(rm, 1)} ({fmt(rlo, 0)}..{fmt(rhi, 0)}) | "
                 f"{fmt(q[3], 1)} ({fmt(q[4], 0)}..{fmt(q[5], 0)}) |")
        for name, (med, lo, hi) in (("send_rows", (sm, slo, shi)),
                                    ("resched_rows", (rm, rlo, rhi))):
            rec(4, p, f"{name}_median", fmt(med, 1), f"min {fmt(lo, 0)} max {fmt(hi, 0)}")
    L += ["", "The predicted columns are predict.txt's and stand there for information: "
              "spec.md attaches no pass or fail to the decision counts. The replay resolves "
              "every poll it sends and writes rows past the instant the testbed's agent "
              "stops, so the measured reschedule counts are expected to sit slightly below "
              "the predicted ones.", "",
          "### The share per arm, which is the check that the uniform draw is uniform", "",
          "N is the cell's reschedule rows and the tolerance is "
          "`1/3 +- 3*sqrt((2/9)/N)` read at that measured N, which is the formula spec.md "
          "fixed before the run. The pass or fail word of this line is whether all three "
          "shares lie inside it, per cell.", "",
          "| p | N | tolerance | arm | rows | share | distance from 1/3 | inside | "
          "predicted share |", "|---|---|---|---|---|---|---|---|---|"]
    arms_word = {}
    for p in PS:
        runs = cells[p]
        n = total(runs, "resched_rows")
        tol = tolerance(n)
        inside_all = True
        for i, (arm, col) in enumerate(zip(ARMS, ARM_COLS)):
            k = total(runs, col)
            share = k / n if n else float("nan")
            dist = share - UNIFORM
            # None, not False, on a cell that counted no row: an unmeasured share is
            # neither inside the tolerance nor outside it, and `no` would read as a draw
            # that came out skewed.
            ok = abs(dist) <= tol if n else None
            inside_all = inside_all and bool(ok)
            seen = "-" if ok is None else ("yes" if ok else "no")
            L.append(f"| {p} | {int(n)} | {fmt(tol, 5)} | {arm} | {int(k)} | "
                     f"{fmt(share, 5)} | {signed(dist, 5)} | {seen} | "
                     f"{fmt(PRED_RES[p][4][i], 3)} |")
            rec(4, p, f"arm_share[{arm}]", fmt(share, 6),
                f"{int(k)} of {int(n)} rows, distance {signed(dist, 6)}, "
                f"tolerance {fmt(tol, 6)}, inside {seen}")
        # A cell with no reschedule row is `not measured` and not `fail`: the uniform draw
        # cannot be checked on no rows, and pass line 5 calls the same empty cell `not
        # measured` too, so one report does not give two words for one absence of data.
        arms_word[p] = ("not measured" if not n
                        else "pass" if inside_all else "fail")
        rec(4, p, "reschedule_rows", int(n))
        rec(4, p, "arm_tolerance", fmt(tol, 6), "3*sqrt((2/9)/N) at the measured N")
    L += ["", "### The resolutions, and the rows a learner would read them off", "",
          "| p | timeout share | predicted | clamped share | predicted | truncated send "
          "rows | truncated reschedule rows | reschedule rows per send row | predicted |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in PS:
        runs = cells[p]
        n = total(runs, "resched_rows")
        sends = total(runs, "send_rows")
        to_share = total(runs, "resched_timed_out") / n if n else float("nan")
        cl_share = total(runs, "resched_clamped") / n if n else float("nan")
        per_send = n / sends if sends else float("nan")
        q = PRED_RES[p]
        L.append(f"| {p} | {fmt(to_share, 4)} | {fmt(q[0], 3)} | {fmt(cl_share, 4)} | "
                 f"{fmt(q[3], 3)} | {int(total(runs, 'trunc_send'))} | "
                 f"{int(total(runs, 'trunc_resched'))} | {fmt(per_send, 4)} | "
                 f"{fmt(PRED_ROWS[p][6], 3)} |")
        rec(4, p, "timeout_share", fmt(to_share, 6), f"predicted {q[0]}")
        rec(4, p, "clamped_share", fmt(cl_share, 6), f"predicted {q[3]}")
        rec(4, p, "truncated_send_rows", int(total(runs, "trunc_send")))
        rec(4, p, "truncated_reschedule_rows", int(total(runs, "trunc_resched")))
        rec(4, p, "reschedule_rows_per_send_row", fmt(per_send, 6),
            f"predicted {PRED_ROWS[p][6]}, the replay's surplus past the stop instant "
            f"{PRED_ROWS[p][7]}")
    L += ["", "A truncated row is one whose state carried fewer origins than the run "
              "tracks, so a learner reading it cannot see every origin. With two origins "
              "none is expected; a nonzero count means the rows are missing origins. The "
              "count is reported and carries no pass or fail word of its own, which this "
              "line's word reads off the three shares alone.", "",
          "The reschedule rows per send row are predicted as 1.000 because the replay "
          "resolves every poll it sends. It also writes rows past t0+130 s, where the "
          "testbed's agent has already left its loop: predict.txt puts those at "
          f"{PRED_ROWS[0.2][7]:.3f} of the reschedule rows at p 0.2 and "
          f"{PRED_ROWS[0.4][7]:.3f} at p 0.4, and that is the size of the gap below 1.000 "
          "to expect here. spec.md records that the same surplus biases the predicted "
          "reschedule count and the predicted timeout share upward.", "",
          "The clamped share is predicted as zero for a structural reason: the shortest "
          "applied interval is 3.695 x 0.9 = 3.3255 s measured from the send, and a "
          "resolution lands at most 2.1 s after the send, so the re-decided instant is "
          "always still ahead. A nonzero clamped share here is therefore evidence that an "
          "agent's loop stalled for over a second, not evidence about the arms.", "",
          "### Also reported, with no pass or fail attached", "",
          "| p | polls checked against the schedule | of those outside the 0.5 s | "
          "answered reschedule rows whose `empty` is null | `state_poll_late_reply` rows | "
          "reschedule rows that are not a uniform draw |", "|---|---|---|---|---|---|"]
    for p in PS:
        runs = cells[p]
        # perrun.py's per-row column, not the sum of its two per-condition counters: a row
        # that is wrong in both ways is one row, and the sum would print it as two under a
        # heading that names rows.
        bad_draw = total(runs, "bad_draw_rows")
        L.append(f"| {p} | {int(total(runs, 'sched_checked'))} | "
                 f"{int(total(runs, 'sched_outside'))} | "
                 f"{int(total(runs, 'empty_null_answered'))} | "
                 f"{int(total(runs, 'late_reply_rows'))} | {int(bad_draw)} |")
        rec(4, p, "schedule_checked_polls", int(total(runs, "sched_checked")))
        rec(4, p, "schedule_outside_slack", int(total(runs, "sched_outside")),
            "window [applied - 0.002 s, applied + 0.5 s], the smoke gate's constants")
        rec(4, p, "empty_null_answered", int(total(runs, "empty_null_answered")))
        rec(4, p, "late_reply_rows", int(total(runs, "late_reply_rows")))
        rec(4, p, "rows_not_a_uniform_draw", int(bad_draw),
            f"action_prob other than {UNIFORM!r} or interval_s outside {list(ARMS)}, "
            f"counted once per row; action_prob is wrong on "
            f"{int(total(runs, 'bad_action_prob'))} rows and interval_s on "
            f"{int(total(runs, 'bad_interval'))}")
    L += ["", "The first two columns are the smoke gate's seventh check run over every "
              "run of the sweep, with its own constants: for a poll re-decided without a "
              "clamp whose neighbour was polled again, whether that next poll left inside "
              "`[applied - 0.002 s, applied + 0.5 s]` of the re-decided instant. A poll "
              "outside it is a poll whose next send does not match the re-decided instant, "
              "which is either a re-decision that was logged and not applied or an agent "
              "loop that stalled, and this count alone cannot tell the two apart. "
              "Unlike the smoke gate, which stops at the first late successor of a "
              "(node, neighbour) pair, every late successor is counted here, so the number "
              "is a count of polls and not of pairs and cannot be compared with a smoke "
              "gate count directly. spec.md attaches no pass or fail to it.", "",
          "A null `empty` on an answered row is a reply the node could not decode. A row "
          "carrying no `empty` field at all is counted in the same column and cannot be "
          "told apart from an explicit null here. The "
          "replay cannot produce one, so this count has nothing in the prediction to be "
          "held against; evidence/20 found none in 11881 answered re-decisions on its two "
          "rule stages. A `state_poll_late_reply` is a reply that arrived after its poll "
          "had already timed out, which the replay cannot produce either.", "",
          "The last column is an addition of this analysis and has no Acceptance item "
          "behind it: reschedule rows whose `action_prob` is not exactly 1/3 or whose "
          "`interval_s` is not one of the three arms. Expected zero. A nonzero count means "
          "the share per arm above was counted on rows that are not the uniform draw this "
          "step ran, and nothing else on this line would say so. It is counted once per "
          "row, so it stays a row count when a row is wrong in both ways at once.", "",
          "evidence/09's record-set sanity counters (`hop_gap_outside_window`, "
          "`reply_before_any_poll` and the rest) are printed on perrun.py's stderr only "
          "and are not read back here, so this file cannot show that the record set the "
          "delay above was counted on is the one the prediction scored. That has to be "
          "read off perrun.py's own output.", ""]
    if not complete:
        word4 = word_incomplete
    elif all(arms_word[p] == "pass" for p in PS):
        word4 = "pass"
    elif any(arms_word[p] == "fail" for p in PS):
        word4 = "fail"
    else:
        word4 = "not measured"
    for p in PS:
        rec(4, p, "arms_inside_tolerance", arms_word[p])
    per_cell = ", ".join(f"p {p} {arms_word[p]}" for p in PS)
    L += [f"Verdict, which reads the share per arm alone: **{word4}**"
          + (f" ({per_cell})." if complete else "."), ""]
    rec(4, "", "verdict", word4)

    # ---- line 5: did the replay predict the measurement? ------------------------------
    L += ["## Line 5: did the replay predict the measurement?", "",
          "The headline delay and the polls per run at both p, four values, each against "
          f"the plus or minus {BAND_PCT:.0f} % band of predict.txt. The headline delay is "
          "the pooled `sum(lat4)/sum(n4)` over the cell's runs, over the records created "
          "at or before t0+72 s; under `w0` that is every record the run holds. The polls "
          "per run is the mean over the cell's runs of evalagg's in-window count over the "
          "8 directed pairs; a reschedule row is not a poll and is never counted as one. "
          "All 4 inside reads `predicted`, 1 or 2 outside `mixed`, 3 or more outside "
          "`missed`.", "",
          "The band ends below are the digits predict.txt itself prints. They part from "
          "the printed median times 1 -+ 6 % by at most 0.008 (the upper end of the polls "
          "per run at p 0.4), which is under 0.006 % of that value, because that file "
          "forms the band from the unrounded median; inside or outside is decided on the "
          "printed ends, and a value on which the two would disagree is named below the "
          "table.", "",
          "| p | quantity | runs | measured | predicted | band | inside or outside | "
          "deviation % |", "|---|---|---|---|---|---|---|---|"]
    # Counted apart rather than as 4 minus the other: a cell with no run has a value that
    # is neither inside nor outside, and calling it outside would read as a measurement
    # that missed the band while calling it inside would read as one that hit it.
    n_inside, n_outside, outside_list, edge_list = 0, 0, [], []
    for p in PS:
        runs = cells[p]
        hn, hlat = total(runs, "hn4"), total(runs, "hlat4")
        delay = hlat / hn / 1000.0 if hn else float("nan")
        polls = total(runs, "polls_sent") / len(runs) if runs else float("nan")
        q = PRED5[p]
        for name, value, mid, lo, hi, nd in (("headline delay s", delay, q[0], q[1], q[2], 3),
                                             ("polls per run", polls, q[3], q[4], q[5], 2)):
            word = ("not measured" if value != value
                    else "inside" if lo <= value <= hi else "outside")
            n_inside += word == "inside"
            n_outside += word == "outside"
            # The other reading of spec.md's band, the printed median times 1 -+ 6 %. A
            # measured value lying between the two sets of ends would get its word from
            # the choice of reading and not from the measurement, so it is named.
            alt = mid * (1 - BAND_PCT / 100) <= value <= mid * (1 + BAND_PCT / 100)
            if word != "not measured" and alt != (word == "inside"):
                edge_list.append(f"p {p} {name} {fmt(value, 4)}")
            dev = 100 * (value / mid - 1) if mid else float("nan")
            if word == "outside":
                outside_list.append((p, name, value, mid, dev, lo, hi))
            L.append(f"| {p} | {name} | {len(runs)} | {fmt(value, nd)} | {fmt(mid, nd)} | "
                     f"{fmt(lo, nd)} .. {fmt(hi, nd)} | {word} | {signed(dev)} |")
            rec(5, p, f"{name}_measured", fmt(value, 6))
            rec(5, p, f"{name}_band", word, f"{lo} .. {hi}")
            rec(5, p, f"{name}_deviation_pct", fmt(dev, 4))
    # The word is read off the inside count and not off the outside count. A value a cell
    # produced no run for is neither, so an undefined value is reported on its own line
    # rather than letting `n_outside == 0` call a stage with two unmeasured values
    # `predicted`. An undefined value also outranks the outside count: with one value
    # missing, spec.md's split over four values is not the split being read.
    n_missing = 4 - n_inside - n_outside
    if not complete:
        key5, word5 = "incomplete", word_incomplete
    elif n_missing:
        key5 = "not measured"
        word5 = f"not measured ({n_missing} of 4 values have no run behind them)"
    elif n_outside == 0:
        key5 = word5 = "predicted"
    elif n_outside <= 2:
        key5 = word5 = "mixed"
    else:
        key5 = word5 = "missed"
    L += ["", f"{n_inside} of 4 values inside their band, {n_outside} outside, "
              f"{n_missing} not measured. Verdict: **{word5}**.", "",
          TEXT5["incomplete" if not complete else key5], ""]
    L += ["Values lying between the printed band ends and the printed median times "
          "1 -+ 6 %, which would read the other way on the second: "
          + ("; ".join(edge_list) if edge_list else "none, so the choice between the two "
             "decides nothing here") + ".", ""]
    rec(5, "", "values_between_the_two_band_readings", len(edge_list), "; ".join(edge_list))
    rec(5, "", "values_inside", n_inside)
    rec(5, "", "values_outside", n_outside)
    rec(5, "", "verdict", word5)
    L += ["Every value outside its band, with how far outside and in which direction. "
          "spec.md asks for this whatever the count is, so the list is printed even when "
          "it is empty.", ""]
    if outside_list:
        for p, name, value, mid, dev, lo, hi in outside_list:
            # Measured from the band end the inside-or-outside column was decided on, not
            # from the predicted value times 6 %. The two part in the third decimal, and a
            # value just past the printed end would otherwise be reported as a negative
            # number of points outside a band the table above calls it outside of.
            above = value > hi
            over = (value - hi if above else lo - value) / mid * 100
            L.append(f"- p {p} {name}: measured {value:.3f}, predicted {mid:.3f}, "
                     f"{dev:+.2f} % against the prediction, which is "
                     f"{over:.3f} points {'above' if above else 'below'} the band end "
                     "predict.txt prints.")
            rec(5, p, f"outside[{name}]", fmt(dev, 4), f"predicted {mid}")
        L += [""]
    else:
        L += ["- no value fell outside its band.", ""]
    L += ["The censored share beside each delay, which spec.md asks for. A record is "
          "censored when it was not stored 4 hops away inside the window, and the delay is "
          "the mean over the uncensored ones, so a censored record is left out of the "
          "delay instead of lengthening it. The headline share is the one that belongs to "
          "the delay above; the all-record share stands beside it, and under `w0` the two "
          "select the same records and are expected to be equal.", "",
          "| p | headline censored share | all-record censored share | predicted |",
          "|---|---|---|---|"]
    for p in PS:
        runs = cells[p]
        hn, hc = total(runs, "hn4"), total(runs, "hc4")
        an, ac = total(runs, "n4"), total(runs, "c4")
        hs = hc / (hn + hc) if (hn + hc) else float("nan")
        a_s = ac / (an + ac) if (an + ac) else float("nan")
        L.append(f"| {p} | {fmt(hs, 5)} | {fmt(a_s, 5)} | {fmt(PRED5[p][6], 3)} |")
        rec(5, p, "censored_share_headline", fmt(hs, 6))
        rec(5, p, "censored_share_all_records", fmt(a_s, 6))
    L += ["", "## What these numbers do not say", "",
          "- This stage's delay is not a quantity any policy is compared on. A uniform "
          "draw over three arms is not a policy; the prediction exists only to check that "
          "the replay and the testbed ran the same thing, and pass line 5 is that check "
          "and nothing more.", "",
          f"- The {BAND_PCT:.0f} % band has no measurement behind it for a uniform draw. "
          "It is the largest per-stage deviation evidence/13 measured on the default load, "
          "5.9 %, on four fixed arms and two policies, carried over here. Nothing says it "
          "is the right band for a uniformly drawn interval or for the re-decision branch, "
          "which is why every value outside it is reported with its size and its "
          "direction whatever the count is.", "",
          "- The re-decision branch of the replay has no measurement behind it anywhere "
          "else. Until this line is read, the replay cannot be said to reproduce this "
          "stage at all, and on `missed` that branch is the first suspect.", "",
          "- No stage here is run from two source versions, so nothing in this step shows "
          "on the testbed that the knob-off path is unchanged since evidence/20's commit. "
          "That rests on the merged change's own tests.", "",
          "- Nothing here says which decision maker is better, or whether a policy can be "
          "estimated off these rows. This stage draws uniformly and runs no policy, and "
          "this step stops at leaving the data.", "",
          "- Nothing about p 0.5, p 0.6, p 0, the no-outage cell, the on/off loads, or "
          "finer arms.", "",
          "- Every number in the prediction is simulated. A predicted band says what the "
          "replay expects the testbed to measure and is itself no measurement.", ""]

    with open(os.path.join(OUT, "results.csv"), "w", newline="", encoding="utf-8") as fh:
        if OVERRIDDEN:
            fh.write(f"# {BANNER}\n")
        w = csv.DictWriter(fh, ["line", "p", "metric", "value", "note"])
        w.writeheader()
        w.writerows(out)
    with open(os.path.join(OUT, "results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
