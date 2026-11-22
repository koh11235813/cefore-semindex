"""Per-run metrics for the s11 runs of spec.md: one CSV row per (p, seed) of one stage.

    AGG=<evalagg output> PYTHONPATH=<repo>/src \\
        /home/lab_shared/cefore-semindex/.venv/bin/python3 -B perrun.py

Writes `<OUT>/perrun.csv`. One process, no pool: the sweep this reads is a timing
measurement on the same machine, and a reader that keeps several cores busy changes what
it reads. ana.py then computes every pass line out of this CSV alone, so the store and the
node logs are read exactly once.

WHAT ONE ROW HOLDS

  - evidence/20's `work`, imported from ../20/perrun.py and not copied: evidence/09's
    all-record columns (`lat4`, `n4`, `c4`, the per-hop sums, the sanity counters) plus
    evidence/20's headline columns (`hlat4`, `hn4`, `hc4`, the schedule columns and
    `workload`). Pass line 5's delay is the headline one, `sum(hlat4)/sum(hn4)` over a
    cell, which is the quantity spec.md "Quantities" fixes.
  - the polls.csv counters of the run's 8 directed pairs, summed. `polls_sent` is the
    in-window poll count of spec.md's "Polls per run" and the second quantity pass line 5
    reads.
  - the decision-row counts of pass line 4, counted here over the whole run from the five
    `events.jsonl` logs, not over the measurement window: spec.md "Quantities" says the
    window belongs to the delay and the polls per run, and that what pass line 4 asks is
    what a run decided. The field names are the source's own
    (`src/semindex/engine.py::_redecide_next_poll` and `tracker.decision_state`):
    `action.trigger`, `action.clamped`, `action.empty`, `action.interval_s`,
    `action.applied_interval_s`, `action.poll_epoch_ms`, `state.truncated`, and
    `action_prob` beside `action` rather than inside it.
  - what the run itself records about how it was made: validity, the semindex commit and
    its dirty flag, the workload, the re-decision knob and the three arms. ana.py's pass
    line 3 reads these off the CSV, so it needs no second pass over the store.

A run without a DONE marker is left out, as in evidence/20: `evalagg.discover_runs`
returns a seed directory whether or not the run inside it finished, and an unfinished one
has no meta.json. The per-stage line on stderr says how many directories exist and how
many are DONE, so a partial stage is visible in this script's own output; that matters
here because the sweep takes nine hours and this script is meant to be runnable before it
ends.

ENVIRONMENT

  AGG     (required) evalagg output, with `<AGG>/<STAGE>/discovery.csv` and `polls.csv`.
  STORE   default /home/lab_shared/semindex-eval.
  STAGE   default s11a, the stage spec.md names.
  SEEDS   default 1-100. Only the banner and ana.py read it; every DONE run of the stage
          is written out whatever its seed, so that a seed the step did not ask for is
          visible to ana.py instead of being filtered away here.
  OUT     default this directory.

Whenever STORE, STAGE, SEEDS or OUT differs from its default, the first line of
perrun.csv is a `#` comment saying that the file is a test input and not the stage's
reading. ana.py skips `#` lines when it reads the CSV and repeats the sentence in its own
output. The banner exists because every test of these two scripts runs them on s11smoke or
on a fake store, and a number out of such a run that reached a reader unlabelled would be
read as the step's result.

Needs PYTHONPATH=<repo>/src for semindex; ../05, ../07 and ../09 are put on the path from
this file's own location, so the script does not have to be run from this directory.
"""
import csv
import glob
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# evidence/05 holds attrib and run_all, evidence/07 what they import in turn; evidence/20's
# perrun.py puts ../09 on the path itself when it is loaded below. Absolute paths rather
# than the relative PYTHONPATH entries evidence/09, 10, 16, 17, 19 and 20 are run with, so
# that this script works from any working directory.
for _d in ("05", "07"):
    sys.path.insert(0, os.path.join(HERE, os.pardir, _d))
import run_all  # noqa: E402
from semindex import evalagg  # noqa: E402

STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")
STAGE = os.environ.get("STAGE", "s11a")
SEEDS = os.environ.get("SEEDS", "1-100")
OUT = os.environ.get("OUT", HERE)
AGG = os.environ["AGG"]
OVERRIDDEN = (STORE != "/home/lab_shared/semindex-eval" or STAGE != "s11a"
              or SEEDS != "1-100" or os.path.abspath(OUT) != HERE)
# No comma anywhere in the banner: it is written as the first line of perrun.csv, and a
# reader that does not skip `#` lines (csv.DictReader, pandas.read_csv) would otherwise
# split it into fields and take it for the header row, losing every column name.
BANNER = (f"test input: stage {STAGE}; seeds {SEEDS}; store {STORE}; output {OUT}. "
          "These numbers are a test of the script and not the reading of stage s11a.")

# evidence/20's perrun.py is loaded under a name of its own rather than imported: this file
# is also called perrun.py, and `import perrun` from this directory would bind the name to
# this module instead. Under another name, evidence/20's own `import perrun as P9` still
# resolves to evidence/09's, because evidence/20 puts ../09 at the head of sys.path first.
# AGG is read at its module level, so it must already be in the environment, which it is.
_spec = importlib.util.spec_from_file_location(
    "perrun20", os.path.join(HERE, os.pardir, "20", "perrun.py"))
P20 = importlib.util.module_from_spec(_spec)
sys.modules["perrun20"] = P20
_spec.loader.exec_module(P20)

# The three arms of spec.md "Runs", as `--interval-choices` passes them and as the engine
# logs them: the logged `interval_s` is the nominal choice, the jitter is applied after it.
# Compared as exact doubles, which is what a share per arm has to be counted on.
ARMS = (3.695, 6.0, 10.043)
ARM_COL = {3.695: "arm_3695", 6.0: "arm_6000", 10.043: "arm_10043"}
ACTION_PROB = 1.0 / len(ARMS)
# smoke.py's two constants, unchanged: spec.md pass line 4 asks for the smoke gate's
# schedule check to be run again over the whole sweep with the same constants. SLACK_S is
# how late after the re-decided instant the next poll may leave, STAMP_S how early a poll
# that left exactly on time can read once both millisecond stamps are rounded.
SLACK_S = 0.5
STAMP_S = 0.002
# The counts this file adds to evidence/20's row. Listed here so that a run whose logs
# yielded nothing still carries every column.
COUNT_COLS = ("node_logs", "send_rows", "resched_rows", "resched_answered",
              "resched_timed_out", "resched_trigger_other", "resched_clamped", "arm_3695", "arm_6000", "arm_10043",
              "arm_other", "trunc_send", "trunc_resched", "empty_null_answered",
              "late_reply_rows", "sched_checked", "sched_outside", "bad_action_prob",
              "bad_interval", "bad_draw_rows")


def decision_counts(run_dir):
    """The pass line 4 counts of one run, over its five node logs and its whole duration.

    One pass per log. Every count is a column of perrun.csv, so ana.py can read a cell's
    pooled count as a sum and a per-run median off the same rows.

    `sched_checked` and `sched_outside` are the smoke gate's seventh check run over a whole
    run: for every poll that was re-decided without a clamp and whose neighbour was polled
    again afterwards, whether that next poll left inside `[applied - STAMP_S,
    applied + SLACK_S]` of the re-decided instant. A clamped row moved the instant to `now`
    and says nothing about the applied interval, and the last poll of a pair has no
    successor, so neither is checked. Unlike smoke.py this counts every successor rather
    than stopping at the first one that is late, because pass line 4 reports how many polls
    left outside and not which pairs did.

    Only `*/semindex/events.jsonl` is opened under `debug/node_dirs`: the rest of that tree
    holds the hosts' Cefore key material.
    """
    x = dict.fromkeys(COUNT_COLS, 0)
    pattern = os.path.join(run_dir, "debug", "node_dirs", "h*", "semindex", "events.jsonl")
    for path in sorted(glob.glob(pattern)):
        x["node_logs"] += 1
        # Per node, because a poll and its successor are sends by the same agent; two
        # nodes polling the same neighbour name are two separate schedules.
        sends, applied = {}, {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                e = json.loads(line)
                name = e.get("event")
                if name == "state_poll_late_reply":
                    # A reply to a poll already resolved as timed out. Its own event, not a
                    # decision row: the schedule the timeout set stands and nothing is
                    # drawn. The replay cannot produce one at all (spec.md, difference 2).
                    x["late_reply_rows"] += 1
                    continue
                if name != "decision":
                    continue
                kind = e.get("kind")
                if kind == "state_poll":
                    x["send_rows"] += 1
                    x["trunc_send"] += e["state"].get("truncated") is True
                    sends.setdefault(e["action"]["neighbor"], []).append(e["ts_ms"])
                elif kind == "state_poll_reschedule":
                    a = e["action"]
                    x["resched_rows"] += 1
                    x["trunc_resched"] += e["state"].get("truncated") is True
                    x["resched_clamped"] += bool(a["clamped"])
                    x[ARM_COL.get(a["interval_s"], "arm_other")] += 1
                    # The analysis's own integrity count, with no pass line behind it: a
                    # row that is not a uniform draw over the three arms is a row the whole
                    # of pass line 4 is counted on wrongly, and nothing else below would
                    # say so.
                    bad_prob = e.get("action_prob") != ACTION_PROB
                    bad_arm = a["interval_s"] not in ARMS
                    x["bad_action_prob"] += bad_prob
                    x["bad_interval"] += bad_arm
                    # Counted once per row beside the two counters above, because a row
                    # that is wrong in both ways is still one row: ana.py prints this
                    # column under a heading that names rows, and the sum of the two
                    # counters would print such a row twice there.
                    x["bad_draw_rows"] += bad_prob or bad_arm
                    # The engine resolves a poll as exactly "answered" or "timed_out"
                    # (`engine.py::_redecide_next_poll` has those two callers and no
                    # other). spec.md defines the timeout share on the rows whose trigger
                    # is `timed_out`, so a third value gets a column of its own instead of
                    # being folded into the timeouts, and main() stops on it.
                    if a["trigger"] == "answered":
                        x["resched_answered"] += 1
                        # `empty` is what the reply advertised. Null on an answer means the
                        # reply could not be decoded, which the replay cannot produce
                        # (spec.md, difference 6), so the count has no prediction to be
                        # held against and is reported on its own. A row carrying no
                        # `empty` field at all is counted here too and cannot be told from
                        # an explicit null; ana.py says so where it prints the count.
                        x["empty_null_answered"] += a.get("empty") is None
                    elif a["trigger"] == "timed_out":
                        x["resched_timed_out"] += 1
                    else:
                        x["resched_trigger_other"] += 1
                    if not a["clamped"]:
                        applied[(a["neighbor"], a["poll_epoch_ms"])] = a["applied_interval_s"]
        for nb, stamps in sends.items():
            for sent, nxt in zip(stamps, stamps[1:]):
                gap = applied.get((nb, sent))
                if gap is None:
                    continue
                x["sched_checked"] += 1
                x["sched_outside"] += not -STAMP_S <= (nxt - sent) / 1000.0 - gap <= SLACK_S
    return x


def meta_columns(run_dir):
    """What the run records about how it was made, for pass line 3 to read off the CSV.

    The knob and the arms are what say the run being scored is a run the re-decision was on
    in: spec.md records that runs.csv has no column for the knob, so meta.json is the only
    place a run carries it.
    """
    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as fh:
        m = json.load(fh)
    v = m.get("validity") or {}
    return {
        "valid": bool(v.get("valid")),
        "flags": ";".join(v.get("flags") or []) or "none",
        "semindex_git_commit": m.get("semindex_git_commit"),
        "semindex_git_dirty": bool(m.get("semindex_git_dirty")),
        "redecide": m.get("state_poll_choice_redecide"),
        "choices": ";".join(repr(c) for c in m.get("state_poll_interval_choices_s") or []),
    }


def work(job):
    """One run's row: evidence/20's columns, then this step's counts and meta columns."""
    _stage, _p, _seed, run_dir = job
    row = P20.work(job)
    row.update(decision_counts(run_dir))
    row.update(meta_columns(run_dir))
    return row


def main():
    # A stage with no evalagg output at all is the normal state of a store that does not
    # hold the stage yet, and the "no DONE run" message below is the readable form of it.
    # Opening discovery.csv first would end the script in a traceback instead.
    disc = os.path.join(AGG, STAGE, "discovery.csv")
    if os.path.exists(disc):
        run_all.load_disc(STAGE, AGG)
    if OVERRIDDEN:
        print(BANNER, file=sys.stderr)
    refs = evalagg.discover_runs(STORE, STAGE)
    done = [r for r in refs if evalagg.run_has_done_marker(r.run_dir)]
    print(STAGE, f"dirs={len(refs)}", f"done={len(done)}", file=sys.stderr, flush=True)
    if not done:
        print(f"no DONE run under {STORE}/{STAGE}; perrun.csv not written", file=sys.stderr)
        raise SystemExit(1)
    # Checked before the attrib pass below rather than at the open further down: a stage
    # with DONE runs but no evalagg output would otherwise end in a FileNotFoundError
    # traceback after the whole pass over the store has been paid for.
    polls_csv = os.path.join(AGG, STAGE, "polls.csv")
    if not os.path.exists(polls_csv):
        print(f"no {polls_csv}; run semindex.evalagg over {STAGE} first", file=sys.stderr)
        raise SystemExit(1)
    rows = [work((STAGE, r.p, r.seed, r.run_dir)) for r in done]
    rows.sort(key=lambda r: (r["p"], r["seed"]))
    for r in rows:
        r.update(dict.fromkeys(P20.POLL_COLS, 0))
    idx = {(r["p"], r["seed"]): r for r in rows}
    with open(polls_csv, encoding="utf-8") as fh:
        for q in csv.DictReader(fh):
            r = idx.get((float(q["p"]), int(q["seed"])))
            if r is None:
                continue
            r["polls_sent"] += int(q["polls_sent"])
            r["state_rx"] += int(q["state_rx"])
            r["resched_ans"] += int(q["reschedule_answered"])
            r["resched_to"] += int(q["reschedule_timed_out"])
            r["clamped"] += int(q["reschedule_clamped"])
            r["late"] += int(q["late_replies"])
            r["n_pollrows"] += 1
    # evidence/20's two stops, both of which would otherwise leave a run's delay right and
    # its polls per run wrong, which is a silent corruption of pass line 5. An AGG built
    # before a run finished has no discovery.csv rows for it, and attrib then skips that
    # run's pair sums instead of failing; a DONE run whose polls.csv rows are missing would
    # be scored with polls_sent 0. Losing the race against the sweep is the expected way to
    # reach either, and the fix for both is to rebuild AGG and rerun.
    stale = [(r["p"], r["seed"]) for r in rows if r["runs_without_discovery_rows"]]
    if stale:
        print("AGG is older than the store; rebuild it for these runs:", stale, file=sys.stderr)
        raise SystemExit(1)
    wrong = [(r["p"], r["seed"], r["n_pollrows"]) for r in rows
             if r["n_pollrows"] != P20.POLLROWS_PER_RUN]
    if wrong:
        print(f"these DONE runs folded a number of polls.csv rows other than "
              f"{P20.POLLROWS_PER_RUN} (p, seed, rows folded); rebuild AGG for them:",
              wrong, file=sys.stderr)
        raise SystemExit(1)
    # A trigger this analysis does not know is a resolution spec.md's timeout share has no
    # place for. Writing the CSV would let ana.py print a share whose two parts no longer
    # sum to the rows, so the script stops and the rows are looked at first.
    odd = [(r["p"], r["seed"], r["resched_trigger_other"]) for r in rows
           if r["resched_trigger_other"]]
    if odd:
        print("reschedule rows with a trigger other than answered or timed_out "
              "(p, seed, rows); perrun.csv not written:", odd, file=sys.stderr)
        raise SystemExit(1)
    path = os.path.join(OUT, "perrun.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        if OVERRIDDEN:
            fh.write(f"# {BANNER}\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, len(rows), file=sys.stderr)
    # The sanity counters of evidence/09, and the three columns evidence/20 added. All of
    # them are expected to be 0; a nonzero one says the rows above rest on a record set
    # that is not the one the prediction scored, and belongs in the analysis's reading.
    parts = [f"{c}={max(r[c] for r in rows) if c == 'exact_maxerr' else sum(r[c] for r in rows)}"
             for c in P20.P9.CHK]
    print(STAGE, f"runs={len(rows)}", *parts, file=sys.stderr)
    for c in ("sched_regen_diff", "head_missing_disc", "all_missing_disc", "node_logs",
              "bad_action_prob", "bad_interval", "bad_draw_rows", "arm_other"):
        print(f"{c} over all runs:", sum(r[c] for r in rows), file=sys.stderr)


if __name__ == "__main__":
    main()
