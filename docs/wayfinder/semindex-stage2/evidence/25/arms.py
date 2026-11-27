#!/usr/bin/env python3
"""Check that every decided interval is one of the configured arms, and, where an
independent record exists, that the logged interval is what the decider actually returned.

The first half answers "did anything leak into the schedule that is not one of the three
choices the decider was built around" straight from the run's own log, with no dependency
on a recording wrapper. The second half answers a different question -- "is the logged
interval really what the decider returned, or did something between the decider and the
log change it" -- and that one needs the independent record, because comparing the log
with itself would only ever confirm that the log agrees with the log.
"""
import sys
# 2026-11-27: no checkout imports needed here (plain stdlib json/argparse over the log
# and record files), but setting this before anything else keeps the same guarantee as
# parity.py's sibling script: this run leaves no __pycache__ for someone else to clean up.
sys.dont_write_bytecode = True

import argparse    # noqa: E402
import json        # noqa: E402
import os          # noqa: E402


def load_jsonl(path):
    """Parse one newline-delimited JSON file into a list of dicts, in file order."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_nodes(run_dir):
    """List of hN names that have an events.jsonl under RUN/debug/node_dirs."""
    base = os.path.join(run_dir, "debug", "node_dirs")
    nodes = []
    for name in sorted(os.listdir(base)):
        if os.path.isfile(os.path.join(base, name, "semindex", "events.jsonl")):
            nodes.append(name)
    return nodes


def decision_rows(rows, kind):
    """The decision rows of one kind, in file order.

    The two kinds are kept apart because they mean different things. A
    `state_poll_reschedule` row is a decision: its `interval_s` is what was chosen when a
    poll resolved, and it is the only kind that is compared against what the decider
    returned. A `state_poll` send row's `interval_s` is the value the pair already held
    going into the send. It is still checked for membership, because a stage with one
    fixed arm and no re-decision writes send rows only, and a check that read reschedule
    rows alone would pass such a stage without having looked at a single interval.
    """
    return [e for e in rows if e.get("event") == "decision" and e.get("kind") == kind]


def returned_index(rec_rows):
    """One pass over a node's record file: ({(neighbor, poll_epoch_ms): returned value},
    number of deferred resolutions whose answer was never released).

    An immediate answer is a `decide` line's own `returned`; a deferred one has
    `returned: null` on the `decide` line and the real value turns up later on a
    `collect` line carrying the same token, which is how the recording wrapper marks a
    deferred answer as released to the engine. Built once, so neither `decide` nor
    `collect` lines are rescanned per logged row. A deferred resolution with no `collect`
    line gets no entry at all: the engine never had its answer, so there is nothing a
    logged row could be compared with, and it is counted instead.
    """
    index = {}
    deferred_tokens = {}
    for e in rec_rows:
        if e.get("rec") == "decide":
            where = e.get("obs", {}).get("where", {})
            if where.get("kind") != "state_poll_reschedule":
                continue
            resolved = e.get("obs", {}).get("resolved") or {}
            key = (where.get("neighbor"), resolved.get("poll_epoch_ms"))
            if e.get("returned") is not None:
                index[key] = e["returned"]
            else:
                deferred_tokens[tuple(e.get("token") or ())] = key
        elif e.get("rec") == "collect":
            key = deferred_tokens.pop(tuple(e.get("token") or ()), None)
            if key is not None:
                index[key] = e["interval"]
    return index, len(deferred_tokens)


def check_choices(node, rows, choices):
    """Membership check: every logged interval_s must equal one of --choices exactly.

    Returns (failures, counts, late_count, clamped_count).
    """
    failures = []
    counts = {}
    late_count = 0
    clamped_count = 0
    for e in rows:
        value = e["action"]["interval_s"]
        if value in choices:
            counts[value] = counts.get(value, 0) + 1
        else:
            failures.append(
                "FAIL node=%s neighbor=%s kind=%s ts_ms=%s interval_s=%s not in choices"
                % (node, e["action"].get("neighbor"), e.get("kind"), e.get("ts_ms"), repr(value)))
        if e["action"].get("late"):
            late_count += 1
        if e["action"].get("clamped"):
            clamped_count += 1
    return failures, counts, late_count, clamped_count


def check_returned(node, rows, record_dir, decider_counts):
    """Join each logged reschedule row to the independent record and compare exactly."""
    record_path = os.path.join(record_dir, node + ".jsonl")
    if not os.path.isfile(record_path):
        return ["FAIL node=%s neighbor=- poll_epoch_ms=- record file missing" % node]

    rec_rows = load_jsonl(record_path)
    index, never_released = returned_index(rec_rows)

    failures = []
    matched_keys = set()
    for e in rows:
        action = e["action"]
        key = (action.get("neighbor"), action.get("poll_epoch_ms"))
        if key not in index:
            failures.append(
                "FAIL node=%s neighbor=%s poll_epoch_ms=%s logged row has no recorded answer"
                % (node, key[0], key[1]))
            continue
        matched_keys.add(key)
        value = index[key]
        if action["interval_s"] != value:
            failures.append(
                "FAIL node=%s neighbor=%s poll_epoch_ms=%s interval_s=%s returned=%s mismatch"
                % (node, key[0], key[1], repr(action["interval_s"]), repr(value)))

    # The other direction, in two parts that must not be mixed. An answer the decider
    # did hand over (returned at once, or released later) always produces a logged row,
    # so one with no row is an answer the engine lost: nothing excuses it. A deferred
    # resolution that was never released has no answer to lose, and the engine counts
    # exactly those as `unanswered`, so the two numbers must be equal.
    for key in sorted(k for k in index if k not in matched_keys):
        failures.append(
            "FAIL node=%s neighbor=%s poll_epoch_ms=%s recorded answer has no logged row"
            % (node, key[0], key[1]))
    unanswered = decider_counts.get("unanswered", 0) if decider_counts else 0
    if never_released != unanswered:
        failures.append(
            "FAIL node=%s deferred resolutions never released=%d but "
            "decider_counts.unanswered=%d" % (node, never_released, unanswered))
    elif never_released:
        failures.append(
            "INFO node=%s deferred resolutions never released=%d equals "
            "decider_counts.unanswered" % (node, never_released))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--choices", required=True, nargs="+", type=float)
    parser.add_argument("--record-dir")
    args = parser.parse_args()

    choices = set(args.choices)
    nodes = find_nodes(args.run)
    if not nodes:
        print("no nodes with events.jsonl found under %s" % args.run)
        return 2

    all_failures = []
    for node in nodes:
        events_path = os.path.join(args.run, "debug", "node_dirs", node, "semindex", "events.jsonl")
        rows = load_jsonl(events_path)
        resched = decision_rows(rows, "state_poll_reschedule")
        sends = decision_rows(rows, "state_poll")
        decider_counts = next((e for e in rows if e.get("event") == "decider_counts"), None)

        if not resched and not sends:
            # A node with no interval at all is a node the check did not look at.
            all_failures.append("FAIL node=%s no state_poll or state_poll_reschedule row" % node)
        for kind, kind_rows in (("state_poll_reschedule", resched), ("state_poll", sends)):
            failures, counts, late_count, clamped_count = check_choices(node, kind_rows, choices)
            all_failures.extend(failures)
            print("INFO node=%s kind=%s rows=%d late=%d clamped=%d"
                  % (node, kind, len(kind_rows), late_count, clamped_count))
            for value in sorted(counts):
                print("INFO node=%s kind=%s interval_s=%s count=%d"
                      % (node, kind, repr(value), counts[value]))

        if args.record_dir:
            all_failures.extend(check_returned(node, resched, args.record_dir, decider_counts))

    if not args.record_dir:
        print("NOT CHECKED: no --record-dir given, so the returned-value half was skipped; "
              "the log is the only record of what was returned here, and comparing it with "
              "itself would be a tautology, not a check.")

    for line in all_failures:
        print(line)

    hard_failures = [f for f in all_failures if f.startswith("FAIL")]
    if not hard_failures:
        print("ARMS PASS")
        return 0
    print("ARMS FAIL failures=%d" % len(hard_failures))
    return 1


if __name__ == "__main__":
    sys.exit(main())
