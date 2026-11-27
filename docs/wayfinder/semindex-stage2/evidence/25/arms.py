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
    """List of all node-directory names under RUN/debug/node_dirs.

    Every directory here is a node the run created. A node directory without
    events.jsonl is not "no data to check", it is the thing this check exists to
    catch: a node the run silently never logged. A missing events.jsonl is reported
    as a FAIL rather than this function quietly leaving the node out.
    """
    base = os.path.join(run_dir, "debug", "node_dirs")
    # 2026-11-27: filter to directories only. node_dirs holds nothing but node
    # directories on every stage this check has ever run against, but a plain file
    # dropped next to them (a README, a tarball) is not a node, and without this
    # filter it would be treated as one and fail with a phantom missing-events.jsonl
    # FAIL.
    return sorted(e for e in os.listdir(base) if os.path.isdir(os.path.join(base, e)))


def decision_rows(rows, kind):
    """The decision rows of one kind, in file order.

    The two kinds are kept apart because they mean different things. A
    `state_poll_reschedule` row is a decision taken when a poll resolved; a `state_poll`
    send row is a decision taken when a send went out. Both are compared against what
    the decider actually returned (see `check_returned`), on different join keys -- a
    reschedule row on (neighbor, poll_epoch_ms), a send row on (neighbor, ts_ms), since
    a send is never deferred and carries no poll_epoch_ms of its own. Checking both
    matters for membership too: a stage with one fixed arm and no re-decision writes
    send rows only, and a check that read reschedule rows alone would pass such a stage
    without having looked at a single interval.
    """
    return [e for e in rows if e.get("event") == "decision" and e.get("kind") == kind]


def returned_index(rec_rows):
    """One pass over a node's record file: ({(neighbor, poll_epoch_ms): returned value},
    number of deferred resolutions whose answer was never released,
    number of `collect` lines that released a token no longer outstanding).

    An immediate answer is a `decide` line's own `returned`; a deferred one has
    `returned: null` on the `decide` line and the real value turns up later on a
    `collect` line carrying the same token, which is how the recording wrapper marks a
    deferred answer as released to the engine. Built once, so neither `decide` nor
    `collect` lines are rescanned per logged row. A deferred resolution with no `collect`
    line gets no entry at all: the engine never had its answer, so there is nothing a
    logged row could be compared with, and it is counted instead. A `collect` whose
    token matches nothing still outstanding -- a second release of the same token, or a
    token that was never asked -- is an orphan the engine itself counts separately
    (`decider_counts.orphan`), so it is counted here too rather than silently dropped.
    """
    index = {}
    deferred_tokens = {}
    orphan_collects = 0
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
            token = tuple(e.get("token") or ())
            key = deferred_tokens.pop(token, None)
            if key is not None:
                index[key] = e["interval"]
            else:
                orphan_collects += 1
    return index, len(deferred_tokens), orphan_collects


def send_returned_index(rec_rows):
    """One pass over a node's record file: {(neighbor, ts_ms): [returned values]} for
    every `decide` line whose `obs.where.kind` is `state_poll` -- a send, not a
    resolution.

    A send is never deferred: nothing later releases it, so its own `returned` is the
    whole answer and there is no `collect` line to also scan here, unlike
    `returned_index`. Values are kept as a list, in file order, rather than one dict
    entry per key: a (neighbor, ts_ms) key that the record names more than once (a
    duplicated recorded line, or two genuinely distinct sends colliding on it) must
    still be matched one-for-one against the logged rows with that key below, not
    have an earlier value silently overwritten by a later one.
    """
    index = {}
    for e in rec_rows:
        if e.get("rec") != "decide":
            continue
        where = e.get("obs", {}).get("where", {})
        if where.get("kind") != "state_poll":
            continue
        index.setdefault((where.get("neighbor"), where.get("ts_ms")), []).append(
            e.get("returned"))
    return index


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


def check_jitter(node, rows, kind, jitter):
    """Every row's applied_interval_s must lie in interval_s's jitter band.

    `jitter` is the arm's configured relative jitter (e.g. 0.1 for +/-10%), so a value
    outside [interval_s*(1-jitter), interval_s*(1+jitter)] came from something other
    than that arm's own jitter -- a wrong clamp, a wrong arm, a unit mix-up -- and is a
    FAIL whatever else about the row is fine. A relative tolerance of 1e-9 absorbs
    float rounding in `interval_s * (1 +/- jitter)` without widening the band enough to
    hide a real miss.
    """
    failures = []
    for e in rows:
        action = e["action"]
        interval_s = action["interval_s"]
        applied = action["applied_interval_s"]
        lo = interval_s * (1 - jitter)
        hi = interval_s * (1 + jitter)
        tol = 1e-9 * max(abs(lo), abs(hi), abs(applied), 1.0)
        if not (lo - tol <= applied <= hi + tol):
            failures.append(
                "FAIL node=%s neighbor=%s kind=%s ts_ms=%s interval_s=%s applied_interval_s=%s "
                "outside jitter band"
                % (node, action.get("neighbor"), kind, e.get("ts_ms"), repr(interval_s),
                   repr(applied)))
    return failures


def check_returned(node, resched_rows, send_rows, record_dir, decider_counts):
    """Join each logged reschedule row, and each logged send row, to the independent
    record and compare exactly: a reschedule row's `interval_s` against the value
    eventually released for that resolution (at once, or later via `collect`), and a
    send row's `interval_s` against the value returned immediately for that send.
    """
    record_path = os.path.join(record_dir, node + ".jsonl")
    if not os.path.isfile(record_path):
        return ["FAIL node=%s neighbor=- poll_epoch_ms=- record file missing" % node]

    rec_rows = load_jsonl(record_path)
    index, never_released, orphan_collects = returned_index(rec_rows)
    send_index = send_returned_index(rec_rows)

    failures = []
    matched_keys = set()
    for e in resched_rows:
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

    orphan = decider_counts.get("orphan", 0) if decider_counts else 0
    if orphan_collects != orphan:
        failures.append(
            "FAIL node=%s orphan collects=%d but decider_counts.orphan=%d"
            % (node, orphan_collects, orphan))

    # Each key's recorded values are consumed one at a time, in file order, so a key
    # the record names more than once is matched against the logged rows with that
    # key one-for-one instead of every logged row checking against the same single
    # dict entry and a duplicate on either side going unnoticed.
    for e in send_rows:
        action = e["action"]
        key = (action.get("neighbor"), e.get("ts_ms"))
        values = send_index.get(key)
        if not values:
            failures.append(
                "FAIL node=%s neighbor=%s ts_ms=%s logged send row has no recorded answer"
                % (node, key[0], key[1]))
            continue
        value = values.pop(0)
        if action["interval_s"] != value:
            failures.append(
                "FAIL node=%s neighbor=%s ts_ms=%s interval_s=%s returned=%s mismatch"
                % (node, key[0], key[1], repr(action["interval_s"]), repr(value)))

    for key in sorted(k for k, v in send_index.items() if v):
        for _ in send_index[key]:
            failures.append(
                "FAIL node=%s neighbor=%s ts_ms=%s recorded send answer has no logged row"
                % (node, key[0], key[1]))

    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--choices", required=True, nargs="+", type=float)
    parser.add_argument("--record-dir")
    parser.add_argument("--jitter", type=float,
                         help="arm's configured relative jitter (e.g. 0.1); when given, "
                              "every row's applied_interval_s is checked against it")
    args = parser.parse_args()

    choices = set(args.choices)
    nodes = find_nodes(args.run)
    if not nodes:
        print("no node directories found under %s" % args.run)
        return 2

    all_failures = []
    for node in nodes:
        events_path = os.path.join(args.run, "debug", "node_dirs", node, "semindex", "events.jsonl")
        if not os.path.isfile(events_path):
            # A node directory with no events.jsonl is a node whose process never
            # wrote a single decision, not a node this check had nothing to look at.
            all_failures.append("FAIL node=%s no events.jsonl" % node)
            continue
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
            print("INFO node=%s kind=%s rows=%d late=%d row_clamped=%d"
                  % (node, kind, len(kind_rows), late_count, clamped_count))
            for value in sorted(counts):
                print("INFO node=%s kind=%s interval_s=%s count=%d"
                      % (node, kind, repr(value), counts[value]))
            if args.jitter is not None:
                all_failures.extend(check_jitter(node, kind_rows, kind, args.jitter))

        if args.record_dir:
            all_failures.extend(check_returned(node, resched, sends, args.record_dir, decider_counts))

    # A record file naming a node this run never created under node_dirs is the same
    # failure as a node directory with no events.jsonl, from the other side.
    if args.record_dir and os.path.isdir(args.record_dir):
        extra = sorted(set(name[:-len(".jsonl")] for name in os.listdir(args.record_dir)
                            if name.endswith(".jsonl")) - set(nodes))
        for node in extra:
            all_failures.append(
                "FAIL node=%s record file present but run has no such node" % node)

    if not args.record_dir:
        print("NOT CHECKED: no --record-dir given, so the returned-value half was skipped; "
              "the log is the only record of what was returned here, and comparing it with "
              "itself would be a tautology, not a check.")
    if args.jitter is None:
        print("NOT CHECKED: no --jitter given, so applied_interval_s was not checked "
              "against any jitter band.")

    for line in all_failures:
        print(line)

    hard_failures = [f for f in all_failures if f.startswith("FAIL")]
    if not hard_failures:
        if args.record_dir:
            print("ARMS PASS (arms and returned values)")
        else:
            print("ARMS PASS (arms only: returned values NOT CHECKED)")
        return 0
    print("ARMS FAIL failures=%d" % len(hard_failures))
    return 1


if __name__ == "__main__":
    sys.exit(main())
