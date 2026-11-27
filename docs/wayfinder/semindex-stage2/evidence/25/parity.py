#!/usr/bin/env python3
"""Check that a live decider run was asked the same question the offline replay asks.

`observe.from_logged_rows` (evidence/22) rebuilds, from one node's own events.jsonl, the
four things a decider is handed at each poll resolution: state, resolved, history and the
instant `now`. A recording wrapper around the live decider (one `REC/hN.jsonl` per node)
writes down what the engine actually handed the decider at that same instant. If the two
disagree, the live decider answered a different question than the one the offline model
was fit and scored on, and nothing about the decider's own correctness would show that.

This check compares the engine against its OWN log for the SAME run: it cannot show that
the log (or the replay built from it) matches some other, independent ground truth, and it
says nothing about the history-building rule's own faithfulness to the original engine --
only whether this run's two recordings of it agree. It also only ever looks at resolution
rows (`state_poll_reschedule`); send rows (`state_poll`) carry no decider inputs to compare
and are not checked here.
"""
import sys
# 2026-11-27: this script imports observe.py and features.py straight out of the
# checkout (via a sys.path insert below) rather than a packaged copy, so a stray
# __pycache__ written under evidence/22 or evidence/24 is bytecode someone else's
# checkout did not ask for and would have to notice and delete by hand.
sys.dont_write_bytecode = True

import argparse    # noqa: E402
import json        # noqa: E402
import os          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
EV24 = os.path.normpath(os.path.join(HERE, "..", "24"))
if EV22 not in sys.path:
    sys.path.insert(0, EV22)
if EV24 not in sys.path:
    sys.path.insert(0, EV24)

import observe as OBS    # noqa: E402
import features as FE    # noqa: E402

# Why a late reply and an undecodable reply are NOT excuses here, although both make the
# live engine differ from the offline replay: the `state` and `resolved` a decider is
# handed are the very values its row logs, taken at the same instant, so within one run
# no event of the log can make the two sides of this comparison differ. A difference in
# either is a defect whatever else the log shows. Both are still counted per node below,
# because those counts are what a reader needs to judge the engine against the replay.
# 2026-11-27: an undecodable reply is counted from the reschedule rows (answered with
# `empty` null) and not from `state_rx` rows with `hi == {}`: a valid reply from a
# neighbour that holds nothing yet writes the same `state_rx` row, and a clean run had
# two to six of those per node.


def is_history_field(field):
    """Whether `field` names the pair's history as the log's row order rebuilds it: the raw
    `history`, or `record.exchanges`, the same history as the model reads it."""
    return field == "history" or field.startswith("history[") or field == "record.exchanges"


def asof_histories(pair_rows, ask_ms, own_epoch):
    """The history a decider must have been handed at `ask_ms`, rebuilt from the log.

    The engine appends a pair's exchange to its history when that exchange's row is
    written, and a deferred answer's row is written when the answer is applied
    (`decided_at_ms`), not when it was asked. So the history at an ask is the pair's rows
    already written by then, which is NOT the same list as "the pair's earlier rows in
    file order" once two resolutions of one pair overlap. Two lists are returned, with and
    without the rows written in the very millisecond of the ask, because the log cannot
    say which side of the ask such a row fell on.
    """
    def build(strict):
        rows = [e for e in pair_rows
                if e["action"]["poll_epoch_ms"] != own_epoch
                and (e["action"].get("decided_at_ms", e["ts_ms"]) < ask_ms if strict
                     else e["action"].get("decided_at_ms", e["ts_ms"]) <= ask_ms)]
        return [{"ts_ms": e["ts_ms"], **{k: e["action"][k] for k in OBS.EXCHANGE_FIELDS}}
                for e in rows][-OBS.MAX_EXCHANGES:]
    return build(False), build(True)


def make_classifier(decider_counts):
    """Build a per-node classifier: mismatch_class(field, key) -> class name.

    Each excuse is tied to the one thing it can explain, and is a budget spent once,
    never a standing permission.
    - A recorded resolution with no logged row (`join_recorded_missing`) is excused as
      `unanswered` only when the node's own `decider_counts` row names that very
      resolution as unanswered (when it carries an `unanswered_asked` list), or, on a
      log written before that list existed, up to `decider_counts.unanswered` times,
      used up one at a time: a run with one unanswered resolution at the stop boundary
      must not be allowed to wave away an unrelated pile of real defects.
    - A history difference (`history`, `history[i]`, `record.exchanges`) is excused as
      `overlap` for at most `decider_counts.overlap` DECISIONS per node, keyed by
      (neighbor, poll_epoch_ms): the history and record.exchanges findings of one
      decision are the same excuse and spend the budget only once, but a decision the
      budget has already run out on is a DEFECT like any other.
    Nothing excuses `history_asof` or a history difference once the budget is spent --
    2026-11-27: a second `agent_start` used to excuse every history finding on the node
    as a "restart", which let a genuinely diverging history hide behind an unrelated
    restart; agent_start_count is still reported in the per-node INFO line but no
    longer changes any class here.
    """
    counts = decider_counts or {}
    budget = {"unanswered_left": counts.get("unanswered", 0),
              "overlap_left": counts.get("overlap", 0)}
    overlap_used_keys = set()
    named = None
    if "unanswered_asked" in counts:
        named = {(u["neighbor"], u["poll_epoch_ms"]) for u in counts["unanswered_asked"]}

    def classify(field, key):
        if field == "join_logged_missing":
            # Nothing in the log explains a decision the engine logged without ever
            # asking the decider: there is no decider-side event that could produce this.
            return "DEFECT"
        if field == "join_recorded_missing":
            if named is not None:
                return "unanswered" if key in named else "DEFECT"
            if budget["unanswered_left"] > 0:
                budget["unanswered_left"] -= 1
                return "unanswered"
            return "DEFECT"
        if is_history_field(field):
            if key in overlap_used_keys:
                return "overlap"
            if budget["overlap_left"] > 0:
                budget["overlap_left"] -= 1
                overlap_used_keys.add(key)
                return "overlap"
        return "DEFECT"

    return classify


def load_jsonl(path):
    """Parse one newline-delimited JSON file into a list of dicts, in file order.

    Order matters everywhere this is used: history reconstruction depends on reading
    rows in the sequence they were written, not some other order a dict or set would
    silently impose.
    """
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
    events.jsonl is not "no data to check", it is the thing this check exists to catch:
    a node the run silently never logged. A missing events.jsonl is reported as a
    DEFECT rather than this function quietly leaving the node out.
    """
    base = os.path.join(run_dir, "debug", "node_dirs")
    # 2026-11-27: filter to directories only. node_dirs holds nothing but node
    # directories on every stage this check has ever run against, but a plain file
    # dropped next to them (a README, a tarball) is not a node, and without this
    # filter it would be treated as one and fail with a phantom missing-events.jsonl
    # DEFECT.
    return sorted(e for e in os.listdir(base) if os.path.isdir(os.path.join(base, e)))


def logged_reschedule_rows(rows):
    """The subset and order of `rows` that `observe.from_logged_rows` walks.

    Needed alongside the generator's own yielded tuples so each yielded record can be
    joined back to the row it came from (ts_ms, kind, node) without reimplementing the
    generator's filter a second time and risking the two filters drifting apart.
    """
    return [e for e in rows if e.get("event") == "decision"
            and e.get("kind") == "state_poll_reschedule"]


def truncate(value):
    """Render a value for a one-line finding without letting a large dict blow up output."""
    text = repr(value)
    return text if len(text) <= 200 else text[:200] + "..."


def check_node(node, run_dir, record_dir):
    """Run the full parity check for one node; returns (mismatches, info_lines,
    defect_count, joined_count, overlap_excused_count, unanswered_excused_count)."""
    events_path = os.path.join(run_dir, "debug", "node_dirs", node, "semindex", "events.jsonl")
    if not os.path.isfile(events_path):
        # 2026-11-27: a node directory with no events.jsonl means its process never
        # wrote a single event. Returning "nothing to check" here would hide exactly
        # the failure this script exists to catch, so it is a DEFECT like any other.
        return (["MISMATCH node=%s neighbor=- poll_epoch_ms=- field=events_file "
                 "class=DEFECT logged=missing recorded=-" % node], [], 1, 0, 0, 0)
    rows = load_jsonl(events_path)
    reschedule_rows = logged_reschedule_rows(rows)

    decider_counts = next((e for e in rows if e.get("event") == "decider_counts"), None)
    late_replies = sum(1 for e in rows if e.get("event") == "state_poll_late_reply")
    empty_unknown = sum(1 for e in reschedule_rows if e["action"].get("trigger") == "answered"
                        and e["action"].get("empty") is None)
    agent_starts = sum(1 for e in rows if e.get("event") == "agent_start")

    classify = make_classifier(decider_counts)

    # Logged side: zip the generator's yields with the filtered rows it walked, in
    # lockstep, since from_logged_rows yields exactly one tuple per reschedule row in
    # file order. The record road is walked the same way, with the sibling function that
    # is what the model actually reads, not a second derivation of the same record.
    logged_by_key = {}
    logged_dupe_keys = set()
    for row, (neighbor, state, resolved, history, now_ms) in zip(
            reschedule_rows, OBS.from_logged_rows(rows)):
        key = (neighbor, resolved["poll_epoch_ms"])
        if key in logged_by_key:
            logged_dupe_keys.add(key)
        logged_by_key[key] = {"row": row, "neighbor": neighbor, "state": state,
                               "resolved": resolved, "history": history, "now_ms": now_ms}

    rows_by_neighbor = {}
    for row in reschedule_rows:
        rows_by_neighbor.setdefault(row["action"]["neighbor"], []).append(row)

    logged_record_by_key = {}
    for row, (neighbor, record) in zip(reschedule_rows, FE.records_from_logged_rows(rows)):
        logged_record_by_key[(neighbor, row["action"]["poll_epoch_ms"])] = record

    record_path = os.path.join(record_dir, node + ".jsonl")
    has_record = os.path.isfile(record_path)

    mismatches = []
    # 2026-11-27: defect_keys counts DECISIONS, not finding lines, so two findings
    # on the same (neighbor, poll_epoch_ms) -- e.g. "history" and the
    # matching "record.exchanges" -- are one DEFECT, the same way they are one spent
    # overlap excuse above. A finding with no decision to key on (decider_counts,
    # reschedule_count, record_file, an empty join) adds a synthetic, node-scoped key
    # instead of a real (neighbor, poll_epoch_ms) pair, so it still counts as exactly
    # one item without colliding with a real decision's key. (events_file is its own
    # early return before this set exists, and is already exactly one item.)
    defect_keys = set()
    excused_keys = {"overlap": set(), "unanswered": set()}

    def report(neighbor, poll_epoch_ms, field, logged_val, recorded_val):
        key = (neighbor, poll_epoch_ms)
        c = classify(field, key)
        if c == "DEFECT":
            defect_keys.add(key)
        elif c in excused_keys:
            excused_keys[c].add(key)
        mismatches.append(
            "MISMATCH node=%s neighbor=%s poll_epoch_ms=%s field=%s class=%s logged=%s recorded=%s"
            % (node, neighbor, poll_epoch_ms, field, c, truncate(logged_val), truncate(recorded_val)))

    if has_record and decider_counts is None:
        # Every excuse above is read out of this row, and a decider run always writes
        # one; a record with no matching decider_counts row means the log is missing
        # something a real run never omits, not that there is nothing to excuse.
        mismatches.append("MISMATCH node=%s neighbor=- poll_epoch_ms=- field=decider_counts "
                           "class=DEFECT logged=missing recorded=present" % node)
        defect_keys.add(("__node__", node, "decider_counts"))

    if decider_counts and decider_counts.get("deferred", 0) > 0:
        # A deferred resolution's row is written once applied (if in time) or on a late
        # reply; `applied` and `late` between them must therefore account for every
        # reschedule row this node logged, deferred or not.
        expected = decider_counts.get("applied", 0) + decider_counts.get("late", 0)
        if len(reschedule_rows) != expected:
            mismatches.append(
                "MISMATCH node=%s neighbor=- poll_epoch_ms=- field=reschedule_count "
                "class=DEFECT logged=%d expected_applied_plus_late=%d"
                % (node, len(reschedule_rows), expected))
            defect_keys.add(("__node__", node, "reschedule_count"))

    recorded_resolutions = []
    recorded_sends = 0
    starts_in_record = 0
    recorded_by_key = {}
    recorded_record_by_key = {}
    recorded_dupe_keys = set()

    if not has_record:
        mismatches.append("MISMATCH node=%s neighbor=- poll_epoch_ms=- field=record_file "
                           "class=DEFECT logged=present recorded=missing" % node)
        defect_keys.add(("__node__", node, "record_file"))
    else:
        rec_rows = load_jsonl(record_path)
        starts_in_record = sum(1 for e in rec_rows if e.get("rec") == "start")
        for e in rec_rows:
            if e.get("rec") != "decide":
                continue
            where = e.get("obs", {}).get("where", {})
            if where.get("kind") == "state_poll_reschedule":
                recorded_resolutions.append(e)
            elif where.get("kind") == "state_poll":
                recorded_sends += 1

        for e in recorded_resolutions:
            where = e["obs"]["where"]
            resolved = e["obs"]["resolved"]
            key = (where.get("neighbor"), resolved.get("poll_epoch_ms") if resolved else None)
            if key in recorded_by_key:
                recorded_dupe_keys.add(key)
            recorded_by_key[key] = e
            recorded_record_by_key[key] = FE.record_from_port_obs(e["obs"])

    for key in sorted(logged_dupe_keys):
        mismatches.append("MISMATCH node=%s neighbor=%s poll_epoch_ms=%s field=join "
                           "class=DEFECT logged=duplicate_key recorded=-" % (node, key[0], key[1]))
        defect_keys.add(key)
    for key in sorted(recorded_dupe_keys):
        mismatches.append("MISMATCH node=%s neighbor=%s poll_epoch_ms=%s field=join "
                           "class=DEFECT logged=- recorded=duplicate_key" % (node, key[0], key[1]))
        defect_keys.add(key)

    joined_keys = set(logged_by_key) & set(recorded_by_key)
    if has_record and not joined_keys:
        # A node on which nothing was compared must not count as a node that agreed:
        # a decider that was never asked leaves both sides empty and every loop below idle.
        mismatches.append("MISMATCH node=%s neighbor=- poll_epoch_ms=- field=join "
                           "class=DEFECT logged=%d recorded=%d joined=0"
                           % (node, len(logged_by_key), len(recorded_by_key)))
        defect_keys.add(("__node__", node, "join_empty"))
    only_logged = set(logged_by_key) - set(recorded_by_key)
    only_recorded = set(recorded_by_key) - set(logged_by_key)

    for key in sorted(only_logged, key=lambda k: (k[0] or "", k[1] or 0)):
        report(key[0], key[1], "join_logged_missing",
               "logged row with no recorded resolution", "none")

    for key in sorted(only_recorded, key=lambda k: (k[0] or "", k[1] or 0)):
        report(key[0], key[1], "join_recorded_missing",
               "none", "recorded resolution with no logged row")

    # A recorded resolution with no logged row still had a history the decider was
    # handed at that instant; having no row to join against does not exempt it from
    # the as-of-ask bookkeeping check that joined pairs get further down.
    for key in sorted(only_recorded, key=lambda k: (k[0] or "", k[1] or 0)):
        neighbor, poll_epoch_ms = key
        rec_entry = recorded_by_key[key]
        where = rec_entry["obs"]["where"]
        rec_hist = rec_entry["obs"]["history"]
        asof = asof_histories(rows_by_neighbor.get(neighbor, []), where.get("ts_ms"), poll_epoch_ms)
        if rec_hist not in asof:
            report(neighbor, poll_epoch_ms, "history_asof", asof[0], rec_hist)

    # The other direction of the same claim: a resolution the engine names as unanswered
    # must be one the decider was asked and that has no row. One that was never recorded,
    # or that has a row after all, means the engine's own account of its pending store
    # is wrong.
    for u in (decider_counts or {}).get("unanswered_asked", []):
        key = (u["neighbor"], u["poll_epoch_ms"])
        if has_record and key not in only_recorded:
            mismatches.append("MISMATCH node=%s neighbor=%s poll_epoch_ms=%s field=unanswered_asked "
                               "class=DEFECT logged=named_unanswered recorded=%s"
                               % (node, key[0], key[1],
                                  "has_logged_row" if key in logged_by_key else "never_asked"))
            defect_keys.add(key)

    for key in sorted(joined_keys, key=lambda k: (k[0] or "", k[1] or 0)):
        neighbor, poll_epoch_ms = key
        logged_entry = logged_by_key[key]
        rec_entry = recorded_by_key[key]
        row = logged_entry["row"]
        obs = rec_entry["obs"]
        where = obs["where"]

        for field in set(logged_entry["state"]) | set(obs["state"]):
            if logged_entry["state"].get(field) != obs["state"].get(field):
                report(neighbor, poll_epoch_ms, "state." + field,
                       logged_entry["state"].get(field), obs["state"].get(field))

        if logged_entry["resolved"] != obs["resolved"]:
            report(neighbor, poll_epoch_ms, "resolved", logged_entry["resolved"], obs["resolved"])

        logged_hist = logged_entry["history"]
        rec_hist = obs["history"]
        asof = asof_histories(rows_by_neighbor[neighbor], where.get("ts_ms"), poll_epoch_ms)
        if rec_hist not in asof:
            # 2026-11-27: excused by nothing -- the restart excuse that used to cover
            # this is gone, so a history_asof difference is a DEFECT like any other.
            # This is the engine's own bookkeeping, and it must hold however late or
            # overlapped answers were.
            report(neighbor, poll_epoch_ms, "history_asof", asof[0], rec_hist)
        elif logged_hist != rec_hist:
            if len(logged_hist) != len(rec_hist):
                report(neighbor, poll_epoch_ms, "history",
                       "length %d" % len(logged_hist), "length %d" % len(rec_hist))
            else:
                for i, (a, b) in enumerate(zip(logged_hist, rec_hist)):
                    if a != b:
                        report(neighbor, poll_epoch_ms, "history[%d]" % i, a, b)
                        break

        if logged_entry["now_ms"] != where.get("ts_ms"):
            report(neighbor, poll_epoch_ms, "where.ts_ms", logged_entry["now_ms"], where.get("ts_ms"))

        if node != where.get("node"):
            report(neighbor, poll_epoch_ms, "where.node", node, where.get("node"))

        if row.get("kind") != where.get("kind"):
            report(neighbor, poll_epoch_ms, "where.kind", row.get("kind"), where.get("kind"))

        # Second half: compare the numeric record the model actually reads, built by
        # the two functions whose whole purpose is to be the one road each side takes
        # to get there (records_from_logged_rows, record_from_port_obs), rather than a
        # third derivation of the same record done here.
        logged_record = logged_record_by_key.get(key)
        recorded_record = recorded_record_by_key.get(key)
        if logged_record != recorded_record:
            for field in set(logged_record or {}) | set(recorded_record or {}):
                lv = (logged_record or {}).get(field)
                rv = (recorded_record or {}).get(field)
                if lv != rv and not (field == "exchanges" and rec_hist not in asof):
                    report(neighbor, poll_epoch_ms, "record." + field, lv, rv)

    info = [
        "INFO node=%s logged_reschedule_rows=%d recorded_resolutions=%d joined_pairs=%d "
        "recorded_sends=%d" % (node, len(reschedule_rows), len(recorded_resolutions),
                               len(joined_keys), recorded_sends),
        "INFO node=%s decider_counts=%s" % (node, decider_counts if decider_counts else "none"),
        "INFO node=%s state_poll_late_reply_count=%d answered_empty_unknown_rows=%d "
        "agent_start_count=%d record_start_count=%d" % (
            node, late_replies, empty_unknown, agent_starts, starts_in_record),
    ]
    return (mismatches, info, len(defect_keys),
            len(joined_keys), len(excused_keys["overlap"]), len(excused_keys["unanswered"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--record-dir", required=True)
    args = parser.parse_args()

    nodes = find_nodes(args.run)
    if not nodes:
        print("no node directories found under %s" % args.run)
        return 2

    all_mismatches = []
    all_info = []
    total_defects = 0
    total_joined = 0
    total_overlap = 0
    total_unanswered = 0
    for node in nodes:
        mismatches, info, defects, joined, overlap_n, unanswered_n = check_node(
            node, args.run, args.record_dir)
        all_mismatches.extend(mismatches)
        all_info.extend(info)
        total_defects += defects
        total_joined += joined
        total_overlap += overlap_n
        total_unanswered += unanswered_n

    # A record file naming a node this run never created under node_dirs is the same
    # failure as a node directory with no events.jsonl, from the other side.
    if os.path.isdir(args.record_dir):
        extra = sorted(set(name[:-len(".jsonl")] for name in os.listdir(args.record_dir)
                            if name.endswith(".jsonl")) - set(nodes))
        for node in extra:
            all_mismatches.append(
                "MISMATCH node=%s neighbor=- poll_epoch_ms=- field=record_file "
                "class=DEFECT logged=missing recorded=present" % node)
            total_defects += 1

    for line in all_info:
        print(line)
    for line in all_mismatches:
        print(line)

    excused_decisions = total_overlap + total_unanswered
    if total_defects == 0:
        print("PARITY PASS joined=%d excused_decisions=%d (overlap=%d unanswered=%d)"
              % (total_joined, excused_decisions, total_overlap, total_unanswered))
        return 0
    print("PARITY FAIL defects=%d joined=%d excused_decisions=%d (overlap=%d unanswered=%d)"
          % (total_defects, total_joined, excused_decisions, total_overlap, total_unanswered))
    return 1


if __name__ == "__main__":
    sys.exit(main())
