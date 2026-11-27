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


def make_classifier(decider_counts, agent_starts):
    """Build a per-node classifier: mismatch_class(field) -> class name.

    Each excuse is tied to the one thing it can explain. A recorded resolution with no
    logged row is excused only when the node's own `decider_counts` row names that very
    resolution as unanswered: a run with one unanswered resolution at the stop boundary
    must not be allowed to wave away an unrelated pile of real defects. A log written
    before the row named them carries a count only, and there the excuse is a budget of
    that many, used up one at a time. A history that differs from
    the log's row-order rebuild is excused as `overlap` only on a node that counted one,
    and only when the as-of-ask rebuild agreed (the caller reports `history_asof`
    otherwise, which nothing but a restart excuses).
    """
    counts = decider_counts or {}
    overlap = counts.get("overlap", 0)
    budget = {"unanswered_left": counts.get("unanswered", 0)}
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
        if field == "history_asof":
            # A restarted agent begins with an empty history while the log keeps the
            # first life's rows; nothing else makes the engine's own history differ.
            return "restart" if agent_starts > 1 else "DEFECT"
        if is_history_field(field):
            if overlap > 0:
                return "overlap"
            if agent_starts > 1:
                return "restart"
        return "DEFECT"

    return classify


def load_jsonl(path):
    """Parse one newline-delimited JSON file into a list of dicts, in file order.

    Order matters everywhere this is used: history reconstruction and the "more than one
    agent_start" restart check both depend on reading rows in the sequence they were
    written, not some other order a dict or set would silently impose.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_nodes(run_dir):
    """List of hN names that have an events.jsonl under RUN/debug/node_dirs.

    A node directory without this file is not "no data to check", it is the thing this
    check exists to catch: a node the run silently never logged.
    """
    base = os.path.join(run_dir, "debug", "node_dirs")
    nodes = []
    for name in sorted(os.listdir(base)):
        if os.path.isfile(os.path.join(base, name, "semindex", "events.jsonl")):
            nodes.append(name)
    return nodes


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
    """Run the full parity check for one node; returns (mismatches, info_lines, defect_count)."""
    events_path = os.path.join(run_dir, "debug", "node_dirs", node, "semindex", "events.jsonl")
    rows = load_jsonl(events_path)
    reschedule_rows = logged_reschedule_rows(rows)

    decider_counts = next((e for e in rows if e.get("event") == "decider_counts"), None)
    late_replies = sum(1 for e in rows if e.get("event") == "state_poll_late_reply")
    empty_unknown = sum(1 for e in reschedule_rows if e["action"].get("trigger") == "answered"
                        and e["action"].get("empty") is None)
    agent_starts = sum(1 for e in rows if e.get("event") == "agent_start")

    classify = make_classifier(decider_counts, agent_starts)

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
    defects = [0]

    def report(neighbor, poll_epoch_ms, field, logged_val, recorded_val):
        c = classify(field, (neighbor, poll_epoch_ms))
        if c == "DEFECT":
            defects[0] += 1
        mismatches.append(
            "MISMATCH node=%s neighbor=%s poll_epoch_ms=%s field=%s class=%s logged=%s recorded=%s"
            % (node, neighbor, poll_epoch_ms, field, c, truncate(logged_val), truncate(recorded_val)))

    recorded_resolutions = []
    recorded_sends = 0
    starts_in_record = 0
    recorded_by_key = {}
    recorded_record_by_key = {}
    recorded_dupe_keys = set()

    if not has_record:
        mismatches.append("MISMATCH node=%s neighbor=- poll_epoch_ms=- field=record_file "
                           "class=DEFECT logged=present recorded=missing" % node)
        defects[0] += 1
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
        defects[0] += 1
    for key in sorted(recorded_dupe_keys):
        mismatches.append("MISMATCH node=%s neighbor=%s poll_epoch_ms=%s field=join "
                           "class=DEFECT logged=- recorded=duplicate_key" % (node, key[0], key[1]))
        defects[0] += 1

    joined_keys = set(logged_by_key) & set(recorded_by_key)
    if has_record and not joined_keys:
        # A node on which nothing was compared must not count as a node that agreed:
        # a decider that was never asked leaves both sides empty and every loop below idle.
        mismatches.append("MISMATCH node=%s neighbor=- poll_epoch_ms=- field=join "
                           "class=DEFECT logged=%d recorded=%d joined=0"
                           % (node, len(logged_by_key), len(recorded_by_key)))
        defects[0] += 1
    only_logged = set(logged_by_key) - set(recorded_by_key)
    only_recorded = set(recorded_by_key) - set(logged_by_key)

    for key in sorted(only_logged, key=lambda k: (k[0] or "", k[1] or 0)):
        report(key[0], key[1], "join_logged_missing",
               "logged row with no recorded resolution", "none")

    for key in sorted(only_recorded, key=lambda k: (k[0] or "", k[1] or 0)):
        report(key[0], key[1], "join_recorded_missing",
               "none", "recorded resolution with no logged row")

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
            defects[0] += 1

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
            # Checked first and excused by nothing but a restart: this is the engine's
            # own bookkeeping, and it must hold however late or overlapped answers were.
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
    return mismatches, info, defects[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--record-dir", required=True)
    args = parser.parse_args()

    nodes = find_nodes(args.run)
    if not nodes:
        print("no nodes with events.jsonl found under %s" % args.run)
        return 2

    all_mismatches = []
    all_info = []
    total_defects = 0
    for node in nodes:
        mismatches, info, defects = check_node(node, args.run, args.record_dir)
        all_mismatches.extend(mismatches)
        all_info.extend(info)
        total_defects += defects

    for line in all_info:
        print(line)
    for line in all_mismatches:
        print(line)

    non_defect = len(all_mismatches) - total_defects
    if total_defects == 0:
        print("PARITY PASS (non_defect_findings=%d)" % non_defect)
        return 0
    print("PARITY FAIL defects=%d non_defect_findings=%d" % (total_defects, non_defect))
    return 1


if __name__ == "__main__":
    sys.exit(main())
