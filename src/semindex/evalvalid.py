"""semindex.evalvalid: validity check for one stored stage-1 evaluation run.

map.md ("有効な run の判定") asks for a per-run pass over a run's own
stored artifacts that checks whether the measurement's *preconditions*
held -- every agent started and wrote its final `counters` line, the put
succeeded, records were created on schedule, the drawn disconnect
schedule actually fired on time -- never whether the Semantic Index
converged (that is a result, not a precondition; map.md is explicit that
convergence "合否に使わない").

Unlike `semindex.check` (which is pinned to one of a handful of fixed
scenario shapes: t1/t2a/t2b/t2c), this module reads nothing but what
`semindex.evalgen` itself wrote -- the stored `<name>.experiment.json`
-- to learn the run's own shape (its agent list, record schedule and
disconnect schedule), so it works for any (p, seed) stage-1 run without
a hardcoded host count or topology (map.md: "semindex.check は...計測に
は使わない。semindex.analyze は台数にもトポロジーにも依存しないので...
使える").

Run as:

    python -m semindex.evalvalid <run_dir>

which prints `{"valid": bool, "flags": [...]}` to stdout and exits 0
(valid), 1 (flagged) or 2 (run_dir itself could not be read at all --
see ValidityUsageError below).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from semindex import evalgen
from semindex.check import (
    CheckUsageError,
    agent_start_t0,
    agent_start_t0_values,
    find_one_put_record,
    read_events,
    read_results,
)

# 2026-10-23: `_check_records` (below) compares each record_created
# event's derived elapsed time against the one `at_s` its own seq was
# scheduled for (map.md's "予定どおりにレコードが作られた"). That
# per-record tolerance is not something the ticket specifies explicitly
# (unlike the ±0.5s/±1s margins on the counters_snapshot/counters/
# link-event checks, which map.md and ticket 04 give directly) -- this
# slack absorbs the agent's own ~10Hz tick granularity (and, on a run with
# no agent_start evidence at all, _derive_t0_ms's record-median fallback
# estimation jitter too -- see that function), so a record that is in
# truth exactly on its own scheduled tick is not flagged for jitter that
# has nothing to do with the record itself.
#
# 2026-10-23 review fix: this constant used to gate a much looser check
# -- only whether a record's elapsed time fell anywhere inside the whole
# [WINDOW_START_S, WINDOW_END_S] measurement window, not whether it
# matched its OWN scheduled `at_s`. A counterexample where every one of
# an origin's 30 records landed on the same single in-window instant
# (instead of each at its own 2s-apart tick) still passed that looser
# check outright. It now gates the per-record comparison below instead.
RECORD_SCHEDULE_SLACK_S = 0.5


class ValidityUsageError(ValueError):
    """`run_dir` itself cannot be read well enough to say anything (exit 2): no checks ran."""


def _node_name(host: int) -> str:
    """The node name a host index maps to -- same `f"h{host}"` convention check.py/gen.py/analyze.py all use."""
    return f"h{host}"


def _node_events_path(run_dir: str, node: str) -> str:
    """Where a stored run keeps one node's events.jsonl -- the same `debug/node_dirs/<node>/semindex/`
    layout ReCefore itself writes, preserved by `semindex.evalrun`'s copy step so this module (and
    semindex.check/semindex.analyze, if ever pointed at a stored run) all read the one layout.
    """
    return os.path.join(run_dir, "debug", "node_dirs", node, "semindex", "events.jsonl")


def _node_agent_log_path(run_dir: str, node: str) -> str:
    """Where a stored run keeps one node's semindex_agent.log (a sibling of its `semindex/` state dir)."""
    return os.path.join(run_dir, "debug", "node_dirs", node, "semindex_agent.log")


def load_experiment(run_dir: str) -> dict:
    """Read the one `<name>.experiment.json` a stored run dir holds.

    This is the run's own ground truth for its shape (agents, record
    schedule, disconnect schedule, publisher) -- `semindex.evalgen`
    writes it before generating the YAML/agent configs, and
    `semindex.evalrun` copies it into the store verbatim, so reading it
    back here never re-derives anything `evalgen` already decided.

    Raises ValidityUsageError (exit 2, not a flag) when this file itself
    cannot be found or parsed -- every other check below assumes the
    run's shape is known, so none of them can even be attempted without it.
    """
    matches = sorted(glob.glob(os.path.join(run_dir, "*.experiment.json")))
    if len(matches) != 1:
        raise ValidityUsageError(
            f"expected exactly one *.experiment.json directly under {run_dir!r}, found {len(matches)}"
        )
    try:
        with open(matches[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except OSError as exc:
        raise ValidityUsageError(f"{matches[0]} could not be read: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ValidityUsageError(f"{matches[0]} is not valid JSON: {exc}") from None


def _read_node_events(run_dir: str, agents: list[int]) -> tuple[dict[str, list[dict]], list[str]]:
    """Every expected agent's events.jsonl, plus a flag for each one missing or unreadable.

    Deliberately does not call `semindex.analyze.discover_node_events`:
    that function globs for whichever node dirs happen to exist and
    raises (aborting entirely) the moment any *one* file it finds fails
    to parse -- exactly the two things this function must not do, since
    a run with one missing or corrupt node must still report that node
    flagged while every other node's checks still run.
    """
    events_by_node: dict[str, list[dict]] = {}
    flags: list[str] = []
    for host in agents:
        node = _node_name(host)
        path = _node_events_path(run_dir, node)
        if not os.path.exists(path):
            flags.append(f"missing_events:{node}")
            continue
        try:
            events_by_node[node] = read_events(path)
        except CheckUsageError:
            flags.append(f"unreadable_events:{node}")
    return events_by_node, flags


def _estimate_t0_ms_from_records(experiment: dict, nodes_events: dict[str, list[dict]]) -> float | None:
    """Estimate t0 (epoch ms) from h0/h4's own record_created timings -- the FALLBACK rung.

    2026-10-23 fix: this used to be the ONLY t0 estimate `_check_records` compared
    records against. But it is derived FROM those same records, so a delay affecting an entire origin's
    schedule uniformly (every one of h0's records landing some fixed amount later than its own
    scheduled `at_s`) shifts this median by that same amount and cancels out of every per-record
    residual below -- a generation delay large enough to matter was passing outright. `_derive_t0_ms`
    (below) now only reaches this estimator when no node wrote an `agent_start` line at all (an older
    run captured before agent.py started doing so); every call site that uses the result is responsible
    for flagging that this weaker rung was used (`t0_estimated`), never silently.

    No stored artifact other than `agent_start` ever records the real t0 value directly: it is a
    `time.time()` reading ReCefore's own disaster.py takes once per run
    and splices into each agent's argv (host_apps.py's `{t0_ms}`
    placeholder) -- never written to a file, and never echoed by the
    agent (semindex.agent prints nothing at startup). Two indirect routes
    were considered and rejected in favor of the one below:

    - results.json's own EventRecord rows carry both a wall-clock `ts`
      and a scheduler-relative `actual_at`, from the same origin instant
      t0 is (disaster.py takes `origin = time.monotonic()` then
      `t0_ms = int(time.time() * 1000)` on the very next line), so
      `t0_ms ~= results_ts_ms(row) - actual_at * 1000` for any row. But a
      p=0 run schedules no link_down/link_up events at all (and the
      seed `put` never goes through the scheduler, so it never gets an
      EventRecord either) -- zero rows, for a fifth of every stage-1
      condition.
    - The agent's own semindex_agent.log: it is stdout-only redirection
      (command_runner.py never writes an argv header to it), and
      semindex.agent prints nothing, so there is nothing in it to read.

    What is always present, at every p: h0 and h4 each create
    RECORD_COUNT own records purely on their own elapsed-time clock
    (`Agent._elapsed_s`), never touched by link state. Each record_created
    event's `seq` (1-indexed, in creation order) maps straight back to
    this experiment's own `schedule["<host>"][seq - 1]["at_s"]` -- the
    exact relative time evalgen scheduled it for. So for every such event,
    `ts_ms - at_s * 1000` is one independent estimate of t0_ms, off only
    by that one record's own scheduling/dispatch jitter (bounded by the
    agent's own step() tick rate, not by anything p-dependent). Taking
    the median over every matched record (up to 60: 30 from h0, 30 from
    h4) cancels that jitter without needing any p>0 data at all.

    Returns None only when neither origin logged a single matching
    record_created event (so even a count-only check already has
    something to flag) -- there is nothing left to estimate from.
    """
    schedule = experiment.get("schedule") or {}
    residuals: list[float] = []
    for host in evalgen.RECORD_ORIGIN_HOSTS:
        node = _node_name(host)
        entries = schedule.get(str(host)) or []
        for event in nodes_events.get(node) or ():
            if event.get("event") != "record_created" or event.get("origin") != node:
                continue
            seq = event.get("seq")
            if not isinstance(seq, int) or not (1 <= seq <= len(entries)):
                continue  # out-of-schedule seq: not usable as a t0 anchor (already a record_seq_mismatch elsewhere)
            expected_at_s = entries[seq - 1]["at_s"]
            residuals.append(event["ts_ms"] - expected_at_s * 1000.0)
    if not residuals:
        return None
    residuals.sort()
    mid = len(residuals) // 2
    if len(residuals) % 2:
        return residuals[mid]
    return (residuals[mid - 1] + residuals[mid]) / 2.0


def _derive_t0_ms(experiment: dict, nodes_events: dict[str, list[dict]]) -> tuple[float | None, list[str]]:
    """This run's t0 (epoch ms), plus any flags describing how it was derived.

    2026-10-23 fix: `agent_start.t0_ms` -- one line every agent process writes before anything else in
    its own life, carrying the literal value it was launched with -- is now the PRIMARY rung, not
    `_estimate_t0_ms_from_records`'s median. That estimator is derived from the very records
    `_check_records` goes on to compare against it, so it cannot catch a delay that shifts an origin's
    whole schedule uniformly (see that function's own docstring for the exact counterexample);
    `agent_start.t0_ms` is independent evidence written by the agent itself, so it exposes that delay
    instead of absorbing it.

    `agent_start_t0` (semindex.check) takes every node's own `agent_start.t0_ms` and returns the median
    plus whether they disagree -- a "t0_agent_start_disagreement" flag is added whenever they do not
    all match exactly (map.md launches every node's agent with one shared t0_ms, so any difference is
    an anomaly worth surfacing, not something to resolve by silently falling back to the weaker
    estimator below).

    `_estimate_t0_ms_from_records` is reached only when NO node wrote an agent_start line at all (an
    older run captured before agent.py started writing one); its result comes back tagged with a
    "t0_estimated" flag, so a weaker, self-referential estimate is never mistaken for one measured
    directly. `(None, [])` when neither rung produced anything at all -- the caller adds its own
    "t0_undetermined" flag in that case, as before this fix.
    """
    agent_start_values = agent_start_t0_values(nodes_events)
    agent_start_result = agent_start_t0(agent_start_values)
    if agent_start_result is not None:
        t0_ms, disagreement = agent_start_result
        flags = ["t0_agent_start_disagreement"] if disagreement else []
        return t0_ms, flags

    estimated = _estimate_t0_ms_from_records(experiment, nodes_events)
    if estimated is None:
        return None, []
    return estimated, ["t0_estimated"]


def _check_counters(nodes_events: dict[str, list[dict]], t0_ms: float | None) -> list[str]:
    """Per node: exactly one counters_snapshot (at t0+10s ±0.5s) and a final counters line (at t0+130s ±1s)."""
    flags: list[str] = []
    # stage 1 fixes the snapshot schedule to exactly one entry (map.md
    # "助走の終わりに...counters_snapshot 行を書く"); evalgen's own
    # AGENT_COUNTERS_SNAPSHOT_AT_S/AGENT_STOP_AT_S are reused here rather
    # than restating 10.0/130.0, so a future change to either stays in
    # one place.
    (expected_snapshot_at_s,) = evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S
    expected_stop_at_s = evalgen.AGENT_STOP_AT_S
    for node, events in nodes_events.items():
        snapshots = [e for e in events if e.get("event") == "counters_snapshot"]
        if len(snapshots) != 1:
            flags.append(f"counters_snapshot_count:{node}")
        elif t0_ms is not None:
            elapsed = (snapshots[0]["ts_ms"] - t0_ms) / 1000.0
            if abs(elapsed - expected_snapshot_at_s) > 0.5:
                flags.append(f"counters_snapshot_timing:{node}")
        counters = [e for e in events if e.get("event") == "counters"]
        # 2026-10-23 review fix: this used to take counters[-1] (the LAST
        # line), on the theory that a restarted node's events.jsonl holds
        # one `counters` line per life (same reasoning as
        # analyze.control_bytes). But stage 1 never schedules a
        # host_app_restart -- evalgen only ever emits link_down/link_up
        # (see evalgen._link_events) -- so for this stage a second
        # `counters` line can only mean an UNPLANNED restart, and
        # CefpycoTransport's byte counters are process-local, reset to 0
        # every life (Agent.__init__'s own comment on why
        # counters_snapshot_at_s has no store-backed restart cursor like
        # schedule/probes do). map.md's control-traffic metric
        # ("最後の counters 行から...counters_snapshot 行を引く") assumes
        # both lines come from the same life; taking the last line
        # silently fed it a reset (possibly negative) number instead of
        # flagging the run. Requiring exactly one line turns that into a
        # flag.
        if len(counters) != 1:
            flags.append(f"counters_count:{node}")
        elif t0_ms is not None:
            elapsed = (counters[0]["ts_ms"] - t0_ms) / 1000.0
            if abs(elapsed - expected_stop_at_s) > 1.0:
                flags.append(f"counters_timing:{node}")
    return flags


# Byte counters CefpycoTransport tracks and `semindex.agent.Agent` writes into both the
# counters_snapshot and the final counters line -- reused by name here (never re-derived) so
# `_check_counter_decrease` compares exactly the fields map.md's control-traffic metric (final minus
# snapshot) actually subtracts.
_COUNTER_FIELDS = ("bytes_sent", "bytes_received", "control_bytes_sent", "control_bytes_received")


def _check_agent_lives(nodes_events: dict[str, list[dict]]) -> list[str]:
    """Per node: exactly one `agent_start` line.

    2026-10-23 fix: stage 1 never schedules a host_app_restart (evalgen only ever emits
    link_down/link_up -- see `_check_counters`'s own note on why this stage treats a second
    `counters` line as unplanned), so a node that wrote more than one `agent_start` -- or, just as
    badly, a node whose log never shows one at all -- had an unplanned restart (e.g. a SIGKILL) or
    crashed before logging its first life. A SIGKILL between a life's counters_snapshot and its own
    final `counters` line, followed by a restart whose NEW life writes the only final `counters` line,
    still leaves exactly one snapshot and exactly one counters line overall -- a shape
    `_check_counters` alone cannot tell apart from a clean single life. Counting lives directly, rather
    than counting the lines a life happens to produce, catches that shape regardless of how many of
    each line either life wrote.
    """
    flags = []
    for node, events in nodes_events.items():
        lives = sum(1 for e in events if e.get("event") == "agent_start")
        if lives != 1:
            flags.append(f"agent_lives:{node}")
    return flags


def _check_counter_decrease(nodes_events: dict[str, list[dict]]) -> list[str]:
    """Per node: no byte counter may be lower in the final `counters` line than in counters_snapshot.

    2026-10-23 fix: a second, independent line of defense alongside `agent_lives`
    above -- CefpycoTransport's byte counters are process-local and reset to 0 every life
    (Agent.__init__'s own comment on why counters_snapshot_at_s has no store-backed restart cursor), so
    an unplanned restart between the snapshot and the final line can leave the final numbers BELOW the
    snapshot's. map.md's control-traffic metric subtracts snapshot from final; left unchecked that
    metric can go negative. Only compared when there is exactly one of each line (requires
    `_check_counters`'s own counts to already hold) -- with any other count, this flag would be
    comparing lines that do not correspond to a single before/after pair in the first place.
    """
    flags = []
    for node, events in nodes_events.items():
        snapshots = [e for e in events if e.get("event") == "counters_snapshot"]
        counters = [e for e in events if e.get("event") == "counters"]
        if len(snapshots) != 1 or len(counters) != 1:
            continue
        snapshot, final = snapshots[0], counters[0]
        for field_name in _COUNTER_FIELDS:
            snap_value = snapshot.get(field_name)
            final_value = final.get(field_name)
            if (
                isinstance(snap_value, (int, float))
                and isinstance(final_value, (int, float))
                and final_value < snap_value
            ):
                flags.append(f"counters_decreased:{node}")
                break
    return flags


def _check_last_line(nodes_events: dict[str, list[dict]]) -> list[str]:
    """Per node: events.jsonl is non-empty and its last line is the final `counters` event."""
    flags = []
    for node, events in nodes_events.items():
        if not events:
            flags.append(f"empty_events:{node}")
        elif events[-1].get("event") != "counters":
            flags.append(f"last_line_not_counters:{node}")
    return flags


def _check_records(experiment: dict, nodes_events: dict[str, list[dict]], t0_ms: float | None) -> list[str]:
    """Per origin host: record_created forms the exact expected seq set, and (once that holds)
    each record's own creation time lands within tolerance of its own scheduled `at_s`.
    """
    flags = []
    schedule = experiment.get("schedule") or {}
    for host in evalgen.RECORD_ORIGIN_HOSTS:
        node = _node_name(host)
        entries = schedule.get(str(host)) or []
        expected_count = len(entries)
        created = [
            e for e in (nodes_events.get(node) or ()) if e.get("event") == "record_created" and e.get("origin") == node
        ]
        if len(created) != expected_count:
            flags.append(f"record_count:{node}")
            # seq/timing below compare one-to-one against the schedule; a wrong headcount
            # has no sound 1:1 mapping to check them against.
            continue

        # 2026-10-23 review fix: a correct COUNT used to be treated as proof
        # the right records were made. A counterexample where all 30 of
        # h0's record_created events were stamped seq=1 (instead of 1..30)
        # still had len(created) == expected_count and passed outright.
        # The exact set {1..expected_count}, with no duplicate or missing
        # seq, is what "予定どおりにレコードが作られた" (map.md) actually
        # needs: every scheduled slot filled exactly once.
        # Non-int seqs are filtered out before sorting, not left to raise:
        # `sorted()` on a mix of int and non-int (e.g. a missing seq,
        # None) would otherwise crash this check outright instead of
        # reporting the mismatch a missing/malformed seq already is.
        seqs = sorted(e.get("seq") for e in created if isinstance(e.get("seq"), int))
        if seqs != list(range(1, expected_count + 1)):
            flags.append(f"record_seq_mismatch:{node}")
            continue  # the seq -> at_s lookup below is meaningless without a 1:1 seq match
        if t0_ms is None:
            continue
        for event in created:
            seq = event["seq"]
            expected_at_s = entries[seq - 1]["at_s"]
            elapsed = (event["ts_ms"] - t0_ms) / 1000.0
            if abs(elapsed - expected_at_s) > RECORD_SCHEDULE_SLACK_S:
                flags.append(f"record_off_schedule:{node}:seq{seq}")
    return flags


def _check_put(results: list[dict], experiment: dict) -> list[str]:
    """The publisher's one put row exists and succeeded."""
    publisher = experiment.get("publisher") or {}
    try:
        record = find_one_put_record(results, publisher.get("uri"))
    except CheckUsageError:
        return ["put_failed"]
    return [] if record.get("success") is True else ["put_failed"]


def _check_link_events(experiment: dict, results: list[dict]) -> list[str]:
    """Every disconnect-schedule event has exactly one matching, on-time, successful results.json row.

    Matched by (event_type, nodes, scheduled_at) rather than
    `check.find_one_event_record`: that helper raises unless there is
    *exactly* one row for a given (event_type, nodes) pair, but a single
    link can go down and come back up several times across the window
    (p>0 draws one Bernoulli per 5s slot), so the same link legitimately
    has several link_down rows to tell apart -- `at` is what disambiguates
    them, since evalgen never schedules two events for the same link at
    the same `at`.
    """
    flags: list[str] = []
    expected = list(experiment.get("events") or [])
    rows = [r for r in results if r.get("op_type") == "event" and r.get("event_type") in ("link_down", "link_up")]
    used = [False] * len(rows)

    for expected_event in expected:
        etype = expected_event["type"]
        nodes = list(expected_event["nodes"])
        at = expected_event["at"]
        label = f"{etype}:{nodes[0]}-{nodes[1]}:at{at:g}"
        match_idx = None
        for i, row in enumerate(rows):
            if used[i] or row.get("event_type") != etype or row.get("scheduled_at") != at:
                continue
            if list((row.get("event") or {}).get("nodes") or []) != nodes:
                continue
            match_idx = i
            break
        if match_idx is None:
            flags.append(f"missing_link_event:{label}")
            continue
        used[match_idx] = True
        row = rows[match_idx]
        if row.get("success") is not True:
            flags.append(f"link_event_failed:{label}")
        actual_at = row.get("actual_at")
        if actual_at is None or abs(actual_at - at) > 0.5:
            flags.append(f"link_event_late:{label}")

    for i, row in enumerate(rows):
        if used[i]:
            continue
        nodes = list((row.get("event") or {}).get("nodes") or [])
        at = row.get("scheduled_at")
        if len(nodes) == 2 and at is not None:
            flags.append(f"unexpected_link_event:{row.get('event_type')}:{nodes[0]}-{nodes[1]}:at{at:g}")
        else:
            flags.append(f"unexpected_link_event:{row.get('event_type')}")
    return flags


def _check_tracebacks(run_dir: str, agents: list[int]) -> list[str]:
    """Per agent node: its semindex_agent.log contains no Python traceback."""
    flags = []
    for host in agents:
        node = _node_name(host)
        path = _node_agent_log_path(run_dir, node)
        if not os.path.exists(path):
            continue  # a missing log is already covered by missing_events/unreadable_events for this node
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if "Traceback (most recent call last):" in f.read():
                flags.append(f"traceback:{node}")
    return flags


def validate(run_dir: str) -> dict:
    """Validate one stored run dir; returns `{"valid": bool, "flags": [sorted flag strings]}`.

    Raises ValidityUsageError (exit 2 at the CLI) only when `run_dir`
    itself, or its `<name>.experiment.json`, cannot be read -- every
    other problem (a missing node, a failed put, a late link event, ...)
    is a flag, not a crash, per ticket 04's "印つきにすることを...作った
    反例で確かめる" contract.
    """
    if not os.path.isdir(run_dir):
        raise ValidityUsageError(f"{run_dir} is not a directory")

    experiment = load_experiment(run_dir)
    agents = list(experiment.get("agents") or [])

    nodes_events, flags = _read_node_events(run_dir, agents)
    flags += _check_last_line(nodes_events)

    t0_ms, t0_flags = _derive_t0_ms(experiment, nodes_events)
    flags += t0_flags
    if t0_ms is None:
        flags.append("t0_undetermined")
    flags += _check_counters(nodes_events, t0_ms)
    flags += _check_agent_lives(nodes_events)
    flags += _check_counter_decrease(nodes_events)
    flags += _check_records(experiment, nodes_events, t0_ms)
    flags += _check_tracebacks(run_dir, agents)

    try:
        results = read_results(run_dir)
    except CheckUsageError:
        flags.append("results_unreadable")
    else:
        flags += _check_put(results, experiment)
        flags += _check_link_events(experiment, results)

    flags.sort()
    return {"valid": not flags, "flags": flags}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the validity checker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="A stored stage-1 evaluation run directory (semindex.evalrun's layout)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: python3 -m semindex.evalvalid <run_dir>."""
    args = parse_args(argv)
    try:
        result = validate(args.run_dir)
    except ValidityUsageError as exc:
        print(f"semindex.evalvalid: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
