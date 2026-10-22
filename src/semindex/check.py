"""semindex.check: PASS/FAIL checkers for the T1/T2 acceptance experiments (map.md).

Reads one experiment's description (the same file semindex.gen consumed,
so N/M/hosts/fault counts are never duplicated as separate constants),
every agent host's events.jsonl under a ReCefore run directory, and that
run's results.json, then prints one line per assertion and exits:

    0  every assertion passed
    1  at least one PROTOCOL assertion failed (the sync/fault/restart
       behavior itself is wrong)
    2  usage error (bad arguments, or the experiment/run directory do not
       match what this mode expects)
    3  at least one PRECONDITION assertion failed (the scenario's own
       timing did not hold up -- e.g. records had not finished
       propagating before a link_down actually fired -- so the protocol
       assertions below it were not meaningfully exercised)

Every PRECONDITION/PROTOCOL check that ran is still printed either way
(this module's own `_check`, like examples/two_host_sync/check_events.py's,
never skips a later assertion just because an earlier one failed); only
the exit code distinguishes "didn't get a fair run" from "got a fair run
and broke."

Run as:

    python -m semindex.check <t1|t2a|t2b|t2c> <experiment.json> <run_dir>

`run_dir` is the ReCefore experiment directory (`<output_dir>/ex<num>_seed<seed>`
-- core.artifacts.experiment_dir_name, in the companion ReCefore checkout).
Node events are read from `<run_dir>/debug/node_dirs/h<N>/semindex/events.jsonl`
(the generated config's own `debug.artifacts: ["node_dirs"]`), and the put
and scheduler-event ACTUAL fire times are read from `<run_dir>/results.json`
(EventRecord/ContentRecord rows -- see that ReCefore checkout's
src/core/records.py), never from the experiment's nominal `at:`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

from semindex.gen import ExperimentError, ExperimentSpec, parse_experiment

_MODES = ("t1", "t2a", "t2b", "t2c")


class CheckUsageError(ValueError):
    """The experiment description or run directory do not fit what this check mode expects."""


# -- events.jsonl / results.json reading --------------------------------------


def _node_events_path(run_dir: str, node: str) -> str:
    """Where ReCefore's debug.artifacts: ["node_dirs"] archives one agent's events.jsonl."""
    return os.path.join(run_dir, "debug", "node_dirs", node, "semindex", "events.jsonl")


def read_events(path: str) -> list[dict]:
    """Parse one events.jsonl file into a list of event dicts, in file order.

    Raises CheckUsageError (not a parse-time crash) naming the bad line, the
    same "never traceback on a truncated last line" reasoning
    examples/two_host_sync/check_events.py documents for a host_apps
    process still writing when ReCefore's SIGKILL lands.
    """
    events = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise CheckUsageError(f"{path} line {lineno} unparsable: {exc}") from None
    except FileNotFoundError:
        raise CheckUsageError(f"{path} does not exist") from None
    return events


def read_results(run_dir: str) -> list[dict]:
    """Parse `<run_dir>/results.json` into its list of record dicts."""
    path = os.path.join(run_dir, "results.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise CheckUsageError(f"{path} does not exist") from None
    except json.JSONDecodeError as exc:
        raise CheckUsageError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(data, list):
        raise CheckUsageError(f"{path} must be a JSON list of records")
    return data


def results_ts_ms(record: dict) -> float:
    """A results.json record's `ts` (timestamp_utc(), ISO 8601 with a UTC offset) as epoch ms.

    This is the ACTUAL wall-clock time ReCefore recorded the outcome at --
    directly comparable to an agent's own events.jsonl `ts_ms` (also wall
    clock; Mininet hosts share the clock). `actual_at` on the same record
    is monotonic-clock seconds since the scheduler's own start, with no
    epoch anchor, so it is not usable for this cross-file comparison.
    """
    return datetime.fromisoformat(record["ts"]).timestamp() * 1000.0


def find_one_event_record(results: list[dict], event_type: str, **field_filters) -> dict:
    """The single results.json EventRecord row matching `event_type` and every `field_filters`
    item against its nested `event` dict (e.g. host=0 for a host_app_restart row).

    Raises CheckUsageError if there is not exactly one match: zero means
    the scenario never actually fired that event (a usage/setup problem,
    not a PASS/FAIL outcome this checker can judge), and more than one
    means this experiment's event list does not match what this check
    mode assumes (one link_down, one link_up, etc).
    """
    matches = []
    for record in results:
        if record.get("op_type") != "event" or record.get("event_type") != event_type:
            continue
        event = record.get("event") or {}
        if all(event.get(k) == v for k, v in field_filters.items()):
            matches.append(record)
    if len(matches) != 1:
        raise CheckUsageError(
            f"expected exactly one results.json {event_type} row matching {field_filters}, found {len(matches)}"
        )
    return matches[0]


def find_one_put_record(results: list[dict], uri: str) -> dict:
    """The single results.json ContentRecord row for the `put` of `uri`."""
    matches = [r for r in results if r.get("op_type") == "put" and r.get("uri") == uri]
    if len(matches) != 1:
        raise CheckUsageError(f"expected exactly one results.json put row for {uri!r}, found {len(matches)}")
    return matches[0]


# -- small query helpers over one node's event list ---------------------------


def _events_of(events: list[dict], event_type: str, **field_filters) -> list[dict]:
    """Every event in `events` whose `event` field is `event_type` and matches every `field_filters` item."""
    return [e for e in events if e.get("event") == event_type and all(e.get(k) == v for k, v in field_filters.items())]


def _record_created_seqs(events: list[dict], origin: str) -> list[int]:
    """Every seq this node's own record_created events reported for `origin`, in file order (duplicates kept)."""
    return [e["seq"] for e in _events_of(events, "record_created", origin=origin)]


def _record_stored(events: list[dict], origin: str) -> list[dict]:
    """Every record_stored event for `origin` in `events`, in file order (duplicates kept)."""
    return _events_of(events, "record_stored", origin=origin)


def _contiguous_unique(seqs: list[int], n: int) -> str | None:
    """None if `seqs` is exactly {1, ..., n} with no duplicate, else a one-line description of how it differs."""
    seq_set = set(seqs)
    if len(seqs) != len(seq_set):
        dupes = sorted({s for s in seqs if seqs.count(s) > 1})
        return f"duplicates present: {dupes}"
    if seq_set != set(range(1, n + 1)):
        return f"got {sorted(seq_set)}, expected 1..{n}"
    return None


# -- PASS/FAIL/PRECOND printing -----------------------------------------------


class _Checker:
    """Accumulates PRECONDITION/PROTOCOL assertion results and prints one line per assertion.

    Mirrors examples/two_host_sync/check_events.py's `_check` (always
    print, combine with `&=` not `and`), split into two independently
    tracked categories so main() can choose exit code 3 vs 1 vs 0.
    """

    def __init__(self) -> None:
        """Start with both categories passing; any failing check below flips the matching flag to False for good."""
        self.precondition_ok = True
        self.protocol_ok = True

    def precondition(self, label: str, condition: bool, detail: str = "") -> bool:
        """Print one PRECOND-PASS/PRECOND-FAIL line for a timing/setup assertion and record its result."""
        status = "PRECOND-PASS" if condition else "PRECOND-FAIL"
        suffix = f": {detail}" if detail else ""
        print(f"{status}: {label}{suffix}")
        self.precondition_ok &= condition
        return condition

    def protocol(self, label: str, condition: bool, detail: str = "") -> bool:
        """Print one PASS/FAIL line for a sync-protocol assertion and record its result."""
        status = "PASS" if condition else "FAIL"
        suffix = f": {detail}" if detail else ""
        print(f"{status}: {label}{suffix}")
        self.protocol_ok &= condition
        return condition


# -- T1 ------------------------------------------------------------------


def check_t1(spec: ExperimentSpec, run_dir: str) -> _Checker:
    """map.md T1: relay through a partition, discovery-only during the outage, full catch-up after."""
    c = _Checker()
    origin_host = spec.publisher.host
    origin = f"h{origin_host}"

    link_down_events = [e for e in spec.events if e.type == "link_down"]
    link_up_events = [e for e in spec.events if e.type == "link_up"]
    if len(link_down_events) != 1 or len(link_up_events) != 1:
        raise CheckUsageError("t1 requires exactly one link_down and one link_up event")
    link_down_at, link_up_at = link_down_events[0].at, link_up_events[0].at

    all_schedule = spec.schedule.get(origin_host, ())
    n_entries = [e for e in all_schedule if e.at_s < link_down_at]
    m_entries = [e for e in all_schedule if e.at_s >= link_down_at]
    n, m = len(n_entries), len(m_entries)
    if n == 0 or m == 0:
        raise CheckUsageError("t1 requires h0 schedule entries both before and after link_down")

    all_probes = [(host, p) for host, probes in spec.probes.items() for p in probes]
    during = [(h, p) for h, p in all_probes if link_down_at <= p.at_s <= link_up_at]
    after = [(h, p) for h, p in all_probes if p.at_s > link_up_at]
    if len(during) != 1 or len(after) != 1:
        raise CheckUsageError("t1 requires exactly one probe nominally during the outage and one after")
    prober_host = during[0][0]
    if after[0][0] != prober_host:
        raise CheckUsageError("t1's during/after probes must be on the same host")
    if prober_host == origin_host:
        raise CheckUsageError("t1's prober host must not be the origin itself")
    prober = f"h{prober_host}"
    far_node = prober

    non_origin_nodes = [f"h{h}" for h in spec.agents if h != origin_host]

    events_by_node = {f"h{h}": read_events(_node_events_path(run_dir, f"h{h}")) for h in spec.agents}
    results = read_results(run_dir)

    put_record = find_one_put_record(results, spec.publisher.uri)
    link_down_record = find_one_event_record(results, "link_down", nodes=list(link_down_events[0].fields["nodes"]))
    link_up_record = find_one_event_record(results, "link_up", nodes=list(link_up_events[0].fields["nodes"]))
    link_down_ts = results_ts_ms(link_down_record)
    link_up_ts = results_ts_ms(link_up_record)

    c.precondition("publisher's put succeeded", put_record.get("success") is True, f"success={put_record.get('success')}")

    far_events = events_by_node[far_node]
    far_stored_before_outage = [
        e for e in _record_stored(far_events, origin) if e["ts_ms"] < link_down_ts and e["seq"] <= n
    ]
    c.precondition(
        f"all {n} pre-outage records reached {far_node} before link_down actually fired",
        len({e["seq"] for e in far_stored_before_outage}) == n,
        f"{len({e['seq'] for e in far_stored_before_outage})}/{n} stored before link_down (actual ts {link_down_ts:.0f})",
    )

    origin_events = events_by_node[origin]
    created = _events_of(origin_events, "record_created", origin=origin)
    post_outage_created = [e for e in created if e["seq"] > n]
    late_enough = [e for e in post_outage_created if e["ts_ms"] > link_down_ts]
    c.precondition(
        f"all {m} post-outage records were actually created after link_down fired",
        len(post_outage_created) == m and len(late_enough) == m,
        f"{len(post_outage_created)} created with seq>{n}, {len(late_enough)} of those after link_down (actual ts {link_down_ts:.0f})",
    )

    prober_events = events_by_node[prober]
    discovered = _events_of(prober_events, "probe_discovered")
    if len(discovered) != 2:
        raise CheckUsageError(f"expected exactly 2 probe_discovered events at {prober}, found {len(discovered)}")
    during_discovered, after_discovered = discovered[0], discovered[1]

    c.precondition(
        "during-outage probe actually fired inside the outage window",
        link_down_ts < during_discovered["ts_ms"] < link_up_ts,
        f"probe ts {during_discovered['ts_ms']:.0f}, outage ({link_down_ts:.0f}, {link_up_ts:.0f})",
    )
    c.precondition(
        "after-recovery probe actually fired after link_up",
        after_discovered["ts_ms"] > link_up_ts,
        f"probe ts {after_discovered['ts_ms']:.0f} vs link_up ts {link_up_ts:.0f}",
    )

    during_matches = {(mm["origin"], mm["seq"]) for mm in during_discovered["matches"]}
    c.protocol(
        "during-outage probe discovered exactly the pre-outage records, nothing more",
        during_matches == {(origin, seq) for seq in range(1, n + 1)},
        f"got {sorted(during_matches)}",
    )

    for node in non_origin_nodes:
        stored = _record_stored(events_by_node[node], origin)
        issue = _contiguous_unique([e["seq"] for e in stored], n + m)
        c.protocol(f"{node} holds exactly {n + m} of {origin}'s records after recovery", issue is None, issue or "")

    after_matches = {(mm["origin"], mm["seq"]) for mm in after_discovered["matches"]}
    c.protocol(
        "after-recovery probe discovers all N+M records",
        after_matches == {(origin, seq) for seq in range(1, n + m + 1)},
        f"got {sorted(after_matches)}",
    )

    retrieved_after = _events_of(prober_events, "probe_retrieved")
    # The during-outage retrieval attempt (if any) is expected to fail --
    # map.md: "本体取得の結果は記録するが合否に使わない" -- so only the
    # LAST probe_retrieved (the after-recovery one) is judged.
    c.protocol(
        "after-recovery probe retrieves the content body",
        bool(retrieved_after) and retrieved_after[-1].get("success") is True,
        f"{[r.get('success') for r in retrieved_after]}",
    )

    for node in [origin] + non_origin_nodes:
        conflicts = _events_of(events_by_node[node], "conflict")
        c.protocol(f"{node} logged no conflict", conflicts == [], f"{len(conflicts)} conflict event(s)")

    return c


# -- T2a: fault.drop_rec retry -------------------------------------------


def check_t2a(spec: ExperimentSpec, run_dir: str) -> _Checker:
    """map.md T2(a): a withheld reply delays but does not prevent convergence; the requester retries."""
    c = _Checker()
    if len(spec.fault) != 1:
        raise CheckUsageError("t2a requires fault.drop_rec on exactly one host")
    (holder_host, drop_specs), = spec.fault.items()
    if len(drop_specs) != 1:
        raise CheckUsageError("t2a requires exactly one fault.drop_rec entry")
    drop = drop_specs[0]
    holder = f"h{holder_host}"
    requester_hosts = [h for h in spec.agents if h != holder_host]
    if len(requester_hosts) != 1:
        raise CheckUsageError("t2a requires exactly one agent host besides the holder")
    requester = f"h{requester_hosts[0]}"
    n = len(spec.schedule.get(holder_host, ()))
    if n == 0:
        raise CheckUsageError("t2a requires a schedule on the holder host")

    events_by_node = {f"h{h}": read_events(_node_events_path(run_dir, f"h{h}")) for h in spec.agents}
    requester_events = events_by_node[requester]

    stored = _record_stored(requester_events, drop.origin)
    seen_seqs = {e["seq"] for e in stored}
    c.precondition(
        f"{requester} eventually stored all {n} of {drop.origin}'s records",
        seen_seqs == set(range(1, n + 1)),
        f"got {sorted(seen_seqs)}, expected 1..{n}",
    )

    issue = _contiguous_unique([e["seq"] for e in stored], n)
    c.protocol(f"{requester} holds exactly {n} records, none duplicated", issue is None, issue or "")

    timeouts = _events_of(requester_events, "fetch_timeout", origin=drop.origin, seq=drop.seq)
    c.protocol(
        f"{requester} logged exactly {drop.times} fetch_timeout(s) for ({drop.origin}, {drop.seq})",
        len(timeouts) == drop.times,
        f"got {len(timeouts)}",
    )

    holders_seen = {e["holder"] for e in stored}
    c.protocol(
        f"every record {requester} stored for {drop.origin} came from holder {holder}",
        holders_seen <= {holder},
        f"holders seen: {sorted(holders_seen)}",
    )

    conflicts = _events_of(requester_events, "conflict")
    c.protocol(f"{requester} logged no conflict", conflicts == [], f"{len(conflicts)} conflict event(s)")

    return c


# -- T2b: host_app_restart of the origin ----------------------------------


def check_t2b(spec: ExperimentSpec, run_dir: str) -> _Checker:
    """map.md T2(b): restarting the Index origin mid-schedule must not duplicate or skip its own seqs."""
    c = _Checker()
    origin_hosts = [h for h in spec.schedule if spec.schedule[h]]
    if len(origin_hosts) != 1:
        raise CheckUsageError("t2b requires a schedule on exactly one host")
    origin_host = origin_hosts[0]
    origin = f"h{origin_host}"
    n = len(spec.schedule[origin_host])

    restarts = [e for e in spec.events if e.type == "host_app_restart"]
    if len(restarts) != 1 or restarts[0].fields["host"] != origin_host:
        raise CheckUsageError("t2b requires exactly one host_app_restart event, on the origin host")

    events_by_node = {f"h{h}": read_events(_node_events_path(run_dir, f"h{h}")) for h in spec.agents}
    origin_events = events_by_node[origin]
    other_nodes = [f"h{h}" for h in spec.agents if h != origin_host]

    created_seqs = _record_created_seqs(origin_events, origin)
    c.precondition(
        f"{origin} eventually created all {n} of its own records (across both lives)",
        set(created_seqs) == set(range(1, n + 1)),
        f"got {sorted(set(created_seqs))}, expected 1..{n}",
    )
    for node in other_nodes:
        stored_seqs = {e["seq"] for e in _record_stored(events_by_node[node], origin)}
        c.precondition(
            f"{node} eventually received all {n} of {origin}'s records",
            stored_seqs == set(range(1, n + 1)),
            f"got {sorted(stored_seqs)}, expected 1..{n}",
        )

    issue = _contiguous_unique(created_seqs, n)
    c.protocol(f"{origin}'s own seqs are contiguous and unique across both lives", issue is None, issue or "")

    counters_origin = _events_of(origin_events, "counters")
    c.protocol(
        f"{origin} (restarted) wrote exactly 2 counters lines, one per life",
        len(counters_origin) == 2,
        f"got {len(counters_origin)}",
    )
    for node in other_nodes:
        counters = _events_of(events_by_node[node], "counters")
        c.protocol(f"{node} (not restarted) wrote exactly 1 counters line", len(counters) == 1, f"got {len(counters)}")

    for node in other_nodes:
        issue = _contiguous_unique([e["seq"] for e in _record_stored(events_by_node[node], origin)], n)
        c.protocol(f"{node} holds exactly {n} of {origin}'s records, none duplicated", issue is None, issue or "")

    for node in [origin] + other_nodes:
        conflicts = _events_of(events_by_node[node], "conflict")
        c.protocol(f"{node} logged no conflict", conflicts == [], f"{len(conflicts)} conflict event(s)")

    return c


# -- T2c: host_app_restart of a holder (relay) ----------------------------


def check_t2c(spec: ExperimentSpec, run_dir: str) -> _Checker:
    """map.md T2(c): restarting a relay (not the origin) must never serve progress it had not actually stored."""
    c = _Checker()
    origin_hosts = [h for h in spec.schedule if spec.schedule[h]]
    if len(origin_hosts) != 1:
        raise CheckUsageError("t2c requires a schedule on exactly one host")
    origin_host = origin_hosts[0]
    origin = f"h{origin_host}"
    n = len(spec.schedule[origin_host])

    restarts = [e for e in spec.events if e.type == "host_app_restart"]
    if len(restarts) != 1:
        raise CheckUsageError("t2c requires exactly one host_app_restart event")
    holder_host = restarts[0].fields["host"]
    if holder_host == origin_host:
        raise CheckUsageError("t2c's restarted host must not be the origin (see t2b for that case)")
    holder = f"h{holder_host}"

    far_hosts = [h for h in spec.agents if h not in (origin_host, holder_host)]
    if len(far_hosts) != 1:
        raise CheckUsageError("t2c requires exactly one agent host besides the origin and the restarted holder")
    far = f"h{far_hosts[0]}"

    events_by_node = {f"h{h}": read_events(_node_events_path(run_dir, f"h{h}")) for h in spec.agents}
    holder_events, far_events = events_by_node[holder], events_by_node[far]

    holder_stored = {e["seq"]: e for e in _record_stored(holder_events, origin)}
    far_stored = {e["seq"]: e for e in _record_stored(far_events, origin)}
    c.precondition(
        f"{holder} eventually received all {n} of {origin}'s records (across both lives)",
        set(holder_stored) == set(range(1, n + 1)),
        f"got {sorted(holder_stored)}, expected 1..{n}",
    )
    c.precondition(
        f"{far} eventually received all {n} of {origin}'s records",
        set(far_stored) == set(range(1, n + 1)),
        f"got {sorted(far_stored)}, expected 1..{n}",
    )

    issue = _contiguous_unique(list(holder_stored), n)
    c.protocol(f"{holder} holds exactly {n} of {origin}'s records, none duplicated", issue is None, issue or "")
    issue = _contiguous_unique(list(far_stored), n)
    c.protocol(f"{far} holds exactly {n} of {origin}'s records, none duplicated", issue is None, issue or "")

    # "progress not ahead": far can only have gotten a record from holder
    # (its only neighbor, so every far_stored row's holder is already
    # holder by construction) at or after the moment holder itself
    # actually stored that same seq -- never before, restart or not.
    late = [
        seq
        for seq, far_row in far_stored.items()
        if seq in holder_stored and far_row["ts_ms"] < holder_stored[seq]["ts_ms"]
    ]
    c.protocol(
        f"{holder} never served a record to {far} before {holder} itself had stored it",
        late == [],
        f"seqs served early: {late}",
    )

    counters_holder = _events_of(holder_events, "counters")
    c.protocol(f"{holder} (restarted) wrote exactly 2 counters lines, one per life", len(counters_holder) == 2, f"got {len(counters_holder)}")
    for node in (origin, far):
        counters = _events_of(events_by_node[node], "counters")
        c.protocol(f"{node} (not restarted) wrote exactly 1 counters line", len(counters) == 1, f"got {len(counters)}")

    for node in (origin, holder, far):
        conflicts = _events_of(events_by_node[node], "conflict")
        c.protocol(f"{node} logged no conflict", conflicts == [], f"{len(conflicts)} conflict event(s)")

    return c


_CHECKERS = {"t1": check_t1, "t2a": check_t2a, "t2b": check_t2b, "t2c": check_t2c}


def run_check(mode: str, experiment_path: str, run_dir: str) -> int:
    """Load `experiment_path`, run `mode`'s checker against `run_dir`, and return its exit code."""
    with open(experiment_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    spec = parse_experiment(raw)
    checker = _CHECKERS[mode](spec, run_dir)
    if not checker.precondition_ok:
        return 3
    if not checker.protocol_ok:
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the checker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=_MODES, help="Which experiment's acceptance criteria to check")
    parser.add_argument("experiment", help="Path to the experiment description JSON file this run was generated from")
    parser.add_argument("run_dir", help="The ReCefore experiment directory this run wrote into")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: python3 -m semindex.check <mode> <experiment.json> <run_dir>."""
    args = parse_args(argv)
    try:
        return run_check(args.mode, args.experiment, args.run_dir)
    except (ExperimentError, CheckUsageError) as exc:
        print(f"semindex.check: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
