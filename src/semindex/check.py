"""semindex.check: PASS/FAIL checkers for the T1/T2 acceptance experiments (map.md).

Reads one experiment's description (the same file semindex.gen consumed,
so N/M/hosts/fault counts are never duplicated as separate constants),
every agent host's events.jsonl under a ReCefore run directory, and that
run's results.json, then prints one line per assertion and exits:

    0  every assertion passed
    1  at least one PROTOCOL assertion failed (the sync/fault/restart
       behavior itself is wrong)
    2  usage error (bad arguments, the experiment/run directory do not
       match what this mode expects, or a results.json/events.jsonl row
       is missing a field this check needs)
    3  at least one PRECONDITION assertion failed (the scenario's own
       timing did not hold up -- e.g. records had not finished
       propagating before a link_down actually fired -- so the protocol
       assertions below it were not meaningfully exercised)

When a run has failures in BOTH categories, 1 wins: a protocol failure
must never be hidden behind a precondition exit code just because a
PRECOND-FAIL line also happened to print above it. A trailing SUMMARY
line names that explicitly, so it is visible even to a reader who only
looks at the end of the output.

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


def restart_record_and_ts(results: list[dict], host: int) -> tuple[dict, float]:
    """The one results.json host_app_restart row for `host`, and its actual ts as epoch ms.

    2026-10-22 fix: T2(b)/(c) used to never read this
    row at all, so an overdue host_app_restart (one that only fires after
    every record the scenario cared about had already been created/
    stored) could still pass -- the restart was never actually exercised.
    This row's `ts` is sampled inside ReCefore's ResultsSink.record_event,
    called only after the scheduler's own handler returns; restart_host()
    itself blocks that handler for its whole stop + downtime_s + start
    sequence, so this ts marks when the restart actually FINISHED (the
    node's new life had already started), not when SIGTERM was sent.
    """
    record = find_one_event_record(results, "host_app_restart", host=host)
    return record, results_ts_ms(record)


# -- small query helpers over one node's event list ---------------------------


def _events_of(events: list[dict], event_type: str, **field_filters) -> list[dict]:
    """Every event in `events` whose `event` field is `event_type` and matches every `field_filters` item."""
    return [e for e in events if e.get("event") == event_type and all(e.get(k) == v for k, v in field_filters.items())]


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


def _contiguous_hi(seqs: list[int]) -> int:
    """The highest N such that every seq in 1..N is present in `seqs` (map.md's own "hi" definition)."""
    seq_set = set(seqs)
    hi = 0
    while (hi + 1) in seq_set:
        hi += 1
    return hi


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
    # 2026-10-22 fix: these two rows' own `success` field was never read,
    # so a link_down or link_up that ReCefore itself reported as FAILED
    # (the interface manipulation it wraps did not actually happen) still
    # let every timing check below treat the outage window as real.
    c.precondition(
        "link_down actually succeeded", link_down_record.get("success") is True, f"success={link_down_record.get('success')}"
    )
    c.precondition(
        "link_up actually succeeded", link_up_record.get("success") is True, f"success={link_up_record.get('success')}"
    )

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
    # 2026-10-22 fix: this used to only require ts_ms > link_down_ts, so an
    # M record created at any point afterward -- including well after
    # link_up, once the network had already recovered -- still satisfied
    # it. map.md's own T1 scenario creates every M record strictly inside
    # the outage window, not merely after it started.
    during_outage = [e for e in post_outage_created if link_down_ts < e["ts_ms"] < link_up_ts]
    c.precondition(
        f"all {m} post-outage records were actually created during the outage",
        len(post_outage_created) == m and len(during_outage) == m,
        f"{len(post_outage_created)} created with seq>{n}, {len(during_outage)} of those strictly between "
        f"link_down (ts {link_down_ts:.0f}) and link_up (ts {link_up_ts:.0f})",
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
    # 2026-10-22 fix: this used to be a PRECONDITION,
    # so a retry bug that lost (origin, seq) forever exited 3 ("this
    # run's own setup/timing didn't hold up") instead of 1 ("the sync
    # protocol itself is wrong"). fault.drop_rec withholds a reply a
    # fixed `times` count, not for a timing window this checker could
    # separately verify "landed" or not -- whether the requester ever
    # converges past that withholding *is* the retry protocol under
    # test here, so it is a protocol assertion like every other check
    # in this function, not a precondition.
    c.protocol(
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

    # 2026-10-22 fix: this mode used to never read
    # results.json at all, so an overdue host_app_restart (firing only
    # after every own-record had already been created in life 1) could
    # still pass -- life 2 would then never create anything, and nothing
    # here would have told the difference. restart_record_and_ts's own
    # docstring explains why its ts marks the restart's actual finish.
    results = read_results(run_dir)
    restart_record, restart_ts = restart_record_and_ts(results, origin_host)
    c.precondition(
        f"host_app_restart on {origin} succeeded",
        restart_record.get("success") is True,
        f"success={restart_record.get('success')}",
    )

    created_events = _events_of(origin_events, "record_created", origin=origin)
    created_seqs = [e["seq"] for e in created_events]
    before_restart = [e for e in created_events if e["ts_ms"] < restart_ts]
    after_restart = [e for e in created_events if e["ts_ms"] > restart_ts]
    c.precondition(
        f"{origin} created records on both sides of the actual restart (life 1 and life 2)",
        bool(before_restart) and bool(after_restart),
        f"{len(before_restart)} before, {len(after_restart)} after (restart finished at ts {restart_ts:.0f})",
    )

    # 2026-10-22 fix: these two were PRECONDITIONs, so
    # a restart that duplicated or dropped the origin's own records
    # exited 3 ("didn't get a fair run") instead of 1 ("broke"). Whether
    # every node eventually converges past a restart is exactly the
    # restart-safety property T2(b) exists to test, so it is a protocol
    # assertion like the contiguity/uniqueness checks right below it.
    c.protocol(
        f"{origin} eventually created all {n} of its own records (across both lives)",
        set(created_seqs) == set(range(1, n + 1)),
        f"got {sorted(set(created_seqs))}, expected 1..{n}",
    )
    for node in other_nodes:
        stored_seqs = {e["seq"] for e in _record_stored(events_by_node[node], origin)}
        c.protocol(
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

    # 2026-10-22 fix: this mode used to never read
    # results.json, so an overdue host_app_restart (firing only after
    # everything had already propagated) could still pass; see
    # restart_record_and_ts's own docstring for why its ts marks the
    # restart's actual finish, not the moment SIGTERM was sent.
    results = read_results(run_dir)
    restart_record, restart_ts = restart_record_and_ts(results, holder_host)
    c.precondition(
        f"host_app_restart on {holder} succeeded",
        restart_record.get("success") is True,
        f"success={restart_record.get('success')}",
    )

    # 2026-10-22 fix: kept as the full event list, not
    # collapsed into a {seq: event} dict -- a dict silently drops a
    # duplicate record_stored for the same seq before _contiguous_unique
    # ever sees it, defeating the one check whose entire job is finding
    # duplicates.
    holder_stored_events = _record_stored(holder_events, origin)
    far_stored_events = _record_stored(far_events, origin)
    holder_stored_seqs = {e["seq"] for e in holder_stored_events}
    far_stored_seqs = {e["seq"] for e in far_stored_events}

    holder_before_restart = [e for e in holder_stored_events if e["ts_ms"] < restart_ts]
    holder_after_restart = [e for e in holder_stored_events if e["ts_ms"] > restart_ts]
    c.precondition(
        f"{holder} received records from {origin} on both sides of the actual restart",
        bool(holder_before_restart) and bool(holder_after_restart),
        f"{len(holder_before_restart)} before, {len(holder_after_restart)} after (restart finished at ts {restart_ts:.0f})",
    )
    far_stored_at_restart = {e["seq"] for e in far_stored_events if e["ts_ms"] < restart_ts}
    c.precondition(
        f"{far} had not yet received all of {origin}'s records when the restart actually fired",
        far_stored_at_restart != set(range(1, n + 1)),
        f"{far} already held {sorted(far_stored_at_restart)} of 1..{n} at restart (ts {restart_ts:.0f})",
    )

    # 2026-10-22 fix: these two used to be
    # PRECONDITIONs, so a restarted holder that lost or stalled a record
    # exited 3 instead of 1. Whether holder/far ever converge past the
    # restart is exactly T2(c)'s own protocol property, not scenario setup.
    c.protocol(
        f"{holder} eventually received all {n} of {origin}'s records (across both lives)",
        holder_stored_seqs == set(range(1, n + 1)),
        f"got {sorted(holder_stored_seqs)}, expected 1..{n}",
    )
    c.protocol(
        f"{far} eventually received all {n} of {origin}'s records",
        far_stored_seqs == set(range(1, n + 1)),
        f"got {sorted(far_stored_seqs)}, expected 1..{n}",
    )

    issue = _contiguous_unique([e["seq"] for e in holder_stored_events], n)
    c.protocol(f"{holder} holds exactly {n} of {origin}'s records, none duplicated", issue is None, issue or "")
    issue = _contiguous_unique([e["seq"] for e in far_stored_events], n)
    c.protocol(f"{far} holds exactly {n} of {origin}'s records, none duplicated", issue is None, issue or "")

    # "progress not ahead", first cut: far can only have gotten a record
    # from holder (its only neighbor, so every far_stored row's holder is
    # already holder by construction) at or after the EARLIEST moment
    # holder itself actually stored that same seq -- never before,
    # restart or not. Earliest, not last: a seq holder stored twice (a
    # duplicate, already flagged above) must not let its later timestamp
    # excuse an early serve that happened between the two. This is
    # necessary but not sufficient: it would not catch a life-2 holder
    # advertising a seq it does not actually hold, since handle_interest
    # only ever answers with a record it actually has -- that bug shows
    # up as a fetch_timeout against holder, never as a bad record_stored.
    # The checks below close that gap.
    holder_first_ts_by_seq: dict[int, float] = {}
    for e in holder_stored_events:
        if e["seq"] not in holder_first_ts_by_seq or e["ts_ms"] < holder_first_ts_by_seq[e["seq"]]:
            holder_first_ts_by_seq[e["seq"]] = e["ts_ms"]
    late = [
        e["seq"]
        for e in far_stored_events
        if e["seq"] in holder_first_ts_by_seq and e["ts_ms"] < holder_first_ts_by_seq[e["seq"]]
    ]
    c.protocol(
        f"{holder} never served a record to {far} before {holder} itself had stored it",
        late == [],
        f"seqs served early: {late}",
    )

    counters_holder = _events_of(holder_events, "counters")

    # 2026-10-22 fix: "post-restart" used to be decided against holder's
    # own first `counters` line (its life-1 shutdown), not against
    # restart_ts. far can log a state_rx from holder DURING the downtime
    # itself (a reply already in flight when holder went down) whose ts
    # still lands after that counters line if the restart happens to
    # finish quickly -- that stale, often-honest reply was then the ONLY
    # one ever judged (index 0 of the old sorted list), so a genuinely
    # dishonest advertisement sent once life 2 actually started went
    # unexamined. restart_ts (results.json's own host_app_restart row,
    # which restart_record_and_ts's docstring says marks the restart's
    # actual FINISH) is sampled by ReCefore's own scheduler, not by
    # anything holder or far wrote, so it is the only boundary here that
    # cannot itself be stale.
    post_restart_state_rx = sorted(
        (e for e in _events_of(far_events, "state_rx", neighbor=holder) if e["ts_ms"] > restart_ts),
        key=lambda e: e["ts_ms"],
    )
    c.precondition(f"{far} heard from {holder} again after its restart (state_rx)", bool(post_restart_state_rx))

    # 2026-10-22 fix: judge EVERY post-restart state_rx, not just the
    # first -- a holder whose first reply after coming back up is honest
    # but whose next one overstates its progress was invisible to a check
    # that only ever looked at the first row.
    for state_rx in post_restart_state_rx:
        state_rx_ts = state_rx["ts_ms"]

        # The actual bug shape -- a life-2 holder advertising a higher hi
        # than it has actually stored -- is checked directly against what
        # the holder's own events.jsonl shows it had stored as of this
        # same reply, instead of only inferred indirectly through a
        # fetch_timeout that may or may not follow. A reply that never
        # mentions `origin` at all (decode failure -> `{}`, or a reply
        # sent before any progress existed to advertise) is a
        # PRECONDITION gap, not a passing advertised_hi=0: defaulting
        # silently would make this whole check a no-op for exactly the
        # malformed/empty-reply runs it most needs to catch.
        advertised = state_rx.get("hi", {})
        have_advertised_hi = c.precondition(
            f"{far}'s post-restart state_rx from {holder} at ts {state_rx_ts:.0f} actually advertised a hi for {origin}",
            origin in advertised,
            f"hi={advertised!r}",
        )
        if have_advertised_hi:
            advertised_hi = advertised[origin]
            holder_contiguous_as_of = _contiguous_hi(
                [e["seq"] for e in holder_stored_events if e["ts_ms"] <= state_rx_ts]
            )
            c.protocol(
                f"{holder}'s advertised hi for {origin} in its post-restart state_rx at ts {state_rx_ts:.0f} never exceeds what it had actually stored by then",
                advertised_hi <= holder_contiguous_as_of,
                f"advertised hi={advertised_hi}, holder's own contiguous progress at ts {state_rx_ts:.0f} was {holder_contiguous_as_of}",
            )

    # "progress not ahead", the second-order case: a false advertisement
    # whose seq the holder (now caught up) manages to serve correctly
    # before being caught above would still show up here too, through the
    # retries it costs -- 2500ms grace (interest_lifetime_ms's own 2000ms
    # default, plus slack for one retry sent just as holder came back up)
    # excludes timeouts that only mean holder was still down, not yet
    # evidence of a false advertisement. Anchored on restart_ts itself,
    # not on the first state_rx row (which may arrive late, or not at
    # all), so this check still runs even when the precondition above
    # already failed.
    grace_deadline = restart_ts + 2500
    late_timeouts = [
        e
        for e in _events_of(far_events, "fetch_timeout", origin=origin, neighbor=holder)
        if e["ts_ms"] > grace_deadline
    ]
    c.protocol(
        f"{holder}'s life 2 never advertised a record it could not actually serve to {far}",
        late_timeouts == [],
        f"fetch_timeout(s) after grace: {late_timeouts}",
    )

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
    # 2026-10-22 fix: precondition was checked first, so a run with BOTH
    # a PRECOND-FAIL and a FAIL line exited 3 outright -- reading as "this
    # run's own timing/setup did not hold up," which can hide a real
    # protocol failure sitting right above it in the same output. 1 now
    # wins whenever both categories failed, and the SUMMARY line below
    # names that explicitly so it is not just a changed exit code with
    # nothing printed to explain it.
    if not checker.protocol_ok and not checker.precondition_ok:
        print("SUMMARY: a protocol failure occurred (see the FAIL line(s) above; some PRECOND-FAIL line(s) also printed)")
        return 1
    if not checker.protocol_ok:
        return 1
    if not checker.precondition_ok:
        return 3
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
    except KeyError as exc:
        # 2026-10-22 fix: a results.json row missing `ts` or an
        # events.jsonl row missing a field a checker reads unconditionally
        # (e.g. `seq`) used to raise an unhandled KeyError traceback
        # instead of the same exit-2-with-a-message every other
        # malformed-input case here gets.
        print(f"semindex.check: missing required field: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
