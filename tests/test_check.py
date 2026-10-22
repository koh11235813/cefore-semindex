"""Tests for the semindex.check checkers, against synthetic events.jsonl/results.json fixtures."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from semindex.check import (
    CheckUsageError,
    _contiguous_unique,
    find_one_event_record,
    find_one_put_record,
    read_events,
    results_ts_ms,
    run_check,
)

# A fixed epoch base for every synthetic timestamp in this file, so offsets
# below are easy to read as "seconds into the run" while still producing
# real epoch values results_ts_ms/ts_ms comparisons need.
_BASE = 1_700_000_000.0


def _ts_ms(offset_s: float) -> int:
    """events.jsonl's own `ts_ms` convention (EventWriter: round(time.time() * 1000))."""
    return round((_BASE + offset_s) * 1000)


def _iso(offset_s: float) -> str:
    """results.json's own `ts` convention (timestamp_utc(): datetime.now(timezone.utc).isoformat())."""
    return datetime.fromtimestamp(_BASE + offset_s, tz=timezone.utc).isoformat()


def _write_jsonl(path: Path, events: list[dict]) -> None:
    """Write `events` to `path` as one JSON object per line, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def _write_experiment(path: Path, raw: dict) -> None:
    """Write an experiment description dict to `path` as JSON."""
    path.write_text(json.dumps(raw), encoding="utf-8")


def _node_events_path(run_dir: Path, node: str) -> Path:
    """Where semindex.check itself expects to find one node's archived events.jsonl under `run_dir`."""
    return run_dir / "debug" / "node_dirs" / node / "semindex" / "events.jsonl"


def _write_results(run_dir: Path, records: list[dict]) -> None:
    """Write a synthetic results.json (a plain list of record dicts) under `run_dir`."""
    (run_dir / "results.json").write_text(json.dumps(records), encoding="utf-8")


# -- small helpers -------------------------------------------------------


def test_contiguous_unique_accepts_an_exact_contiguous_set():
    """Contiguous unique accepts an exact contiguous set."""
    assert _contiguous_unique([1, 2, 3], 3) is None


def test_contiguous_unique_reports_a_duplicate():
    """Contiguous unique reports a duplicate."""
    assert _contiguous_unique([1, 1, 2], 2) == "duplicates present: [1]"


def test_contiguous_unique_reports_a_gap():
    """Contiguous unique reports a gap."""
    assert _contiguous_unique([1, 3], 3) == "got [1, 3], expected 1..3"


def test_results_ts_ms_round_trips_timestamp_utc_format():
    """Results ts ms round trips timestamp utc format."""
    iso = _iso(12.5)
    assert results_ts_ms({"ts": iso}) == pytest.approx((_BASE + 12.5) * 1000.0, abs=1.0)


def test_find_one_event_record_raises_on_zero_matches():
    """Find one event record raises on zero matches."""
    with pytest.raises(CheckUsageError) as exc_info:
        find_one_event_record([], "link_down", nodes=[0, 1])
    assert str(exc_info.value) == (
        "expected exactly one results.json link_down row matching {'nodes': [0, 1]}, found 0"
    )


def test_find_one_event_record_raises_on_multiple_matches():
    """Find one event record raises on multiple matches."""
    row = {"op_type": "event", "event_type": "link_down", "event": {"nodes": [0, 1]}}
    with pytest.raises(CheckUsageError) as exc_info:
        find_one_event_record([row, row], "link_down", nodes=[0, 1])
    assert str(exc_info.value) == (
        "expected exactly one results.json link_down row matching {'nodes': [0, 1]}, found 2"
    )


def test_find_one_put_record_matches_by_uri():
    """Find one put record matches by uri."""
    rows = [
        {"op_type": "put", "uri": "ccnx:/content/h0/body", "success": True},
        {"op_type": "put", "uri": "ccnx:/content/other", "success": True},
    ]
    assert find_one_put_record(rows, "ccnx:/content/h0/body")["success"] is True


def test_read_events_raises_a_usage_error_on_an_unparsable_line(tmp_path: Path):
    """Read events raises a usage error on an unparsable line."""
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "record_created"}\n{truncated\n', encoding="utf-8")
    try:
        json.loads("{truncated")
    except json.JSONDecodeError as parse_exc:
        expected_detail = parse_exc
    with pytest.raises(CheckUsageError) as exc_info:
        read_events(str(path))
    assert str(exc_info.value) == f"{path} line 2 unparsable: {expected_detail}"


def test_read_events_raises_a_usage_error_on_a_missing_file(tmp_path: Path):
    """Read events raises a usage error on a missing file."""
    path = tmp_path / "nope.jsonl"
    with pytest.raises(CheckUsageError) as exc_info:
        read_events(str(path))
    assert str(exc_info.value) == f"{path} does not exist"


# -- T1 -------------------------------------------------------------------


def _t1_experiment() -> dict:
    """A small T1-shaped experiment description: line h0-h1-h2, 2 pre-outage + 2 post-outage records."""
    return {
        "name": "t1mini",
        "num": 1,
        "seed": 1,
        "group": "g",
        "hosts": 3,
        "links": [[0, 1], [1, 2]],
        "duration": 60,
        "output_dir": "logs/mini",
        "agents": [0, 1, 2],
        "publisher": {
            "host": 0,
            "uri": "ccnx:/content/h0/body",
            "file": "./sample-putfile",
            "expiry": 3000,
            "cache_time": 3000,
        },
        "schedule": {
            "0": [
                {"at_s": 3, "zone": "zone1", "disaster": "flood", "media": "image"},
                {"at_s": 5, "zone": "zone1", "disaster": "flood", "media": "image"},
                {"at_s": 22, "zone": "zone1", "disaster": "flood", "media": "image"},
                {"at_s": 24, "zone": "zone1", "disaster": "flood", "media": "image"},
            ]
        },
        "probes": {"2": [{"at_s": 25, "zone": "zone1"}, {"at_s": 50, "zone": "zone1"}]},
        "events": [
            {"at": 20, "type": "link_down", "nodes": [0, 1]},
            {"at": 40, "type": "link_up", "nodes": [0, 1]},
        ],
    }


def _write_t1_happy_path(run_dir: Path) -> None:
    """A full, consistent T1 run: 2 pre-outage + 2 post-outage records, clean recovery."""
    _write_jsonl(
        _node_events_path(run_dir, "h0"),
        [
            {"ts_ms": _ts_ms(3), "event": "record_created", "origin": "h0", "seq": 1},
            {"ts_ms": _ts_ms(5), "event": "record_created", "origin": "h0", "seq": 2},
            {"ts_ms": _ts_ms(12), "event": "record_created", "origin": "h0", "seq": 3},
            {"ts_ms": _ts_ms(14), "event": "record_created", "origin": "h0", "seq": 4},
            {"ts_ms": _ts_ms(50), "event": "counters", "control_bytes_sent": 20, "control_bytes_received": 0},
        ],
    )
    _write_jsonl(
        _node_events_path(run_dir, "h1"),
        [
            {"ts_ms": _ts_ms(4), "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h0"},
            {"ts_ms": _ts_ms(6), "event": "record_stored", "origin": "h0", "seq": 2, "holder": "h0"},
            {"ts_ms": _ts_ms(32), "event": "record_stored", "origin": "h0", "seq": 3, "holder": "h0"},
            {"ts_ms": _ts_ms(34), "event": "record_stored", "origin": "h0", "seq": 4, "holder": "h0"},
            {"ts_ms": _ts_ms(50), "event": "counters", "control_bytes_sent": 10, "control_bytes_received": 10},
        ],
    )
    _write_jsonl(
        _node_events_path(run_dir, "h2"),
        [
            {"ts_ms": _ts_ms(6), "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h1"},
            {"ts_ms": _ts_ms(8), "event": "record_stored", "origin": "h0", "seq": 2, "holder": "h1"},
            {
                "ts_ms": _ts_ms(16),
                "event": "probe_discovered",
                "matches": [
                    {"origin": "h0", "seq": 1, "content_name": "ccnx:/content/h0/body"},
                    {"origin": "h0", "seq": 2, "content_name": "ccnx:/content/h0/body"},
                ],
            },
            {
                "ts_ms": _ts_ms(18),
                "event": "probe_retrieved",
                "origin": "h0",
                "seq": 1,
                "content_name": "ccnx:/content/h0/body",
                "success": False,
                "bytes": 0,
            },
            {"ts_ms": _ts_ms(36), "event": "record_stored", "origin": "h0", "seq": 3, "holder": "h1"},
            {"ts_ms": _ts_ms(38), "event": "record_stored", "origin": "h0", "seq": 4, "holder": "h1"},
            {
                "ts_ms": _ts_ms(45),
                "event": "probe_discovered",
                "matches": [
                    {"origin": "h0", "seq": s, "content_name": "ccnx:/content/h0/body"} for s in (1, 2, 3, 4)
                ],
            },
            {
                "ts_ms": _ts_ms(46),
                "event": "probe_retrieved",
                "origin": "h0",
                "seq": 1,
                "content_name": "ccnx:/content/h0/body",
                "success": True,
                "bytes": 433390,
            },
            {"ts_ms": _ts_ms(50), "event": "counters", "control_bytes_sent": 5, "control_bytes_received": 15},
        ],
    )
    _write_results(
        run_dir,
        [
            {"op_type": "put", "uri": "ccnx:/content/h0/body", "success": True},
            {"op_type": "event", "event_type": "link_down", "ts": _iso(10), "success": True, "event": {"nodes": [0, 1]}},
            {"op_type": "event", "event_type": "link_up", "ts": _iso(30), "success": True, "event": {"nodes": [0, 1]}},
        ],
    )


def test_check_t1_all_pass(tmp_path: Path):
    """Check t1 all pass."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)

    assert run_check("t1", str(experiment_path), str(run_dir)) == 0


def test_check_t1_protocol_fails_when_the_only_surviving_retrieval_is_the_during_outage_one(
    tmp_path: Path, capsys
):
    """2026-10-22 fix: a stale during-outage probe_retrieved row, if it happens
    to be the chronologically last one on file, used to satisfy the after-recovery check all by
    itself once its own `success` was true -- even with the real after-recovery retrieval (offset
    46 in the happy path) missing entirely. Drops that row and flips the during-outage one
    (offset 18, whose own body fetch map.md says must never count toward PASS/FAIL) to success."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    events = [e for e in events if not (e.get("event") == "probe_retrieved" and e["ts_ms"] == _ts_ms(46))]
    for event in events:
        if event.get("event") == "probe_retrieved" and event["ts_ms"] == _ts_ms(18):
            event["success"] = True
    _write_jsonl(h2_path, events)

    rc = run_check("t1", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    # 2026-10-23 fix (review finding): pairing is by the
    # literal next ROW now (see the fix comment above retrieved_after in check.py), so once the
    # real retrieval is gone, the row right after the after-recovery discovery is h2's own
    # shutdown `counters` line -- shown here, not a generic "nothing followed" message.
    assert (
        "FAIL: after-recovery probe retrieves the content body: "
        "{'ts_ms': 1700000050000, 'event': 'counters', 'control_bytes_sent': 5, 'control_bytes_received': 15}"
    ) in out.splitlines()


def test_check_t1_precondition_fails_when_far_node_has_not_converged_before_link_down(tmp_path: Path, capsys):
    """Check t1 precondition fails when far node has not converged before link down."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    # Move h2's second pre-outage record_stored to AFTER the actual link_down
    # time (10s) -- the margin this experiment budgeted did not hold.
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("event") == "record_stored" and event.get("seq") == 2:
            event["ts_ms"] = _ts_ms(11)
    _write_jsonl(h2_path, events)

    rc = run_check("t1", str(experiment_path), str(run_dir))

    assert rc == 3
    out = capsys.readouterr().out
    assert (
        "PRECOND-FAIL: all 2 pre-outage records reached h2 before link_down actually fired: "
        "1/2 stored before link_down (actual ts 1700000010000)"
    ) in out.splitlines()


def test_check_t1_protocol_fails_when_during_probe_discovers_too_much(tmp_path: Path, capsys):
    """Check t1 protocol fails when during probe discovers too much."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("event") == "probe_discovered" and event["ts_ms"] == _ts_ms(16):
            event["matches"].append({"origin": "h0", "seq": 3, "content_name": "ccnx:/content/h0/body"})
    _write_jsonl(h2_path, events)

    rc = run_check("t1", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: during-outage probe discovered exactly the pre-outage records, nothing more: "
        "got [('h0', 1), ('h0', 2), ('h0', 3)]"
    ) in out.splitlines()


def test_check_t1_precondition_fails_when_post_outage_records_are_created_after_link_up(tmp_path: Path, capsys):
    """2026-10-22 fix: a post-outage (M) record used to satisfy this
    precondition as long as it was created any time after link_down -- including well after
    link_up, once the network had already recovered. Moves both of h0's M records to offsets 31
    and 33, after link_up (30s) instead of inside the outage window (10s, 30s)."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    h0_path = _node_events_path(run_dir, "h0")
    events = [json.loads(line) for line in h0_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("event") == "record_created" and event.get("seq") == 3:
            event["ts_ms"] = _ts_ms(31)
        if event.get("event") == "record_created" and event.get("seq") == 4:
            event["ts_ms"] = _ts_ms(33)
    _write_jsonl(h0_path, events)

    rc = run_check("t1", str(experiment_path), str(run_dir))

    assert rc == 3
    out = capsys.readouterr().out
    assert (
        "PRECOND-FAIL: all 2 post-outage records were actually created during the outage: "
        "2 created with seq>2, 0 of those strictly between link_down (ts 1700000010000) and link_up (ts 1700000030000)"
    ) in out.splitlines()


def test_check_t1_precondition_fails_when_link_down_fires_after_the_during_probe_by_actual_time(
    tmp_path: Path, capsys
):
    """Locks in a check that already behaves correctly: timing here is
    judged by results.json's own actual fire time, never the experiment's nominal `at:`. Moving
    link_down's actual ts to offset 17 (after the during-outage probe's own log at offset 16)
    must flag both the outage-window precondition on the M records and the probe-timing
    precondition -- not silently pass because the nominal schedule still looks fine."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    for row in results:
        if row.get("event_type") == "link_down":
            row["ts"] = _iso(17)
    results_path.write_text(json.dumps(results), encoding="utf-8")

    rc = run_check("t1", str(experiment_path), str(run_dir))

    assert rc == 3
    out = capsys.readouterr().out.splitlines()
    assert (
        "PRECOND-FAIL: all 2 post-outage records were actually created during the outage: "
        "2 created with seq>2, 0 of those strictly between link_down (ts 1700000017000) and link_up (ts 1700000030000)"
    ) in out
    assert (
        "PRECOND-FAIL: during-outage probe actually fired inside the outage window: "
        "probe ts 1700000016000, outage (1700000017000, 1700000030000)"
    ) in out


def test_check_t1_precondition_fails_when_link_down_or_link_up_did_not_actually_succeed(tmp_path: Path, capsys):
    """2026-10-22 fix: link_down/link_up rows' own `success` field was
    never read, so ReCefore reporting the interface manipulation itself as FAILED still let
    every timing check in this mode treat the outage window as real."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    for row in results:
        if row.get("event_type") in ("link_down", "link_up"):
            row["success"] = False
    results_path.write_text(json.dumps(results), encoding="utf-8")

    rc = run_check("t1", str(experiment_path), str(run_dir))

    assert rc == 3
    out = capsys.readouterr().out.splitlines()
    assert "PRECOND-FAIL: link_down actually succeeded: success=False" in out
    assert "PRECOND-FAIL: link_up actually succeeded: success=False" in out


def test_check_t1_protocol_fails_on_a_wrong_target_after_recovery_retrieval(tmp_path: Path, capsys):
    """2026-10-23 fix: the after-recovery retrieval check used to accept ANY
    later probe_retrieved row with success=True, with no tie to the discovery it is supposed to
    be the body fetch for. Replaces the real retrieval (offset 46, origin h0 seq 1), IN PLACE so
    it is still the immediate next row after the after-recovery discovery, with one naming a
    completely different (origin, seq, content_name) that still reports success."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for i, event in enumerate(events):
        if event.get("event") == "probe_retrieved" and event["ts_ms"] == _ts_ms(46):
            events[i] = {
                "ts_ms": _ts_ms(46),
                "event": "probe_retrieved",
                "origin": "wrong",
                "seq": 99,
                "content_name": "ccnx:/wrong",
                "success": True,
                "bytes": 5,
            }
    _write_jsonl(h2_path, events)

    rc = run_check("t1", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: after-recovery probe retrieves the content body: "
        "{'ts_ms': 1700000046000, 'event': 'probe_retrieved', 'origin': 'wrong', 'seq': 99, "
        "'content_name': 'ccnx:/wrong', 'success': True, 'bytes': 5}"
    ) in out.splitlines()


def test_check_t1_all_pass_when_discovery_and_retrieval_share_one_ts_ms(tmp_path: Path):
    """2026-10-23 fix: EventWriter rounds ts_ms to whole milliseconds, so a
    fast in-process retrieval can legitimately land on the exact same ts_ms as its own
    discovery. Pairing by file order (not by a strict `>` on ts_ms) must still pass this
    otherwise-honest run."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("event") == "probe_retrieved" and event["ts_ms"] == _ts_ms(46):
            event["ts_ms"] = _ts_ms(45)
    _write_jsonl(h2_path, events)

    assert run_check("t1", str(experiment_path), str(run_dir)) == 0


def test_check_t1_protocol_fails_when_an_intervening_row_separates_the_retrieval_from_its_discovery(
    tmp_path: Path, capsys
):
    """2026-10-23 fix (review finding): the previous fix
    for the wrong-target case above still scanned forward with `next(...)` for the next
    probe_retrieved EVENT, skipping any intervening row that is not itself one -- a reviewer
    reproduced this by moving the REAL after-recovery retrieval (offset 46, the correct target)
    past h2's own shutdown `counters` line (offset 50) instead of leaving it immediately after
    the discovery. Both flags this check reads stayed true; the retrieval simply had no adjacent
    discovery any more. Pairing by the literal next ROW, not the next matching event, must catch
    this."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    retrieved = next(e for e in events if e.get("event") == "probe_retrieved" and e["ts_ms"] == _ts_ms(46))
    events.remove(retrieved)
    events.append(retrieved)  # now follows the counters line instead of the discovery
    _write_jsonl(h2_path, events)

    rc = run_check("t1", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: after-recovery probe retrieves the content body: "
        "{'ts_ms': 1700000050000, 'event': 'counters', 'control_bytes_sent': 5, 'control_bytes_received': 15}"
    ) in out.splitlines()


def test_check_t1_precondition_fails_when_m_records_leak_to_relay_and_far_during_the_outage(
    tmp_path: Path, capsys
):
    """2026-10-23 fix: nothing checked that the partition actually held for
    the M records THEMSELVES -- only that the during-outage probe fired inside the window and
    that the M records were created inside it. Moves h1's and h2's M-record stores to inside the
    outage window (10..30s), after the during-outage probe (offset 16) so the probe-based check
    cannot see the leak."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    for node, holder, base in (("h1", "h0", 17), ("h2", "h1", 18)):
        path = _node_events_path(run_dir, node)
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        events = [e for e in events if not (e.get("event") == "record_stored" and e["seq"] in (3, 4))]
        events.append({"ts_ms": _ts_ms(base), "event": "record_stored", "origin": "h0", "seq": 3, "holder": holder})
        events.append(
            {"ts_ms": _ts_ms(base + 2), "event": "record_stored", "origin": "h0", "seq": 4, "holder": holder}
        )
        _write_jsonl(path, events)

    rc = run_check("t1", str(experiment_path), str(run_dir))

    assert rc == 3
    out = capsys.readouterr().out
    assert (
        "PRECOND-FAIL: no node stored a post-outage (seq > 2) record strictly inside the outage window: "
        "leaked: [('h1', 3, 1700000017000), ('h1', 4, 1700000019000), "
        "('h2', 3, 1700000018000), ('h2', 4, 1700000020000)]"
    ) in out.splitlines()


def test_check_t1_protocol_fails_when_far_claims_every_record_came_straight_from_the_origin(
    tmp_path: Path, capsys
):
    """2026-10-23 fix: holder labels were never asserted in T1 at all. h2 is
    not even linked to h0 (the links are h0-h1-h2), so every h2 record_stored row claiming
    holder=h0 means the relay (h1) was bypassed entirely."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("event") == "record_stored":
            event["holder"] = "h0"
    _write_jsonl(h2_path, events)

    rc = run_check("t1", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL: every record h2 stored for h0 came from holder h1: holders seen: ['h0']" in out.splitlines()


def test_check_t1_protocol_fails_when_far_stores_a_record_before_the_relay_itself_did(
    tmp_path: Path, capsys
):
    """2026-10-23 fix: nothing enforced that h2 (far) never holds a record
    before h1 (the relay) itself had stored it -- the same "progress not ahead" property
    check_t2c already enforces between a restarted holder and far. Delays every one of h1's own
    stores well past h2's (unaffected) ones."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    new_ts = {1: _ts_ms(50), 2: _ts_ms(51), 3: _ts_ms(52), 4: _ts_ms(53)}
    for event in events:
        if event.get("event") == "record_stored":
            event["ts_ms"] = new_ts[event["seq"]]
    _write_jsonl(h1_path, events)

    rc = run_check("t1", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h2 never stored a record before h1 itself had stored it: seqs stored early: [1, 2, 3, 4]"
    ) in out.splitlines()


def test_check_t1_protocol_fails_when_after_probe_matches_name_a_stale_content(tmp_path: Path, capsys):
    """2026-10-23 fix: the after-recovery discovery's own matches were never
    checked against what far_node actually stored or against the publisher's own content name --
    only the (origin, seq) set was compared to the spec's N+M count. Every match here names a
    stale/other content instead of the publisher's own body."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("event") == "probe_discovered" and event["ts_ms"] == _ts_ms(45):
            for match in event["matches"]:
                match["content_name"] = "ccnx:/stale/other"
        if event.get("event") == "probe_retrieved" and event["ts_ms"] == _ts_ms(46):
            event["content_name"] = "ccnx:/stale/other"
    _write_jsonl(h2_path, events)

    rc = run_check("t1", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: after-recovery probe's matches are all records h2 actually stored, naming the "
        "publisher's own content: matched seqs h2 never actually stored: [], content name(s) "
        "other than 'ccnx:/content/h0/body': ['ccnx:/stale/other']"
    ) in out.splitlines()


def test_check_t1_precondition_fails_when_a_node_never_wrote_a_counters_line(tmp_path: Path, capsys):
    """2026-10-23 fix: a node SIGKILLed before its own shutdown finished
    never writes a `counters` line; T1 never checked for this at all before. Drops h0's own line
    only -- h1 and h2 still have theirs."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    h0_path = _node_events_path(run_dir, "h0")
    events = [json.loads(line) for line in h0_path.read_text(encoding="utf-8").splitlines()]
    events = [e for e in events if e.get("event") != "counters"]
    _write_jsonl(h0_path, events)

    rc = run_check("t1", str(experiment_path), str(run_dir))

    assert rc == 3
    out = capsys.readouterr().out
    assert (
        "PRECOND-FAIL: h0 wrote a counters line before this run ended: "
        "h0 wrote no counters line (shutdown did not finish)"
    ) in out.splitlines()


# -- T2a --------------------------------------------------------------------


def _t2a_experiment() -> dict:
    """A small T2(a)-shaped experiment description: holder h0 withholds seq 2 twice for requester h1."""
    return {
        "name": "t2amini",
        "num": 2,
        "seed": 1,
        "group": "g",
        "hosts": 3,
        "links": [[0, 1], [1, 2]],
        "duration": 30,
        "output_dir": "logs/mini",
        "agents": [0, 1],
        "publisher": {
            "host": 0,
            "uri": "ccnx:/content/h0/body",
            "file": "./sample-putfile",
            "expiry": 3000,
            "cache_time": 3000,
        },
        "schedule": {
            "0": [
                {"at_s": 3, "zone": "zone1", "disaster": "flood", "media": "image"},
                {"at_s": 5, "zone": "zone1", "disaster": "flood", "media": "image"},
                {"at_s": 7, "zone": "zone1", "disaster": "flood", "media": "image"},
            ]
        },
        "fault": {"0": {"drop_rec": [{"origin": "h0", "seq": 2, "times": 2}]}},
    }


def _write_t2a_happy_path(run_dir: Path) -> None:
    """A full, consistent T2(a) run: 2 fetch_timeouts for seq 2, then a clean 1..3 convergence."""
    _write_jsonl(
        _node_events_path(run_dir, "h0"),
        [{"ts_ms": _ts_ms(20), "event": "counters", "control_bytes_sent": 5, "control_bytes_received": 5}],
    )
    _write_jsonl(
        _node_events_path(run_dir, "h1"),
        [
            {"ts_ms": _ts_ms(4), "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h0"},
            {"ts_ms": _ts_ms(6), "event": "fetch_timeout", "origin": "h0", "seq": 2, "neighbor": "h0"},
            {"ts_ms": _ts_ms(9), "event": "fetch_timeout", "origin": "h0", "seq": 2, "neighbor": "h0"},
            {"ts_ms": _ts_ms(14), "event": "record_stored", "origin": "h0", "seq": 2, "holder": "h0"},
            {"ts_ms": _ts_ms(16), "event": "record_stored", "origin": "h0", "seq": 3, "holder": "h0"},
            {"ts_ms": _ts_ms(20), "event": "counters", "control_bytes_sent": 5, "control_bytes_received": 5},
        ],
    )


def test_check_t2a_all_pass(tmp_path: Path):
    """Check t2a all pass."""
    experiment_path = tmp_path / "t2a.json"
    _write_experiment(experiment_path, _t2a_experiment())
    run_dir = tmp_path / "run"
    _write_t2a_happy_path(run_dir)

    assert run_check("t2a", str(experiment_path), str(run_dir)) == 0


def test_check_t2a_protocol_fails_on_the_wrong_timeout_count(tmp_path: Path, capsys):
    """Check t2a protocol fails on the wrong timeout count."""
    experiment_path = tmp_path / "t2a.json"
    _write_experiment(experiment_path, _t2a_experiment())
    run_dir = tmp_path / "run"
    _write_t2a_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    events = [e for e in events if not (e.get("event") == "fetch_timeout" and e["ts_ms"] == _ts_ms(9))]
    _write_jsonl(h1_path, events)

    rc = run_check("t2a", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL: h1 logged exactly 2 fetch_timeout(s) for (h0, 2): got 1" in out.splitlines()


def test_check_t2a_protocol_fails_when_never_converges(tmp_path: Path, capsys):
    """2026-10-22 fix: a requester that never converges past fault.drop_rec's
    withheld replies is a protocol failure (exit 1), not a precondition failure (exit 3) --
    the retry protocol itself is what T2(a) exists to test."""
    experiment_path = tmp_path / "t2a.json"
    _write_experiment(experiment_path, _t2a_experiment())
    run_dir = tmp_path / "run"
    _write_jsonl(_node_events_path(run_dir, "h0"), [])
    _write_jsonl(
        _node_events_path(run_dir, "h1"),
        [{"ts_ms": _ts_ms(4), "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h0"}],
    )

    rc = run_check("t2a", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL: h1 eventually stored all 3 of h0's records: got [1], expected 1..3" in out.splitlines()


def test_check_t2a_protocol_fails_when_the_dropped_seq_was_stored_before_its_own_timeout(
    tmp_path: Path, capsys
):
    """2026-10-23 fix: nothing checked that the dropped seq's own store
    landed AFTER its fetch_timeout(s). Moves it 3s BEFORE its only pair of timeouts (both its
    ts_ms AND its file position, as a real events.jsonl -- appended in execution order -- always
    has the two agree), which means the "timeout" was never real evidence the reply had actually
    been withheld."""
    experiment_path = tmp_path / "t2a.json"
    _write_experiment(experiment_path, _t2a_experiment())
    run_dir = tmp_path / "run"
    _write_t2a_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    store_event = next(e for e in events if e.get("event") == "record_stored" and e.get("seq") == 2)
    store_event["ts_ms"] = _ts_ms(6) - 3000
    events.remove(store_event)
    events.insert(1, store_event)  # ahead of both fetch_timeout rows, not just an earlier ts_ms
    _write_jsonl(h1_path, events)

    rc = run_check("t2a", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h1's eventual store of (h0, 2) came after its fetch_timeout(s), not before: "
        "stored at [1700000003000], timed out at [1700000006000, 1700000009000]"
    ) in out.splitlines()


def test_check_t2a_protocol_fails_when_the_dropped_seq_is_stored_immediately_before_its_final_timeout_at_the_same_ts_ms(
    tmp_path: Path, capsys
):
    """2026-10-23 fix (review finding): the check
    above used to compare ts_ms with a strict `<`, so a store and its final timeout rounded to
    the exact same millisecond (EventWriter rounds ts_ms to whole milliseconds) read as "not
    before" even when the store's row in h1's own events.jsonl -- one process's own log, appended
    in true execution order -- came immediately before that timeout. Moves the store to sit right
    before the final timeout in file order and gives both rows the final timeout's own ts_ms."""
    experiment_path = tmp_path / "t2a.json"
    _write_experiment(experiment_path, _t2a_experiment())
    run_dir = tmp_path / "run"
    _write_t2a_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    store_event = next(e for e in events if e.get("event") == "record_stored" and e.get("seq") == 2)
    store_event["ts_ms"] = _ts_ms(9)
    events.remove(store_event)
    final_timeout_idx = max(
        i for i, e in enumerate(events) if e.get("event") == "fetch_timeout" and e.get("seq") == 2
    )
    events.insert(final_timeout_idx, store_event)  # immediately before the final timeout, same ts_ms
    _write_jsonl(h1_path, events)

    rc = run_check("t2a", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h1's eventual store of (h0, 2) came after its fetch_timeout(s), not before: "
        "stored at [1700000009000], timed out at [1700000006000, 1700000009000]"
    ) in out.splitlines()


def test_check_t2a_protocol_fails_when_the_dropped_seqs_store_ts_contradicts_its_later_file_position(
    tmp_path: Path, capsys
):
    """2026-10-23 fix (review finding): the check
    above switched from ts_ms to file-position ordering (see the 2026-10-23 fix comment right
    above it) because EventWriter rounds ts_ms to whole milliseconds, which can make a store and
    its final timeout tie even when file order disagreed. But file position alone then missed the
    opposite contradiction: a store row whose own ts_ms is unambiguously EARLIER than its
    fetch_timeout(s) (no rounding tie at all) still read as "after" just because it happened to
    sit later in the file. events.jsonl's seq-2 store keeps its real, unmoved file position (after
    both fetch_timeouts, same as the happy path) but its ts_ms is rewritten to 3s -- 3s before the
    first of those timeouts -- which the old index-only check waved through as a real withheld
    reply, even though no genuine run could log a store before its own timeout."""
    experiment_path = tmp_path / "t2a.json"
    _write_experiment(experiment_path, _t2a_experiment())
    run_dir = tmp_path / "run"
    _write_t2a_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    store_event = next(e for e in events if e.get("event") == "record_stored" and e.get("seq") == 2)
    store_event["ts_ms"] = _ts_ms(3)  # contradicts file order: this is BEFORE both timeouts (6, 9)
    _write_jsonl(h1_path, events)

    rc = run_check("t2a", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h1's eventual store of (h0, 2) came after its fetch_timeout(s), not before: "
        "stored at [1700000003000], timed out at [1700000006000, 1700000009000]"
    ) in out.splitlines()


def test_check_t2a_protocol_fails_when_the_dropped_seqs_file_position_contradicts_its_later_ts(
    tmp_path: Path, capsys
):
    """2026-10-23 fix (review finding): the check
    above treated ts_ms as authoritative whenever it was unambiguous (not a rounding tie),
    falling back to file position only on an exact-ts_ms tie -- so a store row that sits BEFORE
    both fetch_timeout rows in h1's own events.jsonl (events.jsonl is appended by one process per
    life in true execution order, so file order is authoritative for when things actually
    happened) but carries a ts_ms that reads as "after" still passed as a genuine withheld reply.
    A contradicting ts_ms is not evidence of health either way: this moves the seq-2 store row to
    sit ahead of both timeouts in file order (same as the "stored_before_its_own_timeout" test
    above) while giving it a ts_ms that is unambiguously AFTER both timeouts, so only file order
    objects to it."""
    experiment_path = tmp_path / "t2a.json"
    _write_experiment(experiment_path, _t2a_experiment())
    run_dir = tmp_path / "run"
    _write_t2a_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    store_event = next(e for e in events if e.get("event") == "record_stored" and e.get("seq") == 2)
    store_event["ts_ms"] = _ts_ms(14)  # unambiguously AFTER both timeouts (6, 9) -- ts_ms alone reads "healthy"
    events.remove(store_event)
    events.insert(1, store_event)  # but file order still says it landed ahead of both fetch_timeout rows
    _write_jsonl(h1_path, events)

    rc = run_check("t2a", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h1's eventual store of (h0, 2) came after its fetch_timeout(s), not before: "
        "stored at [1700000014000], timed out at [1700000006000, 1700000009000]"
    ) in out.splitlines()


def test_check_t2a_protocol_fails_on_a_fetch_timeout_for_a_seq_that_was_never_dropped(
    tmp_path: Path, capsys
):
    """2026-10-23 fix: the fetch_timeout-count check only ever looked at
    (drop.origin, drop.seq) -- a timeout against any OTHER seq (the holder failing to serve a
    record it was never faulted to withhold) was invisible to this mode entirely."""
    experiment_path = tmp_path / "t2a.json"
    _write_experiment(experiment_path, _t2a_experiment())
    run_dir = tmp_path / "run"
    _write_t2a_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    events.append({"ts_ms": _ts_ms(12), "event": "fetch_timeout", "origin": "h0", "seq": 3, "neighbor": "h0"})
    _write_jsonl(h1_path, events)

    rc = run_check("t2a", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h1 logged no fetch_timeout other than the configured dropped (h0, 2): "
        "unexpected timeout(s): [{'ts_ms': 1700000012000, 'event': 'fetch_timeout', 'origin': 'h0', "
        "'seq': 3, 'neighbor': 'h0'}]"
    ) in out.splitlines()


def test_check_t2a_protocol_fails_on_a_fetch_timeout_for_an_origin_that_was_never_dropped(
    tmp_path: Path, capsys
):
    """2026-10-23 fix (review finding): the
    unexpected-timeout check used to filter requester events down to origin=drop.origin BEFORE
    checking seq, so a fetch_timeout logged against any OTHER origin was invisible to this mode
    no matter its seq -- including one that happens to share drop.seq's own number, which this
    test pins directly. The check must inspect every requester fetch_timeout and allow only the
    exact (origin, seq) pair the experiment actually dropped."""
    experiment_path = tmp_path / "t2a.json"
    _write_experiment(experiment_path, _t2a_experiment())
    run_dir = tmp_path / "run"
    _write_t2a_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    # drop.origin is "h0", drop.seq is 2 -- this timeout is for a
    # different origin ("h1") but reuses seq 2, the exact combination
    # the old origin-first filter let slip through unnoticed.
    events.append({"ts_ms": _ts_ms(12), "event": "fetch_timeout", "origin": "h1", "seq": 2, "neighbor": "h0"})
    _write_jsonl(h1_path, events)

    rc = run_check("t2a", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h1 logged no fetch_timeout other than the configured dropped (h0, 2): "
        "unexpected timeout(s): [{'ts_ms': 1700000012000, 'event': 'fetch_timeout', 'origin': 'h1', "
        "'seq': 2, 'neighbor': 'h0'}]"
    ) in out.splitlines()


def test_check_t2a_precondition_fails_when_the_holder_never_wrote_a_counters_line(tmp_path: Path, capsys):
    """2026-10-23 fix: see check_t1's identical fix -- T2(a) never checked
    that every node actually finished its shutdown (wrote a counters line) either."""
    experiment_path = tmp_path / "t2a.json"
    _write_experiment(experiment_path, _t2a_experiment())
    run_dir = tmp_path / "run"
    _write_t2a_happy_path(run_dir)
    _write_jsonl(_node_events_path(run_dir, "h0"), [])

    rc = run_check("t2a", str(experiment_path), str(run_dir))

    assert rc == 3
    out = capsys.readouterr().out
    assert (
        "PRECOND-FAIL: h0 wrote a counters line before this run ended: "
        "h0 wrote no counters line (shutdown did not finish)"
    ) in out.splitlines()


# -- T2b --------------------------------------------------------------------


def _t2b_experiment() -> dict:
    """A small T2(b)-shaped experiment description: origin h0 is restarted mid-schedule."""
    return {
        "name": "t2bmini",
        "num": 3,
        "seed": 1,
        "group": "g",
        "hosts": 3,
        "links": [[0, 1], [1, 2]],
        "duration": 40,
        "output_dir": "logs/mini",
        "agents": [0, 1, 2],
        "publisher": {
            "host": 0,
            "uri": "ccnx:/content/h0/body",
            "file": "./sample-putfile",
            "expiry": 3000,
            "cache_time": 3000,
        },
        "schedule": {
            "0": [
                {"at_s": 3, "zone": "zone1", "disaster": "flood", "media": "image"},
                {"at_s": 5, "zone": "zone1", "disaster": "flood", "media": "image"},
                {"at_s": 15, "zone": "zone1", "disaster": "flood", "media": "image"},
                {"at_s": 17, "zone": "zone1", "disaster": "flood", "media": "image"},
            ]
        },
        "events": [{"at": 9, "type": "host_app_restart", "host": 0, "downtime_s": 3}],
    }


def _write_t2b_happy_path(run_dir: Path) -> None:
    """A full, consistent T2(b) run: h0's own 1..4 seqs split across two lives, no gaps or dupes."""
    _write_jsonl(
        _node_events_path(run_dir, "h0"),
        [
            {"ts_ms": _ts_ms(3), "event": "record_created", "origin": "h0", "seq": 1},
            {"ts_ms": _ts_ms(5), "event": "record_created", "origin": "h0", "seq": 2},
            {"ts_ms": _ts_ms(7), "event": "counters", "bytes_sent": 100, "bytes_received": 50},
            {"ts_ms": _ts_ms(15), "event": "record_created", "origin": "h0", "seq": 3},
            {"ts_ms": _ts_ms(17), "event": "record_created", "origin": "h0", "seq": 4},
            {"ts_ms": _ts_ms(30), "event": "counters", "bytes_sent": 80, "bytes_received": 40},
        ],
    )
    for node, holder in (("h1", "h0"), ("h2", "h1")):
        _write_jsonl(
            _node_events_path(run_dir, node),
            [
                {"ts_ms": _ts_ms(6), "event": "record_stored", "origin": "h0", "seq": 1, "holder": holder},
                {"ts_ms": _ts_ms(8), "event": "record_stored", "origin": "h0", "seq": 2, "holder": holder},
                {"ts_ms": _ts_ms(18), "event": "record_stored", "origin": "h0", "seq": 3, "holder": holder},
                {"ts_ms": _ts_ms(20), "event": "record_stored", "origin": "h0", "seq": 4, "holder": holder},
                {"ts_ms": _ts_ms(35), "event": "counters", "bytes_sent": 40, "bytes_received": 60},
            ],
        )
    # restart (at: 9, downtime_s: 3) actually finishes at offset 11 here --
    # strictly between h0's two pre-restart creations (3, 5) and its two
    # post-restart ones (15, 17), the window the "both sides" precondition
    # needs.
    _write_results(
        run_dir,
        [
            {
                "op_type": "event",
                "event_type": "host_app_restart",
                "ts": _iso(11),
                "success": True,
                "event": {"host": 0, "downtime_s": 3},
            },
        ],
    )


def test_check_t2b_all_pass(tmp_path: Path):
    """Check t2b all pass."""
    experiment_path = tmp_path / "t2b.json"
    _write_experiment(experiment_path, _t2b_experiment())
    run_dir = tmp_path / "run"
    _write_t2b_happy_path(run_dir)

    assert run_check("t2b", str(experiment_path), str(run_dir)) == 0


def test_check_t2b_protocol_fails_on_a_duplicate_own_seq(tmp_path: Path, capsys):
    """Check t2b protocol fails on a duplicate own seq."""
    experiment_path = tmp_path / "t2b.json"
    _write_experiment(experiment_path, _t2b_experiment())
    run_dir = tmp_path / "run"
    _write_t2b_happy_path(run_dir)
    h0_path = _node_events_path(run_dir, "h0")
    events = [json.loads(line) for line in h0_path.read_text(encoding="utf-8").splitlines()]
    events.append({"ts_ms": _ts_ms(31), "event": "record_created", "origin": "h0", "seq": 2})
    _write_jsonl(h0_path, events)

    rc = run_check("t2b", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL: h0's own seqs are contiguous and unique across both lives: duplicates present: [2]" in out.splitlines()


def test_check_t2b_protocol_fails_when_a_receiver_never_converges(tmp_path: Path, capsys):
    """2026-10-22 fix: a receiver losing a record across the restart is a
    protocol failure (exit 1), not a precondition failure (exit 3) -- convergence past a
    restart is exactly what T2(b) exists to test."""
    experiment_path = tmp_path / "t2b.json"
    _write_experiment(experiment_path, _t2b_experiment())
    run_dir = tmp_path / "run"
    _write_t2b_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    events = [e for e in events if not (e.get("event") == "record_stored" and e.get("seq") == 4)]
    _write_jsonl(h2_path, events)

    rc = run_check("t2b", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL: h2 eventually received all 4 of h0's records: got [1, 2, 3], expected 1..4" in out.splitlines()


def test_check_t2b_precondition_fails_when_the_restart_did_not_succeed(tmp_path: Path, capsys):
    """2026-10-22 fix: a host_app_restart results.json row with success=False
    means ReCefore's own stop/relaunch mechanism failed -- nothing about the sync protocol
    was meaningfully exercised, so this is a precondition failure, not a protocol one."""
    experiment_path = tmp_path / "t2b.json"
    _write_experiment(experiment_path, _t2b_experiment())
    run_dir = tmp_path / "run"
    _write_t2b_happy_path(run_dir)
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results[0]["success"] = False
    results_path.write_text(json.dumps(results), encoding="utf-8")

    rc = run_check("t2b", str(experiment_path), str(run_dir))

    assert rc == 3
    out = capsys.readouterr().out
    assert "PRECOND-FAIL: host_app_restart on h0 succeeded: success=False" in out.splitlines()


def test_check_t2b_precondition_fails_on_an_overdue_restart(tmp_path: Path, capsys):
    """2026-10-22 fix: the restart must actually interrupt the schedule. If its
    actual fire time (from results.json, not its nominal `at:`) is overdue -- after every one
    of h0's own records had already been created -- life 2 never creates anything, and the
    scenario never tested a restart mid-schedule at all."""
    experiment_path = tmp_path / "t2b.json"
    _write_experiment(experiment_path, _t2b_experiment())
    run_dir = tmp_path / "run"
    _write_t2b_happy_path(run_dir)
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results[0]["ts"] = _iso(25)  # after h0's last record_created (17)
    results_path.write_text(json.dumps(results), encoding="utf-8")

    rc = run_check("t2b", str(experiment_path), str(run_dir))

    assert rc == 3
    out = capsys.readouterr().out
    assert (
        "PRECOND-FAIL: h0 created records on both sides of the actual restart (life 1 and life 2): "
        "4 before, 0 after (restart finished at ts 1700000025000)"
    ) in out.splitlines()


def test_check_t2b_protocol_fails_on_a_duplicate_received_seq(tmp_path: Path, capsys):
    """Locks in a check that already behaves correctly: a receiver logging the same
    record_stored seq twice (e.g. two different holders racing to serve it, or a retried fetch
    both of whose replies land) must FAIL, not silently collapse into the same stored set."""
    experiment_path = tmp_path / "t2b.json"
    _write_experiment(experiment_path, _t2b_experiment())
    run_dir = tmp_path / "run"
    _write_t2b_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    first = next(e for e in events if e.get("event") == "record_stored")
    events.append(dict(first, ts_ms=first["ts_ms"] + 50000))
    _write_jsonl(h1_path, events)

    rc = run_check("t2b", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL: h1 holds exactly 4 of h0's records, none duplicated: duplicates present: [1]" in out.splitlines()


def test_check_t2b_protocol_fails_when_a_receiver_stores_a_seq_before_the_origin_created_it(
    tmp_path: Path, capsys
):
    """2026-10-23 fix: nothing checked that a receiver's own record_stored
    for one of the origin's seqs never predates that seq's record_created on the origin itself --
    a receiver whose progress runs AHEAD of the origin. Moves both receivers' seq-3 store to
    before h0's own life-2 creation of it (offset 15)."""
    experiment_path = tmp_path / "t2b.json"
    _write_experiment(experiment_path, _t2b_experiment())
    run_dir = tmp_path / "run"
    _write_t2b_happy_path(run_dir)
    for node in ("h1", "h2"):
        path = _node_events_path(run_dir, node)
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for event in events:
            if event.get("event") == "record_stored" and event.get("seq") == 3:
                event["ts_ms"] = _ts_ms(10)
        _write_jsonl(path, events)

    rc = run_check("t2b", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out.splitlines()
    assert "FAIL: no record h1 stored for h0 predates that seq's own record_created: seqs stored before being created: [3]" in out
    assert "FAIL: no record h2 stored for h0 predates that seq's own record_created: seqs stored before being created: [3]" in out


def test_check_t2b_protocol_fails_when_a_receiver_keeps_timing_out_on_the_origin_long_after_restart(
    tmp_path: Path, capsys
):
    """2026-10-23 fix: T2(b) had no grace-window check at all for
    fetch_timeouts against the restarted origin -- an origin whose life 2 never actually answers
    again, visible only as timeouts repeated well past the restart's grace window, passed
    silently as long as every record eventually arrived some other way.

    2026-10-23 fix (review finding): the grace
    deadline below also folds in post-restart record-delivery evidence (not just state_rx), which
    legitimately moves this run's own deadline to offset 20.5 (seq 3's happy-path delivery at 18
    is the earliest readiness evidence here, with no state_rx present) -- offsets shifted from
    (20, 24, 28) to (22, 26, 30) so this still demonstrates persistent timeouts well past grace,
    not one sitting right at the old boundary."""
    experiment_path = tmp_path / "t2b.json"
    _write_experiment(experiment_path, _t2b_experiment())
    run_dir = tmp_path / "run"
    _write_t2b_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    for offset in (22, 26, 30):
        events.append({"ts_ms": _ts_ms(offset), "event": "fetch_timeout", "origin": "h0", "seq": 4, "neighbor": "h0"})
    _write_jsonl(h1_path, events)

    rc = run_check("t2b", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h0's life 2 left no fetch_timeout against it standing after the restart's grace window: "
        "fetch_timeout(s) after grace: "
        "[{'ts_ms': 1700000022000, 'event': 'fetch_timeout', 'origin': 'h0', 'seq': 4, 'neighbor': 'h0'}, "
        "{'ts_ms': 1700000026000, 'event': 'fetch_timeout', 'origin': 'h0', 'seq': 4, 'neighbor': 'h0'}, "
        "{'ts_ms': 1700000030000, 'event': 'fetch_timeout', 'origin': 'h0', 'seq': 4, 'neighbor': 'h0'}]"
    ) in out.splitlines()


def test_check_t2b_all_pass_when_a_late_timeout_falls_within_the_state_rx_based_grace_window(
    tmp_path: Path,
):
    """2026-10-23 fix (review finding): restart_ts marks
    when HostAppFleet.restart_host() RETURNS, not when the new life's cefnetd prefix
    registration actually finishes -- a retry sent during that registration lag can legitimately
    time out past the old restart_ts + 2500 deadline while life 2 is already healthy. Reproduces
    an external reviewer's exact scenario: seq 2 is still outstanding across the restart
    (created at offset 5, before it), retries sent at offsets 8.6 and 11.6 time out at 10.6 and
    13.6 (their own 2s interest_lifetime later), h1 hears h0 again (state_rx) at offset 12, and
    the eventual successful retry lands at 15.7 (h1) / 16.0 (h2, relayed) -- 13.6 is 0.1s past the
    OLD deadline (restart at 11 + 2.5) but within 2500ms of the state_rx that actually proves
    h0's life 2 is reachable again."""
    experiment_path = tmp_path / "t2b.json"
    _write_experiment(experiment_path, _t2b_experiment())
    run_dir = tmp_path / "run"
    _write_t2b_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    h1_events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    for event in h1_events:
        if event.get("event") == "record_stored" and event.get("seq") == 2:
            event["ts_ms"] = _ts_ms(15.7)  # the eventual successful retry, not the original offset 8
    h1_events.append({"ts_ms": _ts_ms(10.6), "event": "fetch_timeout", "origin": "h0", "seq": 2, "neighbor": "h0"})
    h1_events.append({"ts_ms": _ts_ms(12), "event": "state_rx", "neighbor": "h0", "hi": {"h0": 1}})
    h1_events.append({"ts_ms": _ts_ms(13.6), "event": "fetch_timeout", "origin": "h0", "seq": 2, "neighbor": "h0"})
    _write_jsonl(h1_path, h1_events)
    h2_path = _node_events_path(run_dir, "h2")
    h2_events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for event in h2_events:
        if event.get("event") == "record_stored" and event.get("seq") == 2:
            event["ts_ms"] = _ts_ms(16.0)  # relayed from h1 shortly after h1's own eventual store
    _write_jsonl(h2_path, h2_events)

    assert run_check("t2b", str(experiment_path), str(run_dir)) == 0


def test_check_t2b_protocol_still_fails_when_timeouts_persist_well_past_the_state_rx_too(
    tmp_path: Path, capsys
):
    """Companion to the healthy-startup case above: proves the
    state_rx-based grace window still rejects a genuine persistent failure, not just slow
    registration -- h1 hears h0 again at offset 12 (so life 2 IS reachable), yet keeps timing out
    on it for seq 4 far past that reply's own 2500ms grace, same as the un-anchored case above."""
    experiment_path = tmp_path / "t2b.json"
    _write_experiment(experiment_path, _t2b_experiment())
    run_dir = tmp_path / "run"
    _write_t2b_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    events.append({"ts_ms": _ts_ms(12), "event": "state_rx", "neighbor": "h0", "hi": {"h0": 2}})
    for offset in (20, 24, 28):
        events.append({"ts_ms": _ts_ms(offset), "event": "fetch_timeout", "origin": "h0", "seq": 4, "neighbor": "h0"})
    _write_jsonl(h1_path, events)

    rc = run_check("t2b", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h0's life 2 left no fetch_timeout against it standing after the restart's grace window: "
        "fetch_timeout(s) after grace: "
        "[{'ts_ms': 1700000020000, 'event': 'fetch_timeout', 'origin': 'h0', 'seq': 4, 'neighbor': 'h0'}, "
        "{'ts_ms': 1700000024000, 'event': 'fetch_timeout', 'origin': 'h0', 'seq': 4, 'neighbor': 'h0'}, "
        "{'ts_ms': 1700000028000, 'event': 'fetch_timeout', 'origin': 'h0', 'seq': 4, 'neighbor': 'h0'}]"
    ) in out.splitlines()


def test_check_t2b_protocol_fails_even_when_a_very_late_first_state_rx_would_otherwise_mask_it(
    tmp_path: Path, capsys
):
    """2026-10-23 fix (review finding): the
    grace deadline above used to be anchored on the first post-restart state_rx alone, with no
    bound -- a state_rx arriving very late for reasons unrelated to h0's own readiness (e.g. that
    neighbor's own poll cadence) drags the deadline out with it, even when an earlier successful
    delivery already proves h0 was reachable long before. Reproduces the reviewer's exact
    scenario: h0 restarts at offset 11, seq 3 is delivered at 18 (proving readiness by then), seq
    4 times out at 20/24/28 before finally landing at 31, and the first post-restart state_rx h1
    hears doesn't arrive until 29 -- which would move the deadline to 31.5 and hide every one of
    those timeouts if state_rx were the only readiness evidence consulted. Folding in the seq-3
    delivery (the earliest readiness evidence available, even though it is not a state_rx) keeps
    the deadline at offset 20.5, so 24 and 28 still read as genuinely late."""
    experiment_path = tmp_path / "t2b.json"
    _write_experiment(experiment_path, _t2b_experiment())
    run_dir = tmp_path / "run"
    _write_t2b_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    h1_events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    for event in h1_events:
        if event.get("event") == "record_stored" and event.get("seq") == 4:
            event["ts_ms"] = _ts_ms(31)  # eventual arrival, after the persistent timeouts below
    for offset in (20, 24, 28):
        h1_events.append({"ts_ms": _ts_ms(offset), "event": "fetch_timeout", "origin": "h0", "seq": 4, "neighbor": "h0"})
    h1_events.append({"ts_ms": _ts_ms(29), "event": "state_rx", "neighbor": "h0", "hi": {"h0": 3}})
    _write_jsonl(h1_path, h1_events)
    h2_path = _node_events_path(run_dir, "h2")
    h2_events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for event in h2_events:
        if event.get("event") == "record_stored" and event.get("seq") == 4:
            event["ts_ms"] = _ts_ms(32)  # relayed from h1 shortly after h1's own eventual store
    _write_jsonl(h2_path, h2_events)

    rc = run_check("t2b", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h0's life 2 left no fetch_timeout against it standing after the restart's grace window: "
        "fetch_timeout(s) after grace: "
        "[{'ts_ms': 1700000024000, 'event': 'fetch_timeout', 'origin': 'h0', 'seq': 4, 'neighbor': 'h0'}, "
        "{'ts_ms': 1700000028000, 'event': 'fetch_timeout', 'origin': 'h0', 'seq': 4, 'neighbor': 'h0'}]"
    ) in out.splitlines()


def test_check_t2b_protocol_fails_when_a_relayed_delivery_would_otherwise_mask_a_registration_lag_failure(
    tmp_path: Path, capsys
):
    """2026-10-23 fix (review finding, #2): the
    readiness-evidence list above folded in every post-restart record_stored for origin
    regardless of its holder -- but a record h2 relays from h1 says nothing about whether h0's
    own life 2 is actually reachable; only a delivery h1 itself got DIRECTLY from h0 (holder=h0)
    is real evidence of that. Reproduces the masking failure: h2 merely re-logs its (already
    pre-restart) seq-2 delivery at offset 12 -- a relay through h1, holder "h1", with nothing to
    do with h0's readiness -- while h1's own FIRST direct contact with h0's life 2 does not land
    until offset 20. A single legitimate registration-lag timeout at offset 15 (well before h1's
    genuine readiness at 20) should pass; counting h2's relay as readiness evidence instead
    anchors the deadline at 14.5 and turns that ordinary lag into a false protocol failure."""
    experiment_path = tmp_path / "t2b.json"
    _write_experiment(experiment_path, _t2b_experiment())
    run_dir = tmp_path / "run"
    _write_t2b_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    h1_events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    for event in h1_events:
        if event.get("event") == "record_stored" and event.get("seq") == 3:
            event["ts_ms"] = _ts_ms(20)  # h1's own first direct post-restart contact with h0
    h1_events.append({"ts_ms": _ts_ms(15), "event": "fetch_timeout", "origin": "h0", "seq": 3, "neighbor": "h0"})
    _write_jsonl(h1_path, h1_events)
    h2_path = _node_events_path(run_dir, "h2")
    h2_events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for event in h2_events:
        if event.get("event") == "record_stored" and event.get("seq") == 2:
            # a relayed (holder h1, NOT h0) re-delivery of a pre-restart
            # record, landing shortly after restart_ts -- irrelevant to
            # whether h0 itself is reachable again.
            event["ts_ms"] = _ts_ms(12)
    _write_jsonl(h2_path, h2_events)

    rc = run_check("t2b", str(experiment_path), str(run_dir))

    assert rc == 0
    out = capsys.readouterr().out
    assert (
        "PASS: h0's life 2 left no fetch_timeout against it standing after the restart's grace window: "
        "fetch_timeout(s) after grace: []"
    ) in out.splitlines()


# -- T2c --------------------------------------------------------------------


def _t2c_experiment() -> dict:
    """A small T2(c)-shaped experiment description: holder h1 (not the origin) is restarted mid-sync."""
    return {
        "name": "t2cmini",
        "num": 4,
        "seed": 1,
        "group": "g",
        "hosts": 3,
        "links": [[0, 1], [1, 2]],
        "duration": 40,
        "output_dir": "logs/mini",
        "agents": [0, 1, 2],
        "publisher": {
            "host": 0,
            "uri": "ccnx:/content/h0/body",
            "file": "./sample-putfile",
            "expiry": 3000,
            "cache_time": 3000,
        },
        "schedule": {
            "0": [
                {"at_s": 3, "zone": "zone1", "disaster": "flood", "media": "image"},
                {"at_s": 5, "zone": "zone1", "disaster": "flood", "media": "image"},
                {"at_s": 7, "zone": "zone1", "disaster": "flood", "media": "image"},
                {"at_s": 9, "zone": "zone1", "disaster": "flood", "media": "image"},
            ]
        },
        "events": [{"at": 8, "type": "host_app_restart", "host": 1, "downtime_s": 3}],
    }


def _write_t2c_happy_path(run_dir: Path) -> None:
    """A full, consistent T2(c) run: h1 restarts mid-sync (actually finishing at offset 11,
    results.json), with two of h0's records reaching h1 before that and two after -- never
    serving either to h2 ahead of its own store, and never advertising (state_rx) a seq it
    had not yet actually stored."""
    _write_jsonl(
        _node_events_path(run_dir, "h0"),
        [
            {"ts_ms": _ts_ms(3), "event": "record_created", "origin": "h0", "seq": 1},
            {"ts_ms": _ts_ms(5), "event": "record_created", "origin": "h0", "seq": 2},
            {"ts_ms": _ts_ms(7), "event": "record_created", "origin": "h0", "seq": 3},
            {"ts_ms": _ts_ms(9), "event": "record_created", "origin": "h0", "seq": 4},
            {"ts_ms": _ts_ms(30), "event": "counters", "bytes_sent": 10, "bytes_received": 0},
        ],
    )
    _write_jsonl(
        _node_events_path(run_dir, "h1"),
        [
            {"ts_ms": _ts_ms(4), "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h0"},
            {"ts_ms": _ts_ms(6), "event": "record_stored", "origin": "h0", "seq": 2, "holder": "h0"},
            {"ts_ms": _ts_ms(8), "event": "counters", "bytes_sent": 5, "bytes_received": 5},
            {"ts_ms": _ts_ms(20), "event": "record_stored", "origin": "h0", "seq": 3, "holder": "h0"},
            {"ts_ms": _ts_ms(26), "event": "record_stored", "origin": "h0", "seq": 4, "holder": "h0"},
            {"ts_ms": _ts_ms(32), "event": "counters", "bytes_sent": 5, "bytes_received": 5},
        ],
    )
    _write_jsonl(
        _node_events_path(run_dir, "h2"),
        [
            {"ts_ms": _ts_ms(6), "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h1"},
            {"ts_ms": _ts_ms(8), "event": "record_stored", "origin": "h0", "seq": 2, "holder": "h1"},
            # h1's life 2 starts after its own counters line at offset 8;
            # this is h2 hearing from it again (state_rx), before h2 asks
            # for and gets seq 3 -- the anchor the "progress not ahead"
            # check measures its late-fetch_timeout grace window from.
            # hi: {"h0": 2} is an honest advertisement -- h1 had actually
            # stored exactly seqs 1-2 by this point (ts offset 19, between
            # its own seq-2 store at 6 and seq-3 store at 20).
            {"ts_ms": _ts_ms(19), "event": "state_rx", "neighbor": "h1", "hi": {"h0": 2}},
            {"ts_ms": _ts_ms(22), "event": "record_stored", "origin": "h0", "seq": 3, "holder": "h1"},
            {"ts_ms": _ts_ms(28), "event": "record_stored", "origin": "h0", "seq": 4, "holder": "h1"},
            {"ts_ms": _ts_ms(35), "event": "counters", "bytes_sent": 3, "bytes_received": 3},
        ],
    )
    # host_app_restart (at: 8, downtime_s: 3) actually finishes at offset
    # 11 here: strictly between h1's two pre-restart stores (4, 6) and its
    # two post-restart ones (20, 26), and while h2 (far) still lacks seq 3
    # and 4 -- the window the two new timing preconditions need.
    _write_results(
        run_dir,
        [
            {
                "op_type": "event",
                "event_type": "host_app_restart",
                "ts": _iso(11),
                "success": True,
                "event": {"host": 1, "downtime_s": 3},
            },
        ],
    )


def test_check_t2c_all_pass(tmp_path: Path):
    """Check t2c all pass."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)

    assert run_check("t2c", str(experiment_path), str(run_dir)) == 0


def test_check_t2c_protocol_fails_when_far_node_is_served_before_the_holder_itself_stored(tmp_path: Path, capsys):
    """The "progress ahead" bug this check guards against: h2 reports holding seq 3 from h1
    before h1's own events.jsonl shows it actually stored seq 3 (e.g. a stale in-memory belief
    surviving a restart instead of being read back from the persisted store)."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("event") == "record_stored" and event.get("seq") == 3:
            event["ts_ms"] = _ts_ms(10)  # before h1's own seq-3 record_stored at offset 20
    _write_jsonl(h2_path, events)

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL: h1 never served a record to h2 before h1 itself had stored it: seqs served early: [3]" in out.splitlines()


def test_check_t2c_protocol_fails_when_holder_keeps_timing_out_long_after_its_restart(tmp_path: Path, capsys):
    """The bug the ts-ordering check above cannot see: h1's life 2 advertises a seq it does not
    actually hold, so h2 never gets a (wrongly early) record_stored for it at all -- only a
    fetch_timeout against h1, repeated well past the grace window right after h1 came back up."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    # state_rx from h1's life 2 is at offset 19; this timeout lands well
    # past the 2.5s grace window that follows it.
    events.append({"ts_ms": _ts_ms(24), "event": "fetch_timeout", "origin": "h0", "seq": 3, "neighbor": "h1"})
    _write_jsonl(h2_path, events)

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h1's life 2 never advertised a record it could not actually serve to h2: "
        "fetch_timeout(s) after grace: "
        "[{'ts_ms': 1700000024000, 'event': 'fetch_timeout', 'origin': 'h0', 'seq': 3, 'neighbor': 'h1'}]"
    ) in out.splitlines()


def test_check_t2c_protocol_fails_on_a_timeout_the_first_state_rx_as_anchor_would_have_missed(
    tmp_path: Path, capsys
):
    """2026-10-22 fix (regression): discriminates the grace deadline's own anchor, not merely
    whether a late fetch_timeout is caught at all. restart_ts (offset 11) + the 2.5s grace puts
    the deadline at offset 13.5; this timeout at offset 15 lands after it. Anchoring the grace
    window on the first post-restart state_rx (offset 19, as check.py's own comment above warns
    against) instead of restart_ts would put the deadline at offset 21.5 -- after this timeout --
    and miss it, producing a false PASS."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    events.append({"ts_ms": _ts_ms(15), "event": "fetch_timeout", "origin": "h0", "seq": 3, "neighbor": "h1"})
    _write_jsonl(h2_path, events)

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h1's life 2 never advertised a record it could not actually serve to h2: "
        "fetch_timeout(s) after grace: "
        "[{'ts_ms': 1700000015000, 'event': 'fetch_timeout', 'origin': 'h0', 'seq': 3, 'neighbor': 'h1'}]"
    ) in out.splitlines()


def test_check_t2c_protocol_fails_when_holder_never_converges(tmp_path: Path, capsys):
    """2026-10-22 fix: a restarted holder that loses one of the origin's records
    for good is a protocol failure (exit 1), not a precondition failure (exit 3) -- convergence
    past the restart is exactly T2(c)'s own protocol property. Drops h1's seq 4 (not seq 3, the
    only one of its post-restart records that remains) so the "both sides of the restart"
    timing precondition this scenario also depends on keeps passing."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    events = [e for e in events if not (e.get("event") == "record_stored" and e.get("seq") == 4)]
    _write_jsonl(h1_path, events)

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h1 eventually received all 4 of h0's records (across both lives): "
        "got [1, 2, 3], expected 1..4"
    ) in out.splitlines()


def test_check_t2c_protocol_fails_when_far_never_converges(tmp_path: Path, capsys):
    """The same precondition/protocol reclassification for the far node's own convergence
    check -- a separate code path from the holder one above, so it needs its own test. Drops
    h2's seq 4, the opposite choice from the holder test above."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    events = [e for e in events if not (e.get("event") == "record_stored" and e.get("seq") == 4)]
    _write_jsonl(h2_path, events)

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL: h2 eventually received all 4 of h0's records: got [1, 2, 3], expected 1..4" in out.splitlines()


def test_check_t2c_precondition_fails_when_the_restart_did_not_succeed(tmp_path: Path, capsys):
    """2026-10-22 fix: success=False in the host_app_restart results.json row
    means ReCefore's own stop/relaunch mechanism failed, so nothing about the sync protocol
    across a restart was meaningfully exercised -- a precondition failure, not a protocol one."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results[0]["success"] = False
    results_path.write_text(json.dumps(results), encoding="utf-8")

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 3
    out = capsys.readouterr().out
    assert "PRECOND-FAIL: host_app_restart on h1 succeeded: success=False" in out.splitlines()


def test_check_t2c_precondition_fails_on_an_overdue_restart(tmp_path: Path, capsys):
    """2026-10-22 fix: an overdue restart (its actual results.json fire time,
    not its nominal `at:`, lands after h1 had already stored every one of h0's records) never
    actually interrupted anything, so T2(c)'s own restart-safety property went untested."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results[0]["ts"] = _iso(27)  # after h1's own last record_stored (seq 4 at offset 26)
    results_path.write_text(json.dumps(results), encoding="utf-8")

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 3
    out = capsys.readouterr().out
    assert (
        "PRECOND-FAIL: h1 received records from h0 on both sides of the actual restart: "
        "4 before, 0 after (restart finished at ts 1700000027000)"
    ) in out.splitlines()


def test_check_t2c_protocol_fails_when_holder_advertises_a_seq_it_had_not_yet_stored(tmp_path: Path, capsys):
    """2026-10-22 fix: the false-advertisement bug this check now catches
    directly -- h1's life 2 claims hi=3 in its post-restart state_rx to h2, but by that
    reply's own ts (offset 19) h1's events.jsonl shows it had actually stored only seqs 1-2 (its
    seq-3 store does not land until offset 20). The old checks alone would miss this: h2's own
    seq-3 record_stored still lands after h1's (not "served early"), and no fetch_timeout
    follows at all, let alone a late one -- only the advertised-vs-stored comparison below
    sees the false claim."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("event") == "state_rx":
            event["hi"] = {"h0": 3}
    _write_jsonl(h2_path, events)

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h1's advertised hi for h0 in its post-restart state_rx at ts 1700000019000 never "
        "exceeds what it had actually stored by then: advertised hi=3, holder's own contiguous "
        "progress at ts 1700000019000 was 2"
    ) in out.splitlines()


def test_check_t2c_precondition_fails_when_the_state_rx_never_mentions_the_origin(tmp_path: Path, capsys):
    """2026-10-22 fix: a state_rx reply that never mentions `origin` at all (a malformed/empty
    reply right as h1 comes back up, or holder_state.decode failing and leaving `hi: {}`) must
    not silently default to advertised_hi=0 and pass -- that would make the advertised-vs-stored
    check above a no-op for exactly the replies it most needs to catch."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("event") == "state_rx":
            event["hi"] = {}
    _write_jsonl(h2_path, events)

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 3
    out = capsys.readouterr().out
    assert (
        "PRECOND-FAIL: h2's post-restart state_rx from h1 at ts 1700000019000 actually advertised a hi for h0: hi={}"
    ) in out.splitlines()


def test_check_t2c_protocol_fails_when_a_stale_downtime_state_rx_masks_a_dishonest_post_restart_one(
    tmp_path: Path, capsys
):
    """2026-10-22 fix: a state_rx h2 received DURING h1's downtime (offset
    9, strictly before the restart's actual finish at offset 11) used to be misclassified as
    "post-restart" because the old selection compared against h1's own first counters line
    (offset 8) instead of restart_ts. An honest-but-stale reply there used to satisfy the whole
    check on its own (it was always index 0 of the sorted list), leaving the genuinely dishonest
    post-restart advertisement at offset 19 (hi=4 while h1 had only stored 1-2 by then) never
    examined at all."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    # h1 had actually stored exactly seqs 1-2 by offset 9 (its seq-3
    # store does not land until offset 20), so hi: {"h0": 2} here is
    # honest -- but it arrived during the downtime (restart finishes at
    # offset 11), not after it.
    events.insert(2, {"ts_ms": _ts_ms(9), "event": "state_rx", "neighbor": "h1", "hi": {"h0": 2}})
    for event in events:
        if event.get("event") == "state_rx" and event["ts_ms"] == _ts_ms(19):
            event["hi"] = {"h0": 4}  # dishonest: h1 had only stored 1-2 by offset 19
    _write_jsonl(h2_path, events)

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h1's advertised hi for h0 in its post-restart state_rx at ts 1700000019000 never "
        "exceeds what it had actually stored by then: advertised hi=4, holder's own contiguous "
        "progress at ts 1700000019000 was 2"
    ) in out.splitlines()


def test_check_t2c_precondition_fails_when_the_only_state_rx_predates_the_restarts_actual_finish(
    tmp_path: Path, capsys
):
    """2026-10-22 fix (regression): discriminates restart_ts itself as the post-restart boundary,
    separately from the "judge every row, not just the first" fix above. With only an honest
    state_rx at offset 9 (h1 had actually stored exactly 1-2 by then) and none later, restart_ts
    (offset 11) correctly finds no post-restart reply to judge at all. The old boundary -- h1's
    own first counters line at offset 8 -- would have let this same offset-9 row count as
    "post-restart" (9 > 8) and, being honest, pass the whole check: a false PASS that never
    actually exercised life 2's post-restart behavior."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    events = [e for e in events if e.get("event") != "state_rx"]
    events.insert(2, {"ts_ms": _ts_ms(9), "event": "state_rx", "neighbor": "h1", "hi": {"h0": 2}})
    _write_jsonl(h2_path, events)

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 3
    out = capsys.readouterr().out
    assert "PRECOND-FAIL: h2 heard from h1 again after its restart (state_rx)" in out.splitlines()


def test_check_t2c_protocol_fails_when_a_later_post_restart_state_rx_is_dishonest_even_though_the_first_was_honest(
    tmp_path: Path, capsys
):
    """2026-10-22 fix: the old check only ever judged the FIRST
    post-restart state_rx (index 0 of the sorted list). A holder whose first reply after
    restarting is honest but whose later one overstates its progress used to pass undetected."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    # h1 stored seq 3 at offset 20 and seq 4 at offset 26 (see
    # _write_t2c_happy_path); this second state_rx at offset 21 claims
    # hi=4 while h1's own contiguous progress at that ts is still 3.
    events.insert(3, {"ts_ms": _ts_ms(21), "event": "state_rx", "neighbor": "h1", "hi": {"h0": 4}})
    _write_jsonl(h2_path, events)

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert (
        "FAIL: h1's advertised hi for h0 in its post-restart state_rx at ts 1700000021000 never "
        "exceeds what it had actually stored by then: advertised hi=4, holder's own contiguous "
        "progress at ts 1700000021000 was 3"
    ) in out.splitlines()


def test_check_t2c_protocol_fails_on_a_duplicate_far_stored_seq(tmp_path: Path, capsys):
    """2026-10-22 fix: record_stored events must be kept as a list, not collapsed
    into a {seq: event} dict, before _contiguous_unique runs -- a dict silently drops a
    duplicate by key, which is exactly the thing this check exists to catch."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    events.append({"ts_ms": _ts_ms(29), "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h1"})
    _write_jsonl(h2_path, events)

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL: h2 holds exactly 4 of h0's records, none duplicated: duplicates present: [1]" in out.splitlines()


def test_check_t2c_protocol_fails_when_far_claims_every_record_came_straight_from_the_origin(
    tmp_path: Path, capsys
):
    """2026-10-23 fix: far's holder label was assumed correct "by
    construction" (its only sync neighbor is the restarted relay) and never actually checked.
    Every one of h2's record_stored rows here claims holder=h0, as if the restarted relay (h1)
    never served anything."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("event") == "record_stored":
            event["holder"] = "h0"
    _write_jsonl(h2_path, events)

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL: every record h2 stored for h0 came from holder h1: holders seen: ['h0']" in out.splitlines()


def test_check_t2c_protocol_fails_when_far_mixes_holder_labels(tmp_path: Path, capsys):
    """2026-10-23 fix: a relay PARTLY bypassed (only some of far's rows
    mislabeled) is just as real a protocol break as every row being wrong; the fix checks the
    whole set of holders seen on far, not a single sampled row."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("event") == "record_stored" and event.get("seq") == 3:
            event["holder"] = "h0"
    _write_jsonl(h2_path, events)

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL: every record h2 stored for h0 came from holder h1: holders seen: ['h0', 'h1']" in out.splitlines()


# -- exit code precedence -------------------------------------------------


def test_run_check_exits_1_and_prints_a_summary_when_both_precondition_and_protocol_fail(tmp_path: Path, capsys):
    """2026-10-22 fix: exit 1 (protocol) wins over exit 3 (precondition)
    when a run has failures in both categories, so a protocol break is never hidden behind a
    precondition exit code; the trailing SUMMARY line makes that explicit even to a reader who
    only looks at the end of the output. Combines the two existing T1 failure fixtures
    (late pre-outage convergence + an over-matching during-outage probe) in one run."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("event") == "record_stored" and event.get("seq") == 2:
            event["ts_ms"] = _ts_ms(11)  # precondition: late pre-outage convergence
        if event.get("event") == "probe_discovered" and event["ts_ms"] == _ts_ms(16):
            event["matches"].append({"origin": "h0", "seq": 3, "content_name": "ccnx:/content/h0/body"})  # protocol break
    _write_jsonl(h2_path, events)

    rc = run_check("t1", str(experiment_path), str(run_dir))

    assert rc == 1
    out = capsys.readouterr().out.splitlines()
    assert (
        "PRECOND-FAIL: all 2 pre-outage records reached h2 before link_down actually fired: "
        "1/2 stored before link_down (actual ts 1700000010000)"
    ) in out
    assert (
        "FAIL: during-outage probe discovered exactly the pre-outage records, nothing more: "
        "got [('h0', 1), ('h0', 2), ('h0', 3)]"
    ) in out
    assert "SUMMARY: a protocol failure occurred (see the FAIL line(s) above; some PRECOND-FAIL line(s) also printed)" in out


# -- CLI / usage errors -------------------------------------------------


def test_main_exits_2_on_a_missing_run_dir(tmp_path: Path, capsys):
    """Main exits 2 on a missing run dir."""
    from semindex.check import main

    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())

    run_dir = tmp_path / "no-such-run"
    rc = main(["t1", str(experiment_path), str(run_dir)])

    assert rc == 2
    assert capsys.readouterr().err == (
        f"semindex.check: {run_dir}/debug/node_dirs/h0/semindex/events.jsonl does not exist\n"
    )


def test_main_exits_2_with_a_message_when_a_results_row_is_missing_ts(tmp_path: Path, capsys):
    """2026-10-22 fix: a results.json row missing `ts` (e.g. a ReCefore
    version mismatch, or a hand-edited fixture) used to raise an unhandled KeyError traceback
    instead of exiting 2 with a message, like every other malformed-input case here."""
    from semindex.check import main

    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    for row in results:
        if row.get("event_type") == "link_down":
            del row["ts"]
    results_path.write_text(json.dumps(results), encoding="utf-8")

    rc = main(["t1", str(experiment_path), str(run_dir)])

    assert rc == 2
    assert capsys.readouterr().err == "semindex.check: missing required field: 'ts'\n"


def test_main_exits_2_with_a_message_when_an_event_is_missing_seq(tmp_path: Path, capsys):
    """2026-10-22 fix: an events.jsonl row missing `seq` (e.g. a truncated
    or hand-edited line) used to raise an unhandled KeyError traceback instead of exiting 2
    with a message."""
    from semindex.check import main

    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)
    h0_path = _node_events_path(run_dir, "h0")
    events = [json.loads(line) for line in h0_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("event") == "record_created" and event.get("seq") == 3:
            del event["seq"]
    _write_jsonl(h0_path, events)

    rc = main(["t1", str(experiment_path), str(run_dir)])

    assert rc == 2
    assert capsys.readouterr().err == "semindex.check: missing required field: 'seq'\n"


def test_main_exits_2_with_a_message_on_a_malformed_state_rx_hi_value(tmp_path: Path, capsys):
    """2026-10-23 fix: state_rx's `hi` value is normally an int; a malformed
    row holding a string there (e.g. a hand-edited fixture, or a holder_state.decode bug) made
    the advertised-vs-stored `<=` comparison in check_t2c raise an unhandled TypeError instead of
    exiting 2 with a message like every other malformed-input case here."""
    from semindex.check import main

    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("event") == "state_rx":
            event["hi"]["h0"] = "2"
    _write_jsonl(h2_path, events)

    rc = main(["t2c", str(experiment_path), str(run_dir)])

    assert rc == 2
    assert capsys.readouterr().err == (
        "semindex.check: malformed input: '<=' not supported between instances of 'str' and 'int'\n"
    )


# -- live-data regression (ReCefore host_app_restart worktree) -------------
#
# 2026-10-23: these four runs were generated from this repo's own
# experiments/*.json against a real ReCefore host_app_restart checkout
# before this round's fixes landed, and are kept as a standing regression
# check that none of the new assertions above turns a real, already-known
# run result into a DIFFERENT one. They are skipped (not failed) when that
# worktree is not present on the machine running the tests.

_LIVE_RUN_DIR_BASE = "/home/lab_shared/wt/recefore-host-app-restart/logs/semindex"
_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not os.path.isdir(f"{_LIVE_RUN_DIR_BASE}/ex9911_seed42"),
    reason="live ReCefore host_app_restart run directory not present on this machine",
)
def test_check_t1_still_passes_against_the_live_ex9911_run():
    """Regression fixture, recorded 2026-10-23: a real `disaster --config
    t1.yaml` run against ReCefore host_app_restart's own checkout exited 0 before this round's
    fixes. Every new check added here (holder labels, relay order, the outage-leak and counters
    preconditions) must still pass against this real data, not just the synthetic fixtures
    above."""
    experiment_path = _REPO_ROOT / "experiments" / "t1.json"
    run_dir = f"{_LIVE_RUN_DIR_BASE}/ex9911_seed42"
    assert run_check("t1", str(experiment_path), run_dir) == 0


@pytest.mark.skipif(
    not os.path.isdir(f"{_LIVE_RUN_DIR_BASE}/ex9912_seed42"),
    reason="live ReCefore host_app_restart run directory not present on this machine",
)
def test_check_t2a_live_ex9912_run_has_exactly_the_one_known_fail_line(capsys):
    """Regression fixture, recorded 2026-10-23: this real run already fails
    with one known FAIL line (an under-investigation Cefore PIT effect, not something this round
    touches -- the fetch_timeout-count assertion itself is explicitly out of scope here). Pins
    that this round's new checks (store-after-timeout, other-seq-timeout) add no FAIL line on
    top of it. Update this pin, not the assertion it is pinning, once that investigation lands."""
    experiment_path = _REPO_ROOT / "experiments" / "t2a.json"
    run_dir = f"{_LIVE_RUN_DIR_BASE}/ex9912_seed42"
    rc = run_check("t2a", str(experiment_path), run_dir)
    fail_lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("FAIL")]
    assert rc == 1
    assert fail_lines == ["FAIL: h1 logged exactly 2 fetch_timeout(s) for (h0, 3): got 3"]


@pytest.mark.skipif(
    not os.path.isdir(f"{_LIVE_RUN_DIR_BASE}/ex9913_seed42"),
    reason="live ReCefore host_app_restart run directory not present on this machine",
)
def test_check_t2b_still_passes_against_the_live_ex9913_run():
    """Regression fixture, recorded 2026-10-23 (see
    test_check_t1_still_passes_against_the_live_ex9911_run for why): a real T2(b) run must still
    exit 0 with the new receiver-progress and restart-grace checks in place."""
    experiment_path = _REPO_ROOT / "experiments" / "t2b.json"
    run_dir = f"{_LIVE_RUN_DIR_BASE}/ex9913_seed42"
    assert run_check("t2b", str(experiment_path), run_dir) == 0


@pytest.mark.skipif(
    not os.path.isdir(f"{_LIVE_RUN_DIR_BASE}/ex9914_seed42"),
    reason="live ReCefore host_app_restart run directory not present on this machine",
)
def test_check_t2c_still_passes_against_the_live_ex9914_run():
    """Regression fixture, recorded 2026-10-23 (see
    test_check_t1_still_passes_against_the_live_ex9911_run for why): a real T2(c) run must still
    exit 0 with the new far-holder-label check in place."""
    experiment_path = _REPO_ROOT / "experiments" / "t2c.json"
    run_dir = f"{_LIVE_RUN_DIR_BASE}/ex9914_seed42"
    assert run_check("t2c", str(experiment_path), run_dir) == 0
