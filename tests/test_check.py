"""Tests for the semindex.check checkers, against synthetic events.jsonl/results.json fixtures."""

from __future__ import annotations

import json
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
        ],
    )
    _write_jsonl(
        _node_events_path(run_dir, "h1"),
        [
            {"ts_ms": _ts_ms(4), "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h0"},
            {"ts_ms": _ts_ms(6), "event": "record_stored", "origin": "h0", "seq": 2, "holder": "h0"},
            {"ts_ms": _ts_ms(32), "event": "record_stored", "origin": "h0", "seq": 3, "holder": "h0"},
            {"ts_ms": _ts_ms(34), "event": "record_stored", "origin": "h0", "seq": 4, "holder": "h0"},
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
        ],
    )
    _write_results(
        run_dir,
        [
            {"op_type": "put", "uri": "ccnx:/content/h0/body", "success": True},
            {"op_type": "event", "event_type": "link_down", "ts": _iso(10), "event": {"nodes": [0, 1]}},
            {"op_type": "event", "event_type": "link_up", "ts": _iso(30), "event": {"nodes": [0, 1]}},
        ],
    )


def test_check_t1_all_pass(tmp_path: Path):
    """Check t1 all pass."""
    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())
    run_dir = tmp_path / "run"
    _write_t1_happy_path(run_dir)

    assert run_check("t1", str(experiment_path), str(run_dir)) == 0


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
    _write_jsonl(_node_events_path(run_dir, "h0"), [])
    _write_jsonl(
        _node_events_path(run_dir, "h1"),
        [
            {"ts_ms": _ts_ms(4), "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h0"},
            {"ts_ms": _ts_ms(6), "event": "fetch_timeout", "origin": "h0", "seq": 2, "neighbor": "h0"},
            {"ts_ms": _ts_ms(9), "event": "fetch_timeout", "origin": "h0", "seq": 2, "neighbor": "h0"},
            {"ts_ms": _ts_ms(14), "event": "record_stored", "origin": "h0", "seq": 2, "holder": "h0"},
            {"ts_ms": _ts_ms(16), "event": "record_stored", "origin": "h0", "seq": 3, "holder": "h0"},
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


def test_check_t2a_precondition_fails_when_never_converged(tmp_path: Path, capsys):
    """Check t2a precondition fails when never converged."""
    experiment_path = tmp_path / "t2a.json"
    _write_experiment(experiment_path, _t2a_experiment())
    run_dir = tmp_path / "run"
    _write_jsonl(_node_events_path(run_dir, "h0"), [])
    _write_jsonl(
        _node_events_path(run_dir, "h1"),
        [{"ts_ms": _ts_ms(4), "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h0"}],
    )

    rc = run_check("t2a", str(experiment_path), str(run_dir))

    assert rc == 3


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


def test_check_t2b_precondition_fails_when_a_receiver_never_converges(tmp_path: Path):
    """Check t2b precondition fails when a receiver never converges."""
    experiment_path = tmp_path / "t2b.json"
    _write_experiment(experiment_path, _t2b_experiment())
    run_dir = tmp_path / "run"
    _write_t2b_happy_path(run_dir)
    h2_path = _node_events_path(run_dir, "h2")
    events = [json.loads(line) for line in h2_path.read_text(encoding="utf-8").splitlines()]
    events = [e for e in events if not (e.get("event") == "record_stored" and e.get("seq") == 4)]
    _write_jsonl(h2_path, events)

    rc = run_check("t2b", str(experiment_path), str(run_dir))

    assert rc == 3


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
            ]
        },
        "events": [{"at": 8, "type": "host_app_restart", "host": 1, "downtime_s": 3}],
    }


def _write_t2c_happy_path(run_dir: Path) -> None:
    """A full, consistent T2(c) run: h1 restarts mid-sync, never serving to h2 ahead of its own store."""
    _write_jsonl(
        _node_events_path(run_dir, "h0"),
        [
            {"ts_ms": _ts_ms(3), "event": "record_created", "origin": "h0", "seq": 1},
            {"ts_ms": _ts_ms(5), "event": "record_created", "origin": "h0", "seq": 2},
            {"ts_ms": _ts_ms(7), "event": "record_created", "origin": "h0", "seq": 3},
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
            {"ts_ms": _ts_ms(25), "event": "counters", "bytes_sent": 5, "bytes_received": 5},
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
            {"ts_ms": _ts_ms(19), "event": "state_rx", "neighbor": "h1"},
            {"ts_ms": _ts_ms(22), "event": "record_stored", "origin": "h0", "seq": 3, "holder": "h1"},
            {"ts_ms": _ts_ms(35), "event": "counters", "bytes_sent": 3, "bytes_received": 3},
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


def test_check_t2c_precondition_fails_when_holder_never_converges(tmp_path: Path):
    """Check t2c precondition fails when holder never converges."""
    experiment_path = tmp_path / "t2c.json"
    _write_experiment(experiment_path, _t2c_experiment())
    run_dir = tmp_path / "run"
    _write_t2c_happy_path(run_dir)
    h1_path = _node_events_path(run_dir, "h1")
    events = [json.loads(line) for line in h1_path.read_text(encoding="utf-8").splitlines()]
    events = [e for e in events if not (e.get("event") == "record_stored" and e.get("seq") == 3)]
    _write_jsonl(h1_path, events)

    rc = run_check("t2c", str(experiment_path), str(run_dir))

    assert rc == 3


# -- CLI / usage errors -------------------------------------------------


def test_main_exits_2_on_a_missing_run_dir(tmp_path: Path, capsys):
    """Main exits 2 on a missing run dir."""
    from semindex.check import main

    experiment_path = tmp_path / "t1.json"
    _write_experiment(experiment_path, _t1_experiment())

    rc = main(["t1", str(experiment_path), str(tmp_path / "no-such-run")])

    assert rc == 2
    assert "does not exist" in capsys.readouterr().err
