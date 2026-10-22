"""Tests for semindex.evalvalid, against synthetic stored-run fixtures.

Every fixture below is built directly from `evalgen.build_experiment`'s own
output (same schedule/link-event shape evalrun would actually store), so a
test mutates exactly one stored artifact away from a run evalgen itself
could have produced, never a shape evalgen could never produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semindex import evalgen
from semindex.evalvalid import ValidityUsageError, validate

# A fixed epoch base (ms) standing in for the one real t0_ms every agent in
# a run shares -- never itself written to any stored artifact (see
# evalvalid._derive_t0_ms's docstring), so every synthetic ts_ms below is
# built as this base plus an elapsed-seconds offset, exactly how a real
# agent's EventWriter would stamp it.
_T0_MS = 1_700_000_000_000


def _ts_ms(elapsed_s: float) -> int:
    return round(_T0_MS + elapsed_s * 1000)


def _write_experiment(run_dir: Path, p: float, seed: int, stage: str = "s1") -> dict:
    raw = evalgen.build_experiment(p, seed, stage, str(run_dir))
    (run_dir / f"{raw['name']}.experiment.json").write_text(json.dumps(raw), encoding="utf-8")
    return raw


def _write_events(run_dir: Path, node: str, events: list[dict]) -> None:
    path = run_dir / "debug" / "node_dirs" / node / "semindex" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def _write_log(run_dir: Path, node: str, text: str = "") -> None:
    path = run_dir / "debug" / "node_dirs" / node / "semindex_agent.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_results(run_dir: Path, rows: list[dict]) -> None:
    (run_dir / "results.json").write_text(json.dumps(rows), encoding="utf-8")


def _counters_fields() -> dict:
    return {"bytes_sent": 100, "bytes_received": 100, "control_bytes_sent": 100, "control_bytes_received": 100}


def _record_events(host: int, schedule_entries: list[dict]) -> list[dict]:
    node = f"h{host}"
    return [
        {"ts_ms": _ts_ms(entry["at_s"]), "event": "record_created", "origin": node, "seq": i + 1}
        for i, entry in enumerate(schedule_entries)
    ]


def _snapshot_event(elapsed_s: float) -> dict:
    return {"ts_ms": _ts_ms(elapsed_s), "event": "counters_snapshot", **_counters_fields()}


def _counters_event(elapsed_s: float) -> dict:
    return {"ts_ms": _ts_ms(elapsed_s), "event": "counters", **_counters_fields()}


def _link_row(event: dict, *, actual_at: float | None = None, success: bool = True) -> dict:
    scheduled_at = event["at"]
    return {
        "op_type": "event",
        "event_type": event["type"],
        "ts": "2023-01-01T00:00:00+00:00",
        "success": success,
        "error": None,
        "scheduled_at": scheduled_at,
        "actual_at": scheduled_at if actual_at is None else actual_at,
        "event": {"nodes": list(event["nodes"])},
    }


def _put_row(uri: str, success: bool = True) -> dict:
    return {"op_type": "put", "uri": uri, "ts": "2023-01-01T00:00:00+00:00", "success": success}


def _build_happy_run(tmp_path: Path, *, p: float = 0.2, seed: int = 7, stage: str = "s1") -> tuple[Path, dict]:
    """A stored run dir that `evalvalid.validate` should accept outright."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    raw = _write_experiment(run_dir, p, seed, stage)

    for host in raw["agents"]:
        node = f"h{host}"
        events = []
        if host in evalgen.RECORD_ORIGIN_HOSTS:
            events += _record_events(host, raw["schedule"][str(host)])
        events.append(_snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]))
        events.append(_counters_event(evalgen.AGENT_STOP_AT_S))
        _write_events(run_dir, node, events)
        _write_log(run_dir, node, "")

    rows = [_put_row(raw["publisher"]["uri"])] + [_link_row(event) for event in raw["events"]]
    _write_results(run_dir, rows)
    return run_dir, raw


# -- happy paths -----------------------------------------------------------


def test_validate_accepts_a_fully_consistent_run(tmp_path: Path):
    """Validate accepts a fully consistent run."""
    run_dir, _raw = _build_happy_run(tmp_path)
    assert validate(str(run_dir)) == {"valid": True, "flags": []}


def test_validate_accepts_p0_with_no_link_events_at_all(tmp_path: Path):
    """Validate accepts p0 with no link events at all (t0 still derivable from record_created)."""
    run_dir, raw = _build_happy_run(tmp_path, p=0.0, seed=1)
    assert raw["events"] == []
    assert validate(str(run_dir)) == {"valid": True, "flags": []}


# -- missing/unreadable run_dir itself (exit 2 path) ------------------------


def test_validate_raises_when_run_dir_is_not_a_directory(tmp_path: Path):
    """Validate raises when run_dir is not a directory."""
    with pytest.raises(ValidityUsageError):
        validate(str(tmp_path / "does-not-exist"))


def test_validate_raises_when_experiment_json_is_missing(tmp_path: Path):
    """Validate raises when experiment.json is missing."""
    run_dir, _raw = _build_happy_run(tmp_path)
    (run_dir / "s1_p0.2_seed7.experiment.json").unlink()
    with pytest.raises(ValidityUsageError):
        validate(str(run_dir))


def test_validate_raises_when_experiment_json_is_unparsable(tmp_path: Path):
    """Validate raises when experiment.json is unparsable."""
    run_dir, _raw = _build_happy_run(tmp_path)
    (run_dir / "s1_p0.2_seed7.experiment.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValidityUsageError):
        validate(str(run_dir))


# -- per-node events.jsonl presence/readability ------------------------------


def test_validate_flags_a_missing_node_events_file(tmp_path: Path):
    """Validate flags a missing node events file."""
    run_dir, _raw = _build_happy_run(tmp_path)
    (run_dir / "debug" / "node_dirs" / "h1" / "semindex" / "events.jsonl").unlink()
    result = validate(str(run_dir))
    assert result["flags"] == ["missing_events:h1"]
    assert result["valid"] is False


def test_validate_flags_an_unreadable_node_events_file(tmp_path: Path):
    """Validate flags an unreadable node events file (one unparsable line)."""
    run_dir, _raw = _build_happy_run(tmp_path)
    path = run_dir / "debug" / "node_dirs" / "h1" / "semindex" / "events.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write("{not json\n")
    assert validate(str(run_dir))["flags"] == ["unreadable_events:h1"]


def test_validate_flags_an_empty_node_events_file(tmp_path: Path):
    """Validate flags an empty node events file (and, since it logged nothing, its missing
    counters_snapshot and counters lines too)."""
    run_dir, _raw = _build_happy_run(tmp_path)
    (run_dir / "debug" / "node_dirs" / "h1" / "semindex" / "events.jsonl").write_text("", encoding="utf-8")
    assert validate(str(run_dir))["flags"] == sorted(["empty_events:h1", "counters_snapshot_count:h1", "counters_count:h1"])


def test_validate_flags_a_last_line_that_is_not_counters(tmp_path: Path):
    """Validate flags a last line that is not counters."""
    run_dir, _raw = _build_happy_run(tmp_path)
    path = run_dir / "debug" / "node_dirs" / "h1" / "semindex" / "events.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts_ms": _ts_ms(131.0), "event": "probe_discovered", "matches": []}) + "\n")
    assert validate(str(run_dir))["flags"] == ["last_line_not_counters:h1"]


# -- counters_snapshot / counters timing -------------------------------------


def test_validate_flags_a_missing_counters_snapshot(tmp_path: Path):
    """Validate flags a missing counters_snapshot (zero, not one)."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h1"
    events = [_counters_event(evalgen.AGENT_STOP_AT_S)]
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == ["counters_snapshot_count:h1"]


def test_validate_flags_a_duplicate_counters_snapshot(tmp_path: Path):
    """Validate flags a duplicate counters_snapshot (two, not one)."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h1"
    events = [
        _snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]),
        _snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0] + 1.0),
        _counters_event(evalgen.AGENT_STOP_AT_S),
    ]
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == ["counters_snapshot_count:h1"]


def test_validate_flags_a_late_counters_snapshot(tmp_path: Path):
    """Validate flags a counters_snapshot more than 0.5s away from t0+10s."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h1"
    events = [
        _snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0] + 2.0),
        _counters_event(evalgen.AGENT_STOP_AT_S),
    ]
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == ["counters_snapshot_timing:h1"]


def test_validate_flags_a_late_final_counters_line(tmp_path: Path):
    """Validate flags a final counters line more than 1s away from t0+130s."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h1"
    events = [
        _snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]),
        _counters_event(evalgen.AGENT_STOP_AT_S + 3.0),
    ]
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == ["counters_timing:h1"]


def test_validate_flags_an_unplanned_restart_even_though_the_final_counters_line_is_on_time(tmp_path: Path):
    """Validate flags a second `counters` line (reviewer counterexample: h1 snapshots at
    t0+10s, writes a `counters` line at t0+20s, then -- after an unplanned restart that reset
    its process-local byte counters -- writes a second `counters` line at t0+130s). The final
    line alone lands exactly on schedule, so only requiring exactly one line catches this."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h1"
    events = [
        _snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]),
        _counters_event(20.0),
        _counters_event(evalgen.AGENT_STOP_AT_S),
    ]
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == ["counters_count:h1"]


# -- record_created schedule --------------------------------------------------


def test_validate_flags_a_short_record_count(tmp_path: Path):
    """Validate flags an origin whose record_created count falls short of its schedule."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h0"
    events = _record_events(0, raw["schedule"]["0"])[:-1]  # drop the last record
    events.append(_snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]))
    events.append(_counters_event(evalgen.AGENT_STOP_AT_S))
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == ["record_count:h0"]


def test_validate_flags_a_record_created_at_the_wrong_scheduled_time(tmp_path: Path):
    """Validate flags a record_created event whose ts_ms does not match its own schedule at_s,
    even when it is still inside the measurement window and its seq/count are untouched."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h0"
    events = _record_events(0, raw["schedule"]["0"])
    seq = len(raw["schedule"]["0"])
    # Move the last record 20s off its own scheduled at_s -- still well
    # inside [WINDOW_START_S, WINDOW_END_S], so only the per-record
    # schedule check (not record_count/record_seq_mismatch) should fire.
    events[-1] = {**events[-1], "ts_ms": _ts_ms(evalgen.WINDOW_START_S + 20.0)}
    events.append(_snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]))
    events.append(_counters_event(evalgen.AGENT_STOP_AT_S))
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == [f"record_off_schedule:h0:seq{seq}"]


def test_validate_flags_every_record_collapsed_onto_one_instant(tmp_path: Path):
    """Validate flags every off-schedule record even when the headcount AND the seq set are
    both still exactly correct (reviewer counterexample: all of h0's records stamped at the
    same single in-window instant instead of each at its own 2s-apart tick)."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h0"
    schedule = raw["schedule"]["0"]
    last_at_s = schedule[-1]["at_s"]
    events = [{**entry, "ts_ms": _ts_ms(last_at_s)} for entry in _record_events(0, schedule)]
    events.append(_snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]))
    events.append(_counters_event(evalgen.AGENT_STOP_AT_S))
    _write_events(run_dir, node, events)
    # Every seq except the last one (whose own at_s IS last_at_s, so it
    # lands exactly on schedule by construction) is now off schedule.
    expected = sorted(f"record_off_schedule:h0:seq{i}" for i in range(1, len(schedule)))
    assert validate(str(run_dir))["flags"] == expected


def test_validate_flags_a_record_seq_mismatch_even_with_the_right_count(tmp_path: Path):
    """Validate flags an origin whose record_created events all share one seq (reviewer
    counterexample: h0's 30 records all stamped seq=1 instead of 1..30), even though the
    headcount alone still matches the schedule exactly."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h0"
    events = [{**e, "seq": 1} for e in _record_events(0, raw["schedule"]["0"])]
    events.append(_snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]))
    events.append(_counters_event(evalgen.AGENT_STOP_AT_S))
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == ["record_seq_mismatch:h0"]


def test_validate_flags_t0_undetermined_when_no_origin_created_any_record(tmp_path: Path):
    """Validate flags t0_undetermined (plus the resulting record_count misses) when h0/h4 create nothing."""
    run_dir, raw = _build_happy_run(tmp_path)
    for host in evalgen.RECORD_ORIGIN_HOSTS:
        node = f"h{host}"
        events = [_snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]), _counters_event(evalgen.AGENT_STOP_AT_S)]
        _write_events(run_dir, node, events)
    result = validate(str(run_dir))
    assert result["flags"] == sorted(["t0_undetermined", "record_count:h0", "record_count:h4"])


# -- put -----------------------------------------------------------------------


def test_validate_flags_a_failed_put(tmp_path: Path):
    """Validate flags a failed put row."""
    run_dir, raw = _build_happy_run(tmp_path)
    rows = [_put_row(raw["publisher"]["uri"], success=False)] + [_link_row(event) for event in raw["events"]]
    _write_results(run_dir, rows)
    assert validate(str(run_dir))["flags"] == ["put_failed"]


def test_validate_flags_a_missing_put_row(tmp_path: Path):
    """Validate flags a run whose results.json has no put row at all."""
    run_dir, raw = _build_happy_run(tmp_path)
    rows = [_link_row(event) for event in raw["events"]]
    _write_results(run_dir, rows)
    assert validate(str(run_dir))["flags"] == ["put_failed"]


def test_validate_flags_unreadable_results_json(tmp_path: Path):
    """Validate flags an unreadable results.json, and skips put/link checks entirely."""
    run_dir, _raw = _build_happy_run(tmp_path)
    (run_dir / "results.json").write_text("{not json", encoding="utf-8")
    assert validate(str(run_dir))["flags"] == ["results_unreadable"]


# -- link events ---------------------------------------------------------------


def test_validate_flags_a_missing_link_event_row(tmp_path: Path):
    """Validate flags a scheduled link event with no matching results.json row."""
    run_dir, raw = _build_happy_run(tmp_path)
    assert raw["events"], "p=0.2 seed=7 must schedule at least one link event for this test"
    missing = raw["events"][0]
    rows = [_put_row(raw["publisher"]["uri"])] + [_link_row(event) for event in raw["events"][1:]]
    _write_results(run_dir, rows)
    label = f"{missing['type']}:{missing['nodes'][0]}-{missing['nodes'][1]}:at{missing['at']:g}"
    assert validate(str(run_dir))["flags"] == [f"missing_link_event:{label}"]


def test_validate_flags_a_failed_link_event_row(tmp_path: Path):
    """Validate flags a link event whose results.json row reports success=False."""
    run_dir, raw = _build_happy_run(tmp_path)
    target = raw["events"][0]
    rows = [_put_row(raw["publisher"]["uri"])]
    rows += [_link_row(event, success=False) if event is target else _link_row(event) for event in raw["events"]]
    _write_results(run_dir, rows)
    label = f"{target['type']}:{target['nodes'][0]}-{target['nodes'][1]}:at{target['at']:g}"
    assert validate(str(run_dir))["flags"] == [f"link_event_failed:{label}"]


def test_validate_flags_a_late_link_event_row(tmp_path: Path):
    """Validate flags a link event whose actual_at missed scheduled_at by more than 0.5s."""
    run_dir, raw = _build_happy_run(tmp_path)
    target = raw["events"][0]
    rows = [_put_row(raw["publisher"]["uri"])]
    rows += [
        _link_row(event, actual_at=event["at"] + 0.8) if event is target else _link_row(event)
        for event in raw["events"]
    ]
    _write_results(run_dir, rows)
    label = f"{target['type']}:{target['nodes'][0]}-{target['nodes'][1]}:at{target['at']:g}"
    assert validate(str(run_dir))["flags"] == [f"link_event_late:{label}"]


def test_validate_flags_an_unexpected_link_event_row(tmp_path: Path):
    """Validate flags a results.json link row that does not correspond to any scheduled event."""
    run_dir, raw = _build_happy_run(tmp_path)
    extra = {"at": 999.0, "type": "link_down", "nodes": [0, 1]}
    rows = [_put_row(raw["publisher"]["uri"])] + [_link_row(event) for event in raw["events"]] + [_link_row(extra)]
    _write_results(run_dir, rows)
    assert validate(str(run_dir))["flags"] == ["unexpected_link_event:link_down:0-1:at999"]


# -- tracebacks -----------------------------------------------------------------


def test_validate_flags_a_traceback_in_an_agent_log(tmp_path: Path):
    """Validate flags a Python traceback found in one node's semindex_agent.log."""
    run_dir, _raw = _build_happy_run(tmp_path)
    _write_log(run_dir, "h3", "Traceback (most recent call last):\nValueError: boom\n")
    assert validate(str(run_dir))["flags"] == ["traceback:h3"]
