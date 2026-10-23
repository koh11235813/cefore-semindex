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


def _agent_start_event(node: str, t0_ms: float = _T0_MS) -> dict:
    """The first line every real agent life writes (semindex.agent.Agent.__init__), carrying the
    literal `t0_ms` it was launched with. EventWriter.write stamps every event, this one included,
    with its own wall-clock `ts_ms` -- this fixture reuses `t0_ms` as that stamp, since a real agent
    writes this line within a tick or two of its own launch."""
    return {"ts_ms": round(t0_ms), "event": "agent_start", "node": node, "t0_ms": t0_ms}


def _record_events(host: int, schedule_entries: list[dict]) -> list[dict]:
    node = f"h{host}"
    return [
        {"ts_ms": _ts_ms(entry["at_s"]), "event": "record_created", "origin": node, "seq": i + 1}
        for i, entry in enumerate(schedule_entries)
    ]


def _snapshot_event(elapsed_s: float) -> dict:
    return {"ts_ms": _ts_ms(elapsed_s), "event": "counters_snapshot", **_counters_fields()}


def _counters_event(elapsed_s: float, **overrides) -> dict:
    return {"ts_ms": _ts_ms(elapsed_s), "event": "counters", **_counters_fields(), **overrides}


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
        events = [_agent_start_event(node)]
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
    assert validate(str(run_dir)) == {
        "valid": True,
        "flags": [],
        "metrics": {"record_late_max_s": 0.0, "record_early_count": 0},
    }


def test_validate_accepts_p0_with_no_link_events_at_all(tmp_path: Path):
    """Validate accepts p0 with no link events at all (t0 still derivable from record_created)."""
    run_dir, raw = _build_happy_run(tmp_path, p=0.0, seed=1)
    assert raw["events"] == []
    assert validate(str(run_dir)) == {
        "valid": True,
        "flags": [],
        "metrics": {"record_late_max_s": 0.0, "record_early_count": 0},
    }


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
    agent_start, counters_snapshot and counters lines too)."""
    run_dir, _raw = _build_happy_run(tmp_path)
    (run_dir / "debug" / "node_dirs" / "h1" / "semindex" / "events.jsonl").write_text("", encoding="utf-8")
    assert validate(str(run_dir))["flags"] == sorted(
        ["empty_events:h1", "counters_snapshot_count:h1", "counters_count:h1", "agent_lives:h1"]
    )


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
    events = [_agent_start_event(node), _counters_event(evalgen.AGENT_STOP_AT_S)]
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == ["counters_snapshot_count:h1"]


def test_validate_flags_a_duplicate_counters_snapshot(tmp_path: Path):
    """Validate flags a duplicate counters_snapshot (two, not one)."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h1"
    events = [
        _agent_start_event(node),
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
        _agent_start_event(node),
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
        _agent_start_event(node),
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
        _agent_start_event(node),
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
    events = [_agent_start_event(node)] + _record_events(0, raw["schedule"]["0"])[:-1]  # drop the last record
    events.append(_snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]))
    events.append(_counters_event(evalgen.AGENT_STOP_AT_S))
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == ["record_count:h0"]


def test_validate_flags_a_record_created_at_the_wrong_scheduled_time(tmp_path: Path):
    """Validate flags a record_created event whose ts_ms does not match its own schedule at_s,
    even when it is still inside the measurement window and its seq/count are untouched."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h0"
    events = [_agent_start_event(node)] + _record_events(0, raw["schedule"]["0"])
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
    events = [_agent_start_event(node)] + [
        {**entry, "ts_ms": _ts_ms(last_at_s)} for entry in _record_events(0, schedule)
    ]
    events.append(_snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]))
    events.append(_counters_event(evalgen.AGENT_STOP_AT_S))
    _write_events(run_dir, node, events)
    # Every seq except the last is off schedule: the last one's own at_s IS last_at_s, so it lands
    # exactly on schedule by construction. The second-to-last is off by one RECORD_INTERVAL_S
    # (2.0s), which must still be flagged -- the slack is kept below the interval for this reason.
    expected = sorted(f"record_off_schedule:h0:seq{i}" for i in range(1, len(schedule)))
    assert validate(str(run_dir))["flags"] == expected


def test_validate_flags_a_missing_seq_without_crashing(tmp_path: Path):
    """Validate flags (rather than crashes on) a record_created event with no `seq` field at
    all -- sorting a mix of int and non-int seqs would otherwise raise before any flag could
    be reported."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h0"
    events = [_agent_start_event(node)] + _record_events(0, raw["schedule"]["0"])
    del events[1]["seq"]  # events[0] is agent_start; the first record_created is at index 1
    events.append(_snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]))
    events.append(_counters_event(evalgen.AGENT_STOP_AT_S))
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == ["record_seq_mismatch:h0"]


def test_validate_flags_a_record_seq_mismatch_even_with_the_right_count(tmp_path: Path):
    """Validate flags an origin whose record_created events all share one seq (reviewer
    counterexample: h0's 30 records all stamped seq=1 instead of 1..30), even though the
    headcount alone still matches the schedule exactly."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h0"
    events = [_agent_start_event(node)] + [{**e, "seq": 1} for e in _record_events(0, raw["schedule"]["0"])]
    events.append(_snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]))
    events.append(_counters_event(evalgen.AGENT_STOP_AT_S))
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == ["record_seq_mismatch:h0"]


def test_validate_flags_t0_undetermined_when_no_agent_start_or_record_evidence_exists_anywhere(tmp_path: Path):
    """Validate flags t0_undetermined only when NEITHER t0 rung has anything to go on at all: no node
    wrote an agent_start line, AND no origin ever created a record (plus the resulting record_count
    and agent_lives misses). A run missing agent_start on just one node, or records on just one origin,
    still has other evidence (agent_start elsewhere) to anchor t0 with -- this is the one case left
    where the fallback chain itself comes up empty, now that agent_start is the primary rung."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    raw = _write_experiment(run_dir, p=0.2, seed=7, stage="s1")
    for host in raw["agents"]:
        node = f"h{host}"
        events = [_snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]), _counters_event(evalgen.AGENT_STOP_AT_S)]
        _write_events(run_dir, node, events)
        _write_log(run_dir, node, "")
    rows = [_put_row(raw["publisher"]["uri"])] + [_link_row(event) for event in raw["events"]]
    _write_results(run_dir, rows)

    result = validate(str(run_dir))
    expected = sorted(
        ["t0_undetermined", "record_count:h0", "record_count:h4"]
        + [f"agent_lives:h{host}" for host in raw["agents"]]
    )
    assert result["flags"] == expected


def test_validate_flags_t0_estimated_when_falling_back_to_the_record_median(tmp_path: Path):
    """Validate flags t0_estimated when no node wrote an agent_start line at all, so t0 falls back to
    the record_created median estimate -- that weaker rung must never be used silently."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    raw = _write_experiment(run_dir, p=0.2, seed=7, stage="s1")
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

    result = validate(str(run_dir))
    # Every node here wrote zero agent_start lines (the whole point of this fixture), which the
    # single-life check flags independently of which t0 rung won.
    expected = sorted(["t0_estimated"] + [f"agent_lives:h{host}" for host in raw["agents"]])
    assert result["flags"] == expected


def test_validate_flags_a_t0_agent_start_disagreement(tmp_path: Path):
    """Validate flags t0_agent_start_disagreement when nodes' own agent_start.t0_ms values do not
    agree exactly -- map.md launches every node's agent with the SAME t0_ms, so any difference is
    itself an anomaly worth surfacing, not something to silently average away (median) over."""
    run_dir, raw = _build_happy_run(tmp_path)
    path = run_dir / "debug" / "node_dirs" / "h2" / "semindex" / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert first == {"ts_ms": _T0_MS, "event": "agent_start", "node": "h2", "t0_ms": _T0_MS}
    first["t0_ms"] = _T0_MS + 50
    lines[0] = json.dumps(first)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # The median of {T0, T0, T0, T0, T0+50} is still exactly T0, so nothing else this run checks
    # against t0 shifts -- only the disagreement itself is new.
    assert validate(str(run_dir))["flags"] == ["t0_agent_start_disagreement"]


def test_validate_flags_a_uniformly_delayed_schedule_when_agent_start_is_present(tmp_path: Path):
    """Validate flags every one of h0's records when its whole schedule is uniformly delayed by
    2.5s, even though agent_start is present and every node agrees on t0 exactly.

    A t0 estimate taken from the median of these same records (the OLD, and now fallback-only,
    rung) shifts by that same 2.5s and the error cancels out of every per-record residual -- a
    delay large enough to matter (beyond the implementation's own 2.0s tolerance, the orchestrator's
    2026-10-23 widened `RECORD_SCHEDULE_SLACK_S`) passed outright. agent_start.t0_ms is independent
    evidence the agent itself wrote at its real launch time, unaffected by anything that happens to
    its records afterward, so comparing against it exposes exactly this delay instead of absorbing
    it.
    """
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h0"
    schedule = raw["schedule"]["0"]
    delayed = [{**e, "ts_ms": e["ts_ms"] + 2500} for e in _record_events(0, schedule)]
    events = [_agent_start_event(node)] + delayed
    events.append(_snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]))
    events.append(_counters_event(evalgen.AGENT_STOP_AT_S))
    _write_events(run_dir, node, events)

    result = validate(str(run_dir))
    expected = sorted(f"record_off_schedule:h0:seq{i}" for i in range(1, len(schedule) + 1))
    assert result["flags"] == expected
    assert result["metrics"]["record_late_max_s"] == pytest.approx(2.5)
    assert result["valid"] is False


# -- record-schedule metrics (orchestrator decision: report timing as numbers, not just flags) --


def test_validate_reports_zero_record_metrics_on_an_exactly_on_schedule_run(tmp_path: Path):
    """Validate reports record_late_max_s=0.0 and record_early_count=0 on a run whose records all
    land exactly on their own scheduled at_s -- metrics are always present, not just when
    something is flagged."""
    run_dir, _raw = _build_happy_run(tmp_path)
    metrics = validate(str(run_dir))["metrics"]
    assert metrics == {"record_late_max_s": 0.0, "record_early_count": 0}


def test_validate_reports_lateness_within_the_widened_tolerance_as_a_metric_while_staying_valid(
    tmp_path: Path,
):
    """A record 1.5s late (inside the new 2.0s RECORD_SCHEDULE_SLACK_S, so not flagged) still
    shows up in metrics -- the whole point of reporting these as numbers instead of only a flag is
    that a run can be valid and still have its actual closeness to schedule visible."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h0"
    schedule = raw["schedule"]["0"]
    events = [_agent_start_event(node)] + _record_events(0, schedule)
    events[-1] = {**events[-1], "ts_ms": events[-1]["ts_ms"] + 1500}
    events.append(_snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]))
    events.append(_counters_event(evalgen.AGENT_STOP_AT_S))
    _write_events(run_dir, node, events)

    result = validate(str(run_dir))
    assert result["flags"] == []
    assert result["valid"] is True
    assert result["metrics"]["record_late_max_s"] == pytest.approx(1.5)
    assert result["metrics"]["record_early_count"] == 0


def test_validate_reports_an_early_record_in_record_early_count(tmp_path: Path):
    """A record 1.5s EARLY (also inside tolerance) is counted in record_early_count, and does not
    contribute to record_late_max_s (which stays 0.0 -- nothing was ever late)."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h0"
    schedule = raw["schedule"]["0"]
    events = [_agent_start_event(node)] + _record_events(0, schedule)
    events[-1] = {**events[-1], "ts_ms": events[-1]["ts_ms"] - 1500}
    events.append(_snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]))
    events.append(_counters_event(evalgen.AGENT_STOP_AT_S))
    _write_events(run_dir, node, events)

    result = validate(str(run_dir))
    assert result["flags"] == []
    assert result["metrics"]["record_late_max_s"] == 0.0
    assert result["metrics"]["record_early_count"] == 1


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


# -- agent lives / byte-counter monotonicity (review r2) -----------------------


def test_validate_flags_a_missing_agent_start(tmp_path: Path):
    """Validate flags a node with zero agent_start lines (e.g. it crashed before writing one)."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h1"
    events = [_snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]), _counters_event(evalgen.AGENT_STOP_AT_S)]
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == ["agent_lives:h1"]


def test_validate_flags_two_agent_starts_even_when_counters_do_not_decrease(tmp_path: Path):
    """Validate flags agent_lives on two agent_start lines even when the final counters line happens
    not to be numerically lower than the snapshot -- the life count itself, not just whether the byte
    values dipped, is what stage 1's single-life invariant rests on."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h1"
    events = [
        _agent_start_event(node),
        _snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]),
        _agent_start_event(node),  # an unplanned restart, with nothing numerically wrong about its counters
        _counters_event(evalgen.AGENT_STOP_AT_S),
    ]
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == ["agent_lives:h1"]


def test_validate_flags_decreasing_byte_counters_independently_of_agent_lives(tmp_path: Path):
    """Validate flags a decrease in any byte counter from counters_snapshot to the final counters
    line on its own (a second, independent line of defense) -- map.md's control-traffic metric, final
    minus snapshot, must never go negative, whatever caused the decrease."""
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h1"
    events = [
        _agent_start_event(node),
        _snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]),
        _counters_event(evalgen.AGENT_STOP_AT_S, bytes_received=5),
    ]
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == ["counters_decreased:h1"]


def test_validate_flags_the_sigkill_restart_counterexample(tmp_path: Path):
    """Validate flags both agent_lives and counters_decreased for a SIGKILL-between-lines shape:
    h1 writes its counters_snapshot (control_bytes_sent=100) in its first life, is
    SIGKILLed before that life ever writes a final `counters` line, restarts (a second agent_start),
    and only the NEW life writes the final `counters` line -- with control_bytes_sent reset to 50 by
    the restart (CefpycoTransport's byte counters are process-local, reset to 0 every life). Exactly
    one counters_snapshot and exactly one counters line in total still passes `_check_counters`'s own
    counts outright; this is the shape only agent_lives and counters_decreased catch.
    """
    run_dir, raw = _build_happy_run(tmp_path)
    node = "h1"
    events = [
        _agent_start_event(node),  # life 1
        _snapshot_event(evalgen.AGENT_COUNTERS_SNAPSHOT_AT_S[0]),  # life 1's snapshot: control_bytes_sent=100
        # life 1 is SIGKILLed here, before it ever writes a final `counters` line.
        _agent_start_event(node),  # life 2, after the unplanned restart
        _counters_event(evalgen.AGENT_STOP_AT_S, control_bytes_sent=50),  # life 2's own (reset) final line
    ]
    _write_events(run_dir, node, events)
    assert validate(str(run_dir))["flags"] == ["agent_lives:h1", "counters_decreased:h1"]
