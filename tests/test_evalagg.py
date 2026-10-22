"""Tests for the semindex.evalagg stage-1 evaluation aggregator.

Hand-computed-expected-value coverage for every output table, plus a
smoke test against the real trial-manual sample at
/home/lab_shared/semindex-eval (skipped if that path is absent -- it is
a manually-stored fixture outside this repo, not something this test
suite can create).
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from semindex import evalgen
from semindex.check import CheckUsageError
from semindex.evalagg import (
    Sample,
    aggregate,
    analyze_run,
    derive_t0_ms,
    discover_runs,
    kaplan_meier_steps,
    km_delivered_fraction_at,
    km_quantile,
    median_q1_q3,
    read_meta,
    run_has_done_marker,
    wilson_interval,
    _node_window_control_bytes,
    _parse_p_dirname,
    _percentile,
    _row_to_sample,
    _split_lives,
)

WINDOW_START_MS = evalgen.WINDOW_START_S * 1000.0
WINDOW_END_MS = evalgen.WINDOW_END_S * 1000.0


# -- fixture helpers ----------------------------------------------------


def _node_events_path(run_dir: Path, node: str) -> Path:
    """Where semindex.analyze (and therefore semindex.evalagg) expects one node's events.jsonl."""
    return run_dir / "debug" / "node_dirs" / node / "semindex" / "events.jsonl"


def _write_jsonl(path: Path, events: list[dict]) -> None:
    """Write `events` to `path` as one JSON object per line, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def _write_run(
    run_dir: Path,
    node_events: dict[str, list[dict]],
    *,
    results: list[dict] | None = None,
    meta: dict | None = None,
    done: bool = True,
) -> None:
    """Write one synthetic run's events.jsonl per node, plus optional results.json/meta.json/DONE."""
    for node, events in node_events.items():
        _write_jsonl(_node_events_path(run_dir, node), events)
    if results is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f)
    if meta is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
    if done:
        (run_dir / "DONE").touch()


def _iso(ts_ms: float) -> str:
    """ts_ms as the ISO-8601 string semindex.check.results_ts_ms expects to parse back."""
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()


def _event_result_row(ts_ms: float, scheduled_at: float, actual_at: float, event_type: str = "link_down") -> dict:
    """One results.json op_type=="event" row, the shape _t0_from_results reads."""
    return {
        "op_type": "event",
        "event_type": event_type,
        "ts": _iso(ts_ms),
        "scheduled_at": scheduled_at,
        "actual_at": actual_at,
        "success": True,
        "event": {"at": scheduled_at, "type": event_type, "nodes": [0, 1]},
    }


def _read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# -- _parse_p_dirname / discover_runs / run_has_done_marker -------------


def test_parse_p_dirname_strips_leading_p_and_parses_float():
    """Parse p dirname strips leading p and parses float."""
    assert _parse_p_dirname("p0.2") == 0.2
    assert _parse_p_dirname("p0") == 0.0
    assert _parse_p_dirname("notp") is None
    assert _parse_p_dirname("pX") is None


def test_discover_runs_sorts_by_p_then_seed_and_ignores_non_digit_seed_dirs(tmp_path: Path):
    """Discover runs sorts by p then seed and ignores non digit seed dirs."""
    for p, seed in [(0.2, 10), (0.2, 2), (0.0, 1)]:
        (tmp_path / "stage" / f"p{p}" / str(seed)).mkdir(parents=True)
    (tmp_path / "stage" / "p0.2" / "not-a-seed").mkdir(parents=True)
    runs = discover_runs(str(tmp_path), "stage")
    assert [(r.p, r.seed) for r in runs] == [(0.0, 1), (0.2, 2), (0.2, 10)]


def test_run_has_done_marker(tmp_path: Path):
    """Run has done marker."""
    assert run_has_done_marker(str(tmp_path)) is False
    (tmp_path / "DONE").touch()
    assert run_has_done_marker(str(tmp_path)) is True


# -- read_meta ------------------------------------------------------------


def test_read_meta_missing_file_is_unknown(tmp_path: Path):
    """Read meta missing file is unknown."""
    assert read_meta(str(tmp_path)) == ("unknown", [])


def test_read_meta_top_level_valid_true(tmp_path: Path):
    """Read meta top level valid true."""
    (tmp_path / "meta.json").write_text(json.dumps({"valid": True, "flags": []}))
    assert read_meta(str(tmp_path)) == ("valid", [])


def test_read_meta_top_level_valid_false_with_flags(tmp_path: Path):
    """Read meta top level valid false with flags."""
    (tmp_path / "meta.json").write_text(json.dumps({"valid": False, "flags": ["bad_put", "late_link"]}))
    assert read_meta(str(tmp_path)) == ("flagged", ["bad_put", "late_link"])


def test_read_meta_nested_one_level_is_found(tmp_path: Path):
    """Read meta nested one level is found."""
    (tmp_path / "meta.json").write_text(json.dumps({"check": {"valid": True, "flags": ["x"]}}))
    assert read_meta(str(tmp_path)) == ("valid", ["x"])


def test_read_meta_malformed_json_is_unknown(tmp_path: Path):
    """Read meta malformed json is unknown."""
    (tmp_path / "meta.json").write_text("{not json")
    assert read_meta(str(tmp_path)) == ("unknown", [])


def test_read_meta_shape_without_valid_and_flags_is_unknown(tmp_path: Path):
    """Read meta shape without valid and flags is unknown."""
    (tmp_path / "meta.json").write_text(json.dumps({"something": "else"}))
    assert read_meta(str(tmp_path)) == ("unknown", [])


# -- numeric helpers ------------------------------------------------------


def test_percentile_single_value_returns_itself_for_any_q():
    """Percentile single value returns itself for any q."""
    assert _percentile([5.0], 0.25) == 5.0
    assert _percentile([5.0], 0.75) == 5.0


def test_percentile_linear_interpolation_on_four_values():
    """Percentile linear interpolation on four values."""
    values = [1.0, 2.0, 3.0, 4.0]
    # idx = 0.5 * 3 = 1.5 -> interpolate between values[1]=2 and values[2]=3
    assert _percentile(values, 0.5) == 2.5
    # idx = 0.25 * 3 = 0.75 -> between values[0]=1 and values[1]=2
    assert _percentile(values, 0.25) == 1.75


def test_median_q1_q3_empty_is_all_none():
    """Median q1 q3 empty is all none."""
    assert median_q1_q3([]) == (None, None, None)


def test_median_q1_q3_matches_percentile():
    """Median q1 q3 matches percentile."""
    assert median_q1_q3([1.0, 2.0, 3.0, 4.0]) == (2.5, 1.75, 3.25)


def test_wilson_interval_zero_successes_of_ten_matches_hand_computed_bound():
    """Wilson interval zero successes of ten matches hand computed bound."""
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    # Exact closed form at x=0: z^2 / (n + z^2).
    z2 = 1.959963984540054**2
    assert hi == pytest.approx(z2 / (10 + z2))


def test_wilson_interval_n_zero_is_none():
    """Wilson interval n zero is none."""
    assert wilson_interval(5, 0) is None


def test_wilson_interval_is_symmetric_around_one_half_at_x_equals_half_n():
    """Wilson interval is symmetric around one half at x equals half n."""
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(1 - hi)
    assert 0.0 < lo < 0.5 < hi < 1.0


# -- _row_to_sample (shared by survival.csv and summary.csv's own pooling) --


def test_row_to_sample_delivered_uses_the_given_key():
    """Row to sample delivered uses the given key."""
    row = {"latency_ms": 1500, "follow_up_ms": 9999, "censored": False}
    assert _row_to_sample(row, "latency_ms") == Sample(1.5, delivered=True)


def test_row_to_sample_censored_uses_follow_up_ms_regardless_of_key():
    """Row to sample censored uses follow up ms regardless of key."""
    row = {"time_ms": None, "follow_up_ms": 4500, "censored": True}
    assert _row_to_sample(row, "time_ms") == Sample(4.5, delivered=False)


# -- Kaplan-Meier ----------------------------------------------------------


def test_kaplan_meier_matches_naive_fraction_when_nothing_is_censored():
    """Kaplan meier matches naive fraction when nothing is censored."""
    samples = [Sample(1.0, True), Sample(2.0, True), Sample(3.0, True), Sample(4.0, True)]
    steps = kaplan_meier_steps(samples)
    assert km_delivered_fraction_at(steps, 2.0) == pytest.approx(0.5)
    assert km_delivered_fraction_at(steps, 4.0) == pytest.approx(1.0)


def test_kaplan_meier_censoring_discriminates_from_naive_fraction():
    """Kaplan meier censoring discriminates from naive fraction.

    Samples at durations (1, delivered), (2, censored), (3, delivered),
    (5, censored). A naive fraction-delivered-by-t=3 would read 2/4 = 0.5
    (ignoring that the censored-at-2 sample never had a chance to be
    delivered at 3). The correct product-limit answer is 0.625:
    at t=1, 4 at risk, 1 event -> S = 1 - 1/4 = 0.75.
    at t=3, 2 at risk ({3, 5}), 1 event -> S = 0.75 * (1 - 1/2) = 0.375.
    delivered_fraction(3) = 1 - 0.375 = 0.625.
    """
    samples = [Sample(1.0, True), Sample(2.0, False), Sample(3.0, True), Sample(5.0, False)]
    steps = kaplan_meier_steps(samples)
    assert km_delivered_fraction_at(steps, 3.0) == pytest.approx(0.625)
    # Before the first event, nothing has been delivered yet.
    assert km_delivered_fraction_at(steps, 0.5) == pytest.approx(0.0)
    # Between the two event times, the fraction holds at the first step's value.
    assert km_delivered_fraction_at(steps, 2.0) == pytest.approx(0.25)


def test_kaplan_meier_all_censored_never_reaches_any_quantile():
    """Kaplan meier all censored never reaches any quantile."""
    samples = [Sample(1.0, False), Sample(2.0, False)]
    steps = kaplan_meier_steps(samples)
    assert km_delivered_fraction_at(steps, 10.0) == 0.0
    assert km_quantile(steps, 0.5) is None


def test_kaplan_meier_empty_samples_is_all_zero():
    """Kaplan meier empty samples is all zero."""
    steps = kaplan_meier_steps([])
    assert km_delivered_fraction_at(steps, 10.0) == 0.0


def test_km_quantile_finds_the_first_grid_crossing():
    """Km quantile finds the first grid crossing."""
    samples = [Sample(1.0, True), Sample(2.0, True), Sample(3.0, True), Sample(4.0, True)]
    steps = kaplan_meier_steps(samples)
    assert km_quantile(steps, 0.5) == 2.0
    assert km_quantile(steps, 0.25) == 1.0
    assert km_quantile(steps, 0.75) == 3.0


# -- derive_t0_ms ----------------------------------------------------------


def test_derive_t0_ms_agent_start_is_the_first_rung_even_when_scheduler_evidence_exists():
    """Derive t0 ms agent start is the first rung even when scheduler evidence exists."""
    t0 = 1_000_000_000_000.0
    results = [_event_result_row(t0 + 20_000, scheduled_at=20.0, actual_at=20.0)]
    nodes = {
        "h0": [{"ts_ms": t0, "event": "agent_start", "node": "h0", "t0_ms": t0}],
        "h1": [{"ts_ms": t0, "event": "agent_start", "node": "h1", "t0_ms": t0}],
    }
    result = derive_t0_ms(nodes, results)
    assert result.source == "agent_start"
    assert result.t0_ms == t0
    assert result.agent_start_disagreement is False
    # Both agent_start and scheduler evidence exist and agree exactly.
    assert result.consistency_ms == pytest.approx(0.0, abs=1e-6)


def test_derive_t0_ms_agent_start_disagreement_is_flagged_not_silently_resolved():
    """Derive t0 ms agent start disagreement is flagged not silently resolved."""
    t0 = 1_000_000_000_000.0
    nodes = {
        "h0": [{"ts_ms": t0, "event": "agent_start", "node": "h0", "t0_ms": t0}],
        "h1": [{"ts_ms": t0, "event": "agent_start", "node": "h1", "t0_ms": t0 + 500.0}],
    }
    result = derive_t0_ms(nodes, results=None)
    assert result.source == "agent_start"
    assert result.agent_start_disagreement is True
    assert result.t0_ms == pytest.approx(t0 + 250.0)  # median of the two disagreeing values
    assert result.consistency_ms is None  # no scheduler evidence to cross-check against


def test_derive_t0_ms_scheduler_rung_from_results_json():
    """Derive t0 ms scheduler rung from results json."""
    t0 = 1_000_000_000_000.0
    results = [_event_result_row(t0 + 20_000, scheduled_at=20.0, actual_at=20.0)]
    nodes = {"h0": [{"ts_ms": t0 + 99_000, "event": "counters"}]}
    result = derive_t0_ms(nodes, results)
    assert result.source == "scheduler"
    assert result.t0_ms == pytest.approx(t0)
    assert result.agent_start_disagreement is False
    assert result.consistency_ms is None  # no agent_start evidence at all in this run


def test_derive_t0_ms_falls_back_to_snapshot_when_no_scheduler_events():
    """Derive t0 ms falls back to snapshot when no scheduler events."""
    t0 = 1_000_000_000_000.0
    nodes = {
        "h0": [{"ts_ms": t0 + WINDOW_START_MS, "event": "counters_snapshot"}],
        "h1": [{"ts_ms": t0 + WINDOW_START_MS, "event": "counters_snapshot"}],
    }
    result = derive_t0_ms(nodes, results=None)
    assert result.source == "snapshot"
    assert result.t0_ms == pytest.approx(t0)


def test_derive_t0_ms_falls_back_to_stop_when_no_snapshot_either():
    """Derive t0 ms falls back to stop when no snapshot either."""
    t0 = 1_000_000_000_000.0
    nodes = {"h0": [{"ts_ms": t0 + WINDOW_END_MS, "event": "counters"}]}
    result = derive_t0_ms(nodes, results=None)
    assert result.source == "stop"
    assert result.t0_ms == pytest.approx(t0)


def test_derive_t0_ms_falls_back_to_first_event_when_nothing_else_available():
    """Derive t0 ms falls back to first event when nothing else available."""
    nodes = {"h0": [{"ts_ms": 500.0, "event": "state_rx"}, {"ts_ms": 700.0, "event": "state_rx"}]}
    result = derive_t0_ms(nodes, results=None)
    assert result.source == "first_event"
    assert result.t0_ms == 500.0


def test_derive_t0_ms_raises_when_there_is_no_evidence_at_all():
    """Derive t0 ms raises when there is no evidence at all."""
    with pytest.raises(CheckUsageError) as exc_info:
        derive_t0_ms({"h0": []}, results=None)
    assert str(exc_info.value) == (
        "no results.json events, counters lines, or any other ts_ms found to derive t0 from"
    )


# -- full pipeline: hand-computed synthetic run tree -----------------------


@pytest.fixture
def synthetic_store(tmp_path: Path) -> Path:
    """A small (p, seed) run tree with every value hand-computable, covering:

    - a valid two-node run (p=0.0, seed=1): one delivered + one censored
      discovery pair, control-byte window subtraction, a window-filtered
      probe pair, a probe outside the window excluded, and one decision
      of each kind (with one more outside the window, excluded).
    - a flagged run (p=0.0, seed=2): counted in runs.csv/summary.csv's
      flagged count, excluded from survival/decisions pooling.
    - an unknown run (p=1.0, seed=1) deriving t0 via the scheduler rung,
      with a three-node line testing hop_distance=2 and an explicit
      "stored after window end counts as censored" case.
    """
    store = tmp_path / "store"
    stage = "test"

    t0_a = 1_000_000_000_000.0
    run_a = store / stage / "p0" / "1"
    _write_run(
        run_a,
        {
            "h0": [
                {"event": "record_created", "origin": "h0", "seq": 1, "ts_ms": t0_a + 15_000},
                {"event": "record_created", "origin": "h0", "seq": 2, "ts_ms": t0_a + 125_500},
                {
                    "event": "counters_snapshot",
                    "ts_ms": t0_a + 10_000,
                    "bytes_sent": 100,
                    "bytes_received": 50,
                    "control_bytes_sent": 100,
                    "control_bytes_received": 50,
                },
                {
                    "event": "counters",
                    "ts_ms": t0_a + 130_000,
                    "bytes_sent": 400,
                    "bytes_received": 250,
                    "control_bytes_sent": 400,
                    "control_bytes_received": 250,
                },
                {
                    "event": "decision",
                    "kind": "state_poll",
                    "ts_ms": t0_a + 11_000,
                    "state": {"in_flight": 0, "last_timeout_s": None, "truncated": False, "origins": {}},
                    "action": {"neighbor": "h1", "interval_s": 2.0},
                    "action_prob": 1.0,
                },
                {
                    "event": "decision",
                    "kind": "state_poll",
                    "ts_ms": t0_a + 200_000,  # outside the window: must be excluded
                    "state": {"in_flight": 5, "last_timeout_s": None, "truncated": False, "origins": {}},
                    "action": {"neighbor": "h1", "interval_s": 2.0},
                    "action_prob": 1.0,
                },
            ],
            "h1": [
                {"event": "record_stored", "origin": "h0", "seq": 1, "holder": "h0", "ts_ms": t0_a + 16_000},
                # seq 2 is never stored on h1 at all -- censored.
                {
                    "event": "counters_snapshot",
                    "ts_ms": t0_a + 10_000,
                    "bytes_sent": 80,
                    "bytes_received": 40,
                    "control_bytes_sent": 80,
                    "control_bytes_received": 40,
                },
                {
                    "event": "counters",
                    "ts_ms": t0_a + 130_000,
                    "bytes_sent": 300,
                    "bytes_received": 180,
                    "control_bytes_sent": 300,
                    "control_bytes_received": 180,
                },
                {
                    "event": "probe_discovered",
                    "ts_ms": t0_a + 20_000,
                    "matches": [{"origin": "h0", "seq": 1, "content_name": "x"}],
                },
                {
                    "event": "probe_retrieved",
                    "ts_ms": t0_a + 20_000,
                    "origin": "h0",
                    "seq": 1,
                    "content_name": "x",
                    "success": True,
                    "bytes": 123,
                },
                {
                    "event": "probe_discovered",
                    "ts_ms": t0_a + 200_000,  # outside the window: must be excluded
                    "matches": [],
                },
                {
                    "event": "decision",
                    "kind": "fetch_plan",
                    "ts_ms": t0_a + 12_000,
                    "state": {
                        "in_flight": 1,
                        "last_timeout_s": None,
                        "truncated": False,
                        "origins": {
                            # advertised h2's notified_max (5) runs 2 ahead
                            # of its own hi (3) -- advertised_gap = 2.
                            "h0": {"hi": 0, "missing": 1, "advertised": {"h2": {"hi": 3, "notified_max": 5}}}
                        },
                    },
                    "action": {"neighbor": "h0", "fetched": [["h0", 1]]},
                    "action_prob": 1.0,
                },
                {
                    "event": "decision",
                    "kind": "fetch_backoff",
                    "ts_ms": t0_a + 13_000,
                    "state": {
                        "in_flight": 0,
                        # Epoch seconds (map.md "last_timeout_s ... 壁時計の
                        # エポック秒"), 1s before this decision's own
                        # ts_ms (t0_a + 13_000) -- age = 1.0 exactly.
                        "last_timeout_s": (t0_a + 12_000) / 1000.0,
                        "truncated": False,
                        "origins": {"h0": {"hi": 0, "missing": 1, "advertised": {}}},
                    },
                    "action": {"neighbor": "h0", "origin": "h0", "seq": 1, "wait_s": 2.0, "cause": "timeout"},
                    "action_prob": 1.0,
                },
            ],
        },
        results=None,  # forces the snapshot t0 rung
        meta=None,  # "unknown" status for this run on purpose -- see test below
    )

    # Flagged run: minimal, no records, counted but excluded from pooling.
    t0_b = 1_000_000_500_000.0
    run_b = store / stage / "p0" / "2"
    _write_run(
        run_b,
        {
            "h0": [
                {
                    "event": "counters_snapshot",
                    "ts_ms": t0_b + 10_000,
                    "bytes_sent": 10,
                    "bytes_received": 5,
                    "control_bytes_sent": 10,
                    "control_bytes_received": 5,
                },
                {
                    "event": "counters",
                    "ts_ms": t0_b + 130_000,
                    "bytes_sent": 20,
                    "bytes_received": 15,
                    "control_bytes_sent": 20,
                    "control_bytes_received": 15,
                },
            ],
            "h1": [
                {
                    "event": "counters_snapshot",
                    "ts_ms": t0_b + 10_000,
                    "bytes_sent": 5,
                    "bytes_received": 2,
                    "control_bytes_sent": 5,
                    "control_bytes_received": 2,
                },
                {
                    "event": "counters",
                    "ts_ms": t0_b + 130_000,
                    "bytes_sent": 15,
                    "bytes_received": 12,
                    "control_bytes_sent": 15,
                    "control_bytes_received": 12,
                },
            ],
        },
        results=None,
        meta={"valid": False, "flags": ["bad_put"]},
    )

    # Unknown run on a second p, deriving t0 via the scheduler rung,
    # three-node line (hop_distance up to 2), one pair delivered and one
    # explicitly stored-after-window-end (still censored).
    t0_c = 2_000_000_000_000.0
    run_c = store / stage / "p1" / "1"
    _write_run(
        run_c,
        {
            "h0": [
                {"event": "record_created", "origin": "h0", "seq": 1, "ts_ms": t0_c + 15_000},
                {
                    "event": "counters_snapshot",
                    "ts_ms": t0_c + 10_000,
                    "bytes_sent": 1,
                    "bytes_received": 1,
                    "control_bytes_sent": 1,
                    "control_bytes_received": 1,
                },
                {
                    "event": "counters",
                    "ts_ms": t0_c + 130_000,
                    "bytes_sent": 2,
                    "bytes_received": 2,
                    "control_bytes_sent": 2,
                    "control_bytes_received": 2,
                },
            ],
            "h1": [
                {"event": "record_stored", "origin": "h0", "seq": 1, "holder": "h0", "ts_ms": t0_c + 16_000},
                {
                    "event": "counters_snapshot",
                    "ts_ms": t0_c + 10_000,
                    "bytes_sent": 1,
                    "bytes_received": 1,
                    "control_bytes_sent": 1,
                    "control_bytes_received": 1,
                },
                {
                    "event": "counters",
                    "ts_ms": t0_c + 130_000,
                    "bytes_sent": 2,
                    "bytes_received": 2,
                    "control_bytes_sent": 2,
                    "control_bytes_received": 2,
                },
            ],
            "h2": [
                # Stored, but AFTER the window's close -- still censored.
                {"event": "record_stored", "origin": "h0", "seq": 1, "holder": "h1", "ts_ms": t0_c + 200_000},
                {
                    "event": "counters_snapshot",
                    "ts_ms": t0_c + 10_000,
                    "bytes_sent": 1,
                    "bytes_received": 1,
                    "control_bytes_sent": 1,
                    "control_bytes_received": 1,
                },
                {
                    "event": "counters",
                    "ts_ms": t0_c + 130_000,
                    "bytes_sent": 2,
                    "bytes_received": 2,
                    "control_bytes_sent": 2,
                    "control_bytes_received": 2,
                },
            ],
        },
        results=[_event_result_row(t0_c + 5_000, scheduled_at=5.0, actual_at=5.0)],
        meta=None,
    )

    return store


def test_analyze_run_run_a_discovery_and_consistency_rows(synthetic_store: Path):
    """Analyze run run a discovery and consistency rows."""
    run_dir = synthetic_store / "test" / "p0" / "1"
    run = analyze_run(0.0, 1, str(run_dir))

    assert run.status == "unknown"
    assert run.t0_source == "snapshot"
    assert run.n_records_created == 2

    rows = {(r["origin"], r["seq"], r["node"]): r for r in run.discovery_rows}
    assert rows[("h0", 1, "h1")]["censored"] is False
    assert rows[("h0", 1, "h1")]["latency_ms"] == 1000
    assert rows[("h0", 1, "h1")]["hop_distance"] == 1
    assert rows[("h0", 2, "h1")]["censored"] is True
    assert rows[("h0", 2, "h1")]["latency_ms"] is None
    assert rows[("h0", 2, "h1")]["follow_up_ms"] == pytest.approx(4_500.0)

    by_seq = {c["seq"]: c for c in run.consistency_rows}
    assert by_seq[1]["time_ms"] == 1000
    assert by_seq[1]["censored"] is False
    assert by_seq[2]["time_ms"] is None
    assert by_seq[2]["censored"] is True


def test_analyze_run_run_a_control_bytes_window_subtraction(synthetic_store: Path):
    """Analyze run run a control bytes window subtraction."""
    run_dir = synthetic_store / "test" / "p0" / "1"
    run = analyze_run(0.0, 1, str(run_dir))
    assert run.control_bytes_sent_window == {"h0": 300, "h1": 220}
    assert run.control_bytes_received_window == {"h0": 200, "h1": 140}


def test_node_window_control_bytes_excludes_a_prior_lifes_traffic_before_the_window_opened():
    """2026-10-23 (external review): a restart before the window even opens must not

    inflate the in-window total with that prior life's own full traffic.
    The exact counterexample from review: a life ending at elapsed 5s
    (control_bytes_sent/received=40) restarts into a second life that
    holds the window-start snapshot (elapsed 10s, value=10) and the
    window-close counters (elapsed 130s, value=100). Summing every
    life's own final counters line (40 + 100) and subtracting only the
    snapshot (-10) gives a wrong 130; the correct in-window total is
    100 - 10 = 90 -- life 1's 40 was never inside the window at all.
    """
    t0 = 1_000_000_000_000.0
    events = [
        {"event": "counters", "ts_ms": t0 + 5_000, "control_bytes_sent": 40, "control_bytes_received": 40},
        {
            "event": "counters_snapshot",
            "ts_ms": t0 + 10_000,
            "control_bytes_sent": 10,
            "control_bytes_received": 10,
        },
        {"event": "counters", "ts_ms": t0 + 130_000, "control_bytes_sent": 100, "control_bytes_received": 100},
    ]
    sent, received = _node_window_control_bytes(events, window_start_ms=t0 + 10_000, window_end_ms=t0 + 130_000)
    assert (sent, received) == (90, 90)


def test_node_window_control_bytes_sums_a_middle_life_entirely_inside_the_window():
    """A life with no snapshot that starts and ends entirely inside the window (a second

    restart, after the snapshot-holding life, before the window's close)
    contributes its own FULL final value, since it started fresh at 0.
    """
    t0 = 1_000_000_000_000.0
    events = [
        {
            "event": "counters_snapshot",
            "ts_ms": t0 + 10_000,
            "control_bytes_sent": 5,
            "control_bytes_received": 5,
        },
        # Life holding the snapshot ends (restarts) at elapsed 50s.
        {"event": "counters", "ts_ms": t0 + 50_000, "control_bytes_sent": 25, "control_bytes_received": 25},
        # A fresh middle life, entirely inside the window, ends at elapsed 90s.
        {"event": "counters", "ts_ms": t0 + 90_000, "control_bytes_sent": 7, "control_bytes_received": 7},
        # A third life, also fresh, is the one active when the window closes.
        {"event": "counters", "ts_ms": t0 + 130_000, "control_bytes_sent": 3, "control_bytes_received": 3},
    ]
    sent, received = _node_window_control_bytes(events, window_start_ms=t0 + 10_000, window_end_ms=t0 + 130_000)
    # (25 - 5) + 7 + 3 = 30
    assert (sent, received) == (30, 30)


def test_node_window_control_bytes_no_snapshot_anywhere_is_no_data():
    """A node with no counters_snapshot line at all returns None (no data), not 0."""
    t0 = 1_000_000_000_000.0
    events = [{"event": "counters", "ts_ms": t0 + 130_000, "control_bytes_sent": 100, "control_bytes_received": 100}]
    assert _node_window_control_bytes(events, window_start_ms=t0 + 10_000, window_end_ms=t0 + 130_000) == (
        None,
        None,
    )


def test_split_lives_drops_a_trailing_life_with_no_terminating_counters_line():
    """A life that never cleanly shut down (no final `counters` line) is dropped, not treated as complete."""
    events = [
        {"event": "record_created", "ts_ms": 1},
        {"event": "counters", "ts_ms": 2},
        {"event": "record_created", "ts_ms": 3},  # never followed by a counters line
    ]
    lives = _split_lives(events)
    assert len(lives) == 1
    assert lives[0] == events[:2]


def test_analyze_run_run_a_probes_are_window_filtered(synthetic_store: Path):
    """Analyze run run a probes are window filtered."""
    run_dir = synthetic_store / "test" / "p0" / "1"
    run = analyze_run(0.0, 1, str(run_dir))
    assert run.probe_issued == 1
    assert run.probe_discovered == 1
    assert run.probe_retrieved == 1


def test_analyze_run_run_a_decisions_are_window_filtered(synthetic_store: Path):
    """Analyze run run a decisions are window filtered."""
    run_dir = synthetic_store / "test" / "p0" / "1"
    run = analyze_run(0.0, 1, str(run_dir))
    kinds = [d["kind"] for d in run.decision_rows]
    assert sorted(kinds) == ["fetch_backoff", "fetch_plan", "state_poll"]
    fetch_backoff = next(d for d in run.decision_rows if d["kind"] == "fetch_backoff")
    assert fetch_backoff["wait_s"] == 2.0
    assert fetch_backoff["missing"] == [1]


def test_analyze_run_run_c_derives_t0_via_scheduler_and_hop_distance_two(synthetic_store: Path):
    """Analyze run run c derives t0 via scheduler and hop distance two."""
    run_dir = synthetic_store / "test" / "p1" / "1"
    run = analyze_run(1.0, 1, str(run_dir))
    assert run.t0_source == "scheduler"
    rows = {(r["node"]): r for r in run.discovery_rows}
    assert rows["h1"]["hop_distance"] == 1
    assert rows["h1"]["censored"] is False
    assert rows["h1"]["latency_ms"] == 1000
    assert rows["h2"]["hop_distance"] == 2
    assert rows["h2"]["censored"] is True  # stored after window end


def test_aggregate_writes_all_six_csvs_with_expected_rows(synthetic_store: Path, tmp_path: Path):
    """Aggregate writes all six csvs with expected rows (--include-unknown, so the synthetic

    tree's two unknown runs are pooled -- see the sibling test below for
    the default, exclude-unknown behaviour).
    """
    outdir = tmp_path / "out"
    aggregate(str(synthetic_store), "test", str(outdir), include_undone=False, include_unknown=True)

    for name in ("runs.csv", "discovery.csv", "survival.csv", "consistency.csv", "summary.csv", "decisions.csv"):
        assert (outdir / name).is_file()

    runs = _read_csv(outdir / "runs.csv")
    by_key = {(r["p"], r["seed"]): r for r in runs}
    assert by_key[("0.0", "1")]["valid"] == "unknown"
    assert by_key[("0.0", "1")]["n_records_created"] == "2"
    assert by_key[("0.0", "1")]["n_pairs"] == "2"
    assert by_key[("0.0", "1")]["n_delivered_in_window"] == "1"
    assert by_key[("0.0", "1")]["n_censored"] == "1"
    assert by_key[("0.0", "1")]["control_bytes_sent_window_total"] == "520"
    assert by_key[("0.0", "1")]["control_bytes_received_window_total"] == "340"
    assert by_key[("0.0", "1")]["control_bytes_sent_per_record"] == "260.0"
    assert by_key[("0.0", "1")]["control_bytes_received_per_record"] == "170.0"
    assert by_key[("0.0", "2")]["valid"] == "flagged"
    assert by_key[("0.0", "2")]["flags"] == "bad_put"
    assert by_key[("0.0", "2")]["n_records_created"] == "0"
    assert by_key[("0.0", "2")]["control_bytes_sent_per_record"] == ""
    assert by_key[("1.0", "1")]["valid"] == "unknown"

    discovery = _read_csv(outdir / "discovery.csv")
    # Run b (flagged) contributed no record_created at all, so it has no
    # discovery rows; run a has 2, run c has 2 (h1, h2).
    assert len(discovery) == 4

    summary = _read_csv(outdir / "summary.csv")
    by_p = {r["p"]: r for r in summary}
    assert by_p["0.0"]["n_valid"] == "0"
    assert by_p["0.0"]["n_flagged"] == "1"
    assert by_p["0.0"]["n_unknown"] == "1"
    # Only run a (unknown, included) contributes to p=0.0's pooled stats.
    assert by_p["0.0"]["probe_issued_total"] == "1"
    assert by_p["0.0"]["probe_discovered_total"] == "1"
    assert by_p["0.0"]["probe_discovery_rate"] == "1.0"
    assert by_p["1.0"]["n_unknown"] == "1"

    decisions = _read_csv(outdir / "decisions.csv")
    by_field = {(r["p"], r["kind"], r["field"]): r for r in decisions}
    assert by_field[("0.0", "state_poll", "interval_s")]["median"] == "2.0"
    assert by_field[("0.0", "state_poll", "truncated")]["frequency"] == "0.0"
    assert by_field[("0.0", "fetch_plan", "missing")]["median"] == "1"
    assert by_field[("0.0", "fetch_plan", "advertised_gap")]["median"] == "2"  # notified_max(5) - hi(3)
    assert by_field[("0.0", "fetch_plan", "fetched_count")]["median"] == "1"
    assert by_field[("0.0", "fetch_backoff", "wait_s")]["median"] == "2.0"
    assert by_field[("0.0", "fetch_backoff", "last_timeout_age_s")]["median"] == "1.0"
    assert by_field[("0.0", "fetch_backoff", "last_timeout_age_s")]["decision_count"] == "1"

    survival = _read_csv(outdir / "survival.csv")
    # (p=0.0, hop=1): run a's two pairs, one delivered at 1s, one censored
    # at follow-up 4.5s -- run b excluded (flagged, and has no pairs
    # anyway). Grid point t=1.0 must already show the delivery.
    # KM at t=1.0: 2 at risk, 1 event -> S = 1 - 1/2 = 0.5, delivered = 0.5
    # (the other pair is censored at follow-up 4.5s and never becomes an
    # event, so it never pulls this back up to 1.0).
    group = [r for r in survival if r["p"] == "0.0" and r["hop_distance"] == "1"]
    at_t1 = next(r for r in group if r["t_s"] == "1.0")
    assert at_t1["delivered_fraction"] == "0.5"
    assert at_t1["n_pairs"] == "2"
    assert at_t1["n_censored"] == "1"
    assert at_t1["censored_fraction"] == "0.5"

    summary_p0 = by_p["0.0"]
    assert summary_p0["n_pairs"] == "2"
    assert summary_p0["n_censored_pairs"] == "1"
    assert summary_p0["pairs_censored_fraction"] == "0.5"


def test_aggregate_excludes_unknown_runs_from_pooling_by_default(synthetic_store: Path, tmp_path: Path):
    """2026-10-23 (revised): a run with no meta.json verdict at all is, by

    default, treated the same as a flagged one for every pooled table --
    counted in summary.csv's n_unknown, but contributing no sample to
    survival.csv/decisions.csv/summary.csv's own medians -- since
    `synthetic_store`'s only two runs are both "unknown", every pooled
    table comes back with no samples for either p at all, while
    runs.csv/discovery.csv (never filtered by status) still carry both.
    """
    outdir = tmp_path / "out"
    aggregate(str(synthetic_store), "test", str(outdir), include_undone=False)  # include_unknown defaults to False

    runs = _read_csv(outdir / "runs.csv")
    assert len(runs) == 3  # unaffected by include_unknown: every processed run, any status

    discovery = _read_csv(outdir / "discovery.csv")
    assert len(discovery) == 4  # unaffected by include_unknown, same as the --include-unknown case above

    summary = _read_csv(outdir / "summary.csv")
    by_p = {r["p"]: r for r in summary}
    assert by_p["0.0"]["n_unknown"] == "1"
    assert by_p["0.0"]["probe_issued_total"] == "0"
    assert by_p["0.0"]["probe_discovered_total"] == "0"
    assert by_p["0.0"]["probe_discovery_rate"] == ""  # n=0: Wilson/rate undefined
    assert by_p["1.0"]["n_unknown"] == "1"
    assert by_p["1.0"]["probe_issued_total"] == "0"

    assert _read_csv(outdir / "survival.csv") == []
    assert _read_csv(outdir / "decisions.csv") == []


def test_discovery_csv_includes_flagged_runs_rows_but_survival_excludes_them(tmp_path: Path):
    """Discovery csv includes flagged runs rows but survival excludes them."""
    store = tmp_path / "store"
    t0 = 1_000_000_000_000.0
    run_dir = store / "test" / "p0" / "1"
    _write_run(
        run_dir,
        {
            "h0": [
                {"event": "record_created", "origin": "h0", "seq": 1, "ts_ms": t0 + 15_000},
                {"event": "counters_snapshot", "ts_ms": t0 + WINDOW_START_MS, "control_bytes_sent": 1, "control_bytes_received": 1},
                {"event": "counters", "ts_ms": t0 + WINDOW_END_MS, "control_bytes_sent": 2, "control_bytes_received": 2},
            ],
            "h1": [
                {"event": "record_stored", "origin": "h0", "seq": 1, "holder": "h0", "ts_ms": t0 + 16_000},
                {"event": "counters_snapshot", "ts_ms": t0 + WINDOW_START_MS, "control_bytes_sent": 1, "control_bytes_received": 1},
                {"event": "counters", "ts_ms": t0 + WINDOW_END_MS, "control_bytes_sent": 2, "control_bytes_received": 2},
            ],
        },
        results=None,
        meta={"valid": False, "flags": ["x"]},
    )
    outdir = tmp_path / "out"
    aggregate(str(store), "test", str(outdir), include_undone=False)
    discovery = _read_csv(outdir / "discovery.csv")
    assert len(discovery) == 1
    survival = _read_csv(outdir / "survival.csv")
    assert survival == []  # the only run is flagged, so no group has any samples


# -- DONE marker / --include-undone --------------------------------------


def test_aggregate_skips_runs_without_done_marker_by_default(tmp_path: Path):
    """Aggregate skips runs without done marker by default."""
    store = tmp_path / "store"
    t0 = 1_000_000_000_000.0
    run_dir = store / "test" / "p0" / "1"
    _write_run(
        run_dir,
        {"h0": [{"event": "counters_snapshot", "ts_ms": t0 + WINDOW_START_MS, "control_bytes_sent": 1, "control_bytes_received": 1}]},
        results=None,
        meta=None,
        done=False,
    )
    outdir = tmp_path / "out"
    analyses = aggregate(str(store), "test", str(outdir), include_undone=False)
    assert analyses == []
    runs = _read_csv(outdir / "runs.csv")
    assert runs == []


def test_aggregate_include_undone_processes_runs_without_done_marker(tmp_path: Path):
    """Aggregate include undone processes runs without done marker."""
    store = tmp_path / "store"
    t0 = 1_000_000_000_000.0
    run_dir = store / "test" / "p0" / "1"
    _write_run(
        run_dir,
        {"h0": [{"event": "counters_snapshot", "ts_ms": t0 + WINDOW_START_MS, "control_bytes_sent": 1, "control_bytes_received": 1}]},
        results=None,
        meta=None,
        done=False,
    )
    outdir = tmp_path / "out"
    analyses = aggregate(str(store), "test", str(outdir), include_undone=True)
    assert len(analyses) == 1


def test_aggregate_keeps_an_unreadable_run_counted_rather_than_dropping_it(tmp_path: Path):
    """2026-10-23: a run with DONE and a `meta.json` verdict, but no events.jsonl

    anywhere, used to simply vanish from every output (analyze_run raised
    CheckUsageError, and aggregate()'s own try/except at the time dropped
    it outright) -- not even counted in n_flagged/n_unknown, silently
    under-reporting a condition's own failure rate. It must now stay in
    `analyses` (and runs.csv), with its real status/flags and
    `analysis_error` set, every metric column empty.
    """
    store = tmp_path / "store"
    broken_dir = store / "test" / "p0" / "1"
    broken_dir.mkdir(parents=True)
    (broken_dir / "DONE").touch()
    (broken_dir / "meta.json").write_text(json.dumps({"valid": False, "flags": ["agents_missing"]}))
    # No events.jsonl anywhere under broken_dir -- discover_node_events
    # (and therefore _analyze_run_metrics) raises CheckUsageError for it.
    outdir = tmp_path / "out"
    analyses = aggregate(str(store), "test", str(outdir), include_undone=False)

    assert len(analyses) == 1
    run = analyses[0]
    assert run.status == "flagged"
    assert run.flags == ["agents_missing"]
    assert run.analysis_error is not None
    assert run.n_records_created is None

    runs = _read_csv(outdir / "runs.csv")
    assert len(runs) == 1
    assert runs[0]["valid"] == "flagged"
    assert runs[0]["flags"] == "agents_missing"
    assert runs[0]["analysis_error"] != ""
    assert runs[0]["n_records_created"] == ""  # empty, not a misleading 0

    summary = _read_csv(outdir / "summary.csv")
    assert len(summary) == 1
    assert summary[0]["n_flagged"] == "1"
    assert summary[0]["n_valid"] == "0"
    assert summary[0]["n_unknown"] == "0"


# -- real-sample smoke test ------------------------------------------------

TRIAL_MANUAL_STORE = "/home/lab_shared/semindex-eval"
TRIAL_MANUAL_STAGE = "trial-manual"


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(TRIAL_MANUAL_STORE, TRIAL_MANUAL_STAGE)),
    reason="trial-manual sample not present on this machine",
)
def test_real_trial_manual_sample_aggregates_without_crashing(tmp_path: Path):
    """Real trial manual sample aggregates without crashing.

    trial-manual is exactly the "manual trial run" --include-unknown
    exists for (map.md 有効な run の判定): it has no meta.json verdict at
    all, so without that flag it would be processed into runs.csv/
    discovery.csv but pooled into nothing.
    """
    outdir = tmp_path / "out"
    analyses = aggregate(TRIAL_MANUAL_STORE, TRIAL_MANUAL_STAGE, str(outdir), include_undone=True, include_unknown=True)
    assert len(analyses) == 1
    run = analyses[0]
    assert run.status == "unknown"  # trial-manual has no meta.json
    # This sample predates agent.py's own `agent_start` event (2026-10-23)
    # and has link_down/link_up events in its results.json, so t0 falls
    # back to the next most precise rung.
    assert run.t0_source == "scheduler"
    assert run.t0_agent_start_disagreement is False
    assert run.t0_consistency_ms is None  # no agent_start evidence in this older sample
    # 60 records (30 each from h0 and h4), all well inside the window
    # (created between offsets 13s and 71s, the window being [10, 130]s),
    # each reaching all 4 non-origin nodes with nothing censored -- this
    # particular seed's disconnect schedule never kept a record out past
    # the window's close.
    assert run.n_records_created == 60
    assert len(run.discovery_rows) == 240
    assert sum(1 for r in run.discovery_rows if r["censored"]) == 0
    # Independently confirmed via `grep -c '"event": "probe_discovered"'`
    # and `grep -c '"event": "probe_retrieved"'` on h2's own events.jsonl:
    # 12 probe_discovered lines (11 with a non-empty match) and 11
    # probe_retrieved lines, of which only 7 have `"success": true` --
    # all fall inside the derived window.
    assert run.probe_issued == 12
    assert run.probe_discovered == 11
    assert run.probe_retrieved == 7

    for name in ("runs.csv", "discovery.csv", "survival.csv", "consistency.csv", "summary.csv", "decisions.csv"):
        assert (outdir / name).is_file()
    runs = _read_csv(outdir / "runs.csv")
    assert len(runs) == 1

    # --include-unknown pooled this run, so summary.csv's own counts are
    # not all zero (the point of the flag existing at all).
    summary = _read_csv(outdir / "summary.csv")
    assert len(summary) == 1
    assert summary[0]["n_unknown"] == "1"
    assert summary[0]["probe_issued_total"] == "12"
    assert summary[0]["probe_discovered_total"] == "11"
