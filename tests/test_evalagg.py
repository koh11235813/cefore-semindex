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

from semindex import evalagg, evalgen, evalrun
from semindex.check import CheckUsageError
from semindex.agent import ScheduledProbe, ScheduledRecord
from semindex.evalagg import (
    FIGURES_AVAILABLE,
    PROBE_FRESH_S,
    RunAnalysis,
    Sample,
    aggregate,
    analyze_run,
    build_decisions_rows,
    build_runs_rows,
    classify_probe,
    derive_t0_ms,
    discover_runs,
    main,
    kaplan_meier_steps,
    km_delivered_fraction_at,
    km_quantile,
    median_q1_q3,
    oldest_match_age_s,
    read_meta,
    read_meta_generation,
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
    """Read one of evalagg's own output CSVs at `path` into a list of {column_name: cell_string} dicts, one per row."""
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


def test_discover_runs_accepts_a_canonical_store_unchanged(tmp_path: Path):
    """Distinct (p, seed) per directory, as evalgen writes them: every run found, nothing refused."""
    for p_dir, seed in [("p0", 1), ("p0", 2), ("p0.1", 1), ("p0.4", 10)]:
        (tmp_path / "stage" / p_dir / str(seed)).mkdir(parents=True)
    runs = discover_runs(str(tmp_path), "stage")
    assert [(r.p, r.seed, os.path.relpath(r.run_dir, tmp_path)) for r in runs] == [
        (0.0, 1, "stage/p0/1"),
        (0.0, 2, "stage/p0/2"),
        (0.1, 1, "stage/p0.1/1"),
        (0.4, 10, "stage/p0.4/10"),
    ]


@pytest.mark.parametrize(
    ("first", "second", "p", "seed"),
    [
        (("p0", "1"), ("p0.0", "1"), "0", 1),  # two spellings of the same p
        (("p0.2", "01"), ("p0.2", "1"), "0.2", 1),  # two spellings of the same seed
    ],
)
def test_discover_runs_refuses_two_directories_with_the_same_numeric_p_and_seed(
    tmp_path: Path, first: tuple[str, str], second: tuple[str, str], p: str, seed: int
):
    """Aliased run dirs would be pooled as one run identity; refuse the store, naming both directories."""
    for p_dir, seed_dir in (first, second):
        (tmp_path / "stage" / p_dir / seed_dir).mkdir(parents=True)
    with pytest.raises(CheckUsageError) as excinfo:
        discover_runs(str(tmp_path), "stage")
    first_dir = os.path.join(str(tmp_path), "stage", *first)
    second_dir = os.path.join(str(tmp_path), "stage", *second)
    assert str(excinfo.value) == (
        f"two run directories map to the same run (p={p}, seed={seed}): {first_dir} and {second_dir}"
    )


def test_main_exits_nonzero_on_an_aliased_store_before_writing_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    """The CLI reports the aliased store on stderr, returns 1, and never creates --out."""
    for p_dir in ("p0", "p0.0"):
        _write_run(tmp_path / "store" / "test" / p_dir / "1", {}, meta={"valid": True, "flags": []})
    outdir = tmp_path / "out"
    code = main(["--store", str(tmp_path / "store"), "--stage", "test", "--out", str(outdir)])
    first_dir = os.path.join(str(tmp_path / "store"), "test", "p0", "1")
    second_dir = os.path.join(str(tmp_path / "store"), "test", "p0.0", "1")
    assert code == 1
    assert capsys.readouterr().err == (
        "semindex.evalagg: two run directories map to the same run (p=0, seed=1): "
        f"{first_dir} and {second_dir}\n"
    )
    assert not outdir.exists()


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


def test_read_meta_reads_evalruns_own_meta_json_shape(tmp_path: Path):
    """read_meta must handle `semindex.evalrun`'s own meta.json shape directly, written by its

    own `_build_meta`/`_write_json` -- not a hand-rolled guess at it.
    `_build_meta` nests the verdict one level down, under a top-level
    "validity" key, never at the top level itself. The verdict used here
    (`agent_lives`) is exactly what `semindex.evalvalid._check_agent_lives`
    now flags for a node whose life count is not exactly one (more than
    one `agent_start` line, or none at all) -- a future change to either
    producer's own shape breaks this test first, not silently inside
    read_meta.
    """
    meta = evalrun._build_meta(
        p=0.2,
        seed=1,
        stage="test",
        k=1,
        state_poll_interval_choices_s=None,
        state_poll_choice_jitter_frac=None,
        recefore="/fake/recefore",
        recefore_rc=0,
        start_s=0.0,
        end_s=1.0,
        verdict={"valid": False, "flags": ["agent_lives:h0"]},
        capture_subprocess=lambda argv, cwd=None: "",
    )
    evalrun._write_json(str(tmp_path / "meta.json"), meta)
    assert read_meta(str(tmp_path)) == ("flagged", ["agent_lives:h0"])


def test_read_meta_generation_reads_k_and_choices_from_evalruns_own_meta_json(tmp_path: Path):
    """k and the interval choices come back from the meta.json evalrun itself writes."""
    meta = evalrun._build_meta(
        p=0.4,
        seed=1,
        stage="test",
        k=4,
        state_poll_interval_choices_s=[1.0, 2.0],
        state_poll_choice_jitter_frac=None,
        recefore="/fake/recefore",
        recefore_rc=0,
        start_s=0.0,
        end_s=1.0,
        verdict={"valid": True, "flags": []},
        capture_subprocess=lambda argv, cwd=None: "",
    )
    evalrun._write_json(str(tmp_path / "meta.json"), meta)
    # No j given with choices: the default the run actually applied is recorded.
    assert read_meta_generation(str(tmp_path)) == (4, [1.0, 2.0], 0.1)


def test_read_meta_generation_is_none_for_a_stage_1_meta_json_or_none_at_all(tmp_path: Path):
    """A stage-1 meta.json (no k key) and a missing meta.json both give (None, None, None)."""
    assert read_meta_generation(str(tmp_path)) == (None, None, None)
    (tmp_path / "meta.json").write_text(json.dumps({"p": 0.2, "validity": {"valid": True, "flags": []}}))
    assert read_meta_generation(str(tmp_path)) == (None, None, None)


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


def test_node_window_control_bytes_last_life_counts_despite_a_few_ms_of_late_stop_detection():
    """2026-10-23: the agent detects crossing stop_at_s on its own poll

    tick and only then writes the final `counters` line -- a few ms of
    wall-clock lateness in the WRITE, not evidence that life started or
    ran outside the window. Counterexample: window
    [10000, 130000] ms, a restart mid-window (life 1 holds the snapshot
    and ends at elapsed 60s with value 50; life 2 is fresh, with no
    snapshot of its own, and is the LAST life -- the one structurally
    active when the window closes), whose own final counters line is
    stamped 130001 ms -- 1ms past window_end_ms. Comparing that
    timestamp against window_end_ms with no slack dropped life 2's whole
    60 outright (a wrong total of 40 instead of the correct 100).
    """
    t0 = 1_000_000_000_000.0
    events = [
        {
            "event": "counters_snapshot",
            "ts_ms": t0 + 10_000,
            "control_bytes_sent": 10,
            "control_bytes_received": 10,
        },
        {"event": "counters", "ts_ms": t0 + 60_000, "control_bytes_sent": 50, "control_bytes_received": 50},
        {"event": "counters", "ts_ms": t0 + 130_001, "control_bytes_sent": 60, "control_bytes_received": 60},
    ]
    sent, received = _node_window_control_bytes(events, window_start_ms=t0 + 10_000, window_end_ms=t0 + 130_000)
    # (50 - 10) + 60 = 100
    assert (sent, received) == (100, 100)


def test_node_window_control_bytes_last_life_slack_is_bounded_not_unlimited():
    """The late-stop-detection slack absorbs a few ms of logging lateness, not an

    unbounded one -- a last life whose final counters line lands well
    past window_end_ms (here, 1 full minute late) is too far from the
    window's close to be explained by stop-detection jitter, so it is
    not counted at all; only the snapshot life's own 40 contributes.
    """
    t0 = 1_000_000_000_000.0
    events = [
        {
            "event": "counters_snapshot",
            "ts_ms": t0 + 10_000,
            "control_bytes_sent": 10,
            "control_bytes_received": 10,
        },
        {"event": "counters", "ts_ms": t0 + 60_000, "control_bytes_sent": 50, "control_bytes_received": 50},
        {"event": "counters", "ts_ms": t0 + 190_000, "control_bytes_sent": 60, "control_bytes_received": 60},
    ]
    sent, received = _node_window_control_bytes(events, window_start_ms=t0 + 10_000, window_end_ms=t0 + 130_000)
    assert (sent, received) == (40, 40)


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


def test_build_decisions_rows_advertised_hi_and_notified_max_distinguish_same_gap():
    """2026-10-23: `advertised_gap` alone cannot tell (hi=3,

    notified_max=5) apart from (hi=30, notified_max=32) -- both have
    gap=2, so a decisions.csv built from only the gap reports an
    IDENTICAL state distribution for two runs whose neighbors are
    advertising wildly different progress. `advertised_hi`/
    `advertised_notified_max` report the two raw values too.
    """
    base_row = {
        "kind": "fetch_plan",
        "in_flight": 0,
        "truncated": False,
        "hi": [],
        "missing": [],
        "last_timeout_age_s": None,
        "wait_s": None,
        "interval_s": None,
        "fetched_count": None,
    }
    run = RunAnalysis(
        p=0.0,
        seed=1,
        status="valid",
        flags=[],
        decision_rows=[
            {**base_row, "advertised_gap": [2], "advertised_hi": [3], "advertised_notified_max": [5]},
            {**base_row, "advertised_gap": [2], "advertised_hi": [30], "advertised_notified_max": [32]},
        ],
    )
    rows = build_decisions_rows([run], include_unknown=False)
    by_field = {r["field"]: r for r in rows}
    assert by_field["advertised_gap"]["median"] == 2.0
    assert by_field["advertised_hi"]["median"] == 16.5  # (3 + 30) / 2
    assert by_field["advertised_notified_max"]["median"] == 18.5  # (5 + 32) / 2


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


def test_aggregate_writes_all_seven_csvs_with_expected_rows(synthetic_store: Path, tmp_path: Path):
    """Aggregate writes all seven csvs with expected rows (--include-unknown, so the synthetic

    tree's two unknown runs are pooled -- see the sibling test below for
    the default, exclude-unknown behaviour).
    """
    outdir = tmp_path / "out"
    aggregate(str(synthetic_store), "test", str(outdir), include_undone=False, include_unknown=True)

    for name in (
        "runs.csv",
        "discovery.csv",
        "survival.csv",
        "consistency.csv",
        "summary.csv",
        "decisions.csv",
        "probes.csv",
    ):
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
    assert by_field[("0.0", "fetch_plan", "advertised_hi")]["median"] == "3"
    assert by_field[("0.0", "fetch_plan", "advertised_notified_max")]["median"] == "5"
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


def test_aggregate_counts_runs_without_done_marker_as_incomplete_by_default(tmp_path: Path):
    """2026-10-23: `semindex.evalrun.run_one` writes DONE unconditionally after a run

    finishes its own pipeline, flagged or not -- a run dir missing DONE
    is therefore an INTERRUPTED run awaiting a redo, not a flagged one,
    and `meta.json`, if any happens to be left over from that
    interrupted attempt, is never consulted for it (its own verdict is
    not this run's real one). It must get an "incomplete" row with every
    metric column empty, never vanish from the output entirely (the
    previous behavior silently excluded exactly the runs most likely to
    need a redo from map.md's own >5%-flagged judgment).
    """
    store = tmp_path / "store"
    t0 = 1_000_000_000_000.0
    run_dir = store / "test" / "p0" / "1"
    _write_run(
        run_dir,
        {"h0": [{"event": "counters_snapshot", "ts_ms": t0 + WINDOW_START_MS, "control_bytes_sent": 1, "control_bytes_received": 1}]},
        # A leftover meta.json from the interrupted attempt, deliberately
        # saying "flagged" -- proves this verdict is never read for an
        # incomplete run.
        meta={"valid": False, "flags": ["agents_missing"]},
        done=False,
    )
    outdir = tmp_path / "out"
    analyses = aggregate(str(store), "test", str(outdir), include_undone=False)
    assert len(analyses) == 1
    assert analyses[0].status == "incomplete"
    assert analyses[0].flags == []
    assert analyses[0].analysis_error is None

    runs = _read_csv(outdir / "runs.csv")
    assert len(runs) == 1
    assert runs[0]["valid"] == "incomplete"
    assert runs[0]["flags"] == ""
    assert runs[0]["n_records_created"] == ""  # empty, not a misleading 0

    summary = _read_csv(outdir / "summary.csv")
    assert len(summary) == 1
    assert summary[0]["n_incomplete"] == "1"
    assert summary[0]["n_valid"] == "0"
    assert summary[0]["n_flagged"] == "0"
    assert summary[0]["n_unknown"] == "0"


def test_aggregate_94_valid_plus_6_incomplete_counterexample(tmp_path: Path):
    """2026-10-23 counterexample: 94 finished, valid runs (DONE present)

    alongside 6 runs an earlier, interrupted sweep never finished (no
    DONE -- regardless of whatever meta.json a half-finished attempt
    happened to leave behind). Before this fix, the 6 no-DONE runs were
    skipped outright: n_valid=94, n_flagged=0, 94 rows in runs.csv --
    silently dropping exactly the runs a redo-rate judgment needs to
    see. They must now show up as n_incomplete=6, never pooled into any
    metric, with 100 rows total.
    """
    store = tmp_path / "store"
    for seed in range(94):
        run_dir = store / "test" / "p0" / str(seed)
        _write_run(run_dir, {}, results=None, meta={"valid": True, "flags": []}, done=True)
    for seed in range(94, 100):
        run_dir = store / "test" / "p0" / str(seed)
        _write_run(
            run_dir,
            {},
            results=None,
            meta={"valid": False, "flags": ["agents_missing"]},
            done=False,
        )
    outdir = tmp_path / "out"
    analyses = aggregate(str(store), "test", str(outdir), include_undone=False)

    assert len(analyses) == 100
    assert sum(1 for r in analyses if r.status == "valid") == 94
    assert sum(1 for r in analyses if r.status == "incomplete") == 6

    runs = _read_csv(outdir / "runs.csv")
    assert len(runs) == 100

    summary = _read_csv(outdir / "summary.csv")
    assert len(summary) == 1
    assert summary[0]["n_valid"] == "94"
    assert summary[0]["n_incomplete"] == "6"
    assert summary[0]["n_flagged"] == "0"


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

    # Exact match, not merely non-empty: this run's events.jsonl is
    # missing for every node, so discover_node_events's own diagnostic
    # ("no events.jsonl found under ...") is the ONE error CheckUsageError
    # can raise here -- a substring/non-empty check would also pass for
    # an unrelated error (e.g. a crash elsewhere in this pipeline), which
    # is not evidence this specific precondition failure was caught.
    expected_diagnostic = f"no events.jsonl found under {broken_dir}/debug/node_dirs/h*/semindex/"

    assert len(analyses) == 1
    run = analyses[0]
    assert run.status == "flagged"
    assert run.flags == ["agents_missing"]
    assert run.analysis_error == expected_diagnostic
    assert run.n_records_created is None

    runs = _read_csv(outdir / "runs.csv")
    assert len(runs) == 1
    assert runs[0]["valid"] == "flagged"
    assert runs[0]["flags"] == "agents_missing"
    assert runs[0]["analysis_error"] == expected_diagnostic
    assert runs[0]["n_records_created"] == ""  # empty, not a misleading 0

    summary = _read_csv(outdir / "summary.csv")
    assert len(summary) == 1
    assert summary[0]["n_flagged"] == "1"
    assert summary[0]["n_valid"] == "0"
    assert summary[0]["n_unknown"] == "0"


# -- real-sample smoke test ------------------------------------------------

# -- probe ground truth (oldest_match_age_s / classify_probe / probes.csv) --


def _rec(at_s: float, zone: str, disaster: str = "unknown", media: str = "video") -> ScheduledRecord:
    """One config schedule entry as agent.validate_config builds it; content_name is irrelevant to matching."""
    return ScheduledRecord(at_s=at_s, content_name="ccnx:/content/x/body", zone=zone, disaster=disaster, media=media)


def _probe(
    at_s: float, zone: str | None = None, disaster: str | None = None, media: str | None = None
) -> ScheduledProbe:
    """One config probes entry as agent.validate_config builds it (None = field left out = wildcard)."""
    return ScheduledProbe(at_s=at_s, zone=zone, disaster=disaster, media=media)


def test_classify_probe_boundary_age_exactly_fresh_threshold_is_established():
    """Age exactly PROBE_FRESH_S is established; just under is fresh; no matching record is none."""
    assert PROBE_FRESH_S == 5.0
    assert classify_probe(None) == "none"
    assert classify_probe(0.0) == "fresh"
    assert classify_probe(4.999) == "fresh"
    assert classify_probe(5.0) == "established"
    assert classify_probe(60.0) == "established"


def test_oldest_match_age_s_uses_earliest_matching_record_across_origins_and_wildcards():
    """Records from two origins pooled; unspecified probe fields are wildcards; later records never count."""
    records = [
        _rec(13.0, "zone1", "flood", "video"),  # origin A
        _rec(11.0, "zone1", "collapse", "image"),  # origin B -- the earliest zone1 record
        _rec(12.0, "zone2", "flood", "video"),  # origin B
        _rec(16.0, "zone3", "fire", "sensor"),  # after every probe below but one
    ]
    assert oldest_match_age_s(_probe(15.0, zone="zone1"), records) == 4.0
    assert oldest_match_age_s(_probe(15.0, zone="zone1", disaster="flood"), records) == 2.0
    assert oldest_match_age_s(_probe(15.0, disaster="flood"), records) == 3.0
    assert oldest_match_age_s(_probe(15.0, media="video", zone="zone2"), records) == 3.0
    assert oldest_match_age_s(_probe(15.0), records) == 4.0  # all three fields wildcard
    assert oldest_match_age_s(_probe(10.0, zone="zone1"), records) is None  # scheduled only after the probe
    assert oldest_match_age_s(_probe(15.0, zone="zone3"), records) is None
    assert oldest_match_age_s(_probe(16.0, zone="zone3"), records) == 0.0  # same instant counts
    assert oldest_match_age_s(_probe(15.0, zone="zone1", media="sensor"), records) is None  # AND, not OR


def _agent_config(node: str, schedule: list[dict], probes: list[dict]) -> dict:
    """A stored agent config shaped like evalgen's own output (placeholders included) that agent.load_config accepts."""
    return {
        "cefore_conf_dir": "/placeholder-overridden-by---cefore-conf-dir",
        "counters_snapshot_at_s": [10.0],
        "group": "test",
        "neighbors": ["h1"] if node != "h1" else ["h0"],
        "node": node,
        "probes": probes,
        "schedule": [{"content_name": f"ccnx:/content/{node}/body", **entry} for entry in schedule],
        "state_dir": "/placeholder-overridden-by---state-dir",
        "stop_at_s": 130.0,
        "t0_ms": 0,
    }


def _write_configs(run_dir: Path, configs: dict[str, dict], dirname: str = "test_p0_seed1") -> None:
    """Write each node's agent config to `<run_dir>/<dirname>/<node>.json`, evalrun's own stored layout."""
    config_dir = run_dir / dirname
    config_dir.mkdir(parents=True, exist_ok=True)
    for node, config in configs.items():
        with open(config_dir / f"{node}.json", "w", encoding="utf-8") as f:
            json.dump(config, f)


T0_PROBE = 2_000_000_000_000.0


def _expected_stderr(warnings: list[str]) -> str:
    """aggregate()'s full stderr for these warning lines: each one, then the no-matplotlib notice when it applies."""
    lines = list(warnings)
    if not FIGURES_AVAILABLE:
        lines.append("semindex.evalagg: matplotlib not importable in this venv -- wrote CSVs only, no figures")
    return "".join(line + "\n" for line in lines)


# h0 creates two records; h2 runs four probes, one of each interesting
# kind. Ages (probe at_s minus earliest matching record at_s):
#   15 zone1 -> 15-13 = 2.0 fresh, discovered
#   18 zone1 -> 18-13 = 5.0 established (boundary), discovered + retrieved
#   20 zone3 -> no zone3 record anywhere -> none
#   25 zone2 -> 25-15 = 10.0 established, missed (match_count 0)
# h2's own record (zone4, at 11) matches none of the probes but proves
# h2's own schedule is pooled too: the 5th probe (h1, zone4 at 12) has
# age 1.0 fresh and comes from a node that is not h2.
_PROBE_SCHEDULES = {
    "h0": [
        {"at_s": 13.0, "zone": "zone1", "disaster": "flood", "media": "video"},
        {"at_s": 15.0, "zone": "zone2", "disaster": "fire", "media": "image"},
    ],
    "h1": [],
    "h2": [{"at_s": 11.0, "zone": "zone4", "disaster": "unknown", "media": "sensor"}],
}
_PROBE_PROBES = {
    "h0": [],
    "h1": [{"at_s": 12.0, "zone": "zone4"}],
    # Deliberately NOT in at_s order: agent.py sorts them before firing.
    "h2": [
        {"at_s": 18.0, "zone": "zone1"},
        {"at_s": 15.0, "zone": "zone1"},
        {"at_s": 20.0, "zone": "zone3"},
        {"at_s": 25.0, "zone": "zone2"},
    ],
}


def _discovered(at_s: float, matches: int) -> dict:
    """One probe_discovered line logged `at_s` seconds after T0_PROBE with `matches` dummy matches."""
    return {
        "event": "probe_discovered",
        "ts_ms": T0_PROBE + at_s * 1000 + 3,
        "matches": [{"origin": "h0", "seq": i + 1, "content_name": "ccnx:/content/h0/body"} for i in range(matches)],
    }


def _retrieved(at_s: float, success: bool) -> dict:
    """The probe_retrieved line following a matched probe_discovered."""
    return {
        "event": "probe_retrieved",
        "ts_ms": T0_PROBE + at_s * 1000 + 50,
        "origin": "h0",
        "seq": 1,
        "content_name": "ccnx:/content/h0/body",
        "success": success,
        "bytes": 10,
    }


def _probe_truth_run(run_dir: Path, *, h2_probe_events: list[dict] | None = None, configs: bool = True) -> None:
    """Write the probe-ground-truth fixture run (see _PROBE_SCHEDULES/_PROBE_PROBES) into `run_dir`."""
    start = {"event": "agent_start", "t0_ms": T0_PROBE, "ts_ms": T0_PROBE + 100}
    if h2_probe_events is None:
        h2_probe_events = [
            _discovered(15.0, 1),
            _retrieved(15.0, False),
            _discovered(18.0, 2),
            _retrieved(18.0, True),
            _discovered(20.0, 0),
            _discovered(25.0, 0),
        ]
    _write_run(
        run_dir,
        {
            "h0": [{**start, "node": "h0"}],
            "h1": [{**start, "node": "h1"}, _discovered(12.0, 1), _retrieved(12.0, True)],
            "h2": [{**start, "node": "h2"}, *h2_probe_events],
        },
        meta={"valid": True, "flags": []},
    )
    if configs:
        _write_configs(run_dir, {n: _agent_config(n, _PROBE_SCHEDULES[n], _PROBE_PROBES[n]) for n in _PROBE_SCHEDULES})


def test_analyze_run_classifies_each_probe_from_stored_configs(tmp_path: Path):
    """Each in-window probe gets its config query, oldest_match_age_s and class, paired in config at_s order."""
    run_dir = tmp_path / "store" / "test" / "p0" / "1"
    _probe_truth_run(run_dir)
    run = analyze_run(0.0, 1, str(run_dir))
    assert run.probe_truth_error is None
    assert run.probe_issued == 5
    assert run.probe_discovered == 3
    assert [
        (r["node"], r["probe_at_s"], r["zone"], r["match_count"], r["oldest_match_age_s"], r["probe_class"])
        for r in run.probe_rows
    ] == [
        ("h1", 12.0, "zone4", 1, 1.0, "fresh"),
        ("h2", 15.0, "zone1", 1, 2.0, "fresh"),
        ("h2", 18.0, "zone1", 2, 5.0, "established"),
        ("h2", 20.0, "zone3", 0, None, "none"),
        ("h2", 25.0, "zone2", 0, 10.0, "established"),
    ]
    assert [r["retrieved_success"] for r in run.probe_rows] == [True, False, True, None, None]
    assert [r["disaster"] for r in run.probe_rows] == [None] * 5
    assert run.probe_none == 1
    assert run.probe_fresh == 2
    assert run.probe_established == 2
    assert run.probe_established_discovered == 1


def test_aggregate_writes_probes_csv_and_established_summary_columns(tmp_path: Path):
    """probes.csv has one row per in-window probe; summary/runs carry the class counts and established rate."""
    store = tmp_path / "store"
    _probe_truth_run(store / "test" / "p0" / "1")
    outdir = tmp_path / "out"
    aggregate(str(store), "test", str(outdir), include_undone=False)

    probes = _read_csv(outdir / "probes.csv")
    assert list(probes[0].keys()) == [
        "p",
        "seed",
        "node",
        "probe_at_s",
        "zone",
        "disaster",
        "media",
        "discovered_ts_ms",
        "match_count",
        "retrieved_success",
        "oldest_match_age_s",
        "probe_class",
    ]
    assert probes[2] == {
        "p": "0.0",
        "seed": "1",
        "node": "h2",
        "probe_at_s": "18.0",
        "zone": "zone1",
        "disaster": "",
        "media": "",
        "discovered_ts_ms": str(T0_PROBE + 18_003),
        "match_count": "2",
        "retrieved_success": "true",
        "oldest_match_age_s": "5.0",
        "probe_class": "established",
    }
    assert [r["probe_class"] for r in probes] == ["fresh", "fresh", "established", "none", "established"]

    runs = _read_csv(outdir / "runs.csv")
    assert (runs[0]["probe_none"], runs[0]["probe_fresh"], runs[0]["probe_established"]) == ("1", "2", "2")
    assert runs[0]["probe_established_discovered"] == "1"
    assert runs[0]["probe_truth_error"] == ""

    summary = _read_csv(outdir / "summary.csv")[0]
    # Existing columns unchanged: 3 of 5 probes discovered anything.
    assert summary["probe_issued_total"] == "5"
    assert summary["probe_discovered_total"] == "3"
    assert summary["probe_discovery_rate"] == "0.6"
    assert summary["probe_none_total"] == "1"
    assert summary["probe_fresh_total"] == "2"
    assert summary["probe_established_total"] == "2"
    assert summary["probe_established_discovered"] == "1"
    assert summary["probe_established_rate"] == "0.5"
    lo, hi = wilson_interval(1, 2)
    assert summary["probe_established_rate_wilson_lo"] == str(lo)
    assert summary["probe_established_rate_wilson_hi"] == str(hi)
    assert (summary["n_runs_probe_truth"], summary["n_runs_probe_truth_missing"]) == ("1", "0")
    assert summary["probe_unclassified_total"] == "0"


def test_probe_count_mismatch_leaves_classes_empty_and_warns(tmp_path: Path, capsys: pytest.CaptureFixture):
    """A node whose probe_discovered count differs from its config probes is never guessed at: whole run unclassified."""
    store = tmp_path / "store"
    run_dir = store / "test" / "p0" / "1"
    # Only three of h2's four configured probes logged.
    _probe_truth_run(
        run_dir,
        h2_probe_events=[_discovered(15.0, 0), _discovered(18.0, 0), _discovered(20.0, 0)],
    )
    run = analyze_run(0.0, 1, str(run_dir))
    expected_error = "h2: 3 probe_discovered event(s) but 4 configured probe(s)"
    assert run.probe_truth_error == expected_error
    assert run.probe_issued == 4  # the existing, config-free counts are unaffected
    assert (run.probe_none, run.probe_fresh, run.probe_established, run.probe_established_discovered) == (
        None,
        None,
        None,
        None,
    )
    assert [r["probe_class"] for r in run.probe_rows] == [None] * 4
    assert [r["probe_at_s"] for r in run.probe_rows] == [None] * 4

    outdir = tmp_path / "out"
    aggregate(str(store), "test", str(outdir), include_undone=False)
    err = capsys.readouterr().err
    assert err == _expected_stderr([f"semindex.evalagg: {run_dir}: probe ground truth unavailable: {expected_error}"])
    runs = _read_csv(outdir / "runs.csv")
    assert runs[0]["probe_established"] == ""
    assert runs[0]["probe_truth_error"] == expected_error
    summary = _read_csv(outdir / "summary.csv")[0]
    assert summary["probe_issued_total"] == "4"
    assert summary["probe_established_total"] == ""
    assert summary["probe_established_rate"] == ""


def test_missing_configs_leave_classes_empty_and_warn(tmp_path: Path, capsys: pytest.CaptureFixture):
    """A run with no stored agent configs still aggregates; class columns are empty and stderr says why."""
    store = tmp_path / "store"
    run_dir = store / "test" / "p0" / "1"
    _probe_truth_run(run_dir, configs=False)
    run = analyze_run(0.0, 1, str(run_dir))
    expected_error = f"no agent configs found at {run_dir}/*/h*.json"
    assert run.probe_truth_error == expected_error
    assert run.probe_issued == 5
    assert run.probe_established is None
    assert [r["probe_class"] for r in run.probe_rows] == [None] * 5

    outdir = tmp_path / "out"
    aggregate(str(store), "test", str(outdir), include_undone=False)
    err = capsys.readouterr().err
    assert err == _expected_stderr([f"semindex.evalagg: {run_dir}: probe ground truth unavailable: {expected_error}"])
    probes = _read_csv(outdir / "probes.csv")
    assert len(probes) == 5
    assert [r["probe_class"] for r in probes] == [""] * 5
    assert [r["oldest_match_age_s"] for r in probes] == [""] * 5
    assert probes[0]["match_count"] == "1"
    summary = _read_csv(outdir / "summary.csv")[0]
    assert summary["probe_none_total"] == ""
    assert summary["probe_discovery_rate"] == "0.6"
    assert (summary["n_runs_probe_truth"], summary["n_runs_probe_truth_missing"]) == ("0", "1")
    assert summary["probe_unclassified_total"] == "5"


def test_config_missing_for_one_node_is_missing_configs(tmp_path: Path):
    """A node that logged events but has no config makes the ground truth incomplete, so none is claimed."""
    run_dir = tmp_path / "store" / "test" / "p0" / "1"
    _probe_truth_run(run_dir)
    (run_dir / "test_p0_seed1" / "h0.json").unlink()
    run = analyze_run(0.0, 1, str(run_dir))
    assert run.probe_truth_error == "no agent config for node(s) h0"
    assert run.probe_established is None


def _write_h0_h2_run(
    run_dir: Path, h2_events: list[dict], h2_probes: list[dict], *, lead: list[dict] | None = None
) -> None:
    """A two-node run: h0 schedules one zone1 record at 13 s, h2 runs `h2_probes` and logs `h2_events`.

    `lead` replaces the default first line of each node's file (an
    agent_start with t0 = T0_PROBE), e.g. to build a legacy run with none.
    """
    if lead is None:
        lead = [{"event": "agent_start", "t0_ms": T0_PROBE, "ts_ms": T0_PROBE + 100}]
    _write_run(
        run_dir,
        {"h0": [{**e, "node": "h0"} for e in lead], "h2": [*({**e, "node": "h2"} for e in lead), *h2_events]},
        meta={"valid": True, "flags": []},
    )
    schedule = [{"at_s": 13.0, "zone": "zone1", "disaster": "flood", "media": "video"}]
    _write_configs(run_dir, {"h0": _agent_config("h0", schedule, []), "h2": _agent_config("h2", [], h2_probes)})


def test_restart_duplicate_first_probe_and_unfired_last_is_unclassified(tmp_path: Path):
    """A re-run first probe plus a never-fired last one cancel in the counts; a second life must still block pairing.

    The crash lands between probe_discovered(15 s) and the restart-cursor
    update, so the 20 s life re-runs the 15 s probe; the 25 s probe then
    succeeds and the 125 s one never fires. Index pairing would shift the
    25 s success onto the 125 s probe and report established 0/1.
    """
    run_dir = tmp_path / "store" / "test" / "p0" / "1"
    _write_h0_h2_run(
        run_dir,
        [
            _discovered(15.0, 0),
            {"event": "agent_start", "node": "h2", "t0_ms": T0_PROBE, "ts_ms": T0_PROBE + 20_000},
            _discovered(20.0, 0),
            _discovered(25.0, 1),
            _retrieved(25.0, True),
        ],
        [{"at_s": 15.0, "zone": "zone3"}, {"at_s": 25.0, "zone": "zone1"}, {"at_s": 125.0, "zone": "zone3"}],
    )
    run = analyze_run(0.0, 1, str(run_dir))
    assert run.probe_truth_error == "h2: 2 agent_start lines (restarted), probe pairing is ambiguous"
    assert (run.probe_established, run.probe_established_discovered) == (None, None)
    assert [r["probe_class"] for r in run.probe_rows] == [None, None, None]
    assert run.probe_issued == 3
    assert run.probe_discovered == 1


def test_event_earlier_than_its_paired_probe_is_unclassified(tmp_path: Path):
    """Within one life, a probe_discovered stamped before its paired probe's scheduled instant is a wrong pairing."""
    run_dir = tmp_path / "store" / "test" / "p0" / "1"
    _write_h0_h2_run(
        run_dir,
        [_discovered(15.0, 1), _retrieved(15.0, True), _discovered(20.0, 1), _retrieved(20.0, True)],
        [{"at_s": 15.0, "zone": "zone1"}, {"at_s": 25.0, "zone": "zone1"}],
    )
    run = analyze_run(0.0, 1, str(run_dir))
    assert run.probe_truth_error == (
        f"h2: probe_discovered at ts_ms {T0_PROBE + 20_003} is earlier than its paired probe's scheduled at_s 25.0"
    )
    assert run.probe_established is None
    assert [r["probe_class"] for r in run.probe_rows] == [None, None]


def test_pairing_follows_file_order_when_the_wall_clock_steps_back(tmp_path: Path):
    """Both probes fire late (after 25 s) and the clock steps back 100 ms between them: pairing must not re-sort by ts.

    In file order the 15 s zone3 probe logged ts 25200 with no match, then
    the 25 s zone1 probe logged ts 25100 and retrieved; a ts sort would
    swap them and report the established probe as a miss (0/1).
    """
    run_dir = tmp_path / "store" / "test" / "p0" / "1"
    match = {"origin": "h0", "seq": 1, "content_name": "ccnx:/content/h0/body"}
    _write_h0_h2_run(
        run_dir,
        [
            {"event": "probe_discovered", "ts_ms": T0_PROBE + 25_200, "matches": []},
            {"event": "probe_discovered", "ts_ms": T0_PROBE + 25_100, "matches": [match]},
            {**_retrieved(0.0, True), "ts_ms": T0_PROBE + 25_110},
        ],
        [{"at_s": 15.0, "zone": "zone3"}, {"at_s": 25.0, "zone": "zone1"}],
    )
    run = analyze_run(0.0, 1, str(run_dir))
    assert run.probe_truth_error is None
    assert [(r["probe_at_s"], r["discovered_ts_ms"], r["match_count"], r["probe_class"]) for r in run.probe_rows] == [
        (15.0, T0_PROBE + 25_200, 0, "none"),
        (25.0, T0_PROBE + 25_100, 1, "established"),
    ]
    assert [r["retrieved_success"] for r in run.probe_rows] == [None, True]
    assert (run.probe_established, run.probe_established_discovered) == (1, 1)


def test_legacy_run_without_agent_start_is_classified_despite_a_late_derived_t0(tmp_path: Path):
    """No agent_start: t0 comes from the snapshot rung, 250 ms late, so the early-event check must not apply."""
    run_dir = tmp_path / "store" / "test" / "p0" / "1"
    snapshot = {
        "event": "counters_snapshot",
        "ts_ms": T0_PROBE + 10_250,
        "bytes_sent": 0,
        "bytes_received": 0,
        "control_bytes_sent": 0,
        "control_bytes_received": 0,
    }
    _write_h0_h2_run(
        run_dir,
        [_discovered(15.0, 0), _discovered(25.0, 1), _retrieved(25.0, True)],
        [{"at_s": 15.0, "zone": "zone3"}, {"at_s": 25.0, "zone": "zone1"}],
        lead=[snapshot],
    )
    run = analyze_run(0.0, 1, str(run_dir))
    assert (run.t0_source, run.t0_ms) == ("snapshot", T0_PROBE + 250)
    assert run.probe_truth_error is None
    assert [r["probe_class"] for r in run.probe_rows] == ["none", "established"]
    assert (run.probe_established, run.probe_established_discovered) == (1, 1)


def test_summary_reports_probe_truth_coverage_for_mixed_runs(tmp_path: Path):
    """One run with ground truth and one without: the established rate stands on 1 run, and summary.csv says so."""
    store = tmp_path / "store"
    _probe_truth_run(store / "test" / "p0" / "1")
    _probe_truth_run(store / "test" / "p0" / "2", configs=False)
    outdir = tmp_path / "out"
    aggregate(str(store), "test", str(outdir), include_undone=False)
    summary = _read_csv(outdir / "summary.csv")[0]
    assert {
        k: summary[k]
        for k in (
            "n_valid",
            "probe_issued_total",
            "probe_discovered_total",
            "n_runs_probe_truth",
            "n_runs_probe_truth_missing",
            "probe_unclassified_total",
            "probe_none_total",
            "probe_fresh_total",
            "probe_established_total",
            "probe_established_discovered",
            "probe_established_rate",
        )
    } == {
        "n_valid": "2",
        "probe_issued_total": "10",
        "probe_discovered_total": "6",
        "n_runs_probe_truth": "1",
        "n_runs_probe_truth_missing": "1",
        "probe_unclassified_total": "5",
        "probe_none_total": "1",
        "probe_fresh_total": "2",
        "probe_established_total": "2",
        "probe_established_discovered": "1",
        "probe_established_rate": "0.5",
    }


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

    for name in (
        "runs.csv",
        "discovery.csv",
        "survival.csv",
        "consistency.csv",
        "summary.csv",
        "decisions.csv",
        "probes.csv",
    ):
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


# -- stage 2 (stage-2 map "生成器と集計"): generator settings and poll-reply decision fields --


def test_runs_rows_carry_k_and_choices_last_and_empty_when_not_recorded():
    """runs.csv gets k, the choices (compact JSON) and the choice jitter j as its last
    three columns; a run whose meta.json does not record them gets empty cells, an
    analysis_error run included."""
    recorded = RunAnalysis(
        p=0.4,
        seed=1,
        status="valid",
        flags=[],
        k=4,
        state_poll_interval_choices_s=[1.0, 2.0],
        state_poll_choice_jitter_frac=0.1,
    )
    stage1 = RunAnalysis(p=0.4, seed=2, status="valid", flags=[])
    broken = RunAnalysis(p=0.4, seed=3, status="valid", flags=[], analysis_error="no t0", k=1)
    rows = build_runs_rows([recorded, stage1, broken])
    assert [(r["k"], r["state_poll_interval_choices_s"], r["state_poll_choice_jitter_frac"]) for r in rows] == [
        (4, "[1.0,2.0]", 0.1),
        (None, None, None),
        (1, None, None),
    ]
    assert evalagg.RUNS_FIELDS[-3:] == ["k", "state_poll_interval_choices_s", "state_poll_choice_jitter_frac"]


def _poll_decision(status, timeouts, action_prob):
    """One analyze_run-shaped state_poll decision row from a stage-2 agent."""
    return {
        "kind": "state_poll",
        "in_flight": 0,
        "truncated": False,
        "hi": [],
        "missing": [],
        "advertised_gap": [],
        "advertised_hi": [],
        "advertised_notified_max": [],
        "last_timeout_age_s": None,
        "wait_s": None,
        "interval_s": 1.0,
        "fetched_count": None,
        "has_poll_reply_state": True,
        "last_poll_status": status,
        "consecutive_poll_timeouts": timeouts,
        "action_prob": action_prob,
    }


def test_decisions_rows_report_poll_reply_status_timeouts_and_action_prob_for_stage_2():
    """For stage-2 state_poll decisions, decisions.csv adds action_prob and
    consecutive_poll_timeouts (median/Q1/Q3) and one frequency row per
    last_poll_status value, in a fixed order, after the stage-1 rows."""
    run = RunAnalysis(
        p=0.4,
        seed=1,
        status="valid",
        flags=[],
        decision_rows=[
            _poll_decision(None, 0, 1 / 3),
            _poll_decision("answered", 0, 1 / 3),
            _poll_decision("timed_out", 1, 1 / 3),
            _poll_decision("timed_out", 2, 1 / 3),
            _poll_decision("pending", 2, 1 / 3),
        ],
    )
    rows = build_decisions_rows([run], include_unknown=False)
    new_rows = [(r["field"], r["n"], r["median"], r["frequency"]) for r in rows][-6:]
    assert new_rows == [
        ("action_prob", 5, 1 / 3, None),
        ("consecutive_poll_timeouts", 5, 1.0, None),
        ("last_poll_status=null", 5, None, 0.2),
        ("last_poll_status=answered", 5, None, 0.2),
        ("last_poll_status=timed_out", 5, None, 0.4),
        ("last_poll_status=pending", 5, None, 0.2),
    ]
    assert rows[-7]["field"] == "truncated"


def test_decisions_rows_add_nothing_for_stage_1_decisions(synthetic_store: Path):
    """A stage-1 decision line (no last_poll_status key) adds none of the stage-2 rows,
    so a stage-1 store's decisions.csv keeps its bytes."""
    run = analyze_run(0.0, 1, str(synthetic_store / "test" / "p0" / "1"))
    state_poll = next(d for d in run.decision_rows if d["kind"] == "state_poll")
    assert state_poll["has_poll_reply_state"] is False
    fields = {r["field"] for r in build_decisions_rows([run], include_unknown=True)}
    assert not fields & {"action_prob", "consecutive_poll_timeouts", "last_poll_status=null"}


def test_analyze_run_reads_the_poll_reply_fields_off_a_stage_2_decision_line(tmp_path: Path):
    """analyze_run carries a stage-2 state_poll line's last_poll_status,
    consecutive_poll_timeouts and action_prob into its decision rows."""
    t0 = 1_000_000_000_000.0
    run_dir = tmp_path / "store" / "s2" / "p0" / "1"
    decision = {
        "event": "decision",
        "ts_ms": t0 + 20_000,
        "kind": "state_poll",
        "state": {
            "in_flight": 0,
            "last_timeout_s": None,
            "truncated": False,
            "origins": {},
            "last_poll_status": "timed_out",
            "consecutive_poll_timeouts": 3,
        },
        "action": {"neighbor": "h1", "interval_s": 0.5},
        "action_prob": 0.5,
    }
    _write_run(run_dir, {"h0": [{"ts_ms": t0, "event": "agent_start", "node": "h0", "t0_ms": t0}, decision]})
    [row] = analyze_run(0.0, 1, str(run_dir)).decision_rows
    assert (row["has_poll_reply_state"], row["last_poll_status"], row["consecutive_poll_timeouts"], row["action_prob"]) == (
        True,
        "timed_out",
        3,
        0.5,
    )


def test_irrational_choices_and_j_round_trip_through_meta_json_and_runs_csv(tmp_path: Path):
    """0.5 * e**n choices and j written by evalrun's own meta.json come back exactly,
    and the runs.csv cell parses back to the same floats."""
    import math

    choices = [0.5 * math.e**n for n in range(4)]
    meta = evalrun._build_meta(
        p=0.0,
        seed=1,
        stage="test",
        k=1,
        state_poll_interval_choices_s=choices,
        state_poll_choice_jitter_frac=0.05,
        recefore="/fake/recefore",
        recefore_rc=0,
        start_s=0.0,
        end_s=1.0,
        verdict={"valid": True, "flags": []},
        capture_subprocess=lambda argv, cwd=None: "",
    )
    evalrun._write_json(str(tmp_path / "meta.json"), meta)
    k, read_choices, jitter = read_meta_generation(str(tmp_path))
    assert (k, read_choices, jitter) == (1, choices, 0.05)
    row = build_runs_rows(
        [RunAnalysis(p=0.0, seed=1, status="valid", flags=[], k=k, state_poll_interval_choices_s=read_choices,
                     state_poll_choice_jitter_frac=jitter)]
    )[0]
    assert json.loads(row["state_poll_interval_choices_s"]) == choices
