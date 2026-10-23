"""Tests for semindex.evalfig, the rows -> figure-files step of the stage-1 aggregator.

Skipped as a whole where matplotlib is not importable: evalfig is only
ever called when semindex.evalagg.FIGURES_AVAILABLE is true, so there is
nothing to test without it. The aggregate-level test reuses
test_evalagg's own hand-made synthetic store, so the figures are checked
against exactly the runs whose CSV values that suite already pins.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from semindex import evalfig  # noqa: E402
from semindex import evalagg  # noqa: E402
from semindex.evalagg import aggregate  # noqa: E402

from test_evalagg import WINDOW_END_MS, WINDOW_START_MS, _write_run  # noqa: E402
from test_evalagg import synthetic_store  # noqa: E402,F401  (pytest fixture, used by name)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

FIGURE_STEMS = ["fig_discovery_survival", "fig_consistency", "fig_probe_established", "fig_control_bytes"]


def _assert_png(path: Path) -> None:
    """`path` exists and starts with the PNG signature followed by more bytes."""
    data = path.read_bytes()
    assert data[:8] == PNG_SIGNATURE
    assert len(data) > len(PNG_SIGNATURE)


def _texts(fig) -> list[str]:
    """Every text artist's string on every axes of `fig` (titles and tick labels excluded)."""
    return [t.get_text() for ax in fig.axes for t in ax.texts]


def _survival_row(p: float, hop: int, t_s: float, frac: float, n_pairs: int, n_censored: int) -> dict:
    """One survival.csv-shaped row (the quantile columns are irrelevant to the figure and left None)."""
    return {
        "p": p,
        "hop_distance": hop,
        "t_s": t_s,
        "delivered_fraction": frac,
        "n_pairs": n_pairs,
        "n_censored": n_censored,
        "censored_fraction": n_censored / n_pairs,
        "median_s": None,
        "q1_s": None,
        "q3_s": None,
    }


def _summary_row(p: float, **overrides) -> dict:
    """One summary.csv-shaped row carrying only the columns evalfig reads, all None unless overridden."""
    row = {
        "p": p,
        "consistency_time_median_ms": None,
        "control_bytes_sent_per_record_median": None,
        "control_bytes_sent_per_record_q1": None,
        "control_bytes_sent_per_record_q3": None,
        "n_runs_probe_truth": 0,
        "n_runs_probe_truth_missing": 0,
        "probe_established_total": None,
        "probe_established_discovered": None,
        "probe_established_rate": None,
        "probe_established_rate_wilson_lo": None,
        "probe_established_rate_wilson_hi": None,
    }
    row.update(overrides)
    return row


def test_aggregate_writes_the_four_figures_as_png(synthetic_store: Path, tmp_path: Path):  # noqa: F811
    """With matplotlib importable, aggregate writes every figure next to the CSVs, each a real PNG."""
    outdir = tmp_path / "out"
    aggregate(str(synthetic_store), "test", str(outdir), include_undone=False, include_unknown=True)
    for stem in FIGURE_STEMS:
        _assert_png(outdir / f"{stem}.png")
        assert (outdir / f"{stem}.pdf").stat().st_size > 0


def test_write_figures_returns_every_path_it_wrote(tmp_path: Path):
    """write_figures reports exactly the PNG+PDF pair per figure, in figure order."""
    paths = evalfig.write_figures(str(tmp_path), [], [], [], [])
    expected = [str(tmp_path / f"{stem}.{ext}") for stem in FIGURE_STEMS for ext in ("png", "pdf")]
    assert paths == expected
    assert [Path(path).is_file() for path in paths] == [True] * len(expected)


def test_empty_rows_still_write_every_figure_with_a_no_data_note(tmp_path: Path):
    """No rows at all (e.g. a stage with no included run) -> four valid PNGs, each saying so instead of crashing."""
    evalfig.write_figures(str(tmp_path), [], [], [], [])
    for stem in FIGURE_STEMS:
        _assert_png(tmp_path / f"{stem}.png")
    assert _texts(evalfig.discovery_survival_figure([])) == ["no data"]
    assert _texts(evalfig.consistency_figure([], [])) == ["no data"]
    assert _texts(evalfig.probe_established_figure([])) == ["no data"]
    assert _texts(evalfig.control_bytes_figure([], [])) == ["no data"]


def test_probe_figure_tells_no_ground_truth_apart_from_no_established_probes():
    """Rate None has two causes, each labelled as itself: no run had ground truth, or none of its probes was established."""
    rows = [
        _summary_row(
            0.0,
            n_runs_probe_truth=2,
            probe_established_total=20,
            probe_established_discovered=19,
            probe_established_rate=0.95,
            probe_established_rate_wilson_lo=0.76,
            probe_established_rate_wilson_hi=0.99,
        ),
        _summary_row(0.2, n_runs_probe_truth=0, n_runs_probe_truth_missing=3),
        _summary_row(
            0.4,
            n_runs_probe_truth=2,
            probe_none_total=1,
            probe_fresh_total=1,
            probe_established_total=0,
            probe_established_discovered=0,
        ),
    ]
    fig = evalfig.probe_established_figure(rows)
    assert _texts(fig) == [
        "19/20\n2 runs",
        "no ground truth\n(0 of 3 runs)",
        "no established probes\n(2 runs with ground truth)",
    ]


def test_consistency_figure_all_censored_p_has_no_box_but_says_so():
    """A p whose every record is censored is reported as such rather than silently missing from the plot."""
    consistency = [
        {"p": 0.0, "seed": 1, "origin": "h0", "seq": 1, "time_ms": 1000.0, "censored": False},
        {"p": 0.0, "seed": 1, "origin": "h0", "seq": 2, "time_ms": 3000.0, "censored": False},
        {"p": 0.4, "seed": 1, "origin": "h0", "seq": 1, "time_ms": None, "censored": True},
        {"p": 0.4, "seed": 1, "origin": "h0", "seq": 2, "time_ms": None, "censored": True},
    ]
    summary = [_summary_row(0.0, consistency_time_median_ms=1000.0), _summary_row(0.4)]
    fig = evalfig.consistency_figure(consistency, summary)
    assert _texts(fig) == ["0/2 censored", "all 2 censored"]
    (ax,) = fig.axes
    # One box only (p=0), spanning Q1..Q3 of its delivered times 1 s and 3 s.
    assert [(b.get_x() + b.get_width() / 2, b.get_y(), b.get_height()) for b in ax.patches] == [(0.0, 1.5, 1.0)]
    # The KM median marker comes from summary rows: present for p=0 only.
    medians = [(ln.get_xdata().tolist(), ln.get_ydata().tolist()) for ln in ax.get_lines() if ln.get_marker() == "D"]
    assert medians == [([0.0], [1.0])]


def test_survival_figure_puts_each_groups_censored_count_in_its_legend():
    """Each (p, hop) line's legend entry carries that group's own censored/pairs count from survival rows."""
    rows = [
        _survival_row(0.0, 1, 0.0, 0.0, 4, 0),
        _survival_row(0.0, 1, 0.5, 1.0, 4, 0),
        _survival_row(0.4, 1, 0.0, 0.0, 4, 4),
        _survival_row(0.4, 1, 0.5, 0.0, 4, 4),
        _survival_row(0.0, 2, 0.0, 0.0, 6, 1),
        _survival_row(0.0, 2, 0.5, 0.8, 6, 1),
    ]
    fig = evalfig.discovery_survival_figure(rows)
    legends = [[t.get_text() for t in ax.get_legend().get_texts()] for ax in fig.axes]
    assert legends == [
        ["p=0 (0/4 censored)", "p=0.4 (4/4 censored)"],
        ["p=0 (1/6 censored)"],
    ]
    assert [ax.get_title() for ax in fig.axes] == ["hop distance 1", "hop distance 2"]
    assert [ln.get_ydata().tolist() for ln in fig.axes[0].get_lines()] == [[0.0, 1.0], [0.0, 0.0]]


def test_control_bytes_figure_skips_runs_with_no_per_record_value():
    """A run whose control_bytes_sent_per_record is None or absent (analysis_error row) contributes no point."""
    runs = [
        {"p": 0.0, "seed": 1, "control_bytes_sent_per_record": 260.0},
        {"p": 0.0, "seed": 2, "control_bytes_sent_per_record": None},
        {"p": 0.0, "seed": 3, "control_bytes_sent_per_record": 280.0},
        # build_runs_rows' shape for a run with analysis_error: no metric keys at all.
        {"p": 0.0, "seed": 4, "valid": "valid", "flags": "", "analysis_error": "no events"},
    ]
    summary = [
        _summary_row(
            0.0,
            control_bytes_sent_per_record_median=270.0,
            control_bytes_sent_per_record_q1=265.0,
            control_bytes_sent_per_record_q3=275.0,
        )
    ]
    fig = evalfig.control_bytes_figure(runs, summary)
    (ax,) = fig.axes
    run_points = ax.collections[0].get_offsets()
    assert [float(y) for _, y in run_points] == [260.0, 280.0]
    (median_bar,) = ax.containers
    data_line, _caps, (bar_lines,) = median_bar.lines
    assert data_line.get_ydata().tolist() == [270.0]
    assert [[float(y) for _, y in seg] for seg in bar_lines.get_segments()] == [[265.0, 275.0]]


def test_probe_figure_tolerates_a_wilson_bound_a_float_hair_below_a_rate_of_one():
    """Rate 1.0 with wilson_hi 0.9999999999999999 (what wilson_interval really returns) draws without raising."""
    rows = [
        _summary_row(
            0.0,
            n_runs_probe_truth=10,
            probe_established_total=110,
            probe_established_discovered=110,
            probe_established_rate=1.0,
            probe_established_rate_wilson_lo=0.9662560647018358,
            probe_established_rate_wilson_hi=0.9999999999999999,
        )
    ]
    fig = evalfig.probe_established_figure(rows)
    assert _texts(fig) == ["110/110\n10 runs"]
    (ax,) = fig.axes
    (bar,) = ax.containers
    data_line, _caps, (bar_lines,) = bar.lines
    assert data_line.get_xydata().tolist() == [[0.0, 1.0]]
    # Lower end at the Wilson bound; upper error length clamped to 0, so it ends at the rate.
    ((x_lo, y_lo), (x_hi, y_hi)) = bar_lines.get_segments()[0]
    assert (x_lo, x_hi, y_hi) == (0.0, 0.0, 1.0)
    assert y_lo == pytest.approx(0.9662560647018358, abs=1e-15)


def _curve(p: float, hop: int, fracs: list[float]) -> list[dict]:
    """survival rows for one (p, hop) curve on a 1 s grid, delivered fraction fracs[i] at t=i."""
    return [_survival_row(p, hop, float(t), f, 10, 0) for t, f in enumerate(fracs)]


def test_survival_x_limit_is_where_every_curve_is_within_one_percent_of_final_rounded_to_5s():
    """x max = earliest t with every curve >= 0.99 x its final value (here 12 s), rounded up to 15 s.

    The slow curve creeps from 0.995 to 1.0 only at t=30, so a "full
    plateau" rule would give 35 s instead; the window runs to 60 s, so
    the cap does not decide it either.
    """
    fast = [min(1.0, t / 7) for t in range(61)]
    slow = [min(0.995, t / 12 * 0.99) if t < 30 else 1.0 for t in range(61)]
    fig = evalfig.discovery_survival_figure(_curve(0.0, 1, fast) + _curve(0.4, 2, slow))
    assert [ax.get_xlim() for ax in fig.axes] == [(0.0, 15.0), (0.0, 15.0)]


def test_p_keeps_its_colour_and_dash_whichever_other_p_values_the_stage_has():
    """p=0.2 is drawn identically alone and alongside p=0 and p=0.4 (style keyed by value, not rank)."""
    flat = [1.0, 1.0]

    def style_of_p02(ps: list[float]) -> tuple:
        """p=0.2's line colour and dash pattern in a survival figure drawn for the p values `ps`."""
        rows = [row for p in ps for row in _curve(p, 1, flat)]
        (ax,) = evalfig.discovery_survival_figure(rows).axes
        line = ax.get_lines()[ps.index(0.2)]
        # Line2D has no public getter for a custom dash pattern (get_linestyle
        # reports every one as "--"), so the unscaled pattern is read directly.
        return line.get_color(), line._unscaled_dash_pattern

    assert style_of_p02([0.2]) == style_of_p02([0.0, 0.2, 0.4])


def _unknown_only_store(store: Path) -> None:
    """A stage whose only run has no meta.json (status unknown) but real events: one delivered pair."""
    t0 = 1_000_000_000_000.0
    counters = [
        {"event": "counters_snapshot", "ts_ms": t0 + WINDOW_START_MS, "control_bytes_sent": 1, "control_bytes_received": 1},
        {"event": "counters", "ts_ms": t0 + WINDOW_END_MS, "control_bytes_sent": 2, "control_bytes_received": 2},
    ]
    _write_run(
        store / "test" / "p0" / "1",
        {
            "h0": [{"event": "record_created", "origin": "h0", "seq": 1, "ts_ms": t0 + 15_000}, *counters],
            "h1": [{"event": "record_stored", "origin": "h0", "seq": 1, "holder": "h0", "ts_ms": t0 + 16_000}, *counters],
        },
    )


def test_no_included_run_gives_every_figure_the_same_no_data_treatment(tmp_path: Path, monkeypatch):
    """Only an unknown run, not pooled: summary.csv still has a status-count row, yet every figure says "no data"."""
    store = tmp_path / "store"
    _unknown_only_store(store)
    captured = {}
    monkeypatch.setattr(evalfig, "write_figures", lambda outdir, *rows: captured.setdefault("rows", rows))
    aggregate(str(store), "test", str(tmp_path / "out"), include_undone=False, include_unknown=False)

    survival, consistency, summary, runs = captured["rows"]
    assert [row["p"] for row in summary] == [0.0]
    assert _texts(evalfig.discovery_survival_figure(survival)) == ["no data"]
    assert _texts(evalfig.consistency_figure(consistency, summary)) == ["no data"]
    assert _texts(evalfig.probe_established_figure(summary)) == ["no data"]
    assert _texts(evalfig.control_bytes_figure(runs, summary)) == ["no data"]


def test_aggregate_builds_runs_and_consistency_rows_once_and_hands_figures_the_csv_rows(
    synthetic_store: Path, tmp_path: Path, monkeypatch  # noqa: F811
):
    """The figure rows are selected from the very lists the CSVs were written from, never rebuilt."""
    calls = {"build_runs_rows": [], "build_consistency_rows": []}
    for name in calls:
        original = getattr(evalagg, name)

        def counting(analyses, _name=name, _original=original):
            """Record the list this builder returned, then return it unchanged."""
            rows = _original(analyses)
            calls[_name].append(rows)
            return rows

        monkeypatch.setattr(evalagg, name, counting)
    captured = {}
    monkeypatch.setattr(evalfig, "write_figures", lambda outdir, *rows: captured.setdefault("rows", rows))
    aggregate(str(synthetic_store), "test", str(tmp_path / "out"), include_undone=False, include_unknown=True)

    assert {name: len(lists) for name, lists in calls.items()} == {"build_runs_rows": 1, "build_consistency_rows": 1}
    _survival, consistency, _summary, runs = captured["rows"]
    # Every figure row is one of the CSV's own row objects (identity, not just equal values).
    csv_runs_ids = {id(row) for row in calls["build_runs_rows"][0]}
    csv_consistency_ids = {id(row) for row in calls["build_consistency_rows"][0]}
    assert runs and all(id(row) in csv_runs_ids for row in runs)
    assert consistency and all(id(row) in csv_consistency_ids for row in consistency)
    # ...and only the pooled runs': the synthetic tree's flagged run (p=0, seed 2) is left out.
    assert (0.0, 2) not in {(row["p"], row["seed"]) for row in runs}


# Run in a fresh interpreter: matplotlib's backend, once resolved, stays
# resolved for the whole process, so only a new process can show that
# drawing never triggered it. rcParams._get_backend_or_none() is private,
# but it is the one way to ask "still unresolved?" without resolving it.
_GLOBAL_STATE_SCRIPT = """
import json, sys
import matplotlib
from semindex import evalfig
rows = json.loads(sys.argv[2])
before = matplotlib.rcParams._get_backend_or_none()
evalfig.write_figures(sys.argv[1], *rows)
print(json.dumps({
    "before": before,
    "after": matplotlib.rcParams._get_backend_or_none(),
    "pyplot": "matplotlib.pyplot" in sys.modules,
}))
"""


def test_drawing_all_figures_leaves_no_process_global_matplotlib_state(tmp_path: Path):
    """All four figures with real data: pyplot never imported, the backend never auto-resolved."""
    survival = _curve(0.0, 1, [0.0, 0.5, 1.0])
    consistency = [
        {"p": 0.0, "seed": 1, "origin": "h0", "seq": seq, "time_ms": ms, "censored": False}
        for seq, ms in enumerate([1000.0, 2000.0, 3000.0, 4000.0], start=1)
    ]
    summary = [
        _summary_row(
            0.0,
            consistency_time_median_ms=2000.0,
            control_bytes_sent_per_record_median=270.0,
            control_bytes_sent_per_record_q1=265.0,
            control_bytes_sent_per_record_q3=275.0,
            n_runs_probe_truth=1,
            probe_established_total=10,
            probe_established_discovered=9,
            probe_established_rate=0.9,
            probe_established_rate_wilson_lo=0.6,
            probe_established_rate_wilson_hi=0.98,
        )
    ]
    runs = [{"p": 0.0, "seed": 1, "control_bytes_sent_per_record": 270.0}]
    env = {k: v for k, v in __import__("os").environ.items() if k != "MPLBACKEND"}
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    result = subprocess.run(
        [sys.executable, "-c", _GLOBAL_STATE_SCRIPT, str(tmp_path), json.dumps([survival, consistency, summary, runs])],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert json.loads(result.stdout) == {"before": None, "after": None, "pyplot": False}
