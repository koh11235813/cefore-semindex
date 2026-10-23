"""semindex.evalfig: draw the stage-1 lab-report figures from semindex.evalagg's own table rows.

One job: row lists in, figure files out. Every number drawn here is read
straight off the same row dicts evalagg writes its CSVs from (survival,
consistency, summary, runs) -- nothing here re-derives a metric, so a
figure and the CSV beside it cannot disagree. Four figures, each written
as `<stem>.png` (print DPI) plus `<stem>.pdf` into the same --out dir:

- fig_discovery_survival: KM delivered fraction vs time since record
  creation, one line per p, one panel per hop_distance (survival.csv).
- fig_consistency: per-p distribution of the all-nodes-consistent time
  of delivered records, with each p's censored count and summary.csv's
  censoring-aware KM median (consistency.csv + summary.csv).
- fig_probe_established: established-probe discovery rate per p with its
  Wilson 95% interval (summary.csv).
- fig_control_bytes: control bytes sent per record, one point per run
  plus summary.csv's median and Q1-Q3 (runs.csv + summary.csv).

matplotlib is imported lazily inside each function and only through its
object-oriented API (Figure + the Agg canvas, never pyplot or
matplotlib.use()), so importing this module costs nothing and drawing
leaves no process-global state behind: no backend switch, no rcParams
mutation, no pyplot figure registry growing across repeated aggregate()
calls in one test process. Styling is applied per artist instead.
"""

from __future__ import annotations

import math
import os
import textwrap

from semindex.evalagg import PROBE_FRESH_S

# Print-friendly raster resolution; the PDF beside each PNG is the vector
# copy for a report that wants one.
PNG_DPI = 200

# Ordinal blue ramp (dataviz reference palette, steps 250..700). p is an
# ordered magnitude (disconnect probability), so one hue light -> dark
# carries "more disconnects = darker"; step 250 is the lightest that still
# clears 2:1 against the surface (validated with --ordinal).
_P_RAMP = [
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
# A one-hue ramp collapses in grayscale print and for some readers, so
# every p also gets its own dash pattern -- identity never rests on
# color alone.
# Ordered so the patterns hardest to tell apart (the two dash-dot
# families) are never given to neighbouring p values, which already have
# the closest colors.
_P_DASHES = [
    (None, None),
    (0, (1, 1.5)),
    (0, (6, 2, 1, 2)),
    (0, (3, 2)),
    # Short enough that a legend handle shows at least two dashes; a
    # longer dash read as a solid line there, i.e. as p=0.
    (0, (7, 3)),
]
# Top of the fixed p scale colors are laid out on (see _p_styles); covers
# every p the stage-1 sweep runs, with room for one more step.
_P_COLOR_MAX = 0.5
_SERIES_COLOR = "#2a78d6"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID_COLOR = "#e4e3df"
_FONT_SIZE = 9

FIGURE_STEMS = ("fig_discovery_survival", "fig_consistency", "fig_probe_established", "fig_control_bytes")


# -- shared figure plumbing -------------------------------------------------


def _new_figure(width_in: float, height_in: float):
    """A Figure bound to an Agg canvas, laid out with constrained_layout.

    constrained_layout rather than tight_layout(): the latter emits a
    UserWarning on layouts it cannot satisfy, and evalagg's tests compare
    aggregate()'s full stderr.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width_in, height_in), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    return fig


def _style_axes(ax) -> None:
    """Recessive chrome: hairline solid grid behind the data, no top/right spines, secondary-ink ticks."""
    ax.set_axisbelow(True)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.8, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_TEXT_SECONDARY)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_TEXT_SECONDARY, labelsize=_FONT_SIZE, labelcolor=_TEXT_PRIMARY)
    ax.xaxis.label.set_fontsize(_FONT_SIZE)
    ax.yaxis.label.set_fontsize(_FONT_SIZE)
    ax.xaxis.label.set_color(_TEXT_PRIMARY)
    ax.yaxis.label.set_color(_TEXT_PRIMARY)


def _no_data(fig, title: str):
    """Turn an empty figure into one axes saying "no data", so every figure file is still written."""
    ax = fig.add_subplot()
    ax.set_axis_off()
    ax.set_title(title, fontsize=_FONT_SIZE + 1, color=_TEXT_PRIMARY)
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color=_TEXT_SECONDARY)
    return fig


def _footnote(fig, text: str) -> None:
    """A caption-like note under the plot area, in secondary ink, wrapped to the figure's width.

    Wrapped by hand because constrained_layout reserves height for the
    note but never breaks its lines: an unwrapped note runs off the
    right edge of the PNG. ~15 characters per inch fits this font size.
    """
    width_chars = int(fig.get_figwidth() * 15)
    fig.supxlabel(textwrap.fill(text, width_chars), fontsize=_FONT_SIZE - 1, color=_TEXT_SECONDARY)


def _p_label(p: float) -> str:
    """How a p value is written in legends: the same `{:g}` form as every p tick label ("p=0", "p=0.1")."""
    return f"p={p:g}"


def _p_styles(ps: list[float]) -> dict[float, tuple[str, object]]:
    """(color, dash) per p, each a function of the p VALUE alone, never of which other p values are present.

    Color: p's position on a fixed [0, _P_COLOR_MAX] scale picks the
    ramp step, so the same p keeps the same color in every stage's
    figures (a stage with only p=0.2 must not paint it the lightest
    blue). The scale widens only if a larger p ever appears. Dash: p's
    0.1-grid index modulo the pattern count, likewise value-keyed; the
    sweep's p values are multiples of 0.1, so neighbours on that grid
    always differ.
    """
    scale_max = max([_P_COLOR_MAX, *ps])
    styles = {}
    for p in set(ps):
        idx = round(p / scale_max * (len(_P_RAMP) - 1))
        styles[p] = (_P_RAMP[idx], _P_DASHES[round(p * 10) % len(_P_DASHES)])
    return styles


def _p_spacing(ps: list[float]) -> float:
    """The smallest gap between distinct p values (1.0 for a single p), used to size per-p marks."""
    ordered = sorted(set(ps))
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    return min(gaps) if gaps else 1.0


def _set_p_axis(ax, ps: list[float]) -> None:
    """x axis with one tick per p, labelled with the bare p value, padded by half a p-gap on each side."""
    ordered = sorted(set(ps))
    gap = _p_spacing(ordered)
    ax.set_xticks(ordered, [f"{p:g}" for p in ordered])
    ax.set_xlim(ordered[0] - gap / 2, ordered[-1] + gap / 2)
    ax.set_xlabel("disconnect probability p")
    ax.grid(False, axis="x")


def _save(fig, outdir: str, stem: str) -> list[str]:
    """Write `fig` as <stem>.png and <stem>.pdf into `outdir`; return both paths, PNG first."""
    paths = []
    for ext in ("png", "pdf"):
        path = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(path, dpi=PNG_DPI, facecolor="white")
        paths.append(path)
    return paths


# -- the four figures ---------------------------------------------------------


def discovery_survival_figure(survival_rows: list[dict]):
    """KM delivered-fraction curves from survival.csv rows: one panel per hop_distance, one line per p.

    Panels per hop rather than one axes: hop distance is the dominant
    structure (latency grows roughly linearly with hops), and every p x
    hop line in one axes would be 20 converging curves. Panels share
    both axes so they compare directly. Each line's legend entry carries
    its own group's censored/pairs count, read off the rows -- censoring
    differs per (p, hop), so a per-panel legend is the honest place.
    """
    title = "Discovery: delivered fraction vs time since record creation (Kaplan-Meier)"
    fig = _new_figure(9.0, 6.5)
    if not survival_rows:
        return _no_data(fig, title)

    groups: dict[tuple[float, int], list[dict]] = {}
    for row in survival_rows:
        groups.setdefault((row["p"], row["hop_distance"]), []).append(row)
    hops = sorted({hop for _, hop in groups})
    styles = _p_styles([p for p, _ in groups])

    # The grid spans the whole window, but curves approach their final
    # value long before its end, often through a long thin tail; cut the x
    # axis at the earliest grid time where every curve is within 1 % of its
    # own final value (rounded up to a 5 s tick), so the rise -- the part
    # the figure is about -- gets the width, and say so in the footnote.
    window_end_s = max(row["t_s"] for row in survival_rows)
    reach_s = 0.0
    for rows in groups.values():
        rows.sort(key=lambda r: r["t_s"])
        final = rows[-1]["delivered_fraction"]
        reach_s = max(reach_s, next(r["t_s"] for r in rows if r["delivered_fraction"] >= 0.99 * final))
    x_max = min(window_end_s, max(5.0, 5.0 * math.ceil(reach_s / 5.0)))

    ncols = 1 if len(hops) == 1 else 2
    nrows = math.ceil(len(hops) / ncols)
    axes = fig.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False).flatten()
    for ax in axes[len(hops) :]:
        fig.delaxes(ax)
    for ax, hop in zip(axes, hops):
        _style_axes(ax)
        for p in sorted(p for p, h in groups if h == hop):
            rows = groups[(p, hop)]
            color, dash = styles[p]
            label = f"{_p_label(p)} ({rows[0]['n_censored']}/{rows[0]['n_pairs']} censored)"
            line = ax.step(
                [r["t_s"] for r in rows],
                [r["delivered_fraction"] for r in rows],
                where="post",
                color=color,
                linewidth=2,
                label=label,
            )[0]
            if dash[0] is not None:
                line.set_linestyle(dash)
        ax.set_title(f"hop distance {hop}", fontsize=_FONT_SIZE + 1, color=_TEXT_PRIMARY)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=_FONT_SIZE - 1, frameon=False, handlelength=4.5)
        ax.set_xlabel("time since record creation [s]")
        ax.set_ylabel("delivered fraction")
        # Shared axes: keep tick labels and axis titles on the outer panels only.
        ax.label_outer()
    fig.suptitle(title, fontsize=_FONT_SIZE + 2, color=_TEXT_PRIMARY)
    note = (
        "Pooled discovery pairs of included runs. Censored = not delivered before the window closed; "
        "KM keeps each one at risk for its own follow-up time."
    )
    if x_max < window_end_s:
        note += (
            f" x axis cut at {x_max:g} s, where every curve is within 1 % of its final value"
            f" (the grid runs to the window end, {window_end_s:g} s)."
        )
    _footnote(fig, note)
    return fig


def consistency_figure(consistency_rows: list[dict], summary_rows: list[dict]):
    """Per-p box of all-nodes-consistent time over DELIVERED records, censored count per p, KM median overlaid.

    Why not a KM curve: consistency.csv rows carry no follow-up time for
    a censored record, so a KM curve could not be built from them -- it
    would need a second computation from the runs, which this module
    must not do. Instead nothing is dropped silently: the box describes
    only the delivered records, each p is labelled with how many of its
    records were censored (all of them -> no box, an explicit note), and
    the authoritative, censoring-aware centre is summary.csv's own KM
    median, drawn as a separate marker. The box's own median line is
    hidden so there is never a second, unlabelled "median" that diverges
    from it under censoring.
    """
    title = "All-nodes-consistent time per record"
    fig = _new_figure(7.0, 4.5)
    if not consistency_rows:
        return _no_data(fig, title)

    by_p: dict[float, list[dict]] = {}
    for row in consistency_rows:
        by_p.setdefault(row["p"], []).append(row)
    ps = sorted(by_p)
    gap = _p_spacing(ps)
    km_median_s = {
        row["p"]: row["consistency_time_median_ms"] / 1000.0
        for row in summary_rows
        if row["consistency_time_median_ms"] is not None
    }

    ax = fig.add_subplot()
    _style_axes(ax)
    delivered_s = {p: [r["time_ms"] / 1000.0 for r in rows if not r["censored"]] for p, rows in by_p.items()}
    with_box = [p for p in ps if delivered_s[p]]
    if with_box:
        ax.boxplot(
            [delivered_s[p] for p in with_box],
            positions=with_box,
            widths=gap * 0.4,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
            boxprops={"facecolor": "#cde2fb", "edgecolor": _SERIES_COLOR, "linewidth": 1.2},
            whiskerprops={"color": _SERIES_COLOR, "linewidth": 1.2},
            capprops={"color": _SERIES_COLOR, "linewidth": 1.2},
            medianprops={"linewidth": 0},
        )
    medians = [(p, km_median_s[p]) for p in ps if p in km_median_s]
    if medians:
        ax.plot(
            [p for p, _ in medians],
            [m for _, m in medians],
            linestyle="none",
            marker="D",
            markersize=6,
            color=_TEXT_PRIMARY,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="KM median (summary.csv, censoring-aware)",
        )
        ax.legend(loc="upper left", fontsize=_FONT_SIZE - 1, frameon=False)
    ax.set_ylim(bottom=0)
    _set_p_axis(ax, ps)
    ax.set_ylabel("time until every node held the record [s]")
    # Censored counts sit just above the plot area's floor, one per p, so
    # they never collide with a box or whisker of a different height.
    for p in ps:
        n = len(by_p[p])
        n_cens = sum(1 for r in by_p[p] if r["censored"])
        text = f"all {n} censored" if n_cens == n else f"{n_cens}/{n} censored"
        ax.text(
            p,
            0.02,
            text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=_FONT_SIZE - 1,
            color=_TEXT_SECONDARY,
        )
    fig.suptitle(title, fontsize=_FONT_SIZE + 2, color=_TEXT_PRIMARY)
    _footnote(
        fig,
        "Box: Q1-Q3 of delivered records, whiskers 5-95 %. Censored records (not consistent before the "
        "window closed) are counted, not boxed.",
    )
    return fig


def probe_established_figure(summary_rows: list[dict]):
    """Established-probe discovery rate per p with its Wilson 95% interval, from summary.csv rows.

    The y axis is not forced to start at 0: these are rates near 1, and
    a 0..1 axis would squash the intervals the figure exists to show
    (points and error bars, not bars, so a cut axis does not distort
    any length the eye reads as magnitude).
    """
    title = "Established-probe discovery rate"
    fig = _new_figure(6.0, 4.5)
    if not summary_rows:
        return _no_data(fig, title)

    ps = [row["p"] for row in summary_rows]
    ax = fig.add_subplot()
    _style_axes(ax)
    with_rate = [row for row in summary_rows if row["probe_established_rate"] is not None]
    lows = [row["probe_established_rate_wilson_lo"] for row in with_rate]
    y_min = math.floor(min(lows) * 20) / 20 if lows else 0.0
    y_min = min(y_min, 0.95)
    if with_rate:
        rates = [row["probe_established_rate"] for row in with_rate]
        ax.errorbar(
            [row["p"] for row in with_rate],
            rates,
            # 2026-10-23 smoke crash fix: at rate 1.0 the Wilson upper
            # bound comes out as 0.9999999999999999, a hair below the rate,
            # and errorbar rejects the resulting negative length; clamp
            # float noise to 0 rather than altering the CSV's own bound.
            yerr=[
                [max(0.0, rate - row["probe_established_rate_wilson_lo"]) for rate, row in zip(rates, with_rate)],
                [max(0.0, row["probe_established_rate_wilson_hi"] - rate) for rate, row in zip(rates, with_rate)],
            ],
            linestyle="none",
            marker="o",
            markersize=7,
            color=_SERIES_COLOR,
            markeredgecolor="white",
            markeredgewidth=1.5,
            ecolor=_SERIES_COLOR,
            elinewidth=1.5,
            capsize=4,
        )
    for row in summary_rows:
        if row["probe_established_rate"] is None:
            n_runs = row["n_runs_probe_truth"] + row["n_runs_probe_truth_missing"]
            text = f"no ground truth\n(0 of {n_runs} runs)"
            ax.text(row["p"], 0.5, text, transform=ax.get_xaxis_transform(), ha="center", va="center",
                    fontsize=_FONT_SIZE - 1, color=_TEXT_SECONDARY)
        else:
            # k/n and the run count sit under the interval's lower end, the
            # only side guaranteed to have room when the rate is 1.0.
            text = f"{row['probe_established_discovered']}/{row['probe_established_total']}\n{row['n_runs_probe_truth']} runs"
            ax.annotate(text, (row["p"], row["probe_established_rate_wilson_lo"]), xytext=(0, -6),
                        textcoords="offset points", ha="center", va="top", fontsize=_FONT_SIZE - 1,
                        color=_TEXT_SECONDARY)
    span = 1.0 - y_min
    ax.set_ylim(y_min - span * 0.18, 1.0 + span * 0.04)
    _set_p_axis(ax, ps)
    ax.set_ylabel("discovered / established probes (Wilson 95 %)")
    fig.suptitle(title, fontsize=_FONT_SIZE + 2, color=_TEXT_PRIMARY)
    _footnote(
        fig,
        f"Only probes whose oldest matching record was >= {PROBE_FRESH_S:g} s old, and only runs with "
        "probe ground truth (stored agent configs).",
    )
    return fig


def control_bytes_figure(runs_rows: list[dict], summary_rows: list[dict]):
    """Control bytes sent per record: one point per run (runs.csv) beside summary.csv's median and Q1-Q3.

    Run points are spread across a fixed band left of each p's tick by
    their order, not random jitter, so the figure is byte-identical run
    to run. The y axis is not zero-based: these are point marks, and the
    per-p differences are a few percent of the value.
    """
    title = "Control bytes sent per record"
    fig = _new_figure(6.0, 4.5)
    # .get: a runs row for a run with analysis_error carries only its
    # identity/status columns, no metric keys at all (see build_runs_rows).
    points = [row for row in runs_rows if row.get("control_bytes_sent_per_record") is not None]
    if not points and not summary_rows:
        return _no_data(fig, title)

    ps = sorted({row["p"] for row in points} | {row["p"] for row in summary_rows})
    gap = _p_spacing(ps)
    ax = fig.add_subplot()
    _style_axes(ax)
    by_p: dict[float, list[float]] = {}
    for row in points:
        by_p.setdefault(row["p"], []).append(row["control_bytes_sent_per_record"])
    xs, ys = [], []
    for p, values in by_p.items():
        for i, value in enumerate(values):
            offset = 0.0 if len(values) == 1 else -0.30 + 0.25 * i / (len(values) - 1)
            xs.append(p + offset * gap)
            ys.append(value)
    ax.scatter(xs, ys, s=22, color=_SERIES_COLOR, alpha=0.7, edgecolors="white", linewidths=0.8,
               label="one run")
    summary = [row for row in summary_rows if row["control_bytes_sent_per_record_median"] is not None]
    if summary:
        med = [row["control_bytes_sent_per_record_median"] for row in summary]
        ax.errorbar(
            [row["p"] + 0.15 * gap for row in summary],
            med,
            yerr=[
                [m - row["control_bytes_sent_per_record_q1"] for m, row in zip(med, summary)],
                [row["control_bytes_sent_per_record_q3"] - m for m, row in zip(med, summary)],
            ],
            linestyle="none",
            marker="_",
            markersize=14,
            markeredgewidth=2,
            color=_TEXT_PRIMARY,
            elinewidth=2,
            capsize=0,
            label="median, Q1-Q3 (summary.csv)",
        )
    ax.legend(loc="best", fontsize=_FONT_SIZE - 1, frameon=False)
    _set_p_axis(ax, ps)
    ax.set_ylabel("control bytes sent per record [bytes]")
    fig.suptitle(title, fontsize=_FONT_SIZE + 2, color=_TEXT_PRIMARY)
    _footnote(fig, "Sync-protocol bytes sent by all nodes inside the window / records created in that run.")
    return fig


def write_figures(
    outdir: str,
    survival_rows: list[dict],
    consistency_rows: list[dict],
    summary_rows: list[dict],
    runs_rows: list[dict],
) -> list[str]:
    """Draw all four figures into `outdir` (PNG + PDF each); return every path written, in FIGURE_STEMS order.

    Every figure is always written, even from empty rows ("no data"
    figure), so the set of files in --out never depends on the data.
    `consistency_rows`/`runs_rows` must already be restricted to the runs
    summary.csv pools, or a figure's per-run marks and its summary
    overlay would describe different run sets.
    """
    figures = [
        discovery_survival_figure(survival_rows),
        consistency_figure(consistency_rows, summary_rows),
        probe_established_figure(summary_rows),
        control_bytes_figure(runs_rows, summary_rows),
    ]
    paths = []
    for stem, fig in zip(FIGURE_STEMS, figures):
        paths.extend(_save(fig, outdir, stem))
    return paths
