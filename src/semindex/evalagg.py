"""semindex.evalagg: aggregate stage-1 evaluation runs into map.md's metrics.

Reads every (p, seed) run stored under `<store>/<stage>/p<p>/<seed>/`
(semindex.evalgen's own layout, map.md "データの置き場所") and writes six
CSV files into --out: runs.csv, discovery.csv, survival.csv,
consistency.csv, summary.csv, decisions.csv. Nothing here duplicates
semindex.analyze's event discovery or pairing -- this module calls into
it for event-reading, (origin, seq) creation times, control-byte summing
and probe pairing, and only adds the window cut, censoring, and the
across-run pooling analyze.py has no reason to know about.

Figures: a plot of each table would still be useful, but this venv has no
matplotlib (checked at import time, see FIGURES_AVAILABLE below) and
nothing may be installed to get one. Every output here is written as a
plain CSV instead, deliberately kept in long/tidy form (one row per
observation, never one column per value) so a plotting tool -- whichever
one a later effort picks -- can read a table directly, no reshaping:

- runs.csv: one row per (p, seed) run. Status (`valid`/`flagged`/
  `unknown`, see `read_meta` -- or `incomplete`, which never goes
  through `read_meta` at all, see `aggregate`) and `flags` are always
  present even when `analysis_error` is set (this run's events/t0
  evidence could not be read at all) -- every other column is empty in
  that case, never a misleading 0. Otherwise: counts (`n_records_created`, `n_pairs`,
  `n_delivered_in_window`, `n_censored`), control bytes strictly inside
  the window (per-node as a JSON cell, plus the total and the per-record
  figure), probe counts, and which t0 rung won (`t0_source`) plus its
  own cross-checks (`t0_agent_start_disagreement`, `t0_consistency_ms`).
- discovery.csv: one row per (run, origin, seq, node) pair -- every
  processed run's rows, regardless of status (see `build_discovery_rows`
  for why). `latency_ms` empty when `censored` is true.
- survival.csv: one row per (p, hop_distance, t_s) grid point -- a KM
  delivered-fraction curve per (p, hop_distance), plus that group's own
  `n_pairs`/`n_censored`/`censored_fraction` and `median_s`/`q1_s`/`q3_s`
  repeated on every row of the group (so a reader can filter to one
  group and immediately have everything needed to plot and annotate it).
- consistency.csv: one row per (run, origin, seq) -- time until every
  non-origin node held that record, or censored.
- summary.csv: one row per p -- run-status counts (including
  `n_incomplete`), pooled discovery-pair
  counts/censoring, KM-quantile latency/consistency medians, control
  bytes per record, and probe discovery rate with its Wilson interval.
- decisions.csv: one row per (p, kind, field) -- see `build_decisions_rows`
  for the full field list and why a boolean field (`truncated`) reports
  a `frequency` instead of a median.

t0 (map.md "計測の窓"): every agent's own `elapsed_s` is `now -
t0_ms/1000`, and `t0_ms` only ever reaches the agent as a CLI override
(`--t0-ms`) baked into the host_apps argv at ReCefore launch time -- the
generated YAML stores that argv as the literal, unresolved template
string "{t0_ms}" (semindex.gen's own placeholder), and a node's own
agent-config JSON carries a "t0_ms": 0 placeholder default that is never
the value actually used. 2026-10-23: agent.py now writes the real value
directly, as an `agent_start` line before anything else in every life
(see Agent.__init__), so this is the preferred, most direct evidence a
run can have. Older runs without it (and any older rung this module
still carries as a fallback -- see `derive_t0_ms`) fall back to deriving
t0 from the one clock every node's events.jsonl and the run's own
results.json already share (Mininet hosts share the wall clock -- see
semindex.check.results_ts_ms's own docstring). `t0_source` on every run
says which rung actually won.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import statistics
import sys
from dataclasses import dataclass, field

from semindex import evalgen
from semindex.analyze import (
    _created_ts_by_key,
    _node_sort_key,
    discover_node_events,
    probe_pairs,
)
from semindex.check import CheckUsageError, agent_start_t0, agent_start_t0_values, read_results, results_ts_ms

# Whether a figure step could even run in this interpreter. Checked once,
# at import time, rather than inside main() on every invocation; a
# venv's installed package set does not change between one run of this
# CLI and the next. No plotting code is written at all when this is
# False (see the module docstring) -- there is nothing here to guard with
# it today, but a later effort adding figures should gate that code on
# this flag instead of a bare `import matplotlib`.
FIGURES_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

# 95% two-sided Wilson score interval z (scipy.stats.norm.ppf(0.975));
# hardcoded rather than computed, since this module has no normal-quantile
# routine of its own and stdlib has none either (map.md only ever asks
# for the 95% interval, not a configurable confidence level).
_WILSON_Z = 1.959963984540054

# KM time grid for survival.csv (map.md "t グリッド 0..120 秒 0.5 秒刻み").
# The grid's own span is WINDOW_END_S - WINDOW_START_S, not a hardcoded
# 120 -- the one evalgen already fixed at 120.0 (DISCONNECT_WINDOW_S) --
# so this module never carries its own copy of that number to drift from
# evalgen's.
_GRID_STEP_S = 0.5


def _window_duration_s() -> float:
    """The measurement window's own span in seconds, from evalgen's constants."""
    return evalgen.WINDOW_END_S - evalgen.WINDOW_START_S


def _survival_grid() -> list[float]:
    """The t grid survival.csv reports delivered-fraction at: 0, 0.5, ..., up to the window's span."""
    duration = _window_duration_s()
    n_steps = round(duration / _GRID_STEP_S)
    return [round(i * _GRID_STEP_S, 4) for i in range(n_steps + 1)]


# -- run discovery -------------------------------------------------------


@dataclass(frozen=True)
class RunRef:
    """One (p, seed) run's location under the store, before anything in it has been read."""

    p: float
    seed: int
    run_dir: str


def _parse_p_dirname(name: str) -> float | None:
    """`"p0.2"` -> 0.2, `"p0"` -> 0.0; None if `name` is not a `p<number>` directory."""
    if not name.startswith("p"):
        return None
    try:
        return float(name[1:])
    except ValueError:
        return None


def discover_runs(store: str, stage: str) -> list[RunRef]:
    """Every (p, seed) run directory under `<store>/<stage>/`, sorted by (p, seed).

    A run directory is any `<store>/<stage>/p<p>/<seed>/` whose `<seed>`
    component is itself a run of ASCII digits (evalgen names every run's
    seed directory after the plain int `--seed` it was generated with;
    anything else under a `p<p>/` directory is not a run this module
    knows how to read).
    """
    runs: list[RunRef] = []
    for p_dir in sorted(glob.glob(os.path.join(store, stage, "p*"))):
        p = _parse_p_dirname(os.path.basename(p_dir))
        if p is None or not os.path.isdir(p_dir):
            continue
        for seed_dir in sorted(os.listdir(p_dir)):
            if not seed_dir.isdigit():
                continue
            full = os.path.join(p_dir, seed_dir)
            if os.path.isdir(full):
                runs.append(RunRef(p=p, seed=int(seed_dir), run_dir=full))
    runs.sort(key=lambda r: (r.p, r.seed))
    return runs


def run_has_done_marker(run_dir: str) -> bool:
    """Whether `<run_dir>/DONE` exists.

    2026-10-23: `semindex.evalrun.run_one` writes DONE unconditionally,
    after a run reaches the end of its own pipeline -- flagged or not
    (only the verdict inside meta.json records pass/fail; DONE only ever
    means "this run finished and its retained files are complete"). A
    run dir missing it is therefore never a run whose preconditions
    failed -- it is one that never finished at all (orchestrator
    restart, machine reboot, a crash partway through `run_one`), which
    `run_one` itself deletes and redoes from scratch the next time it
    sees that dir. `aggregate` counts such a run as "incomplete", never
    "flagged" or "unknown" (see its own docstring). `--include-undone`
    overrides that for the one case with no DONE by construction that is
    NOT awaiting a redo: a hand-stored trial-manual sample, which has
    neither DONE nor meta.json.
    """
    return os.path.isfile(os.path.join(run_dir, "DONE"))


# -- meta.json / validity --------------------------------------------------


def read_meta(run_dir: str) -> tuple[str, list[str]]:
    """This run's validity status and flags from `<run_dir>/meta.json`.

    Status is one of "valid", "flagged" or "unknown". `meta.json` is
    meant to be a run validity-checker's own output; no committed code
    writes one yet, and its shape is only committed to holding
    `{"valid": bool, "flags": [...]}` "under some key" -- so a dict with
    both keys is accepted whether it sits at the top level or one level
    down (e.g. `{"check": {"valid": ..., "flags": [...]}}`). Missing the
    file, an unparsable file, or a file whose shape matches neither all
    mean "unknown", never "flagged": there is no evidence here that the
    run's own preconditions actually failed, only that this module could
    not read a verdict.
    """
    path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(path):
        return "unknown", []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "unknown", []

    def _as_verdict(candidate: object) -> dict | None:
        """`candidate` itself, if it is a dict holding both "valid" and "flags" keys; else None."""
        if isinstance(candidate, dict) and "valid" in candidate and "flags" in candidate:
            return candidate
        return None

    verdict = _as_verdict(data)
    if verdict is None and isinstance(data, dict):
        for value in data.values():
            verdict = _as_verdict(value)
            if verdict is not None:
                break
    if verdict is None:
        return "unknown", []

    flags_raw = verdict.get("flags")
    flags = [str(f) for f in flags_raw] if isinstance(flags_raw, list) else []
    if verdict.get("valid") is True:
        return "valid", flags
    if verdict.get("valid") is False:
        return "flagged", flags
    return "unknown", flags


# -- t0 derivation ---------------------------------------------------------


@dataclass(frozen=True)
class T0Derivation:
    """Everything `derive_t0_ms` figured out about one run's t0, not just the winning value.

    `agent_start_disagreement` and `consistency_ms` exist even when
    `source` is not "agent_start": they are diagnostics about the
    agent_start/scheduler evidence this run happened to have, independent
    of which rung actually won.
    """

    t0_ms: float
    source: str  # "agent_start" | "scheduler" | "snapshot" | "stop" | "first_event"
    agent_start_disagreement: bool
    consistency_ms: float | None  # max |t0_agent_start - t0_scheduler| across nodes, when both exist


def _t0_from_results(results: list[dict] | None) -> float | None:
    """Median of (event ts_ms - actual_at*1000) over results.json's own scheduler event rows.

    The most precise source available (see the module docstring): every
    `op_type == "event"` row (link_down/link_up here) carries both a wall
    clock `ts` and `actual_at`, the scheduler's own monotonic seconds
    since ITS start -- and that start is the same instant every host_apps
    process was launched with as its `t0_ms` (verified against
    trial-manual: the spread across 17 link events there is under 25ms).
    None at p=0, where no link events exist at all, so this is never the
    only rung of the fallback chain.
    """
    if not results:
        return None
    anchors = []
    for record in results:
        if record.get("op_type") != "event":
            continue
        actual_at = record.get("actual_at")
        if actual_at is None:
            continue
        try:
            ts_ms = results_ts_ms(record)
        except (KeyError, ValueError):
            continue
        anchors.append(ts_ms - actual_at * 1000.0)
    return statistics.median(anchors) if anchors else None


def _t0_from_boundary_event(nodes: dict[str, list[dict]], event_name: str, boundary_s: float) -> float | None:
    """Median of (that event's ts_ms - boundary_s*1000) across every node that logged it.

    Used for both fallback rungs below `_t0_from_results`:
    `counters_snapshot` against evalgen.WINDOW_START_S, and the final
    `counters` line against evalgen.WINDOW_END_S (Agent._elapsed_s / the
    stop_at_s and counters_snapshot_at_s checks in agent.py fire these
    once `elapsed_s(now) >= boundary_s`, so the event's own ts_ms is
    `t0_ms + boundary_s*1000` plus at most one poll tick -- a late bias of
    well under a second, not large enough to matter against a 120s
    window).
    """
    anchors = [
        event["ts_ms"] - boundary_s * 1000.0
        for events in nodes.values()
        for event in events
        if event.get("event") == event_name and "ts_ms" in event
    ]
    return statistics.median(anchors) if anchors else None


def _t0_from_first_event(nodes: dict[str, list[dict]]) -> float | None:
    """The earliest ts_ms logged by any node, as a last-resort t0 (least precise rung)."""
    all_ts = [event["ts_ms"] for events in nodes.values() for event in events if "ts_ms" in event]
    return min(all_ts) if all_ts else None


def derive_t0_ms(nodes: dict[str, list[dict]], results: list[dict] | None) -> T0Derivation:
    """This run's t0 derivation, every rung's evidence included (see T0Derivation).

    2026-10-23: `agent_start` (agent.py now writes one `{"event":
    "agent_start", "node", "t0_ms"}` line before anything else in every
    life) is the new first rung -- it is literally the value the agent
    process ran with, not inferred from a boundary it crosses a tick
    late. It wins whenever a run has ANY agent_start lines at all, even
    if they disagree with each other (that disagreement is reported via
    `agent_start_disagreement`, not silently resolved by falling back to
    a different rung -- a run where nodes genuinely got different t0_ms
    values is a real anomaly worth surfacing, not a reason to pretend the
    direct evidence was never written). The four older rungs
    (scheduler/snapshot/stop/first_event) are now only ever reached when
    a run has NO agent_start evidence at all (e.g. a run captured before
    this change). `consistency_ms` cross-checks the agent_start evidence
    against the independently-derived scheduler evidence whenever both
    are available, regardless of which rung wins.

    Raises CheckUsageError only if a run has no evidence at all on any
    rung -- a run this broken has nothing this module could anchor a
    window to.
    """
    agent_start_values = agent_start_t0_values(nodes)
    scheduler = _t0_from_results(results)
    consistency_ms = (
        max(abs(v - scheduler) for v in agent_start_values) if agent_start_values and scheduler is not None else None
    )

    agent_start_result = agent_start_t0(agent_start_values)
    if agent_start_result is not None:
        t0_ms, disagreement = agent_start_result
        return T0Derivation(
            t0_ms=t0_ms,
            source="agent_start",
            agent_start_disagreement=disagreement,
            consistency_ms=consistency_ms,
        )
    if scheduler is not None:
        return T0Derivation(t0_ms=scheduler, source="scheduler", agent_start_disagreement=False, consistency_ms=None)
    snapshot = _t0_from_boundary_event(nodes, "counters_snapshot", evalgen.WINDOW_START_S)
    if snapshot is not None:
        return T0Derivation(t0_ms=snapshot, source="snapshot", agent_start_disagreement=False, consistency_ms=None)
    stop = _t0_from_boundary_event(nodes, "counters", evalgen.WINDOW_END_S)
    if stop is not None:
        return T0Derivation(t0_ms=stop, source="stop", agent_start_disagreement=False, consistency_ms=None)
    first = _t0_from_first_event(nodes)
    if first is not None:
        return T0Derivation(t0_ms=first, source="first_event", agent_start_disagreement=False, consistency_ms=None)
    raise CheckUsageError("no results.json events, counters lines, or any other ts_ms found to derive t0 from")


# -- small numeric helpers -------------------------------------------------


def _percentile(sorted_values: list[float], q: float) -> float | None:
    """The `q`-quantile (0..1) of already-sorted `sorted_values`, linear interpolation (numpy's default "linear" method).

    None for an empty input. A single value returns itself for every q
    (the interpolation index is always 0), which is also numpy's
    behaviour and needs no special case here.
    """
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = q * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def median_q1_q3(values: list[float]) -> tuple[float | None, float | None, float | None]:
    """(median, Q1, Q3) of `values` via `_percentile`; (None, None, None) if `values` is empty."""
    if not values:
        return None, None, None
    ordered = sorted(values)
    return _percentile(ordered, 0.5), _percentile(ordered, 0.25), _percentile(ordered, 0.75)


def wilson_interval(x: int, n: int, z: float = _WILSON_Z) -> tuple[float, float] | None:
    """The two-sided Wilson score interval for `x` successes out of `n` trials; None if n == 0."""
    if n <= 0:
        return None
    p_hat = x / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * ((p_hat * (1 - p_hat) / n) + z2 / (4 * n * n)) ** 0.5
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


# -- Kaplan-Meier product-limit estimator ----------------------------------


@dataclass(frozen=True)
class Sample:
    """One (origin, seq, node) discovery pair's follow-up, as a KM input: a duration and whether it was an event."""

    duration_s: float
    delivered: bool


def _row_to_sample(row: dict, delivered_key: str) -> Sample:
    """Convert one censoring-aware row into its KM Sample.

    Shared by every pooling site that reads a discovery-pair row
    (`latency_ms`) or a consistency row (`time_ms`) -- both carry the
    same three fields (the delivered-at value under a different key,
    `follow_up_ms`, `censored`), and used to be converted inline,
    separately, at each of survival.csv's and summary.csv's own pooling
    sites; a fix to one used to risk silently not reaching the other.
    """
    if row["censored"]:
        return Sample(row["follow_up_ms"] / 1000.0, delivered=False)
    return Sample(row[delivered_key] / 1000.0, delivered=True)


def kaplan_meier_steps(samples: list[Sample]) -> list[tuple[float, float]]:
    """The product-limit survival curve for `samples`: a list of (t, S(t)) steps, ascending t, starting at (0.0, 1.0).

    S(t) is the probability of NOT YET being delivered by time t
    ("surviving" past t without the delivery event); `1 - S(t)` is map.md's
    own "届いた割合" (delivered fraction). At each distinct delivery time
    t_i: n_i = count of samples whose duration is >= t_i (everyone still
    "at risk" -- not yet delivered and not yet censored out, inclusive of
    anyone censored at exactly t_i, which is the standard convention for
    ties between an event and a censoring at the same instant), d_i = how
    many of those samples were delivered at exactly t_i. S is updated
    multiplicatively by (1 - d_i/n_i) at each such t_i; a censoring-only
    time contributes no step on its own (it only shrinks n_i at later
    event times).

    A censored sample's own duration is its follow-up time (map.md
    "記録created late in the window have shorter follow-up"), not the
    window's full span -- this is exactly what keeps a late-created,
    not-yet-censored record from being double-penalized against a record
    that had the whole window to be delivered in.
    """
    if not samples:
        return [(0.0, 1.0)]
    event_times = sorted({s.duration_s for s in samples if s.delivered})
    steps = [(0.0, 1.0)]
    survival = 1.0
    for t in event_times:
        at_risk = sum(1 for s in samples if s.duration_s >= t)
        events_here = sum(1 for s in samples if s.delivered and s.duration_s == t)
        if at_risk > 0:
            survival *= 1 - events_here / at_risk
        steps.append((t, survival))
    return steps


def km_delivered_fraction_at(steps: list[tuple[float, float]], t_query: float) -> float:
    """1 - S(t_query), reading the step function `steps` (ascending t, as `kaplan_meier_steps` returns)."""
    survival = 1.0
    for t, s in steps:
        if t <= t_query:
            survival = s
        else:
            break
    return 1 - survival


def km_quantile(steps: list[tuple[float, float]], q: float) -> float | None:
    """The smallest t at which the KM delivered-fraction first reaches `q`; None if it never does.

    A small epsilon absorbs float rounding in `survival` (repeated
    multiplication) that could otherwise land a hair below an exact `q`
    boundary and miss it.
    """
    for t, s in steps:
        if 1 - s >= q - 1e-9:
            return t
    return None


# -- per-run analysis --------------------------------------------------------


@dataclass
class RunAnalysis:
    """Everything this module computes from one run's events.jsonl files, before any cross-run pooling."""

    p: float
    seed: int
    status: str  # "valid" | "flagged" | "unknown" | "incomplete"
    flags: list[str]
    # Set (and every metric field below left at its default/None) iff
    # this run's own files could not be analyzed at all -- status/flags
    # above are still real, read independently via read_meta before any
    # of this is attempted (see `analyze_run`), so an unreadable run is
    # still counted by validity, just with nothing else to report. Never
    # set for an "incomplete" run (see `aggregate`): there, every metric
    # field is left at its default for a different reason -- the run was
    # never analyzed at all, not analyzed-and-found-unreadable.
    analysis_error: str | None = None
    t0_ms: float | None = None
    t0_source: str | None = None
    t0_agent_start_disagreement: bool = False
    t0_consistency_ms: float | None = None
    window_start_ms: float | None = None
    window_end_ms: float | None = None
    n_records_created: int | None = None
    discovery_rows: list[dict] = field(default_factory=list)  # origin, seq, node, hop, latency_ms, censored
    consistency_rows: list[dict] = field(default_factory=list)  # origin, seq, time_ms, censored
    control_bytes_sent_window: dict[str, int | None] = field(default_factory=dict)
    control_bytes_received_window: dict[str, int | None] = field(default_factory=dict)
    probe_issued: int = 0
    probe_discovered: int = 0
    probe_retrieved: int = 0
    decision_rows: list[dict] = field(default_factory=list)  # kind, in_flight, missing (list), wait_s


def _split_lives(events: list[dict]) -> list[list[dict]]:
    """Split one node's events into lives: maximal runs ending at (and including) a `counters` line.

    agent.py writes exactly one `counters` line per life, at that life's
    own clean shutdown (EventWriter is append-mode across a restart, so
    a restarted node's events.jsonl holds several). A trailing run with
    no terminating `counters` line (the life never cleanly shut down) is
    dropped entirely -- this module's "no data" rule for byte counters,
    the same one semindex.analyze.control_bytes applies to a node that
    never wrote one at all.
    """
    lives: list[list[dict]] = []
    current: list[dict] = []
    for event in events:
        current.append(event)
        if event.get("event") == "counters":
            lives.append(current)
            current = []
    return lives


# 2026-10-23 fix: the agent detects crossing stop_at_s on its own poll
# tick, then writes the final `counters` line a few ms after that --
# wall-clock lateness in the WRITE, not evidence the life itself started
# or ran outside the window. The life structurally active when the
# window closes (the last life in `_split_lives`'s own order, whenever
# it carries no snapshot of its own -- see `_node_window_control_bytes`)
# is still entirely inside the window even when its own final line's
# ts_ms lands a hair past window_end_ms. This slack absorbs exactly that
# lateness, nothing more -- the same 1.0s margin semindex.evalvalid's own
# counters_timing check already allows around this identical boundary.
_STOP_DETECTION_LATE_SLACK_MS = 1000.0


def _node_window_control_bytes(
    events: list[dict], window_start_ms: float, window_end_ms: float
) -> tuple[int | None, int | None]:
    """One node's own (control_bytes_sent, control_bytes_received) strictly inside the window, life-aware.

    2026-10-23 fix: summing every life's own final `counters` line
    (semindex.analyze.control_bytes's own job) and subtracting one
    snapshot used to double-count whenever a restart happened anywhere
    in this node's history -- byte counters reset to 0 every life
    (agent.py's own docstring), so a PRIOR life's full total was never
    inside the window at all, yet it still landed in that sum (a
    reviewer's counterexample: a life ending at elapsed 5s with
    control_bytes_sent=40, entirely before a 10s window-open, inflated
    an in-window total of 90 to a wrong 130). This instead walks each
    life (`_split_lives`) and only ever counts the portion of it that
    actually falls inside the window:

    - the one life holding the window-start `counters_snapshot` line
      (evalgen's own counters_snapshot_at_s is always a single entry, at
      WINDOW_START_S, so at most one life ever has it) contributes its
      own final `counters` line minus that snapshot.
    - the LAST life in this node's history, if it carries no snapshot
      (a restart happened after the window opened): it is, by
      structural position, the life active when the window closes --
      every evalgen-generated config sets stop_at_s == WINDOW_END_S for
      every node, so there is always exactly one life still running at
      that instant, and it is always the last one in `_split_lives`'s
      order. 2026-10-23 fix: this used to be decided from that life's
      own final-line ts_ms (`<= window_end_ms`) instead of this
      structural position, which a few ms of stop-detection lateness
      (see `_STOP_DETECTION_LATE_SLACK_MS` above) could push past
      window_end_ms and drop entirely (counterexample: window [10000,
      130000] ms, a restart mid-window, final counters stamped 130001
      ms -- 1ms late -- dropped that whole life, undercounting a true
      total of 100 as 40). Deciding by position instead means the
      explicit slack constant only ever has to absorb logging lateness,
      never decide "is this even the right life" on its own.
    - any OTHER life with no snapshot, whose own final `counters` line
      lands at or after the window's open (it necessarily started fresh
      at 0, and a life before the snapshot-life cannot satisfy this --
      it ended before the window opened) and at or before the window's
      close, contributes its FULL final value.
    - any other life (one that ended before the window opened, with no
      snapshot in it) contributes nothing.

    None (not 0) for a node with no counters_snapshot evidence anywhere,
    the same "no data" convention as before this fix.
    """
    lives = _split_lives(events)
    if not any(any(e.get("event") == "counters_snapshot" for e in life) for life in lives):
        return None, None

    total_sent = 0
    total_received = 0
    any_life_counted = False
    last_idx = len(lives) - 1
    for idx, life in enumerate(lives):
        final = next((e for e in reversed(life) if e.get("event") == "counters"), None)
        if final is None:
            continue
        snapshot = next((e for e in life if e.get("event") == "counters_snapshot"), None)
        if snapshot is not None:
            total_sent += final["control_bytes_sent"] - snapshot["control_bytes_sent"]
            total_received += final["control_bytes_received"] - snapshot["control_bytes_received"]
            any_life_counted = True
        elif idx == last_idx and final["ts_ms"] <= window_end_ms + _STOP_DETECTION_LATE_SLACK_MS:
            total_sent += final["control_bytes_sent"]
            total_received += final["control_bytes_received"]
            any_life_counted = True
        elif window_start_ms <= final["ts_ms"] <= window_end_ms:
            total_sent += final["control_bytes_sent"]
            total_received += final["control_bytes_received"]
            any_life_counted = True
        # else: this life ended before the window opened (or some other
        # shape this module cannot disambiguate without a window-end
        # snapshot no stage-1 config emits) -- contributes nothing.
    if not any_life_counted:
        return None, None
    return total_sent, total_received


def _window_control_bytes(
    nodes: dict[str, list[dict]], window_start_ms: float, window_end_ms: float
) -> tuple[dict[str, int | None], dict[str, int | None]]:
    """Per-node control bytes sent/received strictly inside the window (see `_node_window_control_bytes`)."""
    sent: dict[str, int | None] = {}
    received: dict[str, int | None] = {}
    for node, events in nodes.items():
        sent[node], received[node] = _node_window_control_bytes(events, window_start_ms, window_end_ms)
    return sent, received


def analyze_run(p: float, seed: int, run_dir: str) -> RunAnalysis:
    """Build one run's full RunAnalysis: window cut, discovery/consistency pairs, control bytes, probes, decisions.

    2026-10-23 fix: validity (status/flags) is read via `read_meta`
    FIRST, independent of whether this run's events/t0 evidence can be
    analyzed at all. It used to be read only at the very end of a single
    function that could raise CheckUsageError (an unreadable
    events.jsonl, or no t0 evidence anywhere) partway through -- so a run
    that failed this early simply vanished from every output, not even
    counted in n_flagged/n_unknown, silently under-reporting a
    condition's own failure rate (map.md's own "印つきの run が 5% を超
    えたら やり直す" cannot be judged against a count that already
    dropped the very runs most likely to need re-running). The metric
    computation itself now lives in `_analyze_run_metrics`; a
    CheckUsageError from it is caught here and still produces a
    RunAnalysis, with this run's real status/flags and every metric
    field left at its None/empty default -- `analysis_error` names what
    went wrong -- rather than this run disappearing outright.
    """
    status, flags = read_meta(run_dir)
    try:
        return _analyze_run_metrics(p, seed, run_dir, status, flags)
    except CheckUsageError as exc:
        return RunAnalysis(p=p, seed=seed, status=status, flags=flags, analysis_error=str(exc))


def _analyze_run_metrics(p: float, seed: int, run_dir: str, status: str, flags: list[str]) -> RunAnalysis:
    """Every metric this module computes from one run's events, given its already-read validity (status/flags).

    Raises CheckUsageError (propagated from discover_node_events/derive_t0_ms)
    when this run's own files are too broken to analyze at all -- an
    unreadable or missing events.jsonl, or no t0 evidence on any rung.
    """
    nodes = discover_node_events(run_dir)
    results_path = os.path.join(run_dir, "results.json")
    results = read_results(run_dir) if os.path.isfile(results_path) else None
    t0 = derive_t0_ms(nodes, results)
    window_start_ms = t0.t0_ms + evalgen.WINDOW_START_S * 1000.0
    window_end_ms = t0.t0_ms + evalgen.WINDOW_END_S * 1000.0

    created = _created_ts_by_key(nodes)
    in_window_created = {
        key: ts for key, ts in created.items() if window_start_ms <= ts <= window_end_ms
    }

    # (node, event_idx) -> ts_ms is not enough to tell duplicates apart
    # from genuinely distinct stores; keep the EARLIEST record_stored per
    # (origin, seq, node), matching semindex.check's own "first store wins"
    # treatment of a holder that re-sends the same record.
    stored_ts: dict[tuple[str, int, str], float] = {}
    for node, events in nodes.items():
        for event in events:
            if event.get("event") != "record_stored":
                continue
            key = (event["origin"], event["seq"], node)
            ts = event["ts_ms"]
            if key not in stored_ts or ts < stored_ts[key]:
                stored_ts[key] = ts

    discovery_rows: list[dict] = []
    consistency_rows: list[dict] = []
    for (origin, seq), created_ts in sorted(in_window_created.items()):
        origin_idx = _node_sort_key(origin)
        last_delivered_ts = created_ts
        all_delivered = True
        for node in nodes:
            if node == origin:
                continue
            node_idx = _node_sort_key(node)
            ts = stored_ts.get((origin, seq, node))
            censored = ts is None or ts > window_end_ms
            latency_ms = None if censored else ts - created_ts
            discovery_rows.append(
                {
                    "origin": origin,
                    "seq": seq,
                    "node": node,
                    "hop_distance": abs(node_idx - origin_idx),
                    "latency_ms": latency_ms,
                    # Follow-up for a censored pair: from this record's own
                    # creation to the window's close, never the window's
                    # full span -- a record created late in the window has
                    # correspondingly less time to be delivered in before
                    # censoring, and the KM estimator (kaplan_meier_steps)
                    # needs that shorter follow-up to keep such a pair from
                    # inflating the at-risk set past when it actually left it.
                    "follow_up_ms": window_end_ms - created_ts,
                    "censored": censored,
                }
            )
            if censored:
                all_delivered = False
            else:
                last_delivered_ts = max(last_delivered_ts, ts)
        consistency_rows.append(
            {
                "origin": origin,
                "seq": seq,
                "time_ms": (last_delivered_ts - created_ts) if all_delivered else None,
                "follow_up_ms": window_end_ms - created_ts,
                "censored": not all_delivered,
            }
        )

    sent_window, received_window = _window_control_bytes(nodes, window_start_ms, window_end_ms)

    probe_rows = [
        p_pair
        for p_pair in probe_pairs(nodes)
        if window_start_ms <= p_pair["discovered_ts_ms"] <= window_end_ms
    ]
    probe_issued = len(probe_rows)
    probe_discovered = sum(1 for row in probe_rows if row["match_count"] > 0)
    probe_retrieved = sum(
        1 for row in probe_rows if row["retrieved"] is not None and row["retrieved"].get("success") is True
    )

    decision_rows: list[dict] = []
    for events in nodes.values():
        for event in events:
            if event.get("event") != "decision":
                continue
            if not (window_start_ms <= event["ts_ms"] <= window_end_ms):
                continue
            kind = event.get("kind")
            state = event.get("state", {})
            action = event.get("action", {})
            origins = state.get("origins", {})
            # map.md's own per-origin shape: {"hi", "missing", "advertised":
            # {neighbor: {"hi", "notified_max"}}}. `hi`/`missing` are
            # flattened one value per origin per decision; `advertised_gap`
            # one value per (origin, advertising-neighbor) pair --
            # notified_max - hi is how far that neighbor's own claimed
            # progress runs ahead of its contiguous hold (map.md's own
            # "origins" docstring: notified_max can exceed hi via a sparse
            # `extra`), the comparison this field exists for.
            # 2026-10-23 fix: `advertised_gap` alone loses the underlying
            # values -- two decision sets with (hi=3, notified_max=5) and
            # (hi=30, notified_max=32) both have gap=2 and, with every
            # other state field equal, produce an IDENTICAL decisions.csv
            # even though the neighbor's own advertised state is nowhere
            # close to the same. `advertised_hi`/`advertised_notified_max`
            # report those two raw distributions alongside the gap, so a
            # reader comparing two runs' state is not told they match when
            # they do not.
            last_timeout_s = state.get("last_timeout_s")
            decision_rows.append(
                {
                    "kind": kind,
                    "in_flight": state.get("in_flight"),
                    "truncated": state.get("truncated"),
                    "hi": [info.get("hi") for info in origins.values() if info.get("hi") is not None],
                    "missing": [info.get("missing") for info in origins.values() if info.get("missing") is not None],
                    "advertised_gap": [
                        adv["notified_max"] - adv["hi"]
                        for info in origins.values()
                        for adv in info.get("advertised", {}).values()
                        if adv.get("notified_max") is not None and adv.get("hi") is not None
                    ],
                    "advertised_hi": [
                        adv["hi"]
                        for info in origins.values()
                        for adv in info.get("advertised", {}).values()
                        if adv.get("hi") is not None
                    ],
                    "advertised_notified_max": [
                        adv["notified_max"]
                        for info in origins.values()
                        for adv in info.get("advertised", {}).values()
                        if adv.get("notified_max") is not None
                    ],
                    # Age, not the raw epoch seconds: last_timeout_s is "壁
                    # 時計のエポック秒" (map.md), not meaningfully poolable
                    # across runs/times on its own, but "how long ago,
                    # relative to this same decision" is.
                    "last_timeout_age_s": (
                        event["ts_ms"] / 1000.0 - last_timeout_s if last_timeout_s is not None else None
                    ),
                    "wait_s": action.get("wait_s") if kind == "fetch_backoff" else None,
                    "interval_s": action.get("interval_s") if kind == "state_poll" else None,
                    "fetched_count": len(action["fetched"]) if kind == "fetch_plan" and "fetched" in action else None,
                }
            )

    return RunAnalysis(
        p=p,
        seed=seed,
        status=status,
        flags=flags,
        t0_ms=t0.t0_ms,
        t0_source=t0.source,
        t0_agent_start_disagreement=t0.agent_start_disagreement,
        t0_consistency_ms=t0.consistency_ms,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        n_records_created=len(in_window_created),
        discovery_rows=discovery_rows,
        consistency_rows=consistency_rows,
        control_bytes_sent_window=sent_window,
        control_bytes_received_window=received_window,
        probe_issued=probe_issued,
        probe_discovered=probe_discovered,
        probe_retrieved=probe_retrieved,
        decision_rows=decision_rows,
    )


# -- CSV cell formatting -----------------------------------------------------


def _cell(value: object) -> str:
    """Render one CSV cell: None -> "", bool -> "true"/"false", everything else -> str()."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    """Write `rows` to `path` as a header plus one line per row, every cell passed through `_cell`."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([_cell(row.get(name)) for name in fieldnames])


# -- building the six tables -------------------------------------------------


def build_runs_rows(analyses: list[RunAnalysis]) -> list[dict]:
    """runs.csv: one row per run -- counts, control bytes in/out of the window, and probe counts.

    2026-10-23: a run with `analysis_error` set (its events/t0 evidence
    could not be analyzed at all, see `analyze_run`) still gets a row
    here -- p/seed/valid/flags/analysis_error filled from what
    `read_meta` actually found, every metric column left empty rather
    than a misleading 0. Dropping the row entirely (the previous
    behavior) silently removed exactly the runs most likely to be the
    ones a condition's own failure-rate judgment needs to see. An
    "incomplete" run (see `aggregate`) gets the identical empty-metrics
    row shape, minus `analysis_error` (left empty too -- there is
    nothing to analyze at all, let alone an error from trying).
    """
    rows = []
    for run in analyses:
        if run.analysis_error is not None or run.status == "incomplete":
            rows.append(
                {
                    "p": run.p,
                    "seed": run.seed,
                    "valid": run.status,
                    "flags": ";".join(run.flags),
                    "analysis_error": run.analysis_error,
                }
            )
            continue
        n_pairs = len(run.discovery_rows)
        n_delivered = sum(1 for row in run.discovery_rows if not row["censored"])
        n_censored = n_pairs - n_delivered
        # `None`, not 0, when every node's own window value is None (no
        # usable data anywhere) -- summing an empty list would silently
        # report "zero traffic", indistinguishable from a run that
        # genuinely sent nothing (same "no data" convention as
        # semindex.analyze.control_bytes's own total).
        sent_values = [v for v in run.control_bytes_sent_window.values() if v is not None]
        received_values = [v for v in run.control_bytes_received_window.values() if v is not None]
        sent_total = sum(sent_values) if sent_values else None
        received_total = sum(received_values) if received_values else None
        rows.append(
            {
                "p": run.p,
                "seed": run.seed,
                "valid": run.status,
                "flags": ";".join(run.flags),
                "n_records_created": run.n_records_created,
                "n_pairs": n_pairs,
                "n_delivered_in_window": n_delivered,
                "n_censored": n_censored,
                "control_bytes_sent_window_per_node": json.dumps(run.control_bytes_sent_window, sort_keys=True),
                "control_bytes_received_window_per_node": json.dumps(
                    run.control_bytes_received_window, sort_keys=True
                ),
                "control_bytes_sent_window_total": sent_total,
                "control_bytes_received_window_total": received_total,
                "control_bytes_sent_per_record": (
                    sent_total / run.n_records_created if sent_total is not None and run.n_records_created else None
                ),
                "control_bytes_received_per_record": (
                    received_total / run.n_records_created
                    if received_total is not None and run.n_records_created
                    else None
                ),
                "probe_issued": run.probe_issued,
                "probe_discovered": run.probe_discovered,
                "probe_retrieved": run.probe_retrieved,
                "t0_ms": run.t0_ms,
                "t0_source": run.t0_source,
                "t0_agent_start_disagreement": run.t0_agent_start_disagreement,
                "t0_consistency_ms": run.t0_consistency_ms,
            }
        )
    return rows


RUNS_FIELDS = [
    "p",
    "seed",
    "valid",
    "flags",
    "analysis_error",
    "n_records_created",
    "n_pairs",
    "n_delivered_in_window",
    "n_censored",
    "control_bytes_sent_window_per_node",
    "control_bytes_received_window_per_node",
    "control_bytes_sent_window_total",
    "control_bytes_received_window_total",
    "control_bytes_sent_per_record",
    "control_bytes_received_per_record",
    "probe_issued",
    "probe_discovered",
    "probe_retrieved",
    "t0_ms",
    "t0_source",
    "t0_agent_start_disagreement",
    "t0_consistency_ms",
]


def build_discovery_rows(analyses: list[RunAnalysis]) -> list[dict]:
    """discovery.csv: one row per (run, origin, seq, node) pair, every run's rows included regardless of status

    (map.md counts flagged runs out of AGGREGATES, not out of this raw
    per-pair table; survival.csv is what actually excludes them, at
    pooling time).
    """
    rows = []
    for run in analyses:
        for row in run.discovery_rows:
            rows.append(
                {
                    "p": run.p,
                    "seed": run.seed,
                    "origin": row["origin"],
                    "seq": row["seq"],
                    "node": row["node"],
                    "hop_distance": row["hop_distance"],
                    "latency_ms": row["latency_ms"],
                    "censored": row["censored"],
                }
            )
    return rows


DISCOVERY_FIELDS = ["p", "seed", "origin", "seq", "node", "hop_distance", "latency_ms", "censored"]


def build_consistency_rows(analyses: list[RunAnalysis]) -> list[dict]:
    """consistency.csv: one row per (run, origin, seq) -- time until the last node stored it, or censored."""
    rows = []
    for run in analyses:
        for row in run.consistency_rows:
            rows.append(
                {
                    "p": run.p,
                    "seed": run.seed,
                    "origin": row["origin"],
                    "seq": row["seq"],
                    "time_ms": row["time_ms"],
                    "censored": row["censored"],
                }
            )
    return rows


CONSISTENCY_FIELDS = ["p", "seed", "origin", "seq", "time_ms", "censored"]


def _included(analyses: list[RunAnalysis], include_unknown: bool) -> list[RunAnalysis]:
    """Every run except flagged and incomplete ones, and -- unless `include_unknown` -- except unknown ones too.

    2026-10-23 (revised): flagged runs (map.md "印つきの run は集計から外
    す") are always excluded. "Unknown" runs (no meta.json at all -- its
    producer does not exist yet, see `read_meta`) default to EXCLUDED as
    well, the same treatment as flagged: a run this module cannot
    positively confirm as passing is not evidence it passed, so by
    default it is left out of every pooled aggregate and only counted
    (summary.csv's own n_unknown). `--include-unknown` overrides that for
    a manual trial run that has no meta.json by construction (e.g.
    trial-manual) and still needs a real number out the other end.
    "Incomplete" runs (see `aggregate`) are ALWAYS excluded, regardless
    of `include_unknown` -- there is no verdict to include even under
    that flag, since the run never finished and has no metrics at all.
    """
    excluded_statuses = {"flagged", "incomplete"} if include_unknown else {"flagged", "unknown", "incomplete"}
    return [run for run in analyses if run.status not in excluded_statuses]


def build_survival_rows(analyses: list[RunAnalysis], include_unknown: bool) -> list[dict]:
    """survival.csv: per (p, hop_distance), the KM delivered-fraction curve plus its median/Q1/Q3.

    Pools every discovery pair from every INCLUDED run (see `_included`)
    sharing a (p, hop_distance), each contributing one KM sample: its
    latency (delivered) or its follow_up_ms (censored, from
    `analyze_run`'s own per-pair computation -- the window's close minus
    that specific record's own creation time, never a flat window span).
    """
    included = _included(analyses, include_unknown)
    samples_by_group: dict[tuple[float, int], list[Sample]] = {}
    for run in included:
        for row in run.discovery_rows:
            key = (run.p, row["hop_distance"])
            samples_by_group.setdefault(key, []).append(_row_to_sample(row, "latency_ms"))

    rows = []
    grid = _survival_grid()
    for key in sorted(samples_by_group):
        p, hop = key
        samples = samples_by_group[key]
        steps = kaplan_meier_steps(samples)
        median_s = km_quantile(steps, 0.5)
        q1_s = km_quantile(steps, 0.25)
        q3_s = km_quantile(steps, 0.75)
        n_pairs = len(samples)
        # 2026-10-23: counts/fraction never actually delivered within the
        # window at all (map.md's own "打ち切りの割合"), so a reader can
        # see e.g. "80% censored" directly next to the curve instead of
        # re-deriving it from runs.csv's per-run counts.
        n_censored = sum(1 for s in samples if not s.delivered)
        censored_fraction = n_censored / n_pairs if n_pairs else None
        for t in grid:
            rows.append(
                {
                    "p": p,
                    "hop_distance": hop,
                    "t_s": t,
                    "delivered_fraction": km_delivered_fraction_at(steps, t),
                    "n_pairs": n_pairs,
                    "n_censored": n_censored,
                    "censored_fraction": censored_fraction,
                    "median_s": median_s,
                    "q1_s": q1_s,
                    "q3_s": q3_s,
                }
            )
    return rows


SURVIVAL_FIELDS = [
    "p",
    "hop_distance",
    "t_s",
    "delivered_fraction",
    "n_pairs",
    "n_censored",
    "censored_fraction",
    "median_s",
    "q1_s",
    "q3_s",
]


def build_summary_rows(analyses: list[RunAnalysis], include_unknown: bool) -> list[dict]:
    """summary.csv: per p, run-status counts (including n_incomplete), discovery/consistency/control-byte summaries, probe discovery rate."""
    by_p: dict[float, list[RunAnalysis]] = {}
    for run in analyses:
        by_p.setdefault(run.p, []).append(run)

    rows = []
    for p in sorted(by_p):
        runs = by_p[p]
        included = _included(runs, include_unknown)
        n_valid = sum(1 for r in runs if r.status == "valid")
        n_flagged = sum(1 for r in runs if r.status == "flagged")
        n_unknown = sum(1 for r in runs if r.status == "unknown")
        # 2026-10-23: a run with no DONE marker (see `aggregate`) never
        # finished at all -- counted separately from n_flagged/n_unknown
        # so map.md's own ">5% flagged" threshold is judged against a
        # condition's actually-finished runs, never silently inflated or
        # deflated by ones still awaiting a redo.
        n_incomplete = sum(1 for r in runs if r.status == "incomplete")

        latency_samples = [
            _row_to_sample(row, "latency_ms") for run in included for row in run.discovery_rows
        ]
        latency_steps = kaplan_meier_steps(latency_samples)
        n_pairs = len(latency_samples)
        n_censored_pairs = sum(1 for s in latency_samples if not s.delivered)
        pairs_censored_fraction = n_censored_pairs / n_pairs if n_pairs else None

        consistency_samples = [
            _row_to_sample(row, "time_ms") for run in included for row in run.consistency_rows
        ]
        consistency_steps = kaplan_meier_steps(consistency_samples)

        sent_per_record = [
            sum(v for v in run.control_bytes_sent_window.values() if v is not None) / run.n_records_created
            for run in included
            if run.n_records_created and any(v is not None for v in run.control_bytes_sent_window.values())
        ]
        received_per_record = [
            sum(v for v in run.control_bytes_received_window.values() if v is not None) / run.n_records_created
            for run in included
            if run.n_records_created and any(v is not None for v in run.control_bytes_received_window.values())
        ]
        sent_med, sent_q1, sent_q3 = median_q1_q3(sent_per_record)
        recv_med, recv_q1, recv_q3 = median_q1_q3(received_per_record)

        probe_issued_total = sum(r.probe_issued for r in included)
        probe_discovered_total = sum(r.probe_discovered for r in included)
        rate = probe_discovered_total / probe_issued_total if probe_issued_total else None
        wilson = wilson_interval(probe_discovered_total, probe_issued_total)

        rows.append(
            {
                "p": p,
                "n_valid": n_valid,
                "n_flagged": n_flagged,
                "n_unknown": n_unknown,
                "n_incomplete": n_incomplete,
                "n_pairs": n_pairs,
                "n_censored_pairs": n_censored_pairs,
                "pairs_censored_fraction": pairs_censored_fraction,
                "discovery_latency_median_ms": _s_to_ms(km_quantile(latency_steps, 0.5)),
                "discovery_latency_q1_ms": _s_to_ms(km_quantile(latency_steps, 0.25)),
                "discovery_latency_q3_ms": _s_to_ms(km_quantile(latency_steps, 0.75)),
                "consistency_time_median_ms": _s_to_ms(km_quantile(consistency_steps, 0.5)),
                "consistency_time_q1_ms": _s_to_ms(km_quantile(consistency_steps, 0.25)),
                "consistency_time_q3_ms": _s_to_ms(km_quantile(consistency_steps, 0.75)),
                "control_bytes_sent_per_record_median": sent_med,
                "control_bytes_sent_per_record_q1": sent_q1,
                "control_bytes_sent_per_record_q3": sent_q3,
                "control_bytes_received_per_record_median": recv_med,
                "control_bytes_received_per_record_q1": recv_q1,
                "control_bytes_received_per_record_q3": recv_q3,
                "probe_issued_total": probe_issued_total,
                "probe_discovered_total": probe_discovered_total,
                "probe_discovery_rate": rate,
                "probe_discovery_rate_wilson_lo": wilson[0] if wilson else None,
                "probe_discovery_rate_wilson_hi": wilson[1] if wilson else None,
            }
        )
    return rows


def _s_to_ms(value: float | None) -> float | None:
    """Convert a KM quantile from seconds to milliseconds, passing None through."""
    return None if value is None else value * 1000.0


SUMMARY_FIELDS = [
    "p",
    "n_valid",
    "n_flagged",
    "n_unknown",
    "n_incomplete",
    "n_pairs",
    "n_censored_pairs",
    "pairs_censored_fraction",
    "discovery_latency_median_ms",
    "discovery_latency_q1_ms",
    "discovery_latency_q3_ms",
    "consistency_time_median_ms",
    "consistency_time_q1_ms",
    "consistency_time_q3_ms",
    "control_bytes_sent_per_record_median",
    "control_bytes_sent_per_record_q1",
    "control_bytes_sent_per_record_q3",
    "control_bytes_received_per_record_median",
    "control_bytes_received_per_record_q1",
    "control_bytes_received_per_record_q3",
    "probe_issued_total",
    "probe_discovered_total",
    "probe_discovery_rate",
    "probe_discovery_rate_wilson_lo",
    "probe_discovery_rate_wilson_hi",
]


# Every numeric decision field this module reports a distribution for,
# regardless of `kind` -- `state` is recorded on every decision
# (map.md's own "判断の記録" shape), so these seven are never kind-
# specific. `action`'s own fields differ by kind instead (_ACTION_FIELDS
# below), since map.md's own `action` shape differs by kind too.
# `advertised_hi`/`advertised_notified_max` (2026-10-23) sit alongside
# `advertised_gap` rather than replacing it: the gap is the comparison
# map.md asks for, but reporting only the gap loses the two values that
# produced it (see analyze_run's own decision_rows construction).
_STATE_NUMERIC_FIELDS = (
    "in_flight",
    "hi",
    "missing",
    "advertised_gap",
    "advertised_hi",
    "advertised_notified_max",
    "last_timeout_age_s",
)

# Which action field each kind's own `action` shape carries (map.md
# "action は種類ごとに違う"): state_poll -> interval_s, fetch_plan ->
# how many records that tick's fetch sent (len(fetched)), fetch_backoff
# -> wait_s. A kind not in this map (there is none today) would simply
# report no action-field row at all.
_ACTION_FIELD_BY_KIND = {
    "state_poll": "interval_s",
    "fetch_plan": "fetched_count",
    "fetch_backoff": "wait_s",
}


def build_decisions_rows(analyses: list[RunAnalysis], include_unknown: bool) -> list[dict]:
    """decisions.csv: per (p, kind, field), a tidy long-format row of that field's own distribution.

    2026-10-23: redesigned from one wide row per (p, kind) covering only
    in_flight/missing/wait_s to one row per (p, kind, field) covering
    every field map.md's own `decision` event records -- state's
    in_flight/hi/missing/advertised_gap/advertised_hi/
    advertised_notified_max/last_timeout_age_s (scalar or flattened-list,
    see analyze_run's own decision_rows construction) as
    median/Q1/Q3, state's own boolean `truncated` as a frequency instead
    (a median of a 0/1 indicator is not the number anyone asking "how
    often" wants), and whichever action field this kind's own `action`
    shape carries (_ACTION_FIELD_BY_KIND) as median/Q1/Q3 too. Long,
    tidy rows rather than ever-more wide-format columns, so a reader (or
    a plotting tool) can filter on `field` directly instead of needing a
    new column name memorized for every field this module might report.
    """
    included = _included(analyses, include_unknown)
    groups: dict[tuple[float, str], list[dict]] = {}
    for run in included:
        for decision in run.decision_rows:
            groups.setdefault((run.p, decision["kind"]), []).append(decision)

    def _field_row(p: float, kind: str, decision_count: int, field_name: str, values: list[float]) -> dict:
        """One tidy row for a numeric field's own distribution (median/Q1/Q3), `frequency` left empty."""
        med, q1, q3 = median_q1_q3(values)
        return {
            "p": p,
            "kind": kind,
            "decision_count": decision_count,
            "field": field_name,
            "n": len(values),
            "median": med,
            "q1": q1,
            "q3": q3,
            "frequency": None,
        }

    rows = []
    for key in sorted(groups):
        p, kind = key
        decisions = groups[key]
        decision_count = len(decisions)

        for field_name in _STATE_NUMERIC_FIELDS:
            raw = [d[field_name] for d in decisions if field_name in d]
            # hi/missing/advertised_gap/advertised_hi/advertised_notified_max
            # are already-flattened lists on each decision row;
            # in_flight/last_timeout_age_s are scalars.
            values = [v for entry in raw for v in (entry if isinstance(entry, list) else [entry]) if v is not None]
            rows.append(_field_row(p, kind, decision_count, field_name, values))

        action_field = _ACTION_FIELD_BY_KIND.get(kind)
        if action_field is not None:
            values = [d[action_field] for d in decisions if d.get(action_field) is not None]
            rows.append(_field_row(p, kind, decision_count, action_field, values))

        truncated_values = [d["truncated"] for d in decisions if d.get("truncated") is not None]
        frequency = (
            sum(1 for v in truncated_values if v) / len(truncated_values) if truncated_values else None
        )
        rows.append(
            {
                "p": p,
                "kind": kind,
                "decision_count": decision_count,
                "field": "truncated",
                "n": len(truncated_values),
                "median": None,
                "q1": None,
                "q3": None,
                "frequency": frequency,
            }
        )
    return rows


DECISIONS_FIELDS = ["p", "kind", "decision_count", "field", "n", "median", "q1", "q3", "frequency"]


# -- CLI --------------------------------------------------------------------


def aggregate(
    store: str, stage: str, outdir: str, include_undone: bool, include_unknown: bool = False
) -> list[RunAnalysis]:
    """Discover every run, analyze the ones this call will process, write all six CSVs into `outdir`.

    Returns every run's RunAnalysis -- every discovered run gets one now,
    including a run missing its DONE marker. `analyze_run` itself no
    longer raises for a run whose events/t0 evidence is unreadable (see
    its own docstring and RunAnalysis.analysis_error), so every such run
    still gets a row too, not just the ones with clean data.

    2026-10-23 fix: a run dir missing DONE used to be skipped outright
    (no row anywhere) under the assumption that DONE marks "this run's
    own preconditions passed" -- but `semindex.evalrun.run_one` writes
    DONE unconditionally, after a run reaches the end of its own
    pipeline, flagged or not (see `run_has_done_marker`'s own docstring).
    A missing DONE therefore means the run never finished at all (an
    interrupted sweep, not a failed one), and silently dropping it from
    every table under-reported the OPPOSITE of what map.md's own
    ">5% flagged" threshold needs: a condition cannot be judged against
    a count that already excluded the runs most likely to still need a
    redo. Such a run now gets a status="incomplete" row (p/seed and
    status only, every metric column and `flags` empty -- there is
    nothing stored yet to analyze), counted in summary.csv's own
    n_incomplete, and always excluded from pooling (see `_included`) regardless of
    `include_unknown` -- there is no verdict to pool either way.
    `--include-undone` keeps its own prior meaning unchanged: it is for
    the one case that has no DONE by construction and is NOT awaiting a
    redo (a hand-stored manual trial run), and routes straight into
    `analyze_run` as before, with whatever status its own meta.json (or
    lack of one) gives it.

    `include_unknown` only changes which runs `survival.csv`/
    `summary.csv`/`decisions.csv` POOL (see `_included`) -- runs.csv and
    discovery.csv always carry every discovered run, regardless of
    either flag.
    """
    refs = discover_runs(store, stage)
    analyses: list[RunAnalysis] = []
    for ref in refs:
        if not include_undone and not run_has_done_marker(ref.run_dir):
            analyses.append(RunAnalysis(p=ref.p, seed=ref.seed, status="incomplete", flags=[]))
            continue
        try:
            analyses.append(analyze_run(ref.p, ref.seed, ref.run_dir))
        except CheckUsageError as exc:
            # Last-resort safety net: analyze_run's own try/except already
            # turns an unreadable run into a RunAnalysis with
            # analysis_error set rather than raising, so reaching this
            # clause means something unanticipated broke even reading
            # meta.json/DONE -- still skip only this one run (not the
            # whole batch), with the same "not processed at all" shape,
            # since there is no status/flags to attach a row to here
            # either.
            print(f"semindex.evalagg: skipping {ref.run_dir}: {exc}", file=sys.stderr)
            continue

    os.makedirs(outdir, exist_ok=True)
    _write_csv(os.path.join(outdir, "runs.csv"), RUNS_FIELDS, build_runs_rows(analyses))
    _write_csv(os.path.join(outdir, "discovery.csv"), DISCOVERY_FIELDS, build_discovery_rows(analyses))
    _write_csv(
        os.path.join(outdir, "survival.csv"), SURVIVAL_FIELDS, build_survival_rows(analyses, include_unknown)
    )
    _write_csv(os.path.join(outdir, "consistency.csv"), CONSISTENCY_FIELDS, build_consistency_rows(analyses))
    _write_csv(
        os.path.join(outdir, "summary.csv"), SUMMARY_FIELDS, build_summary_rows(analyses, include_unknown)
    )
    _write_csv(
        os.path.join(outdir, "decisions.csv"), DECISIONS_FIELDS, build_decisions_rows(analyses, include_unknown)
    )
    if not FIGURES_AVAILABLE:
        print("semindex.evalagg: matplotlib not importable in this venv -- wrote CSVs only, no figures", file=sys.stderr)
    return analyses


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the aggregation script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="Root directory holding <store>/<stage>/p<p>/<seed>/ runs")
    parser.add_argument("--stage", required=True, help="Stage label (the <stage> path component under --store)")
    parser.add_argument("--out", required=True, dest="outdir", help="Directory to write the six CSV files into")
    parser.add_argument(
        "--include-undone",
        action="store_true",
        help="Also process runs with no DONE marker (e.g. a hand-stored trial run)",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Also pool runs with no meta.json verdict into survival/summary/decisions (e.g. a manual trial run)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: python3 -m semindex.evalagg --store ... --stage ... --out ..."""
    args = parse_args(argv)
    aggregate(args.store, args.stage, args.outdir, args.include_undone, args.include_unknown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
