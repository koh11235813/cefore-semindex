# ev13 spec: replay of the CURRENT rule, and the pass line (fixed before any replay number was seen)

Written 2026-11-17 before replay.py existed. The rules come from facts-timers.md, facts-reply-fetch.md and
facts-metrics.md in this directory (read them first; they cite the source lines and their own log checks).

## Purpose

Question Q61 wants the byte and delay effect of a different state-poll interval rule, computed offline on the
existing runs. That is only worth doing if an offline replay of the rule the runs actually used lands on the
logged numbers. This step builds that replay and scores it. No new rule is tried in this step.

## What is exogenous (taken from the run) and what is simulated

Taken from the run, never simulated:
- topology (line h0..h4) and the link_down/link_up times (the mapping facts-reply-fetch.md fact 8 found best);
- record creation times at h0 and h4 (record_created lines of the origins' logs);
- meta.json policy settings; stop_at_s; t0_ms.

Simulated for all five nodes jointly: every state poll (send time, answered / timed out, reply content),
every fetch (plan, window of 8, timeout, 2/4/8/16 s retry), every record_stored time.

## Two modes

- Mode A, pinned. Jitter draws regenerated from the RNG streams of facts-timers.md fact 4 (or read from
  applied_interval_s, say which and why). The tick lag of each poll (logged ts minus due time) is taken from
  the log, in order, per (node, neighbor); when the simulated run has more polls than the log, lag = 50 ms.
  Purpose: with the same draws and the same lags a correct model reproduces the logged run almost event for
  event, so a logic error shows as a per-event difference.
- Mode B, free. Same jitter streams; tick lag drawn uniform on [0, 100] ms from a simulator RNG
  (seeded by (stage, p, seed, replicate)); R = 5 replicates per run. Purpose: this is the mode a changed rule
  has to run in, since a changed rule has no logged lags. Its fidelity is the one that matters for Q61.

Fixed constants in both modes unless the facts files force something else: poll RTT 1 ms, state_rx to
fetch_plan 0 ms, fetch_plan to record_stored 2 ms, poll timeout 2000 ms judged at the next tick, fetch
timeout 2000 ms + tick lag (Mode A: 50 ms). Constants must be named at the top of the script.

## Measured side

Recomputed from events.jsonl by the script itself (json only, no semindex import), with the definitions of
facts-metrics.md: polls_sent, polls answered, polls timed out, state_rx, in-window; 4-hop delay (uncensored
mean, pooled per cell), cens4 (censored share); state-poll bytes = answered*(INT_B+REP_B[p]) + timed_out*INT_B
where REP_B exists. The simulated side is scored with the same code path on simulated events, so that a
definition slip cannot favour either side. Cross-check the measured side against evidence/10 live_main.csv /
perrun.csv and evidence/11 cells.csv where they hold the same quantity; any difference is fatal.

## Scope

All 42 cells of evidence/11 and 12 (840 runs). Nothing else.

## Pass line

Mode A (per cell, then the number of cells that pass, out of 42):
- A1 poll outcome: simulated poll i of (node, neighbor) has the same outcome as logged poll i, >= 99.5 %.
- A2 store time: |simulated - logged| record_stored time per (node, origin, seq), over records stored in
  both, <= 150 ms for >= 99 %. Records stored in only one of the two are counted and reported separately and
  must be <= 0.5 % of record-node pairs.
- A3 polls_sent within 0.5 %; 4-hop delay within max(1 %, 50 ms); cens4 within 0.5 percentage points.

Mode B (per metric: delay, cens4, polls_sent, polls timed out, state-poll bytes where defined):
- B1 per cell: simulated value (mean over the 5 replicates of the 20-seed cell value) lies inside the 95 %
  seed-bootstrap interval of the measured cell value (2000 resamples, default_rng(20261117)). Under a perfect
  model about 95 % of cells land inside; the line is >= 90 % of cells (38 of 42; 27 of 30 for bytes).
- B2 pooled bias: over the cells, median of (sim - measured) / measured within +-2 % for delay, +-1 % for
  polls_sent and bytes; for cens4 median absolute difference <= 1 percentage point.

Verdict: PASS only if every line holds. Anything else is FAIL or PARTIAL, reported with the layer that broke
(outcome model, reply content, fetch model, timer) and the cells where it broke. Do not tune constants to
move a number across a line; if a constant is changed after seeing results, say so, give the before and
after, and count the result as tuned.

## Deliverables (all in this directory)

- replay.py: the simulator plus scoring; `python3 replay.py` rebuilds everything. Environment STORE.
- validate_cells.csv: one row per cell, measured / mode A / mode B values and each pass flag.
- validate.md: the tables, the verdict per line, the worst cells, and every deviation from this spec.
- A trace option: `replay.py --trace <stage> <p> <seed> <mode>` prints the simulated events of one run in the
  same shape as events.jsonl lines (node, rel time, event, key fields) so that a reader can diff one run.
