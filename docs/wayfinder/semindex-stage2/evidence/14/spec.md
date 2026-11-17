# evidence/14 spec: the same replay under workloads whose start and period are not known in advance

Written 2026-11-17 before any number of this step was seen. User decision Q63.

## Purpose

evidence/13 found that a rule which lengthens the state-poll interval while replies are empty loses to a
fixed interval on the testbed workload (each origin creates 4 records every 2 s from t0+13 s to t0+71 s). The
inference was that it loses because that workload never goes quiet while it runs. This step checks the
inference in the only way available without new runs: keep the outages of the logged runs, replace the record
creation times, and compare the same rules again. If the inference is right, the lengthening rule should
beat a fixed interval where records come rarely or in bursts.

What this can and cannot show: the replay was checked against measurements for the testbed workload only
(evidence/13). For a replaced workload there is no measurement to compare with. Every number here is a
simulated value on an unvalidated input, to be read as a pointer to which testbed runs are worth doing.

## Simulator

`../13/replay.py`, unchanged, Mode B, 5 replicates, imported (not copied). Rules are passed as in
`../13/trickle.py`. The only input replaced is `run["created"]`, the list of (ts_ms, origin, seq).

Constraints on a replaced list: origins h0 and h4 only; per origin seq = 1..N in non-decreasing time order;
all times inside [t0+13 s, t0+110 s] so that every record has at least 20 s before the window ends
(t0+130 s); times are whole milliseconds.

## Workloads

Each workload is a function of (p, seed, origin) through `random.Random(f"ev14:{name}:{p}:{seed}:{origin}")`,
so every rule sees the same creation times on one (p, seed), and the two origins are independent.

- W0 control: the logged creation times, untouched. Must reproduce `../13/trickle_cells.csv` for the cells
  and rules both steps share, to every printed digit. Checked before anything else is read.
- WP periodic, start and period unknown: period drawn from (1, 2, 4) s, start uniform on [13, 50] s, 60 s of
  generation, records per tick = 2 * period (so 120 records per origin, the testbed's count and rate).
- WB bursts: 6 bursts of 20 records per origin, burst times independent uniform on [13, 110] s.
- WR rare: 12 single records per origin, times independent uniform on [13, 110] s.

## Rules

Fixed intervals, 7 arms: 0.5, 0.8, 1.359, 2.2, 3.695, 6.0, 10.043 s, jitter 0.1. The four testbed arms use
their stage settings as in ../13/trickle.py; the three added arms (0.8, 2.2, 6.0) are the rule with m = 1 and
I_min = I_max = X, which ../13/identity.py shows to be a fixed arm. Seven arms instead of four because
../13/after.md showed the four-arm chord flatters a rule that lands mid-segment.

Tested against that frontier: short, wait, and the five variants of ../13/spec2.md (T05, T136, T37, T136s,
T136c), unchanged. No other variant.

## Cells

Donors and p: s6b10p043 p 0.2 and p 0.4 (outages of 8-12 s), s5b10p043 p 0 (no outage). 20 seeds.
3 cells x 4 workloads.

## Comparison

As ../13/spec2.md Part 3: gain = 1 - delay(rule) / delay_frontier(polls_sent(rule)), the frontier being
piecewise linear in log-log through the seven simulated fixed arms of the same cell and workload; save is the
same read the other way; 95 % interval by bootstrap over the 20 seeds (2000 resamples,
default_rng(20261117)), frontier and rule recomputed per resample; "outside" per axis as in evidence/13;
a gain within 1e-9 of zero counts as zero; cens flag when cens4 exceeds both bracketing arms by more than
1 percentage point. delay is the 4-hop uncensored mean; cens4 is reported beside every gain.

Also reported per (cell, workload, rule), because evidence/13 found them to be what explains a gain:
EMPTY share of answered polls; mean gap and length-weighted gap (sum g^2 / sum g) between consecutive polls
to one neighbor; polls sent after the node held every record of the run.

## Reading rule, fixed now

For each workload: "the lengthening rule helps" is claimed only for a variant whose gain interval lies
entirely above zero in at least 2 of the 3 cells without a cens flag. "It does not help" is claimed for a
variant whose gain is at or below zero, or whose interval spans zero, in all 3 cells. Anything else is
reported as mixed. The frontier of WR may be flat or non-monotone (few records, noisy delay); if it is not
monotone in a cell, no gain is computed for that cell and the raw delay and polls are shown instead.

## Checks required

- W0 reproduces ../13/trickle_cells.csv (above).
- Each generated list satisfies the constraints (asserted in code for every run).
- Rerun gives byte-identical csv.
- ruff clean with the repo settings.

## Deliverables (this directory)

- workload.py (generators, runs, tables), workload_cells.csv (one row per cell x workload x rule),
  workload_gain.csv (one row per cell x workload x tested rule), workload.md (tables, the W0 check,
  deviations).
- README.md and RUN.txt are written by the main session after review.
