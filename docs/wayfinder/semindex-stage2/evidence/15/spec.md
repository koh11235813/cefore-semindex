# evidence/15 spec: on/off workloads and a rule that also reacts to timeouts

Written 2026-11-17 before any number of this step was seen. User decision Q64.

## Purpose

evidence/14 left two inferences untested:

1. The lengthening rule with I_min 3.695 s (T37) wins only where "a record that just came makes another
   one likely". The testbed workload has that property only as one 60 s block. A workload where generation
   switches on and off several times should then favour T37 over a fixed interval, where there is no outage.
2. With outages T37's gain disappears. T37 leaves its interval unchanged on a timeout. If that is the
   reason, a T37 that moves its interval on a timeout should recover some of the gain in the outage cells.

Both are checked on the evidence/13 replay, with the outages of the logged runs and replaced record
creation times, as in evidence/14. The replaced workloads have no measurement behind them: every number is
a simulated value on an unvalidated input and points at which testbed runs are worth doing.

## Simulator

`../13/replay.py` is imported and must not be modified. The timeout variants need one behaviour the
simulator does not have (changing the rule's interval when the newest poll times out). Implement it in this
directory as a subclass of the replay engine that overrides only that step, and run every rule of this step
through the subclass. Proof required that the subclass changes nothing for rules without the new key: the W0
check below.

## Workloads

Functions of (p, seed, origin) through `random.Random(f"ev15:{name}:{p}:{seed}:{origin}")`; origins h0, h4
independent; times whole milliseconds in [t0+13 s, t0+110 s]; per origin seq 1..N in time order.
While ON an origin creates 4 records every 2 s (the testbed's rate and batch size).

- W0 control: logged creation times. Must reproduce `../14/workload_cells.csv` (W0 rows) for the rules both
  steps share, to every printed digit.
- WO on/off regular: first ON starts at U[13, 33] s; ON 20 s, OFF 20 s, repeating until 110 s (a tick is
  emitted only if its time is <= 110 s).
- WV on/off variable: first ON starts at U[13, 33] s; each ON and each OFF length is drawn from an
  exponential with mean 20 s, clipped to [4, 60] s; until 110 s.

## Rules

Frontier: the seven fixed arms of evidence/14 (0.5, 0.8, 1.359, 2.2, 3.695, 6.0, 10.043 s).

Tested, all with I_min 3.695 s, m 2, I_max 10.043 s, jitter 0.1:
- T37: on a timeout the interval is unchanged (evidence/13's rule).
- T37r ("reset"): when the newest poll times out, I = I_min.
- T37b ("back off"): when the newest poll times out, I = I_max.
Also short and wait, as reference points. No other rule.

The timeout step applies only to the newest outstanding poll of that neighbor, exactly where the simulator
resolves a timeout today; the next poll time is then computed as for an answered poll (same jitter draw,
max(sent + applied, now)).

## Cells

s6b10p043 p 0.2 and p 0.4 (outages 8-12 s), s5b10p043 p 0 (no outage). 20 seeds, Mode B, 5 replicates.

## Comparison

As ../14/spec.md: gain against the seven-arm frontier in log-log over polls_sent, bootstrap over seeds
(2000, default_rng(20261117)), 1e-9 rule, per-axis "outside", cens flag.

Because evidence/14 found that late records are censored more under sparse rules and that the planned flag
cannot catch it, the HEADLINE gain of this step is computed on records created at or before t0+72 s
(delay and cens4 over those records; polls over the whole window), as ../14/after.py does. The gain over
all records is reported beside it.

Also per (cell, workload, rule): records per run, EMPTY share, mean gap, length-weighted gap, and the
interval in force when a non-empty reply arrived (mean applied interval of the answered poll), which is the
direct reading of "is the rule asking fast while data is flowing".

## Reading rule, fixed now

Inference 1 (on/off favours T37 without outages): supported if, in the no-outage cell, T37's headline gain
interval lies entirely above zero under WO and under WV. Not supported if it spans zero or lies below in
both. Otherwise mixed.

Inference 2 (reacting to timeouts recovers the gain under outages): for each of T37r and T37b, compute per
outage cell and workload the paired difference gain(variant) - gain(T37) with its bootstrap interval (same
resamples). Supported for a variant if the difference interval lies entirely above zero in at least 4 of
the 6 (2 outage cells x 3 workloads) cases. Not supported if it does so in at most 1. Otherwise mixed.
In the no-outage cell the three rules must give identical numbers (no timeout occurs); checked.

## Checks required

- W0 reproduces ../14/workload_cells.csv for the shared rules (seven arms, short, wait, T37), all columns
  both files print.
- T37, T37r, T37b identical in the no-outage cell for every workload.
- Generated lists satisfy the constraints (asserted for every run).
- Rerun gives byte-identical csv. ruff clean with the repo settings.

## Deliverables (this directory)

- onoff.py, onoff_cells.csv, onoff_gain.csv (one row per cell x workload x tested rule, with headline and
  all-record gains and the paired differences), onoff.md (tables, checks, deviations).
- README.md and RUN.txt are written by the main session after review.
