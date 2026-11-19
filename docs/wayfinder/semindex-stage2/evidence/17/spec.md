# evidence/17 spec: more seeds and a third fixed arm for the timeout-reset rule

Written 2026-11-19 before any run of this step. User decision Q70 (do all three follow-ups; this is the
one called C). The pass lines below are not moved after the runs.

## Why

evidence/16 measured T37 and T37r at p 0.4 on 20 seeds. The gain difference T37r minus T37 against the line
through the two fixed arms was +9.76 points with a 95 % interval of -2.24 to +19.68, so by the pre-set
reading it was "not seen". Two things were left open there: whether 20 seeds were simply too few, and
whether the straight line between two arms is where the fixed-interval trade-off really runs. This step adds
20 new seeds for the four stages and a measured fixed arm at 6.0 s.

Not answered here: other p, other workloads, whether a learned policy beats a formula.

## Reading the same hypothesis twice

The difference was already read once on seeds 1 to 20. Adding seeds until an interval clears zero would be
optional stopping, so the two readings are kept apart and both are fixed now:

- the replication (pass line 2) uses seeds 21 to 40 only and is independent of what evidence/16 read;
- the pooled value (pass line 3) uses seeds 1 to 40 and is the best estimate, but it contains the 20 seeds
  already read and is not an independent test. It is reported with that sentence beside it.

No further seeds are added to this question after this step, whatever the verdicts are.

## Runs (`main7.sh`, from a detached clean checkout of main `eadee32`; `src/` is identical to `077a700`)

All p 0.4, common settings as evidence/16: `--k 4 --store-synchronous NORMAL --disconnect-duration-s 8 12`.
The outage schedule depends on (p, seed) only.

| stage | setting | seeds | runs |
|---|---|---|---|
| `s8t37` | rule, `on_timeout` keep (as `s7t37`) | 21 to 40 | 20 |
| `s8t37r` | rule, `on_timeout` reset (as `s7t37r`) | 21 to 40 | 20 |
| `s8b3p695` | fixed 3.695 s, jitter 0.1 (as `s6b3p695`) | 21 to 40 | 20 |
| `s8b10p043` | fixed 10.043 s, jitter 0.1 (as `s6b10p043`) | 21 to 40 | 20 |
| `s8b6p0` | fixed 6.0 s, jitter 0.1 | 1 to 40 | 40 |

120 runs, about 5 h 20 min. New stage names, so that the stores evidence/10 and evidence/16 were computed
from keep exactly the runs they had. Order: the four seed-21-to-40 stages first, then the 6.0 s arm.

For seeds 1 to 20 the arms are the evidence/10 runs (`s6b3p695`, `s6b10p043`, measured 2026-11-12 with the
code before the rule was added) and the rules are the evidence/16 runs (`s7t37`, `s7t37r`). evidence/16
pass line 4 showed the existing settings unchanged by the rule's code on one seed each; that is the only
support for mixing the two code versions and it is stated wherever the mixed set is used.

## Quantities

As evidence/16 `spec.md` pass line 2: delay = pooled `sum(lat4) / sum(n4)` over the runs of a stage;
censored share = `sum(c4) / sum(n4 + c4)`; polls per run = mean of windowed `polls_sent`; line = straight
line in log-log through the 3.695 s and 10.043 s arms; gain = 1 - delay(rule) / line(polls(rule)); no gain
when polls fall outside the arms. Interval = 2000 resamples of the seeds in use with
`default_rng(20261119)`, every stage recomputed from the same resampled seeds; a resample in which a gain is
undefined is counted and left out.

Three-arm frontier (pass line 4): piecewise linear in log-log through the 3.695 s, 6.0 s and 10.043 s arms,
read at the rule's polls per run. If the three arms are not monotone (more polls, less delay) in the full
sample, no frontier gain is computed and the raw values are shown; a resample in which they are not
monotone is counted and left out.

## Predictions available before the runs

From evidence/16 `predict.txt` (replay on the logged outages of `s6b10p043` p 0.4, seeds 1 to 20):

- fixed 6.0 s: delay 29.744 s, polls per run 159.4, censored share 0.003.
- gain difference T37r minus T37 against the three-arm frontier: mean +15.8 points over the 40 replay
  experiments, range +9.2 to +22.5.

There is no replay prediction for seeds 21 to 40: the replay needs the logged outages of a run with those
seeds, and none exists before this step.

## Pass lines

1. All 120 runs valid (no `evalvalid` flag).
2. Replication, seeds 21 to 40 only, the four `s8` stages with those seeds: gain difference T37r minus T37
   against the two-arm line. Interval entirely above zero: "seen". Spanning zero: "not seen". Entirely
   below zero: "reversed".
3. Pooled, seeds 1 to 40 (arms `s6` + `s8`, rules `s7` + `s8`): the same quantity and the same three words.
   Reported as the best estimate, not as an independent test.
4. Against the three-arm frontier, seeds 1 to 40 (6.0 s arm from `s8b6p0`, the other two arms and the rules
   as in line 3): for each rule, gain and interval. Entirely above zero: "above the fixed frontier".
   Spanning zero: "not distinguishable from the fixed frontier". Entirely below zero: "below the fixed
   frontier". The gain difference T37r minus T37 against this frontier is reported with the words of line 2.
   Also reported: the 6.0 s arm's own gain against the two-arm line (it says how far the straight line is
   from the measured trade-off).
5. Replay check of the new arm: `s8b6p0` on seeds 1 to 20, delay and polls per run inside plus or minus 6 %
   of the replay (delay 27.959 to 31.529 s, polls 149.84 to 168.96). The 6 % is the band evidence/16 used.

Also reported, without a pass line: every stage's delay, censored share and polls per run on seeds 1 to 20,
21 to 40 and 1 to 40; the extra quantities of evidence/16 (`empty` share, undecodable replies, late replies,
clamped re-decisions, interval histogram) for `s8t37` and `s8t37r`; the plain ratio of T37r to T37 in delay
and polls with its interval on seeds 21 to 40 and on 1 to 40.

## Deliverables (this directory)

`spec.md`, `main7.sh` before the runs. After the runs: the analysis script, its csv and md output,
`README.md`, `RUN.txt`.
