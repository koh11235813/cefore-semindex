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

## Amendments of 2026-11-19, written while the runs were in progress and before any number was read

An adversarial review of this file was done after the sweep had been started (it takes over five hours).
No run of this step had been aggregated or opened when the following was added; the only thing known was
the count of finished runs. Where an amendment conflicts with the text above, the amendment holds.

1. Completeness. Pass line 1 reads "all 120 runs DONE and valid". Every other line uses the seeds that are
   DONE and unflagged in all of the stages that line names; the number of seeds used is printed beside the
   verdict, and a line computed on fewer seeds than it names gets the word "incomplete" in place of its
   verdict.
2. Which reading carries the verdict. The replication (pass line 2, seeds 21 to 40) is the reading that
   goes into `README.md` and `map.md` as the answer to "is the difference seen". The pooled value (pass
   line 3) is reported beside it as the estimate of the size, with its interval, and its word is never
   quoted without the sentence that it is not an independent test.
3. What the replication cannot show. On 20 seeds the interval half width in evidence/16 was about 11
   points. If the true difference is the +9.8 points evidence/16 measured, a 20-seed interval clears zero
   only about half the time. A "not seen" on line 2 therefore does not say the difference is absent, and a
   split result (line 2 "not seen", line 3 "seen") is the single most likely outcome under that size. It is
   then reported as: not confirmed by the independent replication; pooled estimate so-and-so.
4. Bootstrap unit. The unit resampled is the seed number. A seed carries the value of every stage in use
   for that seed (for seeds 1 to 20 the arms from `s6`, the rules from `s7`; for 21 to 40 all from `s8`;
   the 6.0 s arm from `s8b6p0` for every seed). 20 units for the replication, 40 for the pooled set.
5. Three-arm frontier and code versions. In the seeds 1 to 40 frontier the 6.0 s arm is all new code while
   the other two arms are half old code, and evidence/16's rerun check saw delay shifts up to 1.8 % between
   the versions on single runs, twice the 0.9 % curvature the replay expects. So the reading of pass line 4
   is taken on seeds 21 to 40, where all three arms and both rules are the same code and none of the seeds
   was read before. The seeds 1 to 40 version is reported beside it without verdict words. The same holds
   for the 6.0 s arm's own gain against the two-arm line: read on seeds 21 to 40, with the same bootstrap
   interval as the other gains; spanning zero reads "the two-arm line and the three-arm frontier are not
   distinguishable", entirely above zero "the trade-off bends below the straight line", entirely below zero
   "the trade-off bends above the straight line".
6. When the frontier was chosen. Measuring the 6.0 s arm and attaching words to the three-arm frontier was
   decided after evidence/16 had read "not seen". The frontier's definition itself is the one
   `../16/predict.py` printed before those runs. The replay expects the three-arm frontier to raise each
   rule's own gain by about 0.9 points and to leave the difference between the rules almost unchanged
   (15.9 against 15.8 points).
7. Pass line 5, what the band is centred on. 29.744 s and 159.4 polls come from `predict.txt`'s pooled
   line, the mean over 40 replicate streams, which `predict.py` itself says a one-run-per-seed testbed
   cannot be expected to match as closely as the per-experiment median the evidence/16 bands used. No
   per-experiment range was printed for this arm, so how much of the 6 % is single-run noise is unknown.
   The replay simulates this arm as a rule with `i_min_s` equal to `i_max_s` (6.0, jitter 0.1); the testbed
   stage uses `--interval-choices 6.0 --choice-jitter 0.1`. An outside value is reported as "the replay's
   6.0 s arm missed"; lines 2 to 4 use measured arms only and are read regardless.
8. Run environment. `main7.sh` sets `PYTHONDONTWRITEBYTECODE=1`; the evidence/10 and evidence/16 runs did
   not. Each process therefore compiles the package at import instead of reading a cache. The compile
   happens at import, before t0; no effect on the windowed quantities is known and none was measured. It is
   the one named environment difference between the halves that line 3 pools, and one more reason the
   verdicts are read on seeds 21 to 40.
