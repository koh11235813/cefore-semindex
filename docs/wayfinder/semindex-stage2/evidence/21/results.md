# evidence/21 results

Computed by ana.py from perrun.csv and the frozen predict.txt of this directory. The pass lines, the bands, the tolerance formula and the verdict words are spec.md's; the words are read off the numbers mechanically and nothing here adds to them.

One stage at two p: `s11a` p 0.2 and p 0.4, workload `w0`, seeds 1-100 in each cell, 200 runs, the choices mode (3.695 / 6.0 / 10.043 s, jitter 0.1) re-deciding its interval at every resolution. Decision rows are counted over the whole run, the polls per run over the window [t0+10 s, t0+130 s], and the delay over the records created at or before t0+72 s, which is what spec.md "Quantities" fixes for each.

The predicted constants carried in ana.py were checked against predict.txt at run time: 6 rows of predicted numbers, every number in them equal to the digits that file prints. A mismatch stops the script, so the analysis cannot drift from the frozen prediction.

## Line 3: all 200 runs DONE and valid

DONE means the run has a row in perrun.csv: perrun.py writes only runs that carry a DONE marker. Valid is `meta.json` -> `validity`, written by semindex.evalrun out of semindex.evalvalid. The flag names carry the node and the sequence number (`record_off_schedule:h0:seq5`), so the breakdown counts them by the first word of the name. It is printed whether or not any flag appeared.

| p | runs expected | runs DONE | valid and unflagged | flagged | flag names |
|---|---|---|---|---|---|
| 0.2 | 100 | 100 | 100 | 0 | none |
| 0.4 | 100 | 100 | 100 | 0 | none |

No run carries a validity flag, so there is no breakdown by cell and by flag to print.

200 of 200 runs DONE, 200 of them valid and unflagged. Lines 4 and 5 below are counted on the valid and unflagged runs only: a flagged run is one evalvalid says did not run as specified, and counting its decisions among the rest would put rows from a run nobody can read into the share per arm. Leaving a run out also lowers N, which widens the tolerance on the share per arm, so a cell read on fewer runs is checked less tightly than spec.md's 100 runs would check it.

What the runs themselves record about how they were made. These are spec.md's own settings, and a run recording anything else is not a run of this step: such a row stops this script before a single measured number is printed, because every count below would then be counted on a stage nobody specified. Fewer runs than the cell asks for does not stop anything and reads `incomplete` instead.

| p | runs | semindex commit | dirty runs | workload | knob | arms | seeds | node logs per run |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 100 | `a8add2cff4275317ddceea0a6068e5a1dd18e3da` | 0 | `w0` | True | `3.695;6.0;10.043` | 1 to 100 | 5 |
| 0.4 | 100 | `a8add2cff4275317ddceea0a6068e5a1dd18e3da` | 0 | `w0` | True | `3.695;6.0;10.043` | 1 to 100 | 5 |

Every run must have written 5 node logs, one per host: a run with fewer logged its decisions somewhere this analysis did not read, and its decision counts would be short with nothing else saying so. Every run has 5.

The two smoke runs of `s11smoke` seed 101, reported separately and never counted among the 200. spec.md gates the sweep on them, and seed 101 lies outside both this step's seeds and evidence/20's held-out seeds, so neither set is spent on it.

| p | seed | valid | flags | knob | arms | semindex commit |
|---|---|---|---|---|---|---|
| 0.2 | 101 | True | none | True | `[3.695, 6.0, 10.043]` | `a8add2cff4275317ddceea0a6068e5a1dd18e3da` |
| 0.4 | 101 | True | none | True | `[3.695, 6.0, 10.043]` | `a8add2cff4275317ddceea0a6068e5a1dd18e3da` |

Verdict on 200 runs: **pass**.

## Line 4: the decision log is material a learner can read

Counted over each cell's valid runs, over the whole run rather than over the measurement window: spec.md "Quantities" gives the window to the delay and the polls per run, and says that what this line asks is what a run decided. A send row is one `state_poll` decision, a reschedule row one `state_poll_reschedule`.

### Decision rows per run

| p | runs | send rows median (min..max) | predicted | reschedule rows median (min..max) | predicted |
|---|---|---|---|---|---|
| 0.2 | 100 | 160.5 (149..173) | 160.0 (148..176) | 160.0 (148..173) | 160.0 (148..176) |
| 0.4 | 100 | 160.0 (149..173) | 161.0 (146..174) | 160.0 (148..173) | 161.0 (146..174) |

The predicted columns are predict.txt's and stand there for information: spec.md attaches no pass or fail to the decision counts. The replay resolves every poll it sends and writes rows past the instant the testbed's agent stops, so the measured reschedule counts are expected to sit slightly below the predicted ones.

### The share per arm, which is the check that the uniform draw is uniform

N is the cell's reschedule rows and the tolerance is `1/3 +- 3*sqrt((2/9)/N)` read at that measured N, which is the formula spec.md fixed before the run. The pass or fail word of this line is whether all three shares lie inside it, per cell.

| p | N | tolerance | arm | rows | share | distance from 1/3 | inside | predicted share |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 16025 | 0.01117 | 3.695 | 5265 | 0.32855 | -0.00478 | yes | 0.329 |
| 0.2 | 16025 | 0.01117 | 6.0 | 5430 | 0.33885 | +0.00551 | yes | 0.337 |
| 0.2 | 16025 | 0.01117 | 10.043 | 5330 | 0.33261 | -0.00073 | yes | 0.333 |
| 0.4 | 16006 | 0.01118 | 3.695 | 5301 | 0.33119 | -0.00215 | yes | 0.331 |
| 0.4 | 16006 | 0.01118 | 6.0 | 5375 | 0.33581 | +0.00248 | yes | 0.336 |
| 0.4 | 16006 | 0.01118 | 10.043 | 5330 | 0.33300 | -0.00033 | yes | 0.333 |

### The resolutions, and the rows a learner would read them off

| p | timeout share | predicted | clamped share | predicted | truncated send rows | truncated reschedule rows | reschedule rows per send row | predicted |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.1888 | 0.191 | 0.0000 | 0.000 | 0 | 0 | 0.9964 | 1.000 |
| 0.4 | 0.3646 | 0.368 | 0.0000 | 0.000 | 0 | 0 | 0.9950 | 1.000 |

A truncated row is one whose state carried fewer origins than the run tracks, so a learner reading it cannot see every origin. With two origins none is expected; a nonzero count means the rows are missing origins. The count is reported and carries no pass or fail word of its own, which this line's word reads off the three shares alone.

The reschedule rows per send row are predicted as 1.000 because the replay resolves every poll it sends. It also writes rows past t0+130 s, where the testbed's agent has already left its loop: predict.txt puts those at 0.004 of the reschedule rows at p 0.2 and 0.006 at p 0.4, and that is the size of the gap below 1.000 to expect here. spec.md records that the same surplus biases the predicted reschedule count and the predicted timeout share upward.

The clamped share is predicted as zero for a structural reason: the shortest applied interval is 3.695 x 0.9 = 3.3255 s measured from the send, and a resolution lands at most 2.1 s after the send, so the re-decided instant is always still ahead. A nonzero clamped share here is therefore evidence that an agent's loop stalled for over a second, not evidence about the arms.

### Also reported, with no pass or fail attached

| p | polls checked against the schedule | of those outside the 0.5 s | answered reschedule rows whose `empty` is null | `state_poll_late_reply` rows | reschedule rows that are not a uniform draw |
|---|---|---|---|---|---|
| 0.2 | 15283 | 0 | 0 | 0 | 0 |
| 0.4 | 15287 | 0 | 0 | 0 | 0 |

The first two columns are the smoke gate's seventh check run over every run of the sweep, with its own constants: for a poll re-decided without a clamp whose neighbour was polled again, whether that next poll left inside `[applied - 0.002 s, applied + 0.5 s]` of the re-decided instant. A poll outside it is a poll whose next send does not match the re-decided instant, which is either a re-decision that was logged and not applied or an agent loop that stalled, and this count alone cannot tell the two apart. Unlike the smoke gate, which stops at the first late successor of a (node, neighbour) pair, every late successor is counted here, so the number is a count of polls and not of pairs and cannot be compared with a smoke gate count directly. spec.md attaches no pass or fail to it.

A null `empty` on an answered row is a reply the node could not decode. A row carrying no `empty` field at all is counted in the same column and cannot be told apart from an explicit null here. The replay cannot produce one, so this count has nothing in the prediction to be held against; evidence/20 found none in 11881 answered re-decisions on its two rule stages. A `state_poll_late_reply` is a reply that arrived after its poll had already timed out, which the replay cannot produce either.

The last column is an addition of this analysis and has no Acceptance item behind it: reschedule rows whose `action_prob` is not exactly 1/3 or whose `interval_s` is not one of the three arms. Expected zero. A nonzero count means the share per arm above was counted on rows that are not the uniform draw this step ran, and nothing else on this line would say so. It is counted once per row, so it stays a row count when a row is wrong in both ways at once.

evidence/09's record-set sanity counters (`hop_gap_outside_window`, `reply_before_any_poll` and the rest) are printed on perrun.py's stderr only and are not read back here, so this file cannot show that the record set the delay above was counted on is the one the prediction scored. That has to be read off perrun.py's own output.

Verdict, which reads the share per arm alone: **pass** (p 0.2 pass, p 0.4 pass).

## Line 5: did the replay predict the measurement?

The headline delay and the polls per run at both p, four values, each against the plus or minus 6 % band of predict.txt. The headline delay is the pooled `sum(lat4)/sum(n4)` over the cell's runs, over the records created at or before t0+72 s; under `w0` that is every record the run holds. The polls per run is the mean over the cell's runs of evalagg's in-window count over the 8 directed pairs; a reschedule row is not a poll and is never counted as one. All 4 inside reads `predicted`, 1 or 2 outside `mixed`, 3 or more outside `missed`.

The band ends below are the digits predict.txt itself prints. They part from the printed median times 1 -+ 6 % by at most 0.008 (the upper end of the polls per run at p 0.4), which is under 0.006 % of that value, because that file forms the band from the unrounded median; inside or outside is decided on the printed ends, and a value on which the two would disagree is named below the table.

| p | quantity | runs | measured | predicted | band | inside or outside | deviation % |
|---|---|---|---|---|---|---|---|
| 0.2 | headline delay s | 100 | 24.320 | 24.225 | 22.771 .. 25.678 | inside | +0.39 |
| 0.2 | polls per run | 100 | 145.58 | 145.51 | 136.78 .. 154.24 | inside | +0.05 |
| 0.4 | headline delay s | 100 | 34.738 | 34.930 | 32.835 .. 37.026 | inside | -0.55 |
| 0.4 | polls per run | 100 | 145.81 | 145.70 | 136.96 .. 154.45 | inside | +0.08 |

4 of 4 values inside their band, 0 outside, 0 not measured. Verdict: **predicted**.

All four values lie inside the band spec.md fixed before the runs. That is a statement about the replay on these four values and about no decision maker: this stage draws uniformly and runs no policy.

Values lying between the printed band ends and the printed median times 1 -+ 6 %, which would read the other way on the second: none, so the choice between the two decides nothing here.

Every value outside its band, with how far outside and in which direction. spec.md asks for this whatever the count is, so the list is printed even when it is empty.

- no value fell outside its band.

The censored share beside each delay, which spec.md asks for. A record is censored when it was not stored 4 hops away inside the window, and the delay is the mean over the uncensored ones, so a censored record is left out of the delay instead of lengthening it. The headline share is the one that belongs to the delay above; the all-record share stands beside it, and under `w0` the two select the same records and are expected to be equal.

| p | headline censored share | all-record censored share | predicted |
|---|---|---|---|
| 0.2 | 0.00100 | 0.00100 | 0.000 |
| 0.4 | 0.01933 | 0.01933 | 0.021 |

## What these numbers do not say

- This stage's delay is not a quantity any policy is compared on. A uniform draw over three arms is not a policy; the prediction exists only to check that the replay and the testbed ran the same thing, and pass line 5 is that check and nothing more.

- The 6 % band has no measurement behind it for a uniform draw. It is the largest per-stage deviation evidence/13 measured on the default load, 5.9 %, on four fixed arms and two policies, carried over here. Nothing says it is the right band for a uniformly drawn interval or for the re-decision branch, which is why every value outside it is reported with its size and its direction whatever the count is.

- The re-decision branch of the replay has no measurement behind it anywhere else. Until this line is read, the replay cannot be said to reproduce this stage at all, and on `missed` that branch is the first suspect.

- No stage here is run from two source versions, so nothing in this step shows on the testbed that the knob-off path is unchanged since evidence/20's commit. That rests on the merged change's own tests.

- Nothing here says which decision maker is better, or whether a policy can be estimated off these rows. This stage draws uniformly and runs no policy, and this step stops at leaving the data.

- Nothing about p 0.5, p 0.6, p 0, the no-outage cell, the on/off loads, or finer arms.

- Every number in the prediction is simulated. A predicted band says what the replay expects the testbed to measure and is itself no measurement.

