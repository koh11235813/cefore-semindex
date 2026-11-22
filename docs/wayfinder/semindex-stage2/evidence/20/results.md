# evidence/20 results

Computed by ana.py from perrun.csv here, the frozen predict.txt, main10.sh and fpin.txt of this directory, and the stored runs' meta.json and event logs. The pass lines, the bands, the predictions and the verdict words are spec.md's; the words are read off the numbers mechanically and nothing here adds to them. Every line names the number of seeds it used, and a line computed on fewer than 40 seeds gets the word `incomplete` in place of its verdict.

One cell: p 0.2, workload w0, the s6 outage shape, seeds 121 to 160, 200 runs over five stages. There is no gate stage: spec.md records `git diff --stat 03f5ecc d009cfb -- src tests` and its empty output as the evidence that the code did not move since evidence/19. The semindex commit and the dirty flag each run recorded for itself are printed per stage under line 1, so that argument can be read off the runs here.

Two scorings, as spec.md fixes them. The headline scoring takes the records whose scheduled creation time is at or before t0+72.0 s, selected by the experiment description's `schedule[host][seq-1]["at_s"] <= 72.0` read against discovery.csv's `(origin, seq)`, and its delay is the mean over the uncensored ones 4 hops away. The all-record scoring takes every record of the run, which is the quantity every earlier step reported. Polls per run is counted over the whole window [t0+10 s, t0+130 s] under both scorings. spec.md says the two scorings select the same 240 records under `w0`; the headline scoring carries each verdict word, the all-record value stands beside it, and whether the two are equal is a measured column below and not an assumption here.

| arm | stage | setting |
|---|---|---|
| B3 | `s10b3p695` | fixed 3.695 s, jitter 0.1 |
| B6 | `s10b6p0` | fixed 6.0 s, jitter 0.1 |
| B10 | `s10b10p043` | fixed 10.043 s, jitter 0.1 |
| T37r | `s10t37r` | rule T37r: i_min 3.695, factor 2, i_max 10.043, reset |
| F | `s10f` | the pinned F: i_min 3.695, factor 1.5, i_max 10.043, reset |

The predicted constants carried in ana.py were checked against predict.txt at run time: 10 rows of predicted numbers, every number in them equal to the digits that file prints. A mismatch stops the script, so the analysis cannot drift from the frozen prediction.

## Line 1: all 200 runs DONE and unflagged

Validity is `<run_dir>/meta.json` -> `validity` (`valid`, `flags`), written by semindex.evalrun from semindex.evalvalid. The flag names carry the node and the sequence number (`record_off_schedule:h0:seq5`), so the breakdown below counts flags by the first word of the name and lists the names seen. It is printed whether or not any flag appeared.

| stage | runs expected | runs DONE | valid and unflagged | flagged | flag names seen |
|---|---|---|---|---|---|
| `s10b3p695` | 40 | 40 | 40 | 0 | none |
| `s10b6p0` | 40 | 40 | 40 | 0 | none |
| `s10b10p043` | 40 | 40 | 40 | 0 | none |
| `s10t37r` | 40 | 40 | 40 | 0 | none |
| `s10f` | 40 | 40 | 40 | 0 | none |

No run carries a validity flag, so there is no breakdown by stage and by flag to print.

200 of 200 runs DONE, 200 of them valid and unflagged.

Seeds every other line is read on, the intersection of the DONE and unflagged seeds over the five stages: 40 of 40, 121 to 160.

Every `interval_s` in every decision log against the set its stage's setting allows. The decision log is every `decision` event of kind `state_poll` or `state_poll_reschedule` on every host, which is the whole log of the arm's or the rule's own arithmetic. The allowed set is built here from what main10.sh launches: the `--interval-choices` value for a fixed arm, and for a rule the walk `i_min_s`, `min(interval * factor, i_max_s)` repeated until it stops moving, which is the only way `engine._advance_rule_interval` ever changes the interval. The products are computed and not written down, so the comparison is on exact IEEE doubles; F's second value is 5.5424999999999995 and no rounded literal equals it. This check reads every DONE run, including a flagged one. A stage whose decision logs yielded no row at all reads `no` here: the empty set is a subset of every allowed set, so a subset test on its own would pass a stage whose logs were never read and hide that the check measured nothing.

| stage | runs read | allowed values | values seen (count) | every value allowed |
|---|---|---|---|---|
| `s10b3p695` | 40 | 3.695 | 3.695 (11219) | yes |
| `s10b6p0` | 40 | 6.0 | 6.0 (7033) | yes |
| `s10b10p043` | 40 | 10.043 | 10.043 (4231) | yes |
| `s10t37r` | 40 | 3.695, 7.39, 10.043 | 3.695 (8772), 7.39 (2447), 10.043 (3277) | yes |
| `s10f` | 40 | 3.695, 5.5424999999999995, 8.313749999999999, 10.043 | 3.695 (9504), 5.5424999999999995 (2496), 8.313749999999999 (1882), 10.043 (1862) | yes |

Which seeds and which workload the rows are, and which semindex commit the runs record. The seeds must be exactly 121 to 160 in every stage and every row's `workload` must be `w0`, because every other line here reads perrun.csv for whatever rows it holds and would otherwise score a different cell without saying so. The commit and the dirty flag come from each run's own `meta.json`, and they are what makes spec.md's "no gate stage" argument checkable here: that argument says the code did not move since evidence/19, and one clean commit across all five stages is the run-by-run form of it. A stage with more than one commit, or with any run made from a dirty tree, fails this line.

| stage | seeds DONE | seeds are 121 to 160 | workloads seen | workload is `w0` | semindex commit | runs from a dirty tree | one clean commit |
|---|---|---|---|---|---|---|---|
| `s10b3p695` | 40 | yes | `w0` | yes | `d009cfb270909beda1f277943f99c93acb9e96e2` | 0 | yes |
| `s10b6p0` | 40 | yes | `w0` | yes | `d009cfb270909beda1f277943f99c93acb9e96e2` | 0 | yes |
| `s10b10p043` | 40 | yes | `w0` | yes | `d009cfb270909beda1f277943f99c93acb9e96e2` | 0 | yes |
| `s10t37r` | 40 | yes | `w0` | yes | `d009cfb270909beda1f277943f99c93acb9e96e2` | 0 | yes |
| `s10f` | 40 | yes | `w0` | yes | `d009cfb270909beda1f277943f99c93acb9e96e2` | 0 | yes |

The pinned F's five values, restated against main10.sh's launched `FPIN` string and fpin.txt's winner line, both read from those files at run time. spec.md's "The pinned F" table is the third column. All three must agree, and `i_min_s` must be at or above the floor 3.695 s that spec.md and the ticket both set.

| field | main10.sh FPIN | fpin.txt winner line | spec.md's table | agree |
|---|---|---|---|---|
| `i_min_s` | 3.695 | 3.695 | 3.695 | yes |
| `factor` | 1.5 | 1.5 | 1.5 | yes |
| `i_max_s` | 10.043 | 10.043 | 10.043 | yes |
| `jitter_frac` | 0.1 | 0.1 | 0.1 | yes |
| `on_timeout` | 'reset' | 'reset' | 'reset' | yes |

`i_min_s` at or above the floor 3.695 s: yes. Keys in the launched string beyond the five above: none. spec.md names `k_cap` as a key the family has no member with, and the launched string carries no such key.

spec.md also reports that both validators on the `--poll-rule` path accept this string and T37r, with the bounds they enforce. That check was made by calling them when the member was pinned and its result stands in spec.md; it is not re-run here, and this script imports no semindex module.

Verdict on 200 runs and 40 seeds: **pass**.

## Line 2: gain of T37r and of the pinned F against the three-arm fixed frontier

The line is the three fixed arms of the same cell (3.695, 6.0, 10.043 s), polls per run on the horizontal axis and delay on the vertical, joined piecewise in log-log. Gain is `1 - delay(rule) / the line's delay at the rule's own polls per run`. The arms must form a falling line, which here means delay falls strictly as polls per run rise; a resample whose arms are not monotone, and a resample whose rule polls fall outside the outer arms, is counted and left out rather than replaced. If the full sample's arms do not fall, or a rule's own polls fall outside them, the verdict slot reads `no line`.

The reading is attached to each rule separately and the two are not required to agree.

### Measured values, 40 of 40 seeds

| arm | stage | scoring | delay s | censored share | polls per run |
|---|---|---|---|---|---|
| B3 | `s10b3p695` | headline | 13.783 | 0.00000 | 256.48 |
| B3 | `s10b3p695` | all records | 13.783 | 0.00000 | 256.48 |
| B6 | `s10b6p0` | headline | 20.335 | 0.00000 | 159.82 |
| B6 | `s10b6p0` | all records | 20.335 | 0.00000 | 159.82 |
| B10 | `s10b10p043` | headline | 30.140 | 0.00458 | 96.42 |
| B10 | `s10b10p043` | all records | 30.140 | 0.00458 | 96.42 |
| T37r | `s10t37r` | headline | 18.343 | 0.00000 | 165.47 |
| T37r | `s10t37r` | all records | 18.343 | 0.00000 | 165.47 |
| F | `s10f` | headline | 17.460 | 0.00000 | 180.97 |
| F | `s10f` | all records | 17.460 | 0.00000 | 180.97 |

Where the censored 4-hop records are, seed by seed. A record is censored when it was not stored 4 hops away inside the measurement window, and the delay is the mean over the uncensored ones, so a censored record is left out of the delay instead of lengthening it. The replay predicted a share for each stage and those stand in the last column.

| stage | scoring | seeds carrying a censored record | censored records | records | measured share | replay's predicted share |
|---|---|---|---|---|---|---|
| `s10b3p695` | headline | none | 0 | 9600 | 0.00000 | 0.000 |
| `s10b3p695` | all records | none | 0 | 9600 | 0.00000 | 0.000 |
| `s10b6p0` | headline | none | 0 | 9600 | 0.00000 | 0.000 |
| `s10b6p0` | all records | none | 0 | 9600 | 0.00000 | 0.000 |
| `s10b10p043` | headline | 135 (40); 149 (4) | 44 | 9600 | 0.00458 | 0.000 |
| `s10b10p043` | all records | 135 (40); 149 (4) | 44 | 9600 | 0.00458 | 0.000 |
| `s10t37r` | headline | none | 0 | 9600 | 0.00000 | 0.000 |
| `s10t37r` | all records | none | 0 | 9600 | 0.00000 | 0.000 |
| `s10f` | headline | none | 0 | 9600 | 0.00000 | 0.000 |
| `s10f` | all records | none | 0 | 9600 | 0.00000 | 0.000 |

spec.md says that under `w0` pooling over records and averaging over seeds coincide. That holds while every seed contributes the same number of uncensored records, and censoring makes the counts unequal. Where they are unequal, the pooled delay of that stage weights its seeds by how many records each one delivered, so the sentence is not exact for it:

- `s10b10p043`: most seeds contribute 240 uncensored 4-hop records, seed 135 contributes 200, seed 149 contributes 236.

Which segment of the frontier each rule's gain is read off. The frontier is two segments joined at the middle arm, and a gain is read off the one segment that contains the rule's own polls per run, so an arm outside that segment does not enter the gain. The count is over the same 2000 resamples the intervals are drawn from, headline scoring.

| rule | segment in the full sample | segments over the resamples |
|---|---|---|
| T37r | 6.0 s to 3.695 s arms | 6.0 s to 3.695 s arms 2000 of 2000 |
| F | 6.0 s to 3.695 s arms | 6.0 s to 3.695 s arms 2000 of 2000 |

Consequence: the 10.043 s arm enters no gain at all, in the full sample or in any resample, because no segment either rule's gain was read off touches it. That arm is the only one carrying censored records, so the censoring above reaches no gain on this cell. The gain called a three-arm gain is therefore in effect a gain against the line between two arms, 6.0 s and 3.695 s.

Arm points of the full sample, headline scoring, sorted by polls per run: (96.42 polls, 30.140 s), (159.82 polls, 20.335 s), (256.48 polls, 13.783 s). The three arms form a falling line: yes.

Two rules for leaving a resample out, both computed from the one index matrix. The `joint` columns are predict.py's `boot_gains`: a resample in which either rule's gain is undefined is left out of both gains and of their difference. That is the rule the frozen prediction's own intervals were drawn under, so **the joint columns carry the verdict** and an interval drawn the other way would not be the statistic predict.txt prints. The `per quantity` columns leave a resample out of each quantity it leaves undefined and no other, which is evidence/19's rule; they are printed beside the joint ones, never instead of them.

| rule | scoring | seeds | gain % | 95 % interval % (joint) | half width pp (joint) | verdict | why no gain | predicted gain % and interval | predicted half width pp | predicted word | left out (joint) | 95 % interval % (per quantity) | half width pp (per quantity) | left out (per quantity) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T37r | headline | 40 | +7.18 | [+1.77, +12.51] | 5.37 | **seen** | - | +7.17 [+2.28, +12.00] | 5.01 | seen | 0 | [+1.77, +12.51] | 5.37 | 0 |
| F | headline | 40 | +4.90 | [+0.55, +9.21] | 4.33 | **seen** | - | +7.80 [+3.95, +11.76] | 3.92 | seen | 0 | [+0.55, +9.21] | 4.33 | 0 |

F minus T37r on the same frontier, headline scoring, on 40 seeds: -2.28 pp [-6.10, +1.50], half width 3.80 pp, reading **not seen**, resamples left out 0, all under the joint drop rule. Under the per quantity rule the same difference reads [-6.10, +1.50], half width 3.80 pp, resamples left out 0. The replay predicted +0.72 pp [-3.91, +4.83], half width 4.52 pp, with 0 of 40 streams entirely above zero. spec.md attaches no pass line to this difference.


Arm points of the full sample, all records scoring, sorted by polls per run: (96.42 polls, 30.140 s), (159.82 polls, 20.335 s), (256.48 polls, 13.783 s). The three arms form a falling line: yes.

Two rules for leaving a resample out, both computed from the one index matrix. The `joint` columns are predict.py's `boot_gains`: a resample in which either rule's gain is undefined is left out of both gains and of their difference. That is the rule the frozen prediction's own intervals were drawn under, so **the joint columns carry the verdict** and an interval drawn the other way would not be the statistic predict.txt prints. The `per quantity` columns leave a resample out of each quantity it leaves undefined and no other, which is evidence/19's rule; they are printed beside the joint ones, never instead of them.

| rule | scoring | seeds | gain % | 95 % interval % (joint) | half width pp (joint) | verdict | why no gain | predicted gain % and interval | predicted half width pp | predicted word | left out (joint) | 95 % interval % (per quantity) | half width pp (per quantity) | left out (per quantity) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T37r | all records | 40 | +7.18 | [+1.77, +12.51] | 5.37 | **seen** | - | +7.17 [+2.28, +12.00] | 5.01 | seen | 0 | [+1.77, +12.51] | 5.37 | 0 |
| F | all records | 40 | +4.90 | [+0.55, +9.21] | 4.33 | **seen** | - | +7.80 [+3.95, +11.76] | 3.92 | seen | 0 | [+0.55, +9.21] | 4.33 | 0 |

F minus T37r on the same frontier, all records scoring, on 40 seeds: -2.28 pp [-6.10, +1.50], half width 3.80 pp, reading **not seen**, resamples left out 0, all under the joint drop rule. Under the per quantity rule the same difference reads [-6.10, +1.50], half width 3.80 pp, resamples left out 0. The replay predicted +0.72 pp [-3.91, +4.83], half width 4.52 pp, with 0 of 40 streams entirely above zero. spec.md attaches no pass line to this difference.

The verdict of a rule is the headline scoring's; the all-record word stands beside it in the tables above. What spec.md says about each reading:

| rule | seeds | headline verdict | all-record verdict | predicted word | spec.md's text for the headline reading |
|---|---|---|---|---|---|
| T37r | 40 | **seen** | seen | seen | spec.md: seen is the reading when the whole interval of the gain lies above zero, and seen is the prediction for both rules. |
| F | 40 | **seen** | seen | seen | spec.md: seen is the reading when the whole interval of the gain lies above zero, and seen is the prediction for both rules. |

The pinned F's row only: spec.md's prediction for F is a three-arm gain on seeds F has never seen. fpin.txt's +11.1 is a different quantity, a seven-arm gain in sample on the 20 seeds that chose the member, and this row is not read against it.

Both rules read the word spec.md predicted, so neither sends the step to investigate a reason.

## Line 3: the pinned F against T37r directly

The delay ratio and the polls ratio, from the same resamples, printed as the percent change of F against T37r so that they sit beside the replay's numbers. The censored share of each rule stands beside the delay below. spec.md attaches no pass or fail to this line: the two rules differ by one coordinate (factor 1.5 against 2), so this is the measurement of a size and not a test of a direction. The reading is whether each ratio spans zero. The reading is the joint drop rule's, as pass line 2's is, so that the two lines rest on one resample set; the per quantity interval stands beside it.

| quantity | scoring | seeds | measured % | 95 % interval % (joint) | measured half width pp (joint) | predicted % and interval | predicted half width pp | spans zero | 95 % interval % (per quantity) | half width pp (per quantity) |
|---|---|---|---|---|---|---|---|---|---|---|
| delay | headline | 40 | -4.82 | [-8.75, -0.53] | 4.11 | -7.91 [-12.21, -3.32] | 4.54 | no | [-8.75, -0.53] | 4.11 |
| delay | all records | 40 | -4.82 | [-8.75, -0.53] | 4.11 | -7.91 [-12.21, -3.32] | 4.54 | no | [-8.75, -0.53] | 4.11 |
| polls | headline | 40 | +9.37 | [+8.39, +10.31] | 0.96 | +9.81 [+8.95, +10.58] | 0.80 | no | [+8.39, +10.31] | 0.96 |
| polls | all records | 40 | +9.37 | [+8.39, +10.31] | 0.96 | +9.81 [+8.95, +10.58] | 0.80 | no | [+8.39, +10.31] | 0.96 |

predict.py's `boot_ratios` leaves no resample out of a ratio, so neither drop rule drops one here either and the two pairs of columns above are expected to be equal number for number. Whether they are is a computed comparison in the resample table near the end of this file, not an assumption here.

The polls ratio is the same number under both scorings, because polls are counted over the whole window whichever records the delay is taken on. predict.txt prints it once for that reason and the same predicted numbers stand under both rows here.

The censored share of each rule, beside the delays the ratio is built from:

| stage | scoring | delay s | censored share |
|---|---|---|---|
| `s10t37r` | headline | 18.343 | 0.00000 |
| `s10t37r` | all records | 18.343 | 0.00000 |
| `s10f` | headline | 17.460 | 0.00000 |
| `s10f` | all records | 17.460 | 0.00000 |

## Line 4: did the replay predict the measurement?

The delay and the polls per run over the whole window of all five stages, ten values, each against the plus or minus 6 % band of predict.txt, copied from that file as printed and checked against it at run time. 9 or more of the 10 inside reads `predicted`, exactly 8 reads `mixed`, 7 or fewer reads `missed`. spec.md records that the band comes from the largest deviation the replay showed on the default load, 5.9 % (evidence/13), measured on four arms and two policies that do not include the 6.0 s arm, T37r or F, so every value outside it is reported with how far outside and in which direction whatever the count is. The band is a Mode B band on per-stage aggregates and says nothing about agreement decision by decision.

| stage | quantity | informative | measured | prediction | band | inside or outside | deviation % |
|---|---|---|---|---|---|---|---|
| `s10b3p695` | delay s | yes | 13.783 | 13.908 | 13.074 .. 14.742 | inside | -0.90 |
| `s10b3p695` | polls per run | no | 256.48 | 256.68 | 241.27 .. 272.08 | inside | -0.08 |
| `s10b6p0` | delay s | yes | 20.335 | 20.386 | 19.162 .. 21.609 | inside | -0.25 |
| `s10b6p0` | polls per run | no | 159.82 | 159.75 | 150.16 .. 169.34 | inside | +0.05 |
| `s10b10p043` | delay s | yes | 30.140 | 30.437 | 28.611 .. 32.264 | inside | -0.97 |
| `s10b10p043` | polls per run | no | 96.42 | 96.08 | 90.31 .. 101.84 | inside | +0.36 |
| `s10t37r` | delay s | yes | 18.343 | 18.517 | 17.406 .. 19.628 | inside | -0.94 |
| `s10t37r` | polls per run | yes | 165.47 | 165.01 | 155.11 .. 174.91 | inside | +0.28 |
| `s10f` | delay s | yes | 17.460 | 17.047 | 16.024 .. 18.070 | inside | +2.42 |
| `s10f` | polls per run | yes | 180.97 | 180.95 | 170.09 .. 191.81 | inside | +0.01 |

10 of 10 values inside their band. Verdict on 40 seeds: **predicted**.

spec.md: "The 6.0 s arm, T37, T37r and F have no testbed number in this cell at all. Pass line 4 reads this." T37 is not a stage of this step, so the testbed numbers this line reads cover the 6.0 s arm, T37r and F, and T37 still has none in this cell.

spec.md names three of the ten values near-tautological: a fixed arm's poll count follows from its interval and the window length, so the three fixed arms' polls are expected inside the band whatever the replay does. The seven values that can inform are the five delays and the two rules' poll counts, and 7 of 7 of them are inside their band. The verdict word above is spec.md's, read on the ten; this breakdown is reported beside it and carries no word of its own.

Every value outside its band, one by one, with how far outside and in which direction. spec.md asks for this whatever the count is, so the table is printed even when it is empty. A value outside the band is also read against the numbers that bound the substitution: on seeds 1 to 20 the generated schedules moved the delay by at most 0.3 % and a borrowed clock by at most 1.1 % (predict.txt), so a miss much larger than their sum is the replay's and not the borrowing's.

- no value fell outside its band.

The censored share beside every delay, which spec.md asks for on all five stages, is in line 2's measured table. The comparison of decision sequences that spec.md asks for when this line reads `mixed` or `missed` is not computed here: it is a reading of two logs against each other, which this script does not do. The word this line read is printed above, so the reader can see whether spec.md asks for that comparison at all.

## Also reported, without a pass line

Every stage's measured values beside the replay's, both scorings together, and whether the two scorings came out equal. spec.md says they must under `w0` because every record is due at or before t0+71 s; the last two columns are the measurement of that.

| stage | seeds | headline delay s | headline censored | all-record delay s | all-record censored | polls per run | replay delay s | replay polls | replay censored | headline equals all records |
|---|---|---|---|---|---|---|---|---|---|---|
| `s10b3p695` | 40 | 13.783 | 0.00000 | 13.783 | 0.00000 | 256.48 | 13.908 | 256.68 | 0.000 | yes |
| `s10b6p0` | 40 | 20.335 | 0.00000 | 20.335 | 0.00000 | 159.82 | 20.386 | 159.75 | 0.000 | yes |
| `s10b10p043` | 40 | 30.140 | 0.00458 | 30.140 | 0.00458 | 96.42 | 30.437 | 96.08 | 0.000 | yes |
| `s10t37r` | 40 | 18.343 | 0.00000 | 18.343 | 0.00000 | 165.47 | 18.517 | 165.01 | 0.000 | yes |
| `s10f` | 40 | 17.460 | 0.00000 | 17.460 | 0.00000 | 180.97 | 17.047 | 180.95 | 0.000 | yes |

Three counters from perrun.csv, over every DONE run of the five stages. `sched_regen_diff` counts a run's record origins whose instants the committed generator does not draw again for that run's own (host, seed, k, workload); no run of these seeds existed when predict.py drew the prediction, so predict.py built the record schedule from that same generator, and this counter is what says run by run that the testbed ran the instants the prediction scored. The two `missing_disc` counters count records the schedule names that never appeared in the aggregation; a nonzero count would deflate a run's sums silently. All three are expected to be zero.

| counter | total over all DONE runs | runs with a nonzero count |
|---|---|---|
| `sched_regen_diff` | 0 | 0 of 200 |
| `head_missing_disc` | 0 | 0 of 200 |
| `all_missing_disc` | 0 | 0 of 200 |

Records the schedule lays down per run, from the schedule each run was handed. predict.txt puts the held-out schedules at 240.0 [240 .. 240] records per run and the same for the headline cut, so the headline cut taking fewer than all of them would be visible here. All five stages of one seed are handed the same schedule, so one row per seed is the whole of it, and the `stages agree` column says that every stage of that seed carried the same two counts.

| seed | all records | headline records | stages agree | instants are the generator's draw |
|---|---|---|---|---|
| 121 | 240 | 240 | yes | yes |
| 122 | 240 | 240 | yes | yes |
| 123 | 240 | 240 | yes | yes |
| 124 | 240 | 240 | yes | yes |
| 125 | 240 | 240 | yes | yes |
| 126 | 240 | 240 | yes | yes |
| 127 | 240 | 240 | yes | yes |
| 128 | 240 | 240 | yes | yes |
| 129 | 240 | 240 | yes | yes |
| 130 | 240 | 240 | yes | yes |
| 131 | 240 | 240 | yes | yes |
| 132 | 240 | 240 | yes | yes |
| 133 | 240 | 240 | yes | yes |
| 134 | 240 | 240 | yes | yes |
| 135 | 240 | 240 | yes | yes |
| 136 | 240 | 240 | yes | yes |
| 137 | 240 | 240 | yes | yes |
| 138 | 240 | 240 | yes | yes |
| 139 | 240 | 240 | yes | yes |
| 140 | 240 | 240 | yes | yes |
| 141 | 240 | 240 | yes | yes |
| 142 | 240 | 240 | yes | yes |
| 143 | 240 | 240 | yes | yes |
| 144 | 240 | 240 | yes | yes |
| 145 | 240 | 240 | yes | yes |
| 146 | 240 | 240 | yes | yes |
| 147 | 240 | 240 | yes | yes |
| 148 | 240 | 240 | yes | yes |
| 149 | 240 | 240 | yes | yes |
| 150 | 240 | 240 | yes | yes |
| 151 | 240 | 240 | yes | yes |
| 152 | 240 | 240 | yes | yes |
| 153 | 240 | 240 | yes | yes |
| 154 | 240 | 240 | yes | yes |
| 155 | 240 | 240 | yes | yes |
| 156 | 240 | 240 | yes | yes |
| 157 | 240 | 240 | yes | yes |
| 158 | 240 | 240 | yes | yes |
| 159 | 240 | 240 | yes | yes |
| 160 | 240 | 240 | yes | yes |

Whole-run counts over every decision row of all five stages, not cut to the measurement window, because what they describe is the rule's own arithmetic over every poll the run sent. The three fixed arms have no re-decision at all: a choices arm picks its interval once per poll and never re-decides, so their rows are all zero past the poll count and that is what says the arms were run as fixed arms. An answered re-decision carries `empty`: true when the reply held nothing new, false when it held something new, null when the reply could not be decoded. spec.md's "One difference between the replay and the testbed" is about exactly that null case, so its count is printed whether or not it is zero. A timed-out re-decision carries `empty` null by construction and is not counted among the answered ones. These counters are read over every DONE run, as the interval check above is.

| stage | runs | polls | re-decisions | answered | timed out | empty true share of answered | empty false | empty null and answered | clamped | late replies |
|---|---|---|---|---|---|---|---|---|---|---|
| `s10b3p695` | 40 | 11219 | 0 | 0 | 0 | - | 0 | 0 | 0 | 0 |
| `s10b6p0` | 40 | 7033 | 0 | 0 | 0 | - | 0 | 0 | 0 | 0 |
| `s10b10p043` | 40 | 4231 | 0 | 0 | 0 | - | 0 | 0 | 0 | 0 |
| `s10t37r` | 40 | 7259 | 7237 | 5690 | 1547 | 0.5276 | 2688 | 0 | 0 | 0 |
| `s10f` | 40 | 7879 | 7865 | 6191 | 1674 | 0.5271 | 2928 | 0 | 0 | 0 |

## How the bootstrap was drawn

One index matrix, `numpy.random.default_rng(20261122).integers(0, n, size=(2000, n))`, drawn once at the seed count `n` the intersection has and reused for every quantity and both scorings. That is how predict.py drew its own matrix, so the measured bounds and the predicted bounds rest on the same resampled positions. The unit is the seed number: one index row selects the seeds, and all five stages are recomputed from that one selection, so a resample holds the same seeds in every stage at once and the stages stay paired. A seed set short of a seed is drawn at its own size from a generator with the same seed, which is why a run on fewer than 40 seeds gives bounds that are not the bounds predict.txt was drawn against; every size drawn is listed here.

| index matrix | seeds per resample | resamples |
|---|---|---|
| 1 | 40 | 2000 |

Resamples left out of every interval, under both drop rules, with the reason each one was left out. A gain is undefined in a resample whose three arms do not fall and in one whose rule polls fall outside the arms. Under the `joint` rule a resample in which either rule's gain is undefined is left out of both gains and of their difference, which is predict.py's `boot_gains` and the rule every verdict above reads. Under `per quantity` it is left out of each quantity it leaves undefined and no other. spec.md counts such a resample and leaves it out; no value is substituted for it.

| scoring | quantity | drop rule | resamples left out | of | reasons |
|---|---|---|---|---|---|
| headline | diff | joint | 0 | 2000 | none |
| headline | diff | per quantity | 0 | 2000 | none |
| headline | gain_F | joint | 0 | 2000 | none |
| headline | gain_F | per quantity | 0 | 2000 | none |
| headline | gain_T37r | joint | 0 | 2000 | none |
| headline | gain_T37r | per quantity | 0 | 2000 | none |
| headline | ratio_delay | joint | 0 | 2000 | none |
| headline | ratio_delay | per quantity | 0 | 2000 | none |
| headline | ratio_polls | joint | 0 | 2000 | none |
| headline | ratio_polls | per quantity | 0 | 2000 | none |
| all records | diff | joint | 0 | 2000 | none |
| all records | diff | per quantity | 0 | 2000 | none |
| all records | gain_F | joint | 0 | 2000 | none |
| all records | gain_F | per quantity | 0 | 2000 | none |
| all records | gain_T37r | joint | 0 | 2000 | none |
| all records | gain_T37r | per quantity | 0 | 2000 | none |
| all records | ratio_delay | joint | 0 | 2000 | none |
| all records | ratio_delay | per quantity | 0 | 2000 | none |
| all records | ratio_polls | joint | 0 | 2000 | none |
| all records | ratio_polls | per quantity | 0 | 2000 | none |

Whether the two rules kept the same resamples, per scoring and per quantity. The joint rule drops a resample only when some rule's gain is undefined, and the per quantity rule then drops that same resample from at least that quantity, so each quantity's per quantity kept set contains its joint one. Equal counts therefore mean equal sets, and the counts are compared here rather than assumed equal.

| scoring | quantity | joint left out | per quantity left out | same resample set |
|---|---|---|---|---|
| headline | diff | 0 | 0 | yes |
| headline | gain_F | 0 | 0 | yes |
| headline | gain_T37r | 0 | 0 | yes |
| headline | ratio_delay | 0 | 0 | yes |
| headline | ratio_polls | 0 | 0 | yes |
| all records | diff | 0 | 0 | yes |
| all records | gain_F | 0 | 0 | yes |
| all records | gain_T37r | 0 | 0 | yes |
| all records | ratio_delay | 0 | 0 | yes |
| all records | ratio_polls | 0 | 0 | yes |

## Where spec.md left a choice

These are the places spec.md leaves a choice open and what this script took in each. Most of them are the plainest reading of spec.md's own text. The drop rule below is not: spec.md's text reads one quantity at a time and says nothing about binding the quantities together, and the rule taken there is the frozen predict.py's. None of them moves a pass line, a band or a verdict word.

- The verdict of a pass line 2 or line 3 row is the headline scoring's. spec.md's Quantities makes the headline the scoring the pass lines read and says the all-record value is reported anyway, so the all-record word is printed beside the headline word in every table. Under `w0` the two select the same records, and the `headline equals all records` column above is the measurement of that.

- The seed set is the intersection of the DONE and unflagged seeds over the five stages, because spec.md's bootstrap recomputes all five stages from the same resampled seeds. Line 1's own interval check and the decision-row counters are read over every DONE run instead, flagged or not, because spec.md asks the interval check of every decision log.

- Which resamples a quantity's interval is drawn from. spec.md asks for a resample that loses the frontier to be counted and left out, which reads one quantity at a time; it leaves open whether a resample that one rule loses is also left out of the other rule's gain. The rule taken here is the frozen predict.py's `boot_gains`, which leaves a resample out of both rules' gains and of their difference whenever either rule's gain is undefined, while evidence/19's ana.py leaves it out of each quantity separately. It is taken because the frozen prediction's own intervals were drawn under it, so the measured interval and the predicted interval are the same statistic; an interval drawn per quantity would be a different one from what predict.txt prints. The joint rule therefore carries every verdict, every printed interval that a verdict reads and every half width of pass lines 2 and 3, and the per quantity interval and drop count are printed beside it in the same tables. predict.py's `boot_ratios` drops no resample from a ratio, so the two rules are expected to agree on pass line 3; the `same resample set` table above compares their counts instead of assuming it.

- `no line` is read off the full sample, not off the resamples: spec.md's `no line` is the case where the arms of the measurement do not fall or a rule's own polls fall outside them, while a resample that loses the frontier is counted and left out. `incomplete` is tested before `no line`, because spec.md's parenthesis puts it in place of any verdict computed on fewer than 40 seeds.

- A rule whose own full-sample gain is defined but whose every resample the joint drop rule left out, because the OTHER rule lost its frontier in all of them. There is then no interval to read a word off, and spec.md has no word for that case. The slot reads `no line` as well, with a `why` cell saying the joint rule left no resample because the other rule lost its frontier; a new word would be a reading spec.md never defined. The same applies to the F minus T37r sentence, which has no `why` cell of its own. On this step's data it did not arise: the joint rule left 0 resamples out of 2000, over both scorings and all three quantities it binds together.

- The bands of line 4 are predict.txt's own printed band ends, not the printed median times 1 -+ 6 %. The two differ in the third decimal because predict.txt rounds the median after forming the band. spec.md pass line 4 says "the prediction tabled above", and predict.txt is the file that table was printed from.

- Line 4 reads the delay of the headline scoring. predict.txt prints one delay per stage and says the two scorings give the same digits under `w0`, so there is one predicted number to compare against and the all-record delay stands beside it in the table above.

- The allowed interval set of a stage is built from the string main10.sh launches, not from the values spec.md lists. spec.md lists the same sets, and building them from the launcher is what makes the check a check of what the sweep ran. The fixed arms' sets come from `--interval-choices`, which is logged as the nominal value because the jitter of 0.1 is applied after the choice is logged.

- spec.md reports that both validators on the `--poll-rule` path accept the pinned F and T37r, with the bounds they enforce. That was checked by calling them when the member was pinned; this script restates the five values against main10.sh and fpin.txt and imports no semindex module, so the validators are not called again here.

- spec.md gives no reading for the case where a stage has no scored record at all. On a partial store that happens, and such a stage is reported as `-` rather than as a number, with the resamples that lose the frontier counted under `stage scored nothing`. In line 4 such a value reads `not measured` and counts as neither inside nor outside its band, because calling it outside would read as a measurement that missed.

