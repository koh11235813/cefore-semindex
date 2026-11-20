# evidence/19 results

Computed by ana.py from perrun.csv here, ../16/perrun.csv and the stored runs' meta.json and event logs. The pass lines, the bands, the predictions and the verdict words are spec.md's; the words are read off the numbers mechanically and nothing here adds to them. Every line names the number of seeds it used, and a line computed on fewer seeds than it names gets the word `incomplete` in place of its verdict.

Two scorings, as spec.md fixes them. The headline scoring takes the records whose scheduled creation time is at or before t0+72.0 s, selected by the experiment description's `schedule[host][seq-1]["at_s"] <= 72.0` read against discovery.csv's `(origin, seq)`, and its delay is the mean over the uncensored ones 4 hops away. The all-record scoring takes every record of the run, which is the quantity every earlier step reported. Polls per run is counted over the whole window [t0+10 s, t0+130 s] under both scorings. The headline scoring carries each row's verdict word and the all-record value stands beside it.

| load | shape | stages |
|---|---|---|
| `wo` | ON 20 s / OFF 20 s alternating | `s9wo_b3p695`, `s9wo_b6p0`, `s9wo_b10p043`, `s9wo_t37`, `s9wo_t37r` |
| `wv` | ON and OFF lengths exponential, mean 20 s | `s9wv_b3p695`, `s9wv_b6p0`, `s9wv_b10p043`, `s9wv_t37`, `s9wv_t37r` |

| arm | setting |
|---|---|
| B3 | fixed 3.695 s |
| B6 | fixed 6.0 s |
| B10 | fixed 10.043 s |
| T37 | rule, on_timeout keep |
| T37r | rule, on_timeout reset |

## Line 1: the comparison across code versions holds (the gate)

The three `s9w0_*` runs, seed 1, the default load. The first part of spec.md's line 1, that a generated run with `--workload w0` is byte-identical to one generated without it, is a test checked at merge and is not re-read here.

Every `interval_s` in the decision log of a gate run must be a value the setting allows. The decision log is every `decision` event of kind `state_poll` or `state_poll_reschedule` on every host, which is the whole log of the rule's own arithmetic.

| gate stage | allowed values | values seen (count) | exact |
|---|---|---|---|
| `s9w0_b3p695` | 3.695 | 3.695 (279) | yes |
| `s9w0_b10p043` | 10.043 | 10.043 (105) | yes |
| `s9w0_t37r` | 3.695, 7.39, 10.043 | 3.695 (266), 7.39 (81), 10.043 (49) | yes |

The two fixed arms against spec.md's bands, and `s9w0_t37r` beside the original `s7t37r` seed-1 run of ../16/perrun.csv with no band on it. On the default load every record is scheduled at or before t0+71 s, so the headline set is the whole set of records and the two scorings give the same delay here; the equality is checked in the last column.

| gate stage | polls | polls band | delay s | delay band | censored share | ../16 seed-1 run: polls / delay s / censored share | headline equals all records |
|---|---|---|---|---|---|---|---|
| `s9w0_b3p695` | 255.00 | 242.25 .. 267.75 | 19.329 | 15.771 .. 23.656 | 0.0000 | `s7b3p695` 255.00 / 19.713 / 0.0000 | yes |
| `s9w0_b10p043` | 97.00 | 91.20 .. 100.80 | 34.908 | 29.035 .. 43.553 | 0.0167 | `s7b10p043` 96.00 / 36.294 / 0.0167 | yes |
| `s9w0_t37r` | 182.00 | reported only | 18.865 | reported only | 0.0000 | `s7t37r` 182.00 / 21.577 / 0.0333 | yes |

Gate runs DONE and unflagged: 3 of 3.
Verdict on 3 runs: **pass**.

## Line 2: all 200 sweep runs DONE and valid

Validity is `<run_dir>/meta.json` -> `validity` (`valid`, `flags`), written by semindex.evalrun from semindex.evalvalid. Of the flags, `record_off_schedule` and `record_count` say directly whether the on/off schedule was kept by the testbed, so the count of each is printed per stage even when it is zero. The flag names carry the node and the sequence number (`record_off_schedule:h0:seq5`), so what is counted is the flags whose name starts with those two words.

| stage | load | runs expected | runs DONE | valid and unflagged | flagged | record_off_schedule | record_count | other flags seen |
|---|---|---|---|---|---|---|---|---|
| `s9wo_b3p695` | `wo` | 20 | 20 | 20 | 0 | 0 | 0 | none |
| `s9wo_b6p0` | `wo` | 20 | 20 | 20 | 0 | 0 | 0 | none |
| `s9wo_b10p043` | `wo` | 20 | 20 | 20 | 0 | 0 | 0 | none |
| `s9wo_t37` | `wo` | 20 | 20 | 20 | 0 | 0 | 0 | none |
| `s9wo_t37r` | `wo` | 20 | 20 | 20 | 0 | 0 | 0 | none |
| `s9wv_b3p695` | `wv` | 20 | 20 | 20 | 0 | 0 | 0 | none |
| `s9wv_b6p0` | `wv` | 20 | 20 | 20 | 0 | 0 | 0 | none |
| `s9wv_b10p043` | `wv` | 20 | 20 | 20 | 0 | 0 | 0 | none |
| `s9wv_t37` | `wv` | 20 | 20 | 20 | 0 | 0 | 0 | none |
| `s9wv_t37r` | `wv` | 20 | 20 | 20 | 0 | 0 | 0 | none |

200 of 200 sweep runs DONE, 200 of them valid and unflagged.
Verdict on 200 runs: **pass**.

Seeds each load's lines are read on, the intersection over that load's five stages:

| load | seeds used | seeds |
|---|---|---|
| `wo` | 20 of 20 | 1 to 20 |
| `wv` | 20 of 20 | 1 to 20 |

## Line 3: gain against the fixed-interval line

The line is the three arms of the same load and p (3.695, 6.0, 10.043 s), polls per run on the horizontal axis and delay on the vertical, joined piecewise in log-log. Gain is `1 - delay(rule) / the line's delay at the rule's own polls per run`. The arms must be monotone, which here means delay falls strictly as polls per run rise; a resample whose arms are not monotone, and a resample whose rule polls fall outside the outer arms, is counted and left out.

The reading is attached to each of the four (load, rule) rows separately and the four rows are not required to agree.

### Load `wo`, 20 of 20 seeds

| arm | stage | scoring | delay s | censored share | polls per run |
|---|---|---|---|---|---|
| B3 | `s9wo_b3p695` | headline | 22.131 | 0.00000 | 256.15 |
| B3 | `s9wo_b3p695` | all records | 21.043 | 0.05037 | 256.15 |
| B6 | `s9wo_b6p0` | headline | 29.554 | 0.00510 | 159.35 |
| B6 | `s9wo_b6p0` | all records | 28.298 | 0.11018 | 159.35 |
| B10 | `s9wo_b10p043` | headline | 41.869 | 0.02211 | 96.10 |
| B10 | `s9wo_b10p043` | all records | 39.264 | 0.23610 | 96.10 |
| T37 | `s9wo_t37` | headline | 39.006 | 0.03061 | 136.75 |
| T37 | `s9wo_t37` | all records | 35.571 | 0.18573 | 136.75 |
| T37r | `s9wo_t37r` | headline | 27.265 | 0.00000 | 180.75 |
| T37r | `s9wo_t37r` | all records | 25.927 | 0.07555 | 180.75 |

Arm points of the full sample, headline scoring, sorted by polls per run: (96.10 polls, 41.869 s), (159.35 polls, 29.554 s), (256.15 polls, 22.131 s). Monotone: yes.

| rule | scoring | gain % | 95 % interval % | half width pp | word | predicted gain % and interval | predicted word | resamples left out |
|---|---|---|---|---|---|---|---|---|
| T37 | headline | -18.79 | [-25.82, -11.58] | 7.12 | **reversed** | -17.73 [-26.03, -10.01], half width 7.80 pp | reversed | 0 |
| T37r | headline | +0.38 | [-10.79, +10.31] | 10.55 | **not seen** | +1.64 [-8.03, +9.91], half width 8.81 pp | not seen | 0 |

T37r minus T37 on the same frontier, headline scoring: +19.17 pp [+7.76, +29.54], half width 10.89 pp, word **seen**. The replay predicted +19.95 pp [+8.80, +29.69], half width 10.30 pp, with 40 of 40 experiments entirely above zero. spec.md attaches no pass line to this difference.


Arm points of the full sample, all records scoring, sorted by polls per run: (96.10 polls, 39.264 s), (159.35 polls, 28.298 s), (256.15 polls, 21.043 s). Monotone: yes.

| rule | scoring | gain % | 95 % interval % | half width pp | word | predicted gain % and interval | predicted word | resamples left out |
|---|---|---|---|---|---|---|---|---|
| T37 | all records | -13.85 | [-20.69, -7.27] | 6.71 | **reversed** | -13.16 [-20.42, -6.29], half width 6.83 pp | reversed | 0 |
| T37r | all records | +0.88 | [-7.97, +8.80] | 8.38 | **not seen** | +1.17 [-7.10, +8.73], half width 7.73 pp | not seen | 0 |

T37r minus T37 on the same frontier, all records scoring: +14.73 pp [+5.10, +23.67], half width 9.28 pp, word **seen**. The replay predicted +14.55 pp [+4.64, +22.81], half width 8.91 pp, with 40 of 40 experiments entirely above zero. spec.md attaches no pass line to this difference.

### Load `wv`, 20 of 20 seeds

| arm | stage | scoring | delay s | censored share | polls per run |
|---|---|---|---|---|---|
| B3 | `s9wv_b3p695` | headline | 22.174 | 0.00000 | 255.95 |
| B3 | `s9wv_b3p695` | all records | 21.043 | 0.05312 | 255.95 |
| B6 | `s9wv_b6p0` | headline | 29.300 | 0.00000 | 159.35 |
| B6 | `s9wv_b6p0` | all records | 28.090 | 0.12395 | 159.35 |
| B10 | `s9wv_b10p043` | headline | 42.729 | 0.02191 | 96.10 |
| B10 | `s9wv_b10p043` | all records | 40.313 | 0.27493 | 96.10 |
| T37 | `s9wv_t37` | headline | 37.034 | 0.04382 | 140.60 |
| T37 | `s9wv_t37` | all records | 33.003 | 0.16589 | 140.60 |
| T37r | `s9wv_t37r` | headline | 26.422 | 0.00000 | 180.70 |
| T37r | `s9wv_t37r` | all records | 25.397 | 0.05638 | 180.70 |

Arm points of the full sample, headline scoring, sorted by polls per run: (96.10 polls, 42.729 s), (159.35 polls, 29.300 s), (255.95 polls, 22.174 s). Monotone: yes.

| rule | scoring | gain % | 95 % interval % | half width pp | word | predicted gain % and interval | predicted word | resamples left out |
|---|---|---|---|---|---|---|---|---|
| T37 | headline | -15.13 | [-24.08, -6.70] | 8.69 | **reversed** | -16.51 [-27.46, -7.06], half width 10.13 pp | reversed | 0 |
| T37r | headline | +2.90 | [-8.86, +13.70] | 11.28 | **not seen** | +2.51 [-7.90, +11.84], half width 9.71 pp | not seen | 0 |

T37r minus T37 on the same frontier, headline scoring: +18.03 pp [+2.53, +32.60], half width 15.03 pp, word **seen**. The replay predicted +19.03 pp [+5.08, +32.03], half width 13.90 pp, with 37 of 40 experiments entirely above zero. spec.md attaches no pass line to this difference.


Arm points of the full sample, all records scoring, sorted by polls per run: (96.10 polls, 40.313 s), (159.35 polls, 28.090 s), (255.95 polls, 21.043 s). Monotone: yes.

| rule | scoring | gain % | 95 % interval % | half width pp | word | predicted gain % and interval | predicted word | resamples left out |
|---|---|---|---|---|---|---|---|---|
| T37 | all records | -7.44 | [-14.91, -0.37] | 7.27 | **reversed** | -10.35 [-18.65, -2.75], half width 7.41 pp | reversed | 0 |
| T37r | all records | +2.38 | [-7.39, +11.51] | 9.45 | **not seen** | +2.60 [-4.88, +10.89], half width 7.81 pp | not seen | 0 |

T37r minus T37 on the same frontier, all records scoring: +9.82 pp [-1.02, +20.56], half width 10.79 pp, word **not seen**. The replay predicted +12.97 pp [+2.05, +23.44], half width 10.90 pp, with 34 of 40 experiments entirely above zero. spec.md attaches no pass line to this difference.

What spec.md says about each reading. The word of the headline scoring is the row's reading; the all-record word is printed beside it.

| load | rule | headline word | all-record word | predicted word | spec.md's text for the headline word |
|---|---|---|---|---|---|
| `wo` | T37 | **reversed** | reversed | reversed | Reversed is the prediction, and it says that T37 loses to the fixed interval on that load. |
| `wo` | T37r | **not seen** | not seen | not seen | Not seen is the prediction, and even then nothing is claimed about the rule. It is not a demonstration that the rule adds nothing. The interval is wider than the effect, so the only claim is that neither direction can be told. |
| `wv` | T37 | **reversed** | reversed | reversed | Reversed is the prediction, and it says that T37 loses to the fixed interval on that load. |
| `wv` | T37r | **not seen** | not seen | not seen | Not seen is the prediction, and even then nothing is claimed about the rule. It is not a demonstration that the rule adds nothing. The interval is wider than the effect, so the only claim is that neither direction can be told. |

Every one of the four rows read the word spec.md predicted, so none of them sends the step to investigate a reason.

For `wv` T37 spec.md asks what each scoring read, because the replay's two scorings already differ in how wide the interval is there. Both words are in the table above and the two intervals are in the per-load tables.

## Line 4: T37r against T37 directly, per load

The delay ratio and the polls ratio, from the same resamples, printed as the percent change of T37r against T37 so that they sit beside the replay's numbers. The censored share of each rule stands beside the delay. spec.md attaches no pass or fail to this line: the direction is known from the replay and this line is the measurement of the size. The default load measured -23.0 % in delay and +26.8 % in polls, neither interval spanning zero (../16/after.md).

| load | seeds | quantity | scoring | measured % | 95 % interval % | measured half width pp | predicted % and interval | predicted half width pp | spans zero |
|---|---|---|---|---|---|---|---|---|---|
| `wo` | 20 | delay | headline | -30.10 | [-37.25, -22.94] | 7.15 | -30.31 [-37.11, -22.77] | 6.84 | no |
| `wo` | 20 | delay | all records | -27.11 | [-33.15, -20.62] | 6.27 | -26.50 [-32.87, -19.42] | 6.44 | no |
| `wo` | 20 | polls | headline | +32.18 | [+28.51, +35.88] | 3.68 | +31.27 [+27.75, +35.07] | 3.65 | no |
| `wo` | 20 | polls | all records | +32.18 | [+28.51, +35.88] | 3.68 | +31.27 [+27.75, +35.07] | 3.65 | no |
| `wv` | 20 | delay | headline | -28.66 | [-39.03, -16.77] | 11.13 | -29.39 [-38.54, -18.05] | 10.09 | no |
| `wv` | 20 | delay | all records | -23.04 | [-31.77, -13.84] | 8.96 | -24.89 [-33.13, -15.29] | 8.89 | no |
| `wv` | 20 | polls | headline | +28.52 | [+24.42, +32.37] | 3.98 | +27.89 [+23.37, +32.45] | 4.49 | no |
| `wv` | 20 | polls | all records | +28.52 | [+24.42, +32.37] | 3.98 | +27.89 [+23.37, +32.45] | 4.49 | no |

The polls ratio is the same number under both scorings, because polls are counted over the whole window whichever records the delay is taken on. predict.txt prints it once per load for that reason and the same predicted numbers stand under both rows here.

Censored shares of the two rule stages, beside the delays above:

| load | stage | scoring | delay s | censored share |
|---|---|---|---|---|
| `wo` | `s9wo_t37` | headline | 39.006 | 0.03061 |
| `wo` | `s9wo_t37` | all records | 35.571 | 0.18573 |
| `wo` | `s9wo_t37r` | headline | 27.265 | 0.00000 |
| `wo` | `s9wo_t37r` | all records | 25.927 | 0.07555 |
| `wv` | `s9wv_t37` | headline | 37.034 | 0.04382 |
| `wv` | `s9wv_t37` | all records | 33.003 | 0.16589 |
| `wv` | `s9wv_t37r` | headline | 26.422 | 0.00000 |
| `wv` | `s9wv_t37r` | all records | 25.397 | 0.05638 |

## Line 5: did the replay predict the measurement?

The headline delay and the polls per run over the whole window of all ten sweep stages, 20 values, each against the plus or minus 6 % band of spec.md, copied from it as printed. 18 or more of the 20 inside reads `predicted`, 15 to 17 reads `mixed`, 14 or fewer reads `missed`. The 6 % band comes from the largest deviation the replay showed on the default load, 5.9 % (evidence/13); that band was measured on the default load and nothing says it is the right band for an on/off load, so every value outside it is reported with how far outside and in which direction whatever the count is.

| stage | quantity | measured | prediction | band | inside or outside | deviation % |
|---|---|---|---|---|---|---|
| `s9wo_b3p695` | headline delay s | 22.131 | 21.549 | 20.256 .. 22.842 | inside | +2.70 |
| `s9wo_b3p695` | polls per run | 256.15 | 256.15 | 240.78 .. 271.52 | inside | +0.00 |
| `s9wo_b6p0` | headline delay s | 29.554 | 29.537 | 27.764 .. 31.309 | inside | +0.06 |
| `s9wo_b6p0` | polls per run | 159.35 | 159.40 | 149.84 .. 168.96 | inside | -0.03 |
| `s9wo_b10p043` | headline delay s | 41.869 | 41.218 | 38.745 .. 43.691 | inside | +1.58 |
| `s9wo_b10p043` | polls per run | 96.10 | 95.90 | 90.15 .. 101.65 | inside | +0.21 |
| `s9wo_t37` | headline delay s | 39.006 | 38.452 | 36.145 .. 40.759 | inside | +1.44 |
| `s9wo_t37` | polls per run | 136.75 | 137.00 | 128.78 .. 145.22 | inside | -0.18 |
| `s9wo_t37r` | headline delay s | 27.265 | 26.596 | 25.000 .. 28.191 | inside | +2.52 |
| `s9wo_t37r` | polls per run | 180.75 | 179.88 | 169.08 .. 190.67 | inside | +0.48 |
| `s9wv_b3p695` | headline delay s | 22.174 | 21.471 | 20.183 .. 22.759 | inside | +3.27 |
| `s9wv_b3p695` | polls per run | 255.95 | 256.15 | 240.78 .. 271.52 | inside | -0.08 |
| `s9wv_b6p0` | headline delay s | 29.300 | 29.252 | 27.497 .. 31.007 | inside | +0.16 |
| `s9wv_b6p0` | polls per run | 159.35 | 159.40 | 149.84 .. 168.96 | inside | -0.03 |
| `s9wv_b10p043` | headline delay s | 42.729 | 42.501 | 39.951 .. 45.051 | inside | +0.54 |
| `s9wv_b10p043` | polls per run | 96.10 | 95.90 | 90.15 .. 101.65 | inside | +0.21 |
| `s9wv_t37` | headline delay s | 37.034 | 37.415 | 35.170 .. 39.660 | inside | -1.02 |
| `s9wv_t37` | polls per run | 140.60 | 141.07 | 132.61 .. 149.54 | inside | -0.33 |
| `s9wv_t37r` | headline delay s | 26.422 | 26.452 | 24.864 .. 28.039 | inside | -0.12 |
| `s9wv_t37r` | polls per run | 180.70 | 180.40 | 169.58 .. 191.22 | inside | +0.17 |

20 of 20 values inside their band. Verdict on 20 seeds per load: **predicted**.

If this line reads predicted, the on/off numbers of evidence/14 and 15 have support for the first time. That is the largest thing this step buys.

No value fell outside its band.

This line does not show that the creation times are what WO and WV are defined to be. The prediction reads the creation times out of the testbed's own experiment descriptions, so a mistake in the tick-laying code would be in the prediction too. That those instants have the defined shape is shown by `tests/test_evalgen.py`.

That the testbed and the replay ran the same instants is what this line's comparison states, and it is checked run by run. perrun.py reads the instants out of each run's own experiment description and draws them again from the committed generator for that run's own (host, seed, k, workload), which is the builder every description's schedule block is written from and the one the replay's instants come from. Runs whose two origins both agree, instant by instant and in order: 203 of 203.

The comparison of decision sequences that spec.md asks for when this line reads `mixed` or `missed` is not computed here. It is a reading of two logs against each other, which this script does not do. The word this line read is printed above, so the reader can see whether spec.md asks for that comparison at all.

## Also reported, without a pass line

Every stage's measured values beside the replay's, both scorings together. The 6.0 s arm appears here and in line 3's tables: spec.md runs it to measure the bend of the trade-off on these loads and attaches no pass line to it.

| stage | seeds | headline delay s | headline censored | all-record delay s | all-record censored | polls per run | replay headline delay s | replay all-record delay s | replay headline censored | replay all-record censored |
|---|---|---|---|---|---|---|---|---|---|---|
| `s9wo_b3p695` | 20 | 22.131 | 0.00000 | 21.043 | 0.05037 | 256.15 | 21.549 | 20.945 | 0.000 | 0.047 |
| `s9wo_b6p0` | 20 | 29.554 | 0.00510 | 28.298 | 0.11018 | 159.35 | 29.537 | 28.176 | 0.005 | 0.107 |
| `s9wo_b10p043` | 20 | 41.869 | 0.02211 | 39.264 | 0.23610 | 96.10 | 41.218 | 38.862 | 0.023 | 0.241 |
| `s9wo_t37` | 20 | 39.006 | 0.03061 | 35.571 | 0.18573 | 136.75 | 38.452 | 35.013 | 0.031 | 0.165 |
| `s9wo_t37r` | 20 | 27.265 | 0.00000 | 25.927 | 0.07555 | 180.75 | 26.596 | 25.627 | 0.000 | 0.075 |
| `s9wv_b3p695` | 20 | 22.174 | 0.00000 | 21.043 | 0.05312 | 255.95 | 21.471 | 20.747 | 0.000 | 0.043 |
| `s9wv_b6p0` | 20 | 29.300 | 0.00000 | 28.090 | 0.12395 | 159.35 | 29.252 | 28.023 | 0.000 | 0.120 |
| `s9wv_b10p043` | 20 | 42.729 | 0.02191 | 40.313 | 0.27493 | 96.10 | 42.501 | 39.771 | 0.022 | 0.266 |
| `s9wv_t37` | 20 | 37.034 | 0.04382 | 33.003 | 0.16589 | 140.60 | 37.415 | 33.503 | 0.044 | 0.169 |
| `s9wv_t37r` | 20 | 26.422 | 0.00000 | 25.397 | 0.05638 | 180.70 | 26.452 | 25.003 | 0.000 | 0.073 |

The 6.0 s arm against the straight line between the 3.695 s and 10.043 s arms of the same load, computed as 1 minus the measured delay over the delay the straight line reads at the arm's own polls. A positive value means the measured 6.0 s arm is faster than the straight line, so the trade-off bends below it, and that is the bend spec.md runs the third arm to measure. A negative value means the arm is slower than the line. No pass line is attached.

| load | scoring | bend % | 95 % interval % | resamples left out |
|---|---|---|---|---|
| `wo` | headline | +1.93 | [-5.22, +8.50] | 0 |
| `wo` | all records | +0.57 | [-4.65, +5.09] | 0 |
| `wv` | headline | +3.79 | [-2.36, +9.75] | 0 |
| `wv` | all records | +2.53 | [-3.50, +8.14] | 0 |

Record counts per seed, from the schedule each run was handed. predict.py printed the mean and the range of these counts over the seeds and no value per seed, so what stands per seed beside the counts is the `instants match` column: it reads `yes` when every stage of that cell had both origins' instants equal to the generator's own draw, which is the check line 5 reports the total of. All five stages of one (load, seed) are handed the same schedule, so one row per (load, seed) is the whole of it; the `stages agree` column says that every stage of that cell carried the same two counts.

| load | seed | all records | headline records | stages agree | instants match |
|---|---|---|---|---|---|
| `wo` | 1 | 180 | 108 | yes | yes |
| `wo` | 2 | 196 | 124 | yes | yes |
| `wo` | 3 | 196 | 124 | yes | yes |
| `wo` | 4 | 196 | 124 | yes | yes |
| `wo` | 5 | 176 | 104 | yes | yes |
| `wo` | 6 | 200 | 128 | yes | yes |
| `wo` | 7 | 220 | 148 | yes | yes |
| `wo` | 8 | 184 | 108 | yes | yes |
| `wo` | 9 | 184 | 112 | yes | yes |
| `wo` | 10 | 192 | 120 | yes | yes |
| `wo` | 11 | 204 | 132 | yes | yes |
| `wo` | 12 | 208 | 136 | yes | yes |
| `wo` | 13 | 168 | 96 | yes | yes |
| `wo` | 14 | 224 | 152 | yes | yes |
| `wo` | 15 | 200 | 128 | yes | yes |
| `wo` | 16 | 196 | 124 | yes | yes |
| `wo` | 17 | 180 | 104 | yes | yes |
| `wo` | 18 | 176 | 104 | yes | yes |
| `wo` | 19 | 160 | 80 | yes | yes |
| `wo` | 20 | 172 | 96 | yes | yes |
| `wv` | 1 | 236 | 128 | yes | yes |
| `wv` | 2 | 128 | 84 | yes | yes |
| `wv` | 3 | 192 | 112 | yes | yes |
| `wv` | 4 | 228 | 112 | yes | yes |
| `wv` | 5 | 192 | 124 | yes | yes |
| `wv` | 6 | 204 | 168 | yes | yes |
| `wv` | 7 | 160 | 68 | yes | yes |
| `wv` | 8 | 192 | 148 | yes | yes |
| `wv` | 9 | 292 | 168 | yes | yes |
| `wv` | 10 | 260 | 144 | yes | yes |
| `wv` | 11 | 192 | 108 | yes | yes |
| `wv` | 12 | 236 | 100 | yes | yes |
| `wv` | 13 | 220 | 108 | yes | yes |
| `wv` | 14 | 316 | 208 | yes | yes |
| `wv` | 15 | 120 | 48 | yes | yes |
| `wv` | 16 | 240 | 160 | yes | yes |
| `wv` | 17 | 208 | 136 | yes | yes |
| `wv` | 18 | 256 | 160 | yes | yes |
| `wv` | 19 | 192 | 120 | yes | yes |
| `wv` | 20 | 228 | 152 | yes | yes |

| load | all records: mean [min .. max] | headline: mean [min .. max] | replay all records | replay headline | thinnest headline seed |
|---|---|---|---|---|---|
| `wo` | 190.6 [160 .. 224] | 117.6 [80 .. 152] | 190.6 [160 .. 224] | 117.6 [80 .. 152] | seed 19 with 80 records, the replay's seed 19 with 80 |
| `wv` | 214.6 [120 .. 316] | 127.8 [48 .. 208] | 214.6 [120 .. 316] | 127.8 [48 .. 208] | seed 15 with 48 records, the replay's seed 15 with 48 |

Whole-run counts over every decision row of the five rule stages, not cut to the measurement window, because what they describe is the rule's own arithmetic over every poll the run sent. An answered re-decision carries `empty`: true when the reply held nothing new, false when it held something new, null when the reply could not be decoded. A timed-out re-decision carries `empty` null by construction and is not counted among the answered ones.

| stage | runs | polls | re-decisions | answered | timed out | empty true share of answered | empty false | empty null and answered | clamped | late replies |
|---|---|---|---|---|---|---|---|---|---|---|
| `s9w0_t37r` | 1 | 198 | 198 | 122 | 76 | 0.5656 | 53 | 0 | 0 | 0 |
| `s9wo_t37` | 20 | 3055 | 3037 | 1934 | 1103 | 0.6350 | 706 | 0 | 0 | 0 |
| `s9wo_t37r` | 20 | 3935 | 3906 | 2240 | 1666 | 0.5982 | 900 | 0 | 0 | 0 |
| `s9wv_t37` | 20 | 3132 | 3113 | 1986 | 1127 | 0.5957 | 803 | 0 | 0 | 0 |
| `s9wv_t37r` | 20 | 3934 | 3905 | 2282 | 1623 | 0.5833 | 951 | 0 | 0 | 0 |

Every `interval_s` the rule stages logged, against the three values the rule can stand at (i_min 3.695, one back-off 7.39, the ceiling 10.043). Any other value would fill the `other` column.

| stage | 3.695 | 7.39 | 10.043 | other | all rows |
|---|---|---|---|---|---|
| `s9w0_t37r` | 266 | 81 | 49 | 0 | 396 |
| `s9wo_t37` | 2480 | 1464 | 2148 | 0 | 6092 |
| `s9wo_t37r` | 5247 | 1466 | 1128 | 0 | 7841 |
| `s9wv_t37` | 2811 | 1301 | 2133 | 0 | 6245 |
| `s9wv_t37r` | 5264 | 1431 | 1144 | 0 | 7839 |

The three fixed-arm stages of each load log one `interval_s` each, the arm's own nominal value; the jitter of 0.1 is applied after the choice is logged, so the logged value is the nominal one. Those stages are left out of the table above, which covers the stages whose interval is the rule's own output.

| stage | 3.695 | 6.0 | 10.043 | other | all rows |
|---|---|---|---|---|---|
| `s9wo_b3p695` | 5603 | 0 | 0 | 0 | 5603 |
| `s9wo_b6p0` | 0 | 3507 | 0 | 0 | 3507 |
| `s9wo_b10p043` | 0 | 0 | 2106 | 0 | 2106 |
| `s9wv_b3p695` | 5599 | 0 | 0 | 0 | 5599 |
| `s9wv_b6p0` | 0 | 3507 | 0 | 0 | 3507 |
| `s9wv_b10p043` | 0 | 0 | 2106 | 0 | 2106 |

## How the bootstrap was drawn

One index matrix, `numpy.random.default_rng(20261117).integers(0, n, size=(2000, n))`, drawn once at the seed count `n` the loads have and reused for both loads and both scorings. That is how predict.py drew its own matrix, so the measured bounds and the predicted bounds rest on the same resampled positions. The unit is the seed number: one index row selects the seeds, and all five stages of the load are recomputed from that one selection, so a resample holds the same seeds in every stage at once and the stages stay paired. A load short of a seed would be drawn at its own size from a generator with the same seed, and every size drawn is listed here.

| index matrix | seeds per resample | resamples |
|---|---|---|
| 1 | 20 | 2000 |

Resamples left out of every interval. A gain is left out of a resample when that resample's rule polls per run fall outside the arms, and also when that resample's three arms are not monotone. The ratios of line 4 are defined in every resample, so none is left out of them. spec.md counts such a resample and leaves it out; no value is substituted for it.

| load | scoring | quantity | resamples left out | reasons |
|---|---|---|---|---|
| `wo` | headline | bend_B6 | 0 | none |
| `wo` | headline | diff | 0 | none |
| `wo` | headline | gain_T37 | 0 | none |
| `wo` | headline | gain_T37r | 0 | none |
| `wo` | headline | ratio_delay | 0 | none |
| `wo` | headline | ratio_polls | 0 | none |
| `wo` | all records | bend_B6 | 0 | none |
| `wo` | all records | diff | 0 | none |
| `wo` | all records | gain_T37 | 0 | none |
| `wo` | all records | gain_T37r | 0 | none |
| `wo` | all records | ratio_delay | 0 | none |
| `wo` | all records | ratio_polls | 0 | none |
| `wv` | headline | bend_B6 | 0 | none |
| `wv` | headline | diff | 0 | none |
| `wv` | headline | gain_T37 | 0 | none |
| `wv` | headline | gain_T37r | 0 | none |
| `wv` | headline | ratio_delay | 0 | none |
| `wv` | headline | ratio_polls | 0 | none |
| `wv` | all records | bend_B6 | 0 | none |
| `wv` | all records | diff | 0 | none |
| `wv` | all records | gain_T37 | 0 | none |
| `wv` | all records | gain_T37r | 0 | none |
| `wv` | all records | ratio_delay | 0 | none |
| `wv` | all records | ratio_polls | 0 | none |

## Where spec.md left a choice

Each of these is the plainest reading of spec.md's own text; none of them moves a pass line, a band or a verdict word.

- The word of a (load, rule) row of line 3 is the headline scoring's word. spec.md's Quantities makes the headline the scoring the pass lines read and says the all-record value is reported beside it everywhere, so the all-record word is printed beside the headline word in every table.

- A load's seed set is the intersection over that load's five stages, because spec.md's bootstrap recomputes all five stages of that load from the same resampled seeds. The gate is counted separately, on its own three runs.

- A resample is left out of each quantity it leaves undefined, one quantity at a time. predict.py drops a resample from both rules at once when either rule has no readable frontier. Both counts are zero or near zero in the replay, so the two readings cannot part by much; the drop counts are printed above so the reader can see which case applied.

- The decision log of line 1 is every `decision` event of kind `state_poll` or `state_poll_reschedule`, which is every row in which the rule or the arm names an `interval_s`.

- Line 4's ratios are printed as percent changes of T37r against T37, which is the form predict.txt printed them in, and the raw ratio minus one is what results.csv carries.

- The headline sums are built from the discovery rows of the records the schedule puts at or before t0+72.0 s, while the all-record sums keep attrib's own censoring chain, which is where every earlier step's numbers come from. perrun.py computes the all-record sums the second way as well and reports how many runs the two ways disagree on; the count was 0 over all 203 runs, so the two scorings rest on one censoring rule.

- Line 5 reads `incomplete` on the smaller of the two loads' seed counts, since it is one word over the ten stages of both loads together.

- On the default load every record is scheduled at or before t0+71 s, so the gate's headline set is the whole set of records. Line 1's table checks that equality and prints the result, so it is a measurement here and not an assumption.

- spec.md's note under line 5 calls the check of the instants a file-against-file comparison. What is computed is each run's own experiment description against a fresh draw of the generator's schedule builder, which is the function every such file's schedule block is written from, so the second file is not written again to be read back. The plainest reading of the note is that the instants the testbed ran and the instants the replay scored are the same instants, and that is what this comparison answers.

