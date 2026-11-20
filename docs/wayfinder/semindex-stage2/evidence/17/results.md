# evidence/17 results

Computed by ana.py from perrun.csv here, ../10/perrun.csv, ../16/perrun.csv and the stored runs' meta.json and event logs. The pass lines, the bands and the verdict words are spec.md's and its amendments of 2026-11-19; the words are read off the numbers mechanically and nothing here adds to them. Every line names the number of seeds it used, and a line computed on fewer seeds than it names gets the word `incomplete` in place of its verdict (amendment 1).

Logical arms and the stage each seed's run comes from:

| arm | setting | seeds 1 to 20 | seeds 21 to 40 |
|---|---|---|---|
| B3 | fixed 3.695 s | `s6b3p695` | `s8b3p695` |
| B10 | fixed 10.043 s | `s6b10p043` | `s8b10p043` |
| B6 | fixed 6.0 s | `s8b6p0` | `s8b6p0` |
| T37 | rule, on_timeout keep | `s7t37` | `s8t37` |
| T37r | rule, on_timeout reset | `s7t37r` | `s8t37r` |

## Line 1: all 120 runs DONE and valid

DONE is a run with a DONE marker, which is what perrun.py puts in perrun.csv. Validity is `<run_dir>/meta.json` -> `validity` (`valid`, `flags`), written by semindex.evalrun from semindex.evalvalid. `results.json` carries the run's operations and no verdict, so it is not read.

| stage | runs expected | runs DONE | valid and unflagged | flagged | flags seen |
|---|---|---|---|---|---|
| s8t37 | 20 | 20 | 20 | 0 | none |
| s8t37r | 20 | 20 | 20 | 0 | none |
| s8b3p695 | 20 | 20 | 20 | 0 | none |
| s8b10p043 | 20 | 20 | 20 | 0 | none |
| s8b6p0 | 40 | 40 | 40 | 0 | none |

120 of 120 runs DONE, 120 of them valid and unflagged.
Verdict: **pass**.

## Line 2: the replication on seeds 21 to 40

Seeds 21 to 40 only, all four stages from this step's s8 runs. This reading is the answer to the question whether the difference is seen (amendment 2). It is independent of evidence/16: none of these seeds was read before.

Seeds used: 20 of 20, DONE and unflagged in `s8b3p695`, `s8b10p043`, `s8t37` and `s8t37r`.

| arm | stage | runs | delay ms | delay s | censored share | polls per run |
|---|---|---|---|---|---|---|
| B3 | `s8b3p695` | 20 | 21448.5 | 21.449 | 0.00000 | 256.20 |
| B10 | `s8b10p043` | 20 | 43100.8 | 43.101 | 0.03417 | 96.30 |
| T37 | `s8t37` | 20 | 33161.4 | 33.161 | 0.02833 | 142.75 |
| T37r | `s8t37r` | 20 | 26122.6 | 26.123 | 0.00000 | 180.45 |

Log-log line through the two fixed arms: through (96.30 polls, 43100.8 ms) and (256.20 polls, 21448.5 ms), slope d log delay / d log polls = -0.713.

| quantity | value | points | 95 % interval | interval in points |
|---|---|---|---|---|
| gain of T37 against the two-arm line | -0.0188 | -1.88 | [-0.0855, 0.0465] | [-8.55, 4.65] |
| gain of T37r against the two-arm line | 0.0515 | 5.15 | [-0.0131, 0.1159] | [-1.31, 11.59] |
| difference T37r minus T37 | 0.0702 | 7.02 | [-0.0202, 0.1585] | [-2.02, 15.85] |

Interval from 2000 resamples of the 20 seeds with `default_rng(20261119)`, the unit being the seed number and every stage recomputed from the same resampled seeds. Resamples left out of the difference: 0.

Verdict on 20 seeds: **not seen**.

## Line 3: the pooled estimate on seeds 1 to 40

Arms `s6b3p695` and `s6b10p043` on seeds 1 to 20 and `s8b3p695` and `s8b10p043` on seeds 21 to 40; rules `s7t37` and `s7t37r` on seeds 1 to 20 and `s8t37` and `s8t37r` on seeds 21 to 40.

This is not an independent test and it mixes two code versions: it contains the 20 seeds evidence/16 already read, and its seed-1-to-20 half was measured with the code before the rule was added and without `PYTHONDONTWRITEBYTECODE`. It is the best estimate of the size of the difference, and its word is never quoted without this sentence.

Seeds used: 40 of 40.

| arm | stage | runs | delay ms | delay s | censored share | polls per run |
|---|---|---|---|---|---|---|
| B3 | `s6b3p695/s8b3p695` | 40 | 21456.0 | 21.456 | 0.00000 | 256.07 |
| B10 | `s6b10p043/s8b10p043` | 40 | 42773.8 | 42.774 | 0.03208 | 96.12 |
| T37 | `s7t37/s8t37` | 40 | 33829.9 | 33.830 | 0.01708 | 142.10 |
| T37r | `s7t37r/s8t37r` | 40 | 26339.0 | 26.339 | 0.00083 | 179.90 |

| quantity | value | points | 95 % interval | interval in points |
|---|---|---|---|---|
| gain of T37 against the two-arm line | -0.0415 | -4.15 | [-0.0951, 0.0082] | [-9.51, 0.82] |
| gain of T37r against the two-arm line | 0.0426 | 4.26 | [-0.0146, 0.0942] | [-1.46, 9.42] |
| difference T37r minus T37 | 0.0841 | 8.41 | [0.0058, 0.1554] | [0.58, 15.54] |

Interval from 2000 resamples of the 40 seeds with `default_rng(20261119)`, unit the seed number. Resamples left out of the difference: 0.

Verdict word on 40 seeds: **seen**. This line is the estimate of the size of the difference. It is not an independent test.

Lines 2 and 3 split, which amendment 3 names as the single most likely outcome at this size. Reported as: not confirmed by the independent replication; pooled estimate 8.41 points, 95 % interval [0.58, 15.54] points.

## Line 4: the three-arm frontier

The frontier is piecewise linear in log-log through the 3.695 s, 6.0 s and 10.043 s arms, read at the rule's polls per run. The arms must be monotone, which here means delay falls strictly as polls per run rise. A resample whose arms are not monotone, and a resample whose rule polls fall outside the outer arms, is counted and left out.

The verdict words are read on seeds 21 to 40, where all three arms and both rules are one code version and no seed was read before (amendment 5). The seeds 1 to 40 numbers stand beside them with no verdict word.

Seeds used: 20 of 20 for the words, 40 of 40 for the numbers beside them.

| arm | stage | runs | delay ms | delay s | censored share | polls per run |
|---|---|---|---|---|---|---|
| B3 | `s8b3p695` | 20 | 21448.5 | 21.449 | 0.00000 | 256.20 |
| B6 | `s8b6p0` | 20 | 31503.9 | 31.504 | 0.00167 | 159.70 |
| B10 | `s8b10p043` | 20 | 43100.8 | 43.101 | 0.03417 | 96.30 |
| T37 | `s8t37` | 20 | 33161.4 | 33.161 | 0.02833 | 142.75 |
| T37r | `s8t37r` | 20 | 26122.6 | 26.123 | 0.00000 | 180.45 |

Read on seeds 21 to 40 (20 seeds), with the verdict words.

Full-sample arm points, sorted by polls per run: (96.30 polls, 43100.8 ms), (159.70 polls, 31503.9 ms), (256.20 polls, 21448.5 ms). Monotone: yes.

| quantity | value | points | 95 % interval | interval in points |
|---|---|---|---|---|
| gain of T37 against the three-arm frontier | 0.0181 | 1.81 | [-0.0492, 0.0772] | [-4.92, 7.72] |
| gain of T37r against the three-arm frontier | 0.0842 | 8.42 | [0.0174, 0.1474] | [1.74, 14.74] |
| difference T37r minus T37 against the frontier | 0.0661 | 6.61 | [-0.0199, 0.1476] | [-1.99, 14.76] |
| gain of the 6.0 s arm against the two-arm line | -0.0485 | -4.85 | [-0.1294, 0.0247] | [-12.94, 2.47] |

| quantity | resamples left out | reasons |
|---|---|---|
| gain of T37 against the three-arm frontier | 0 | none |
| gain of T37r against the three-arm frontier | 0 | none |
| difference T37r minus T37 against the frontier | 0 | none |
| gain of the 6.0 s arm against the two-arm line | 0 | none |

- gain of T37 against the three-arm frontier: **not distinguishable from the fixed frontier**
- gain of T37r against the three-arm frontier: **above the fixed frontier**
- difference T37r minus T37 against the frontier: **not seen**
- gain of the 6.0 s arm against the two-arm line: **the two-arm line and the three-arm frontier are not distinguishable**

The same quantities on seeds 1 to 40 (40 seeds), with no verdict word. The 6.0 s arm is all new code on every seed while the other two arms are the old code on seeds 1 to 20, which is why no word is attached here.

Full-sample arm points, sorted by polls per run: (96.12 polls, 42773.8 ms), (159.57 polls, 30442.2 ms), (256.07 polls, 21456.0 ms). Monotone: yes.

| quantity | value | points | 95 % interval | interval in points |
|---|---|---|---|---|
| gain of T37 against the three-arm frontier | -0.0281 | -2.81 | [-0.0771, 0.0189] | [-7.71, 1.89] |
| gain of T37r against the three-arm frontier | 0.0546 | 5.46 | [-0.0113, 0.1124] | [-1.13, 11.24] |
| difference T37r minus T37 against the frontier | 0.0826 | 8.26 | [0.0056, 0.1492] | [0.56, 14.92] |
| gain of the 6.0 s arm against the two-arm line | -0.0169 | -1.69 | [-0.0760, 0.0363] | [-7.60, 3.63] |

| quantity | resamples left out | reasons |
|---|---|---|
| gain of T37 against the three-arm frontier | 0 | none |
| gain of T37r against the three-arm frontier | 0 | none |
| difference T37r minus T37 against the frontier | 0 | none |
| gain of the 6.0 s arm against the two-arm line | 0 | none |

## Line 5: the replay check of the 6.0 s arm

`s8b6p0` on seeds 1 to 20 against the plus or minus 6 % bands of spec.md, copied from it as written. The centres are the replay's pooled prediction, the mean over 40 replicate streams; spec.md's amendment 7 records that a one-run-per-seed testbed is not expected to match a pooled replay line as closely as the per-experiment medians the evidence/16 bands used, and that how much of the 6 % is single-run noise is unknown. Lines 2 to 4 use measured arms only and are read whatever this line says.

Seeds used: 20 of 20.

| quantity | measured | prediction | band | inside or outside | deviation % |
|---|---|---|---|---|---|
| delay s | 29.378 | 29.744 | 27.959 .. 31.529 | inside | -1.23 |
| polls per run | 159.45 | 159.40 | 149.84 .. 168.96 | inside | +0.03 |

Verdict on 20 seeds: **pass**.

## Also reported, without a pass line

Every arm's delay, censored share and polls per run on each seed range, each on that arm's own DONE and unflagged seeds in the range. These counts are per arm, so they are not cut to the intersections the pass lines use.

| arm | seed range | stage | runs | delay ms | delay s | censored share | polls per run |
|---|---|---|---|---|---|---|---|
| B3 | 1 to 20 | `s6b3p695` | 20 | 21463.5 | 21.463 | 0.00000 | 255.95 |
| B3 | 21 to 40 | `s8b3p695` | 20 | 21448.5 | 21.449 | 0.00000 | 256.20 |
| B3 | 1 to 40 | `s6b3p695/s8b3p695` | 40 | 21456.0 | 21.456 | 0.00000 | 256.07 |
| B10 | 1 to 20 | `s6b10p043` | 20 | 42448.3 | 42.448 | 0.03000 | 95.95 |
| B10 | 21 to 40 | `s8b10p043` | 20 | 43100.8 | 43.101 | 0.03417 | 96.30 |
| B10 | 1 to 40 | `s6b10p043/s8b10p043` | 40 | 42773.8 | 42.774 | 0.03208 | 96.12 |
| B6 | 1 to 20 | `s8b6p0` | 20 | 29377.8 | 29.378 | 0.00417 | 159.45 |
| B6 | 21 to 40 | `s8b6p0` | 20 | 31503.9 | 31.504 | 0.00167 | 159.70 |
| B6 | 1 to 40 | `s8b6p0` | 40 | 30442.2 | 30.442 | 0.00292 | 159.57 |
| T37 | 1 to 20 | `s7t37` | 20 | 34483.2 | 34.483 | 0.00583 | 141.45 |
| T37 | 21 to 40 | `s8t37` | 20 | 33161.4 | 33.161 | 0.02833 | 142.75 |
| T37 | 1 to 40 | `s7t37/s8t37` | 40 | 33829.9 | 33.830 | 0.01708 | 142.10 |
| T37r | 1 to 20 | `s7t37r` | 20 | 26555.7 | 26.556 | 0.00167 | 179.35 |
| T37r | 21 to 40 | `s8t37r` | 20 | 26122.6 | 26.123 | 0.00000 | 180.45 |
| T37r | 1 to 40 | `s7t37r/s8t37r` | 40 | 26339.0 | 26.339 | 0.00083 | 179.90 |

The plain ratio of T37r to T37, with no line and no frontier in it. A ratio is defined in every resample, so none is left out.

| seed range | seeds | quantity | ratio | 95 % interval | resamples left out |
|---|---|---|---|---|---|
| 21 to 40 | 20 | delay T37r / T37 | 0.7877 | [0.7172, 0.8664] | 0 |
| 21 to 40 | 20 | polls T37r / T37 | 1.2641 | [1.2374, 1.2911] | 0 |
| 1 to 40 | 40 | delay T37r / T37 | 0.7786 | [0.7235, 0.8456] | 0 |
| 1 to 40 | 40 | polls T37r / T37 | 1.2660 | [1.2453, 1.2874] | 0 |

Whole-run counts over every decision row of `s8t37` and `s8t37r`, not cut to the measurement window, because what they describe is the rule's own arithmetic over every poll the run sent. An answered re-decision carries `empty`: true when the reply held nothing new, false when it held something new, null when the reply could not be decoded. A timed-out re-decision carries `empty` null by construction and is not counted among the answered ones.

| stage | runs | polls | re-decisions | answered | timed out | empty true share of answered | empty false | empty null and answered | clamped | late replies |
|---|---|---|---|---|---|---|---|---|---|---|
| s8t37 | 20 | 3175 | 3160 | 2002 | 1158 | 0.5919 | 817 | 0 | 0 | 0 |
| s8t37r | 20 | 3929 | 3908 | 2277 | 1631 | 0.5955 | 921 | 0 | 0 | 0 |

Every `interval_s` the two rule stages logged, against the three values the rule can stand at (i_min 3.695, one back-off 7.39, the ceiling 10.043). Any other value would fill the `other` column.

| stage | 3.695 | 7.39 | 10.043 | other | all rows |
|---|---|---|---|---|---|
| s8t37 | 2926 | 1289 | 2120 | 0 | 6335 |
| s8t37r | 5244 | 1436 | 1157 | 0 | 7837 |

## How the bootstrap was drawn

Every seed set is resampled with a fresh `numpy.random.default_rng(20261119)` of its own. The unit is the seed number: one index row of a draw selects the seeds, and every arm in use is recomputed from that one selection, so a resample holds the same seeds in all stages at once. A seed set is drawn once, as `rng.integers(0, n, size=(2000, n))`, and every interval on that seed set comes from that draw, so the intervals on one seed set share their resamples. A seed set that a later line asks for again reuses the draw it already has. A draw is fixed by the number of seeds in the set, so the intervals one line reports cannot move because another line's seed set changed.

| first wanted by | seeds | seed range |
|---|---|---|
| line 2, seeds 21 to 40, four s8 stages | 20 | 21 to 40 |
| line 3, seeds 1 to 40, s6 plus s8 arms and s7 plus s8 rules | 40 | 1 to 40 |

Resamples left out of every interval, over all lines. A gain is left out of a resample when that resample's rule polls per run fall outside the arms, and a frontier gain also when that resample's three arms are not monotone. A quantity that a seed set cannot define at all, because the seed set is short of a seed in an arm that quantity needs, has no row here.

| seed set | quantity | resamples left out | reasons |
|---|---|---|---|
| seeds 21 to 40, 20 seeds | diff_front | 0 | none |
| seeds 21 to 40, 20 seeds | diff_two | 0 | none |
| seeds 21 to 40, 20 seeds | gain_front_T37 | 0 | none |
| seeds 21 to 40, 20 seeds | gain_front_T37r | 0 | none |
| seeds 21 to 40, 20 seeds | gain_two_B6 | 0 | none |
| seeds 21 to 40, 20 seeds | gain_two_T37 | 0 | none |
| seeds 21 to 40, 20 seeds | gain_two_T37r | 0 | none |
| seeds 21 to 40, 20 seeds | ratio_delay | 0 | none |
| seeds 21 to 40, 20 seeds | ratio_polls | 0 | none |
| seeds 1 to 40, 40 seeds | diff_front | 0 | none |
| seeds 1 to 40, 40 seeds | diff_two | 0 | none |
| seeds 1 to 40, 40 seeds | gain_front_T37 | 0 | none |
| seeds 1 to 40, 40 seeds | gain_front_T37r | 0 | none |
| seeds 1 to 40, 40 seeds | gain_two_B6 | 0 | none |
| seeds 1 to 40, 40 seeds | gain_two_T37 | 0 | none |
| seeds 1 to 40, 40 seeds | gain_two_T37r | 0 | none |
| seeds 1 to 40, 40 seeds | ratio_delay | 0 | none |
| seeds 1 to 40, 40 seeds | ratio_polls | 0 | none |
