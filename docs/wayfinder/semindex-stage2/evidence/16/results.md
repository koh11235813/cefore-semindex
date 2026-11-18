# evidence/16 results

Computed by ana.py from perrun.csv here, ../10/perrun.csv and the stored runs' meta.json and event logs. The pass lines, the bands and the verdict words are spec.md's; the words are read off the numbers mechanically and nothing here adds to them.

## Line 1: all 40 runs valid

Validity source: `<run_dir>/meta.json` -> `validity` (`valid`, `flags`), written by semindex.evalrun from semindex.evalvalid. `results.json` carries the run's operations and no verdict, so it is not read. A stage with fewer than 20 DONE runs is incomplete: its pass word is withheld.

| stage | runs DONE | valid | flagged | flags seen | verdict |
|---|---|---|---|---|---|
| s7t37 | 20 | 20 | 0 | none | pass |
| s7t37r | 20 | 20 | 0 | none | pass |

## Line 2: is the difference seen?

Delay is pooled over records (`sum(lat4) / sum(n4)`, ms); polls per run is the mean of `polls_sent`; censored share is `sum(c4) / sum(n4 + c4)`. The arms come from ../10/perrun.csv at p 0.4. Seeds used: the 20 DONE in all four stages.

| stage | runs | delay ms | delay s | censored share | polls per run |
|---|---|---|---|---|---|
| s6b3p695 | 20 | 21463.5 | 21.463 | 0.00000 | 255.95 |
| s6b10p043 | 20 | 42448.3 | 42.448 | 0.03000 | 95.95 |
| s7t37 | 20 | 34483.2 | 34.483 | 0.00583 | 141.45 |
| s7t37r | 20 | 26555.7 | 26.556 | 0.00167 | 179.35 |

Log-log line through the two arms: through (95.95 polls, 42448.3 ms) and (255.95 polls, 21463.5 ms), slope d log delay / d log polls = -0.695.

| rule | polls per run | line delay ms at those polls | measured delay ms | gain | polls inside arms |
|---|---|---|---|---|---|
| s7t37 | 141.45 | 32412.1 | 34483.2 | -0.0639 | yes |
| s7t37r | 179.35 | 27482.1 | 26555.7 | 0.0337 | yes |

Gain difference s7t37r minus s7t37: 0.0976 (9.76 points). Interval from 2000 resamples of the 20 seeds with `default_rng(20261117)`, all four stages recomputed from the same resampled seeds: [-0.0224, 0.1968] ([-2.24, 19.68] points) over 2000 resamples with a defined gain; 0 resamples put a rule's polls per run outside the arms and were left out.

Verdict: **not seen**.

## Line 3: did the replay predict it?

Measured on each rule stage's own DONE runs. Bands are spec.md's plus or minus 6 % bands, copied from it; the deviation is against the prediction medians it prints.

| stage | runs | quantity | measured | prediction | band | inside or outside | deviation % |
|---|---|---|---|---|---|---|---|
| s7t37 | 20 | delay s | 34.483 | 34.627 | 32.549 .. 36.704 | inside | -0.42 |
| s7t37 | 20 | polls per run | 141.45 | 142.50 | 133.95 .. 151.05 | inside | -0.74 |
| s7t37r | 20 | delay s | 26.556 | 25.160 | 23.650 .. 26.669 | inside | +5.55 |
| s7t37r | 20 | polls per run | 179.35 | 180.15 | 169.34 .. 190.96 | inside | -0.44 |

Verdict: **pass**. All four values are inside their band.

## Line 4: existing policies unchanged (the three reruns)

Polls per run and delay are the windowed per-run values of the single seed-1 run on each side. The interval check is over every `state_poll` and `state_poll_reschedule` decision row of the rerun, whole run, not cut to the measurement window.

| rerun | original | polls rerun | polls original | polls band +-5 % | delay rerun s | delay original s | delay band +-20 % | decision rows | rows by kind | bad intervals |
|---|---|---|---|---|---|---|---|---|---|---|
| s7short | s6short | 140 | 136 | inside | 29.910 | 29.809 | inside | 296 | state_poll 148 + state_poll_reschedule 148 | 0 |
| s7b3p695 | s6b3p695 | 255 | 256 | inside | 19.713 | 19.356 | inside | 279 | state_poll 279 | 0 |
| s7b10p043 | s6b10p043 | 96 | 96 | inside | 36.294 | 36.339 | inside | 105 | state_poll 105 | 0 |

The two exact checks of this line are the interval check above and the policy tests at merge; the two bands are the coarse guide spec.md calls them, since answered against timed out is a race on the testbed.

## Also reported, without a pass line

Whole-run counts over every decision row of the rule stages' DONE runs. An answered re-decision carries `empty`: true when the reply held nothing new, false when it held something new, null when the reply could not be decoded. A timed-out re-decision carries `empty` null by construction and is not counted among the answered ones.

| stage | runs | polls | re-decisions | answered | timed out | empty true share of answered | empty false | empty null and answered | clamped | late replies |
|---|---|---|---|---|---|---|---|---|---|---|
| s7t37 | 20 | 3149 | 3137 | 1979 | 1158 | 0.6099 | 772 | 0 | 0 | 0 |
| s7t37r | 20 | 3907 | 3885 | 2260 | 1625 | 0.5947 | 916 | 0 | 0 | 0 |

Every `interval_s` the rule stages logged, against the three values the rule can stand at (i_min 3.695, one back-off 7.39, the ceiling 10.043). Any other value would be a fourth column here.

| stage | 3.695 | 7.39 | 10.043 | other | all rows |
|---|---|---|---|---|---|
| s7t37 | 2898 | 1244 | 2144 | 0 | 6286 |
| s7t37r | 5213 | 1416 | 1163 | 0 | 7792 |
