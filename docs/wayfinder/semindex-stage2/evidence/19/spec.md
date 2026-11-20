# evidence/19 spec: the on/off loads on the testbed

Written 2026-11-19, before any run of this step. Ticket `../../tickets/06-onoff-workload.md`, accepted by
the user the same day (its 決まったこと section). The pass lines below are translated from that ticket's
Acceptance and are not moved after the runs. Where this file says something the ticket does not, it is
marked as such.

## Question, and what the step is for

The ticket names three purposes, in this order.

**(a) The first check of a replayed load that has never been held against a measurement.** evidence/13
compared the replay with the testbed on the default load only: records every 2 s from t0+13 s to t0+71 s.
The numbers evidence/14 and 15 produced after replacing that load have no measurement behind them at all,
so their conclusions hang in the air. Pass line 5 reads this.

**(b) Decision logs on a load that switches generation on and off, for the later comparison of decision
makers.** The stage-2 map's note of 2026-11-17 said that a load where past arrivals predict the next
arrival is the one place a learned decision maker could beat a formula. No log of such a load exists. The
user's next step (2026-11-19) compares a distilled gemma4-e2b decision maker, an ML policy learned from
logs, and the formula rules; the 80 rule runs here (2 loads x T37, T37r x 20 seeds) are the material the
learning side is fed, and the 120 fixed-arm runs (2 loads x 3 arms x 20 seeds) are the values it is aimed
at.

**(c) The ranking of the rules, predicted to stay unresolved.** The replay puts the rules' gain over the
fixed-interval line at a few percent, far inside the interval half width that 20 seeds buy. The section
"what this step cannot show" gives the widths and the number of seeds that would be needed.

The nine hours of testbed time are spent mainly on (a) and (b). (c) staying unresolved is not a failure of
this step. Running these 203 runs only to learn the ranking would not be worth it, and would not be done.

Not answered here: p 0.2, p 0.5, p 0.6, p 0; the variant that lengthens on a timeout (T37b); all seven
fixed arms; whether a learned decision maker beats a formula; the probe numbers.

## The loads

The knob is `--workload`, implemented and committed on branch `feat/semindex-workload`.

| name | what it creates | records per run at k 4 |
|---|---|---|
| `w0` | the default load: 30 generation times, 2 s apart, from t0+13.0 s | 240, the same for every seed |
| `wo` | per origin, ON 20 s and OFF 20 s alternating, first ON start U[13, 33] s | see the counts below |
| `wv` | the same alternation with ON and OFF lengths from an exponential of mean 20 s clipped to [4, 60] s | see the counts below |

An ON block `[b, b+L)` is half open: ticks at b, b+2, ... strictly below b+L, k records sharing each tick's
`at_s`, no tick above t0+110 s. The times are drawn per origin from a
`random.Random` of the string `ev19:<NAME>:<seed>:h<host>`, with no `p` in it, so the same seed draws the
same schedule at every p and the
instants are a different draw from evidence/15's. That is why none of evidence/15's gain values is used as
a prediction for these runs: the prediction below is drawn again from the schedules the testbed will
actually be given.

## Record counts of these schedules (not in the ticket; its table is evidence/15's draw)

Counted from the experiment descriptions the committed generator writes for seeds 1 to 20, k 4, both
origins, p 0.4. These supersede the ticket's ranges, which came from `../15/onoff_cells.csv`.

| load | all records: mean [min .. max] | headline records (`at_s` <= 72.0): mean [min .. max] |
|---|---|---|
| `wo` | 190.6 [160 .. 224] | 117.6 [80 .. 152] |
| `wv` | 214.6 [120 .. 316] | 127.8 [48 .. 208] |

The thinnest headline sample is `wv` seed 15 with 48 records; `wo`'s thinnest is seed 19 with 80. The
default load puts all 240 of its records in the headline set, so the headline sample here is 80 to 152
records on `wo` and 48 to 208 on `wv`, against the 240 the measurements so far were scored on: a third to
two thirds of that on `wo`, and a fifth to seven eighths on `wv`. The precision of pass lines 3 to 5 is
limited by that, and it is why the step does not assume the precision evidence/16 had.

## Runs (`main9.sh`, from a detached clean checkout of the merged commit at `/home/lab_shared/wt/semindex-main9`)

Common settings, as the s6 and s7 stages: `--p 0.4 --k 4 --store-synchronous NORMAL
--disconnect-duration-s 8 12`. The outage schedule depends on (p, seed) only, so these stages see the
outages the s6 and s7 stages saw. Stage prefix `s9`; `s8` belongs to evidence/17.

| order | stages | load | seeds | runs | time |
|---|---|---|---|---|---|
| 1 gate | `s9w0_b3p695`, `s9w0_b10p043`, `s9w0_t37r` | `w0` | 1 | 3 | 8 min |
| 2 | `s9wo_b3p695`, `s9wo_b6p0`, `s9wo_b10p043`, `s9wo_t37`, `s9wo_t37r` | `wo` | 1 to 20 | 100 | 4 h 27 min |
| 3 | `s9wv_b3p695`, `s9wv_b6p0`, `s9wv_b10p043`, `s9wv_t37`, `s9wv_t37r` | `wv` | 1 to 20 | 100 | 4 h 27 min |

203 runs at 160 s each, about 9 h 1 min. The arms are `--interval-choices X --choice-jitter 0.1` at 3.695,
6.0 and 10.043 s; the rules are `--poll-rule` with `i_min_s` 3.695, `factor` 2, `i_max_s` 10.043,
`jitter_frac` 0.1 and `on_timeout` `keep` (T37) or `reset` (T37r).

- The gate runs first. Part 2 starts only once pass line 1 holds; if it does not, the sweep is not started
  and the gate is reported instead.
- The fixed arms are run again for each load: the line is a different line under a different load, so
  `w0`'s arms cannot be borrowed.
- Three arms rather than two. With arms only at 3.695 and 10.043 s the rules land between them, and the straight
  line between two points is off the real trade-off by an amount that the replay puts at 0.9 % for the
  default load and 0.6 % the other way for `wo`. The 6.0 s arm measures that bend on these loads for the
  first time. The 6.0 s arm of `w0` is not run here: evidence/17's `s8b6p0` measures that cell.
- Seeds run outer (for each load: for each seed, the five stages). A sweep stopped part way then holds
  every (seed, stage) pair of the seeds it finished, so nothing has to be retaken. All of `wo` first, then
  all of `wv`.
- Before each part, by hand: no emulator and no Cefore daemon running. The checkout is not edited while
  runs load it.

## Prediction (`predict.py`, output `predict.txt`, both placed here before the runs)

The evidence/15 engine on the outages, `agent_start`, t0 and topology logged for `s6b10p043` p 0.4, seeds 1
to 20, with the creation times read out of the experiment descriptions the committed generator writes for
these runs. 40 replicate streams; one stream is one experiment of 20 single runs, which is the condition
the testbed has. Every number below is the median over the 40 experiments.

Before the prediction is printed, `predict.py` runs the same code path on the default load and reproduces
`../16/predict.txt`'s archived medians, T37 34.627 s / 142.50 polls and T37r 25.160 s / 180.15 polls, to
the digits they were printed at. Both matched. A mismatch stops the script with the difference and prints
no prediction.

Headline delay = records whose scheduled creation is at or before t0+72 s, time until stored 4 hops away,
mean of the uncensored ones. Polls = over the whole window `[t0+10 s, t0+130 s]`. Bands are the median
times 1 plus or minus 6 % and are what pass line 5 reads.

| stage | headline delay s (band) | polls per run (band) | headline cens | all-record delay s | all-record cens |
|---|---|---|---|---|---|
| `s9wo_b3p695` | 21.549 (20.256 .. 22.842) | 256.15 (240.78 .. 271.52) | 0.000 | 20.945 | 0.047 |
| `s9wo_b6p0` | 29.537 (27.764 .. 31.309) | 159.40 (149.84 .. 168.96) | 0.005 | 28.176 | 0.107 |
| `s9wo_b10p043` | 41.218 (38.745 .. 43.691) | 95.90 (90.15 .. 101.65) | 0.023 | 38.862 | 0.241 |
| `s9wo_t37` | 38.452 (36.145 .. 40.759) | 137.00 (128.78 .. 145.22) | 0.031 | 35.013 | 0.165 |
| `s9wo_t37r` | 26.596 (25.000 .. 28.191) | 179.88 (169.08 .. 190.67) | 0.000 | 25.627 | 0.075 |
| `s9wv_b3p695` | 21.471 (20.183 .. 22.759) | 256.15 (240.78 .. 271.52) | 0.000 | 20.747 | 0.043 |
| `s9wv_b6p0` | 29.252 (27.497 .. 31.007) | 159.40 (149.84 .. 168.96) | 0.000 | 28.023 | 0.120 |
| `s9wv_b10p043` | 42.501 (39.951 .. 45.051) | 95.90 (90.15 .. 101.65) | 0.022 | 39.771 | 0.266 |
| `s9wv_t37` | 37.415 (35.170 .. 39.660) | 141.07 (132.61 .. 149.54) | 0.044 | 33.503 | 0.169 |
| `s9wv_t37r` | 26.452 (24.864 .. 28.039) | 180.40 (169.58 .. 191.22) | 0.000 | 25.003 | 0.073 |

Gain against the three-arm log-log frontier, with the 95 % seed-bootstrap interval and that interval's half
width, and how many of the 40 experiments put the interval entirely above zero ("seen"), across zero ("not
seen") and entirely below zero ("reversed"):

| load | rule | scoring | gain % | 95 % interval | half width pp | seen / not seen / reversed |
|---|---|---|---|---|---|---|
| `wo` | T37 | headline | -17.73 | -26.03 .. -10.01 | 7.80 | 0 / 0 / 40 |
| `wo` | T37 | all records | -13.16 | -20.42 .. -6.29 | 6.83 | 0 / 1 / 39 |
| `wo` | T37r | headline | +1.64 | -8.03 .. +9.91 | 8.81 | 2 / 38 / 0 |
| `wo` | T37r | all records | +1.17 | -7.10 .. +8.73 | 7.73 | 1 / 39 / 0 |
| `wv` | T37 | headline | -16.51 | -27.46 .. -7.06 | 10.13 | 0 / 1 / 39 |
| `wv` | T37 | all records | -10.35 | -18.65 .. -2.75 | 7.41 | 0 / 7 / 33 |
| `wv` | T37r | headline | +2.51 | -7.90 .. +11.84 | 9.71 | 0 / 40 / 0 |
| `wv` | T37r | all records | +2.60 | -4.88 .. +10.89 | 7.81 | 0 / 40 / 0 |

T37r minus T37, same frontier: `wo` headline +19.95 pp (+8.80 .. +29.69, half width 10.30, 40 of 40
experiments entirely above zero) and all records +14.55 pp (+4.64 .. +22.81, half width 8.91, 40 of 40);
`wv` headline +19.03 pp (+5.08 .. +32.03, half width 13.90, 37 of 40) and all records +12.97 pp (+2.05 ..
+23.44, half width 10.90, 34 of 40).

Direct T37r against T37, no frontier (pass line 4): `wo` delay -30.31 % headline (-37.11 .. -22.77, half
width 6.84) and -26.50 % all records (-32.87 .. -19.42, half width 6.44), polls +31.27 % (+27.75 .. +35.07,
half width 3.65); `wv` delay -29.39 % headline (-38.54 .. -18.05, half width 10.09) and -24.89 % all
records (-33.13 .. -15.29, half width 8.89), polls +27.89 % (+23.37 .. +32.45, half width 4.49).

So the predicted readings of the four (load, rule) rows of pass line 3 are: `wo` T37 reversed, `wv` T37
reversed, `wo` T37r not seen, `wv` T37r not seen. Those are the ticket's predicted readings as well, drawn
again from these schedules.

## Quantities

Fixed here, before the runs.

- **Headline delay.** Records whose scheduled creation time is at or before t0+72 s: the selection is
  `discovery.csv`'s `(origin, seq)` read against the experiment description's
  `schedule[host][seq-1]["at_s"] <= 72.0`. Delay is the time until the record is stored 4 hops away, mean
  over the uncensored ones. The selection is on the scheduled time rather than the real one, because a record is
  created a median 91 ms after it was due and a selection on the real time would put a different set of
  records on each side of the boundary in the testbed than in the replay. The all-record value is reported
  beside it everywhere.
- **Censored share** `c4/(n4+c4)`, reported beside every delay, on the headline set and on all records, for
  all ten stages. The sparse end of the line is scored on a different set of records from the dense end, so
  without the shares a gain cannot be told apart from a difference in who dropped out. The replay puts the
  headline shares at or below 0.044 everywhere and the all-record share of the 10.043 s arm at 0.241 (`wo`)
  and 0.266 (`wv`), which is the reason the headline exists.
- **Polls per run** counted over the whole window `[t0+10 s, t0+130 s]`.
- **The line.** The three arms of the same load and p (3.695, 6.0, 10.043 s), polls per run on the
  horizontal axis and delay on the vertical, joined piecewise in log-log. If a rule's polls per run fall
  outside the arms, no gain is computed and that is reported instead.
- **Gain** = 1 - delay(rule) / the line's delay at the rule's own polls per run.
- **Interval** = 2000 resamples of seeds 1 to 20 with `default_rng(20261117)`, each resample recomputing
  all five stages of that load from the same resampled seeds. `predict.py` draws the one index matrix of
  2000 by 20 positions once and reuses it for every experiment and both loads, so the printed bounds depend
  on the seed of the generator alone and not on the order the sections are printed in (`../16/predict.py`
  drew a generator per section instead).
- **Bootstrap unit: the seed number.** One unit carries that seed's value in all five stages of its load
  (three arms, T37, T37r), so the stages stay paired inside a resample. 20 units per load. A resample in
  which the arms are not monotone (more polls, less delay) or the rule's polls leave the arms yields no
  gain; it is counted and left out rather than replaced. (The monotone condition is not in the ticket,
  which names only the arms' range; it is evidence/17's rule, carried over because a frontier that does not
  fall cannot say what a rule's polls are worth.)
- **Words.** Entirely above zero: "seen". Spanning zero: "not seen". Entirely below zero: "reversed".

## Pass lines

### 1. The comparison across code versions holds (the gate, before the sweep)

- A generated run with `--workload w0` is byte-identical to one generated without it (test, checked at
  merge).
- In all three gate runs, every `interval_s` in the decision log is a value the setting allows:
  `s9w0_b3p695` only 3.695, `s9w0_b10p043` only 10.043, `s9w0_t37r` only 3.695, 7.39 and 10.043. Exact.
- The two fixed arms are within 5 % in polls and 20 % in delay of the same setting on seed 1 in
  `../16/perrun.csv` (`s7b3p695` and `s7b10p043`, p 0.4, seed 1). Those rows give delay 19.713 s with 255
  polls and censored share 0.000 for `s7b3p695`, and delay 36.294 s with 96 polls and censored share 0.017
  for `s7b10p043`, so the bands are:

  | gate stage | polls band (+-5 %) | delay band (+-20 %) |
  |---|---|---|
  | `s9w0_b3p695` | 242.25 .. 267.75 | 15.771 .. 23.656 s |
  | `s9w0_b10p043` | 91.200 .. 100.800 | 29.035 .. 43.553 s |

  The censored share of each gate run is reported beside the delay.
- **`s9w0_t37r` is reported only and is not part of the gate.** Whether a poll is answered or times out is
  a race on the testbed, so a rule stage does not reproduce its own numbers run to run: `s7t37r` moved
  between 166 and 192 polls across seeds in `../16/perrun.csv`, and seed 1 of it had 182 polls, delay
  21.577 s and censored share 0.033. A 5 % band on this stage would stop the 200-run plan with nothing
  broken. The numbers are reported beside the original run's.
- If any of the above fails, the 200 runs are not started and the gate is reported.

### 2. All 200 sweep runs DONE and valid

No `evalvalid` flag on any of them. Of the flags, `record_off_schedule` and `record_count` say directly
whether the on/off schedule was kept by the testbed, so the count of each, by stage, is reported with the
breakdown even when it is zero.

(Not in the ticket, taken from evidence/17's amendment 1: pass line 2 reads "DONE and valid", and every
other line uses the seeds that are DONE and unflagged in all of the stages that line names. The number of
seeds used is printed beside each verdict, and a line computed on fewer seeds than it names reads
"incomplete" in place of its verdict.)

### 3. Gain against the fixed-interval line

Computed as "Quantities" above fixes it. **The reading is attached to each of the four (load, rule) rows
separately. The four rows are not required to agree.** The gain values in the table are `predict.txt`'s.
They stand in place of the ticket's table, whose values are evidence/15's draw on another seed string; the
ticket itself says those cannot be used as a prediction for these runs. The predicted readings are the
same four words the ticket predicted.

| load | rule | replay gain and 95 % interval (headline; all records in parentheses) | predicted reading |
|---|---|---|---|
| `wo` | T37 | -17.73 %, -26.03 .. -10.01 (all: -13.16 %, -20.42 .. -6.29) | reversed |
| `wv` | T37 | -16.51 %, -27.46 .. -7.06 (all: -10.35 %, -18.65 .. -2.75) | reversed |
| `wo` | T37r | +1.64 %, -8.03 .. +9.91 (all: +1.17 %, -7.10 .. +8.73) | not seen |
| `wv` | T37r | +2.51 %, -7.90 .. +11.84 (all: +2.60 %, -4.88 .. +10.89) | not seen |

- **The two T37 rows** (predicted "reversed"). "Reversed" is the prediction, and it says that T37 loses to
  the fixed interval on that load. "Not seen" has the shape the default load's measurement had
  (evidence/16, -6.4 %, interval -14.4 .. +1.8) and says nothing at all about the rule. "Seen" is the
  opposite sign from the prediction, so the reason is investigated before any other line's conclusion is
  written.
- **The two T37r rows** (predicted "not seen"). "Not seen" is the prediction, and **even then nothing is
  claimed about the rule. It is not a demonstration that the rule adds nothing.** The interval is wider
  than the effect, so the only claim is that neither direction can be told. "Seen" means a larger effect
  than the replay predicts and the reason is investigated; "reversed" is the opposite sign from the
  prediction and is likewise investigated.
- Both scorings are reported for every row. For `wv` T37 the replay's two scorings already differ in how
  wide the interval is, so that row says what each scoring read.

### 4. T37r against T37 directly (no line), per load

The delay ratio and the polls ratio, with intervals from the same resamples, and the censored share beside
the delay. Predicted half widths, from `predict.txt`: delay 6.84 pp (`wo`) and 10.09 pp (`wv`) on the
headline scoring, polls 3.65 pp and 4.49 pp. The measured half widths are reported beside them. The
default load measured -23.0 % in delay and +26.8 % in polls, neither interval spanning zero
(`../16/after.md`). The reading is whether each of the two ratios spans zero, on each load.
**No pass or fail is attached to this line.** The direction is known from the replay; this line is the
measurement of the size.

### 5. Did the replay predict the measurement?

The **headline delay** and the **polls per run over the whole window** of all ten stages, 20 values, each
inside plus or minus 6 % of the prediction tabled above. **18 or more of the 20: "predicted". 15 to 17:
"mixed". 14 or fewer: "missed".**

- "Mixed" means: every value outside the band is reported one by one, naming the stage and the quantity,
  and the decision sequences of those stages are compared between the testbed log and the replay. Whether
  the replay can be used on this load is written only after that.
- The 6 % band comes from the largest deviation the replay showed on the default load, 5.9 %
  (evidence/13). **That band was measured on the default load and nothing says it is the right band for an
  on/off load.** So every value outside it is reported with how far outside and in which direction, even if
  the count is 18 or more.
- "Missed" triggers evidence/16's procedure: the decision sequences around timeouts and around the ON/OFF
  switches are compared between the testbed log and the replay, and where the two part is reported before
  any other line's conclusion.
- **User's decision of 2026-11-19 (ticket, 決まったこと 2): if this line reads "missed" on `wo`, the 100
  `wv` runs are still run through.** The decision logs of purpose (b) come first. The comparison of the
  disagreement is carried out while `wv` is running.
- **This line does not show that the creation times are what WO and WV are defined to be.** The prediction
  reads the creation times out of the testbed's own experiment descriptions, so a mistake in the new
  tick-laying code would be in the prediction too. What the file-against-file comparison states is that the
  two ran the same instants. That those instants have the defined shape is shown by
  `tests/test_evalgen.py`, which compares the generator against hand-written expectations (tick positions,
  the half-open block, the 110 s cut, k records per tick) and never against another generator output.
- If this line reads "predicted", the on/off numbers of evidence/14 and 15 have support for the first
  time. That is the largest thing this step buys.

## What this step cannot show

- **Whether the rules beat the fixed-interval line.** The replay puts T37r's gain at +1.64 % (`wo`) and
  +2.51 % (`wv`) on the headline scoring, and both intervals span zero. The half widths that come with
  them are 8.81 and 9.71 pp, and for T37 7.80 and 10.13 pp. An effect of a few percent is far inside
  those. On the default load the replay's half widths were 5.9 (T37) and 6.6 (T37r) pp while the testbed's
  were 8.1 and 9.1 pp (`../16/after.md`, reason never investigated), so the replay read the width about
  1.4 times too narrow there. Carrying that factor over puts the testbed's half widths here at 11 to 14
  pp, and the record count per run moves so much across seeds that they could be wider still.
- Narrowing the half width to 4 pp means about 125 seeds at 10 pp, since the width shrinks with the square
  root of the seed count. In the shape above (five stages, two loads) that is 1250 runs, about 55 h 30 min;
  one load alone is 625 runs, about 28 h.
- Which rule is best; p 0.2 or p 0; the variant that lengthens on a timeout; whether a learned decision
  maker beats a formula. Also not read: the probe numbers, whose class mix moves when generation stops.
- Every number in the prediction is simulated. The replay's agreement with a measurement has only been
  checked on the default load, which is what pass line 5 is for. A predicted band says what the replay
  expects the testbed to measure, and is itself no measurement.

## Deliverables (this directory)

`spec.md`, `predict.py`, `predict.txt`, `main9.sh`, before the runs. After the runs: the analysis script,
its csv and md output, `README.md`, `RUN.txt`.
