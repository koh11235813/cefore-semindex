# evidence/25 spec: L, M and the formula on the held-out seeds, with the five baselines re-measured

Written 2026-11-28, before any of the 280 runs below, before any run of seeds 121 to 160 on `c333655`, and
before the held-out prediction is run. (The emulator runs of `seam.md` 18 were on learning seed 1.) Ticket
`../../tickets/12-three-way-comparison.md`, its decisions 1 to 10 and the user's rulings of 2026-11-28 on the
handling of late and unanswered decisions, on the recorder and on the prediction (recorded below where each
applies). The pass lines below are translated from that ticket's Acceptance 1 to 9 and are not moved after the
prediction or after the runs. Where this file says something the ticket does not, or departs from it, it is
marked as an addition of this file.

**Order of commits, fixed here.** (1) This file, with its pass lines and its two thresholds, is committed
before the held-out prediction is run. (2) The prediction is run once (`PREDICT_RUN.md` steps 4 to 7). (3) Its
numbers are written into the last section, "Predicted values", in a separate commit; nothing above that
section moves in that commit. (4) The 280 testbed runs start only after (3) is committed.

## Question, and what the step is for

The ticket names three things this step can show ("この一手が示せること"), in this order.

1. **The testbed gain of each of the three decision makers in the main cell, with its interval.** L is
   gemma-4-E2B tuned on hindsight labels (ticket 10, candidate b), M is the model learned from logged
   randomised decisions (ticket 11, candidate a, `../24/coef_a.npy`), and the formula side is the pinned F of
   evidence/20, with T37r beside it as a second baseline (ticket 12, frame item 3).
2. **The difference between L and M**, on the same inputs, at the same instants, against the same measuring
   stick.
3. **Whether the replay's score predicted the testbed's.**

Each decision maker sends exactly one candidate to the testbed, selected in the replay on seeds taken from 1
to 100 (ticket, decisions 1 and 2). Seeds 121 to 160 are the measuring stick and have touched no learning,
labelling, selection or tuning (ticket, frame item 1 and decision 2).

**The five baselines are measured again** (ticket, decision 6). Ticket 07 measured them on `src` version
`d009cfb`; this step runs on the version that carries the decider seam, and no run has shown that the two
versions give the same testbed result. Ticket 07's values stand beside the new ones in the same tables, as
the old version's values.

Not answered here: the general ranking of the three decision makers (one cell, one load, three arms, 40
seeds); whether foresight's gap can be closed; anything at a finer grid of arms; the p 0.4 cell; the on/off
loads (all from the ticket's "示せないこと" and "やらないこと").

## Runs (`main12.sh`, from a detached clean checkout of the docs commit, whose `src` equals `c333655`)

`c333655` is the merge commit of PR #46 (parents `ade661d` and `98d63be`), which carries the decider seam of
`seam.md` and ADR 0005. The docs branch sits on top of it, so one detached checkout of the docs commit that
holds this file with its predicted values gives the merged `src` and the decider files that commit records.
`main12.sh` refuses that checkout unless it descends from `c333655`, its `src` is byte-equal to `c333655`'s
and it holds no change or untracked file (`MAIN12_RUN.md` step 1). `RUN.txt` records the checkout's commit
and that its `src` equals `c333655`, as evidence/20 recorded its commits (`../20/spec.md` "Runs").

Common settings, as evidence/20: `--p 0.2 --k 4 --store-synchronous NORMAL --workload w0
--disconnect-duration-s 8 12`, the s6 outage shape. The outage schedule depends on (p, seed) only
(`../20/spec.md:96-97`), so the seven stages are a comparison on one set of outages per seed, expected to be
the same set ticket 07's five stages ran on (expected and not checked here: the generator gate of the
prediction reads seeds 1 to 20 only). Stage prefix `s12` (ticket, decisions of 2026-11-27).

| stage | decision maker | settings beyond the common ones |
|---|---|---|
| `s12b3p695` | fixed 3.695 s | `--interval-choices 3.695 --choice-jitter 0.1` |
| `s12b6p0` | fixed 6.0 s | `--interval-choices 6.0 --choice-jitter 0.1` |
| `s12b10p043` | fixed 10.043 s | `--interval-choices 10.043 --choice-jitter 0.1` |
| `s12t37r` | rule T37r | `--poll-rule '{"i_min_s": 3.695, "factor": 2, "i_max_s": 10.043, "jitter_frac": 0.1, "on_timeout": "reset"}'` (`../20/main10.sh:26`) |
| `s12f` | the pinned F | `--poll-rule '{"i_min_s": 3.695, "factor": 1.5, "i_max_s": 10.043, "jitter_frac": 0.1, "on_timeout": "reset"}'` (`../20/main10.sh:29`, `../20/spec.md:60`) |
| `s12m` | M, `../24/coef_a.npy` through `mdecider.py` | `--interval-choices 3.695 6.0 10.043 --choice-jitter 0.1 --choice-redecide --poll-decider` naming `mdecider.py` |
| `s12l` | L, candidate b, through `ldecider.py`, answered by `lserver.py` on `hpc-debian` over the relay | `--interval-choices 3.695 6.0 10.043 --choice-jitter 0.1 --choice-redecide --poll-decider` naming `ldecider.py` (the choice, jitter and redecide flags are this file's, taken from M's row, since `--poll-decider` requires them and L must be read on M's arms) |

Seeds 121 to 160 for every stage: 7 stages of 40 runs, 280 runs at 160 s each, 280 x 160 s = 44800 s,
12 h 27 min (ticket, decision 6, line 114).

- **Seeds run outer.** One `evalrun` call per (stage, seed), so a sweep stopped part way holds every (seed,
  stage) pair of the seeds it finished and nothing has to be retaken (`../20/spec.md:109-111`). The order of
  the stages within one seed is `main12.sh`'s and `RUN.txt` records it.
- **The flag set of `s12m` and `s12l` is the one `evalrun` accepts**: `--poll-decider` requires the choices
  and `--choice-redecide` and refuses the table and the rule (`src/semindex/evalrun.py:35-38`,
  `src/semindex/gen.py:602-608`).
- **The input recorder is attached on all 80 decider runs** (`s12m` and `s12l`), writing one record file per
  node to a directory outside the run's `debug/node_dirs`, and `parity.py` runs on every one of those 80 runs
  (user, Q133, 2026-11-28). `arms.py` runs on all 280 (pass line 2). The record directory of each run is
  recorded in `RUN.txt`.
- **L's server.** `lserver.py` on `hpc-debian`'s GPU, reached from the nodes through the socket on the shared
  file system and the relay of `seam.md` section 8.4. L's answers are not waited for inside the loop (ticket,
  decision 8); an answer that has not arrived is counted (pass line 1). `seam.md` 18.4 states that the launcher
  probes the server through the forward before each run; this file requires `main12.sh` to do so (an
  addition of this file). How the server is started is `main12.sh`'s and `RUN.txt`'s.
- **What the re-measured baselines can and cannot bound.** The five baseline stages run in rule mode or
  single-choice mode, and the only line of the seam they execute is one guard that is false for all five
  (`seam.md` 13.3, "One thing the re-measured baselines cannot do"). They bound testbed drift plus commit
  `1c3f758` plus that guard between `d009cfb` and `c333655`; **they do not validate the seam.** Pass line 7
  reads this drift.
- Before the sweep, by hand: no emulator and no Cefore daemon running; the run checkout is not edited while
  runs load it.

## Prediction (`predict.py`, `heldout.py`, `lpredict.py`, run as `PREDICT_RUN.md` lists)

**Run exactly once, after this file is committed** (user, Q134, 2026-11-28): 40 replicate streams of 40 single
runs on seeds 121 to 160 for every stage. One stream is one experiment, the condition the testbed has: one run
per seed, 40 seeds. Every predicted number is the median over the 40 streams (`predict.py:26-36`). The six
CPU stages run on this machine (`predict.py --held-out-once`, about 3 min, `PREDICT_RUN.md:63`); L runs on
`hpc-debian`'s GPU (`lpredict.py --held-out-once --k 4`, about 80 min, `PREDICT_RUN.md:80`); `predict.py
--report` then joins the two and simulates no held-out seed (`PREDICT_RUN.md:99`). `predict.py --held-out-once`
and `lpredict.py --held-out-once` each refuse to redo already-finished work (`predict.py:705-707`,
`lpredict.py:15-17`); `predict.py --report` has no such guard of its own, so `PREDICT_RUN.md` runs steps 4, 6
and 7 under `set -C`, which refuses a repeated line before it can truncate `predict.txt`, `lpredict.log` or
`predict_report.txt` (`PREDICT_RUN.md:3`).

**Which engine, and why it is not evidence/20's.** All seven stages run on `../22/port.py`'s engine with the
scoring's pair streams, `pair_lag_seed = pair_jitter_seed = rep_seed("pairlag:s6b10p043", 0.2, seed, stream)`,
the union choice set, choice jitter 0.1 and redecide, which is how every candidate of L and M was scored
(`predict.py:38-49`). The five baselines are deciders on that engine, so the frontier, F, T37r, M and L are
scored on the same draws. evidence/20 predicted with evidence/18's `bound.sim` on one shared stream instead.
The two engines are bridged by gate 4 below and compared, as a printed check and not a gate, by the "against
evidence/20" table of the output.

**The gates, before any held-out number is printed** (`predict.py:51-78`): 0, the two `src/semindex` trees
byte-equal and the engine copy's five files at the scoring's digests; 1, M's pins as `mdecider.py` checks them
at a node's start; 2, the generator against the stored experiments of seeds 1 to 20; 3, the run dicts against
`label.run_dict` and the held-out substitution against evidence/20's; 4, the bridge: the five baselines on this
engine but on evidence/18's shared stream reproduce evidence/20's gate 1 to its printed digits; 5, the
scoring's own rows (`../23/eval_dry.csv`, `../23/eval.csv`, `../24/eval24_dry.csv`, `../24/eval24.csv`) through
the very path the held-out stages take. A failed gate prints the difference and no prediction. L's rows are
held to 24 check runs against the scoring's b rows and to the input, model and code digests
(`predict.py:643-676`).

**What the held-out seeds cannot supply** is carried over from evidence/20 unchanged: the outage timeline and
the record schedule are generated from (p, seed, k, workload), t0 and `agent_start` are borrowed from a donor
run of seeds 1 to 20, and on seeds 1 to 20 the generated schedules moved the delay by at most 0.3 % and a
borrowed clock by at most 1.1 % (`../20/spec.md:153-185`).

**The replay does not model answer latency.** In the replay every answer of L and M is applied at its own
resolution. On the testbed an L answer arrives after the round trip, and a late one leaves the pair's held
interval running (`seam.md` 6.5, case 2). Pass line 5 carries the consequence.

The predicted values are not in this section. They go in the last section, "Predicted values", which is
filled in the separate commit (3) above, so that nothing above it moves when they are written.

## Quantities

Fixed here, before the prediction and the runs. Carried over from `../20/spec.md` "Quantities" (lines 222-254)
except where marked.

- **Delay**: pooled `sum(lat4) / sum(n4)` over the runs of a stage, the time until a record is stored 4 hops
  away. **Headline delay**: records whose scheduled creation is at or before t0+72 s; under `w0` all 240
  records of a run are scheduled at or before t0+71 s, so the two are one number (`../20/spec.md:187-189`).
  This analysis never reads `predict_rows.csv`'s or `predict_l.csv`'s own `delay_ms` column (`heldout.py`'s
  copy of `../22/score.py`'s `delay`, which pools every censored record at its lower bound); the Delay above
  is the number it computes instead.
- **Censored share** `c4/(n4+c4)`, reported beside every delay.
- **Polls per run** over the whole window `[t0+10 s, t0+130 s]`.
- **The line**: the three fixed arms of this step (`s12b3p695`, `s12b6p0`, `s12b10p043`), polls per run
  against delay, piecewise linear in log-log, read at the rule's own polls per run. A rule whose polls fall
  outside the arms gets no gain.
- **Gain** = 1 - delay(rule) / the line's delay at the rule's own polls per run. Computed for T37r, F, M and L.
- **M minus L**: the gain of M minus the gain of L, in percentage points, on the same line and the same
  resamples. **The sign is `predict.py`'s** (`predict.py:527-528`, "M minus L"); a positive value says M's
  gain is the larger. Ticket Acceptance 7 names the difference without a sign; this file fixes it to match
  the prediction (an addition of this file).
- **Interval**: 2000 resamples of seeds 121 to 160 with `default_rng(20261128)` (`predict.py:109-110`). One
  index matrix of 2000 by 40 positions is drawn once and reused for every quantity, so the printed bounds depend
  on the generator's seed alone. evidence/20 used `default_rng(20261122)`; the seed is new because the
  generator seed is the date the script was written, by evidence/20's own convention (`../20/spec.md:240-241`).
- **Bootstrap unit: the seed number.** One unit carries that seed's value in all seven stages, so the stages
  stay paired inside a resample. 40 units.
- **Dropped resamples, per quantity (a departure from evidence/20).** A resample in which the three arms are
  not monotone (more polls, less delay) or in which a rule's polls leave the arms yields no value for the
  quantities that need that rule; it is counted and left out of those quantities only. evidence/20 dropped such
  a resample jointly from every rule. The per-quantity form is used so that M's interval is the same whether
  L's rows are present or not (`predict.py:519-523`); evidence/20 dropped 0 of 80000 resamples, where the two
  forms are one (`../20/spec.md:208-209`). The count dropped is printed per quantity.
- **Half width**: half the distance between the two ends of the 95 % percentile interval. In the prediction
  each end and the half width are medians over the streams taken separately, and the predicted word is read
  from those medians; a stream whose value reads "no line" is left out of the medians, and `predict.py` prints
  beside each word how many streams it is taken from (40 when no stream reads "no line"); the measurement is one
  experiment, so its half width is half the difference of its own two ends (`../20/spec.md:249-252`).
- **Words.** Entirely above zero: "seen". Spanning zero: "not seen". Entirely below zero: "reversed". No
  value computable: "no line" (ticket, frame item 5; `../20/spec.md:253-254`).

## Thresholds, re-set from ticket 07's measurement

The ticket starts from evidence/18's H = 6.5 points for one gain and Hd = 7.8 points for a difference of two
gains, which were evidence/16's 20-seed half widths of about 9 and about 11 points scaled by `1/sqrt(2)`
(`../18/spec.md:381-386`), and requires them to be put again from the intervals ticket 07 measured on the
testbed (ticket, frame item 6). ticket 07 measured half widths of 5.37 points for T37r's gain, 4.33 for F's
and 3.80 for F minus T37r (`../20/README.md:102`). So:

- **H = 5.37 points** for one gain: the larger of the two measured single-gain half widths (T37r's; F's was
  4.33).
- **Hd = 3.80 points** for a difference of two gains: the measured half width of F minus T37r.

**What they are for.** They are the size this step can resolve on 40 seeds: a true gain smaller than about H,
or a true difference smaller than about Hd, is not expected to give an interval clear of zero, so "not seen"
at that size is the expected reading and says nothing about absence. **The verdict words come from the
intervals and not from H or Hd.** No pass line compares a point value with H or Hd. The measured half widths of
this step are printed beside H and Hd, so a reader sees whether this step bought the resolution ticket 07 did.

**Hd is likely to understate the half width of M minus L.** 3.80 is the paired difference of two rules that
differ in one coordinate (factor 1.5 against 2, `../20/spec.md:324-325`), so most of their seed-to-seed
variation cancels in the difference. M and L differ in everything they read and answer, so less of it cancels
and the M minus L interval may be wider than Hd. The prediction prints its own half width for M minus L
("Predicted values"). In ticket 07 the replay's predicted half width for F minus T37r was 4.52 against the
measured 3.80, and the single-gain ones 5.01 and 3.92 against 5.37 and 4.33 (`../20/README.md:102`), so the
replay's half widths are a guide to within that range and not a measurement.

## Pass lines

The ticket's nine Acceptance items, and where each is read here, so that a cross reference between the frozen
files lands on the right line:

| ticket 12 Acceptance | where |
|---|---|
| 1 (the seam and the inference placement answered before any run) | answered before this file: `seam.md` (sections 2 to 11, and 18 for what the emulator runs changed) and ticket decisions 4 to 10; not a pass line here |
| 2 (every candidate scored in the replay, selection on seeds 1 to 100 only) | answered before this file: the result sections of tickets 10 and 11 (`../23/`, `../24/`); its seed half is read again in pass line 6 |
| 3 (the run's shape written in `spec.md` before the runs) | "Runs" above; the 280-run count replaces the ticket's "2 stages, 80 runs" (ticket, decision 6) |
| 4 (every run valid) | 1 |
| 5 (logged intervals are the decider's values, 3.695, 6.0, 10.043 only) | 2 |
| 6 (gains of the three decision makers, with ticket 07's values in the same table) | 3 |
| 7 (L and M's difference, against Hd) | 4 |
| 8 (did the replay predict the testbed, ±6 %) | 5 |
| 9 (seeds 121 to 160 nowhere on the learning side) | 6 |

Pass line 7, the re-measured baselines against ticket 07, has no Acceptance item. **It is an addition of this
file** (user, 2026-11-28), carries no pass or fail, and is there so that a version drift between `d009cfb`
and `c333655` is printed rather than assumed away.

### 1. All 280 runs DONE and valid, and the decider's own counts

**`evalvalid`** (Acceptance 4). No flag on any run. If a flag appears, the count by stage and by flag is
reported with the breakdown. As evidence/20's pass line 1 (`../20/spec.md:273-278`): every other line uses
the seeds that are DONE and unflagged in all seven stages, the number of seeds used is printed beside each
verdict, and a line computed on fewer seeds than it names reads "incomplete" in place of its verdict. **One
consequence is stated now**: since the bootstrap unit carries a seed through all seven stages, one flagged run
in any stage, `s12l` included, takes that seed out of every line, and every verdict then reads "incomplete",
the baselines' included (user, Q137, 2026-11-28). Whether a flagged run is retaken is asked of the user when
the flag appears, with the breakdown; this file does not decide it in advance.

**No run is excluded for its decider counts** (user, Q132, 2026-11-28). `late`, `clamped_by_wait`, `overlap`
and `unanswered` (`seam.md` 6.6) are reported per run and per stage, for `s12m` and `s12l`, together with
`deferred`, `applied` and `orphan`. Excluding runs on those counts would credit L wrongly with removing its
own latency: the runs it would drop are the ones where L's answers were slow, and those belong to L's result.

**Each unanswered resolution is classified** from the `decider_counts` row's `unanswered_asked` list
(`src/semindex/agent.py:1340-1351`, `src/semindex/engine.py:1090-1111`) against the stop instant carried by
the same row, which is that row's own `ts_ms`, stamped when `Agent.shutdown` writes it, after the one drain
and after `self._poll_decider.close()` runs (`agent.py:1334-1339`); `d` below therefore also includes whatever
`close()` itself takes (the same comparison `seam.md` 18.1 item 2 made for `t25n45`). The row carries no other
instant: its other keys are the counts (`src/semindex/engine.py:1079-1088`) and `unanswered_asked`. For each
entry, `d = ts_ms - asked_ms`.

- `d > 1125.5 ms`: **"the server did not answer"**. 1.1255 s is the smallest margin to the receive deadline,
  the 3.695 s arm resolved by a timeout, 1.2255 s, less the 0.1 s loop tick (ticket, decision 10, lines
  137-138). A resolution asked earlier than that before the stop had time to be answered by a working server.
  Any such resolution is reported beside the verdicts of pass lines 3 to 5, with its run, node, pair and `d`.
- `d <= 1125.5 ms`: **"the run ended"**. Counted only. The boundary value itself falls on this side, because the
  user's wording puts "within 1.1255 s" here (an addition of this file as to the equality case). `ts_ms` and
  `asked_ms` are both integer milliseconds (`engine.py:1106-1111`), so `d` is always an integer and the
  boundary value itself is never actually computed: the split that runs is `d >= 1126 ms` against `d <= 1125
  ms`, equivalent to the rule above.

**The 1 % note** (user, Q132). If, over the 40 `s12l` runs and all their nodes, the sum of `late` plus
`clamped_by_wait` exceeds 1 % of the sum of `deferred` (for L every resolution is deferred, so `deferred` is the
number of resolutions L was asked), that is written beside pass line 5's reading as a candidate reason for a
miss, because the replay does not model answer latency. Pooling over the stage, and `deferred` as the
denominator, are additions of this file.

**Defects of the seam are reported before any other conclusion** (`seam.md` 13.3): any `orphan` above zero, a
`parity.py` FAIL on any of the 80 decider runs, or an `arms.py` FAIL (the `arms.py` FAIL is an addition of this
file: user Q132 names orphan and parity only). A run with one is not excluded by this
file; the defect, its count and its runs are written first, and no reading of the stage that holds it is
written until the user has ruled on it (that last clause is an addition of this file). A mismatch `parity.py`
classifies as an `overlap` or `unanswered` caveat is a caveat and is reported with its count (`seam.md` 13.3,
18.2 items 2 and 3). A `(stage, seed)` retaken because its store run lacked `DONE` has its stale record
directory moved aside by `main12.sh` before the retake, never deleted, so `parity.py` and `arms.py` read only
the current attempt's directory and a defect reported here is never an artifact of the resume itself. When a
stage's reading is held this way, every line that reads that stage is held with it: pass lines 3, 4, 5 and 7,
not only the line where the defect appeared.

### 2. Every decided interval is a value the stage allows (Acceptance 5)

`arms.py --run RUN --choices ... --jitter 0.1` on all 280 runs, with `--record-dir` on the 80 decider runs.
The check is on `interval_s`, the decided value, on both send rows and reschedule rows (`seam.md` 18.2 item
4), and not on the jittered value; `--jitter 0.1` additionally requires every `applied_interval_s` to lie
inside its arm's band, which the ticket does not ask for (an addition, from `arms.py`'s own check).

| stage | allowed `interval_s` |
|---|---|
| `s12b3p695`, `s12b6p0`, `s12b10p043` | 3.695; 6.0; 10.043, one value each |
| `s12t37r` | 3.695, 7.39, 10.043 (`../20/spec.md:290-293`) |
| `s12f` | 3.695, 5.5424999999999995, 8.313749999999999, 10.043, the IEEE doubles `_advance_rule_interval` forms (`../20/spec.md:291-293`) |
| `s12m`, `s12l` | 3.695, 6.0, 10.043 |

- The ticket's "only 3.695, 6.0 and 10.043" is read for the two decider stages. The rule stages carry the sets
  evidence/20 checked, because a rule's ladder is not the three arms (an addition of this file).
- On the 80 decider runs the recorder makes the second half of `arms.py` readable: every logged interval is
  joined to what the decider returned. **On the 200 baseline runs that half is not checked**, since they have no
  decider and no recorder, and comparing the log with itself would be no check (`arms.py:323-325`).
- **`arms.py` has not yet been run on a rule-mode run.** The emulator runs of `seam.md` 18 are a fixed arm and
  deciders only. A FAIL on `s12t37r` or `s12f` is reported, and whether it comes from the engine or from the
  script's reading of rule rows is found before the line is read.
- Two preconditions `seam.md` 17.4 leaves here: the send-time answer `send_start` is `../22/port.py:45`'s
  `ARMS[1]` = 6.0, an element of the choices, which `main12.sh` asserts (`seam.md` 8.2); and the bound of
  `seam.md` section 5 item 1, the smallest applied interval above the poll timeout plus one tick, is a refusal in
  the agent's config validation (`src/semindex/agent.py:465`; `seam.md` 15, choice 19), so a stage that broke
  it would be refused at launch. For these choices the smallest applied interval is 3.695 x 0.9 = 3.3255 s
  (`seam.md` 18.3) against 2000 ms + 100 ms (`seam.md` section 3.2).

### 3. Gains of T37r, F, M and L against the three-arm fixed frontier (Acceptance 6)

On seeds 121 to 160, as "Quantities" fixes it. The reading is attached to each rule separately and the four are
not required to agree. The censored share is reported beside every delay. The table carries ticket 07's
testbed values as the old version's, in the same table (ticket, decision 6):

| rule | `src` version | gain and 95 % interval | half width | word | source |
|---|---|---|---|---|---|
| T37r | `d009cfb` (ticket 07) | +7.18 %, +1.77 .. +12.51 | 5.37 | seen | `../20/README.md:47` |
| F | `d009cfb` (ticket 07) | +4.90 %, +0.55 .. +9.21 | 4.33 | seen | `../20/README.md:48` |
| T37r | `c333655` | measured | measured | | |
| F | `c333655` | measured | measured | | |
| M | `c333655` | measured | measured | | |
| L | `c333655` | measured | measured | | |

- The old rows are copied as evidence/20 printed them, on `s10`'s own frontier and `default_rng(20261122)`;
  they are not recomputed.
- The predicted words are the prediction's only ("Predicted values").
- **"Not seen" does not show that a decision maker adds nothing**; it says only that the interval is wider
  than the effect (ticket, "やらないこと"). Where a measured word is "reversed" and the predicted word is not,
  the reason is investigated before any other line's conclusion is written (carried over from
  `../20/spec.md:309-311`).
- The three fixed arms must form a falling line in the full sample. If they do not, no gain is computed, the
  raw values are shown and every rule reads "no line", as does a rule whose polls fall outside the arms. **"No
  line" stops the reading at this step** until the user has decided what the measuring stick is (carried over
  from `../20/spec.md:312-316`).

### 4. M minus L (Acceptance 7)

The gain difference M minus L, with its interval from the same resamples, against Hd = 3.80 points, read with
the same three words: "seen" says M's gain is the larger, "reversed" says L's is. The measured half width is
printed beside Hd. As "Thresholds" says, the interval may be wider than Hd, and a "not seen" says only that.

### 5. Did the replay predict the testbed? (Acceptance 8)

The **delay** and the **polls per run over the whole window** of all seven stages, 14 values, each inside
plus or minus 6 % of the predicted median ("Predicted values"). **13 or more of the 14 inside: "predicted".
11 or 12: "mixed". 10 or fewer: "missed".**

- **The split is a carried-over convention and nothing more.** evidence/19 set 18 or more / 15 to 17 / 14 or
  fewer out of 20 by judgement, and evidence/20 scaled it to 10 values as 9 / 8 / 7 or fewer
  (`../20/spec.md:330-336`). Scaled to 14 values, 18, 15 and 14 out of 20 become 12.6, 10.5 and 9.8. Rounded
  as evidence/20 rounded (the "predicted" bound up to the next whole count, the "mixed" lower bound up, and
  every count below it "missed"; evidence/20 turned 9 and 7.5 into 9 and 8), that gives 13 / 11 to 12 / 10 or
  fewer. The rounding rule is an addition of this file. **A count of exactly 10 falls between the scaled
  bounds 9.8 and 10.5 and is put on the "missed" side** (user, Q136, 2026-11-28). evidence/20's scaled bounds had no such gap (its 7 was
  the "missed" bound itself, `../20/spec.md:334-335`), so where 10 goes is a choice of this file.
- **Three of the 14 values are near-tautological.** A fixed arm's poll count follows from its interval and the
  window length, so the polls of `s12b3p695`, `s12b6p0` and `s12b10p043` are expected inside the band whatever
  the replay does. The 11 values that can inform are the seven delays and the poll counts of T37r, F, M and L.
  The verdict is reported for the 14 and the breakdown for the 11.
- **The band is a Mode B band**, per-stage aggregates, and says nothing about agreement decision by decision
  (ticket, Acceptance 8; evidence/13's Mode A failed). It comes from the largest deviation evidence/13 showed on
  the default load, 5.9 % (`../20/README.md:107`), set on other cells; in ticket 07 all ten values were inside
  it and the largest deviation was `s10f`'s delay, +2.42 % (`../20/README.md:75`). Every value outside the band
  is reported with how far outside and in which direction, whatever the count is.
- **For `s12l`, the 1 % note of pass line 1 is written beside this reading** when it applies, as a candidate
  reason for a miss on L: the replay applies every L answer at its resolution, and the testbed does not.
- "Missed" triggers evidence/16's procedure (`../20/spec.md:348-350`): the decision sequences around timeouts
  are compared between the testbed log and the replay, and where the two part is reported before any other
  line's conclusion. A value outside the band is also read against the substitution's measured cost (generated
  schedules at most 0.3 %, a borrowed clock at most 1.1 % in delay, `../20/spec.md:181-185`).

### 6. Seeds 121 to 160 nowhere on the learning side (Acceptance 9)

Shown by tracing the scripts of tickets 09, 10 and 11, as the ticket asks:

- **The learning side's gate.** `../22/port.py`'s `check_seeds` refuses every seed above `MAX_SEED = 100`
  (`../22/port.py:37`, `:93-100`), and `port.Engine`'s constructor refuses a run dict whose seed is above 100
  (`heldout.py:108-109`); ticket 10's result records that every simulation of evidence/23 goes through
  `port.sim` (`../../tickets/10-llm-system-one.md:200`). The scripts of `../22/`, `../23/` and `../24/`
  call it; the narrower per-script ranges and the outputs that show the gate refusing are those the results of
  ticket 10 (`../../tickets/10-llm-system-one.md:200`: tuning 1 to 70, 71 to 80, selection 81 to 100) and ticket
  11 (`../../tickets/11-ml-from-logs.md:130`: learning seeds up to 80, `port.check_seeds` above 100) already
  record, and ticket 09's results record seeds 1 to 100 only (`../../tickets/09-replay-labels.md:126`). This
  line lists those places and does not run them again.
- **The single armed door.** `heldout.HeldOutEngine` is the only door past `port.Engine`'s seed guard
  (`../22/port.py:169`), and it opens only after `heldout.arm_held_out()` has run in the same process
  (`heldout.py:92-124`). It is not the first use of these seeds in a replay: ticket 07's prediction
  (`../20/predict.py`) simulated the five baselines on them outside `check_seeds` (ticket 12, line 163), and
  that use was baselines only. `arm_held_out` is called in
  `predict.py --held-out-once` (`predict.py:762`) and in `lpredict.py --held-out-once`, parent and workers
  (`lpredict.py:155-156`, `:323`), and nowhere else; `predict.py`'s mutation checks show `port.Engine` refusing seed 121
  and the door refusing seed 121 unarmed and seed 101 (`predict.py:457-464`). The delivered form of this line
  is a listing of every call of `arm_held_out`, every construction of `HeldOutEngine` and every construction
  of evidence/18's unguarded base engine (`B.Engine`, `bound.sim`) under `evidence/`, taken at the commit the
  prediction ran from (an addition of this file). A listing by name cannot detect a held-out seed reaching an
  engine through a path it does not name.
- What this line cannot show is carried in "What this step cannot show".

### 7. The re-measured baselines against ticket 07 (no pass or fail; an addition of this file)

For each of the five baseline stages, the paired relative difference of the new version against the old one,
`delay(s12) / delay(s10) - 1` and `polls(s12) / polls(s10) - 1`, with a 95 % interval from the same index
matrix, the seed as the unit carrying its `s10` and `s12` values together. The `s10` per-run values are read
from `../20/perrun.csv`, never from the store. The reading is whether each interval spans zero. Ticket 07's
values sit in the same table as the old version's (ticket, decision 6):

| stage | `d009cfb` delay s / polls (`../20/README.md:58-62`) | `c333655` delay s / polls | delay difference, interval | polls difference, interval | spans zero |
|---|---|---|---|---|---|
| `b3p695` | 13.783 / 256.48 | measured | | | |
| `b6p0` | 20.335 / 159.82 | measured | | | |
| `b10p043` | 30.140 / 96.42 | measured | | | |
| `t37r` | 18.343 / 165.47 | measured | | | |
| `f` | 17.460 / 180.97 | measured | | | |

- The relative form, rather than a difference in seconds and polls, is a choice of this file, made so the line
  reads on the same scale as evidence/20's pass line 3 and the 6 % band.
- **An interval clear of zero cannot be attributed from this line alone** to the code (`1c3f758` and the
  guard) or to the testbed (runs on a different day). It says the two versions' baselines differ by
  that much, and it is reported beside pass lines 3 and 5, since both read the new baselines. An interval that
  spans zero does not show the versions equal.

## What this step cannot show

- **A general ranking of the three decision makers.** One cell, one load, three arms, 40 seeds (ticket).
- **That "not seen" means no gain.** It means the interval is wider than the effect (ticket).
- **Whether foresight's gap closes.** In the replay `V_at_F - F` is +70.1 points (+63.9 .. +76.1), and that F
  is `F_reach` (+8.9), not `F_restricted` (+10.7) (ticket, lines 25-27); V knows the record and outage times.
- **Anything at a finer grid than three arms.**
- **Agreement decision by decision.** The 6 % band is Mode B only (pass line 5).
- **That the seam is right because the baselines agree.** The baselines do not run the seam (`seam.md` 13.3).
  What checks the seam here is `parity.py` and `arms.py` on the 80 decider runs, within the limits their
  docstrings state (`parity.py:10-16`).
- **That every pair kept being polled to the end of the run, that the interval a late row says was held is the
  one that ran, and anything about a log and a record altered consistently with each other.** Neither check
  reads these (`seam.md` 18.4, "Not shown by either check"). This file does not add a check for the first; it
  is listed so that a pass of pass lines 1 and 2 is not read as covering it (an addition of this file).
- **How the replay would score L with its latency.** The replay applies every answer at once; pass line 1's
  counts and the 1 % note are the only measure of what latency did on the testbed.
- **What the borrowed exogenous half costs on seeds 121 to 160.** It was measured on seeds 1 to 20 only
  (`../20/spec.md:153-185`).
- **The selection bias of the candidates is corrected only for the one candidate each.** M's a and b, and L's
  candidates, were compared and chosen on seeds 81 to 100 (`../../tickets/11-ml-from-logs.md:153`); this step
  reads the chosen one on new seeds and says nothing about the ones not sent.
- **The rule on the held-out inputs is not enforced by code on the authors.** Since ticket 07 ran, the outage
  timelines, t0, `agent_start` and topology of seeds 121 to 160 sit in the store, readable by anyone
  (`../20/spec.md:376-380`); pass line 6 shows the scripts' gates, not what a person read.

## What `seam.md` left to this file, and where it is answered

| `seam.md` | left to `spec.md` | here |
|---|---|---|
| 17.1 | the share of unanswered, late or overlapping decisions at which a run is excluded | none is excluded (pass line 1, user Q132) |
| 17.2 | parity per stage or per run | every one of the 80 decider runs (user Q133) |
| 17.3 | which venv the server runs under | a measurement of `hpc-debian`, recorded by `RUN.txt`; not this file's |
| 17.4 | stage names, run counts, choice set, two preconditions | "Runs" (ticket decision 1); preconditions in pass line 2 |
| 17.5, 17.6 | ADR 0005; refusal or note for the bound of section 5 item 1 | outside this file; this file takes `c333655` as it stands and adds no precondition of its own |
| 17.7 | stderr into the agent log; a node's view of `/tmp`; `hpc-debian` | not needed by any line here; the socket is on the shared file system (`seam.md` 18.2 item 7), the server was measured in 18.3 |
| 18.1 item 2 | where "the run ended" and "the server did not answer" part | 1.1255 s (pass line 1) |
| 18.3, last paragraph | what a run with `late` or `clamped_by_wait` does to the comparison | counted and reported, the 1 % note (pass line 1) |
| 18.4, last paragraph | whether "every pair kept being polled to the end" is needed | not added; listed in "What this step cannot show" |

## Deliverables (this directory)

Before the prediction: `spec.md`, `predict.py`, `heldout.py`, `lpredict.py`, `PREDICT_RUN.md`, and the seam
scripts already committed (`mdecider.py`, `ldecider.py`, `lserver.py`, `lrun.sh`, `parity.py`, `arms.py`,
`fakes.py`).
After the prediction and before the runs: `predict.txt`, `predict_report.txt`, `predict_runs.json`,
`predict_rows.csv`, `predict_l.csv`, `lpredict.log`, and the "Predicted values" section below filled in.
Before the prediction: `main12.sh` and `MAIN12_RUN.md` as well. After the runs: the analysis script, its csv and md output, the `parity.py`
and `arms.py` outputs of every run they ran on, `README.md`, `RUN.txt`.

## Predicted values

Filled from the single held-out run in commit (3) of the order above. The six CPU stages' rows come from
`predict.txt` (`PREDICT_RUN.md` step 4) and the full seven-stage tables, L and M minus L included, from
`predict_report.txt` (step 7), which re-reads the same rows; the placeholder below names `predict.txt` for
both.

**Per stage: delay, polls per run, each with its ±6 % band, and the censored share.**

(filled from predict.txt after the single held-out run; nothing above this line moves)

**Gains against the three-arm frontier, and M minus L: gain, 95 % interval, half width, streams seen / not
seen / reversed / no line, predicted word, dropped resamples.**

(filled from predict.txt after the single held-out run; nothing above this line moves)

**The five baselines against evidence/20's prediction for the same seeds (a printed check, not a gate).**

(filled from predict.txt after the single held-out run; nothing above this line moves)
