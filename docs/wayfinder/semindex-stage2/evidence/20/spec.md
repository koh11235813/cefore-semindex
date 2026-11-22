# evidence/20 spec: the baseline stage on the held-out seeds

Written 2026-11-22, before any run of this step. Ticket `../../tickets/07-baseline-heldout.md`, accepted by
the user the same day (its 決まったこと section). The pass lines below are translated from that ticket's
Acceptance and are not moved after the runs. The pinned formula F, the stage settings and every predicted
number are frozen here before the first run. Where this file says something the ticket does not, or departs
from it, it is marked as such.

## Question, and what the step is for

The ticket names three purposes, in this order.

**(a) The measuring stick of the whole three-way comparison, laid on seeds no learning touches.** The gain of
a decision maker is read against the line through the fixed-interval arms. A line drawn on the seeds that
trained the decision maker would dirty the measuring stick itself, so seeds 121 to 160 are reserved for
measuring and are never used for labels, training, model selection or tuning. 121 is the first free number:
the store's largest seed is 100 and evidence/10's replay tuning used generated schedules of seeds 21 to 120.

**(b) The exogenous inputs the final replay scoring of the learned decision makers will borrow.** Candidates
for decision maker L (a gemma-4-E2B tuned on hindsight labels) and decision maker M (a model learned from
logged randomised decisions) are screened and selected in the replay on seeds taken from 1 to 100, never on
these seeds. The one candidate each decision maker sends to the testbed is then scored once in the replay on
seeds 121 to 160, as the prediction written before its testbed runs (ticket 12). If the outage timeline, t0,
`agent_start` and the topology that scoring borrows come from 200 runs that actually happened, the prediction
and the testbed run stand on the same inputs.

**(c) The first testbed measurement of the 6.0 s arm, of T37r and of F at p 0.2.** evidence/13 held the
replay against the testbed in this cell for the fixed 0.5, 1.359, 3.695 and 10.043 s arms and the `short`
and `wait` policies only; its Mode B per-stage error there ran -2.38 to +0.46 % (`../13/validate_cells.csv`).
**The 6.0 s arm, T37, T37r and F have no testbed number in this cell at all.** Pass line 4 reads this.

The nine hours of testbed time are spent mainly on (a) and (b). (c) is what makes the step a check and not
only a baseline.

Not answered here: which of the three decision makers is better, since this step runs none of them; the p 0.4
cell, which evidence/18 read as undecidable (F +0.4 %, interval -8.7 .. +8.7); p 0.5, p 0 and the no-outage
cell; T37, which evidence/19 read as "reversed" on both on/off loads; the four fixed arms below 3.695 s;
whether the search that found F's family is a good search, since the family is evidence/18's own narrowed and
is neither widened nor validated here.

## The pinned F

evidence/18 reports F as a cross-fitted procedure and not as one rule: in this cell fold A (seeds 1 to 10)
and fold B (seeds 11 to 20) selected different members (`../18/bound.md`). A testbed stage launches one
`--poll-rule` string, so the member has to be fixed before any run. `fpin.py` fixed it and `fpin.txt` is its
output.

- **How.** evidence/18's own coordinate ascent (3 sweeps, ties to the earliest value, start T37r), re-run on
  **seeds 1 to 20 pooled as one set**, with no fold split, over the sub-family `src/semindex` can express:
  `i_min_s` and `i_max_s` on evidence/18's arm grid inside [3.695, 10.043] s with `i_max >= i_min`, `factor`
  in {1.2, 1.5, 2.0, 3.0}, `jitter_frac` 0.1, `on_timeout` keep or reset, no `k_cap`. 48 members, 11 visited
  by the ascent; every one of the 48 was also scored and the ascent's winner is the best of them on those 20
  seeds.
- **The winner.**

  | i_min_s | factor | i_max_s | jitter_frac | on_timeout |
  |---|---|---|---|---|
  | 3.695 | 1.5 | 10.043 | 0.1 | reset |

  `--poll-rule '{"i_min_s": 3.695, "factor": 1.5, "i_max_s": 10.043, "jitter_frac": 0.1, "on_timeout": "reset"}'`

- **On which seeds, and what its gain there is.** Seeds 1 to 20 chose this member and seeds 1 to 20 scored it:
  `fpin.txt` puts it at +11.1 (+5.8 .. +15.5) against the seven-arm frontier, delay 16665.0 ms, 182.07 polls
  per run. **That gain is in sample. It is not an estimate of this member's gain on new seeds, and its
  bootstrap interval does not make it one.** The prediction for seeds 121 to 160 is drawn again in
  `predict.txt` and is a different number against a different (three-arm) frontier.
- **It is evidence/18's fold B member, not a new rule.** The ticket predicted the pinned member would differ
  from evidence/18's F. It does not: the ascent landed on fold B's winner exactly, with the same three
  numbers. Fold A's winner (i_min 6.0 with a reset target of 3.695) is outside the family at all, because
  `engine._advance_rule_interval` resets to `i_min_s` and nowhere else. evidence/18's +10.7 % is still no
  prediction for this one rule on held-out seeds.
- **Why the floor is 3.695 s.** The decision this plan compares three makers on is the next state-poll
  interval over 3.695 to 10.043 s (`../../map.md`, the 2026-11-22 section on the three-way comparison), so no
  coordinate of F may leave that band and the floor is the band's lower end. The ticket's "何を足すか" section
  carries the same floor. In this cell `../18/bound.md`'s F table gives `F_open` (floor 0) and `F_restricted`
  (floor 2.2) the same two fold members, `i_min` 6.0 for fold A and 3.695 for fold B, so neither of
  evidence/18's own floors bound here either. **What the tighter floor costs was measured**: `fpin.py 2.2`, whose
  output is kept here as `fpin_floor22.txt`, grows the family from 48 members to 80
  and the ascent's path from 11 members to 13, lands on the same winner, and still reports that winner as the
  best of the 80 on these 20 seeds. The reason evidence/18 had a floor at all is that its 6 % replay fidelity
  transfers only to members whose intervals stay at or above 2.2 s (`../18/bound.md` line 5).
- **The empirical claim the family rests on.** `i_timeout == i_min` in evidence/18's member space and
  `on_timeout: reset` in evidence/15's rule space are different code paths. `fpin.py` ran T37r through both
  and got the same row, so the family does map onto `--poll-rule`.
- **One difference between the replay and the testbed, found by reading the two.** A reply that arrives and
  fails to decode hands the testbed `empty is None`, and the testbed then leaves the interval where it is,
  while the replay's `if empty` drops the interval to `i_min`. The replay never produces that case
  (`../13/replay.py`'s `no_new` returns a bool for every simulated reply), so no number printed here moves.
  On the testbed a decode failure would leave a backed-off interval backed off, so the rule would poll later
  than the replay says. `empty is None` is a reply that could not be read, and
  the replay's falsy `if empty` is what lumps it together with a reply that brought something new.

## Runs (`main10.sh`, from a detached clean checkout of main `d009cfb` at `/home/lab_shared/wt/semindex-main10`)

Common settings, as the s6 stages and evidence/19: `--p 0.2 --k 4 --store-synchronous NORMAL --workload w0
--disconnect-duration-s 8 12`. The outage schedule depends on (p, seed) only, so the five stages are a
comparison on one set of outages per seed. Stage prefix `s10`; `s9` belongs to evidence/19.

| order | stage | decision maker | seeds | runs | time |
|---|---|---|---|---|---|
| 1 | `s10b3p695` | fixed 3.695 s, jitter 0.1 | 121 to 160 | 40 | 1 h 47 min |
| 2 | `s10b6p0` | fixed 6.0 s, jitter 0.1 | 121 to 160 | 40 | 1 h 47 min |
| 3 | `s10b10p043` | fixed 10.043 s, jitter 0.1 | 121 to 160 | 40 | 1 h 47 min |
| 4 | `s10t37r` | rule T37r (`i_min` 3.695, factor 2, `i_max` 10.043, reset) | 121 to 160 | 40 | 1 h 47 min |
| 5 | `s10f` | the pinned F | 121 to 160 | 40 | 1 h 47 min |

200 runs at 160 s each (evidence/19's measured 159.5 s per run), about 8 h 53 min.

- Seeds run outer: for each seed, the five stages. A sweep stopped part way then holds every (seed, stage)
  pair of the seeds it finished, so nothing has to be retaken. `main10.sh` therefore makes one `evalrun` call
  per (stage, seed) rather than one call per stage with the whole seed range.
- **No gate stage, unlike `main9.sh`.** A gate catches a code version that moved between two stages that will
  be compared. `git -C /home/lab_shared/wt/semindex-plan3 diff --stat 03f5ecc d009cfb -- src tests` prints
  nothing: 03f5ecc is the commit evidence/19 ran from (`../19/RUN.txt`) and `d009cfb` is the commit this sweep
  runs from, so `src` and `tests` are the same version in both and no gate rerun is needed. That command and
  its empty output are the evidence.
- **This step changes no source, so it runs from main rather than waiting for a merge.** `spec.md`,
  `main10.sh`, `fpin.py`, `fpin.txt`, `fpin_floor22.txt`, `predict.py` and `predict.txt` are committed and pushed on the docs
  branch first, the run checkout copies only the launcher, and `RUN.txt` records both the `src` commit and the
  launcher's commit.
- Before the sweep, by hand: no emulator and no Cefore daemon running. The checkout is not edited while runs
  load it.

## Prediction (`predict.py`, output `predict.txt`, both placed here before the runs)

evidence/18's engine (`../18/bound.py`'s `sim`, which subclasses evidence/15's and evidence/13's), 40
replicate streams of 40 single runs. One stream is one experiment, the condition the testbed has: one run per
seed, 40 seeds. Every number is the median over the 40 streams.

**What the held-out seeds cannot supply.** evidence/13's run dict carries four exogenous inputs: the outage
timeline, the record creation stamps, t0 and the per-node `agent_start`. No run of seeds 121 to 160 exists,
so none of the four can be read off a run. The outage timeline and the record schedule are drawn from this
worktree's own `evalgen.build_experiment`, which depends on (p, seed, k, workload) alone; t0 and
`agent_start` are borrowed from the logged `s6b10p043` p 0.2 run of donor seed `1 + (seed - 121) % 20`, so
each donor's clock carries two held-out seeds. `build_experiment` is used in place of `generate_run` because
`generate_run` writes a ReCefore yaml and a per-node config beside each description and those configs carry
key material.

**Two gates, before any predicted number is printed.**

1. evidence/18's `bound_cells.csv`, cell `s6b10p043` p 0.2 W0, seeds 1 to 20: the early-record delay and
   polls per run of the three arms and of T37r, to the digits that csv prints, and the pinned F member's
   delay and polls to the digits `fpin.txt` prints. All five matched. This says the engine, the frontier
   inputs and the replicate stream are the ones F was pinned with.
2. `build_experiment`'s outage events and record schedule against the stored `*.experiment.json` of the same
   20 runs. Every event and every schedule equal. evidence/19 compared a written file against the stored
   file; here the comparison is dict against stored file, for the key-material reason above. **This does not
   state that the instants have the shape `w0` and the outage shape are defined to have**;
   `tests/test_evalgen.py` answers that against hand-written expectations.

A mismatch in either gate prints the difference and no prediction.

**What the borrowed exogenous half costs, measured on seeds 1 to 20 at p 0.2 and not corrected for.**

| what | measured |
|---|---|
| realized outage instant against the scheduled one (`results.json` `actual_at` against the description's `at`) | at most 27 ms over every link event of the 20 runs |
| `agent_start` minus t0, over 20 donors and 5 nodes | 382 .. 2090 ms, median 421 |
| the agent's creation lag, logged stamp minus scheduled instant, over 4800 records | median 52 ms, 5th to 95th percentile 5 .. 96 |

The held-out timeline is therefore the scheduled one, and the replay creates a record at exactly the
scheduled instant while the testbed's agent creates it a little later.

**What the substitution moves, measured on the same seeds 1 to 20.** A held-out run carries two replacements
at once: generated schedules in place of logged ones, and a borrowed clock in place of the run's own. The five
stages were therefore scored three times on seeds 1 to 20, so that each replacement is bounded on its own.
Every number in the table is the median over the 40 replicate streams, as `predict.txt`'s own caption of it
says.
The third column takes t0 and `agent_start` from seed `1 + seed mod 20`, a clock as unrelated to the run's
schedules as a held-out seed's borrowed clock is to its own.

| stage | logged, delay s / polls | generated schedules, own clock | and a borrowed clock | schedules move | clock moves |
|---|---|---|---|---|---|
| `s10b3p695` | 13.899 / 256.10 | 13.892 / 256.10 | 14.043 / 256.05 | delay -0.1 %, polls +0.0 % | delay +1.1 %, polls -0.0 % |
| `s10b6p0` | 21.076 / 159.25 | 21.024 / 159.25 | 20.851 / 159.20 | delay -0.3 %, polls +0.0 % | delay -0.8 %, polls -0.0 % |
| `s10b10p043` | 30.821 / 95.80 | 30.802 / 95.80 | 30.627 / 95.90 | delay -0.1 %, polls +0.0 % | delay -0.6 %, polls +0.1 % |
| `s10t37r` | 18.909 / 166.35 | 18.938 / 166.25 | 18.888 / 166.53 | delay +0.2 %, polls -0.1 % | delay -0.3 %, polls +0.2 % |
| `s10f` | 16.689 / 182.70 | 16.664 / 182.70 | 16.605 / 182.35 | delay -0.1 %, polls +0.0 % | delay -0.4 %, polls -0.2 % |

This is not a gate: it cannot be exact, since the generated stamps carry no agent lag and the generated
timeline is the scheduled one. What it says is that on these 20 seeds the generated schedules move the delay
by at most 0.3 % and the polls by at most 0.1 %, and a borrowed clock moves the delay by at most 1.1 % and
the polls by at most 0.2 %. Both are well inside pass line 4's 6 % band. **The two columns are measured
separately and are not added here**, so what a held-out run's two replacements do together is bounded only
loosely by their sum, 1.4 % in delay.

**The predicted values.** Every record of a `w0` run is scheduled at or before t0+71 s, so the headline set
and the all-record set are the same 240 records and the two scorings print the same digits; `predict.py`
checks that equality on every stage and every stream and prints one delay per stage.

| stage | delay s (band) | polls per run (band) | censored share |
|---|---|---|---|
| `s10b3p695` | 13.908 (13.074 .. 14.742) | 256.68 (241.27 .. 272.08) | 0.000 |
| `s10b6p0` | 20.386 (19.162 .. 21.609) | 159.75 (150.16 .. 169.34) | 0.000 |
| `s10b10p043` | 30.437 (28.611 .. 32.264) | 96.08 (90.31 .. 101.84) | 0.000 |
| `s10t37r` | 18.517 (17.406 .. 19.628) | 165.01 (155.11 .. 174.91) | 0.000 |
| `s10f` | 17.047 (16.024 .. 18.070) | 180.95 (170.09 .. 191.81) | 0.000 |

Gain against the three-arm log-log frontier, with the 95 % seed-bootstrap interval, that interval's half
width, and how many of the 40 streams put the interval entirely above zero ("seen"), across zero ("not seen")
and entirely below zero ("reversed"):

| rule | gain % | 95 % interval | half width pp | seen / not seen / reversed | predicted word |
|---|---|---|---|---|---|
| T37r | +7.17 | +2.28 .. +12.00 | 5.01 | 36 / 4 / 0 | **seen** |
| the pinned F | +7.80 | +3.95 .. +11.76 | 3.92 | 40 / 0 / 0 | **seen** |

F minus T37r on the same frontier: +0.72 pp (-3.91 .. +4.83), half width 4.52 pp, 0 of 40 streams entirely
above zero. 0 of 80000 resamples had an unreadable frontier.

F against T37r directly, no frontier: delay -7.91 % (-12.21 .. -3.32, half width 4.54 pp), polls +9.81 %
(+8.95 .. +10.58, half width 0.80 pp).

**So the predicted readings of pass line 2 are "seen" for T37r and "seen" for the pinned F.** Those are not
the ticket's words: the ticket left the readings to be drawn here, and said explicitly that evidence/18's
+10.7 % (+4.8 .. +16.4) is a different member on different seeds and cannot stand as this step's prediction.
**The pinned F's +7.80 % here is a three-arm gain on seeds F has never seen, and is not `fpin.txt`'s +11.1,
which is a seven-arm gain in sample on the 20 seeds that chose the member.**

## Quantities

Fixed here, before the runs. Where a definition comes from is named.

- **Delay** (evidence/17 `spec.md` "Quantities", itself evidence/16's pass line 2): pooled
  `sum(lat4) / sum(n4)` over the runs of a stage, the time until a record is stored 4 hops away. Under `w0`
  every seed contributes the same 240 records, so pooling over records and averaging over seeds coincide.
- **Headline delay** (evidence/19 `spec.md` "Quantities", from evidence/14's scorer): records whose scheduled
  creation is at or before t0+72 s. Under `w0` that selects all 240 records, so the headline delay and the
  pooled delay above are one number. Both are reported anyway, since the later cells of this plan will not
  have that property.
- **Censored share** `c4/(n4+c4)` (evidence/19 `spec.md`), reported beside every delay for all five stages.
  The sparse end of the line is scored on a different set of records from the dense end, so without the
  shares a gain cannot be told apart from a difference in who dropped out.
- **Polls per run** counted over the whole window `[t0+10 s, t0+130 s]` (evidence/17 `spec.md`).
- **The line** (evidence/17 `spec.md`, "Three-arm frontier"): the three arms of the same cell (3.695, 6.0,
  10.043 s), polls per run on the horizontal axis and delay on the vertical, piecewise linear in log-log,
  read at the rule's own polls per run. If a rule's polls fall outside the arms, no gain is computed and that
  is reported instead. The code is evidence/19 `predict.py`'s `interp`.
- **Gain** = 1 - delay(rule) / the line's delay at the rule's own polls per run (evidence/17 `spec.md`).
- **Interval**: 2000 resamples of seeds 121 to 160 with `default_rng(20261122)`, the seed being the date this
  spec was written, as evidence/19 used `default_rng(20261117)`. One index matrix of 2000 by 40 positions is
  drawn once and reused for every stream and every quantity, so the printed bounds depend on the generator's
  seed alone and not on the order the sections are printed in.
- **Bootstrap unit: the seed number.** One unit carries that seed's value in all five stages, so the stages
  stay paired inside a resample. 40 units. A resample in which the three arms are not monotone (more polls,
  less delay) or in which a rule's polls leave the arms yields no gain; it is counted and left out rather
  than replaced. (The monotone condition is evidence/17's, carried over because a frontier that does not fall
  cannot say what a rule's polls are worth.)
- **Half width**: half the distance between the two ends of the 95 % percentile interval. In `predict.txt`
  the lower end, the upper end and the half width are each the median over the 40 streams, taken separately,
  so a printed half width is not half the difference of the two printed ends. The measurement is one
  experiment, so its half width is half the difference of its own two ends.
- **Words.** Entirely above zero: "seen". Spanning zero: "not seen". Entirely below zero: "reversed". No gain
  computable (pass line 2 says when): "no line".

## Pass lines

The ticket's six Acceptance items are folded into the four lines below. The mapping, so that a cross
reference between the two frozen files lands on the right line:

| ticket 07 Acceptance | pass line here |
|---|---|
| 1 (F pinned to one rule) | 1, the first of the two exactly checked things reported with it |
| 2 (all 200 runs valid) | 1 |
| 3 (every `interval_s` as configured) | 1, the second exactly checked thing |
| 4 (the three fixed arms form a falling line) | 2, the monotone condition inside it |
| 5 (gain of T37r and of F against the line) | 2 |
| 6 (did the replay predict the measurement) | 4 |

Pass line 3, F against T37r directly, has no Acceptance item in the ticket. **It is an addition of this file**,
carries no pass or fail, and is here because the size of the difference is worth printing.

### 1. All 200 runs DONE and valid

No `evalvalid` flag on any of them. If a flag appears, the count by stage and by flag is reported with the
breakdown. (Taken from evidence/17's amendment 1 and evidence/19's pass line 2: every other line uses the
seeds that are DONE and unflagged in all five stages, the number of seeds used is printed beside each verdict,
and a line computed on fewer seeds than it names reads "incomplete" in place of its verdict.)

Two things are checked exactly and reported with this line, from the ticket's Acceptance 1 and 3.

- The pinned F's five values stand in "The pinned F" above and are the JSON `main10.sh` launches. Both
  validators on the `--poll-rule` path accept it and T37r, checked by calling them: `agent._validate_poll_rule`
  returns `StatePollRule(i_min_s=3.695, factor=1.5, i_max_s=10.043, jitter_frac=0.1, on_timeout='reset')` and
  `gen.parse_state_poll_rule` returns the same five fields. The bounds they enforce are `i_min_s` > 0,
  `factor` >= 1, `i_max_s` >= `i_min_s`, `jitter_frac` in [0, 1) and `on_timeout` in {keep, reset}; there is no
  `k_cap` key at all. `i_min` is at or above 3.695 s, the floor the ticket and this file both set. `fpin.txt`
  also carries the member each of the three ascent sweeps ended on and which coordinate moved in that sweep,
  which is the rest of what the ticket's Acceptance 1 asks the script's output to keep.
- Every `interval_s` in a decision log is a value the setting allows. `s10b3p695` only 3.695; `s10b6p0` only
  6.0; `s10b10p043` only 10.043; `s10t37r` only 3.695, 7.39 and 10.043; `s10f` only 3.695, 5.5425, 8.31375
  and 10.043. The last two sets are the products `_advance_rule_interval` forms, `min(interval * factor,
  i_max_s)` from `i_min_s`, in IEEE doubles 3.695, 5.5424999999999995, 8.313749999999999, 10.043.

### 2. Gain of T37r and of the pinned F against the three-arm fixed frontier

On seeds 121 to 160, computed as "Quantities" above fixes it: bootstrap of 2000 resamples with
`default_rng(20261122)`, the unit the seed number, 40 units, one unit carrying that seed's value in all five
stages, non-monotone resamples counted and left out. The reading is attached to each rule separately and the
two are not required to agree.

| rule | predicted gain and 95 % interval | predicted word |
|---|---|---|
| T37r | +7.17 %, +2.28 .. +12.00 | seen |
| the pinned F | +7.80 %, +3.95 .. +11.76 | seen |

- The predicted words are `predict.txt`'s, which is the only place they come from. Reading them off
  evidence/18's cross-fitted +10.7 % would be reading another member on other seeds.
- "Seen" is the prediction for both. **"Not seen" would not show that the rule adds nothing**; it would say
  only that the interval is wider than the effect. "Reversed" is the opposite sign from the prediction, and
  the reason is investigated before any other line's conclusion is written.
- The three fixed arms must form a falling line (more polls, less delay) in the full sample. If they do not,
  no gain is computed, the raw values are shown and that is reported instead. The verdict slot of this line
  then reads "no line" for both rules, as it does for a rule whose polls per run fall outside the arms.
  **"No line" stops the plan at this step**: ticket 12 does not start its comparison on these seeds before
  the user has decided what the measuring stick is.
- The censored share is reported beside every delay.

### 3. The pinned F against T37r directly

The delay ratio and the polls ratio with intervals from the same resamples, and the censored share beside the
delay. Predicted from `predict.txt`: delay -7.91 % (-12.21 .. -3.32, half width 4.54 pp) and polls +9.81 %
(+8.95 .. +10.58, half width 0.80 pp). The measured half widths are reported beside them. The reading is
whether each of the two ratios spans zero. **No pass or fail is attached to this line.** The two rules differ
by one coordinate (factor 1.5 against 2), so this is the measurement of a size, not a test of a direction.

### 4. Did the replay predict the measurement?

The **delay** and the **polls per run over the whole window** of all five stages, ten values, each inside plus
or minus 6 % of the prediction tabled above. **9 or more of the 10 inside: "predicted". Exactly 8: "mixed".
7 or fewer: "missed".**

- **The ticket's Acceptance 6 carries this same split.** Both files scale evidence/19's 18 or more / 15 to 17 /
  14 or fewer out of 20 down to 10 values, and halving those bounds gives 9 / 7.5 / 7, which puts a count of 7
  on the "missed" side. **Neither split is backed by a measurement**: evidence/19's own bounds were set by
  judgement, so this one is a carried-over convention and nothing more.
- **Three of the ten values are near-tautological.** A fixed arm's poll count follows from its interval and
  the window length, so `s10b3p695`, `s10b6p0` and `s10b10p043` polls are expected inside the band whatever
  the replay does. The seven values that can inform are the five delays and the two rules' poll counts. The
  verdict is reported for the ten and the breakdown for the seven.
- **The 6 % band was set on other cells.** It comes from the largest deviation the replay showed on the
  default load, 5.9 % (evidence/13), and that band was measured on the four arms and two policies evidence/13
  covered, at p 0.4 as well as p 0.2. Nothing says it is the right band for the 6.0 s arm, for T37r or for F.
  So every value outside it is reported with how far outside and in which direction, whatever the count is.
  **The band is a Mode B band, per-stage aggregates. It says nothing about agreement decision by decision.**
  evidence/13's Mode A passed in 39 of 42 cells on A1, 24 on A2 and 41 on A3, and failed only because each of
  those lines required all 42 (`../13/README.md`, the first results table).
- "Missed" triggers evidence/16's procedure: the decision sequences around timeouts are compared between the
  testbed log and the replay, and where the two part is reported before any other line's conclusion. Tickets
  09 to 12 screen candidates by replay scoring, so a "missed" here puts the screening itself in question.
- A value outside the band is also read against the numbers above that bound the substitution: on seeds 1 to
  20 the generated schedules moved the delay by at most 0.3 % and a borrowed clock by at most 1.1 %, so a
  miss much larger than their sum is the replay's and not the borrowing's.

## What this step cannot show

- **Nothing about decision maker L or decision maker M.** This step runs no decision maker. It places the baselines and leaves the
  link schedules behind; which of the three decision makers is better is ticket 12.
- **Nothing about whether foresight's gain is reachable.** evidence/18 puts `V_at_F - F` at +70.1 points
  (+63.9 .. +76.1) in the replay, but V knows when records are created and when links go down. Whether that
  distance is reachable from what a node can observe is not read here.
- **F's selection used seeds 1 to 20 while the reading is on seeds 121 to 160.** The two sets do not overlap,
  so the reading is out of sample for F. What that costs is not knowable from this step alone: the gain F
  shows here is one draw of one member on 40 new seeds, and nothing says how much of the +11.1 in sample was
  the member and how much was the 20 seeds.
- **The half width 40 seeds buys.** `predict.txt` puts it at 5.01 pp for T37r and 3.92 pp for the pinned F,
  and 4.52 pp for the difference between them. The predicted F minus T37r difference is +0.72 pp, far inside
  its own half width, so **this step is not expected to separate F from T37r** and 0 of the 40 streams did.
  On the default load at p 0.4 the replay read the half width about 1.4 times too narrow against the testbed
  (`../19/spec.md` "What this step cannot show", which reads it off `../16/after.md`; the reason was never
  investigated), which would put the measured half widths here nearer 5 to 7 pp.
- **Nothing about the p 0.4 cell, p 0.5, p 0 or the no-outage cell**, and nothing about T37 or the four arms
  below 3.695 s.
- **Every number in the prediction is simulated.** A predicted band says what the replay expects the testbed
  to measure, and is itself no measurement.
- **The held-out link schedules become visible to the authors once this stage has run.** Running the 200 runs
  writes the outage timelines, t0, `agent_start` and the topology of seeds 121 to 160 into the store, where
  anyone building decision maker L or M can read them. **The rule that binds their use: these inputs are never used
  for training, for label making, for model selection or for tuning. They are used only to score, in the replay, a candidate that has already been selected (the prediction ticket 12 writes before that candidate's testbed runs). The seeds a candidate is selected on are taken from 1 to 100.** Nothing in the code enforces this; it
  is a rule on the authors, recorded here so that a later step can be checked against it.

## Deliverables (this directory)

`fpin.py`, `fpin.txt`, `fpin_floor22.txt`, `spec.md`, `predict.py`, `predict.txt`, `main10.sh`, all before the runs. After the
runs: the analysis script, its csv and md output, `README.md`, `RUN.txt`.
