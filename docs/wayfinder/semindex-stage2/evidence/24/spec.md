# evidence/24 spec: decision maker M, learned from the s11a logs by stepwise value learning, and the plan frozen before any fit on the training half

Written 2026-11-25, **before any emulator log of a seed above 20 has been read, before any coefficient of M
exists, and before any replay simulation of this effort has been run**. Ticket
`../../tickets/11-ml-from-logs.md`; its `決まったこと` sections (2026-11-22 items 1 to 4, 2026-11-24 items 5 to 9,
2026-11-25 item 10) are the user's decisions and are not reopened here. Everything a later script needs is
decided in this file, with the reason, so that no choice is left to be made after a number is seen. Where this
file chooses something the user did not decide, the line says **worker's choice** and names the alternative not
taken; those are collected again in one section near the end. Where this file departs from the design handed to
it, the line says so and the departures are collected under "Deviations from the design handed to this file".

Every number quoted here comes from a generated file of the repository and the file is named: `charges.txt`,
`features.txt`, `precheck.txt` and `labelsig.txt` in this directory, `RUN.txt` here, `../22/weight.json`,
`../22/weight.txt`, `../22/stability.txt`, `../22/README.md`, `../23/ana.txt`, `../23/agree.txt`,
`../23/eval.csv`, `../23/README.md`, `../23/RUN.txt`, `../13/validate.md`, and
`../../tickets/11-ml-from-logs.md` for the user's own decisions. Three numbers this plan would have liked are
in no repository file and each is named where it would have been used: the span of one outage in decisions, the
p 0.4 cell's own poll weight, and the row counts of the logs of seeds 21 to 80. The first gets a script of this
directory (`outage.py`); the second is dropped and the claim that needed it is rewritten; the third is an
extrapolation from `charges.txt` part 6's in-window counts, over `precheck.txt` part 1's denominators, until
`rows24.py` prints the counts, and every line that quotes it says so.

What came before, in this directory: `features.py` (the numeric form of a decision's input), `charges.py` (the
cost of one decision), `precheck.py` (what the logs can carry) and `labelsig.py` (whether ticket 09's labels
carry learnable signal), with `RUN.txt` as the log of those four runs and of the two reviews of them.
**Nothing there is a policy, nothing there is a score, and no replay simulation was run.** This file is the step
between that preparation and the first fit.

## Question, and what the step is for

**Does a policy learned from the s11a logs by stepwise value learning -- one interval per state, fitted to the
cost per second of wall-clock time -- decide the interval of a state poll better than a formula rule and better
than a fixed arm, on the replay port, in the main cell?**

M is the second of the three decision makers of the comparison (L in ticket 10, M here, the rule F already
built). The shape is fixed by the user: the learning reads the s11a logs (uniform draws among 3.695, 6.0 and
10.043 s at each resolution), it is stepwise and bootstrapped rather than a regression of whole-sequence
returns (`決まったこと` 5), it reads everything L reads (`決まったこと` 7), it is linear and numpy-only
(`決まったこと` 9), it has two candidates (`決まったこと` 8), and every setting of it is chosen from the logs
alone (`決まったこと` 10).

This step fits two candidates, scores them on the replay port against the three fixed arms and F, and chooses
one to hand to ticket 12. It is the step in which the plan is frozen; the measurements it describes have not
been made.

## What this step can show

The ticket's own `示せること`, kept, and sharpened where this directory's preparation already settled something.

1. **One policy built from the logs**: a map from the 261-cell feature vector of a decision to one of the three
   intervals, deterministic, with no draw in it (ticket Acceptance 2).
2. **M's score on the replay port, in the main cell** (p 0.2, the s6 outage shape, workload w0, store stage
   `s6b10p043`), for both candidates, beside the three fixed arms and F run through the same port on the same
   per-pair random streams, and beside ticket 10's archived rows for L (ticket Acceptance 4).
3. **That the fit reproduces bit for bit.** The solve is closed form, so two processes must write the same
   coefficient bytes. `precheck.txt` part 7 already shows the same arithmetic reproducing in three processes on
   this machine (p 0.2 digest `8ca3d05c...`, p 0.4 `167094e9...`, 6288 bytes each). **This is the check ticket
   10's torch fit could not pass** (`../23/README.md`: four byte-different adapters from two runs of one
   recipe).
4. **M's agreement with ticket 09's labels on seeds 81 to 100**, beside the baseline pre-registered below, with
   no pass or fail attached (ticket Acceptance 3).
5. **What one decision costs in time on a CPU**, in three parts, beside 100 ms, with no verdict (ticket
   Acceptance 6).
6. **That no seed above 100 is read and that seeds 81 to 100 reach no setting**, by what every script prints and
   asserts (ticket Acceptance 1).

## What this step cannot show

1. **Nothing about M on the testbed.** No run of this step touches the emulator; every log it reads is already
   on disk. Inserting a decision maker into the live loop is ticket 12's.
2. **Nothing per decision about the two worlds M straddles.** M learns from emulator logs and is scored in the
   replay. `../13/validate.md`'s Mode A, the per-decision agreement, is on record as failing (39 of 42 cells for
   A1, 24 for A2, 41 for A3, where all 42 were required), and Mode B's band is a per-stage band. **M's values
   are fitted per decision and there is no per-decision bound between the world they were fitted in and the
   world they are scored in.** This is the largest gap of the step and it is named here rather than in the
   threats alone.
3. **Nothing about the value of any policy other than the logging policy, from the logs.** The logs hold
   trajectories of one policy: the uniform draw. Every number this step computes about an improved policy's
   value is the fitted model's own extrapolation, and the only thing the logs can test directly is the logging
   policy's own continuations, which is what the penalty-selection criterion of section 5 reads and the reason
   that section says it cannot be done for any other policy.
4. **Nothing about an interval finer than the three, and nothing outside them.** The logs drew three intervals;
   M can express nothing else (ticket, 腕が 3 つしかないことが、M に効く).
5. **Nothing causal about which interval is better, from the per-interval rates.** `precheck.txt` part 2's
   40.0586 / 38.4132 / 40.1021 score ms/s (p 0.2) are three different mixtures of the five cells and of clock
   position, and that file says so in its own output; `charges.txt` part 7 shows the dearest and the cheapest
   arm changing with the charge convention in 5 of 10 (p, cell) combinations.
6. **Nothing about p 0.4 on the testbed, other outage shapes, or other workloads.** The p 0.4 logs are learning
   material for candidate b (ticket `決まったこと` 8, and 示せないこと: p 0.4 は、主のマスの結果が出るまで、
   なぞり直しの上だけで読む), and nothing of this step scores the p 0.4 cell.
7. **Nothing that corrects the selection bias of choosing one of two candidates on the same 20 seeds**, nor the
   accumulated bias of those seeds having been scored on before. See Threats 5 and 6.
8. **Nothing about the backlog and history fields being the right thing for a policy to read.** ADR 0004 line 5
   permits a policy two scalars; the user chose the full input (`決まったこと` 7, ADR 0004 が許した範囲より広く
   見る事になる。それを承知で進める). This step reads what the user chose and measures nothing about the choice.

## Names and definitions

- **E** = `docs/wayfinder/semindex-stage2/evidence/24` (this directory). **W** = `/home/lab_shared/wt/semindex-ev24`,
  the only repository touched. **store** = `/home/lab_shared/semindex-eval/s11a/p{0.2,0.4}/<seed>`, read only.
  **labels** = `/home/lab_shared/semindex-learn/ev22/labels_p0.2.jsonl.gz`, read only, pinned by
  `../22/labels.sha256`.
- **ROOT** = `/home/yamakawa/Development/semindex-learn` on `hpc-debian` (user `yamakawa`, CPU only, no card,
  nothing installed; Ticket 10's directories there are untouched). Filtered `s11a` seeds 1 to 80 logs are copied into
  `ROOT/store/s11a/p{0.2,0.4}/<seed>` and the worktree code into `ROOT/repo24` by `push_inputs24.sh`.
- **host** = `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 STORE=ROOT/store PYTHONPATH=ROOT/repo24/src ROOT/.venv/bin/python -B`
  on `hpc-debian` (Python 3.13.5, numpy 2.5.4), used for `rows24.py` and `learn24.py`.
- **local** = `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=W/src /home/lab_shared/cefore-semindex/.venv/bin/python3 -B`,
  the interpreter `RUN.txt` names (Python 3.12.13, numpy 2.5.4), used for `inventory.py`, `outage.py`, the
  replay control, dry/real evaluations, analysis, agreement and CPU latency. **No card and nothing installed
  anywhere**: numpy only (`決まったこと` 9).
  - **`決まったこと` 4 and the 2026-11-26 host split.** `決まったこと` 4 places learning tools in
    `/home/lab_shared/semindex-learn/.venv` and not in the repository's venv, and per the user's 2026-11-26
    instruction, the learning scripts (`rows24.py` and `learn24.py`) run on `hpc-debian` under `ROOT/.venv` (the
    remote learning environment) while installing no new package (`決まったこと` 9: numpy only). The replay
    evaluations, agreement and CPU latency run on `local` reading the committed `coef_a.npy` and `coef_b.npy`.
    Because `local` and `hpc-debian` run different Python/OpenBLAS builds (`numpy.linalg.solve` of a shared
    787 x 787 test matrix differs in the lowest bits between the two hosts while reproducing bit for bit across
    processes on each host), the two-process fit reproducibility gate runs on `hpc-debian` and the exact fitted
    `.npy` bytes are transferred back, verified by sha256 on both hosts, and committed.
- **The arms** are the exact floats `(3.695, 6.0, 10.043)` (`../22/port.py:45`, re-exported as `charges.ARMS`).
  An interval a decider returns must be an element of the run's `choices` by exact float identity or the port
  raises.
- **The choice set** of every simulation is `port.union_choices()`, the three arms plus every interval F reaches
  from them, because `label.run_dict` sets it and F leaves the three arms as soon as it lengthens. **So the port
  would accept more than three values and M must assert its own three** (ticket Acceptance 5).
- **The main cell** = p 0.2, workload w0, the s6 outage shape, store stage `s6b10p043`.
- **The weight** W = **104.29595912307323 ms per poll**, read from `../22/weight.json` by both `charges.py` and
  `../22/label.py` so the two cannot drift (`charges.txt` part 0; `charges.py:73-76`).
- **The score window** is `[t0 + 10000 ms, t0 + 130000 ms]` and records are created by `t0 + 72000 ms`
  (`charges.txt` part 0). The live agent stops at `t0 + 130000 ms` (`../13/replay.py:68`,
  `STOP_AT_MS = 130_000.0`). **N** = 240 records per run (`charges.txt` part 0).
- **A decision** is a `state_poll_reschedule` row. The interval on a `state_poll` row is re-drawn at the
  resolution and never governs: the ticket records that the applied interval matched the re-drawn one on every
  one of the 40 runs of seeds 1 to 20 (ticket, 作業者が決めたこと), and the mechanism is the identity early
  return at `../13/replay.py:1103-1109`. `charges.py`'s periods are cut at reschedule rows, so the send-time
  rows are outside every row set of this plan by construction.
- **A period** is one owned decision period of `charges.py`: from one decision of an ordered pair to that pair's
  next decision. **An owned period** is one a decision owns; the pair's last decision of a run owns none
  (`charges.txt` part 2).
- **A run** is one (seed, replicate stream) pair in the evaluation, and one (p, seed) emulator run in the
  learning. **A stream** is one of the 40 replicate streams (`../21/predict.py:102`, `NREP = 40`).
- **The selecting score** is `../22/score.py`'s `score_run(events, t0, weight)[0]` = `delay + weight x polls`,
  the delay taking every censored record at its lower bound. This is the statistic `決まったこと` 8 selects on
  and the one ticket 10 selected on (`../23/ana.txt` part 2).
- **The reported frontier gain** is `../20/predict.py`'s `gain = 1 - delay(decider) / interp(frontier,
  polls(decider))`, whose delay is `lat / n4` with the censored records **dropped** (`score.delay_dropped`), read
  against a log-log piecewise-linear frontier through the three arm points. Two different delays and two
  different directions of better: the score selects, the gain is reported beside it.
- **The node rule**, in the form `../13/replay.py:826-841` states it and `../23/spec.md` adopted: no node name
  and no ordinal may reach the model input or be an argument of the answer, while keying a decider's own
  per-pair memory on `(node, neighbor)` is real-node context and is what F and L both do. **M keys its send-time
  memory the same way and for the same reason**: `run["decider"]` is one object shared by all five Nodes
  (`../13/replay.py:580`), so `neighbor` alone would merge five nodes' memories into one. `features.vector`
  takes the record and nothing else, so the vector cannot read a name at all (`features.txt` part 1, and the
  rename equality there).

## 1. The row

**One row is one owned decision period of `charges.py`**, exactly as `precheck.py`'s `table_of_run` already
builds it (`precheck.txt` part 0), with one field added:

| field | what it is | where it comes from |
|---|---|---|
| `vec` | the 261-cell feature vector of the decision's own record | `features.vector(record)`, the record through `observe.from_logged_rows` -> `observe.observation` |
| `arm` | the interval the log drew at this decision, one of the three | the row's `action.interval_s` |
| `tau` | the period length in seconds | `charges.periods`' `length_s` |
| `c` | the period cost in **score ms** | `charges.charge_run`'s `cost_ms` divided by N = 240 |
| `cell` | the five-cell scalar state built from the two fields ADR 0004 line 5 permits | `charges.cell(row)` |
| `pos`, `chain`, `next_pos` | the decision's place in its pair's chain of owned periods, and how the pair's next decision is reached | `precheck.table_of_run` |
| **`t_rel_ms`** | **the decision's own instant, relative to the run's t0** | `per["start"] - run["t0"]`, attached by walking `run["periods"][pair]`'s owned periods in order and matching `pos` |

`t_rel_ms` is the one field `precheck.py`'s row does not carry and the window of section 2 is read off it alone.
It is attached rather than recomputed: `table_of_run` returns `(rows, run)`, and the k-th owned period of a
pair's chain in `run["periods"]` is the row with `pos = k`, so the instant is read off the period the row was
built from and not re-derived from the log.

**x' is the feature vector of the same pair's next decision**, i.e. `chain[pos + 1]["vec"]`, and **a' is that
decision's own `arm`**. A row whose `next_pos` is None has no x' and is a **terminal row**; section 4 says what
is done with it.

**The vector is the one shared writing of a decision's input** (ticket `決まったこと` 7, and 入力は L と同じ
もの): 261 cells, name list sha256 `16c2aa02c5738abcb819fa4503067eeaff3a82a3a0fa2020cca6ae29d79aafce`
(`features.txt` lines 20-21), proved to carry what the rendered text carries by rendering, inverting and
re-rendering every record of both sources -- 6423 of 6423 logged records and 12868 of 12868 label records, with
0 texts carrying two vectors and 0 vectors carrying two texts over all 19291 records (`features.txt` parts 2
and 3). **So on the measured records a difference between L and M is a difference of learning and not of
input**: that bijection, and not an argument about rounding, is the evidence for the user's `決まったこと` 7.
The differences between the two roads that **are** real are of another kind and are named in Threats 10:
what M provably cannot read, where M's standardisation was fitted, and how a send-time decision reaches the
vector.

**Every fitted row must carry `action_prob` = 1/3.** It is a top-level field of the decision row (ticket
`決まったこと` 2: `action_prob` に 1/3 が乗る). `rows24.py` asserts that every row of both candidates carries
the same float and refuses otherwise. **It is used for nothing else**: under a uniform behaviour policy a
per-row importance weight is one global constant that cancels in every ratio and in every argmin, so the
ticket's 重みの付け所はそこ has no numerical content here (ticket, 作業者が決めたこと, second item). This plan
states that rather than performing a reweighting that could not change a decision.

## 2. Which rows are fitted on: an arm-independent window in decision time

**A row is fitted on if and only if `10000 <= t_rel_ms <= 102000`.** Both bounds are read off the decision's own
instant and nothing else, so **no arm can be kept or dropped more often than another for what followed the
decision**.

Why a filter is needed at all, and why this one. Filtering on an outcome the interval moves biases the
comparison between the intervals, and this directory's own output says so where the filter is chosen:
`charges.txt` part 6, "The second filter reads the decision's own instant only, so it cannot keep one arm more
often than another for having landed its send in the window, which is what a caused-send filter does and the
reason none is used here". The size of what a convention can move is `charges.txt` part 7, which prints **two
separate comparisons and they must not be credited to each other**: the two send-to-send filters, `gap/caused`
against `gap/start`, disagree about the dearest or the cheapest arm in **2 of the 10 (p, cell) combinations**,
and the charge **unit**, `gap/start` against `period`, disagrees in **5 of the 10**. The decision-time filter is
`period`, so its only printed comparison is the 5-of-10 one, and that file says the filter move and the unit
move "are separable". `charges.txt` part 7 also names the mechanism for the long arm: the caused-send filter
"keeps a pre-window decision -- one poll charge over a gap holding almost no record time -- far more often for
the long arm, which is the defect the earlier verification named".

**The lower bound, 10000 ms**, is where the score's poll window opens (`../13/replay.py:66`). A period starting
before it can hold a send the score does not price: 148 owned periods at p 0.2 and 143 at p 0.4 hold no
in-window send (`charges.txt` part 4). Those are periods charged a delay and no poll, and the arm did not choose
that.

**The lower bound removes that cause and only that cause, and a second cause stays in every fitted row.** The
same `charges.txt` part 4, counting **every** send of the pair rather than only the in-window ones, shows owned
periods holding no send at all and owned periods holding two: **10 with zero and 9 with two at p 0.2, and 5 with
zero and 3 with two at p 0.4**, of 3054 and 3049 owned rows. That file names the mechanism: "a send that went
out BEFORE its predecessor's resolution was taken, which is what the send-time provisional timer does and which
leaves one period with none and the one before it with two". **No bound in decision time can remove it**, because
it fires when the pair's provisional timer beats the resolution, a condition the interval length enters, and the
window only controls which rows are kept. One poll charge is about 104.3 score ms (`charges.txt` part 0) against
mean period costs of 152 to 404 score ms, so such a row's own cost is short or long by about a quarter to two
thirds of a period.

**So the residual is printed per arm rather than assumed away.** `rows24.py` prints, **over the FITTED rows of
each candidate and broken down by arm and cell, how many hold 0, 1 and 2 in-window sends**, so the rate of the
misallocation is a number per arm. **What a reader may not conclude until that table is read: that a difference
in fitted cost between two arms is free of the one-poll misallocation.** The arm-independence the window buys is
of the row **selection**, which reads the decision's own instant alone; it is not arm-independence of the cost
inside a kept row. This is a correction to the design handed to this file; see "Changes to the worker's design
made in review".

**The upper bound, 102000 ms, is 130000 less twice a period ceiling of 14000 ms.** The ceiling, derived and then
checked against the data:

- the longest interval is 10.043 s and the choice jitter is +-10 % (`JF = 0.1`, `../21/predict.py:101`), so the
  applied interval is at most 10.043 x 1.1 = **11.0473 s**;
- a period runs from one resolution to the pair's next resolution, and the next resolution follows its send by
  the poll's own age, which is at most **2.100 s** over the 6423 logged decisions (`features.txt`,
  `res_poll_age_s` max 2.100). A period's length is the applied interval less the time this resolution had
  already taken, plus the next resolution's lag (`charges.txt` part 5), so it is at most 11.0473 + 2.100 =
  **13.147 s**;
- the largest period length on record is **13.097 s** at p 0.2 and **13.144 s** at p 0.4 (`charges.txt` part 5),
  both below that ceiling, which is the triangulation that says the derivation is not loose.
- **14.0 s** is that ceiling rounded up to the whole second -- worker's choice, 0.853 s of margin. Alternative
  not taken: 13.147 s exactly, which would leave the guarantee below resting on one observed maximum with no
  slack.

**Why twice the ceiling and not once.** Once would guarantee only that the row's own period completes. It would
not guarantee that the row is not a **terminal** row, and the terminal rows are dropped (section 4), so the drop
itself would be arm-dependent: a row's successor decision falls later when the row's own arm is longer, and the
later that successor falls, the more likely it is the pair's last decision and owns no period. With `t_rel_ms <=
102000` the row's successor falls at or before 115.147 s and the successor's successor at or before 128.294 s,
both before the agent stops, so **every fitted row has two further decisions of its pair and no fitted row is
terminal**. The exclusion of terminal rows then drops nothing from the fitted set and cannot be arm-dependent.
This is a departure from the design handed to this file, which asked for one period's completion; see
"Deviations".

**The guarantee is asserted and not assumed.** `rows24.py` refuses the step if any fitted row's `tau` exceeds
14.0 s, or if any fitted row is terminal. Either failure voids the window's whole argument, so neither is
printed and carried on from.

**And the stop the argument rests on is read from the emulator's own logs, not from the replay's constant.**
The 130000 ms above is `../13/replay.py:68`'s `STOP_AT_MS`, and `charges.txt` part 0 carries
`[t0 + 10000, t0 + 130000]` as the **score window** of the emulator logs rather than as the emulator's run
length; **no file of this directory carries the instant of a run's last decision.** So `rows24.py` prints, per
cell, **the instant of each ordered pair's last decision relative to its run's t0 -- the minimum and the maximum
over the 8 pairs and the runs it reads** -- and the two-ceiling arithmetic above is then read against that
printed minimum. If any pair's last decision falls before 128294 ms, the argument's conclusion holds only up to
that instant and the plan says so rather than inheriting the replay's number.

**What this window costs, and what no file carries yet.** Over seeds 1 to 20, the periods starting inside the
score window number 2756 at p 0.2 (952 + 936 + 868) and 2751 at p 0.4 (944 + 933 + 874) of 3054 and 3049 owned
rows (`charges.txt` part 6, `precheck.txt` part 1). The 28 s tail this window removes is 23.3 % of the 120 s
window, so the fitted rows should number about 2100 per cell per 20 seeds, i.e. about **7400 fitted rows for
candidate a (seeds 1 to 70, p 0.2)** and about **14800 for candidate b**, with about 1060 and 2120 held-out rows
(seeds 71 to 80). **Those four numbers are an extrapolation that assumes decisions are spread evenly in time,
not a count: no file of the repository carries a row count for the logs of seeds 21 to 80.** `rows24.py` prints
the counts and every later line of this effort quotes those and not these.

## 3. The objective: cost per second of wall-clock time, undiscounted

**The learning is built on cost per second and not on cost per decision.** The reason is measured. A period pays
one poll charge whatever interval it chose, so counting by decisions prices 3.695 s over 3.7 seconds the same as
10.043 s over 10 seconds: `charges.txt` part 6's mean cost per period rises monotonically with the interval
(152.466 / 232.972 / 404.030 score ms at p 0.2) while the cost per second does not (40.059 / 38.413 / 40.102),
and that file names the first ordering an artefact of counting by decisions. Converted to time, the two halves
of the cost move in opposite directions and nearly cancel: over the in-window periods at p 0.2 the poll cost per
second falls with the interval (27.4 / 17.2 / 10.4 score ms/s) while the waiting cost per second rises
(16.2 / 24.3 / 33.5), leaving the three arms within **5.8 %** of each other pooled over those same in-window
periods (`charges.txt` part 6, which prints that spread beside those splits; the 4.4 % of the same part is the
every-owned-period row, and candidate b's second cell is **11.0 %** in-window and 9.1 % every-owned).
**That near-cancellation, and not any arm's name, is what the learning has to resolve per state.**

**There is no discount.** The score is `delay + weight x polls` over a fixed window with no discount anywhere
(`../22/score.py:121`), so the undiscounted average-cost form is the only one faithful to it. A discount factor
would be a regulariser under another name and this plan does not use one.

**The differential (average-cost, semi-Markov) Bellman equation.** For a policy pi with cost rate rho_pi in
score ms per second, the differential action value Q^pi satisfies, for every state x and interval a,

```
    Q^pi(x, a)  =  c(x, a)  -  rho_pi * tau(x, a)  +  E[ Q^pi(x', pi(x')) | x, a ]
```

where c is the period's cost in score ms, tau its length in seconds, and x' the state at the pair's next
decision. Q is a differential value: it is defined only up to one constant added to every state and interval,
because adding k to Q on both sides leaves the equation unchanged. **rho is what prices elapsed time, and it is
the whole reason a long interval can be worth its waiting**: without the `- rho * tau` term the equation would
rank the short interval first for the same reason counting by decisions does.

The logging policy's own rate over any row set is the pooled ratio
`rho = sum(c) / sum(tau)` (`precheck.rho_of`), which is **39.5698 score ms/s at p 0.2 and 49.5153 at p 0.4** over
**every owned period** of seeds 1 to 20, with a per-run standard error of 0.8402 and 1.5872 over the 20 runs
(`precheck.txt` part 2).

**Those two figures are not the fitted rows' rate, and every expectation this plan sets must be read against the
fitted rows' own.** `precheck.txt` part 2 takes no window; the fit takes the window of section 2. Recomputing the
pooled ratio from `charges.txt` part 6's own **in-window** rows -- 952 / 936 / 868 periods at mean cost 165.954 /
250.966 / 440.349 score ms and mean length 3.803 / 6.041 / 10.015 s at p 0.2, and 944 / 933 / 874 at 201.583 /
308.546 / 571.981 and 3.742 / 6.047 / 10.099 at p 0.4 -- gives **43.1390 score ms/s at p 0.2 and 54.3352 at
p 0.4**, which is 3.57 and 4.82 above the every-owned-period figures, i.e. more than four of those per-run
standard errors. **Those two are themselves arithmetic on part 6's rounded means, and the 28 s tail section 2's
window removes will move them again**, so they are stated as the order of the shift and not as the fit's number:
`rows24.py` prints the fitted rows' own rate with its run-clustered standard error, and section 4.7's rate
diagnostic and section 15's surprise list are read against that. This is a correction to the design handed to
this file; see "Changes to the worker's design made in review".

## 4. The learner: least-squares policy iteration on that equation

### 4.1 The block feature map

Let d = 261 be the number of feature cells and let (mu, sigma) be the fit rows' own mean and standard deviation
per cell, with a zero standard deviation replaced by 1 (`precheck.standardiser`, whose reason is quoted in
section 5 below). For a record with raw vector x,

```
    phi(x)  =  [ 1 , (x - mu) / sigma ]            in R^(1+d) = R^262
```

and for an interval a with index j(a) in the fixed order (3.695, 6.0, 10.043),

```
    Phi(x, a)  =  the vector in R^(3*(1+d)) = R^786 holding phi(x) in block j(a)
                  and zeros in the other two blocks
```

so that, writing w = (w_1, w_2, w_3) for the three coefficient vectors,

```
    Q_w(x, a)  =  w . Phi(x, a)  =  w_{j(a)} . phi(x)
```

One coefficient vector per interval, and not one pooled vector with the interval as a column: a pooled fit could
only shift the level of the three, while the three intervals need different slopes on the same input
(`precheck.per_arm_fit`'s own reason). The standardisation is taken over all the fit rows and not per interval,
so the three coefficient sets are in one scale.

### 4.2 The linear system of one evaluation

Fix a policy pi. For each fitted row i let

```
    Psi_i  =  Phi(x_i, a_i)                        the instrument, 786 cells
    z_i    =  Phi(x_i, a_i) - Phi(x'_i, pi(x'_i))  the regressor, 786 cells
```

so that the sample Bellman residual of row i at (w, rho) is

```
    delta_i(w, rho)  =  c_i  -  rho * tau_i  -  w . z_i
```

and the equation the row asserts is `w . z_i + rho * tau_i = c_i`. Over m fitted rows, write

```
    A  =  (1/m) * sum_i  Psi_i z_i^T          786 x 786
    t  =  (1/m) * sum_i  Psi_i tau_i          786
    b  =  (1/m) * sum_i  Psi_i c_i            786
    P  =  diag, 0 at the three block intercepts, 1 at the 3d other cells
    u  =  the indicator of the three block intercepts (1,0,...,0 | 1,0,...,0 | 1,0,...,0)
```

**One evaluation is one closed-form solve of the square system**

```
    [  A + lambda * P      t  ]   [ w   ]     [ b ]
    [                         ] * [     ]  =  [   ]
    [  u^T                 0  ]   [ rho ]     [ 0 ]
```

solved with `numpy.linalg.solve`, no iteration anywhere inside it, which is what makes two processes write the
same bytes (`precheck.ridge`'s own reason, and `precheck.txt` part 7 is that property measured on the same
arithmetic).

**Why these equations and not a least squares on the rows themselves.** x' is random given (x, a), so minimising
the squared sample residual over the rows is the double-sample problem and is biased. The instrument Psi_i is a
function of (x_i, a_i) alone, so it is uncorrelated with the noise in Phi(x'_i, .), which is what makes this the
projected fixed point (LSTD-Q) rather than a residual regression. It is `../22`-style arithmetic in one respect
worth naming: the matrix `A` is **not symmetric**, because Psi_i and z_i are different vectors.

**The mean, not the sum.** A, t and b carry the factor 1/m so that `lambda` is a per-row penalty and means the
same thing on candidate a's rows and on candidate b's twice-as-many rows. `precheck.py` uses sums with
`LAMBDA = 10.0`, which at its 3054 rows is a per-row 10 / 3054 = **0.00327** in this scale; the grid of section 5
brackets that by about one and a half decades below and two and a half above (section 5).

**Dimensions, unknowns against equations.** Unknowns: 3 x 262 = **786** weights and **1** rate = **787**. The
system has 786 instrument rows and 1 normalisation row = **787**. Square. The normalisation row is not
decoration and the count is not the reason it is there:

> For every non-terminal row, `u . z_i = 1 - 1 = 0`, because exactly one block of Phi(x_i, a_i) and exactly one
> block of Phi(x'_i, pi(x'_i)) carries the leading 1. **So u is in the null space of the regressors**, which is
> the common-constant freedom of a differential value in matrix form, and the 786 instrument rows determine w
> only up to a multiple of u. The row `u . w = 0` -- the three block intercepts summing to zero -- is orthogonal
> to that direction (`u . u = 3`), so it fixes exactly the one degree of freedom that is free and nothing else.

**rho is identified even before the normalisation**, and that is worth saying because it is what the step rests
on: the solution set of the 786 instrument rows is a line `theta* + k * (u; 0)` whose rho coordinate is constant,
so rho does not depend on which point of the line the normalisation picks. **That requires the 786 x 787
instrument block to have full row rank 786**, i.e. that t is not in the range of `A + lambda * P`, which is not
implied by `u^T P = 0` and is not an algebraic consequence of anything above: it holds generically and it is a
property of the data. **So it is a read number and not an assertion.** `learn24.py` prints the rank and the two
smallest singular values of that block (788 x 789 for candidate b) beside the full system's condition number,
and that printed rank is the evidence for this paragraph. The looser sentence the earlier draft carried -- that
the ridge does not touch "the one linear combination of the instrument rows that identifies rho" -- is replaced
by this one, because that combination fixes rho only jointly with `sum_i z_i . w`.

**What the ridge does and does not penalise.** `lambda * P` adds lambda to the diagonal of the 3d non-intercept
weight cells. It does **not** penalise the three block intercepts and it does **not** penalise rho.

- It does not penalise the intercepts because a differential value has no meaningful zero: shrinking Q towards 0
  would shrink it towards a number the equation does not fix, and because, with the normalisation above, the
  three intercepts are exactly the three intervals' level difference -- the quantity the decision rule reads --
  so penalising them would shrink the answer towards "the three arms are equal".
- It does not penalise rho because rho is a rate with a physical meaning, and because of the identity below,
  which it must leave alone: since u is zero everywhere P is non-zero, `u^T P = 0`, so **the ridge leaves the
  u-combination of the instrument rows exactly as the data wrote it**, which is what (IDENTITY) below is. That
  is not by itself the statement that rho is identified, which is the paragraph above and is a printed rank.
- What it does, besides shrinking: it makes the dead cells solvable at all. A cell that is constant on the fit
  rows is exactly 0 in every row of the design after centring (section 5), so its instrument row and column are
  zero and its equation reads `lambda * w_j = 0`, giving `w_j = 0` exactly. At lambda = 0 that row would be
  singular, which is why **0 is not in the grid**.

**The exact identity the solution satisfies, which is the implementation check with teeth.** Summing the 786
instrument rows against u, and using `u^T P = 0`:

```
    sum_i  z_i . w  +  rho * sum_i tau_i  =  sum_i c_i
```

Under the logging policy, where `Phi(x'_i, pi(x'_i)) = Psi_{s(i)}` is the instrument of the very next row, the
fitted rows of one pair in one run form a contiguous block of its chain (the window of section 2 is an interval
in time and decisions are ordered in time), so `sum z_i` telescopes within each block and

```
    rho_hat * sum_i tau_i  +  sum_(blocks) [ Q_w(first row of the block) - Q_w(successor of its last row) ]
                           =  sum_i c_i                                                  (IDENTITY)
```

holds to solver tolerance. There are 8 blocks per run (one per directed pair; `charges.txt` part 3 names the 8
directed pairs). **So the logging policy's fitted rate is the logged rate `sum(c)/sum(tau)` minus the mean
endpoint-value difference of the blocks per fitted second, and nothing else.** That is the exact size of the gap
between the two, not an unexplained residual.

### 4.3 Terminal rows

A **terminal row** is a fitted-table row whose `next_pos` is None: the pair's last **owned** period of a run. Its
own successor decision is the pair's last decision of the run, which owns no period, so that successor has no
tau and no c and is in no table. There are exactly **160 such rows per cell over seeds 1 to 20** -- one per
directed pair per run, 8 x 20 -- and they sit overwhelmingly in one cell: 132 `ans_empty`, 1 `ans_content`, 14
`to1`, 10 `to2`, 3 `to3` at p 0.2, and 89 / 6 / 33 / 19 / 13 at p 0.4 (`precheck.txt` part 1, the `last` column).

**What is done with them: a terminal row contributes no equation, and still serves as the successor state of the
row before it.** The two halves are separate and both matter.

- No equation, because Q at its successor does not exist. Treating the missing successor as Q = 0 would be wrong
  in a way that does not announce itself: a differential value is defined up to a common constant, so anchoring
  one state at 0 is an arbitrary statement about that constant, and it would also break the null-space argument
  above (`u . z_i = 1` for such a row), i.e. it would silently fix the common constant through the terminal rows
  instead of through the normalisation.
- Still a successor, because its own `vec` and `arm` exist: the row before it reads `Phi(x', a')` off it. Only
  its own equation is absent.

**By section 2's window no fitted row is terminal**, so this exclusion removes nothing from the fitted set; it is
written out because the row builder sees terminal rows (they exist in the table for seeds 1 to 70) and must
refuse them rather than drop them silently, and because the successor of a block's last fitted row is frequently
a row that is itself outside the window -- which is correct and intended.

Alternative not taken: giving a terminal row an equation by reaching the pair's last decision's own feature
vector, which exists as a logged decision. Not taken because that successor is a state whose own period is cut
off by the agent's stop, so its value is a boundary artefact of another kind, and because reaching it means
building a record outside `precheck.py`'s own table.

### 4.4 The next-action rule for the logging policy

For the logging policy pi_0, **the next action is the arm the log drew at the next decision**:
`Phi(x'_i, pi_0(x'_i)) = Phi(x'_i, a'_i)` with `a'_i = chain[pos+1]["arm"]` -- worker's choice.

Three reasons, in order of weight:

1. **It is the same functional form as every later evaluation.** An improved policy's next action is a single
   interval, `pi(x') = argmin_a w_a . phi(x')`. One next-action rule means one code path and one equation for
   every iteration, and the penalty chosen on the logging policy's residual is then the penalty of the estimator
   the iteration actually uses.
2. **It alone lets the identity be re-derived through the block endpoints.** This is a narrower claim than the
   earlier draft of this plan made, and the narrowing matters. The **general** identity
   `sum_i z_i . w + rho * sum_i tau_i = sum_i c_i` holds for the 1/3-average rule too: for that successor
   `u . z_i = 1 - 3 * (1/3) = 0` still, so u is still a null direction and the u-row of the system still closes.
   What the average rule forfeits is only the **telescoped** reading of it -- the independent walk through the
   block endpoints of section 4.2 -- and by section 4.7 that walk is the only form of Check A with any reach at
   all. So the reason is the walk and not the identity. See "Changes to the worker's design made in review".
3. It is unbiased: the log drew the interval uniformly and independently of the state (`action_prob` 1/3 on every
   row), so `Q(x', a'_i)` is an unbiased draw of `E_{a' ~ pi_0}[Q(x', a')]`.

Alternative not taken: `Phi(x'_i, pi_0(x'_i)) = (1/3) * sum_a Phi(x'_i, a)`, the exact conditional expectation
under the uniform policy. It has strictly lower variance, by Rao-Blackwell, and this plan does not take it
because it forfeits reasons 1 and 2: it is a second next-action rule, and the telescoped walk is not available
for it.

**The cost of the choice is variance, and it is measured rather than named.** It costs one extra solve, so
`learn24.py` prints, **at `lambda*`, the logging policy's `rho_hat`, the three block intercepts and the
held-out one-step residual's mean under both next-action rules**, for each candidate. The variance cost of this
plan's choice is then a printed pair and not an unknown.

### 4.5 The greedy improvement

Given an evaluation's coefficients w,

```
    pi_greedy(x)  =  argmin over a in (3.695, 6.0, 10.043) of  w_{j(a)} . phi(x),
                     the FIRST minimum in that order
```

Lower is better everywhere in this plan, so the improvement is an argmin and not an argmax. The first-minimum
rule in the fixed arm order is the rule ticket 10's read-out and `labelsig.py`'s `pick` both use, so a tie is
resolved the same way in all three places. **rho does not appear in the argmin**, and that is not a sign that
rho is idle: rho entered the coefficients through the `- rho * tau` term of the fit, which is where the price of
elapsed time lives.

### 4.6 The iteration, its convergence test, its cap and its fallback

```
    pi_0            = the logging policy (section 4.4)
    for k = 0, 1, 2, ... :
        (w_k, rho_k) = the solve of section 4.2 for pi_k, at the penalty of section 5
        pi_(k+1)     = pi_greedy(w_k)
        changed_k    = the share of FITTED rows whose greedy interval differs between
                       pi_(k+1) and pi_k, evaluated on the fitted rows' own x
        stop when changed_k = 0, or when k reaches the cap, or when a bound below is breached
    M = pi_greedy(w_k) of the last evaluation performed, IF the loop stopped at changed_k = 0
        (the cap and the bounds each name their own M below)
```

- **The convergence test is `changed_k = 0`**: the greedy interval of every fitted row is unchanged, so the next
  evaluation would be the same solve and the iteration has a fixed point on these rows. The share is printed at
  every iteration whatever it is. For k = 0, `changed_0` is measured against the logged arm of each row, so it is
  printed as what it is -- how often the one-step improvement of the logging policy disagrees with the log -- and
  it is never a stopping condition, because pi_0 is not a deterministic policy and could not be reproduced by a
  greedy rule.
- **The cap is 20 evaluations** -- worker's choice. Alternative not taken: 10, which is the usual order for
  least-squares policy iteration to settle and would make the fallback more likely to be what ships. 20 is
  priced at about 20 solves of a 787 x 787 system plus 20 passes over the rows, which is minutes.
- **The fallback, fixed here so that nothing is judged after a number is seen**: if the cap is reached with
  `changed_k` non-zero, **M is `pi_greedy(w_0)`, the one-step improvement of the logging policy's own
  evaluation**. That policy is chosen in advance, it needs no comparison to pick it, and it is the one
  improvement whose evaluation the logs can test (section 5). The run still prints every iteration's
  `changed_k`, every iteration's rho and every iteration's arm shares, so a reader can see what the iteration
  was doing when the cap stopped it.
- **The condition-number bar is `numpy.linalg.cond` of the system matrix above, in the 2-norm, at 1e10** --
  worker's choice. Double precision carries about 16 digits, so a bar of 1e10 leaves about 6; the live part of
  the matrix has a diagonal of order the arm shares (about 1/3) and the dead cells sit at lambda, so the
  expected value at the grid's smallest penalty is of order 1e3 to 1e6. **That expectation is an inference from
  the matrix's own scale and not a measurement**, and the bar is set where it catches a genuinely singular
  system -- a penalty not applied, a duplicated column, a mis-built normalisation row -- rather than where it
  trims an ordinary one. The condition number is printed at every solve.
  - If the solve at iteration k > 0 exceeds the bar, the iteration stops there and **M is `pi_greedy(w_(k-1))`**,
    the last evaluation that stayed under the bar.
  - If the solve at k = 0 exceeds the bar for either candidate, **nothing proceeds**: the logging policy's
    evaluation is the one every later number leans on, and this plan does not say what to do without it. The
    step stops and reports, and reopening it is the user's.
- **Two pre-registered numerical bounds on the iteration, and a cycle test, because neither the convergence test
  nor the condition bar can see a diverging iteration.** Off-policy least-squares policy iteration has no
  contraction behind it: the projected Bellman operator under the logging distribution is not a contraction for
  the policies being evaluated, so `rho_k` and `||w_k||` can grow or change sign while the argmin pattern stands
  still, and `changed_k = 0` is only a fixed point of the greedy map on the fitted rows. The condition number
  does not stand in for either: it is the same matrix structure at every iteration and it does not move with the
  weights. **So the following are fixed now, with the existing fallback as the single consequence.** Write
  `R_fit = sum(c) / sum(tau)` over the candidate's own fit rows, per cell for candidate b. At every iteration
  `k > 0`:
  - **the rate stays a rate**: `rho_k > 0` and `|rho_k - R_fit| <= 0.25 * R_fit`. A cost rate is a sum of
    non-negative costs over positive time, so a negative or zero `rho_k` is divergence and nothing else; and the
    in-window arms sit within 5.8 % of each other per second at p 0.2 and 11.0 % at p 0.4 (`charges.txt` part 6),
    so a fitted greedy rate a quarter away from the logged one is reading an improvement the three arms cannot
    supply. **Worker's choice**; alternative not taken: positivity alone, which would let a drifting rate run.
  - **the weights stay bounded**: `||w_k||_2 <= 10 * ||w_0||_2`.
  - **the iteration is not cycling**: `pi_(k+1)` is compared against **every** earlier `pi_j` and not only
    against `pi_k`, and a repeat that is not `pi_k` is reported as **a cycle of that length with the iteration
    index**. Without it the plan would reach the cap, call the fallback and print `changed_k`, and never say
    whether the iteration was cycling or drifting.
  - **A breach of either bound sets `M = pi_greedy(w_0)`**, the fallback already written, and is printed with the
    iteration index and the breached quantity. A cycle is printed and the cap then decides, so a cycle reaches
    the same fallback. This is an addition to the design handed to this file; see "Changes to the worker's design
    made in review".
  - **Why this exit ships `pi_greedy(w_0)` while the condition bar at k > 0 ships `pi_greedy(w_(k-1))`**, which
    is a difference and not an oversight: a conditioning breach is a statement about the k-th **solve** and says
    nothing against the k-1-th, which stayed under the bar, so the last good iterate is the right thing to ship;
    a rate or weight-norm breach is a statement about the **iteration**, which by then has been moving away from
    anything the logs can price, so no iterate after the first is trusted and the plan falls back to the one
    improvement whose own evaluation the logs can test.

### 4.7 The checks on the logging policy's evaluation, and the penalty each is read at

**The order is fixed here, because one of the quantities below is a function of the penalty.** Section 5 selects
`lambda*` first, by its own criterion and before any check of this section is computed. **Check A is then
computed at every grid penalty and at every iteration**, because it holds by algebra at any penalty and costs one
pass over the rows. **The centring gate and the rate diagnostic are computed at `lambda*` alone**; their values
at the other four penalties are printed as a curve and **the curve is not a gate**. **No failure of any check of
this section may be answered by re-selecting the penalty**, and the plan leaves no later moment at which the
penalty could be re-selected.

**Check A, the exact one, on the construction. This is a gate.**

```
    | rho_hat * sum_i tau_i
      + sum_(blocks) [ Q_w(first row) - Q_w(successor of last row) ]
      - sum_i c_i |                                   <=  1e-6 * sum_i |c_i|
```

This is (IDENTITY) of section 4.2 rearranged. The tolerance is the one `charges.py` and `precheck.py` already use
for their closing identities (`charges.txt` part 1, `precheck.check_cost_closes`). **If it fails, the
implementation is wrong and nothing proceeds.**

**Check A has teeth only if its endpoint walk is re-derived, and its reach is narrower than it looks.** Two
sentences the implementer must not misread:

- **The walk must be built from the chain of owned periods and not from the stored `z_i` array.** The
  **un**telescoped form of the same statement, `sum_i z_i . w + rho * sum_i tau_i = sum_i c_i`, is just the
  u-combination of the system's own rows and holds for **any** `z` whatever, so computed off the stored `z` it
  checks nothing about the successor at all. The gate is the telescoped form against an independent walk:
  `rows24.py`'s chain gives each block's first row and the successor record of its last row, and `m24.py` reads
  `Q_w` off those records.
- **What it then catches is an error that changes the sum of the successor values**: a flipped successor sign and
  an off-by-one successor both fail it. **An error that permutes the successors inside the fitted set does
  not**, because `sum_i Phi(x'_i, a'_i)` is unchanged by a permutation, and neither does rolling the successors
  by a whole block. So **Check A is not a check that each row's successor is the right one.**

**What Check A does not catch at all, and the direct assertions that replace the claim.** Check A is a
u-combination, so it is blind to anything that combination cannot see. `rows24.py` and `m24.py` therefore assert
the following on the construction itself, each printed:

- **the penalty matrix**: `P` is exactly 0 at the three block-intercept indices and exactly `lambda` elsewhere,
  and `u^T P` is the zero vector, cell by cell;
- **the normalisation row**: equal to `u` cell by cell. A normalisation row other than `u` never enters
  (IDENTITY), which is derived from the 786 instrument rows alone;
- **tau's unit**: every fitted `tau` lies in `[1.0, 14.0]` seconds and the per-arm medians sit inside the range
  `charges.txt` part 5 prints (medians 1.364 to 12.172 s over the twelve (p, trigger, interval) rows). A tau in
  milliseconds rescales `rho_hat` by 1/1000 and leaves (IDENTITY) exact, so only a printed range catches it;
- **the successor key**: for every fitted row, the successor record's `pos` is the row's `pos + 1` and its
  `(p, seed, pair)` equals the row's. This, and not Check A, is what catches a mis-keyed successor.

**The rate diagnostic, which is not a gate.** `learn24.py` prints, at `lambda*`, `rho_hat` beside the logged rate
`sum(c)/sum(tau)` over the same fitted rows, their difference, **the per-block endpoint values
`Q_w(first) - Q_w(successor of last)`, the sign distribution over the roughly 560 fitted blocks, and the
difference's run-clustered standard error** over the fit runs -- **(p, seed) pairs, 70 for candidate a and 70 in
each of candidate b's two cells**, since section 6 compares each cell's rate with that cell's own logged rate.
Over seeds 1 to 20 that standard error is 0.8402 score ms/s at p 0.2 and 1.5872 at p 0.4 on 20 runs
(`precheck.txt` part 2); on 70 runs it would be about 0.45, **which is sqrt(20/70) of the measured figure and an
inference, not a measurement**. Three sentences fix how it is read:

- **It is a diagnostic and not a gate, and the step proceeds whatever it shows.** By Check A the difference is
  exactly the blocks' endpoint-value difference per fitted second, which is a property of the chain lengths and
  of the states' drift through a run, not of the arithmetic. A gate on it would be a gate the plan expects to
  fire for a legitimate reason: the author's own estimate of the endpoint term is about 0.57 score ms/s (about 50
  score ms per block over roughly 560 blocks and roughly 49000 fitted seconds) against that inferred 0.45, and
  section 15 calls the quantity surprising only above 2.0 score ms/s. Making it a gate would leave the most
  likely path of this step ending in a decision taken with `rho_hat` and the whole fit in hand, which is what
  this freeze exists to prevent. This is a departure from the design handed to this file and from the plan's own
  earlier draft; see "Changes to the worker's design made in review".
- **A small difference at a large penalty is not evidence of a good fit.** Shrinking the non-intercept weights
  flattens `Q_w`, which mechanically shrinks the endpoint term the diagnostic measures, whether or not the
  flatter `Q` is nearer the truth. That is why the number is read at `lambda*` and why the curve over the five
  penalties is printed beside it rather than compared.
- **Its reference is the fitted rows' own rate** and not `precheck.txt` part 2's 39.5698 and 49.5153, which are
  every-owned-period figures (section 3). `rows24.py` prints the fitted rows' rate with its run-clustered SE and
  every later line quotes that.

**The held-out residual must be centred. This is a gate, at `lambda*` alone.** The mean of the held-out one-step
residual of section 5 must satisfy `|mean| <= 2 * SE`, the standard error clustered on the held-out runs (10 for
candidate a, 20 for candidate b). **The in-sample version of the same statement is not a check**: on the fit rows
`u . Psi_i = 1` and `u^T P = 0`, so the u-combination of the system reads `sum_i delta_i = 0` exactly, by the
same algebra as Check A and for any `z`. `precheck.txt` part 2 asserts that in-sample statement for its own
target ("the one-period target sums to 0 over each cell's own rows, asserted to 1e-6") and it is an identity
there too.

**What the held-out centring gate can and cannot see, pre-registered.** `learn24.py` prints the held-out mean
residual, its run-clustered SE, and **the implied detectable level in score ms per row and in score ms per
second** (the mean divided by the held-out rows' mean tau). On the extrapolated 1060 held-out rows in 10 runs,
with period costs of 152 to 404 score ms (`charges.txt` part 6), a two-SE bar sits at the order of tens of score
ms per row. **So it would catch a level error of that size and it would not catch a rate error of the size the
rate diagnostic measures** -- about 0.45 to 0.57 score ms/s over a mean tau of about 6.5 s is about 3 score ms per
row, an order of magnitude below the bar. **If it fails, nothing proceeds**: a held-out level error of tens of
score ms per row is not a boundary term, it is a fit whose values the logs can see are wrong.

## 5. The one free setting: the ridge penalty

**The grid is fixed here: `lambda in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)`**, five values a decade apart in the per-row
scale of section 4.2. It brackets `precheck.py`'s own untuned penalty, which is 10.0 on sums over 3054 rows,
i.e. 0.00327 per row (`precheck.txt` part 3 and part 8, "The penalty 10.0 is fixed and not tuned"): **about one
and a half decades below it (0.00327 / 1e-4 = 32.7) and about two and a half above (1.0 / 0.00327 = 306)**, which
is not symmetric and is not claimed to be. 0 is not in the grid because the dead cells would make the system
singular (section 4.2). Worker's choice; alternative not taken: a finer grid of nine half-decade values, which
would spend more of the held-out rows' resolving power on a setting whose effect this step cannot separate from
noise anyway (Threats 17).

**The criterion is a frozen-tail on-policy criterion, whose target does not depend on the penalty being
scored.** Two numbers are fixed here, before any number of this step is read: **the reference penalty
`lambda_ref = 1e-4`**, the smallest of the grid, and **the horizon `n = 4` decisions**.

1. Fit the logging policy's evaluation on the fit rows (seeds 1 to 70, the window of section 2) **once at
   `lambda_ref`**, giving `(w_ref, rho_ref)`, and the fit rows' own `(mu, sigma)`.
2. **The criterion's rows are the held-out rows whose decision instant satisfies
   `10000 <= t_rel_ms <= 60000`** -- `130000 - (n+1) * 14000`, by the same ceiling as section 2. Read off the
   decision's own instant and nothing else, so the row set is arm-independent for the reason section 2 gives, and
   by section 2's argument every such row has `n + 1` further decisions of its pair, so the `n` rows
   `i, ..., i+n-1` each own a period and carry a `c` and a `tau`. That window is 50 s of the fit window's 92 s,
   so it costs about 46 % of the held-out rows; `learn24.py` prints the count `m''`.
3. For each such row i, with phi standardised by the **fit** rows' mu and sigma, and `rho_ref` the rate of row
   i's own cell for candidate b,

```
    y_i  =  sum_(j = 0 .. n-1) [ c_(i+j)  -  rho_ref * tau_(i+j) ]
            +  w_ref,{j(a_(i+n))} . phi(x_(i+n))
```

```
    Crit(lambda)  =  (1/m'') * sum_i [ w_{j(a_i)}(lambda) . phi(x_i)  -  y_i ]^2
    lambda*       =  argmin over the grid of Crit(lambda), ties to the LARGER lambda
```

**Why the target is frozen and why that is the whole point.** `y_i` is built from `(w_ref, rho_ref)` and from the
logged costs, lengths and arms alone, so **it is the same number for every penalty**: its own randomness cancels
out of the comparison across the grid, and the only thing that moves between grid values is the prediction
`w_{j(a_i)}(lambda) . phi(x_i)`. **What the criterion ranks penalties up to is the reference's own error**: a
penalty is scored by how near its fitted `Q` comes to a target built at `lambda_ref`, so if `lambda_ref` is badly
chosen the whole ranking is shifted. That is a named cost of a pre-registered reference and it is the price of
having no decision left to take after a number is seen.

**Why the obvious criterion -- the held-out squared one-step residual -- may not select the penalty.** For a row
of the logging policy,

```
    E[ delta_i^2 | x_i, a_i ]  =  ( c_i - rho * tau_i - w . zbar_i )^2
                                  +  Var( w . Phi(x'_i, a'_i) | x_i, a_i )
```

and the second term, the double-sampling term, **depends on w and therefore on the penalty being scored**:
shrinking the non-intercept weights flattens `Q` and mechanically shrinks it, whether or not the flatter `Q` is
nearer the truth. So the criterion's own noise floor falls with the penalty and the argmin is dragged to the
largest value of the grid. The three reasons below are all reasons that one row's residual is an unbiased sample
**at a fixed parameter**; **none of them licenses comparing the statistic across penalties that change its
target**, and that is the distinction the earlier draft of this plan missed. Rao-Blackwellising the `a'` draw
does not fix it, because the term it removes is only part of the floor. This is a departure from the design
handed to this file; see "Changes to the worker's design made in review".

**The one-step residual stays in the output as a diagnostic, and may not be read as a selector.** For each
penalty `learn24.py` prints the held-out one-step differential residual row by row, in **both** forms -- with the
logged next arm, and with the exact Rao-Blackwellised bootstrap `(1/3) * sum_a w_{j(a)}(lambda) . phi(x'_i)`,
which is computable per row -- as

```
    delta_i(lambda)  =  c_i
                        -  rho(lambda) * tau_i
                        +  w_{j(a'_i)}(lambda) . phi(x'_i)
                        -  w_{j(a_i)}(lambda) . phi(x_i)
```

over the held-out rows of the window of section 2, where `a_i` is the arm the log drew at row i, `x'_i` and
`a'_i` are the state and the arm of the same pair's next decision, and for candidate b `rho(lambda)` is the rate
of row i's own cell (section 6). Its mean, its run-clustered standard error and its mean square are printed for
all five penalties, and the mean is what the centring gate of section 4.7 reads at `lambda*`. **Neither form of
it is a selector, and a reader may not take its minimum as a better penalty.**

**lambda\* is held fixed through the whole policy iteration**: no later evaluation re-selects it, no check of
section 4.7 may be answered by re-selecting it, and no replay score is ever looked at to choose it
(`決まったこと` 10). `Crit` is printed for all five penalties beside its own run-clustered standard error, so a
reader can see whether the five are separable at all (Threats 17).

**Why these held-out rows can be read for the logging policy at all.** The held-out rows are logged transitions
of the logging policy itself. Three things make a held-out row an unbiased sample of that policy's own quantities
at `(x_i, a_i)`: the next arm `a'_i` was drawn uniformly and independently of the state, so `Q(x'_i, a'_i)` is an
unbiased draw of the expectation the equation takes; the period `(c_i, tau_i, x'_i)` was produced with the other
seven directed pairs deciding under the logging policy, which is this policy's own environment; and the rows are
distributed under the logging policy's own state distribution, which is the distribution its value error should
be weighted by. The held-out seeds appear in no fit, which `precheck.check_held_out` is the existing test for
(`precheck.txt` part 6 shows it firing).

**Why it cannot be computed without bias for any other policy.** Two reasons, and the first is the one that
matters here:

1. **The one-step transition is not policy-free in this system.** The period from one decision of a pair to the
   pair's next runs while the other seven directed pairs are deciding too, and a record's wait inside this
   period was set partly by where its arrival fell in the other pairs' poll phases. Under an improved policy
   those pairs decide differently, so `(c_i, tau_i, x'_i)` is a draw from the logging policy's environment and
   not from the improved policy's. `charges.txt` part 9 states the same fact from the cost side: "the same
   decision also moves where a record arrives inside the DOWNSTREAM pair's periods, and that cost is charged to
   the downstream pair. The partition is additive across decisions and is not separable across pairs."
2. Even setting that aside, the held-out rows are the logging policy's states, while an improved policy's score
   is driven by the states it visits itself, and nothing in the logs samples those.

**So the logs cannot validate the value of any policy other than the logging one, and the improved policies'
values are extrapolation.** This is exactly where ADR 0004's warning about accumulated bootstrapping error
applies, and section 12 is that warning's own section.

## 6. Candidate b: one rate per cell

**Candidate b pools the p 0.2 and p 0.4 rows of seeds 1 to 70 into one fit, with every row keeping its own cost
in score ms at the main cell's weight** (user, `決まったこと` 8). Two things follow and both are stated.

**What the main cell's weight means for a p 0.4 row.** The cost of a period is its waiting milliseconds plus W x
N per in-window send, divided by N (`charges.txt` part 6), with the same W = 104.29595912307323 ms per poll in
both cells (`charges.txt` part 0). W is the main cell's indifference point: `../22/weight.json` records its own
cell as `p 0.2, w0, the s6 8-12 s outage shape, seeds 1 to 100, nrep 40`. So a p 0.4 row's poll is priced at the
exchange rate between a poll and a millisecond of waiting **that holds in the main cell**, which is also the rate
the selecting score of section 8 uses. **Whether p 0.4 has a different indifference weight is in no file of this
repository**: `../22/weight.json` carries one weight and `../22/weight.txt` solves it on the main cell only. This
plan therefore makes no claim about the size of that mispricing and drops the figure a scratch report carried for
it. **Pooling is still what the user asked for**, and it is the right thing to ask for under one reading that
this plan states: candidate b buys coefficient rows from a second environment for the shared map from state to
interval, and it is scored, like a, on the main cell alone.

**What candidate b's estimand actually is, which is not the main cell's own fixed point.** With **one** shared w
and **two** kernels, b's solution is the projected fixed point of the **row-count mixture** of the two cells, and
on `charges.txt` part 6's own in-window counts that mixture is about even: 2756 p 0.2 rows against 2751 p 0.4
rows over seeds 1 to 20, a p 0.4 share of 2751 / 5507 = **0.4995**. **So about half of candidate b's moment
conditions come from a cell nothing in this step scores**, and the two rates of this section fix the **level**
of that mixture per cell without changing which mixture it is. `rows24.py` prints the share
`m_0.4 / (m_0.2 + m_0.4)` on the real fit rows.

**So the extra rows can move the map for the scored cell and not only its level, and the step prints how far.**
Before any score exists, `learn24.py` prints **the two candidates' coefficient vectors' difference** (the 2-norm
per arm block, and the largest ten cells by absolute difference) and **the share of candidate a's own fit rows
on which a's argmin and b's argmin disagree**. A score difference between a and b can then be read against how
far the map itself moved, instead of being attributed to the poll's price alone. This is an addition to the
design handed to this file; see "Changes to the worker's design made in review".

**One rate per cell, which is a departure from the design handed to this file.** The two cells' logged rates are
39.5698 and 49.5153 score ms/s (`precheck.txt` part 2), 25 % apart. A single pooled rho would be neither, and rho
multiplies tau, whose spread between the arms is about 6.3 s per period (`charges.txt` part 6: mean lengths 3.806
/ 6.065 / 10.075 s at p 0.2); a rho error of 5 score ms/s therefore mis-prices the long interval against the
short one by about 30 score ms per decision, against period costs of 152 to 404 score ms. **That is the very
term the decision rule turns on, so it is not a rounding.** The well-posed and faithful form takes two rates and
one extra instrument:

```
    unknowns:  w (786),  rho_(0.2),  rho_(0.4)                                      = 788
    rows 1..786:  (A + lambda*P) w  +  t_(0.2) rho_(0.2)  +  t_(0.4) rho_(0.4)  =  b
                  with  t_(q) = (1/m) sum over rows of cell q of  Psi_i tau_i
    row 787:      (1/m) * sum over rows of cell 0.4 of [ z_i . w + rho_(0.4) tau_i - c_i ]  =  0
    row 788:      u . w  =  0
                                                                      equations    = 788
```

**Why exactly one extra instrument and not two.** The natural instruments for two rates are the two cell
indicators, `g = 1{p = 0.2}` and `g = 1{p = 0.4}`. Their sum is the constant 1, and the constant is already in
the span of the 786 instrument rows -- summing the three block intercept rows gives `u^T`, and `u^T Psi_i = 1` for
every row -- so **exactly one of the two is new**. Adding the p 0.4 one gives the square count above, and the
pair (the already-present pooled row, the new p 0.4 row) spans the same space as the two indicators, so the
choice of which one to add is not a choice about the answer. The null direction is still u alone, so one
normalisation still suffices, and `u^T P = 0` still holds, so the ridge still touches neither rate.

**The identity of section 4.2 then holds once per cell**, since a block of fitted rows belongs to one (p, seed)
run: Check A is run per cell for candidate b, and section 4.7's rate diagnostic compares each cell's `rho_hat`
with that cell's own logged rate, its standard error taken over that cell's own 70 fit runs.

**Candidate a has one rate and the 787-row system of section 4.2**, because all of its rows are p 0.2.

## 7. Standardisation, and what M is blind to

**The mean and the standard deviation of every cell come from the fit rows only** (seeds 1 to 70 of the
candidate's own cells, inside the window of section 2), and the held-out rows, the replay decisions and the label
rows of the agreement are all standardised by those numbers. A cell whose standard deviation is zero on the fit
rows gets 1 in its place (`precheck.standardiser`, whose docstring gives the reason: dividing by zero would make
every coefficient NaN and spread the NaN to all of them), so after centring that cell is exactly 0 in every row
and the ridge holds its weight at exactly 0. **A constant cell therefore gets no weight**, as the design
requires, and the mechanism is arithmetic rather than a filter.

**The dead list is recomputed on the fit rows and printed in full by `rows24.py`.** What `features.txt` already
measured is a different row set and is quoted here as the expectation, not as the definition: over the 6423
logged reschedule rows of seeds 1 to 20 of **both** cells, **150 of the 261 cells are constant** and each is
named (`features.txt` lines 371-521). They are, in words, what M is blind to on that data:

- **a poll that is pending**: `last_poll_status_none` and `last_poll_status_pending` never move;
- **anything beyond two origins**: `o3_*` to `o8_*`, 6 origins x **20** cells, all dead, and `origins_omitted` and
  every `o*_advertisers_omitted` dead, so **a truncated advertisement is invisible**;
- **anything beyond two advertisers per origin**: `o1_p3_*`, `o1_p4_*`, `o2_p3_*`, `o2_p4_*`;
- **a clamped interval**: `x1_clamped` to `x8_clamped`;
- `res_present`, which is 1 on every row because every fitted row is a resolution.

**Three more cells are nearly constant, and a weight resting on a handful of rows is worse than no weight
because it does not announce itself**: `in_flight` departs from 0 on 2 of 6423 rows, `fetch_timeout_present` and
`fetch_timeout_age_s` on 9, and `o2_missing` on 2 (`features.txt` lines 527-531). They are not dead, so the
standardisation keeps them and the fit may lean on them. `rows24.py` prints the same near-constant table on the
fit rows, and **this plan adds no filter for them**: a cell that moves on three runs of seventy is a real cell
whose coefficient is badly determined, and the honest statement is Threats 9 rather than a hand-placed cut.

**What the vector buys over the five scalar cells, measured.** Over the same 6423 rows the vector separates 5865
of the decisions and the five-cell state separates 5 (`features.txt` part 5); held out on seeds the fit never
saw, the full vector explains 0.5008 of the one-period target's variance at p 0.2 against the five cells' 0.2117,
a paired difference of 0.2891 +- 0.0117, and the gap holds at every horizon and in both cells
(`precheck.txt` part 3). **That is explanation and not effect**, and that file says so: the largest standardised
coefficients are the backlog and the exchange history (`o2_hi`, `o1_hi`, `x8_age_s`, `o1_p1_present`,
`consecutive_poll_timeouts`; `precheck.txt` part 3), and the first three move through a run, so part of what the
vector explains is where in the run the decision sits.

## 8. M as a decider on the port

`m24.py` defines **class M**, a decider in the port's sense (`decide(obs) -> float`), built from one coefficient
file:

- **At a resolution** (`where["kind"] == "state_poll_reschedule"`): build the record with
  `features.record_from_port_obs(obs)`, the vector with `features.vector(record)`, standardise with the
  coefficient file's own (mu, sigma), and return the **first minimum** of `w_a . phi(x)` over the three arms in
  the order (3.695, 6.0, 10.043). The chosen arm is stored in the per-pair memory.
- **At a send**: return the interval that ordered pair chose at its last resolution, and **6.0 s before the pair
  has resolved anything** -- the same convention and the same default as L (`../23/spec.md`'s send-time
  convention, `決まったこと` 9 of ticket 10, `SEND_START = ARMS[1]`), **so that M and L differ in the decision
  rule and not in the convention.** F starts a pair at 3.695 instead (`../22/port.py:52`), and that difference
  between F and the two learned deciders is inherited from ticket 10 rather than chosen here.
- **No model call at a send**, so one decision costs one evaluation of three dot products; `m24.py` counts
  `reads` and `sends` and `eval24.py` checks them against the port's own `decider_ord` counters, as
  `../23/eval.py`'s `check_counters` does.
- **The pair is `(node, neighbor)`** and the node name is read for that and nothing else; `where["ordinal"]` is
  never read. The vector cannot read a name at all, since `features.vector` takes the record
  (`features.txt` part 6's two checks that no input can make fail, and the rename equality of part 1).
- **A fresh memory per simulation.** The coefficients are loaded once; only the memory is rebuilt, which is what
  "fresh" means here, exactly as for L.
- **M asserts that every value it returns is one of the three arms**, by exact float identity against
  `port.ARMS`, and counts them per arm. The port's own guard would accept more, because the run's choice set is
  `port.union_choices()` (ticket Acceptance 5).
- **The tie count is printed.** An exact tie between two of the three dot products is possible and the
  first-minimum rule resolves it; the count of ties is printed by every script that produces decisions, so the
  rule's reach is a number and not a footnote. `../23/README.md` is why this matters: candidate a of ticket 10
  had the tie rule deciding up to 0.948 of its non-`long` answers.

**The repeat check (ticket Acceptance 2: 同じ入力に同じ腕を返すこと).** `m24.py`'s `self_check` and **every
script that produces a decision** -- `learn24.py`, `eval24.py`, `agree24.py`, `latency24.py` -- read **one fixed
logged row, the first `state_poll_reschedule` row of p 0.2 seed 1 in log order** (named in every output, so the
row is the same in all five places and is fixed here rather than when the first script is written), twice
through the same class M, and print that the three dot products are bit-identical and that the two passes chose
the same arm, **shown failing** by perturbing one of the two stored triples by one unit before the comparison. It is three numbers' worth of work for a closed-form linear model with no draw in it, which is
exactly why it costs nothing and is still printed: the ticket asks it of the read-out that decides, and a
coefficient file loaded with the wrong dtype or a memory consulted at a resolution would show up here and
nowhere else.

**The freshness assertion must be made to apply to M's type.** Ticket 10's harness wraps a decider in
`FirstCallEmpty` and only asserts an empty memory when `keeps_memory(decider)` is true, and that function is
`isinstance(decider, (D.L, LBL.F))` (`../23/eval.py:555-565`). **A new class returns False there, so the
assertion would silently check nothing for M.** `eval24.py` restates the wrapper with `isinstance(decider,
(m24.M, LBL.F))`, and `FirstCallEmpty.memory()` reads the attributes `mem` or `ival`, so **M names its memory
`mem`** and the wrapper raises rather than passes when a stateful decider carries neither
(`../23/eval.py:520-553`). `eval24.py` shows the assertion failing on a memory seeded by hand, and prints, for
every decider it runs, whether the assertion was made.

**`../23/decider.py` is not imported.** It imports `prefill.py`, which raises `SystemExit` at import unless
`argv[1]` is a directory or `SMOKE_ROOT`/`ROOT` is set (`../23/prefill.py:39-42`), which `labelsig.py` already
had to work around for `pick` (`RUN.txt` section 4). So the conventions that module holds -- the send-time default
6.0, the per-pair keying, the first-minimum tie rule, the empty-memory assertion -- are **restated** in `m24.py`
against `port.ARMS` and against the sentences of this section.

**The send-time default is tied back to L's code by a source check, not by an import.** Importing is not the only
road, and `labelsig.py` already measures a sibling module's source with `ast` rather than executing it
(`RUN.txt` section 4, and Deviations 7). So `m24.py`'s `self_check` **parses `../23/decider.py` with `ast` -- no
import and no execution -- and asserts that `SEND_START` is assigned `ARMS[1]` and that the send branch of L's
`decide` returns the per-pair memory's default**, printing both. What that check **cannot** cover is named: it
reads L's source and not L's running behaviour, so it says the convention M restates is the one L's code
expresses, and nothing about whether the archived rows of `../23/eval.csv` were produced by that code state.
This is a change from the earlier draft, which said no check of this step could tie the convention back; see
"Changes to the worker's design made in review".

## 9. The two candidates, and reproducibility before any selection score

| candidate | fit rows (seeds 1 to 70) | held-out rows (seeds 71 to 80) | rates |
|---|---|---|---|
| **a** | p 0.2 only | p 0.2 only | one |
| **b** | p 0.2 and p 0.4 pooled | p 0.2 and p 0.4 pooled | one per cell (section 6) |

Both use the same window (section 2), the same vector, the same equations, the same grid, the same selection
criterion with the same `lambda_ref` and the same horizon, the same cap, the same bounds, the same
bar and the same fallback. **The penalty is selected separately for each candidate on its own held-out rows**
(user, `決まったこと` 10, and the design handed to this file). Row counts are `rows24.py`'s output; the
extrapolation of section 2 is about 7400 and 14800 fit rows and about 1060 and 2120 held-out rows.

**Unknowns against rows.** 787 unknowns for a and 788 for b, of which only the live cells are determined by data:
if 150 of 261 cells are dead on the fit rows as they are on seeds 1 to 20, the live count is 3 x (1 + 111) + 1 =
337 for a, so about 22 fitted rows per live unknown for a and about 44 for b. **Both figures move with
`rows24.py`'s real counts and with the real dead list.**

**Reproducibility, before any selection score** (the design's item 9, and the failure of ticket 10 it answers):

- each candidate is fitted **in two separate processes on `hpc-debian`**, each re-reading the logs and
  re-solving, and the coefficient file's sha256 must be **identical** across both processes and after transfer
  to `local`. `precheck.txt` part 7 is the same check on the same arithmetic in three processes on `local`,
  and it passed;
- if the two digests differ, **nothing proceeds**: a closed-form fit that does not reproduce is a defect, not a
  spread to be measured, and this is the one property that lets M be shipped as bytes in the repository instead
  of as a recipe;
- **every script prints the sha256 of itself, of every module of this directory it imports, and of the
  coefficient file it read**, and `eval24.py` writes the coefficient digest into the csv's `model_sha` column, so
  every score names the coefficients that produced it. `../23/RUN.txt` section 20.3 is why: ticket 10's first
  tuning log named no code digest, and its adapters could not be tied to any code state.
- **the coefficient files are committed**: `coef_a.npy` and `coef_b.npy`, float64 arrays of shape (5, 262) whose
  rows are the three arms' coefficients in order (each the intercept then the 261 cells), then the fit rows' mu,
  then the fit rows' sigma (each with 0.0 in the leading cell, where the coefficient rows carry their intercept,
  so that one array holds all five rows at one width and `m24.py` asserts that leading 0.0 on load); beside each, a
  `coef_<c>.json` with the rate or rates, the selected penalty, the iteration index and `changed_k` at every
  iteration, the condition numbers, the row counts, the dead-cell names, and the `.npy`'s own sha256. Together
  they are of the order of ten kilobytes (`precheck.txt` part 7: 6288 bytes for three coefficient arrays of the
  same width), so **M, unlike L, is reproducible from the repository alone**.

## 10. The evaluation on seeds 81 to 100

**Six deciders**, every one of them through `port.sim` on a run dict from `../22/label.py`'s `run_dict` (which
carries `"seed"` and calls `port.check_seeds` itself), with `choices = list(port.union_choices())`,
`choice_jitter = 0.1` and `redecide = True`:

| decider | what it is | role |
|---|---|---|
| fixed 3.695 s | `port.const_decider(3.695)` | frontier point, and CONTROL |
| fixed 6.0 s | `port.const_decider(6.0)` | frontier point, and CONTROL |
| fixed 10.043 s | `port.const_decider(10.043)` | frontier point, and CONTROL |
| F | `../22/label.py`'s `F`, built fresh per simulation | comparator, and CONTROL |
| M candidate a | `m24.M` on `coef_a.npy` | scored |
| M candidate b | `m24.M` on `coef_b.npy` | scored |

**L's rows are read from `../23/eval.csv` and are not re-run** (the design's item 10). The untuned model of
ticket 10 is not re-run either and is quoted only where ticket 10 quotes it.

**The streams.** 20 seeds (81 to 100) x 40 replicate streams = **800 runs per decider**, 4800 simulations in all,
with `base = R.rep_seed("pairlag:s6b10p043", 0.2, seed, rep)` passed as **both** `pair_lag_seed` and
`pair_jitter_seed` and `sim_rng = None`, which is `../22/weight.py:149-168`'s convention, ticket 10's step 4, and
the convention `../23/eval.py` scored L under. The csv keeps **ticket 10's own column tuple**
(`../23/eval.py:109-113`), so a row of `eval24.csv` and a row of `../23/eval.csv` are comparable field by field.

**The control, and it is a gate, and it runs before anything of M exists.** The 3200 rows of the three fixed
arms and F must equal `../23/eval.csv`'s 3200 rows of the same four deciders **byte for byte in every column
except `script_sha`** -- which includes `model_sha`, whose value is the literal `none` on those rows
(`../23/eval.csv` line 2). If any row differs, **nothing is scored**: the port path, the stream keying, the run
dict, the weight and the scorer are then not ticket 10's, and no comparison between M and L would mean anything.
This is cheaper and stronger than it looks: it re-derives 3200 archived rows from a different script on a
different day.

**It needs no coefficient file, so it is `eval24.py --control`, its own mode, and section 19 puts it ahead of
`learn24.py`.** It reads only `../22/port.py`, `label.run_dict`, `../22/score.py` and `../23/eval.csv`. **It runs
twice**: once there, as step 3, with nothing of M in existence, and again as the real run's first act, where its
3200 rows also have to equal `eval24_control.csv`'s own, so the second run is a check that the port path did not
move between the two. Both comparisons are printed. Running
it after the dry run, as the earlier draft's order did, would mean that if it failed the user's choice of whether
to reopen the port path was made with M's dry scores and `ana24.py --dry`'s a-against-b gap already printed.
Those are training-seed numbers and the selection rule is mechanical, so that is a weaker leak than a selection
score; it is still a number in hand at a moment the plan admits a design choice is open, and it costs one row of
the order table to remove. See "Changes to the worker's design made in review".

**The two statistics**, both as ticket 10 fixed them:

1. **The selecting statistic**: the mean of `score.score_run(events, t0, 104.29595912307323)[0]` over the 800
   runs of a decider, lower better. **This number, and nothing else, chooses between candidate a and candidate
   b** (user, `決まったこと` 8).
2. **The reported frontier gain**, with a 95 % interval, reusing `../20/predict.py`'s `interp`, `gain`,
   `boot_index`, `ci` and `word`, under ticket 10's own rules, restated here so nothing is left to be
   rediscovered: the per-seed row is `[n4, c4, lat, polls]` in that order; a seed's 40 streams are pooled by the
   column-wise sum, not by a median of per-stream gains; the frontier's three points are this step's own re-run
   arms, stored under `../20/predict.py:115`'s three `ARM_STAGES` keys so that `gain` is called unchanged;
   `boot_gains` is not called, because its `TESTED` keys name `../20`'s two rules; the resample unit is the seed,
   2000 resamples from one (2000, 20) index matrix drawn with `numpy.random.default_rng(20261122)`; a resample
   whose arms are not monotone or whose decider's polls leave the arms yields no gain and is counted and
   dropped, not substituted; and a gain that cannot be read prints the reason with the re-run arms' own poll
   counts.

**How the choice between a and b is made.** The candidate with the lower mean selecting score is chosen, unless
the two are **close**, fixed as `|mean score(a) - mean score(b)| < 297.0 ms`. The band is ticket 10's and its
arithmetic is `../22/weight.txt`'s: the weight's jackknife standard error **1.685586 ms per poll** times the
midpoint **176.220 polls per run** of the two outer arms' step-4 poll counts = 297.0 ms (`../23/ana.py:85-87`,
`../23/README.md`). **What the band is: the amount by which re-solving the weight on another draw of seeds would
move a typical run's score.** It is not a confidence interval on the difference. The verdict words are ticket
10's three, and they are reported as it reported them:

- `|gap| < 297.0 ms` -> **CLOSE**, and the user's relabelling clause of ticket 10 is not reached by this step;
- `|gap| >= 297.0 ms` and the gap lies inside the paired bootstrap's 95 % interval -> **SEPARATED, NOT
  RESOLVED**;
- `|gap| >= 297.0 ms` and the gap lies outside it -> **SEPARATED**.

The chosen candidate is the lower one in every branch, because the user asked for a decision. Beside the band,
and not as the rule: the paired bootstrap's 95 % percentile interval of `score(a) - score(b)` over the same 20
seeds (2000 resamples, the same index matrix, a seed's value being the mean of its 40 run scores), whether it
contains 0, and `1.685586 x |polls(a) - polls(b)|`, the part of the band that does not cancel between two
candidates that poll alike.

**The paired comparisons, pre-registered here and reported whatever they show.** Each candidate against F,
against the fixed 6.0 s arm, and against ticket 10's candidate b, with the seed bootstrap above: six
comparisons, listed before any score exists, so that no comparator is chosen after a number is seen. **This is
the one place this plan is stricter than ticket 10's, whose eight extra comparisons were written after the scores
and whose comparators were chosen then** (`../23/README.md`: この 9 本のうち、凍結した段取りが持つのは 1 行め
だけである). No multiplicity correction is applied to the six and the intervals are read as what 20 seeds do to
each difference; that is stated here rather than discovered later.

**Cost.** 4800 simulations, no card. `../22/weight.txt`'s 40000 simulations took 161.4 s on 5 workers
(quoted in `../23/spec.md`), so the simulations are of the order of a minute, and M's own decisions add about
0.096 ms each (`precheck.txt` part 9: 0.0184 + 0.0713 + 0.0064 ms medians), i.e. of the order of 10 s per
candidate at ticket 10's own decision counts. **This step needs no parallelism, no gate on parallel arithmetic
and no card**, which removes four of ticket 10's own machinery items at once.

## 11. The dry run, and the order that keeps the freeze readable

**After this file is committed**, and before the real evaluation, the whole pipeline is run in its dry mode on
**seeds 1 and 2, streams 0 and 1** -- ticket 10's own `DRY_SEEDS` and `DRY_REPS` (`../23/eval.py:82-83`), which
are training seeds, so a dry score is a score on training data and is read as a proof that the code runs and as
nothing else. The dry control compares the four no-model deciders' 16 rows against `../23/eval_dry.csv`.

**`ana24.py --dry` computes the a-against-b band and the verdict word, prints them, and they are not read.**
Saying so is the point: the rule is fixed in this file before any score exists, the dry seeds are fit rows, and
the dry run's purpose is to prove the arithmetic runs end to end, which includes the branch. **The dry verdict
reaches nothing**: it is not quoted in `README.md`, it is not compared with the real one, and the real run
recomputes it from `eval24.csv`. The alternative not taken is dropping the block from the dry analysis, which
would leave the one branch of `ana24.py` that has three exits untested until the real scores exist.

**No setting changes after a dry score is seen.** How a reader can check that:

- **commit order.** This file is committed alone, before any script of the later steps exists. Every archive of
  the dry run and of the real run is committed after it. `../23/RUN.txt` section 26.3 is the precedent for
  reading the history this way, including its own caveat that a commit time is this machine's clock.
- **digests.** Every archive prints the sha256 of the scripts that produced it and of the coefficient file it
  read, and `eval24.csv` carries both in its own columns. A setting changed after a dry score would change a
  script digest, and the dry archive would name the digest of a file that no longer exists.
- **the coefficient files are the artefact.** The dry and the real evaluation read the same `coef_a.npy` and
  `coef_b.npy`, and both archives print their digest, so a refitted M after a dry score would be visible as a
  changed coefficient digest and not only as a changed script.

## 12. ADR 0004's warning, what was measured about it, and what is not held

ADR 0004 is the record this plan departs from, and the ticket requires this section (Acceptance 7).

**What the warning says.** ADR 0004 rejects a per-decision contextual bandit, because "a bandit learned on A
would rank the short arms on costs that the next decisions, not the chosen arm, produced" (line 11), and rejects
learning "values of whole decision sequences directly" (line 3). It **keeps** one option by name: "Model-free
sequential learning (values of state and arm learned from A's logged sequences by bootstrapping, or on-policy
runs): kept as the next step if the model's predictions fail validation", and warns that "for the 0.5 s arm a 10
s outage spans about 20 consecutive decisions, over which bootstrapping errors accumulate" (line 12). **This step
is that kept option**, and the user's `決まったこと` 5 names it as such.

**Two things about the warning's own standing, both stated plainly.**

1. **Its precondition was never met.** The model ADR 0004 would have fallen back from was never built: the stage
   2 map records 「ADR 0004 のモデルは、まだ作っていない」 (quoted in ticket 11's `決まったこと` 5). **This step
   takes the fallback without its trigger**, and the user's decision is what authorises that, not a failed
   validation.
2. **The 20-decision figure is for an arm this ticket does not have.** ADR 0004's own line 12 names the 0.5 s
   arm, and this effort's arms are 3.695, 6.0 and 10.043 s (`../22/port.py:45`). **The span of one outage in
   decisions for this arm set is in no file of this repository.** It is not in `precheck.txt`, which reads only
   `events.jsonl` and never a run's `results.json`, where the link events live (`precheck.txt` part 0, "not read
   here"). This plan therefore makes no claim about it and `outage.py` is the script that will print it: it pairs
   each `link_down` with the next `link_up` of the same node pair in the `results.json` of seeds 1 to 20 of both
   cells, counts the reschedule decisions of the cut pair inside each outage in both directions, and prints the
   distribution with the outage durations beside it and with the pairing rule's own weakness named (overlapping
   cuts are not resolved). **It is the first script of this directory to open a `results.json`, which
   `RUN.txt` section 5 recorded as unopened, and its output says so.** It influences no setting of this plan, and
   section 14 puts it where that is visible.

**Where the accumulation now sits, and it is not where ADR 0004 put it.** The bootstrap of this step is one step
deep, over a chain of decisions of **one directed pair**. The thing a per-pair value does not represent is the
**coupling across the pairs of the line**: a record crosses four hops, and its wait inside this pair's period was
set partly by where its arrival fell in the next pair's poll phase. `charges.txt` part 9 states it for the cost
("the partition is additive across decisions and is not separable across pairs"), and section 5 of this file
states it for the transition (the period is not policy-free, because the other seven pairs are deciding during
it). **So the accumulation risk of this design is width across the eight directed pairs, not depth in time.**

**What is held.** The per-decision cost itself: `charges.txt` part 1 closes `(lat + censored) + W x N x polls`
against the sum over every period of waiting plus W x N per in-window send, with a difference of 0.0e+00 on 40 of
40 runs asserted to 1e-6 ms per run, 120 censored records of p 0.4 placed at the hop they never crossed, and a
residual of 0.000000 ms in both cells. The cost no decision owns is 0.184 % at p 0.2 and 0.228 % at p 0.4,
largest in one run 0.809 % at p 0.2 (seed 19) and 0.604 % at p 0.4 (seed 6) (`charges.txt` part 1). **So the objective this step learns on sums exactly to the
score it is judged by, up to two tenths of a percent that no decision owns.**

**What is not held, in one list.**

- **No per-decision agreement between the emulator and the replay.** Mode A of `../13/validate.md` fails (39 /
  24 / 41 of 42 cells required). M's values are per decision.
- **No bound on the value of a policy other than the logging one** (section 5).
- **No representation of the cross-pair coupling** anywhere in the model.
- **No measurement of the outage span for this arm set in this repository**, until `outage.py` prints it.
- **No claim that this plan's one-step bootstrap escapes the accumulation.** It replaces depth with a
  single-step projection whose error is weighted by the logging policy's state distribution, and nothing here
  bounds that error at the states M itself would visit.

## 13. The label agreement, the timing, and the inventory

**The agreement (Acceptance 3).** `agree24.py`, **after every score of this step is written**, on the labelled
rows of seeds 81 to 100 of the main cell: 3203 rows, of which **2841 carry a label**, whose arm shares are 640 /
703 / 1498 and whose most-frequent-arm baseline is therefore **1498 / 2841 = 0.5273** (`../23/agree.txt` part 1,
which recomputes the figure ticket 10 pre-registered, and `../23/README.md`). **The baseline is pre-registered
here as 0.5273 before any coefficient exists**, and `agree24.py` recomputes it from the rows it reads and stops
if it does not reproduce it. For each candidate it prints the share of the 2841 rows whose argmin equals the
label, the same per label class, the 3 x 3 table of label against answer, the share of answers falling to each
arm, and the tie count. **No pass or fail is attached** (the ticket's own instruction), and the untuned-model
column of ticket 10 has no counterpart here because M has no untuned state.

**What the agreement does not mean.** The labels are the replay's own hindsight -- the arm that is best now **if F
takes over afterwards** -- and two independent 8-pass labellings of the same decisions agreed on 0.7577 of the
1366 **non-tie** decisions, over seeds 1 to 10 of the main cell (`../22/stability.txt:11,25` and
`../23/README.md`; `labelsig.txt` part 7 quotes it as "decisions both labelled", which is the looser wording). M is fitted to the cost per
second of the s11a logs and to no label: **the s11a logged decisions carry no label at all**, because the labels
were attached to decisions inside a replay roll-in, so there is no row-level join between M's training rows and
the labels (`RUN.txt` section 4, and the ticket's 下調べ). So the agreement is a number about two different
questions meeting, and this step reads it as nothing else.

**How the label rows are read, which needs its own sentence.** `features.label_rows` **refuses** a label seed
above 80 and that refusal is load-bearing for every other script of this directory (`features.txt` part 6 shows
it firing). So `agree24.py` carries **its own reader**: it takes the path and the committed digest from
`features.label_file()` (so the digest check stays single-sourced), reads the gzip bytes, checks the sha256
against `../22/labels.sha256`, parses each line, keeps the rows with `81 <= seed <= 100`, and passes that seed
list through `port.check_seeds` and through its own guard refusing anything outside 81 to 100. **Threading a
range through `features.label_rows` instead would weaken the guard every other script leans on**, which is why
about eight lines are written twice here and the duplication is named rather than hidden. The record of a label
row goes through `features.record_from_label_row`, which is the same road as the replay's.

**The timing (Acceptance 6).** `latency24.py`, on the local machine, **for both candidates**, over **the first
1200 distinct decisions of p 0.2 seeds 1 to 8 in the order the logs hold them** -- `precheck.txt` part 9's own
row set, fixed here so that the two figures are comparable and so that the choice is not made when the script is
written -- times the three parts separately and prints medians, p99 and max **beside 100 ms with no verdict**:
`observe.observation`, `features.vector`, and the three dot products plus the argmin. The seed list and the
count are printed, as `precheck.txt` part 9 prints them. `precheck.txt` part 9 already timed exactly those three on the same machine with another
coefficient set -- medians 0.0184, 0.0713 and 0.0064 ms, p99 0.0260, 0.0819 and 0.0118 -- so the figure this
script adds is M's own, on M's own coefficients, in the same three parts. **Both candidates and not only the
chosen one** -- worker's choice, because at these costs it removes a dependence of the timing on the selection;
the alternative not taken is ticket 10's, which timed the chosen candidate alone.
The three parts are timed apart and never summed into one number, because the sum is comparable with nothing on
record; L's own comparable sum is 70.13 ms (`../23/README.md`), of which 69.47 ms is one forward pass on a GPU.
**It is not a node's loop and it is not a node's machine**, and the loop question stays ticket 12's.

**The inventory (the design's item 13).** `inventory.py`, **before the real evaluation**, prints what has read
seeds 81 to 100 before M's scores exist, **built by a script and not from memory**, because ticket 10 miscounted
this list twice from memory: amendment A2 of `../23/spec.md` said two items and `../23/RUN.txt` section 27.2
finding 4 found three, the omission being `parity.py`'s stream-0 runs of seeds 81 and 100. How it builds the
list:

1. for every `*.csv` of `../23`, `../22` and this directory, it parses the `seed` column and prints the file, the
   distinct seeds in 81 to 100, the row count and the deciders those rows name;
2. for every `*.txt` and `*.md` of those directories, it prints every line matching a decimal token in 81 to 100
   bounded by non-digits, with its line number, and every line matching the role phrases `selection`,
   `81 to 100` and `81-100`;
3. **it asserts nothing about those matches, and the earlier draft's assertion is struck.** The assertion it
   carried -- that no file of this directory matches either test -- **is false of the repository as it stands,
   including of this file**: `spec.md` is a `*.md` of this directory and carries `selection` and `81 to 100`
   dozens of times, and the token test fires 6 times in `charges.txt`, 8 in `features.txt`, 16 in
   `precheck.txt`, 24 in `labelsig.txt` and 27 in `RUN.txt` (row-count columns, "refuses one above 100", "the
   100 ms tick" and the like), with `spec.md` itself at 58. An assertion that stops deliverable 1 on the
   existence of the plan is not a gate, and an exemption list invented after `inventory.txt` is read would be a
   decision taken after seeing a number. **So item 2's printed evidence is the whole of it**, which is what the
   closing sentence below already said it was. See "Changes to the worker's design made in review";
4. it prints ticket 10's own corrected count, **reading every figure off the files item 1 already parses rather
   than typing it**: `../23/eval.csv`'s no-model and adapter row counts by decider, `../23/parity.txt`'s
   single-stream runs of seeds 81 and 100, and `../23/agree.txt`'s labelled row count and most-frequent-arm
   baseline. The only thing typed is the one scratch probe `../23/spec.md` amendment A2 records, because no file
   carries it, and it is quoted from that amendment. **The reason is ticket 10's own defect class**:
   `../23/RUN.txt` 27.2 item 1 is `agree.py:518` typing "tune.txt's 408/568" where the script already held the
   parsed value, and `RUN.txt` section 19 of this directory removed the same thing from three other places.

**It prints evidence and not a verdict**: a token match in prose is not a read of a seed, and the reader judges
each line. **And it cannot see what is not in the repository.** Two things of that kind are therefore written
here, by hand, because no script can print them: **the author of this plan knows ticket 10's seven deciders'
mean scores on these 20 seeds** (`../23/ana.txt` part 2, quoted in section 15's predictions), and knows the four
adapters' scores of `../23/spread.txt`. **Those reconcile with the user's own count of nine**
(`決まったこと` 8, seed 81〜100 は、もう 9 つの決め方と用意の段の試しに使われている): `../23/spread.txt`'s four
rows are a run1, a run2, b run1 and b run2 (`../23/spread.txt` lines 68-76), and the two run2 rows are already
two of `../23/eval.csv`'s seven, so the distinct deciders are **seven plus spread.txt's two first-run adapters =
nine**. That knowledge reached the predictions of section 15 and the band of section 10, and it could not be
unlearned by any ordering of scripts. See Threats 6.

## 14. The proof that no seed above 100 is read, and that the halves stay apart

Every script of this step does all of the following and prints it.

1. **Builds every simulation through `port.sim` on a run dict from `label.run_dict`**, which carries `"seed"` and
   calls `port.check_seeds` itself, so the constructor guard at `../22/port.py:167-169` is on the one path.
2. **Passes every seed list it computes through `port.check_seeds(..., "<script>")`**, which refuses above 100,
   **and additionally through its own `check_learn_seeds`, which refuses anything outside 1 to 80, wherever the
   list is a learning list.** `port.check_seeds` lets 81 to 100 through and therefore cannot stand in for the
   learning guard: a number computed from a selection seed would make the choice between a and b circular. This
   is ticket 10's two-guard structure (`../23/decider.py:93`'s `check_seeds_tuning`) restated, since that module
   cannot be imported (section 8).
3. **Prints the sorted seed set of each role** -- fit 1 to 70, held out 71 to 80, selection 81 to 100 -- and
   asserts that each role's set is inside its range, that the fit and held-out sets are disjoint, and that the
   union's intersection with `range(121, 161)` is empty.
4. **Prints the min, max and count of the seed of every data file it reads**, so the claim covers the data and
   not only the simulations.
5. **Shows every guard refusing, in its own output**, on a mutated input inside the process, as all four scripts
   of this directory already do (15, 23, 17 and 18 mutations, `RUN.txt` after-the-fixes block).

**The one edit to an archived script of this directory, and why it is the smallest one.**
`charges.check_log_seeds` refuses any emulator seed that is not a whole number 1 to `MAX_LOG_SEED = 20`, and that
guard sits inside `charges.load_run`, the only function of this directory that opens a log
(`charges.txt` part 0). The fit reads seeds 1 to 80. So:

- `charges.check_log_seeds` and `charges.load_run` take the bound as a keyword argument **defaulting to
  `MAX_LOG_SEED`**, `precheck.table_of_run` passes it through, and `charges.STORE` reads
  `os.environ.get("STORE", "/home/lab_shared/semindex-eval")`. The message and the default path keep their
  exact values, so with the defaults every line of `charges.txt` and `precheck.txt` is unchanged;
- **all four scripts are re-run and all four archives compared with `cmp`** (`precheck.txt` only through line
  400, its timing marker, as `RUN.txt` section 3 already does), and the new file digests are recorded in
  `RUN.txt`. Not only the two edited scripts: `features.py` opens every log through `CH.load_run` (lines 679 and
  701) and prints that function's refusal message (lines 1003-1004, archived at `features.txt:578`), and
  `labelsig.py` sits downstream of it. With the bound defaulting to `MAX_LOG_SEED` neither output can move, but
  the re-run and the `cmp` are the four commands `RUN.txt`'s own after-the-fixes block already runs, so the
  byte-identity of all four is **measured** rather than argued;
- the learning scripts pass 80 and also run their own `check_learn_seeds`;
- **`charges.txt`'s own sentence then describes the default and not the ceiling the new callers use**, and
  `rows24.py` prints its own cap beside it so the two cannot be confused by a reader of either file.

Alternative not taken: a second loader in a new module, which would put the only code that opens a log in two
places; and raising `MAX_LOG_SEED` to 80, which would rewrite two committed archives and weaken the guard under
which the archived work was done.

**The rule for a wording-only fix to a committed script whose output is archived, fixed here because ticket 10
left it open.** `../23/RUN.txt` 27.3 records five scripts still carrying stale sentences, with the reason: the
scripts sit in a digest that two committed csvs carry, so "they are to be corrected together the next time one
of those scripts is changed for a reason of its own". **This plan adopts that as the rule and names it**, for
every script of this directory and for every script whose digest a csv of this effort carries:

- **a wording-only fix is not applied on its own.** The stale sentence stands, and the correction is recorded in
  `RUN.txt` with the file, the line and the right wording, so the record carries the fix even though the script
  does not;
- **it is applied the next time that script changes for a functional reason**, together with that change, and
  the archive is then re-run, `cmp`'d where it should be byte-identical, and re-digested;
- **the same holds when the script's digest is already written into a csv**, which is the case ticket 10 hit:
  editing the script would make the csv's `script_sha` name a file that no longer exists, and re-running the csv
  to repair that is a bigger change than the wording it fixes;
- **a functional fix is never deferred this way.** The distinction is whether any number or any decision of any
  output moves; if one does, the script is edited and every archive that reads it is re-run.

This is the one rule the deliverable order of section 19 cannot express, because it governs what happens after
an archive exists.

## 15. Predictions, written before the data

Short, each with one clause a number can falsify, and read against ticket 10's archived scores on the same 20
seeds and the same 40 streams (`../23/ana.txt` part 2): F **36150.829**, fixed 6.0 s **37035.920**, L's candidate
b **37210.550**, L's candidate a **37828.390**, fixed 10.043 s **39855.665**, fixed 3.695 s **40811.181**,
untuned **40823.648** ms.

1. **Both candidates score below 39855.665 and below 40811.181**, i.e. below the two outer fixed arms. The
   objective M is fitted to is cost per second, and `charges.txt` part 6 shows the two outer arms paying the most
   per second in the pooled reading at p 0.2 (40.102 and 40.059 against 38.413 for 6.0 s).
2. **At least one candidate scores below the fixed 6.0 s arm's 37035.920.** This is the open question of the
   step and this file takes a side. The reason: on held-out seeds the one-period cost of the rows whose logged
   arm the ridge called cheapest is -81.72 score ms against +88.47 for the rows whose arm it called dearest, a
   monotone spread of about 170 score ms per decision with a run-clustered standard error of 9.57 to 12.61, and
   the three rank groups are three samples of the same state distribution because the log drew the arm
   independently of the state (`precheck.txt` part 4). A per-state ordering with that much room should beat the
   best single arm.
3. **F's 36150.829 stays below both candidates' means.** Agreeing with the design's own prior, and with a
   mechanism: F's choice set is `port.union_choices()`, which holds every interval F reaches by its 1.5x
   lengthening, while M is confined to the three arms the logs drew (ticket, 腕が 3 つしかないことが、M に効く).
   F also reads the resolution M reads.
4. **a and b are separated by the band**, i.e. `|mean(a) - mean(b)| >= 297.0 ms`. **This contradicts the design's
   own prior** and the reason is arithmetic: 297.0 ms is 2.85 polls per run at the weight, the two candidates are
   two deterministic fits on row sets that differ by a whole second cell, and ticket 10's two candidates sat 39
   polls per run apart. Two policies that differ at all are unlikely to land within three polls per run of each
   other. **And unlike ticket 10, a difference here is a difference between the row sets**, because the fit is
   closed form and reproduces bit for bit (section 9) -- ticket 10's own a-against-b gap of 617.840 ms was
   smaller than the 634.457 ms its recipe moved between two runs (`../23/README.md`).
5. **Agreement with ticket 09's labels on seeds 81 to 100 lies within 0.03 of the 0.5273 baseline**, i.e. in
   [0.4973, 0.5573], for both candidates. The labels answer a different question (the best arm now if F takes
   over, measured over 8 passes), and when a linear fit on the same vector was fitted **to the labels** it
   reached only 0.5396 against the 0.5102 baseline of its own held-out rows (`labelsig.txt` part 3). A fit to
   something else should sit at the baseline.
6. **The iteration settles -- `changed_k = 0` by evaluation 5 -- for both candidates.** One printed iteration
   index decides it: settling at evaluation 10 falsifies it, although that is still before the cap of 20 and
   would still ship the converged policy rather than the fallback of section 4.6. The earlier draft carried both
   clauses and an index of 10 would have read as confirmation under one and falsification under the other.

**What would be surprising**, and would be reported as surprising rather than smoothed over: a candidate below
F; a candidate above both outer arms; the two candidates within 50 ms of each other; an agreement with the
labels above 0.60; a condition number above 1e8 at any solve; a rate-diagnostic gap above 2 score ms/s **against
the fitted rows' own rate, which `rows24.py` prints with its run-clustered SE** (section 3, not
`precheck.txt` part 2's every-owned-period figures); `changed_k` oscillating without settling; a repeat of an
earlier policy that is not the previous one, which section 4.6 reports as a cycle.

## 16. Threats

1. **Learning in the emulator, scoring in the replay.** The only agreement on record between the two worlds is
   per cell: `../13/validate.md`'s Mode B passes and its Mode A, the per-decision agreement, fails at 39 / 24 /
   41 of the 42 cells required. **What a reader may therefore not conclude: that a per-decision value fitted in
   the emulator describes the decision the replay asks M for.** Every coefficient of M was fitted on per-decision
   costs from a world whose per-decision agreement with the scoring world is on record as failed, and no number
   of this step bounds the difference.
2. **The values of every policy but the logging one are extrapolation.** The logs hold one policy's
   trajectories; the penalty is chosen on the logging policy's held-out residual; the improved policies'
   evaluations are the same linear family read at states and actions no logged continuation visited. **What a
   reader may therefore not conclude: that M's fitted value, or its fitted rate, says anything about M's own
   cost per second.** M's own rate is measured in exactly one place in this step, the replay score, and that is
   not what the fit produced.
3. **Coupling across the pairs of the line.** A record crosses four hops and each hop's wait is set partly by the
   downstream pair's own decisions, which a per-pair value does not represent (`charges.txt` part 9, and
   section 5 of this file). **What a reader may therefore not conclude: that M's one-step bootstrap is a model of
   the process that produces the score.** It models one pair's own chain and prices the rest of the line as if it
   were the environment, while under M the rest of the line is M as well.
4. **Three intervals only.** The logs drew three, so M can express nothing finer and nothing outside them, while
   the port would accept six (`port.union_choices`). **What a reader may therefore not conclude: that M's score
   is what a learned policy with a finer grid would score.** The user has asked for a finer grid later, and that
   is a new stage, not a re-reading of this one.
5. **Selecting one of two candidates on the same 20 seeds biases the chosen one's reported score.** Lower is
   better, so the winner's figure is the minimum of two draws and is biased in the flattering direction. The bias
   reaches the gain as well, because it is a different delay on the same runs, and it reaches the step's own
   question, because the figure compared against F and against the arms is the chosen candidate's. **What a
   reader may therefore not conclude: that the chosen candidate's score is an estimate of its quality.** It is a
   bound in the favourable direction, and the only clean correction is a set of seeds nobody has looked at, which
   is 121 to 160 and belongs to ticket 12.
6. **The author of this plan knew every earlier score on the selection seeds.** Ticket 10's seven deciders'
   means, the four adapters' means of `../23/spread.txt`, and `../23/parity.txt`'s eight single-stream runs of
   seeds 81 and 100 were all on record and were read while this file was written; two of them are quoted in
   section 15 and one in section 10's band. **What a reader may therefore not conclude: that the predictions and
   the comparators of this plan are independent of the selection half.** They are pre-registered, which is a
   different and weaker property, and `inventory.py` can print the files but not unlearn them.
7. **Unequal random draws between a decider that changes its mind and a fixed arm.** A decider whose send-time
   value differs from its resolution-time value skips `resolve`'s identity early return
   (`../13/replay.py:1103-1109`) and draws one extra `tick_lag` from that pair's stream; the three fixed arms
   never differ and never draw it, F and M do. `../22/README.md` records the measured consequence as a check
   that did **not** hold -- `h2->h3` going from 26 to 27 lag draws, the draw-kind order diverging at the 21st
   draw -- and its own sentence is 「候補どうしが同じ乱数を見ている、とはまだ言えない」. **What a reader may
   therefore not conclude: that the difference between M's score and an arm's score is the difference of the
   decision rules.** **The size of each decider's share of it is a count this step produces, so it is measured
   rather than named**: `eval24.py` prints, **for every decider it runs, the three fixed arms and F included, the
   number and the share of decisions whose returned interval differs from that ordered pair's previously applied
   interval**, beside the per-pair lag-draw counts the port keeps. The three fixed arms must print 0, which is
   the control on the count itself. **And what that count still does not hold: that two deciders with equal
   mind-change counts saw equal random streams.** It measures how often the extra draw is triggered, not what
   the stream did afterwards, because one extra draw shifts every later draw of that pair.
8. **The penalty is chosen on the logging policy's own data and used for every other policy.** `Crit` measures
   how well a penalty's fitted `Q` predicts a four-decision frozen tail **of the logging policy**, and nothing
   says the penalty that does that best is the penalty that makes a greedy policy score well; the step has no
   way to find out without a replay score, which `決まったこと` 10 forbids looking at for a setting. **And
   `Crit` ranks penalties only up to the error of its own reference solve at `lambda_ref`** (section 5), which is
   a second thing nothing here measures. **What a reader may therefore not conclude: that lambda\* is the best
   penalty for M.** It is the best penalty, up to the reference's error, for predicting four steps of the policy
   M is not.
9. **The cells M is blind to, and the cells it can lean on from three rows.** 150 of the 261 cells are constant
   on seeds 1 to 20 of both cells and are listed in `features.txt`; three more move on 2 to 9 of 6423 rows
   (`in_flight`, the two fetch-timeout cells, `o2_missing`). **What a reader may therefore not conclude: that any
   number of this step transfers to a condition in which one of them moves** -- a loaded node with polls in
   flight, a truncated advertisement, a pending poll, a third origin. M's answer there is unconstrained by
   anything it was fitted on, and for the nearly-constant cells it is a coefficient fitted on a handful of rows
   rather than no coefficient at all.
10. **M reads the vector and L read the text, and the differences between the two roads are not about
    precision.** There is no precision difference to find: `../22/observe.py` rounds every age to one decimal
    (`AGE_DECIMALS = 1`, inside `observation`) and every interval and wait to three (`INTERVAL_DECIMALS = 3`)
    **before any vector and any text exists**, and `render` prints the wait with those same three decimals. So
    `x1_waited_s`'s 2205 distinct values (`features.txt` part 4) are three-decimal values the text prints in
    full, and `features.txt` part 3 measured the consequence: 18032 distinct texts, 18032 distinct vectors, 0
    texts rendered by more than one vector and 0 vectors rendering more than one text over all 19291 records.
    Three differences are real instead. **M provably ignores the cells that are constant on its fit rows**,
    because the standardisation holds their weight at exactly 0 (section 7), while nothing stopped L's text from
    reading them. **M's standardisation is fitted on seeds 1 to 70 inside the window of section 2** and is then
    applied to replay decisions and to label rows drawn from elsewhere. **And a send-time decision reaches the
    vector as a different record** (`res_present` 0), while M does not call the model at a send at all
    (section 8). **What a reader may therefore not conclude: that the bijection makes M's and L's inputs
    interchangeable.** It makes them carry the same information on the measured records, which is a statement
    about those records and not about what each decider can use.
11. **One fit per candidate, but a deterministic one.** Each candidate is one draw of one row set, one window,
    one grid and one vector, and nothing here measures the variance over any of them. **What a reader may
    therefore not conclude: that a difference between a and b is a difference between the two cells' logs rather
    than between two arbitrary fits.** What **can** be concluded, and could not in ticket 10, is that re-running
    the recipe reproduces the bytes (section 9), so the difference carries no tuning noise -- the failure
    `../23/README.md` reports, where one recipe's run-to-run move exceeded the difference being read.
12. **The five-cell state, which is about what ADR 0004 permits, is not what M reads.** The user decided that
    (`決まったこと` 7) with the departure named. One half-sentence on the five cells themselves: they are the
    two permitted fields **as they appear in this data**, with the answered case split by whether the reply
    brought anything new (`charges.cell`'s `ans_empty` against `ans_content`), and ADR 0004 line 20 lists that
    split separately as something "the policies of this ADR may not look at". So the five cells are the
    ticket's own wording (`2 つの欄... だけでは 5 通りにしかならず`) and `precheck.txt` part 3's "the most the
    ADR 0004 scalars plus the interval can say", and it is the ADR that says less than the name suggests. `labelsig.txt` part 7 shows the five cells do carry label
    information while never moving a three-arm argmax, so the departure is not free of content on either side.
    **What a reader may therefore not conclude: that M's score is evidence about what a policy inside ADR 0004's
    permitted information set would score.**
13. **p 0.4 rows priced at the main cell's weight in candidate b, and carrying about half of its moment
    conditions.** Section 6: W is a p 0.2 indifference point and the p 0.4 cell's own weight is in no file of
    this repository, so the size of the mispricing is unknown here, not small; and the p 0.4 share of the rows
    is about 0.4995 on `charges.txt` part 6's counts, so b's solution is the projected fixed point of an even
    mixture of the two cells and not of the scored one. **What a reader may therefore not conclude: that
    candidate b's extra rows only add information about the main cell.** They are rows of the same functional
    form with a poll priced at another cell's rate, and they move the shared map itself, which is why section 6
    requires the coefficient difference and the argmin disagreement to be printed before any score.
14. **The window of section 2 removes nearly a quarter of the score window.** The 28 s tail is dropped from
    every fit, so the states of the last quarter of a run are in no fit row while the replay scores M over the
    whole window. **What a reader may therefore not conclude: that M was fitted on the states it is scored
    over.** The end of a run is, by construction, not in its training rows.
15. **`polled_peer` is absent on a fifth of the records.** `../22/observe.txt` counts 6077 of 32219 records,
    0.189, where the state names no such peer, and `features.txt`'s `polled_peer_present` is the cell that
    carries it. **What a reader may therefore not conclude: that M is a per-neighbour decider on those
    decisions.** There its answer is a function of the node's own state and the clock alone, while the interval
    it returns is applied to one ordered pair.
16. **A fifth of M's own decisions in the replay may be the tie rule's.** Ticket 10's candidate a had the
    first-maximum rule deciding 19429 of 95860 replay decisions and up to 0.948 of its non-`long` answers
    (`../23/README.md`, `../23/versus.txt` part 3). M's dot products are continuous and exact ties should be
    rarer, but the count is printed and not assumed. **What a reader may therefore not conclude, until that
    count is read: that M's answers are M's model's answers.**
17. **The five grid penalties may be inseparable on the criterion that chose one of them.** Section 5 prints
    `Crit` and its run-clustered standard error for all five, and nothing says the five differ by more than that
    standard error. `../23/spec.md`'s amendment A5 is the lesson from ticket 10: "candidate a's kept epoch is
    not distinguished from its other two", i.e. the kept thing was one of three the rule could not tell apart on
    the quantity it used. **What a reader may therefore not conclude: that `lambda*` is distinguished from its
    grid neighbours.** If the printed interval covers the neighbours, M is one of several deciders the criterion
    could not separate, and the one the tie rule or the noise happened to leave at the minimum.
18. **Every interval of this step is a 20-seed interval.** The bootstrap unit is the seed and this step has 20
    of them, where `../20` used 40, so the half widths should be about sqrt(2) wider than `../20`'s -- **an
    inference, not a measurement**. `../23/ana.txt` part 3 printed half widths of 2.243 % to 2.560 % on gains of
    0.0411 to 0.0631 at this seed count, and `../23/spec.md` Threat 2 carries the same caveat: "a 'not-seen'
    verdict at 20 seeds is weak evidence of no effect". This step reads six pre-registered paired comparisons
    and the frontier gain against such intervals, and its a-against-b verdict branches on whether the gap lies
    inside one. **What a reader may therefore not conclude: that any of the six comparisons containing 0, or the
    `SEPARATED, NOT RESOLVED` branch, says the two deciders are alike.** It says 20 seeds did not resolve them.
19. **The cells M leans on drift through a run, and nothing in this step separates reading a state from reading
    the clock.** `precheck.txt` part 3's largest standardised coefficients are `o2_hi`, `o1_hi`, `x8_age_s`,
    `o1_p1_present` and `consecutive_poll_timeouts`, and that file's own text says the backlog, the age of the
    oldest exchange and the pair's exchange count "all move through a run, so part of what it explains is WHERE
    IN THE RUN the decision sits". That is load-bearing twice here: section 2's window removes the last 28 s of
    every run, and section 4.7's rate diagnostic can carry a systematic sign across all roughly 560 blocks for
    exactly this reason. **What a reader may therefore not conclude: that M's ordering of the arms is a function
    of the state rather than of where in the run the decision sits.** Separating the two needs a run-position
    control that this step does not build.

## 17. Every worker's choice, in one place

| choice | reason | alternative not taken |
|---|---|---|
| the fit window is `10000 <= t_rel_ms <= 102000` | arm-independent by construction, and it guarantees two further decisions so the terminal exclusion drops nothing arm-dependently | one period's ceiling (116000), which leaves the terminal drop arm-dependent; or no upper bound at all |
| the period ceiling is 14.0 s | 11.0473 + 2.100 = 13.147 s derived, rounded up to the whole second; observed maxima 13.097 and 13.144 (`charges.txt` part 5) | 13.147 s exactly, with no slack over one observed maximum |
| terminal rows contribute no equation and still serve as successors | Q at their successor does not exist, and anchoring it at 0 would fix the common constant through them instead of through the normalisation | reaching the pair's last decision's own vector, a successor whose period the agent's stop cuts off |
| the normalisation is `u . w = 0` (the three block intercepts sum to zero) | it is orthogonal to the one null direction and fixes nothing else | `sum_i Q(x_i, a_i) = 0`, which mixes the anchor with the row distribution |
| the logging policy's next action is the arm the log drew | one next-action rule for every evaluation, and it alone lets Check A's identity be re-derived through the block endpoints (the identity itself holds for both rules) | the 1/3 average over the three arms: lower variance, two rules, no independent endpoint walk. Its variance cost is printed at `lambda*` under both rules |
| A, t, b carry 1/m | lambda means the same on a's rows and b's twice-as-many rows | sums, as `precheck.ridge` uses, where lambda changes meaning with the row count |
| the grid is (1e-4, 1e-3, 1e-2, 1e-1, 1.0) | brackets `precheck.py`'s own 0.00327 per row by about one and a half decades below and two and a half above; 0 would be singular on the dead cells | nine half-decade values, spending more of the criterion's own rows -- about 46 % of an extrapolated 1060 for candidate a -- on a setting this step cannot separate from noise |
| the penalty is chosen by the frozen-tail criterion `Crit`, not by the held-out squared one-step residual | `Crit`'s target is built once at `lambda_ref` and is the same number for every penalty, while the one-step residual's noise floor falls with the penalty and drags the argmin to the grid maximum | the held-out squared one-step residual, which the design handed here asked for and which stays printed as a diagnostic in both its forms |
| the reference penalty is `lambda_ref = 1e-4` and the horizon is `n = 4` | both fixed before any number of this step is read, which is the only thing that makes `Crit` comparable across the grid; the cost is that the ranking is up to the reference's own error | a larger reference or a longer horizon, either of which would be a second free setting |
| `Crit`'s rows are the held-out rows with `10000 <= t_rel_ms <= 60000` | read off the decision's own instant alone, so the row set is arm-independent for section 2's reason, and every such row has `n + 1` further decisions | every held-out row that happens to have `n` further decisions, which is a selection on what the drawn arms reached and is exactly what section 2 forbids |
| ties in the penalty criterion go to the larger lambda | mechanical, and it prefers the smoother fit when the criterion cannot tell them apart | the smaller lambda, which prefers the better in-sample fit |
| the rate comparison with the logged rate is a diagnostic and not a gate | the plan's own arithmetic expects it to fire for a legitimate reason (about 0.57 against an inferred SE of about 0.45), so a gate would leave the step's most likely path ending in a decision taken with `rho_hat` in hand | keeping it a gate with a bar pre-registered at section 15's "surprising" 2.0 score ms/s, which would still make the endpoint term a stopping condition |
| the iteration carries a rate bound, a weight-norm bound and a cycle test | `changed_k` is only a fixed point of the greedy map and the condition number does not move with the weights, so neither existing safeguard can see a diverging iteration | the condition bar alone, which is three to eight decades away from anything the iteration does |
| the 3200-row control is `eval24.py`'s own mode and runs before `learn24.py` | it needs no coefficient file, so nothing forces it behind the dry run, and behind it a control failure is read with M's dry scores already printed | running it as the real evaluation's first step, as the design handed here implied |
| the cap is 20 evaluations | about 20 solves and 20 passes, i.e. minutes, so the cap is not a cost decision | 10, which would make the fallback more likely to ship |
| the fallback is `pi_greedy(w_0)` | fixed in advance, needs no comparison, and is the one improvement whose evaluation the logs can test | the last iteration's policy, which would ship a policy the cap caught mid-oscillation |
| the condition bar is 1e10 | leaves about 6 digits of double precision; set to catch a singular system, not to trim an ordinary one | 1e12, or a bar on the smallest singular value, which would need a scale this plan has not measured |
| one rate per cell for candidate b | the two cells' rates are 39.5698 and 49.5153 (`precheck.txt` part 2) and rho multiplies tau, the term that ranks the arms | one pooled rate, the literal reading of the design handed here |
| M's send-time default is 6.0 s | L's, so M and L differ in the rule and not in the convention | F's 3.695, which would make M differ from L in two things at once |
| the latency is measured for both candidates | at these costs it removes a dependence of the timing on the selection | ticket 10's, which timed the chosen candidate only |
| `agree24.py` carries its own label reader | `features.label_rows`'s refusal above seed 80 is what every other script of this directory leans on | threading a seed range through that guard, weakening it everywhere |
| the seed cap is a keyword argument of `charges.load_run` | the only code that opens a log stays in one place and both archives stay byte-identical | a second loader; or raising `MAX_LOG_SEED`, rewriting two committed archives |
| `versus24.py` is a separate script from `ana24.py` | it is the only script reading two csvs, so the selecting statistic stays in a script that cannot depend on ticket 10's rows | one script, one output, fewer files |
| the six paired comparisons are pre-registered here | ticket 10's eight extra comparisons were written after the scores and their comparators chosen then | reporting them post hoc as ticket 10 did |
| the coefficient files are committed | M is then reproducible from the repository alone, which no adapter of ticket 10 was | keeping them outside, as ticket 10 had to for 10 GB of weights |

## 18. Deviations from the design handed to this file

1. **The fit window's upper bound is two period ceilings (28 s), not one.** With one, the terminal-row exclusion
   stays arm-dependent, because a longer arm pushes the successor later and a later successor is more often the
   pair's last decision. Section 2 carries the derivation and the guarantee.
2. **Candidate b gets one rate per cell, with one extra instrument, instead of one pooled rate.** The design's
   single rho is solvable but misspecified: the two cells' logged rates are 25 % apart and rho multiplies tau.
   Section 6 gives the well-posed system and the dimension count (788 = 788).
3. **The design's ridge-with-one-normalisation count is right for candidate a, and the reason it is needed is
   sharper than "differential values are defined up to a constant".** The constant is a null direction of the
   regressors that the ridge cannot remove, because `u^T P = 0`; section 4.2 shows that, and shows that rho's
   identification needs the instrument block to have full row rank, which is a printed number and not an
   algebraic consequence of `u^T P = 0`.
4. **The design's built-in check on rho is split into a gate and a diagnostic.** The exact statement is Check A,
   an algebraic identity that holds to solver tolerance and is the gate a construction error fails, with its
   reach and its blind spots stated and with direct assertions replacing the catches it was credited with. The
   design's statistical comparison with the logged rate becomes **a printed diagnostic and not a gate**, because
   by Check A its gap is exactly the fitted blocks' endpoint-value difference per fitted second and this plan's
   own arithmetic expects that to exceed one standard error. Both are in section 4.7, and the change is in
   "Changes to the worker's design made in review".
5. **The outage span in decisions is not in the repository.** `precheck.txt` opens no `results.json`. The claim
   is not made from the scratch report; `outage.py` is named, its inputs and output are fixed, and section 12
   says it is unmeasured in the repository until that script runs.
6. **The p 0.4 cell's own poll weight is not in the repository** and the claim that needed it is rewritten to say
   what pricing a p 0.4 row at W does and nothing about the size of it (section 6, Threats 13).
7. **`../23/decider.py`, `../23/ana.py` and `../23/versus.py` cannot be imported** (the first raises at import
   without a ROOT; the other two run their whole analysis at import, which is the property `labelsig.py` already
   measured with `ast` for `../23/rows.py`). `../23/decider.py`'s send-time convention is nevertheless tied back
   to its source by an `ast` parse (section 8). Their conventions and their arithmetic are therefore restated, and
   the restatements are gated on reproducing the archives: `ana24.py` must reproduce `../23/ana.txt` part 2's
   seven means and part 3's four gains from `../23/eval.csv`, and `versus24.py` must reproduce
   `../23/versus.txt` part 2's a-against-b line (617.840 ms, -58.424 to 1357.299) from the same csv, or each
   stops.
8. **Two archived scripts of this directory take one new keyword argument each** (`charges.check_log_seeds` /
   `charges.load_run` and `precheck.table_of_run`), with the defaults unchanged, because the fit reads seeds 1 to
   80 and those guards stop at 20. **All four archives** of this directory are re-run and `cmp`'d and the new
   digests are recorded, and section 14 fixes the rule for a wording-only fix to a script whose output is already
   archived. Section 14.
9. **The latency is timed for both candidates** rather than the chosen one, and the agreement has no untuned
   column, because M has no untuned state.

## 19. Deliverables

**Before any fit** (this step): `spec.md`. Already here: `features.py`, `charges.py`, `precheck.py`,
`labelsig.py`, their four archives, and `RUN.txt`.

**Then, in this order.** The order is chosen so that nothing that could influence a setting, and nothing of M,
is seen before the setting is frozen: `spec.md` is committed first and fixes every setting; the edit of section 14
precedes the first script that needs it, because `rows24.py` cannot read seed 21 without it; `inventory.py` runs
before any score of this step exists, which is what its subject requires; **the 3200-row control runs before any
coefficient of M exists**, so a control failure is read with no score of M in hand (section 10); `rows24.py` and
`outage.py` read only logs of the learning half or earlier and choose nothing; `learn24.py` sees no seed above 80
and no replay score at all; the dry run precedes the real one; and `agree24.py` and `latency24.py` run after the
scores, which removes even the opportunity for an agreement number or a timing to reach the selection.

| # | file | what it is | reads | where it runs | output |
|---|---|---|---|---|---|
| 0 | `spec.md` | this plan, committed alone | -- | local | -- |
| 1 | the section 14 edit | the seed-bound keyword argument in `charges.check_log_seeds` / `charges.load_run` and `precheck.table_of_run`, then all four scripts of this directory re-run, their archives `cmp`'d (`precheck.txt` through line 400) and the new digests recorded | the four scripts and their archives | local | four `cmp`s and a `RUN.txt` block |
| 2 | `inventory.py` | what has read seeds 81 to 100 before M's scores exist, built by the script (section 13) | the csv and text archives of `../22`, `../23` and this directory | local | `inventory.txt` |
| 3 | `eval24.py --control` | the 3200 rows of the three fixed arms and F on seeds 81 to 100, compared with `../23/eval.csv` byte for byte in every column except `script_sha`; nothing of M exists yet, and step 10 runs it again and compares against this csv too | `../22/port.py`, `label.run_dict`, `../22/score.py`, `../23/eval.csv` | local | `eval24_control.csv`, `eval24_control.txt` |
| 3b | `push_inputs24.sh` | transfers filtered `s11a` seeds 1 to 80 logs (`events.jsonl`, `meta.json`, `results.json`, `*.experiment.json`) and worktree code to `ROOT/store` and `ROOT/repo24` on `hpc-debian`, verified by sha256 on both sides | `s11a` seeds 1 to 80, worktree code | local -> hpc-debian | `ROOT/store`, `ROOT/repo24` |
| 4 | `rows24.py` | the row counts of both candidates, the window, the per-arm 0/1/2 in-window-send table, the dead and nearly-constant lists on the fit rows, the fitted rows' own logged rate and its run-clustered SE, the p 0.4 row share, the per-pair last-decision instants, the `action_prob` assertion, the construction assertions of section 4.7, the two seed guards shown refusing | the s11a logs of seeds 1 to 80, both cells, through `charges.load_run` with cap 80 | hpc-debian | `rows24.txt` |
| 5 | `outage.py` | the span of one outage in decisions for this arm set, with the outage durations and the pairing rule's weakness (section 12) | the `results.json` and `events.jsonl` of seeds 1 to 20, both cells | local | `outage.txt` |
| 6 | `m24.py` | the block map, the evaluation solve, the policy iteration with its bounds and cycle test, Check A and the construction assertions, class M, the restated conventions, the `ast` check on `../23/decider.py`, and a `self_check` that shows every card-free check failing on a mutated input. Imported, no `main` | `../23/decider.py` as source text only | local | -- |
| 7 | `learn24.py` | the fit of both candidates: the `Crit` curve and the one-step residual diagnostics over the grid, the instrument block's rank, the iteration with its `changed_k`, bounds, cycle test, condition numbers and rates, the rate diagnostic and the centring gate at `lambda*`, both next-action rules at `lambda*`, the a-against-b coefficient difference and argmin disagreement, the two-process digest comparison, and the coefficient files | `rows24.py`'s own row builder, i.e. the logs of seeds 1 to 80 | hpc-debian | `learn24.txt`, `coef_a.npy`, `coef_a.json`, `coef_b.npy`, `coef_b.json` |
| 8 | `eval24.py --dry` | the whole pipeline on seeds 1 and 2, streams 0 and 1, with the dry control against `../23/eval_dry.csv` | the coefficient files, `../23/eval_dry.csv` | local | `eval24_dry.csv`, `eval24_dry.txt` |
| 9 | `ana24.py --dry` | the same arithmetic on the dry csv, the band branch included and not read (section 11) | `eval24_dry.csv` | local | `ana24_dry.txt` |
| 10 | `eval24.py` | the six deciders x 20 seeds x 40 streams through the port, the control re-run as its first act and compared against both `../23/eval.csv` and step 3's own csv, the freshness assertion on M's type, the three-arm assertion, the tie and arm counts, and the mind-change count per decider (Threats 7) | the coefficient files, `../23/eval.csv` | local | `eval24.csv`, `eval24.txt` |
| 11 | `ana24.py` | the selecting score, the a-against-b rule against the 297.0 ms band with its three verdict words, the paired bootstrap, the frontier gain with its 95 % interval; gated on reproducing `../23/ana.txt`'s seven means and four gains from `../23/eval.csv` | `eval24.csv`, `../23/eval.csv` | local | `ana24.txt` |
| 12 | `versus24.py` | the six pre-registered paired comparisons, each candidate against F, against the fixed 6.0 s arm and against ticket 10's candidate b; gated on reproducing `../23/versus.txt`'s a-against-b line | `eval24.csv`, `../23/eval.csv` | local | `versus24.txt` |
| 13 | `agree24.py` | the label agreement on seeds 81 to 100 against the pre-registered 0.5273 baseline, no verdict | the label file, through its own reader (section 13) | local | `agree24.txt` |
| 14 | `latency24.py` | one decision in three parts beside 100 ms, both candidates, no verdict | the coefficient files, the logs of seeds 1 to 20 | local | `latency24.txt` |
| 15 | `README.md`, `RUN.txt` | the result in Japanese, and the commands with their real output and their digests | -- | local | -- |

Every one of these scripts shows each of its own checks failing on a mutated input, in its own output, and a
check that cannot be made to fail by any input is named as such rather than counted -- the discipline all four
scripts of this directory already follow (`RUN.txt` section 6, item 5).

**What stays out of the repository**: nothing but the logs and the label file themselves. There is no card, no
adapter and no external weight file in this step (learning runs on CPU on `hpc-debian` under `ROOT`), so the
coefficient files, every script and every archive are committed.

## 20. What this plan does not decide

**Nothing, in the sense of a gate whose consequence is unnamed.** Every gate of this plan now names its own
response, and the responses are of one kind only: find the defect and re-run. The list, so that no reader has to
assemble it:

| gate | section | the only response |
|---|---|---|
| Check A's identity, at any penalty or iteration | 4.7 | the construction is wrong. Find it with the assertions beside it (the penalty matrix, the normalisation row, tau's range, the successor key) and re-run. Nothing proceeds meanwhile |
| any construction assertion of section 4.7 | 4.7 | the same: the named quantity is wrong, fix it and re-run |
| the held-out centring gate, at `lambda*` | 4.7 | a held-out level error of tens of score ms per row is a wrong fit and not a boundary term. Find it and re-run |
| the condition bar at k = 0 | 4.6 | the logging policy's evaluation is unusable and every later number leans on it. Find the singular structure and re-run |
| the two processes writing different coefficient bytes | 9 | a closed-form fit that does not reproduce is a defect. Find the non-determinism and re-run |
| the 3200-row control against `../23/eval.csv`, at step 3 and again at step 10 | 10 | the port path, the stream keying, the run dict, the weight or the scorer is not ticket 10's, or it moved between the two runs. Find which and re-run. The first run is before any coefficient exists, so no score of M is in hand |
| `rows24.py`'s window assertions (a fitted tau above 14.0 s, a terminal fitted row) | 2 | the window's argument is void. Find the row and re-run |
| the archive reproductions (`../23/ana.txt`'s means and gains, `../23/versus.txt`'s line) | 18 item 7 | the restatement of ticket 10's arithmetic is wrong. Fix it and re-run |

**The iteration's own exits are not gates** and need no response: the cap, the rate bound, the weight bound and
the condition bar at `k > 0` each name the M that ships (section 4.6), and the rate comparison with the logged
rate is a printed diagnostic that stops nothing (section 4.7).

**What is genuinely open, and it is the user's.** One item: whether the step's **result** reopens anything. If
the rate diagnostic's gap is large, or `Crit` cannot separate the five penalties, or the iteration reaches its
fallback, or M scores above F, the step still completes and reports; **what the next plan should be is the
user's**, because reopening the window, reopening the pooling, reopening the arm set and accepting M as it stands
mean different things and this plan has no basis for choosing between them. That is a decision about the next
ticket and not a decision this step's own scripts need.

Everything a later script needs is above: the row, the window and its ceiling, the per-arm send table, the
equations and their dimensions, the normalisation, the identification rank, the penalty grid, its frozen-tail
criterion with its reference penalty, horizon and row window, the diagnostics that may not select, the
next-action rule and its measured cost, the iteration with its test, cap, bar, bounds, cycle test and fallback,
the terminal rule, the two rates of candidate b and its estimand, the standardisation, the decider's two
conventions and its assertions, the `ast` source check, the two candidates, the reproducibility rule, the six
deciders, the streams, the control and its position, the two statistics, the band and its three verdict words,
the six pre-registered comparisons, the baseline of the agreement, the three parts of the timing, the inventory's
construction, the seed guards, the archived edit and the rule for a wording-only fix, the predictions and the
threats. **The five things the earlier draft of this plan left to the implementer are now fixed in their own
sections**, and they are five and not four: the penalty at which each check of section 4.7 is read (4.7); the
standard error's denominator for candidate b (4.7 and 6); **which 1200 logged decisions `latency24.py` times**
(section 13); **which single logged row the repeat check reads twice** (section 8); and **where the archived-script
edit of section 14 sits in the deliverable order** (section 19, step 1). `決まったこと` 4, which the earlier
draft resolved silently, is carried in Names.

## Changes to the worker's design made in review

Two reviewers read this plan before it was frozen. Each item below is a change to what a later script will do,
not a note beside it, and the section named carries the change itself. Nothing here is a second version of an
equation: where a criterion or a check was replaced, the replaced form is either gone or stated as a diagnostic
that may not be read as a selector.

1. **The penalty is selected by a frozen-tail criterion and not by the held-out squared one-step residual**
   (section 5). The residual's expectation carries a double-sampling variance term that depends on the weights
   and therefore on the penalty being scored, so shrinking the weights shrinks the criterion's own noise floor
   and the argmin is dragged to the largest penalty of the grid, whether or not the flatter `Q` is nearer the
   truth. The replacement builds its target once at a reference penalty fixed in advance, so the target is the
   same number at every grid value and only the prediction moves.
2. **`Crit`'s row set is cut by the decision instant, `10000 <= t_rel_ms <= 60000`, and not by whether a row
   happens to have four further decisions** (section 5). Selecting the rows by what the drawn arms reached would
   keep a short-arm row more often than a long-arm one, which is the exact bias section 2's window exists to
   remove. It costs about 46 % of the held-out rows and `learn24.py` prints the count.
3. **The order in which the checks of section 4.7 are read, and the penalty each is read at, are fixed**
   (section 4.7). The penalty is selected first; Check A runs at every penalty and every iteration because it is
   algebra; the centring gate and the rate diagnostic run at `lambda*` alone. Without this an implementer meeting
   a failure could satisfy the letter of the plan by reading the checks at another grid point, which is a decision
   taken after seeing a number.
4. **The rate comparison with the logged rate is a diagnostic and no longer a gate** (section 4.7, and
   Deviations 4). The plan's own arithmetic puts the expected endpoint term at about 0.57 score ms/s against an
   inferred standard error of about 0.45, so it was a gate the plan expected to fire for a legitimate reason, and
   its failure would have been answered with `rho_hat` and the whole fit in hand. It is replaced by a printed gap,
   the per-block endpoint values, their sign distribution and the gap's run-clustered standard error.
5. **Check A's claimed catches are replaced by direct assertions, and its endpoint walk must be re-derived from
   the chain** (section 4.7). Check A is a u-combination of the system's own rows, so a tau in milliseconds, a
   ridge applied to the three intercepts and a normalisation row other than `u` all leave it exact; and computed
   off the stored `z` array it holds for any `z` whatever. The assertions on the penalty matrix, the normalisation
   row, tau's range and the successor key are cheaper and cannot be fooled.
6. **The iteration carries a rate bound, a weight-norm bound and a cycle test, with the existing fallback as the
   single consequence** (section 4.6). `changed_k` is only a fixed point of the greedy map and the condition
   number does not move with the weights, so neither of the plan's two safeguards could see a rate or a weight
   vector drifting, and a repeat of an earlier policy that is not the previous one was reported as drift rather
   than as a cycle.
7. **The standard error's denominator is one cell's 70 fit runs and not candidate b's 140** (section 4.7 and
   section 6). The two readings differ by sqrt(2), and section 6 compares each cell's rate with that cell's own
   logged rate, so the per-cell denominator is the one that matches.
8. **The in-window rates are quoted where the fit's expectations are set, and `precheck.txt` part 2's figures are
   labelled as every-owned-period** (section 3, section 4.7, section 15). The fitted set is the in-window subset,
   whose pooled rates are about 43.14 and 54.34 score ms/s against the quoted 39.5698 and 49.5153, a shift of more
   than four of the quoted standard errors.
9. **The window's arm-independence claim is narrowed to the row selection, and the residual it cannot remove is
   printed per arm** (section 2). The send-time provisional timer leaves 10 owned periods at p 0.2 and 5 at p 0.4
   with no send at all and 9 and 3 with two, which no bound in decision time can reach, so `rows24.py` prints the
   0/1/2 in-window-send counts over the fitted rows by arm and by cell.
10. **Candidate b's estimand is stated and the step prints how far the shared map moved** (section 6). With one
    shared w, b's solution is the projected fixed point of a row-count mixture in which the unscored cell carries
    about half the weight, so the two rates fix the level of that mixture and not which mixture it is. The
    coefficient difference and the share of a's fit rows where a's and b's argmins disagree are printed before any
    score.
11. **Section 4.4's second reason for the logged next-action rule is narrowed to what holds, and its cost is
    measured** (section 4.4). The general identity holds for the 1/3-average rule too, since `u . z_i = 0` for it
    as well; only the telescoped walk does not. The variance cost is one extra solve, so both rules' `rho_hat`,
    block intercepts and held-out mean residual are printed at `lambda*`.
12. **The send-time default is tied back to L's own code by an `ast` parse of `../23/decider.py`** (section 8),
    which the earlier draft said no check of this step could do. `SEND_START = ARMS[1]` and the send branch's
    return of the per-pair memory's default are read from the source text without importing it, and what the
    check cannot cover is named.
13. **`inventory.py`'s assertion is struck and only its printed evidence remains** (section 13). The assertion was
    false of the repository as it stands, including of this file, so it would have stopped the first script after
    the freeze, and any exemption list would have been written after `inventory.txt` was read.
14. **`inventory.py` reads the counts it prints off the files it already parses** (section 13), rather than
    carrying them as typed prose, which is the defect class `../23/RUN.txt` 27.2 item 1 recorded and this
    directory's `RUN.txt` section 19 removed from three other places.
15. **The 3200-row control is its own mode and runs before any coefficient of M exists** (section 10,
    section 19). It needs no coefficient file, and behind the dry run a control failure would have been read with
    M's dry scores and `ana24.py --dry`'s gap and verdict already printed.
16. **The section 14 edit is a numbered step of the deliverable order, all four archives are re-run and `cmp`'d,
    and a rule for a wording-only fix to an archived script is written** (section 14, section 19). `rows24.py`
    cannot read seed 21 before that edit, `features.py` and `labelsig.py` also route through
    `charges.load_run`, and ticket 10's open wound was a digested script whose printed prose was wrong after its
    archive existed, with no rule for it.
17. **Three threats are added and one is replaced** (section 16). Added: 17, that the five penalties may be
    inseparable on the criterion that chose one; 18, that every interval of this step is a 20-seed interval; 19,
    that the cells M leans on drift through a run and nothing separates reading a state from reading the clock.
    Replaced: Threat 10, whose claimed precision difference between M's vector and L's text does not exist --
    `../22/observe.py` rounds every age and every interval before either road begins, and `features.txt` part 3
    measured zero collisions in both directions -- and which now names the three differences that are real.
18. **Prediction 6 carries one threshold** (section 15), where it had two that the same printed iteration index
    could satisfy and falsify at once.
19. **`latency24.py`'s row set and the repeat check's row are fixed in this file** (section 13, section 8), and
    `決まったこと` 4 is carried explicitly (Names), so the four decisions the earlier draft left to the
    implementer are decided here. Section 20 now lists every gate with its single response and names what is
    genuinely open.
20. **Smaller corrections of the same kind, each in its own section**: the 2-of-10 and 5-of-10 filter counts are
    attributed to the comparisons `charges.txt` part 7 prints (section 2); the near-cancellation's spread is the
    in-window 5.8 % and p 0.4's is 11.0 % (section 3); the two-ceiling argument's stop is read from the
    per-pair last-decision instants `rows24.py` prints and not from `../13/replay.py`'s constant (section 2);
    rho's identification is a printed rank of the instrument block, with its two smallest singular values,
    and not an assertion (section 4.2); the in-sample centring statement is named as an identity and the held-out gate's
    detectable level is pre-registered (section 4.7); the dead list is 20 cells per origin (section 7); the nine
    deciders of `決まったこと` 8 are reconciled with seven plus two (section 13); the grid brackets
    `precheck.py`'s penalty by about one and a half decades below and two and a half above (section 5); the
    pseudocode's closing line names only the converged exit (section 4.6); the row-count extrapolation cites
    `charges.txt` part 6 (preamble); the 0.809 % run names its cell (section 12); the label-stability figure is
    over non-tie decisions of seeds 1 to 10 of the main cell (section 13); and the five cells are named as the
    two permitted fields as they appear in this data, with ADR 0004 line 20's separate listing noted
    (Threats 12).
