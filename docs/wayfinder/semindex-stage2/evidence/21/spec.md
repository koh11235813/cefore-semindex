# evidence/21 spec: the randomly-drawn stage, with a re-decision at every resolution

Written 2026-11-22, before any run of this step. Ticket `../../tickets/08-choice-redecide.md`, accepted by the
user the same day (its 決まったこと section). The pass lines below are translated from that ticket's Acceptance
3, 4 and 5 and are not moved after the runs. The stage settings and every predicted number are frozen here
before the first run. Where this file says something the ticket does not, or departs from it, it is marked as
such.

## Question, and what the step is for

The ticket names three purposes.

**(a) Make the three ways of deciding decide at the same instant from the same input.** The choices mode draws
the N+1-th interval at the N-th poll's **send**, so the state logged beside that draw describes the poll
before last. A `state_poll_rule` treats the send-time interval as provisional and decides again when the poll
resolves. Two makers that decide at two different instants cannot be compared as two answers to one question,
so the uniform draw is moved onto the rule's instant. That move is the source change this step runs; it is
merged (`1c3f758`, merged as `a8add2c`).

**(b) Leave a log a learner can read.** Every re-decision is a uniform draw over three arms and records
`action_prob` 1/3, so an off-policy estimator has a real probability to divide by, and the row carries
`tracker.decision_state(neighbor)` so the learner sees the per-origin progress the two-field row could not
show. Ticket 11 builds a policy from these rows; this step only produces them.

**(c) The first testbed runs of the re-decision branch.** No run anywhere was made with the knob on. The 200
runs are therefore also the first measurement the replay's copy of the branch can be held against, which is
pass line 5.

Not answered here: which decision maker is better, since this step runs none; whether a policy can be built
out of these rows, which is ticket 11; whether a decision maker can be put in the live loop, which is ticket
12; p 0.5, p 0.6, p 0, the on/off loads `wo` and `wv`, and finer arms. **This stage's delay is not a quantity
any policy is compared on**: a uniform draw is not a policy, and the delay is predicted here only so that the
replay and the testbed can be checked against each other.

## The knob

- **The flag is `--choice-redecide`, the config key `state_poll_choice_redecide`.** The ticket's
  `--poll-choice-redecide` was a provisional name and was replaced when the change was written, to match the
  naming of `--choice-jitter` and `--interval-choices`. `evalgen` and `evalrun` both carry the flag.
- **What it does**, read off `src/semindex/engine.py` at `a8add2c`. At a send: one draw from
  `poll_choice_rng`, one from `poll_choice_jitter_rng`, a `state_poll` row, and the jitter draw is kept on the
  poll. When the **newest** poll to a neighbour resolves, answered or past its lifetime: exactly one more draw
  from the same `poll_choice_rng`, the poll's own send-time jitter draw applied to it, the next poll time set
  to `max(sent_at + applied, now)`, and a `state_poll_reschedule` row with `action_prob = 1/len(choices)`. A
  reply to a poll that is not the newest draws nothing and writes no reschedule row; a reply to a poll already
  resolved as timed out writes a `state_poll_late_reply` event instead.
- **The reschedule row's state is snapshotted after the reply has been folded into the tracker** (user
  decision, 2026-11-22). The row is read as everything the node could observe at the instant it chose the next
  interval, and by that instant the node has read the reply, so a pre-fold snapshot would hide what this very
  reply advertised. It stays ahead of the fetch plan, so `in_flight` counts the same thing in the send-time
  row and in the reschedule row. A rule and a policy keep their earlier positions, which is why the source
  decides both gates before anything moves. **One case escapes that order**: a reply that fails to decode is
  resolved where a rule's is, before any fold, with no emptiness to report, because there is nothing to fold
  and nothing to read. The rows of such a resolution carry the tracker state as it stood before a reply this
  node could not read.
- **`runs.csv` has no column for the knob.** The stage name and each run's `meta.json` are what identify these
  runs; `meta.json` carries `state_poll_choice_redecide: true`, written only when the knob was on, so a run
  made before the key existed and a run made with the knob off are the same bytes.

### Acceptance 1 and 2 of the ticket were met before this step

They are checks on the source, not on a run, and the merged change carries them. The suite holds 1046 tests
at `a8add2c`, re-counted here with `pytest --collect-only`; the 1005 it held before the change is the merged
change's own record and was not re-counted.

| what the ticket asks | the test that answers it |
|---|---|
| 1. with the knob off, not one `state_poll_reschedule` row | `tests/test_engine.py::test_choices_without_redecide_never_reschedule` |
| 1. `evalgen`'s generated bytes unchanged with the knob off | `tests/test_evalgen.py::test_the_redecide_knob_off_generates_the_bytes_it_did_before_the_key` |
| 2. a re-decision on an answered poll and on a timed-out one | `test_redecide_draws_a_new_choice_on_an_answered_poll`, `test_redecide_draws_a_new_choice_on_a_timed_out_poll_and_clamps` |
| 2. `action_prob` 1/len and an interval out of the set | `test_every_redecided_interval_is_one_of_the_choices_at_one_over_len` |
| 2. the row carries the send-time row's state fields | `test_redecide_row_carries_exactly_the_send_time_row_s_state_fields`, `test_redecide_row_state_shows_what_the_reply_advertised_but_not_its_fetches` |
| 2. the jitter draw is the send-time one, drawn once per resolution | `test_redecide_draws_once_per_resolution_with_logging_off_as_well` |
| 2. choices with a policy or a rule is still refused | `tests/test_agent.py::test_validate_config_rejects_the_redecide_knob_without_choices_or_as_a_non_boolean` and the existing exclusion tests |

## Runs (`main11.sh`, to be run from a detached clean checkout of main `a8add2c` at `/home/lab_shared/wt/semindex-main11`)

Common settings: `--k 4 --store-synchronous NORMAL --workload w0 --disconnect-duration-s 8 12
--interval-choices 3.695 6.0 10.043 --choice-jitter 0.1 --choice-redecide`. The outage schedule depends on
(p, seed) only. Stage prefix `s11`; `s10` belongs to evidence/20.

| order | stage | p | seeds | runs | time |
|---|---|---|---|---|---|
| 0 | `s11smoke` | 0.2 and 0.4 | 101 | 2 | 5 min |
| 1 | `s11a` | 0.2 | 1 to 100 | 100 | 4 h 27 min |
| 2 | `s11a` | 0.4 | 1 to 100 | 100 | 4 h 27 min |

200 runs at 160 s each (evidence/19's measured 159.5 s per run), about 8 h 53 min, plus the two smoke runs.

- **Every flag of that command line is accepted by this worktree's `evalrun`**, checked by calling
  `evalrun.parse_args` on the exact argument list and reading the namespace back: `--choice-redecide` arrives
  as `choice_redecide=True`, `--interval-choices 3.695 6.0 10.043` as the three floats, and
  `--disconnect-duration-s 8 12` as `[8.0, 12.0]`. Parse only; no run was started by that check.
- **p outer, p 0.2 first**, unlike evidence/20's seed-outer order: the two cells are two stages of the same
  settings rather than five makers on one seed, so there is no (seed, stage) pairing to keep whole, and p 0.2
  is the main cell. A sweep stopped part way then holds all of p 0.2 rather than half of each cell.
  **This departs from one clause of the ticket.** The ticket's 回すもの section says in one sentence both
  "seed outer" and "run p 0.2 to the end first", and the two cannot both hold for one stage at two p; the
  second clause carries the ticket's stated reason (the main cell is not left short when a sweep stops), so
  this file follows it and drops the first.
  `main11.sh` makes one `evalrun` call per (p, seed), so a sweep that is interrupted redoes one run only.
- **The source change was merged before the run** (upstream PR #39, 2026-11-22), so the ticket's path of
  running from a commit pushed to a branch and checked out detached was not needed. The runs are to be made
  from main at `a8add2c`. **That checkout does not exist while this file is written**, so this line is the
  plan and not a record: what ties the runs to a commit is `main11.sh`'s header line, which prints the
  checkout's `rev-parse HEAD` and its dirty count at the head of every real run, and `RUN.txt` afterwards,
  which carries the commit that was actually loaded. A header line naming another commit, or a nonzero dirty
  count, says the runs are not the runs this file describes.
- **No gate stage.** A gate stage catches a code version that moved between two stages that are compared with
  each other. `src` did move between `d009cfb`, the commit evidence/20's runs came from, and `a8add2c`: the
  only commit touching `src` or `tests` between them is `1c3f758`, the re-decision change itself. **What that
  change's tests establish is narrower than an untouched knob-off path.** They establish that the change
  rewrote no existing test and removed none, choices-mode tests included (598 test lines added across five
  files, none deleted), that with the knob off the engine emits no `state_poll_reschedule` row and leaves the
  next poll at the send-time draw
  (`test_choices_without_redecide_never_reschedule`), and that `evalgen`'s generated bytes are unchanged
  (`test_the_redecide_knob_off_generates_the_bytes_it_did_before_the_key`). **No test compares an event stream
  produced at `d009cfb` with one produced at `a8add2c`, and `engine.py` was rewritten rather than extended
  across them: 143 lines added and 51 removed in that file alone.** **What is therefore not shown: nothing in
  this step measures an s10 stage and an s11 stage against each other on the testbed, so the claim that the two
  versions behave identically with the knob off rests on those tests and not on a run.** No pass line below
  compares an s11 stage with an s10 stage.
- Before the sweep, by hand: no emulator and no Cefore daemon running. The checkout is not edited while runs
  load it.

## The smoke gate before the sweep (`smoke.py`)

Stage `s11smoke`, seed 101, one run at p 0.2 and one at p 0.4, the sweep's settings exactly. Seed 101 lies
outside the sweep's seeds 1 to 100 and outside evidence/20's held-out seeds 121 to 160, so neither set is
spent on it. The sweep starts only when every line of `smoke.py` reads PASS:

- `evalvalid`'s verdict in `meta.json` carries no flag;
- `meta.json` records the knob and the three arms, so the run being read is a run the knob was on in;
- every node wrote `state_poll_reschedule` rows, so the knob reached every agent;
- every such row records `action_prob` 1/3 and an interval out of the three arms;
- every such row carries the field names that node's `state_poll` rows carry;
- no (node, neighbour) pair re-decided more polls than it sent;
- **the next poll leaves when the re-decision said it would**: for every poll re-decided without a clamp, the
  gap from its send to the next send to the same neighbour lies between the reschedule row's
  `applied_interval_s` and that plus 0.5 s. This is the one check on the testbed that the re-decision is
  applied and not only logged. A source that wrote correct rows and then left the schedule at the send-time
  draw would pass every other check here, and it would also pass lines 4 and 5 below, because the ignored draw
  and the applied one come out of the same uniform stream and give the same delay, poll count and share per
  arm. The merged tests do assert the applied instant (`test_redecide_draws_a_new_choice_on_an_answered_poll`
  reads `_next_state_poll` and the name of the next poll), so this is a second check, on a real run. The
  0.5 s comes from two runs of evidence/20's rule stages, which re-decide through the same code, read for
  this file and stored in no other (`s10f` seed 121 and `s10t37r` seed 140): over their 360 re-decided polls
  the next poll left 0.0001 s to 0.1002 s after the re-decided instant. The nearest two arms' applied
  intervals are 1.3 s apart, so a schedule that followed the send-time draw cannot hide inside 0.5 s. **The same check is run over all 200 runs after the sweep and reported
  under pass line 4**, with the number of polls outside the 0.5 s.

`smoke.py` reads `meta.json` and `*/semindex/events.jsonl` and opens no other file under `node_dirs`, which
also hold Cefore key material. **It prints no delay, no poll count, no row count and no gap.** A number out of a pilot
run, read before the prediction is read, is a number that can steer the reading of the prediction; what the
smoke gate prints is a shape and never an outcome.

## The replay's copy of the branch (`gate.py`)

The ticket says that if pass line 5 misses, the replay's copy of the re-decision is the first suspect, because
evidence/13's replay returned early in the choices branch and never re-decided. That branch is now in
`../13/replay.py` and `gate.py` checks it, before the sweep:

- **the knob off changes nothing**: the edited replay and the replay as it stood before the edit produce
  equal event dicts on a fixed choices arm, a policy stage and a rule, in Mode A and in Mode B. The key the
  edited `load_run` writes is dropped from the run dict before each of those comparisons, because the frozen
  scripts (`../20/predict.py` among them) build their run dicts by hand and what holds them on the old path is
  the default of the engine's `run.get("redecide", ...)`. Left in, a wrong default there would move those
  scripts and no line of this gate would say so;
- **the knob on over a single choice is the fixed arm**: with `choices = [X]` for X = 3.695, 6.0 and 10.043,
  the re-decision redraws the only value there is and recomputes the due instant the send already set, so the
  two event streams agree event for event apart from the reschedule rows the knob adds. This held exactly, in
  both modes and on every seed tried, with no field excluded;
- **the three choices behave as the source does**, in Mode A and in Mode B: one reschedule row per resolved
  poll and never two for one poll, every interval out of the choice set, the applied interval built from the
  poll's own send-time jitter draw, and exactly one choice-stream draw per send plus one per reschedule row.
  The resolutions that got past the newest-poll guard are also counted inside the engine and held equal to the
  reschedule rows, because a resolution that returned without writing a row would leave nothing in the log for
  any of the row-side checks to miss.

**Six places where the replay is knowingly poorer than the source, and what each costs.**

1. **The replay reports `consecutive_poll_timeouts` as 0 in choices mode, in both rows of a re-decided poll.**
   The source counts timed-out polls out of its poll history and reports the same count with the knob off and
   on; the replay maintains that counter only in the paths that read it, and the choices path has never read
   it. **The gap is older than this change and the knob neither opens nor widens it**: with the knob off the
   choices path never reaches the resolution at all, so the counter was already standing at 0 in every
   `state_poll` row of every choices stage the replay has ever simulated. What the knob adds is a second row
   that reports the same 0. Leaving the counter alone is what makes the single-choice identity above exact. No
   metric, no pass line and no scheduling decision in the replay reads the field, and the testbed's own rows
   carry the real count.
2. **The replay emits no `state_poll_late_reply`.** It cannot: delivery is decided at send time, so a reply
   either lands one RTT after its poll or never, and a reply after a timeout is a case the replay cannot
   produce. The testbed can produce it, so a count of those rows in the runs has nothing in the prediction to
   be held against.
3. **The replay's reschedule row carries no `empty` and no tracker state.** The source puts what the reply
   held on the row in this mode, beside the whole `tracker.decision_state(neighbor)` block; the replay keeps
   the row shape it has written since the policy stages, `last_poll_status` and `consecutive_poll_timeouts`
   and no `empty`. **A replayed row would therefore fail `smoke.py`'s state-field check**, which is the
   property the testbed's rows are gated on and the replay's rows do not have. What the replay logs for a
   learner to read belongs to a later ticket, and nothing in this step's pass lines reads either dropped part
   off a replayed row. The testbed's rows carry both.
4. **The replay resolves a poll that is still outstanding when the agent stops.** It sends no poll at or after
   t0+130 s, but it does write a reschedule row there, where the testbed's agent has already left its loop:
   the source tests `should_stop` before `transport.poll()` and before `on_tick`, so past that instant no
   reply is dispatched and no timed-out poll is resolved. **The surplus rows are timeouts**: this replay
   answers every reply a fixed 1 ms after its poll, so an answered poll lands past the stop only when it was
   sent inside the last millisecond before it, while a timed-out poll reaches there from anywhere in the last
   2 s. Counted with this replay on the closest cell the store holds, `s6b3p695` p 0.4 seeds 1 to 5 with this
   stage's three arms, both modes: 11 of 1580 reschedule rows fall past the stop, all 11 `timed_out`, against
   0 of 1580 `state_poll` rows. That count is a one-off reading and no line of `gate.py` holds it.
   **So the surplus biases three predicted numbers upward and not one**: the reschedule count, the timeout
   share, and the N the share-per-arm tolerance is evaluated at. `predict.txt` prints the share of reschedule
   rows that fall in that region, 0.4 % at p 0.2 and 0.6 % at p 0.4, and the measured reschedule counts should
   sit that much below the predicted ones.
5. **On a clamped timeout the replay sends the next poll one tick lag late.** The source resolves the
   timed-out newest poll at the head of `on_tick` and then tests the re-decided due instant in that same tick,
   so a due instant clamped to `now` is polled immediately; the replay arms the next poll at
   `due + tick_lag`, which is 50 ms (Mode A) or U[0, 100] ms (Mode B) later. The replay inherits this from the
   rule path, where it was written. **It cannot fire at this stage's choices**: the shortest applied interval
   is 3.3255 s measured from the send and a resolution lands at most 2.1 s after the send, so the due instant
   is always ahead of `now` and the clamp branch is never entered, which is also why the clamped share is
   predicted as zero. It is recorded here so that a later ticket drawing an arm shorter than the lifetime does
   not inherit it unseen.

6. **The replay cannot produce a reply that fails to decode.** It builds every reply out of the holder's own
   state, so every delivered reply is folded and planned on. The source resolves an undecodable reply as
   answered with no emptiness to report, folds nothing and sends no fetch plan for it. **If the testbed
   produces such replies, the measured delay sits above the predicted one** and the poll count does not move,
   since both sides resolve the poll and draw once. evidence/20 counted the case on its two rule stages and
   found none in 11881 answered re-decisions. It is counted here after the run (answered reschedule rows whose
   `empty` is null) and reported under pass line 4.

**One further asymmetry, not constructible on either side.** The source tests `state_poll_choice_redecide`
ahead of both the table and the rule, while the replay drops the knob whenever the node has no choices to draw
from, so a run carrying a rule *and* the knob would re-decide by the rule in the replay and by the choices in
the source. Neither side can be asked to run that config: `validate_config` refuses choices beside a rule or a
policy, and the source would index a `None` choice list if it got through. The replay's order is the safer of
the two, and nothing is changed for it.

## Prediction (`predict.py`, output `predict.txt`, both placed here before the runs)

evidence/18's engine (`../18/bound.py`'s `sim`, which subclasses evidence/15's and evidence/13's), 40
replicate streams of 100 single runs per p. One stream is one experiment, the condition the testbed has: one
run per seed, 100 seeds. Every number is the median over the 40 streams.

**Where the exogenous inputs come from.** evidence/13's run dict carries four: the outage timeline, the record
creation stamps, t0 and the per-node `agent_start`. Unlike evidence/20's held-out seeds, many of these seeds
already have a run with this outage shape, and those runs give all four exactly.

| p | from logged runs | generated |
|---|---|---|
| 0.2 | seeds 1 to 20, from `s6b10p043` | seeds 21 to 100 |
| 0.4 | seeds 1 to 40, from `s8b6p0` | seeds 41 to 100 |

**The 40 replicate streams are drawn with the label `s6b10p043` at both p**, which is a string that names a
stream and not a stage that is read: evidence/14 onward derive every tick-lag stream from `trickle:<label>`,
and keeping that label is what makes these streams the streams those steps drew. At p 0.4 the label is
therefore not the donor stage, which is `s8b6p0`. It is recorded here because the label alone decides which
random numbers the frozen predictions below came out of.

A generated seed takes its outage timeline and its creation stamps from this worktree's own
`evalgen.build_experiment`, which depends on (p, seed, k, workload) alone, and borrows t0 and `agent_start`
from the logged run of donor seed `1 + (seed - 1) mod <number of logged seeds>`. `build_experiment` is used in
place of `generate_run` because `generate_run` writes a ReCefore yaml and a per-node config beside each
description and those configs carry key material. The two halves are not separated in the numbers: the sweep
runs all 100 seeds of a p as one stage, and so does the prediction.

**The gate, before any predicted number is printed.** `build_experiment`'s outage events and record schedule
against the stored `*.experiment.json` of every seed that has a run with this shape, 60 descriptions in all.
Every event and every schedule equal. This is evidence/20's gate 2, widened to both cells. **It does not state
that the instants have the shape `w0` and the outage shape are defined to have**; `tests/test_evalgen.py`
answers that against hand-written expectations. A mismatch prints the difference and no prediction.

**The predicted values.**

| p | headline delay s (band) | polls per run (band) | censored share |
|---|---|---|---|
| 0.2 | 24.225 (22.771 .. 25.678) | 145.51 (136.78 .. 154.24) | 0.000 |
| 0.4 | 34.930 (32.835 .. 37.026) | 145.70 (136.96 .. 154.45) | 0.021 |

The four values in the first two columns are what pass line 5 reads.

| p | send rows per run, median (min..max) | reschedule rows per run, median (min..max) | timeout share | clamped share | share per arm 3.695 / 6.0 / 10.043 |
|---|---|---|---|---|---|
| 0.2 | 160.0 (148 .. 176) | 160.0 (148 .. 176) | 0.191 | 0.000 | 0.329 / 0.337 / 0.333 |
| 0.4 | 161.0 (146 .. 174) | 161.0 (146 .. 174) | 0.368 | 0.000 | 0.331 / 0.336 / 0.333 |

- The ticket guessed 174 to 280 decisions per run off three earlier stages and called that a guess to be
  re-measured. The send rows above are this replay's answer for the uniform draw.
- **The clamped share is predicted as zero, and the reason is structural in the replay.** The shortest applied
  interval is 3.695 × 0.9 = 3.3255 s, and a resolution lands at most lifetime + one tick at 2.1 s after the
  send, so `sent_at + applied` is still in the future at every resolution. On the testbed a clamp needs the
  resolution to land later than the poll's own provisional due instant, which needs the agent's loop to stall
  for over a second. **A nonzero clamped share in the runs is therefore evidence of a stall, not of the arms**,
  and is reported as such. The ticket's note that a 3.695 s arm drawn after a 2 s timeout need not be clamped
  is the same reading.
- **Every poll of the replay is resolved, so the two counts are equal.** With the shortest applied interval
  above the 2 s lifetime, a poll is always resolved before its successor is sent, and the replay re-decides
  exactly once per poll. **The testbed's reschedule count should sit slightly below its send count**, because
  the agent stops at t0+130 s with its last polls still outstanding: `predict.txt` puts the rows the replay
  writes past that instant at 0.4 % of the reschedule rows at p 0.2 and 0.6 % at p 0.4, and that is the size
  of the gap to expect.
- **The predicted row count per p is about 16000** (16078 at p 0.2, 16099 at p 0.4, medians over the 40
  streams), which puts the pass line's tolerance at ±0.0112 and ±0.0111 around 1/3. The predicted shares sit
  inside it in both cells. **The tolerance the pass line uses is the one read at the measured N, not these.**

## Quantities

Fixed here, before the runs. Where a definition comes from is named.

- **Headline delay** (`../20/spec.md` "Quantities", from evidence/14's scorer): pooled `sum(lat4) / sum(n4)`
  over the runs of a stage, over the records whose creation is at or before t0+72 s, the time until a record
  is stored 4 hops away. Under `w0` every seed contributes the same 240 records and every one of them is
  created at or before t0+71 s, so the headline set is the whole set and this is also evidence/17's pooled
  delay.
- **Polls per run** counted over the whole window `[t0+10 s, t0+130 s]` (`../20/spec.md`, from evidence/17),
  and **the mean over the 100 runs of a cell**, which is what `predict.txt` prints per stream and what
  evidence/20 measured. The measured count is `evalagg`'s in-window poll count summed over the 8 directed
  pairs, the count evidence/20's analysis read. A `state_poll` row is a poll; **a `state_poll_reschedule` row
  is not and is never counted as one.**
- **Censored share** `c4/(n4+c4)` (`../20/spec.md`), reported beside every delay.
- **Decisions per run**: `state_poll` rows and `state_poll_reschedule` rows of one run's five node logs,
  counted separately and over the whole run rather than over the window. The window belongs to the delay and
  the polls per run; what pass line 4 asks is what a run decided.
- **Timeout share**: reschedule rows whose `trigger` is `timed_out`, over all reschedule rows, per p.
- **Clamped share**: reschedule rows whose `clamped` is true, over all reschedule rows, per p.
- **Share per arm**: reschedule rows whose `interval_s` is that arm, over all reschedule rows, per p.
- **The tolerance on the share per arm**: `1/3 ± 3*sqrt((2/9)/N)`, N the number of reschedule rows counted
  after the run, p 0.2 and p 0.4 counted separately. The formula is the ticket's and is fixed before the run;
  it is three standard deviations of a binomial share at 1/3, so about 1.1 % at the row counts expected. The
  value `predict.txt` prints beside its own N is for information: the pass line reads the formula at the
  measured N.
- **The band on pass line 5**: the predicted value times 1 ∓ 6 %, as `../20/spec.md` fixes it. It comes from
  the largest per-stage deviation evidence/13 measured on the default load, 5.9 %, on four fixed arms and two
  policies at p 0.2 and p 0.4. **Nothing says it is the right band for a uniformly drawn interval or for the
  re-decision branch**, so every value outside it is reported with how far outside and in which direction,
  whatever the count is.

## Pass lines

The ticket's Acceptance 1 and 2 are checks on the source and were met before this step (the table above).
Acceptance 3, 4 and 5 are the three lines below, copied without moving a threshold.

### 3. All 200 runs valid

No `evalvalid` flag on any of them. **If a flag appears, how many runs, in which cell and which flag is
reported with the breakdown.** The two smoke runs are reported separately and are not part of the 200.

### 4. The decision log is material a learner can read

Counted over the 200 runs and reported, p 0.2 and p 0.4 separately:

- **decisions per run, median and minimum and maximum**, with the reschedule rows and the send-time rows
  counted separately;
- **the share of the reschedule rows that fell to each of the three arms**, which is the check that the
  uniform draw is uniform. **The tolerance is fixed by the formula above before the run: every arm's share
  inside `1/3 ± 3*sqrt((2/9)/N)`, N the reschedule rows counted after the run, per p.** The formula does not
  move once a share has been seen. A wide tolerance such as 30 to 37 % would let a skewed draw pass
  undetected, which is the reason the ticket fixes a formula and not an interval;
- **the share of resolutions that were timeouts**, per p, as Quantities defines the timeout share: reschedule
  rows whose `trigger` is `timed_out` over all reschedule rows of the cell, pooled over its 100 runs;
- **the number of rows whose state was truncated** (`truncated` true, i.e. a run tracking more origins than a
  decision row can carry), counted on the send-time rows and on the reschedule rows separately. Expected zero
  with two origins; a nonzero count means the learner's rows are missing origins.

Three more counts are reported with these, **all additions of this file with no Acceptance item behind them
and no pass or fail attached**: the polls whose successor left outside the 0.5 s of the smoke gate's schedule
check, over all 200 runs; the answered reschedule rows whose `empty` is null (item 6 of the list above); and
the **clamped share**. On the clamped share: the ticket's 何を足すか section asks how often the flag appears to
be counted off the stage's own log, and the prediction above says what the replay expects.

### 5. Did the replay predict the measurement?

The **headline delay** and the **polls per run over the whole window** at p 0.2 and p 0.4, four values, each
inside plus or minus 6 % of the prediction tabled above. **All 4 inside: "predicted". 1 or 2 outside:
"mixed". 3 or more outside: "missed".**

- The split is the ticket's, which scaled evidence/19's 20-value split down to 4. **Neither split is backed by
  a measurement**: evidence/19's own bounds were set by judgement, so this one is a carried-over convention
  and nothing more.
- **A value outside the band is reported with its size and its direction**, naming the cell and the quantity,
  whatever the count is.
- **On "missed", the replay's copy of the re-decision branch is the first suspect.** It is the one piece of
  this step that has never been held against a measurement: `gate.py` shows it costs nothing over a single
  choice and draws as the source does, which is a statement about the replay and not about the testbed. The
  comparison to run then is evidence/16's: the decision sequences around timeouts in a testbed log against
  the replay's, and where the two part is reported before any other conclusion is written.
- The delay here is **not** a number any policy is compared on, so "missed" costs this step's own purpose
  nothing. What it costs is the later screening: tickets 09 to 12 pick candidates by replay scoring, and a
  replay that cannot reproduce a uniformly drawn stage is a replay whose screening is in question.

## What this step cannot show

- **Nothing about which decision maker is better.** This stage draws uniformly and runs no policy. Ticket 11
  builds a policy out of these rows and ticket 12 compares the three makers.
- **Nothing about whether a policy can be estimated off these rows.** evidence/05 and evidence/07 left the
  short arms biased in exactly that kind of off-policy reading, and ADR 0004 turned the path down once. This
  step stops at leaving the data.
- **Nothing about the cost of one decision in the live loop.** That is ticket 12's.
- **No testbed comparison across the two source versions.** evidence/20's runs came from `d009cfb` and these
  from `a8add2c`; the knob-off path is held to be unchanged by the merged tests only, and no stage in this
  step is run from both versions to show it on the testbed.
- **The re-decision branch of the replay has no measurement behind it.** Every predicted number rests on a
  branch read off the source and checked against the source's own single-choice behaviour. Until pass line 5
  is read, the replay cannot be said to reproduce this stage at all.
- **Six things the replay does differently from the source** (the `gate.py` section lists them): a
  `consecutive_poll_timeouts` of 0 in choices mode, no `state_poll_late_reply` at all, a reschedule row
  without `empty` and without the tracker state, reschedule rows past the agent's stop instant, a next
  poll one tick lag late after a clamped timeout, and no reply that fails to decode. **The `state_poll_late_reply` count and the two dropped row
  parts have nothing in the prediction to be held against**, so pass line 4 reports them and pass line 5 does
  not read them; the stop-instant surplus has a predicted share and biases the predicted reschedule count and
  timeout share upward, and the clamp cannot be reached at this stage's choices at all.
- **Nothing about p 0.5, p 0.6, p 0, the no-outage cell, the on/off loads, or finer arms.**
- **Every number in the prediction is simulated.** A predicted band says what the replay expects the testbed
  to measure, and is itself no measurement.

## Deliverables (this directory)

`spec.md`, `gate.py`, `smoke.py`, `main11.sh`, `predict.py`, `predict.txt`, all before the runs. After the
runs: the analysis script, its output, `README.md`, `RUN.txt`.
