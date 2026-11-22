# evidence/22 spec: a port for a decision maker that reads the state, and the weight on one poll

Written 2026-11-23, with the code of this directory and before anything was labelled. Ticket
`../../tickets/09-replay-labels.md`. This half covers that ticket's Acceptance 1, 2, 3 and 4 only.
Acceptance 5, 6 and 7 are not specified here and are marked as such below. Where this file says
something the ticket does not, or departs from it, it is marked as such.

## Question, and what this half is for

A label on one decision is "which of the three arms, run to the end of the run, scores best". Three
things have to exist before the first label can be made, and this half is those three plus the number
the score needs.

**(a) A place to put a decision maker.** Both decision instants of the replay draw their interval
themselves: the choices branch of `decide_interval` at the send, and the re-decision branch of
`resolve` at the resolution. A decision maker that reads the state has nowhere to sit, and a learner
and a hand-written formula have no single input to share. The port is that place, and the state it
hands over is the state a real node logs beside the same decision.

**(b) Randomness that is less coupled between candidates.** The base engine draws every Mode B tick
lag from one shared stream in event order, so changing one decision moves every later lag of every
pair, and the three runs of one label differ everywhere rather than at one decision. Per-pair streams
make each pair's lag VALUES a function of the pair and the seed alone: pair X's n-th lag draw is the
same number whatever another pair decides. **They do not make the other pairs hold still in the
stronger sense of serving the same events**, and that limit is measured, not assumed: the per-node
jitter stream still shifts another pair's send instants, and on the one run `lagcheck.py` measures,
one other pair draws 26 lags in one run and 27 in the other. See "The per-pair lag streams" below for
the numbers and "What this step cannot show" for what is therefore still open.

**(c) One weight.** The score is `delay + weight x polls`, and the weight has to be a single number
fixed before any label exists, or a label is a statement about whatever weight produced it.

Not answered here: whether any decision maker is better than the fixed arms, which needs the labels;
whether a label is the testbed's answer, which it is not and cannot be (see "What this step cannot
show"); how the observable state should be rendered as text, which is Acceptance 7 and waits for a
decision by the user.

## Corrections to the ticket

- **`last_timeout_s` is the instant of the most recent FETCH timeout on that neighbour**, not a poll
  timeout (`src/semindex/tracker.py:251-262`, written in `_fail` at `tracker.py:454`). The ticket's
  line 67, which says the value "is known at the point of `resolve`", is wrong: `resolve` is a poll
  resolution and never sees a fetch timeout. The replay writes it in `ev_ftimeout`
  (`../13/replay.py:1117`), after the guard that drops a stale timeout.
- **The ticket's line numbers for `../13/replay.py` are stale.** At `fd6ce55`, before this step's
  edit: `Node` was 268-299 (the ticket says 263-287), `decide_interval` 675 (653), `resolve` 759
  (727), the choice draw 691-694 (668-671), Mode A's lags `logged_lags` 210-242 plus `tick_lag`
  534-541 (205-241), the Mode B shift comment 636-641 and the draw 542 (627-636), and evidence/18's
  per-pair lag stream `bound.py:173-180` (160-168).
- **The exchange history the ticket asks for is not a logged field.** The source keeps no bounded
  per-neighbour exchange history: its `_PollReplies.polls` is bounded in TIME by the Interest
  lifetime plus a grace second (`src/semindex/engine.py:160, :259-274`) and only two scalars of it
  ever reach a row. The history this port hands over is therefore built from the SEQUENCE of that
  pair's own `state_poll_reschedule` rows, which is a new construct; see "The history".
- **The class is `Tracker`** (`src/semindex/tracker.py:89`), not `SyncTracker`.

## The port's contract

### How it is reached

- A run dict may carry a new key **`decider`**, default `None` (`../13/replay.py:580`). It is valid
  only together with `choices` and `redecide=True`, and it is exclusive with `rule` and `policy`.
  A bad combination raises `ValueError` at `Node` construction (`replay.py:303-314`), not at the
  first decision: a decider beside a rule would otherwise run the rule for a whole sweep and the
  numbers would be filed as the decider's. Unlike `policy` and `choices`, a `rule=` argument does
  **not** silence it; it makes the construction raise.
- It rides the existing re-decision path, so the six predicates keyed on `nd.redecide` are untouched
  and no new flag exists. The two places that drew `nd.choices[int(nd.choice_rng() * len(nd.choices))]`
  call the decider instead (`replay.py:881` at the send, `replay.py:1012` at the resolution) and do
  **not** draw `choice_rng`. The jitter draw stays exactly where it was, in the same expression, or
  every later jitter value of that node would shift and the run would no longer be the same run with
  one decision moved.
- **A returned interval that is not one of `choices` raises `ValueError`** (`replay.py:846-849`).
  Every reader of these runs indexes the choice set with the interval.

### What the decider is handed

`decider(obs) -> interval_s`, with `obs` a dict of exactly three keys (`replay.py:807-850`).

`obs["state"]` is the six fields of a testbed decision row, built by `Node.decision_state`
(`replay.py:341-391`) and nothing else. Field by field:

| field | the source it mirrors | the replay's own source | the trap |
|---|---|---|---|
| `in_flight` | `tracker.py:304`, accessor `:242-249` | `Node.in_flight[nb]` | none; the same counter |
| `last_timeout_s` | `tracker.py:305`, `:251-262`, written `:454` | `Node.last_timeout_ms[nb]/1000` | a FETCH timeout, not a poll one; absolute seconds, not an age; a stale timeout must not set it |
| `truncated` | `tracker.py:306`, `:289-290` | `len(self.origins) > 8` | none |
| `origins[o].hi` | `tracker.py:299` | `_Origin.hi` | none |
| `origins[o].missing` | `tracker.py:300`, `missing_total` `:205-236` | computed inline | the UNCAPPED backlog, not `Node.missing`'s fetch-planning list capped at `MAX_MISSING_PER_ORIGIN` |
| `origins[o].advertised[nb].hi` | `tracker.py:296-297` | `_Origin.advertised[nb][0]` | none |
| `origins[o].advertised[nb].notified_max` | `tracker.py:297` | computed inline | PER NEIGHBOUR; `Node.notified_max` maxes over every neighbour and is the wrong function here |
| `last_poll_status` | `engine.py:529-533` at a send, `:680-684` at a resolution; `_status` `:300-307` | `Node.last_status[nb]` | nothing stored it before; see the deviation below |
| `consecutive_poll_timeouts` | the same two lines; `snapshot` `:284-298` | `Node.real_consec[nb]` | `Node.consec` has the same name and is constant 0 under the re-decision path, on purpose |

Origins are capped at the eight lowest names, as the source caps them, and `truncated` says whether
any were left out. Every neighbour that has ever advertised an origin appears under `advertised`, not
only the neighbour the decision is about.

**Nothing a real node cannot see may enter it.** The state BUILDER reads neither a neighbour's
`_Origin.hist` (`replay.py:411`) nor any other node's object at all. `statecheck.py` check 5 enforces
that, by poisoning every history and by reading the builder's source.

**What that check does not say, stated because its pass line could be read as more.** The builder not
reading `hist` is not the same as nothing hist-derived being in the state, and one field is:
`origins[o].advertised[nb].hi` is computed from THAT NEIGHBOUR's `_Origin.hist`, by
`progress_map_asof` in `ev_rx` (`replay.py:897`, `:909`), and copied into the deciding node's own dict
at reply time. Poisoning the histories at the decision instant therefore cannot detect it and the
rebuild is equal by construction. **The argument for that field is not the poison check but the
source**: a real state reply carries (hi, extra) per origin and the source hands exactly that pair to
its tracker (`src/semindex/engine.py:842`), so a live node sees it. What the poison check does buy is
that a FUTURE edit cannot reach into a neighbour's store at decision time. The companion path, a
sparse extra above the advertised `hi`, is never exercised: in the `s11a` p 0.2 seed 1 run no reply
carries one, so `notified_max > hi` holds in 0 of 1519 advertised cells.

### The history

`obs["history"]` is the earlier RESOLVED polls of this node towards this neighbour, oldest first, at
most `DECIDER_HISTORY = 8` (`replay.py:93`), each one a dict of the fields a real node writes on its
own `state_poll_reschedule` row plus that row's stamp: `ts_ms`, `poll_epoch_ms`, `trigger`, `empty`,
`interval_s`, `applied_interval_s`, `clamped`. The decision being made is not in it. It is appended
in `resolve` after the row is emitted and **before** the identity early-return (`replay.py:1063`),
because a poll whose provisional timer already pointed at the new instant takes that return and a
history that skipped exactly those polls would hide the cheapest decisions of the run.

**`neighbor` is the one action field the element drops**, because `where` already carries it and a
learner may not read `where`. This departs from nothing in the ticket, which does not list the
element's fields; it is stated here because `statecheck.py` tests that exact set.

### `where`, and who may read it

`obs["where"]` is `{node, neighbor, kind, ts_ms, ordinal}`, where `kind` is `state_poll` or
`state_poll_reschedule` and `ordinal` counts this run's decider calls of that kind from 0 in call
order. **It exists so that a tool can name one decision of a run again on a later pass, and a learner
must never read it.** A policy that keyed on the node name, the neighbour or the ordinal would score
on something the live loop cannot hand it, and the score would not transfer to the testbed at all.
The docstring of `ask_decider` says so.

### What does not move

**With `decider` None every existing mode emits event dicts equal to `fd6ce55`'s, draw for draw**;
that is `gate.py` check 1, over six configurations in both modes. The new bookkeeping
(`real_consec`, `last_status`, `last_timeout_ms`, `poll_hist`, `replay.py:331-339`) lives in new
attributes, is maintained in every mode and is read only when a decider is set; `nd.consec` and every
emitted row of an old mode are untouched.

**In decider mode the rows widen.** The `state_poll` and `state_poll_reschedule` rows carry the full
`state` -- the very dict the decider was handed, built once -- and the reschedule action carries
`empty`, as the source's own rows do under a rule or the re-decision knob (`engine.py:700-707`). So a
tool can read these rows like testbed rows. The old modes keep the one-field and two-field states
they had: widening those would rewrite rows already measured against the testbed.

### One thing the second half will hit, and it needs a decision

The ticket gained a 決まったこと section dated 2026-11-23 (its lines 107-121, written by another
session while this half was being built) saying that while one decision is probed, **every later
decision of every node is decided by the formula rule F** -- `../20/predict.py`'s `F_MEM`, i.e.
I_min 3.695, m 1.5, I_max 10.043, back to 3.695 on a timeout. F's reachable intervals are 3.695,
5.5425, 8.31375 and 10.043, and **two of those four are not in the three-arm choice set**, so a
decider that emulates F raises on its second consecutive empty reply under the guard above.

The port as specified therefore cannot carry that labelling construction unchanged. Two ways out, and
the cheaper one is the first:

1. **The labelling run passes `choices` as the union of the three arms and F's reachable values.**
   `choices` is the validation set and the set the send-time draw would use; a labelling run never
   takes the send-time draw (it decides every decision) and never reports `action_prob`, so the only
   cost is that a reader counting rows per arm on such a run sees six intervals instead of three.
   Nothing already measured is touched, because no existing stage gets a wider set.
2. **Run the suffix on the rule path instead of the decider.** D1 makes a decider and a rule
   exclusive, so this needs a mode switch inside one run, which the port does not have and which
   would reopen D1.

Neither is decided here. It is flagged because the two documents were written the same day and the
reconciliation is not in either of them.

### Three deviations from the source, stated

1. **`last_poll_status` of a poll that never resolved.** The replay writes "pending" at the send and
   the real status at the resolution. A poll superseded before it resolved therefore reads "pending"
   where the source, judging status off the clock, would read "timed_out", and such a poll is also
   missing from `real_consec`, which the source would fold in as an expired timeout. How many such
   polls there are is exactly what the provisional-timer count below measures; in `weight.py`'s runs
   it is 0. **So "pending" is written and never read** on this half's runs: the only
   `last_poll_status` a decider here ever saw was `None` (the first send of a pair) or a real
   resolution status. `statecheck.py`'s printed type set for the field, `NoneType|str`, is the same as
   the logged rows'.

   **The bound under which the deviation is unreachable, stated because the second half has to check
   its own choice set against it.** A poll can only be superseded before it resolves if the next send
   comes first, so the deviation is unreachable while the SMALLEST APPLIED interval,
   `min(choices) * (1 - choice_jitter)`, exceeds `POLL_TIMEOUT_MS + TICK_PERIOD_MS` -- 2 s, the
   Interest lifetime, plus one 100 ms tick, because the timeout is DETECTED at
   `sent + POLL_TIMEOUT_MS + tick_lag("timeout")` and not at `sent + POLL_TIMEOUT_MS`. For this
   half's choice set that is 3.695 x 0.9 = 3.3255 s against 2.1 s, a margin of 1.2255 s. The F-union
   set flagged above adds 5.5425 and 8.31375, both larger, so it is inside the bound too -- but it is
   the second half's job to check its own set against this sentence, not an assumption it may
   inherit. A choice set that breaks the bound would need the divergence modelled (resolve the
   stale pending poll in `ev_poll` before building the state) rather than documented. The comments at
   `replay.py:721-728` and `:975-985` carry the bound beside the two lines it constrains.
2. **`last_timeout_s` is null in practice.** A fetch is planned only after a reply arrived, so the
   link has to drop inside the few milliseconds between the plan and the fetch's arrival for a fetch
   timeout to happen at all. `statecheck.py` counts 0 in the simulated `s11a` p 0.2 seed 1 run and 0
   in the logged one. The field is structurally present and carries no information on this cell; its
   float branch is exercised only against a real `Tracker` fixture.
3. **`obs["state"]` is handed over by reference and is the same object the row carries.** A decider
   that mutated it would change the emitted row; `history` is copied per call, `state` is not. The
   two are treated differently on purpose, and `ask_decider`'s docstring now carries the reason: the
   history is a list the node keeps and appends to for the rest of the run, so one mutated element
   would travel to every later decision of the pair, while the state is rebuilt from the store at
   every call. A deep copy of the state -- `origins` is nested, so a shallow one would guard nothing
   -- would be about 8.3 million nested copies over `weight.py`'s runs, which is not worth paying to
   guard against a decider this directory does not have. **So the contract is the guard, not a copy:
   a decider must not write into `obs["state"]`.** Every decider here returns a constant or reads
   `where`, and none writes.

## The per-pair lag streams

**Opt-in, default off, and the default is bit-identical to what the engine did.** The option lives in
this directory, in `port.Engine`, a subclass of evidence/18's engine: with `pair_lag_seed` None it is
evidence/18 draw for draw, and with a seed every Mode B tick lag of the ordered pair (node,
neighbour) comes from `random.Random(f"pair_lag:{seed}:{node}:{nb}")` and each node's first-poll
offset from `random.Random(f"node_start_lag:{seed}:{node}")`. The precedent is evidence/18's own V
path (`../18/bound.py:173-180`), which is live there only when a clairvoyant plan is set, so no
ordinary run has ever drawn this way. **"Bit-identical" is measured and not asserted from the
override**: `gate.py` check 2b runs `port.Engine` with `pair_lag_seed=None` against evidence/18's own
`Engine` on two stages in both modes and requires equal event dicts. It matters for `weight.txt`'s
"old fixed-arm path" row, which is a `port.Engine` run with the streams off.

**One line had to move in the base file.** `go`'s first-poll offset was drawn inline, and `go` cannot
be overridden for one line without copying its loop, so the draw is now `_Engine.start_lag`
(`../13/replay.py:647-657`), called from `go` at `:673`. Same draw, same stream, same position in the
node loop; `gate.py` check 1 is what says the refactor moved nothing. Everything else of the option
is in this directory.

### What the guarantee is, exactly

**Pair X's n-th lag draw is the same VALUE whatever another pair decides.** The seed string
`pair_lag:<seed>:<node>:<nb>` holds nothing a decision can touch, so position i of a pair's sequence
is the i-th draw of that stream in any two runs.

**WHICH EVENT that draw serves can still move, on the other pairs as well as on the target pair.**
`Node.jitter_rng` is per NODE and is drawn once per send over BOTH of that node's neighbours, in event
order, so moving one pair's sends re-assigns that node's later jitter draws between its two pairs,
shifts the other pair's send instants, and can change how many draws that pair makes -- a lag is drawn
once per send AND once per detected poll timeout.

**Measured, on `s11a` p 0.2 seed 1 with one resolution of `h2->h1` changed from 6.0 s to 10.043 s**
(`lagcheck.py`'s own printed output):

- The two pairs of `h2` **exchange** their jitter draws for exactly four sends, positions 6 to 9 of
  each pair's send sequence: `h2->h3` changed `[6:10]` is `[6.3619, 5.539, 6.4301, 6.4867]`, which is
  `h2->h1` base `[6:10]`, and symmetrically. From position 10 on the per-node draws re-interleave in
  the original order and **each pair carries its own values again**; the script prints the slice and
  the "own again" verdict so this cites output rather than inference.
- The residue is the cumulative shift of `h2->h3`'s later send instants. That is how a pair's lag
  COUNT moves while its send count does not: `h2->h3` sends 22 polls in both runs, draws 26 lags in
  the base run and 27 in the changed one, and the extra draw is a poll-timeout detection (4 against
  5) inserted at draw index 21, i.e. one shifted send instant landed on the far side of a `link_down`.
  From its 22nd draw on, that pair's lag values are attached to different events.
- So `lagcheck.py`'s check 1 (Acceptance 3's own wording, value by value at equal indices over the
  common length) **cannot fail for any decider or any change** and is labelled in the script as a
  seeding regression guard, not as evidence about this run. The quantity a decision can move is
  printed as check 1b, whose verdict on this run is **NOT HELD**.

Aligning the jitter as well would change what every existing stage does and is not asked for by the
ticket; the ticket's own remedy for the residue is to average over Mode B passes, and how many passes
that takes is still open. **Whether that has to be measured before the first label is made is now a
decision, not a detail**: check 1b says the residue reaches the other pairs of the run, not only the
pair being probed.

## The resolution-time decision is the one that acts

The send-time interval only arms a provisional timer that the resolution replaces, so the labels of
the second half will target resolution-time decisions only. `port.Engine` counts the send-time
provisional timers that fired before their own poll had resolved, which is the only way to see one:
such a poll leaves nothing in the event log that tells it apart from a poll that resolved first. The
count is a count of polls whose send-time interval decided the next send on its own. **Over the
24 000 simulations of `weight.py`, of which the 16 000 port simulations are the ones that can show
any, it is 0.**

**Which runs that number covers.** The 24 000 is `weight.py`'s own count. The other three scripts
built `port.Engine` instances and discarded the counter, which left the claim narrower than "the runs
of this half", so each of them now prints its own count beside its checks: `gate.py` over check 2b's
port runs, `statecheck.py` over its one `s11a` run, `lagcheck.py` over its four. All are 0. The
reason they are is the bound of deviation 1: the smallest applied interval of the choice set exceeds
the Interest lifetime plus a tick, so every poll resolves before the next send can supersede it.
`gate.py` check 2's own simulations are not covered at all and cannot be: they run through
evidence/18's `B.Engine`, which has no counter. They are also outside the question D5 asks -- they are
rule and fixed-arm runs with no decider, so no decision of theirs is a candidate for a label.

## The weight

**`weight = (delay(10.043) - delay(3.695)) / (polls(3.695) - polls(10.043))`, in milliseconds of
record delay per state poll** -- the value at which `delay + weight x polls` is the same for the two
outer arms. `delay` is the headline delay of evidence/20 and 21 (evidence/14's early-record scorer,
pooled `sum(lat4)/sum(n4)` over the runs of one replicate stream, `../14/after.py:39-65`), `polls` is
that same scorer's windowed poll count per run, and both are the median over the replicate streams,
which is what evidence/21's own pass line reads.

**The cell** is p 0.2, workload `w0`, the s6 outage shape (8 to 12 s), seeds 1 to 100, 40 Mode B
streams, with the run dicts built by `../21/predict.py`'s own `run_dict` (logged exogenous inputs
where a run of that shape exists, the generator's instants and a borrowed clock otherwise). The two
arms are run **through the port**, as constant deciders over the three s11a arms, with the per-pair
lag streams on: the weight is meant to be the weight of the mechanism the labels will be made with.

The value, the two scores, the spread, p 0.4, the old fixed-arm path and the ticket's own estimate
are in `weight.txt`; the number alone is in `weight.json`.

**The arms being level under it is an arithmetic identity, not evidence** -- the weight was solved for
it, and the two scores agree to every digit of a double. What carries information is the spread of
the per-stream weights, which `weight.txt` prints, and the distance to the other two values printed
beside it.

## Seeds

**Seeds 121 to 160 appear nowhere here.** They are the measuring seeds of this effort and the
learning side may not touch them. Every script of this directory that takes seeds passes them through
`port.check_seeds`, which refuses any seed above 100 with an error. (`weight.py` loads
`../21/predict.py`, which loads `../20/predict.py`, whose module body defines that seed range as a
constant; no script here reads it.)

## What each check proves

| script | what it states | what it does not |
|---|---|---|
| `gate.py` check 1 | with no decider the edited replay is `fd6ce55`'s, event dict for event dict, on a fixed choices arm, a policy stage, a formula rule on a donor run, three choices with the re-decision off and with it on, and the logged `s11a` stage, in Mode A and Mode B | nothing about a run WITH a decider |
| `gate.py` check 2 | the main cell (p 0.2, `w0`, donor `s6b10p043`, seeds 1 to 20, 12 rules) reproduces every printed column of `../18/bound_cells.csv` and `../15/onoff_cells.csv`, through evidence/18's own job and row builders; a csv column the built row does not carry is a failure, not a column skipped | it is the replay against its own archived output, not against the testbed. On a difference it prints the simulated and archived string of each differing column, as the precedent does; it derives no number of its own, so there is nothing further to suppress |
| `gate.py` check 2b | `port.Engine` with `pair_lag_seed=None` emits the same event dicts as `../18`'s `Engine`, on two stages in both modes, so the subclass is inert with the streams off | nothing about a run with the streams ON; that is `lagcheck.py`'s |
| `gate.py` check 3 | the port refuses a decider without choices, without the re-decision, beside a policy, beside a rule, and refuses an interval outside the choice set | beyond the ticket's Acceptance 1. The refusal is `interval not in choices`, so a numerically equal value of another type is accepted; see "What this step cannot show" |
| `gate.py` check 4 | `git status --porcelain` is unchanged across the run, so no committed csv was overwritten | |
| `statecheck.py` | the key sets of the port's state, of a real `Tracker` plus `_PollReplies`, and of the logged `state_poll` and `state_poll_reschedule` rows of one `s11a` run are equal, nested, with one extra or missing key anywhere a failure; every observed value type is inside a set declared from the source; the history element's keys are the logged action's less `neighbor` plus `ts_ms`; the emitted row carries the decider's own input; and the state the BUILDER produces is unchanged when every `_Origin.hist` is poisoned | **SHAPE ONLY.** Nothing about the VALUES agreeing: a wrong number in any of the six fields passes every line of it. Nothing about `history` mirroring a real field, because the source has no such field. And the poison check is about the builder, not the content: the advertised `hi` IS hist-derived, at reply time |
| `lagcheck.py` check 1 | with the streams on, one changed resolution of one pair leaves every other pair's lag sequence equal value by value over the common length and every node's offset equal -- which is Acceptance 3's wording and a TAUTOLOGY under the per-pair seeding, so it is a regression guard on the seed strings and on the routing (`sim_rng` None, so an unrouted draw raises) | **it cannot fail**, for any decider and any change, so it says nothing about this run |
| `lagcheck.py` check 1b | the quantity a decision CAN move, printed and not gated: each other pair's lag draw count, draw-kind sequence, applied intervals, and the moved slice with whose values it carries. On this run the verdict is **NOT HELD** -- `h2->h3` draws 26 lags against 27, its kind sequence first differs at index 21, and the two pairs of `h2` exchange four sends | it is not a pass/fail gate, because Acceptance 3 asks for check 1 and making 1b the gate would redefine it |
| `lagcheck.py` checks 2, 3 | the change is not vacuous; and with the streams off the same change moves 7 of the 7 other pairs' lag values | that two candidates see the same lags |
| `weight.py` | one number, with the two scores under it, the per-stream spread, the censored share of each arm, the same four numbers at p 0.4, the old fixed-arm path's value and the ticket's own estimate beside them; and the provisional-timer count | that the weight is right for the testbed, or for p 0.4, or for a labelled decision |

## What this step cannot show

- **That a label is the testbed's answer.** A label is a score inside the replay. No run anywhere has
  been made with a decision maker that reads the state, so the port itself has no measurement behind
  it; the nearest thing is evidence/13's validation of the replay's fixed arms and two stage-1
  policies, and evidence/21's own gate, both of which read no decision state.
- **That the state is enough.** The strict match is of shape, not of values and not of sufficiency.
  Whether `decision_state` plus eight resolutions carries what a policy needs is ticket 11's question.
- **That the state's VALUES are right.** No check in this directory compares a number in
  `obs["state"]` against a reference. `statecheck.py` compares path sets and type-name sets, and its
  checks 4 and 5 compare the port against itself, so a decider could have been handed a wrong
  `missing`, a wrong `notified_max` or a `last_timeout_s` in milliseconds and every line would still
  print PASS. Three fields have no value coverage from any check or any run of this half.
- **That the situations the main cell never enters are built right.** In the `s11a` p 0.2 seed 1 run,
  `in_flight` is 0 at every call, `truncated` is False at every call (at most two origins per node
  against a cap of eight), `notified_max` never exceeds `hi` (0 of 1519 advertised cells),
  `last_timeout_s` is null throughout and `last_poll_status` is never "pending". The origin cap, the
  truncation flag, the sparse-extra arithmetic and the millisecond-to-second conversion are therefore
  exercised for their SHAPE only, against a `Tracker` fixture.
- **That a decider returning a numerically equal value of another type is caught.** The guard is
  `interval not in nd.choices`, so an `int` 6 passes beside the `float` 6.0 and the emitted row then
  carries `interval_s: 6`. Harmless for a reader that keys on the number, wrong for one that formats
  the interval into a string key. No reader in this directory does, so it is left as it is and named
  here.
- **That the weight transfers.** It is this cell's indifference point between two arms. At p 0.4 the
  same construction wants a value about 1.28 times as large, printed in `weight.txt` and acted on
  nowhere.
- **That two candidates are compared on the same randomness.** See the jitter residue above. The lag
  streams hold each pair's lag VALUES still, and `lagcheck.py` check 1b measures that they do not hold
  the events those draws serve still, on the other pairs of the run and not only on the pair being
  probed. **Whether the ticket's remedy (averaging over Mode B passes) has to be measured before the
  first label is made is therefore an open decision, not a detail.**
- **That a label is cheap enough, or stable.** Both are Acceptance 5 and 6 and are not in this half.
- **Anything about the live loop.** The port is on the replay only; whether a decision maker fits
  inside the agent's 100 ms loop is ticket 12's question.

## Not specified here

**Acceptance 5 (one run labelled), 6 (the stability of a label under two Mode B pass counts) and 7
(the single text rendering of the observable state) are out of this half** and wait for a decision by
the user on the text form. No labelling tool is written, no `README.md`, no `RUN.txt`, and the
ticket's Result section is not filled in. The ticket's own open points that this half does not close:
how many Mode B passes damp the jitter residue, whether a weight fixed on the main cell flips a label
at p 0.4, and the field order, rounding and history form of the text rendering.

## Deliverables (this directory)

| file | what it is |
|---|---|
| `spec.md` | this file |
| `port.py` | the seed guard, the per-pair lag streams and the provisional-timer count, in one place each. Its engine subclasses evidence/18's, so every run of this directory goes through the same chain `gate.py` check 2 reproduces the published rows with -- and that pass-through is covered by `gate.py` check 2b, not assumed from the two-line override; evidence/18's `resolve` override is itself a pure pass-through when no clairvoyant plan and no rule are set, which is the case for every run here |
| `gate.py` | Acceptance 1 |
| `statecheck.py` | Acceptance 2 |
| `lagcheck.py` | Acceptance 3 |
| `weight.py`, `weight.txt`, `weight.json` | Acceptance 4 |

The edit outside this directory is `../13/replay.py`: the port (the `decider` key, `Node.decision_state`,
`_Engine.ask_decider`, the two branches, the four new per-neighbour attributes) and the one-line
`start_lag` refactor the lag streams need.
