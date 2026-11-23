# evidence/22 spec: a port for a decision maker that reads the state, the weight on one poll, and the hindsight labels

Written 2026-11-23. Ticket `../../tickets/09-replay-labels.md`, all seven Acceptance items.

**It was written in two halves, a third pass and a fourth, and they are marked**, because each one
changes facts the one before it stated. The first half (Acceptance 1 to 4: the port, the lag streams,
the weight) was written before anything was labelled. The second half (Acceptance 5, 6 and 7: the
labels, their stability, the text) implements the user's 決まったこと of 2026-11-23 (the ticket's
lines 108-121) and is in the sections marked **second half**. The **third pass** re-measured the
weight, because the weight and the label score turned out to use two different definitions of a
delay, relabelled both cells under the corrected weight, and corrected eleven things the first
half's files got wrong -- mostly checks whose pass lines claimed more than they could fail on; its
sections are marked **third pass**. The **fourth pass** changed no value: it measured why the third
step of the weight moved and how much seed noise the fixed value carries, and it put the numbers
this file reads off the label files under a script of their own. Where a later pass corrects an
earlier one, the earlier sentence is rewritten rather than left standing beside its correction; the
corrections are listed in "What the second half changed in this file", "What the third pass
changed in this file" and "What the fourth pass changed in this file".

Where this file says something the ticket does not, or departs from it, it is marked as such.

## Question, and what the halves are for

**A label on one decision is "which of the three arms is best now, if the formula rule F takes over
afterwards"** -- the first half wrote this as "which of the three arms, run to the end of the run,
scores best", which was the question before the user fixed the construction; the full definition is in
"The label". Four things have to exist before the first label can be made. The first half is the first
three plus the number the score needs; the second half is the fourth, the labels themselves, and the
one road from the state to text; the third pass is the number the score needs again, because the
first half measured it in a delay the labels were never scored in.

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
one other pair draws 26 lags in one run and 27 in the other. **The second half adds a per-pair jitter
stream, also opt-in, and on that run no other pair moves at all** -- see "The per-pair jitter stream".

**(c) One weight.** The score is `delay + weight x polls`, and the weight has to be a single number
fixed before any label exists, or a label is a statement about whatever weight produced it. **The
first half fixed a number but not the right quantity**: it levelled the arms on a delay that leaves
the censored records out, while a label's score enters them at their lower bound, so the weight was
the indifference point of something no label was measured in. The third pass re-measured it under
the label's own score and relabelled both cells; see "The correction".

**(d) A suffix.** A probed decision has to be followed by something, and what follows it is what the
label means. The user fixed it: a uniform roll-in before, one forced arm at the decision, F after.
That is the second half's own construction and it needed `obs["resolved"]` and a wider choice set to
be expressible at all.

Not answered here: whether any decision maker is better than the fixed arms, which needs a learner;
whether a label is the testbed's answer, which it is not and cannot be (see "What this step cannot
show"); and whether the labels are stable enough to learn from, which "The stability of a label"
measures and answers in the negative at every pass count tried -- with the third pass's one
qualification, that the instability is not spread evenly and a statistic on the label row orders it.

## Corrections to the ticket

- **`last_timeout_s` is the instant of the most recent FETCH timeout on that neighbour**, not a poll
  timeout (`src/semindex/tracker.py:251-262`, written in `_fail` at `tracker.py:454`). The ticket's
  line 67, which says the value "is known at the point of `resolve`", is wrong: `resolve` is a poll
  resolution and never sees a fetch timeout. The replay writes it in `ev_ftimeout`
  (`../13/replay.py:1157`), after the guard that drops a stale timeout.
- **The ticket's line numbers for `../13/replay.py` are stale.** At `fd6ce55`, before this step's
  edit: `Node` was 268-299 (the ticket says 263-287), `decide_interval` 675 (653), `resolve` 759
  (727), the choice draw 691-694 (668-671), Mode A's lags `logged_lags` 210-242 plus `tick_lag`
  534-541 (205-241), the Mode B shift comment 636-641 and the draw 542 (627-636), and evidence/18's
  per-pair lag stream `bound.py:173-180` (160-168).
- **The exchange history the ticket asks for is not a logged field.** The source keeps no bounded
  per-neighbour exchange history: its `_PollReplies.polls` is bounded in TIME by the Interest
  lifetime plus a grace second (`src/semindex/engine.py:160, :259-282`) and only two scalars of it
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
  call the decider instead (`replay.py:911` at the send, `replay.py:1051` at the resolution) and do
  **not** draw `choice_rng`. The jitter draw stays exactly where it was, in the same expression, or
  every later jitter value of that node would shift and the run would no longer be the same run with
  one decision moved.
- **A returned interval that is not one of `choices` raises `ValueError`** (`replay.py:876-879`).
  Every reader of these runs indexes the choice set with the interval.

### What the decider is handed

`decider(obs) -> interval_s`, with `obs` a dict of exactly four keys: `state`, `history`,
`resolved` and `where` (`ask_decider`, `../13/replay.py:809-880`). **`resolved` is the second half's
addition**; the first half had three.

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
`progress_map_asof` in `ev_rx` (`replay.py:927`, `:939`), and copied into the deciding node's own dict
at reply time. Poisoning the histories at the decision instant therefore cannot detect it and the
rebuild is equal by construction. **The argument for that field is not the poison check but the
source**: a real state reply carries (hi, extra) per origin and the source hands exactly that pair to
its tracker (`src/semindex/engine.py:842`), so a live node sees it. What the poison check does buy is
that a FUTURE edit cannot reach into a neighbour's store at decision time. The companion path, a
sparse extra above the advertised `hi`, is never exercised: in the `s11a` p 0.2 seed 1 run no reply
carries one, so `notified_max > hi` holds in **0 of the 923 advertised cells of `statecheck.py`'s
348 port states**, which that script now prints beside the four other situations it does not reach.
(An earlier draft gave this denominator as 1519, which is no population of this run: the 348 port
states hold 923 cells, the 297 logged rows hold 738, and `valuecheck.py`'s 316-call roll-in holds the
774 it prints. The substance did not change -- every one of those populations comes to 0 -- but the
number was not one any script produced.)

### The history

`obs["history"]` is the earlier RESOLVED polls of this node towards this neighbour, oldest first, at
most `DECIDER_HISTORY = 8` (`replay.py:93`), each one a dict of the fields a real node writes on its
own `state_poll_reschedule` row plus that row's stamp: `ts_ms`, `poll_epoch_ms`, `trigger`, `empty`,
`interval_s`, `applied_interval_s`, `clamped`. The decision being made is not in it. It is appended
in `resolve` after the row is emitted and **before** the identity early-return (`replay.py:1103`),
because a poll whose provisional timer already pointed at the new instant takes that return and a
history that skipped exactly those polls would hide the cheapest decisions of the run.

**`neighbor` is the one action field the element drops**, because `where` already carries it once
per call and repeating it on all eight elements would carry nothing: every element of one history is
the same pair's. This departs from nothing in the ticket, which does not list the element's fields;
it is stated here because `statecheck.py` tests that exact set. (The second half's reason for the
drop was that "a learner may not read `where`"; the third pass narrowed that rule -- the neighbour is
real-node context -- so the drop now stands on its own reason, and the field set is unchanged.)

### `resolved` (second half)

`obs["resolved"]` is the outcome of the poll being resolved AT THIS INSTANT:
`{poll_epoch_ms, trigger, empty}` at a resolution, and **None at a send**, where no poll is being
resolved. It is built in `resolve` and handed through `ask_decider`
(`../13/replay.py:1049-1050`, handed through `ask_decider` at `:1051`).

**Why it had to be added.** The labelling suffix is the formula rule F, which lengthens its interval
on an EMPTY reply and resets on a reply with content. Nothing else in `obs` carries the emptiness at
the decision instant: `state` carries the poll STATUS (`answered`, `timed_out`) and not whether the
reply was empty, and `history` ends one decision earlier, because the decision being made is
deliberately not in it. Without `resolved`, a decider could not BE F and "the rest of the run is F"
would have been a claim about something else.

**Why it is admissible as an input.** A real node knows all three at that instant and writes them
into the `action` of the same `state_poll_reschedule` row (`src/semindex/engine.py:692-707`).
`statecheck.py` check 4b holds the port to that twice over: the KEY set against the logged action's,
less `{neighbor, interval_s, applied_interval_s, clamped}` -- the first for the reason the history
drops it, the other three because they are the decision's own OUTPUT and handing a decider the
interval it is about to return would be handing it the answer -- and the VALUES against the action of
the very row that decision wrote. That value comparison is the one place in `statecheck.py` that
looks at a number.

### `where`, and who may read which half of it

`obs["where"]` is `{node, neighbor, kind, ts_ms, ordinal}`, where `kind` is `state_poll` or
`state_poll_reschedule` and `ordinal` counts this run's decider calls of that kind from 0 in call
order. **It is two different things in one dict, and the third pass splits them**, because the first
half's rule ("a learner must never read `where`") was too wide and withheld context a real node has:

| field | what it is | who may read it |
|---|---|---|
| `neighbor` | the peer this decision is about | **real-node context.** The source decides the interval per neighbour (`_choose_poll_interval(neighbor)` at `src/semindex/engine.py:710`, `_advance_rule_interval(neighbor, ...)` at `:571`), keys the next due instant by it (`self._next_state_poll[neighbor]`, `:691`) and writes it into the `action` of the row it logs (`:692-699`). So a node knows it. `observe.py` reads it -- never as a name, only to say which of the advertisers listed in the state is the peer being polled |
| `kind` | send or resolution | real-node context: a node knows whether it is arming a timer or reacting to a reply |
| `ts_ms` | the decision's own stamp | real-node context: a node has a clock. It is only ever subtracted, so no record carries an absolute instant |
| `node` | the deciding node's NAME | **tool only.** A policy keyed on `h2` is a policy about one topology |
| `ordinal` | this run's decider calls of that kind, counted across all five nodes | **tool only.** No node can know another node's decision count; this is what lets a tool name one decision of a run again on a later pass |

**The rule is therefore not "never read `where`"**: it is that nothing which reaches a score may
depend on a NAME or on the `ordinal`, because the live loop cannot hand either over in a form that
transfers to the testbed. `ask_decider`'s docstring now carries the same split. What the first half's
wider rule cost is in "The polled neighbour": an interior node (`h1`, `h2`, `h3`) has two
neighbours, and a record that may not know which one the decision is about cannot tell its own peer's
advertised value from the other peer's.

### What does not move

**With `decider` None every existing mode emits event dicts equal to `fd6ce55`'s, draw for draw**;
that is `gate.py` check 1, over seven configurations in both modes (the seventh, the stage-1 fixed
branch, is the third pass's addition: the list said "every mode" while that branch was never run).
The new bookkeeping
(`real_consec`, `last_status`, `last_timeout_ms`, `poll_hist`, `replay.py:331-339`) lives in new
attributes, is maintained in every mode and is read only when a decider is set; `nd.consec` and every
emitted row of an old mode are untouched.

**In decider mode the rows widen.** The `state_poll` and `state_poll_reschedule` rows carry the full
`state` -- the very dict the decider was handed, built once -- and the reschedule action carries
`empty`, as the source's own rows do under a rule or the re-decision knob (`engine.py:700-707`). So a
tool can read these rows like testbed rows. The old modes keep the one-field and two-field states
they had: widening those would rewrite rows already measured against the testbed.

### The choice set a labelling run declares (second half; this is the decision the first half asked for)

The first half flagged that F's reachable intervals leave the three-arm set, so a decider that
emulates F would raise under the guard above, and it listed two ways out without choosing. **The
first one is taken: the labelling run declares `choices` as the union of the three arms and every
interval F can reach from them.** The guard is not loosened. The second way out -- running the suffix
on the rule path -- needs a mode switch inside one run, which the port does not have and which would
reopen the decider/rule exclusivity.

The set is computed, not typed: `port.union_choices()` iterates F's own arithmetic
`min(I * m, I_max)` from each arm until it closes, so every value in the set is the float that
arithmetic produces and `interval not in choices` holds by construction rather than because a typed
literal happened to round the same way. It comes out as

    [3.695, 5.5424999999999995, 6.0, 8.313749999999999, 9.0, 10.043]

-- the three arms, 5.5425 and 8.31375 above 3.695, and 9.0 above 6.0; everything above 10.043 is
clamped back to 10.043, which is already an arm.

**The uniform draw of the roll-in is over the THREE ARMS only, never over this union.** The draw is
made by the decider itself (`label.Labeller._draw`), with the engine's own index arithmetic, so a
roll-in decision is the decision the uniformly drawn stage would have made. The union exists for the
guard, not for the draw.

**What the wider set costs.** A reader counting rows per arm on a labelling run sees six intervals
instead of three. Nothing already measured is touched: no existing stage gets a wider set, and
`weight.py` still runs the three.

**It is inside the bound of deviation 1**, which the first half said was the second half's job to
check and not to inherit: `labelcheck.py` check 5 computes
`min(choices) * (1 - choice_jitter) * 1000.0` as 3325.5 ms against the 2100 ms of the Interest
lifetime plus one tick, a margin of 1225.5 ms -- the check always had the `x 1000`, which is why the
formula's missing factor never made a run wrong. So `pending` stays unreachable on the labelling runs
too, and the provisional-timer count is 0 over every simulation any of the three passes made.

### Three deviations from the source, stated

1. **`last_poll_status` of a poll that never resolved.** The replay writes "pending" at the send and
   the real status at the resolution. A poll superseded before it resolved therefore reads "pending"
   where the source, judging status off the clock, would read "timed_out", and such a poll is also
   missing from `real_consec`, which the source would fold in as an expired timeout. How many such
   polls there are is exactly what the provisional-timer count below measures; it is 0 in every run
   of all three passes. **So "pending" is written and never read** anywhere in this directory: the only
   `last_poll_status` a decider here ever saw was `None` (the first send of a pair) or a real
   resolution status. `statecheck.py`'s printed type set for the field, `NoneType|str`, is the same as
   the logged rows'.

   **The bound under which the deviation is unreachable, stated because the second half has to check
   its own choice set against it.** A poll can only be superseded before it resolves if the next send
   comes first, so the deviation is unreachable while the SMALLEST APPLIED interval,
   `min(choices) * (1 - choice_jitter) * 1000.0`, exceeds `POLL_TIMEOUT_MS + TICK_PERIOD_MS`. **The
   x 1000 is the third pass's correction**: `choices` are in SECONDS and the two engine constants are
   in milliseconds, so the formula as the first half wrote it compared 3.3255 against 2100.0 and was
   out by a factor of a thousand -- here and at both comment sites in `replay.py`. Every number
   plugged into it was always converted correctly, so no run and no check was ever wrong; the formula
   was. The right-hand side is 2 s, the
   Interest lifetime, plus one 100 ms tick, because the timeout is DETECTED at
   `sent + POLL_TIMEOUT_MS + tick_lag("timeout")` and not at `sent + POLL_TIMEOUT_MS`. For the first
   half's choice set that is 3.695 x 0.9 = 3.3255 s against 2.1 s, a margin of 1.2255 s. **The
   labelling's wider union set is checked against this sentence by `labelcheck.py` check 5**, which
   is where the second half discharges it rather than inheriting it. A choice set that breaks the
   bound would need the divergence modelled (resolve the
   stale pending poll in `ev_poll` before building the state) rather than documented. The comments at
   `replay.py:721-731` and `:1005-1019` carry the bound beside the two lines it constrains.
2. **`last_timeout_s` is null on the one run the first half measured, and rare but NOT absent over the
   labelling** (corrected by the second half). A fetch is planned only after a reply arrived, so the
   link has to drop inside the few milliseconds between the plan and the fetch's arrival for a fetch
   timeout to happen at all. `statecheck.py` counts 0 in the simulated `s11a` p 0.2 seed 1 run and 0
   in the logged one, and `valuecheck.py`'s roll-in run of the same cell counts 0 in its 316 calls.
   Over the full labelling the field is set in **12 of the 16 071 labelled decisions at p 0.2 (all of
   them seed 90) and 17 of the 16 148 at p 0.4 (12 on seed 36, 5 on seed 48)**, where `observation`
   turned the stamp into `fetch_timeout_age_s` values from 2.1 s to 43.0 s and from 0.4 s to 93.9 s
   respectively, all of it `specnums.txt`'s and recounted at every run of it (the second half
   wrote 49.6 s for the p 0.4 maximum, which is the twelfth of the 17
   values and not the largest; the third pass re-read all 17 from both the superseded and the new
   label file and they are the same list, since the field is weight-independent). So the
   millisecond-to-second conversion and the age rendering DID run on labelled
   data that a learner will read, and the field is not information-free on these cells -- it is rare.
   What is still true is that the VALUE has no reference: nothing compares those 29 numbers against a
   source object, because the rebuilt `Tracker` of `valuecheck.py` has planned no fetch, so the float
   branch is verified only against a real `Tracker` fixture, by `statecheck.py`.
3. **`obs["state"]` is handed over by reference and is the same object the row carries.** A decider
   that mutated it would change the emitted row; `history` is copied per call, `state` is not. The
   two are treated differently on purpose, and `ask_decider`'s docstring now carries the reason: the
   history is a list the node keeps and appends to for the rest of the run, so one mutated element
   would travel to every later decision of the pair, while the state is rebuilt from the store at
   every call. **What a mutation would and would not reach** (third pass): it corrupts the row that
   decision wrote and nothing else. `Node.decision_state` allocates a fresh dict at every level on
   every call and shares no mutable container with the node (`_Origin.advertised` holds `(int,
   tuple)`), so it cannot reach the NEXT decision's input; `ask_decider`'s docstring said it could
   and no longer does. A deep copy of the state -- `origins` is nested, so a shallow one would guard
   nothing -- would be one nested copy per decider call, and `weight.txt` now prints that count from
   its own port runs: **12 354 458 over the 32 000 port simulations**, which is the price the contract
   buys out of. The first half's "about 8.3 million" was an estimate and a wrong one: it
   multiplied all 24 000 simulations by the 348 calls `statecheck.py` measures for a 6.0 s arm, which
   counts the 8000 decider-free runs of the old fixed-arm path and prices both outer arms at the
   middle arm's call count. **So the contract is the guard, not a copy:
   a decider must not write into `obs["state"]`.** Every decider here returns a constant or reads
   `where`, and none writes. **`resolved` is copied per call** (second half), for the history's
   reason rather than the state's: it is a small flat dict of three scalars, so the copy costs
   nothing, and `label.py`'s own decider keeps a reference to nothing it was handed. So of the four
   keys of `obs`, three are copies and only `state` is shared with the row.

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
  `h2->h1` base `[6:10]`, and symmetrically. **From position 10 on `h2->h3` carries its own values
  again** -- `own again: True`, read over positions 10 to 22 of two sequences of 22 -- while the
  probed pair `h2->h1` prints `own again: False`, because it sent 22 polls in the base run and 21 in
  the changed one, so there is no comparable tail of it at all.
  **The verdict is a real one as of the third pass, and was not before.** It used to be `a[hi:n]
  == b[hi:n]`, where `hi` is one past the LAST differing index below `n`: that slice has no
  differing index by construction, so the comparison was `[] == []` or a run of equal values and
  could not come out False -- not even when the exchange ran to the end of the shorter sequence,
  and not even when the two sequences had different lengths and there was a tail it never looked
  at. It is now `hi < n and len(a) == len(b)` over the per-send APPLIED INTERVALS -- which is
  where a re-assigned jitter draw shows up, the raw lag draws being what check 1 compares at equal
  indices -- and the printed row carries `n` and both lengths so a reader can see which case it
  is. The first half cited the old verdict as output; what it cited could not fail.
- The residue is the cumulative shift of `h2->h3`'s later send instants. That is how a pair's lag
  COUNT moves while its send count does not: `h2->h3` sends 22 polls in both runs, draws 26 lags in
  the base run and 27 in the changed one, and the extra draw is a poll-timeout detection (4 against
  5) inserted at draw index 21, i.e. one shifted send instant landed on the far side of a `link_down`.
  From its 22nd draw on, that pair's lag values are attached to different events.
- So `lagcheck.py`'s check 1 (Acceptance 3's own wording, value by value at equal indices over the
  common length) is **near-certain to pass for any decider and any change** under the per-pair
  seeding, and is labelled in the script as a seeding regression guard, not as evidence about this
  run. It is not inert: re-assigning a draw between two pairs of one node makes it FAIL, which is
  exactly what the per-node jitter stream does to the applied intervals. The quantity a decision can
  move is printed as check 1b, whose verdict on this run is **NOT HELD**.
- **Its first-poll-offset half was a conjunct that could not come out False, and the third pass
  removed it from the verdict.** `start_lag` runs in the engine's initialisation loop before any
  event or decider executes, so no decision can move an offset in ANY configuration, and check 3's
  streams-off control does not control it either. The offsets are now printed as a reported fact
  ("reported and not gated"), and the PASS line gates on the moved lags alone.

### The per-pair jitter stream (second half), and what it fixed

The first half left the residue open and said that aligning the jitter as well would change what
every existing stage does. **The second half adds it as a second opt-in option, `pair_jitter_seed`,
default off**, so no existing stage changes: with it None the jitter of a poll comes from
`Node.jitter_rng`, the node's own stream, which is what every published cell was measured with, and
it is passed through on Mode A as well, which replays a logged run's own lags. With a seed, the
jitter comes from `random.Random(f"pair_jitter:{seed}:{node}:{nb}")`.

The whole change is in `port.Engine.decide_interval`, which rebinds `Node.jitter_rng` to the pair's
stream and then calls the base method: `jitter_rng` is drawn in `decide_interval` and nowhere else in
the engine (a resolution re-applies `pend[1]`, the send's own draw), so one rebinding covers every
draw site without copying the base method's four branches.

**Measured, on the same run and the same single change `lagcheck.py` check 1b uses** (the fifth
`state_poll_reschedule` of `h2->h1` on s11a p 0.2 seed 1, 6.0 s to 10.043 s), printed by the script
as check 1c:

| | 1b, per-node jitter | 1c, per-pair jitter |
|---|---|---|
| other pairs whose lag draw count or kind sequence moved | 1 of 7 (`h2->h3`, 26 against 27 draws) | **0 of 7** |
| other pairs whose per-send applied intervals moved | 1 of 7 | **0 of 7** |
| draw counts that moved | `('h2','h3','timeout')` 4 against 5 | only the probed pair, `('h2','h1','poll')` 22 against 23 |
| verdict | **NOT HELD** | **held on this run** |

So on this run the per-pair jitter stream closes the residue the first half measured: the only pair
whose draws move is the pair whose decision was changed, and it moves because it really did send one
more poll.

**The two columns are measured on two different runs, which the third pass made visible.** Each
column compares a base against a changed run drawn from its OWN jitter source, and turning the
per-pair jitter on changes the run: `lagcheck.py` now prints all six decider-call counts, and the
jitter-on base makes **352** calls where the streams-on and streams-off bases both make **348**.
Different jitter values move the send instants, so the run reaches a different number of decisions.
Nothing in 1b or 1c is wrong because of it -- each column is internally consistent -- but "0 of 7"
and "1 of 7" are not two readings of one run.

**Neither 1b nor 1c is a gate**, for the first half's reason: Acceptance 3 asks for check 1, and
making either the gate would redefine it.

**What 1c does not say.** It is one run, one changed decision, one pair. And a residue that survives
this option on another run need not be randomness at all: a shifted send moves WHEN a record is
advertised, so another pair's store, its fetch plan and therefore its fetch-timeout lag draws can
move through the simulated world rather than through a generator. `lagcheck.py`'s docstring says to
read the draw-KIND sequence before calling such a residue a seeding problem.

**The residue is damped twice over, not once.** The ticket's own remedy -- averaging over Mode B
passes -- is still used, and the pass count is Acceptance 6's question; see "The stability of a
label". The two remedies answer different things: the jitter stream removes the coupling between
pairs at one decision, and the passes average the randomness that follows the probed decision on the
probed pair itself.

## The resolution-time decision is the one that acts

The send-time interval only arms a provisional timer that the resolution replaces, so the labels
target resolution-time decisions only, which is what the second half does. `port.Engine` counts the
send-time
provisional timers that fired before their own poll had resolved, which is the only way to see one:
such a poll leaves nothing in the event log that tells it apart from a poll that resolved first. The
count is a count of polls whose send-time interval decided the next send on its own. **Over the
40 000 simulations of `weight.py`, of which the 32 000 port simulations are the ones that can show
any, it is 0.**

**Which runs that number covers.** The 40 000 is `weight.py`'s own count, which the third pass's two
extra sweeps raised from 24 000 (the script now derives both totals from its sweep list instead of
printing a literal, so a sweep added later cannot leave them wrong). The other three scripts
built `port.Engine` instances and discarded the counter, which left the claim narrower than "the runs
of this directory", so each of them now prints its own count beside its checks: `gate.py` over
check 2b's port runs, `statecheck.py` over its one `s11a` run, `lagcheck.py` over its six.
**The second half's scripts print theirs too**: `labelcheck.py` over its F runs, `valuecheck.py` over
its roll-in run, `stability.py` over the 154 032 arm simulations and 20 roll-in runs of its 20 runs
(3209 decisions x 3 arms x 16 passes), and `label.py` over every
simulation of the full labelling, and `label.py` and `stability.py` EXIT NON-ZERO on a non-zero
count rather than printing it. All are 0. The
reason they are is the bound of deviation 1: the smallest applied interval of the choice set exceeds
the Interest lifetime plus a tick, so every poll resolves before the next send can supersede it.
`gate.py` check 2's own simulations are not covered at all and cannot be: they run through
evidence/18's `B.Engine`, which has no counter. They are also outside the question D5 asks -- they are
rule and fixed-arm runs with no decider, so no decision of theirs is a candidate for a label.

## The weight (first half, re-measured by the third pass)

**`weight = (delay(10.043) - delay(3.695)) / (polls(3.695) - polls(10.043))`, in milliseconds of
record delay per state poll** -- the value at which `delay + weight x polls` is the same for the two
outer arms.

**`delay` is `score.py`'s, the one a label is decided by** (third pass; the first half's definition is
in the correction below): per run,
`(sum of the delays of the delivered records + sum over the censored records of the time from the
record's creation to the end of the window) / (delivered + censored)`. `polls` is evidence/14's
windowed poll count per run (`../14/after.py:39-65` fixes which records, which window and which
polls). Both are the **mean over the 4000 runs** of an arm, a run being one (seed, stream).

**The cell** is p 0.2, workload `w0`, the s6 outage shape (8 to 12 s), seeds 1 to 100, 40 Mode B
streams, with the run dicts built by `../21/predict.py`'s own `run_dict` (logged exogenous inputs
where a run of that shape exists, the generator's instants and a borrowed clock otherwise); the
generator gate that `run_dict` rests on is run first by `weight.py` and passed. The two arms are run
**through the port**, as constant deciders over the three s11a arms, with the per-pair lag streams
AND the per-pair jitter streams on -- the labelling's own engine configuration. Both per-pair seeds
are `rep_seed("pairlag:s6b10p043", p, seed, rep)`, which carries no arm; **the prefix is part of the
answer** and is named in `weight.txt` and in `weight.json`, because another stream family moves the
number.

**The value is 104.29595912307323 ms per poll.** The four steps, the two arms' mean scores, the
per-stream spread, p 0.4, the old fixed-arm path and the ticket's own estimate are in `weight.txt`;
the number and the superseded one are in `weight.json`.

**The arms being level under it is an arithmetic identity, not evidence** -- the weight was solved for
it, and the mean of a sum is the sum of the means, so the two mean scores agree to every digit of a
double (both 40954.465203126296, difference exactly 0). What carries information is the spread of
the per-stream weights -- **min 99.899, median 104.661, max 107.251 over the 40 streams** -- and the
distance to the other values printed beside it.

### The correction: the weight and the label score were two different delays (third pass)

**The first weight was measured before the label score was decided, the two disagreed, and this is a
correction of a definition.** It is not a choice made after looking at labels: the four steps below
were fixed as a sequence before any of them was run, step 1 is required to reproduce the committed
value exactly, and nothing was tuned at any step.

What the two definitions were:

- the committed weight levelled the arms on `../14/after.py`'s `early_score`, whose delay is
  `sum(lat4) / sum(n4)` over the **delivered records alone**, pooled over the 100 seeds of one
  replicate stream and then taken as the **median over the 40 streams**;
- a label's score enters a **censored record at its lower bound** and is a **per-run** number.

So the weight was the indifference point of a quantity no label was ever measured in, and every
label was a comparison under a weight solved for a different number. Leaving the censored records
out also pays an arm for the records it never delivered, which is the reason the label score enters
them in the first place: the 10.043 s arm censors 0.00400 of its records at p 0.2 and 0.05458 at
p 0.4 while the 3.695 s arm censors none, so dropping them shortened the long arm's delay and only
the long arm's, and the committed weight was biased **downward**.

`score.py` now holds the one definition and both scripts call it. `labelcheck.py` check 6 still
holds the rewrite to `early_score` on the four numbers they share, and `weight.py` makes the same
comparison on **all 40 000 of its own simulations** (0 disagreed), so the record set, the window and
the poll count are still evidence/14's.

**The four steps, each printed on its own in `weight.txt`** so that a reader sees what each change
moved rather than one number replacing another:

| step | what changes | weight, ms per poll | moved |
|---|---|---|---|
| 1 | as committed: censored left out, pooled per stream, median over streams, per-node jitter | **100.62168514446252** | -- (reproduces the committed value exactly) |
| 2 | the same with every censored record at its lower bound | 101.46704705857933 | +0.845362 (+0.84 %) |
| 3 | step 2 with the per-pair jitter streams on, i.e. the labelling's engine | 104.46891354254315 | +3.001866 (+2.96 %) |
| 4 | the per-run mean of the label's own score under step 3's engine | **104.29595912307323** | -0.172954 (-0.17 %) |

Step 4 and not step 3 because a label is one run's score and labels are compared run by run.
**How much of the step 3 to step 4 move is real**: every run of these cells has exactly 240 records
created at or before the cut (`weight.txt` prints the set of values it saw over BOTH cells, `[240]`),
so the denominator is the same in every run and step 3's pooled delay equals the mean of the per-run
delays *inside* a stream. What separates the two steps here is only the mean over the 4000 runs
against the median
over the 40 streams. On a cell where the record count varied per run the two would differ by the
weighting as well.

At p 0.4 the same four steps give 128.854951, 139.289936, 140.082421 and 139.668108, so the
step-4 weight there is 1.34 times the main cell's. It is printed and acted on nowhere.

### Why step 3 moved the weight, and how much seed noise the fixed value carries (fourth pass)

**Step 3 was the largest of the four moves, +3.001866 ms, and the file used to say only that it
happened.** The question a reader is left with is whether the move is a property of the jitter keying
or sampling noise of the step-2 measurement. There is a mechanism that would make it noise: the
node's own jitter stream is seeded `state_poll_interval_jitter:{seed}:{name}`
(`../13/replay.py:329`) and carries NO replicate, so in steps 1 and 2 all 40 streams of a seed draw
their jitter from one sequence per node and the jitter is averaged over 100 independent draws, while
steps 3 and 4 key it per (seed, replicate) and average over 4000. A jitter draw happens in both
configurations -- `port.Engine.decide_interval` rebinds `Node.jitter_rng` only when a per-pair seed
is given and leaves the node's own stream in place otherwise -- so this is a change of keying and not
jitter being switched off.

`weight.py` measures it and reports; it changes no value, and `weight.json` carries none of these
numbers. **The reading rule was written into the script before any of them existed**: claim 1, step
2's jackknife standard error over the seeds is at least 2.0 times step 3's, which is what averaging
the same jitter over 100 draws instead of 4000 would do; claim 2, the per-seed PAIRED move of the
weight between the two steps is within 2.0 standard errors of zero. Both hold and the hypothesis is
supported; claim 2 failing at 3.0 standard errors or more and it is refuted; anything between is
undecided.

| quantity (`weight.txt`) | value |
|---|---|
| step 2's weight, jackknife standard error over the 100 seeds | 101.467047, se 3.167161 ms |
| step 3's | 104.468914, se 2.094254 ms |
| the step 2 to step 3 move against those | +3.001866 ms = 0.95 x step 2's se, 1.43 x step 3's |
| step 2's se against step 3's | 1.51 x, where claim 1 asks for 2.0 x |
| the per-seed paired move of the weight | +2.928395 ms per poll, se 2.493554, 1.17 se from zero |
| the paired move of the numerator | +468.050293 ms, se 399.214386, 1.17 se |
| the paired move of the denominator | -0.098500 polls, se 0.153737, 0.64 se |
| the paired move of each arm's delay | -185.794261 ms (3.695 s), +282.256031 ms (10.043 s) |
| step 4, the FIXED value, jackknife standard error | 104.295959, se 1.685586 ms |

**The rule's verdict is UNDECIDED and `weight.txt` prints it as that.** Claim 2 holds: the paired
move is 1.17 standard errors from zero, so the two keyings do not measure a different quantity at
these 100 seeds. Claim 1 does not hold: the per-pair keying cuts the standard error by 1.51 and not
by the 2.0 the rule asked for. **What that failure says is that the jitter is not the only thing
varying from seed to seed.** The outage instants and the record schedule vary too, and they are
IDENTICAL in steps 2 and 3, so they sit inside both standard errors and dilute the ratio however much
of the variance the jitter carries. Splitting the two would need a sweep that holds the jitter draws
fixed while the lags vary, and no run in this directory does that. So this measurement bounds the move
and does not explain it.

**The two numbers to carry away are the first row of that table and the last**, and they are two
kinds of quantity. The +3.001866 ms keying move is 0.95 times the step-2 measurement's own standard
error and 1.17 of its own paired one, so **the keying move is not larger than the seed noise of the
measurement it corrects**: at 100 seeds this cell cannot tell step 2's 101.467047 from step 3's
104.468914. The whole distance from the committed value to the fixed one, +3.674274 ms, is **1.16
times step 2's standard error -- larger than one standard error and inside two**, and +0.845362 ms
of it is the step 1 to step 2 move, which is the same 8 000 simulations scored twice and carries no
seed-to-seed draw a standard error could cover. So "the correction is inside the noise" is a
sentence about the keying move and not about the correction, and the part of the distance a rerun on
other seeds could move is the keying move alone. (This paragraph said the stronger thing until the
fifth pass; `weight.txt`'s own reading, which it now follows, is in the lines quoted above.) The
correction still stands on its reasoning -- the weight
has to level the arms on the delay a label is scored by, and step 1 reproduces the committed value
exactly, so the four steps are one quantity changing -- but it does not stand on the size of the
move. And the fixed value is **104.295959 ms per poll with a jackknife standard error of 1.685586 ms**
over the 100 seeds, about 1.6 % of itself: larger than two of the three moves the four steps print
and over half the third. Any later reading of a margin against "one poll's weight" carries that much
noise on the poll's price.

Two caveats on the method, both in `weight.txt`. The jackknife of steps 2 and 3 is taken on a MEDIAN
over the 40 streams, where the delete-one estimator is an approximation and its leave-one-out values
can sit at a few levels; step 4's is a mean and has no such caveat, and the paired comparison does not
use the median at all. And the per-seed paired quantities pool the 40 streams of a seed where steps 2
and 3 take the median over them, so the paired move (+2.928395) and the four-step table's move
(+3.001866) are two estimators of one quantity rather than the same arithmetic twice.

**What this cost.** The labels were remade from scratch under the new weight; the count of labels
that changed is in "What came out". The old `weight.json` sha256 was recorded in the second half's
hand-off and is not pinned by any file in this repository, so nothing in the tree had to be
unpinned; the superseded value and the reason it was replaced are in `weight.json` itself.

## The label (second half)

### What a label means

**To label one resolution-time decision k of a run: every decision before k is a uniform draw over
the three arms, decision k is forced to one arm, and every decision of every node after k is made by
the formula rule F. The run is simulated to its end once per arm and the arm with the best score is
the label.** So a label says **which arm is best now if F takes over afterwards**.

F is `../20/predict.py`'s pinned `F_MEM = (3.695, 1.5, 10.043, 3.695, None)`, turned into a rule by
`../18/bound.py`'s `member_rule`: start at 3.695 s, multiply by 1.5 on an empty reply up to
10.043 s, back to 3.695 s on a reply with content and on a timeout.

**A label is NOT the hindsight-optimal sequence of intervals, and no sentence of this file may be
read as saying it is.** One decision is moved and the rest of the run is F; a run of the main cell has
about 161 of these decisions (`labels.txt`: 160.7 a run, 150 to 170) and the three sequences over
all of them are never enumerated. What a
decision maker learned from these labels leans towards is F improved by one step.

**The two rejected constructions, and why** (the user's own two, restated because they are what fixes
the meaning above): a uniform draw after k as well buries the difference between the arms in the
later randomness; F before k as well leaves every state F does not itself reach unlabelled, while the
other learner learns from the uniformly drawn stage's logs and needs labels over that same spread of
states.

### Which decisions get one

**Only resolution-time decisions** (kind `state_poll_reschedule`), and k is the ordinal of that kind
in the roll-in run, counted over every node and pair in call order. The reason is the first half's own
"The resolution-time decision is the one that acts": a send-time interval only arms a provisional
timer that the resolution replaces.

**Send-time decisions are provisional and are never labelled.** Before the switch they are roll-in
draws, from a stream of their own; after it they are F's current interval of that pair, which is what
the rule path returns at a send.

**At the switch**, F's per-pair interval is that pair's most recent resolution-time interval of the
roll-in, or F's own `i_min` (3.695 s) if the roll-in never resolved a poll of that pair; and for the
pair of decision k it is the forced arm. From there F updates exactly as the rule path does, which
means it updates BEFORE reporting at a resolution and reports the current value untouched at a send.

**`labelcheck.py` check 1 is what makes "F after k" mean the real F**: F run through the port from
the FIRST decision on places the same polls, pair by pair, and reports the same resolution-time
interval sequence, as the rule path with `member_rule(F_MEM)`, on two runs of each cell with the
shared lag stream. The reference goes through `port.Engine` with a rule rather than through the base
engine, because F's timeout coordinate is written by evidence/18's `resolve` override and the base
engine's rule leaves the interval alone on a timeout -- a different F. The same `F` class serves the
check and the labelling suffix, so the check is not about a second emulation.

### The score of one simulated run

**In ms: `(sum of the delays of the delivered records + sum over the censored records of the time
from the record's creation to the end of the window) / (delivered + censored) + weight x polls`.**

Records, window and polls are evidence/14's early-record scorer's (`../14/after.py:39-65`): records
created at or before `t0 + 72 s`, delivery measured at the far node (`h4` for origin `h0`, `h0`
otherwise) at or before `t0 + 130 s`, polls counted over `[t0 + 10 s, t0 + 130 s]`.

**A censored record enters the numerator at its lower bound**, `t0 + 130 s` minus its creation,
rather than being dropped. Dropping it would pay an arm for the records it never delivered: an arm
that polls rarely delivers fewer records and would score better on the ones it did deliver.

**One definition, in `score.py`, which `weight.py` levels the arms on as well** (third pass; until
then `weight.py` used `early_score`'s delay and the two files disagreed about what a delay is -- see
"The correction"). `score.score_run` repeats evidence/14's traversal rather than calling
`early_score`, because that function returns the censored COUNT and not the censored TIME.
`labelcheck.py` check 6 requires the four numbers the two have in common -- delivered, censored,
delay sum, polls -- to agree exactly, on three runs, and `weight.py` requires the same on all 40 000
of its own simulations.

**The same run is also scored with the censored records dropped**, and `labels.txt` reports how many
labels that moves, per cell. A decision that ties under one scoring and carries a label under the
other counts as moved.

### The randomness

| stream | keyed by | why |
|---|---|---|
| roll-in resolution-time draws | run, ordered pair | per pair, so one pair's decision count cannot shift another pair's draws; nothing in the key moves with the arm, k or the pass, so every simulation of one run draws the same roll-in |
| roll-in send-time draws | run, ordered pair, separately | so the send draws, which only arm a provisional timer, cannot shift the resolution draws |
| tick lags and jitter, before k | run, ordered pair | the run before k is ONE run for every arm and every pass |
| tick lags and jitter, from k on | run, ordered pair, k, pass | fresh at the probed decision, so the randomness after it is what the passes average; the ARM is not in the key, so the three arms of one pass draw the same values at the same positions |
| each node's first-poll offset | run, node | drawn in `go`, before any decision exists, so no re-key can reach it |

The switch happens inside the decider call for decision k, i.e. inside `resolve`
(`port.Engine.rekey_streams`), and the first draw after the decider returns is the re-arm's tick lag.

**What the alignment is, exactly.** Within one pass the three arms draw the same VALUES at the same
POSITION of each pair's stream. Which EVENT a draw serves can still differ, because one arm may
re-arm its timer where another takes the identity early return and draws nothing at all. That is the
same distinction the first half drew for the lag streams, one level down.

`labelcheck.py` check 2 is the positive statement that goes with it: for sampled k, the three arms'
runs carry the same events at every stamp strictly before decision k's own, and the same events the
roll-in run carries there; and each arm's row at that instant carries the roll-in's state and
resolution and its own forced arm.

### What is written, and where

**Outside the repository**, because a label file is learning data and not evidence:
`/home/lab_shared/semindex-learn/ev22/labels_p0.2.jsonl.gz` and `labels_p0.4.jsonl.gz`, one gzipped
JSON line per labelled decision, rows sorted by (seed, k).

One row carries: the cell `p`, the `seed`, `k`, `node`, `neighbor`, `ts_rel_ms` (the decision's stamp
relative to the run's `t0`) and `ts_ms`, the `state`, `resolved` and `history` the decision was made
in, `rollin_arm` (the roll-in's own arm at that decision), the `passes` used, `mean_score` and
`pass_scores` per arm, the `label`, the `margin` (second best minus best), the `tie` flag,
`label_drop_censored`, and the per-arm per-pass `polls` and `censored` counts.

**No field was added for the third pass's paired t**, because none is needed: `pass_scores` carries
every per-pass score per arm, and `label.paired_t` computes the t from exactly that. A reader of the
file can bin by it without the file growing, which is why "Two breakdowns of the J = 8 agreement" is
something a learner can act on rather than a number only this directory holds.

**`ts_ms` is kept beside `ts_rel_ms`** although the ticket asks only for the relative one: the stamps
inside `resolved` and `history` are the port's own and are absolute, and a reader needs one clock for
all three. The rendered observation text carries no absolute time at all -- see "The observation
text".

**A tie gets no label.** `label` is null when the best and second-best mean scores are exactly equal,
because that is two runs that scored the same and not a weak preference. The ordering tie-break
inside `label.pick` is the arm order, and it only decides which of two tied arms is listed first; it
is never reported as a label. What such a tie actually is was measured -- see "The stability of a
label".

**Reproducibility is a property of the bytes, not of the numbers**: `labelcheck.py` check 3 labels one
run twice and requires identical bytes, gzip header included (`mtime=0`, `sort_keys=True`, rows
sorted by the caller).

**Into the repository** go the scripts, `labels_summary.csv` (one row per run: cell, seed, decisions,
ties, labels per arm, median margin, wall seconds), `labels.txt` (the printed summary) and
`labels.sha256` (the label files' digests).

### What came out

Seeds 1 to 100 of both cells at J = 8, 3706.3 s on 5 workers, under the corrected weight
104.29595912307323 ms per poll.

| | p 0.2 | p 0.4 |
|---|---|---|
| runs | 100 | 100 |
| labelled decisions | 16 071 | 16 148 |
| decisions per run | 160.7 mean, 150 to 170 | 161.5 mean, 146 to 173 |
| label 3.695 s | 3221 (0.2004) | 3390 (0.2099) |
| label 6.0 s | 3675 (0.2287) | 3872 (0.2398) |
| label 10.043 s | 7383 (0.4594) | 7564 (0.4684) |
| tie, no label | 1792 (0.1115) | 1322 (0.0819) |
| of those ties, decisions at or after `t0 + 130 000 ms` | 60 | 78 |
| margin ms, median / max | 104.2960 / 11013.6194 | 130.3699 / 10987.5126 |
| labels that move when the censored records are dropped | 10 (0.0006) | 204 (0.0126) |
| **labels that changed against the superseded weight** | **46 (0.0029)** | **53 (0.0033)** |
| provisional timers fired | 0 | 0 |

**The relabel moved 46 of p 0.2's 16 071 labels and 53 of p 0.4's 16 148** (`specnums.txt`), which
is the count the weight correction cost. The directions are in `specnums.txt` too, seven of them per
cell; the net flow is towards the long arm on both, because the higher weight prices a poll more
dearly.

**That count no longer needs the superseded label files.** A label row carries each pass's score and
that pass's poll count per arm, so the pass's delay comes back as `score - weight x polls` and its
superseded score as `delay + superseded weight x polls`; `specnums.py` relabels every row that way
and compares. The arithmetic is not bit-exact -- the subtraction loses up to about 3e-11 ms, and the
loss differs between two arms because their poll counts differ -- so a margin that was EXACTLY zero
comes back as a few 1e-12 and a tie would be read as a preference. A margin at or below 1e-9 ms is
therefore read back as the tie it was, and `specnums.txt` prints how many reconstructed margins sit
within a thousandfold of that tolerance (0 on both cells). What licenses the tolerance is not a gap
in the margins: the reconstruction was checked row by row against the real superseded files while
they are still in the third pass's scratch, and gave **the same label on every one of the 32 219 rows
and the same seven-way direction breakdown per cell.**

**There is no gap, and the files say so.** `label.py` calls a decision a tie only when the margin is
exactly 0.0, and `specnums.txt` counts **1 labelled row at p 0.2 and 4 at p 0.4 whose margin is at or
below 1e-9 ms** -- a few 1e-12, the last bits of a mean over 8 passes. Those five labels stand on
rounding rather than on a preference, the reconstruction cannot carry them from one weight to the
other, and a learner reading the margin as a strength reads those five as the weakest preferences
there are rather than as the ties they amount to. It is five rows of 32 219 and nothing here is
relabelled for it, but the tie rule is an exact comparison on a mean and that is what it costs.

**The decisions themselves did not move**: the roll-in run does not read the weight, and every row's
`(node, neighbour, ts_ms, rollin_arm)` and its `state`, `resolved` and `history` were compared row by
row against the superseded file and came out identical in all 16 071 and all 16 148. The roll-in's
own arm counts are the same three numbers as before on both cells, which is the same statement from
the other side. **That row-by-row identity is the one comparison of this section that needs the
superseded files, and they are kept nowhere a reader can reach, so nothing here re-derives it**; it
would take a relabel under the superseded weight, 62 minutes, to make it again.

**The longest arm wins 46 % of p 0.2's decisions and 47 % of p 0.4's**, and that is the first number in
this effort that says anything about what a state-reading decision maker should do. Read it against
the weight: the weight was solved to make the 3.695 s and 10.043 s arms level ON THE WHOLE RUN, so a
per-decision preference for the long arm is not the weight leaking in -- it is this construction
saying that, with F taking over afterwards, waiting longer at this one decision is usually the
cheaper move. **It is not evidence that a long-arm policy is better**: a label is one decision moved,
and somewhere between an eighth and a quarter of the labels are pass noise (see "The stability of a
label").

**The median margin over ALL decisions is one poll** at p 0.2 (104.2960 ms, the weight
104.29595912307323 to four decimals; 1797 of the 16 071 margins sit within 1e-9 of it). The consequence
is NOT that half the labels turn on one poll: 1792 of those decisions are ties that carry no label
(60 of them at or after `t0 + 130 000 ms`, where a live agent has stopped)
at all and enter the median at margin 0. **Over the 14 279 decisions that do carry a label the
median margin is 127.4200 ms.** At p 0.4 the median over all decisions is 130.3699 ms and over the
14 826 labelled ones 146.1007 ms. All six of those numbers are `specnums.txt`'s.

**The share "decided by one poll or less" cannot be stated as a single number, because the
comparison lands on the weight itself.** At p 0.2, 5282 of the 14 279 labelled margins are at or
below the weight exactly (0.3699), and 652 of those 5282 sit a last bit BELOW it, within 1e-9
relative. A further **1145 sit a last bit ABOVE** it, so with any tolerance at all the share is
6427 (0.4501) instead. The 652 and the 1145 are the same 1797-margin atom as the sentence above, cut
by exact floating-point `<=`. At p 0.4 the same pair is 4566 (0.3080) exact against 5799 (0.3911)
tolerant, out of an atom of 1780 (547 below, 1233 above). **So read it as an interval: between about
0.37 and 0.45 of p 0.2's labels, and between about 0.31 and 0.39 of p 0.4's, turn on one poll of the
windowed count or less** -- which is a statement about how coarse this score is, not a share the
workload fixes. The same caution applies to the companion "of all decisions, ties included": 0.4402
exact, 0.5114 tolerant, at p 0.2. Every count and share of this paragraph is `specnums.txt`'s, which
prints the two sides of the band and both shares per cell.

**Dropping the censored records instead of entering them at their lower bound moves almost nothing**
at p 0.2 (10 of 16 071) and little at p 0.4 (204 of 16 148), where more records are censored. So the
choice is defensible on its reasoning and barely matters in these cells -- which is a statement about
these cells, not about the choice. **Set beside the weight correction it is the larger of the two at
p 0.4 and the smaller at p 0.2**: dropping the censored records moves 204 labels there against the
weight correction's 53, and 10 here against 46. So at p 0.4 these labels are about four times as
sensitive to how a censored record is scored as they are to the correction of the weight, and at
p 0.2 about a quarter as sensitive. The weight is the term the censoring choice feeds into, which is
why the two are worth reading together, and p 0.4 is the cell where the choice would have to be
argued rather than assumed.

**Which of these numbers a file in this directory prints.** The table's rows and the two digests are
`labels.txt` and `labels.sha256` verbatim; the per-run rows are `labels_summary.csv`. The four things
that no script here printed until the fourth pass -- the median margin over the LABELLED rows, the
share of labels inside one poll, how many margins sit within 1e-9 of the weight, and the
old-against-new moved counts -- are now `specnums.py`'s, which reads the two label files and writes
`specnums.txt` in about 5 seconds. The one number of this section left outside a generated file is
the row-by-row identity against the superseded files, named in the paragraph above. `specnums.py`
refuses to count anything if either label file's sha256 is not the one `labels.sha256` records, so
the numbers in `specnums.txt` are tied to the two files the digests name.

**The two label files' digests** are in `labels.sha256`; they are 4.3 MB and 4.5 MB.

## Seeds

**Seeds 121 to 160 appear nowhere here.** They are the measuring seeds of this effort and the
learning side may not touch them. Every script of this directory that takes seeds passes them through
`port.check_seeds`, which refuses any seed above 100 with an error. **The library doors refuse them
too** (fifth pass): `label.run_dict` checks its own argument, `label.label_run` reaches it for any
(p, seed) a caller drives directly, and `port.Engine` checks `run["seed"]` when the run dict carries
one, which is the constructor every run of this directory goes through. Until then the guard was on
the command line alone, and an importer of `label` or `port` could reach seed 121 without passing
it. (`weight.py` loads
`../21/predict.py`, which loads `../20/predict.py`, whose module body defines that seed range as a
constant; no script here reads it.)

## The stability of a label (second half, with the third pass's two breakdowns)

**It carries no pass line**, by the ticket's own wording: Acceptance 6 asks for the size of the
residue to be seen as a number for the first time, not for a threshold on it. The one decision it
makes is which J the full labelling runs with, and that rule was written before any of these numbers
existed.

### What was measured

On seeds 1 to 10 of each cell, every resolution-time decision was labelled twice, from two DISJOINT
sets of 8 passes (A = passes 0 to 7, B = passes 8 to 15), and the two labels recomputed from the
first J passes of each set, for J in 1, 2, 4 and 8. Three populations, because one share would hide
where the disagreement sits, and a fourth column that restricts the third to the decisions the other
two columns are about:

- **all** decisions: a decision that ties on one side and carries a label on the other counts as a
  disagreement, so this share can sit below the non-tie one (it does at J = 1 in both cells);
- **non-tie**: both sides carry a label. This is the population the J rule reads;
- **the larger-margin half**: the decisions of that cell whose set-A margin at that J is in the upper
  half, ranked descending with (seed, k) as the tie-break so the split is reproducible. The half is
  ranked over ALL decisions, which is how the measurement was defined, so it is over a **wider
  population than the non-tie column beside it**: a member of the half that ties on set B only
  carries no label there and is a forced disagreement. A fourth column therefore gives the same half
  restricted to the members that carry a label on both sides, so that the two can be read against
  each other; its own `n` is how many members the restriction drops.

| cell | J | all | non-tie | n | larger-margin half | n | that half, both sides labelled | n |
|---|---|---|---|---|---|---|---|---|
| p 0.2 | 1 | 0.6683 | 0.6847 | 1183 | 0.6066 | 793 | 0.6312 | 762 |
| p 0.2 | 2 | 0.7055 | 0.7016 | 1287 | 0.6431 | 793 | 0.6448 | 791 |
| p 0.2 | 4 | 0.7377 | 0.7296 | 1335 | 0.6948 | 793 | 0.6948 | 793 |
| p 0.2 | 8 | 0.7623 | 0.7577 | 1366 | 0.7276 | 793 | 0.7276 | 793 |
| p 0.4 | 1 | 0.6328 | 0.6717 | 1252 | 0.5845 | 811 | 0.6156 | 770 |
| p 0.4 | 2 | 0.6648 | 0.6778 | 1372 | 0.6128 | 811 | 0.6182 | 804 |
| p 0.4 | 4 | 0.7135 | 0.7123 | 1439 | 0.6535 | 811 | 0.6543 | 810 |
| p 0.4 | 8 | 0.7585 | 0.7522 | 1473 | 0.7411 | 811 | 0.7411 | 811 |

1586 decisions at p 0.2 and 1623 at p 0.4, over 10 runs each; 774.8 s on 5 workers, which is
`stability.txt`'s own last line after the fifth pass reran it. **These are the
third pass's numbers, under the corrected weight.** The decisions themselves are the same ones: the
roll-in run does not depend on the weight, so the 1586 and the 1623 are unchanged. Against the
second half's table, measured under the superseded weight, none of the 32 shares moved by more than
0.0138 and none by more than 0.0038 at J = 8, and no population size by more than 6 decisions; both
of the two largest moves are in p 0.4's J = 1 row (the half 0.5721 to 0.5845, that half restricted
0.6018 to 0.6156), where the ranking is one pass's spread and the weight reorders it most. **No
conclusion in this section changed sign or
kind.**

**The two populations differ only at the small J**, which the last two `n` columns say: the half holds
31 one-sided ties at J = 1 and 2 at J = 2 on p 0.2, and 41, 7 and 1 at J = 1, 2 and 4 on p 0.4;
none at J = 8 in either cell. At J = 1 the margin is one pass's spread, so a decision can be high in
set A's ranking and tied in set B; by J = 8 the ranking's upper half carries a label on both sides
throughout.

### The J the full labelling runs with

**The rule, fixed before the numbers**: the smallest J in (1, 2, 4, 8) whose agreement over the
non-tie decisions of the main cell is at least 0.90, and 8 if none reaches it.

**J = 8, and not because it reached the bar.** No J reached it. At J = 8 on the main cell the
agreement is **0.7623 over all decisions, 0.7577 over the 1366 non-tie decisions, 0.7276 over the
larger-margin half** -- and 0.7276 over that half restricted to both-labelled decisions too, because
at J = 8 every member of the half carries a label on both sides. On p 0.4, which the rule does not
read, 0.7585 / 0.7522 / 0.7411 / 0.7411.

**Set A IS the full labelling**, not a proxy for it: a pass's streams are keyed by its index alone,
so pass j's score is the same number whether it was asked for beside 15 others or alone (measured:
pass 0's three per-arm scores on all 158 decisions of p 0.2 seed 1 are identical in a one-pass run
and in a two-pass run, while pass 0 and pass 8 differ). The labels set A produces at J = 8 are
therefore exactly the labels on disk for seeds 1 to 10, so the table's numbers are statements about
those labels and not about a parallel construction.

**So 0.2423 of the decisions that got a label from both pass sets got a DIFFERENT label.** That is
the number Acceptance 6 asked to see, and it is a bad number. **It is an upper bound on the share of
the labels on disk that are pass noise, not that share itself**: a disagreement needs only ONE of the
two sides to be wrong, so one side's own error rate is between half of 0.2423 and all of it. Under the
crudest model that fits -- the two sides erring independently at a rate q and an erring side landing
on either of the two other arms with equal probability, so that the disagreement is
2q - 1.5q^2 = 0.2423 -- q is about 0.135. That model is not measured here and nothing downstream may
lean on it; what is measured is the bound. It is stated here rather than buried because the next
ticket's learner will be fitted to these labels and an eighth to a quarter of them are noise. **The
breakdowns below say the noise is not spread evenly**, which is the one thing a learner can act on.

**What the shape of the table says.** Agreement rises monotonically with J at both cells and in all
three populations, so the disagreement is variance that averaging reduces, not a disagreement between
two different constructions. It rises slowly: eight times the passes bought 0.073 on the main cell's
non-ties (0.6847 at J = 1, 0.7577 at J = 8), so reaching 0.90 by passes alone would cost far more
than 8, and the full labelling already takes about an hour on 5 workers at J = 8.

**The larger-margin half agrees LESS than the whole, at every J and both cells.** That is the
opposite of what a wide margin is supposed to mean, and the conclusion a reader would otherwise
reach for first -- keep the wide-margin labels and drop the rest -- does not follow from this
table. **At J = 8 the comparison is clean**, because the half and the non-tie column are then over
the same decisions; at J = 1 part of the gap is the population and not the margin -- 0.6847 against
0.6066 is 0.0781, of which 0.0246 (about a third) goes away when the half's 31 one-sided ties are
removed, leaving 0.0535 that is the margin.

**Two readings of it, and the second half's measurement separated neither.** One: at these pass
counts a wide margin is not a confident decision, it is a decision whose per-pass scores are spread
out. Two: the half is ranked by SET A's own margin, so it selects on set A's noise as much as on any
true separation, which is a winner's curse and would produce exactly this shape on its own. Ranking
by the combined 16-pass margin would tell the two apart and is still **not tried**. What the third
pass's two breakdowns below DO settle is that reading one is the live one: a statistic measured on
the very same set A, the paired t, orders the labels almost cleanly, so the trouble with the margin
is not that it is set A's number -- it is that it is a difference without its spread. The negative
statement about the margin itself stands unchanged: **the margin as this file computes it is not a
usable confidence.**

### Two breakdowns of the J = 8 agreement (third pass)

Since the margin is not a confidence, the question is whether anything else about a decision says
whether its label survives a change of passes. Two breakdowns, **with their bins fixed before the
numbers were seen**, both over ALL decisions of a cell (with the one exception the t bullet names
and the script counts) and both with agreement defined as "set A's label equals set B's label", so
two ties agree and a one-sided tie disagrees -- the `all` column's own rule:

- **set A's own paired t over its 8 passes.** The two best arms by mean score, `d_j` the per-pass gap
  between them, `t = mean(d) / (sd(d) / sqrt(8))` with the sample sd. This is the margin divided by
  its own spread, which is exactly what the margin alone cannot distinguish: a wide gap with wildly
  spread passes and a narrow one with eight near-identical passes rank the same by margin and very
  differently here. Bins `[0,1)`, `[1,2)`, `[2,3)`, `[3,inf)`, plus TWO buckets for `sd` exactly 0,
  which has no t at all: `sd = 0, labelled` and `sd = 0, tie on A`, because the same zero arises from
  opposite situations and the fifth pass had to take them apart (below). A third bucket, printed only
  when it holds something, takes the `sd` None that `label.paired_t` returns below 2 passes: at J = 1
  there is one sample, so there is no t for a reason that has nothing to do with the spread, and the
  J rule can choose J = 1 on other data. `label.paired_t` is the one definition and `stability.py`
  calls it.
  **The t table does not quite cover every decision at p 0.4, and says so**: `stability.txt` prints
  how many decisions the table covers, and at p 0.4 that is 1621 of 1623. The two it leaves out are
  set-A ties whose `fmean(a_j - b_j)` rounds a hair below zero (t = -1.46e-14 and -1.88e-14) and so
  match no bin; their t is printed beside the count. The bins are not widened and no t is clamped,
  because the bins were fixed before the numbers. At p 0.2 the table covers 1586 of 1586;
- **the decision's position in its run**, as equal-count quartiles of `ts_rel_ms`, with the median
  margin printed beside each quartile. A decision near the end of the window has fewer polls left in
  which the arms can separate and less randomness left to average, and the table above cannot say
  which of the two dominates. The quartile rows also carry the set-A ties, the rows labelled on both
  sides with their own agreement, and the decisions at or after the instant a live agent stops.

**Both separate, and hard.** By set A's paired t, at J = 8:

| t bin | p 0.2, n | ties on A | labelled both | agreement | over those | p 0.4, n | ties on A | labelled both | agreement | over those |
|---|---|---|---|---|---|---|---|---|---|---|
| [0,1) | 413 | 5 | 405 | 0.4625 | 0.4716 | 466 | 4 | 460 | 0.4700 | 0.4761 |
| [1,2) | 277 | 0 | 259 | 0.6426 | 0.6873 | 294 | 0 | 283 | 0.6667 | 0.6926 |
| [2,3) | 167 | 0 | 164 | 0.8383 | 0.8537 | 144 | 0 | 142 | 0.8750 | 0.8873 |
| [3,inf) | 472 | 0 | 472 | 0.9746 | 0.9746 | 535 | 0 | 535 | 0.9607 | 0.9607 |
| sd = 0, labelled | 66 | 0 | 66 | 1.0000 | 1.0000 | 53 | 0 | 53 | 1.0000 | 1.0000 |
| sd = 0, tie on A | 191 | 191 | 0 | 0.9110 | -- | 129 | 129 | 0 | 0.9457 | -- |

**The `sd = 0` bucket is two buckets and the fifth pass split it.** `label.paired_t` returns no t
both when every pass separated the two arms by the same NON-ZERO amount and when every pass scored
them equally, and the second is a decision that TIES on set A: `d_j = 0` throughout, so `sd` is
exactly 0 and the row carries no label at all. 191 of p 0.2's 257 and 129 of p 0.4's 182 were that,
under a gloss that described the other 66 and 53. Their "agreement" is `None == None` -- two
labellings agreeing to leave the row unlabelled -- which a learner cannot train on either side of.
So every bucket now prints its set-A ties, its rows labelled on BOTH sides, and the agreement over
those beside the agreement over all of them; the last column is the one a learner reads.

By the decision's position in its run, at J = 8:

| quartile of `ts_rel_ms` | p 0.2 range | n | ties on A | agreement | over the labelled | median margin ms | past the stop | p 0.4 n | ties on A | agreement | over the labelled | p 0.4 median margin ms | past the stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | 391 to 30 251 | 396 | 0 | 0.5101 | 0.5101 | 282.2121 | 0 | 405 | 0 | 0.5012 | 0.5012 | 289.7590 | 0 |
| Q2 | 30 296 to 63 762 | 397 | 0 | 0.7330 | 0.7330 | 213.5172 | 0 | 406 | 0 | 0.6897 | 0.6914 | 241.5358 | 0 |
| Q3 | 63 768 to 97 106 | 396 | 36 | 0.8460 | 0.9130 | 104.2960 | 0 | 406 | 10 | 0.8744 | 0.9046 | 104.2960 | 0 |
| Q4 | 97 189 to 131 907 | 397 | 160 | 0.9597 | 0.9956 | 52.1480 | 4 | 406 | 125 | 0.9680 | 0.9964 | 104.2960 | 9 |

**Every set-A tie of a cell sits in the last two quartiles** (36 + 160 = 196 at p 0.2, 10 + 125 =
135 at p 0.4), so the quartile columns move in opposite directions when the unlabelled rows come
out: Q4's agreement over the rows labelled on both sides is **0.9956 of 228 rows at p 0.2 and 0.9964
of 275 at p 0.4**, HIGHER than the 0.9597 and 0.9680 over everything, while Q1's two columns are the
same number because Q1 holds no tie at all. The 160 and 125 set-A ties of Q4 therefore agree LESS
than its labelled rows, not more. **The last column is the decisions at or after `t0 + 130 000 ms`,
where a live agent has already stopped and the replay keeps resolving polls that were in flight**: 4
of p 0.2's Q4 and 9 of p 0.4's, all of them set-A ties carrying no label, the latest at 131 907 and
131 378 ms. `stability.txt` prints that count per quartile and names it under each table; over all
100 seeds `specnums.txt` counts 60 at p 0.2 and 78 at p 0.4. Nothing is relabelled for them, because
every one is an exact tie and a labelled row therefore never sits there.

**What the paired t says.** Quoting `stability.txt`'s own spread line for each cell, which takes the
range over the four t bins AND both `sd = 0` buckets: agreement runs from **0.4625 to 1.0000 at
p 0.2, a spread of 0.5375**, and from **0.4700 to 1.0000 at p 0.4, a spread of 0.5300**; over the
rows labelled on both sides the same lines read **0.4716 to 1.0000 (0.5284)** and **0.4761 to 1.0000
(0.5239)**. The top of the range is now the `sd = 0, labelled` bucket in both cells, where all 66 and
all 53 decisions kept their label (1.0000): every pass separating the two arms by the same non-zero
amount is as firm as this construction gets. The `[3,inf)` bin is next (0.9746 and 0.9607). Before
the split the top of the p 0.4 range was the merged bucket's 0.9615, where 129 rows were
`None == None` against the 53 that carried a label, so most of what held the top of that
range was two nulls matching; the reading is gone with the merged bucket. So a statistic
computed from set A's own 8 passes DOES
order the labels by whether they survive a change of passes -- the margin's failure is not that
"anything measured on set A selects on set A's noise", which is the strong form of the winner's-curse
reading, because the t is measured on set A too. **The margin's failure is that it is a difference
without its spread.** The t divides by that spread and orders the decisions almost cleanly: above
t = 3 about nineteen in twenty labels survive, below t = 1 fewer than one in two do.

**What that does NOT settle.** The weak form of the winner's curse applies to the t as well -- it is
still set A's own number, so the top bin is partly selected on set A's luck -- and the comparison
that would price that, ranking by the combined 16 passes, is still **not tried**. And the bins are
agreement rates on seeds 1 to 10, not a calibrated probability for any one label.

Three of the eight quartile medians are **104.2960 ms, which is the weight to four decimals** -- one
poll of the windowed count. That is the same fact "Why the labels are this noisy" states from the
other end: over much of a run the three arms are separated by one poll and nothing else, so the
margin has one poll's worth of resolution to work with.

**What the position says.** Agreement rises from 0.5101 in the first quartile of a run to 0.9597 in
the last at p 0.2, and 0.5012 to 0.9680 at p 0.4, while the **median margin FALLS** over the same
quartiles, from 282.2 ms to 52.1 ms. The two move in opposite directions, which is the second and
independent demonstration that the margin is not a confidence: the decisions with the widest margins
are the early ones, and the early ones are the least reproducible. The mechanism is the one the
construction implies -- an early decision leaves 120 s of run in which the randomness after it can
diverge, and a late one leaves almost none -- so the quartile is a proxy for how much randomness the
8 passes had to average, not a property of the decision itself. **The rise is not an artefact of the
ties collecting at the end**: every set-A tie of a cell is in Q3 and Q4, and taking them out raises
the last quartile instead of lowering it, to 0.9956 over 228 labelled rows at p 0.2 and 0.9964 over
275 at p 0.4. What the ties do change is the POPULATION: Q4 of p 0.2 hands a learner 228 usable rows
and not 397.

**What a learner may take from this, and what it may not.** It may weight or filter by the t, which
is on the label row already (`pass_scores` per arm; `label.paired_t` computes it). The selection is
`t >= 3 or (sd = 0 and labelled)`, and it keeps **538 of p 0.2's 1586 decisions -- 0.3392 of all of
them and 0.3871 of the 1390 that carry a set-A label -- at a 0.9777 agreement rate**, and 588 of
p 0.4's 1623 (0.3623 of all, 0.3952 of its 1488 labelled) at 0.9643, against 0.7623 and 0.7585 over
everything. All 538 and all 588 carry a label on both sides, so those two rates are the labelled
rates as well. (`stability.py` prints the selection itself, one line per cell, so every number here
is read off `stability.txt` and none is this file's own arithmetic.)

**The bar sits on a mass point of the t, so neither 538 nor 5696 is a count another implementation
reaches.** 39 rows of each cell have a paired t within 1e-06 of 3, none of them further off than
1e-13 and the nearest row off that clump at least 776 times the tolerance away, and `>=` keeps 19 of p 0.2's and 25 of p 0.4's: the whole-file selection holds 5696 rows where
anything from 5677 to 5716 is the same rule, and the ten measured seeds hold 538 where 536 to 541
is, 588 against 586 to 590 at p 0.4 (`specnums.txt`). That is a weakness of the bar as this step
drew it, and the 0.9777 and 0.9643 above were measured with those rows on the side this directory's
summation order put them. 1246 rows of p 0.2 and 1263 of p 0.4 tie the runner-up with the third arm
in mean as well, where the arm the t is taken against is the order of `ARMS` and nothing in the
data, and 2 and 5 of them cross the selection when the other arm is taken.

**The earlier figure for this selection, `t >= 3 or sd = 0` keeping 0.4596 at 0.9602, counted the
unlabelled set-A ties as kept.** 191 of those 729 decisions carry no label on disk, so a supervised
learner receives 538 rows and not 729, and 0.4596 is not a retention it can read; `stability.txt`
prints the wider line under the selection with that stated. The direction of the error is worth
naming: the retention was overstated by those 191 decisions, 729 of the cell's 1586 claimed
against the 538 a learner receives, while the agreement was UNDERSTATED, because the ties
agree less than the labelled rows they were pooled with.

So the choice a learner faces is not "8 passes or 64" but "all the labels at about three in four, or
a third of the decisions -- two fifths of the labelled ones -- at about forty-nine in fifty". It may
**not** read these rates as the probability that a label is RIGHT: both sides are the same
construction, so they agree about the construction and not about the testbed, and that limit is
unchanged by any breakdown. And the agreement is measured on seeds 1 to 10 only; what the same
selection keeps over all 100 seeds of each file is in `specnums.txt`, as a retention with no
agreement beside it, because no second labelling of those seeds exists.

### Why the labels are this noisy, as far as this step can say

**The arms are typically one poll apart.** On p 0.2 seed 1 at J = 1 the margin's median is
104.2960 ms, which is the weight 104.29595912307323 to four decimals -- one poll of the windowed
count. So the delay term usually separates the arms by less than a single poll does, and a run where
the moved polls happened to deliver nothing new has the SAME delay sum under two different arms, to
the last bit of a double; only the poll count is then left, and when that coincides the score ties
exactly.

**That is where the exact ties come from, as far as it was checked.** On the full labelling at
J = 8, **0.1115 of p 0.2's labels and 0.0819 of p 0.4's are null**, and 60 and 78 of those nulls are
the decisions at or after `t0 + 130 000 ms` that a live agent never reaches (`specnums.txt`), so both
tie shares carry them; the 0.1387 in the stability
measurement is a different quantity -- the share of seeds 1 to 10's decisions that tie on at least
ONE of the two pass sets, which is necessarily larger. On p 0.2 seed 1 at J = 1 there are 28 ties
of 158 decisions. In 23 of them two arms tie and the third differs, and the three event streams of
one such decision were compared: 2259, 2256, 2260 events, so the tie is two DIFFERENT runs
scoring equally and not two identical runs. In the other 5 all three arms score identically, and
those of them that are not near the end of the window were compared event by event as well
(at 89.6 s 2330/2330/2329; at 102.4 s 2360/2359/2358): the runs differ there too, and what makes the
scores equal is that the WINDOWED POLL COUNT comes out the same under all three arms (133 at the
first, 144 at the second) while the delay sums are identical to the last bit -- which is to say none
of those moved sends changed any record's delivery stamp and none of them crossed the window's
edge. The remaining 3 sit after 126 s of the 130 s window, where no arm fits another poll
at all.

**So an exact tie is two or three genuinely different runs, never a forced arm with no effect.** That
matters, because a tie that came from the forced arm having no effect would be a bug in the
construction rather than a property of the score.

### What this cannot show

That the labels are right at J = 8. Two labellings that agree agree about the construction, not about
the testbed. And the agreement was measured on seeds 1 to 10; the full labelling runs 1 to 100 and is
not re-checked.

## The observation text (second half, with the third pass's polled peer)

`observe.py` holds the one road from the observable state to something a learner reads, and both
learners call the first function:

    observation(state, resolved, history, now, neighbor) -> dict   the canonical record
    render(record, legend=True)                          -> str    what the model reads

plus `from_logged_rows`, an adapter that builds the same five inputs from the decision rows of one
node's testbed `events.jsonl`, so that the two functions run unchanged on the `s11a` logs.

### Every choice, with its reason

- **No absolute time.** Every instant becomes an age in seconds before `now`, to one decimal. An
  absolute stamp is an epoch millisecond: it differs between two runs of the same experiment, so a
  policy keyed on one could not transfer, and a language model reading one would spend tokens on 13
  digits that carry nothing. One decimal because a tick is 100 ms, which is the resolution of the
  thing being measured.
- **No node names.** Origins and advertisers get POSITIONS inside one record (`o1..`, `p1..`), never
  names. **The order is stated inside the rendered text**, because an order the reader cannot see is
  a name by another route. Origins are ordered by `missing` then `hi`, both highest first; peers by
  their highest `notified_max` then by how many origins they advertise. The same `p` number is the
  same peer on every line of one record, which is why the peer ranking is computed once per record
  and not once per origin.
- **The tie-break leaks the name ORDER, and that is stated rather than denied.** Ties keep the
  state's own order. For origins that IS the origin-name order, since the source sorts them. For
  advertisers it is first-appearance order over the state -- origin order first, then the
  advertiser-name order inside an origin -- which is `peer_order`'s `seen.index(n)` tie-break and
  is not pure name order. Either way it is the only name-derived ORDER left in the record.
  `polled_peer` comes from a name as well, and **it is not independent of that name either**: over
  the 18 448 labelled records that mark a position and name at least two peers, the position is the
  polled pair's rank in plain advertiser-NAME order in 13 992 of them (0.758, against 0.5 for a coin
  on a two-peer record), and on the logged run in 66 of 81. That is the two orders coinciding on
  this workload, not a name being recoverable: no node name is in the record and the position is the
  whole point of the field. The earlier wording, "it carries nothing about that name", claimed an
  independence nothing here measures. `observe.txt` prints the share.
  **Both tie-breaks are in the legend**, the peer one
  as "ties in the order the state first names them": a reader who cannot reconstruct the order of
  two peers that tie on the two printed keys is a reader who has to read the positions as names,
  which is the one thing this record is built to prevent. On these runs the first-appearance order
  and the advertiser-name order coincide in every record measured (0 of the 148 logged states and
  0 of the 32 219 labelled ones differ, and the ranking differs in none), because every neighbour
  here advertises every origin; the legend states the first-appearance rule, which is the one that
  holds in general.
- **No node name and no ordinal from `where`; the polled neighbour yes** (third pass). `where`'s
  five fields are not one kind -- see "`where`, and who may read which half of it". The node NAME
  and the `ordinal` never enter the record, because the live loop cannot hand a policy either in a
  form that transfers. `neighbor` does enter it, as a POSITION and never as a name: see "The polled
  neighbour".
- **Field order**: the resolution first, because it is the one fact that is new since this pair's
  previous decision and the one every hand-written rule branches on; then the pair's own scalars
  (`in_flight`, `consecutive_poll_timeouts`, `last_poll_status`, the fetch-timeout age); then the
  counts and the per-origin rows, which are the largest block and the one a reader can stop reading;
  then the exchanges.
- **The 8 exchanges are written newest first**, one line each: age, outcome, whether the reply was
  empty, the interval decided, the interval actually waited, and whether the clamp moved it. Newest
  first because the most recent exchange is the one a rule reacts to, and a fixed-length vector built
  from this record then has the live exchange at index 0 whatever the history's length. Both the
  decided and the waited interval are kept: the first is the decision, the second is what the node
  actually waited, and the two differ by a jitter the decider does not control.
- **Rounding**: ages to one decimal, intervals and waited times to three. Three because the six
  intervals a labelling run can produce are 3.695, 5.5425, 6.0, 8.31375, 9.0 and 10.043, and three
  decimals keeps each one recognisable as itself.
- **Length bounded by construction**: at most 8 origins (the source's own cap), at most 4 advertisers
  shown per origin (with the omitted count stated), at most 8 exchanges (the port's own cap). The
  number of FIELDS is a constant; the number of DIGITS is not, and a count in this workload is at
  most three digits.
- **`statecheck.py`'s `walk` is not reused** for the structure comparison, although it does the same
  job one level shallower: it recurses into dicts only, because the state it compares holds no list,
  and this record's two largest blocks are lists. `observe.shape` is six lines, and the alternative
  -- a list branch in `walk` -- would have made the state checks compare something other than what
  they were written to compare, and would have put a check's imports (`Tracker`, `_PollReplies`) on
  the import path of a module both learners load.

### The polled neighbour (third pass)

`state` lists every advertiser of every origin and does not say which of them this decision is
about. The second half concluded that the record therefore could not mark it, and wrote
"the record does not say which peer this decision polls" into the legend. **That was wrong, and it
withheld real-node context**: the source decides the interval per neighbour
(`_choose_poll_interval` at `src/semindex/engine.py:710`) and writes that neighbour into the
`action` of the reschedule row it logs (`:692-699`), so a node knows which peer the decision is
about. The cost was concrete: `h1`, `h2` and `h3` each have
two neighbours, and a record that cannot mark the polled one cannot tell its own peer's `told` value
from the other peer's.

**So `observation` takes the polled neighbour as a fifth input**, and the record carries
`polled_peer`, its POSITION in the record's own peer ranking. The name is what the position is
looked up from and never reaches the record. `render` prints it on the origins line -- "this
decision polls p1" -- and the legend's last clause now says the polled peer is named there.

`polled_peer` is **null when that neighbour advertises none of the shown origins**, which is a real
state and not a missing input. **THREE disjoint states produce it, and the split this file gave
before the fifth pass tested the wrong condition**: it counted `peers_known == 0` as "the state names
no origin at all", and a state that names an origin NO peer has told it about satisfies that
predicate while naming an origin. The three causes, over the 6077 null records of the two label
files, are: **4838 name no origin at all**, so there is no advertiser map for the pair to be absent
from; **598 name an origin that no peer has told them about**, so the advertiser map is empty and
`render` writes "no peer told it"; and **641 have origins peers did tell them about whose advertisers
do not include the polled pair**, which is the case the second half gave as the only one. The 598 are
the records the old predicate misfiled, and they turn the logged run's 35-and-1 into **30, 5 and 1**.
`observe.txt` prints all three counts for both populations.

**In those 6077 records -- 0.189 of the 32 219 labelled ones, and 36 of the 148 logged ones -- the
record still cannot tell a learner which peer the decision is about.** That is the gap the third
pass set out to close, closed for about four fifths of the records and left open for the rest. It is
not a branch being healthy; it is the attribution being absent, in a fifth of the records a learner
will see.

What the record still does not carry is any node NAME and the decider `ordinal`. Those are the two
fields of `where` a live loop cannot hand a policy in a transferable form, and that is a property of
the live loop, not a gap in this function.

### The branches no run exercises

Four branches of these two functions have never been taken, counted over the 32 219 labelled
decisions of both label files and over the 148 logged rows. **Every count of this section is
`specnums.txt`'s**, per cell and totalled, except the logged-run ones, which are `observe.txt`'s:

- **`resolved` None** -- `observation` accepts it and `render` then writes "state poll" with no
  resolution line, because the port hands a decider None at a send. **Nothing in this step renders a
  send-time decision**: only resolution-time decisions are labelled, and the adapter reads only
  `state_poll_reschedule` rows. 0 records of 32 219. It is kept rather than removed because the
  signature is the port's own `obs`, which does carry None at a send, and a function that raised
  there would be a function a learner could not call on every decision;
- **the "(clamped)" suffix** on an exchange -- 0 of the 200 152 exchanges rendered, because no
  applied interval of these runs hit the clamp;
- **the "(+N more)" advertiser count** -- 0, and no state in either file names more than 2
  advertisers on one origin against the cap of four;
- **the origin cap and its " (more left out)"** -- 0, and no state holds more than 2 origins
  against the cap of eight, which is the same fact `truncated` being always False says.

So three of the four are the length bound not biting on this topology, and the fourth is the send-time
record. **Three branches a reader might expect to be dead with them are not**: "no peer told it"
fires 651 times (239 at p 0.2, 412 at p 0.4), the `empty is None` "-" of a timed-out
exchange fires 54 327 times, and the third pass's own `polled_peer` null branch -- "this decision
polls a peer none of these origins names" -- fires 6077 times over the label files
(26 142 of 32 219 records name a peer) and 36 times over the 148 logged rows. **That third one is a
live branch and a MISSING ATTRIBUTION at the same time**, and it should not be read off this list as
a good result: in those 6077 records the marker the third pass added says nothing about which peer
the decision is about. Their three causes are 4838 naming no origin at all, 598 naming an origin no
peer told them about, and 641 whose advertisers do not include the pair.

### What was checked

On the logged `s11a` p 0.2 seed 1 run, all five nodes: every one of the 148 `state_poll_reschedule`
rows renders, and **every record is held to the declared schema, one record at a time** -- the key
set AND the field order of the record and of each of its three nested levels, with each field's
declared value types -- over the 148 logged records, the 158 of a replay roll-in run of the same run,
and all 32 219 of the label files. **A second PASS line holds the same records to the leakage rule**:
no node name, no 10-to-13-digit absolute stamp and no negative age, in the record or in its own
rendered text.

Until the fifth pass that first line compared the UNIONED dotted paths of the two sources, which one
record missing a key could not fail as long as another record supplied the path -- the union masking
`statecheck.py` check 1b was written to end, left standing in the one place that still had it -- and
the leakage the module docstring promises was checked nowhere. The union is still printed, as a
reported fact and not as a gate: **30 paths**, one more than the second half's 29 since `polled_peer`
was added, the same set from both sources. The paths and their value types are
printed in `observe.txt`. The types are printed and not compared across the two sources, because a
type set can legitimately narrow on one run (`last_poll_status` is a `str` at every resolution of
both, `fetch_timeout_age_s` is null throughout both).

### Characters per item

| population | with the legend | without it |
|---|---|---|
| the logged `s11a` run, 148 decisions | median 1171, max 1220 | median 797, max 846 |
| the labels, p 0.2, 16 071 decisions | median 1170, max 1237 | median 796, max 863 |
| the labels, p 0.4, 16 148 decisions | median 1162, max 1232 | median 788, max 858 |
| the labels, both cells, 32 219 decisions | median 1166, max 1237 | |

Every row is the current `observe.txt`, which the script rewrites whole, so the table and the file
move together. **The third pass added about 27 characters an item**: 24 of them are the origins
line's new "this decision polls p1" clause and 3 are the legend's corrected last sentence, which is
why both columns moved and the without-legend column moved by the larger share.

**About 1170 characters an item with the legend and about 795 without.** The maximum is **about 6 %
above the median with the legend** (1237 against 1166 over both cells) and **about 9 % above it
without** (863 against 796 at p 0.2, 858 against 788 at p 0.4), the legend being a constant that
dilutes the variable part. Both margins are small, which is what "bounded by construction" buys;
the without-legend column is the one a caller that hoists the legend will see, and it is the wider
of the two. That is the number ticket 10 budgets
its prompt against: the legend line is 373 characters of the 1170, 374 with the newline that joins it
to the record, and a caller that renders many records in one prompt can hoist it by passing
`legend=False` -- which is why the table gives both columns.

## What each check proves

| script | what it states | what it does not |
|---|---|---|
| `gate.py` check 1 (26 PASS lines over the four checks, was 23) | with no decider the edited replay is `fd6ce55`'s, event dict for event dict, on a fixed choices arm, a policy stage, a formula rule on a donor run, three choices with the re-decision off and with it on, the logged `s11a` stage, and -- third pass -- the stage-1 fixed branch with policy, choices and rule all None, in Mode A and Mode B | nothing about a run WITH a decider |
| `gate.py` check 2 | the main cell (p 0.2, `w0`, donor `s6b10p043`, seeds 1 to 20, 12 rules) reproduces every printed column of `../18/bound_cells.csv` and `../15/onoff_cells.csv`, through evidence/18's own job and row builders; a csv column the built row does not carry is a failure, not a column skipped | it is the replay against its own archived output, not against the testbed. On a difference it prints the simulated and archived string of each differing column, as the precedent does; it derives no number of its own, so there is nothing further to suppress |
| `gate.py` check 2b | `port.Engine` with `pair_lag_seed=None` emits the same event dicts as `../18`'s `Engine`, on two stages in both modes, so the subclass is inert with the streams off | nothing about a run with the streams ON; that is `lagcheck.py`'s |
| `gate.py` check 3 | the port refuses a decider without choices, without the re-decision, beside a policy, beside a rule in the run dict, beside a rule passed as `_sim_run(..., rule=)` (third pass: the keyword is the door that silences `policy` and `choices` without silencing `decider`), and refuses an interval outside the choice set | beyond the ticket's Acceptance 1. The refusal is `interval not in choices`, so a numerically equal value of another type is accepted; see "What this step cannot show" |
| `gate.py` check 4 | `git status --porcelain` AND `git diff HEAD` are unchanged across the run, so no committed csv was overwritten (third pass: the diff was added because a tracked file that was ALREADY dirty keeps its one porcelain line whatever is written over it, so porcelain alone could not see that case -- measured: appending a byte to the dirty `weight.txt` moves the diff and leaves porcelain identical) | |
| `statecheck.py` (28 PASS lines, was 18) | the key sets of the port's state, of a real `Tracker` plus `_PollReplies`, and of the logged `state_poll` and `state_poll_reschedule` rows of one `s11a` run are equal, nested, with one extra or missing key anywhere a failure, the port's two decision kinds compared separately; **and (third pass, check 1b) every INDIVIDUAL sample held to the key set of its level** -- six state keys, exactly `hi`/`missing`/`advertised` per origin row, exactly `hi`/`notified_max` per advertiser row -- with every value type inside a set declared from the source; the history element's keys are the logged action's less `neighbor` plus `ts_ms`; `obs` and `obs["where"]` carry exactly their declared key sets on every call (third pass); the emitted row carries the decider's own input; the state the BUILDER produces is unchanged when every `_Origin.hist` is poisoned; and the `ast` of BOTH `Node.decision_state` and `_Engine.ask_decider` names none of `nodes`, `timeline`, `run`, `heap`, `hist` (third pass: `ask_decider` is the function that assembles `obs` and holds the engine, and it had never been inspected) | **SHAPE ONLY, except check 4b.** Nothing about the VALUES of the six state fields agreeing: a wrong number in any of them passes every line of it, and `valuecheck.py` is where four of the six are compared. Nothing about `history` mirroring a real field, because the source has no such field. And the poison check is about the builder, not the content: the advertised `hi` IS hist-derived, at reply time |
| `lagcheck.py` check 1 | with the streams on, one changed resolution of one pair leaves every other pair's lag sequence equal value by value over the common length -- Acceptance 3's wording, a regression guard on the seed strings and on the routing (`sim_rng` None, so an unrouted draw raises) | **it is near-certain to pass** for any decider and any change under the per-pair seeding, so it says little about this run; it does FAIL if a draw is re-assigned between two pairs of one node. The first-poll offsets are no longer part of it (third pass): `start_lag` runs in the engine's initialisation loop before any event or decider, so no decision can move an offset in ANY configuration, and gating on that was gating on a constant. They are printed as a reported fact instead |
| `lagcheck.py` check 1b | the quantity a decision CAN move, printed and not gated: each other pair's lag draw count, draw-kind sequence, applied intervals, and the moved slice with whose values it carries. On this run the verdict is **NOT HELD** -- `h2->h3` draws 26 lags against 27, its kind sequence first differs at index 21, and the two pairs of `h2` exchange four sends | it is not a pass/fail gate, because Acceptance 3 asks for check 1 and making 1b the gate would redefine it |
| `lagcheck.py` checks 2, 3 | the change is not vacuous -- third pass: the forced 10.043 s really is on the 5th `state_poll_reschedule` row of `h2->h1` (read off the emitted row, not off the decider's bookkeeping) and the event streams differ; and with the streams off the same change moves 7 of the 7 other pairs' lag values | that two candidates see the same lags. Check 2's third disjunct, "the target pair's own lag VALUES moved", is gone: with the per-pair streams on that pair draws one seed string in both runs, so a value at a common position cannot differ and the disjunct could never carry the line (measured: the target drew 25 lags in both runs, so `events_differ` is what holds it) |
| `statecheck.py` check 4b (second half) | `resolved`'s keys are the logged reschedule action's less the four named ones, it is None at every send, and its VALUES are the three fields of the action of the row that decision wrote | nothing about the rest of the state's values; that is `valuecheck.py`'s |
| `lagcheck.py` check 1c (second half) | with the per-pair JITTER stream on as well, the same single change moves no other pair's lag draw count, draw-kind sequence or applied intervals on this run | it is one run, one changed decision, one pair; and it is not a gate. Third pass: it is also a DIFFERENT run from 1b's -- the jitter-on base makes 352 decider calls where 1b's base makes 348 -- so 1b and 1c are not two readings of one run |
| `valuecheck.py` (second half) | the VALUES of four of the six state fields, on a roll-in run of the labelling construction: every origin row (`hi`, `missing`, per-advertiser `hi` and `notified_max`) and `truncated` against a real `Tracker` rebuilt at the decision instant, and `last_poll_status` and `consecutive_poll_timeouts` against a real `_PollReplies` driven by the port's own row stamps | `in_flight` and `last_timeout_s`, which have no reference here: the rebuilt `Tracker` has planned no fetch, so comparing them would compare the port against a fixture. Both are at their default in every call of ITS run, which the script's own situation report states; `last_timeout_s` is not at its default everywhere -- 29 of the 32 219 labelled decisions carry one, and nothing compares those values |
| `labelcheck.py` | the five checks the label tool had to carry, plus the scorer: "F after k" is the rule path's F, the three arms share their prefix and the roll-in's own state at k, a run labelled twice gives identical bytes, a seed above the selection range is refused, the union choice set is inside the "pending" bound, and the scorer's record set is evidence/14's | that a label is right. Every line is about the mechanism. The scorer line now reads `score.py`, the one definition `weight.py` levels the arms on as well; `weight.py` makes the same comparison on all 40 000 of its own simulations, so this line is a sample of a check that also runs at scale |
| `label.py` | the labels of every resolution-time decision of both cells' seeds 1 to 100 at J = 8, the per-cell label counts, the tie share, the margin spread, how many labels move when the censored records are dropped, and the per-run csv; it exits non-zero on a non-zero provisional-timer count | that a label is right, or stable, or the testbed's. It derives no pass line of its own |
| `stability.py` | how far a label moves when the 8 passes behind it are replaced by 8 others, at four pass counts, over three populations plus the larger-margin half restricted to the both-labelled decisions, on seeds 1 to 10 of both cells; the same J = 8 agreement broken down by set A's own paired t and by the decision's position in its run (third pass, bins fixed before the numbers); and the J the full labelling runs with, by a rule fixed before the numbers | **it carries no pass line**, by the ticket's own wording. Two labellings that agree agree about the construction, not about the testbed; and it measures seeds 1 to 10, while the labelling runs 1 to 100. The disagreement it prints is an upper bound on the share of the labels on disk that are pass noise, not that share |
| `observe.py` | every logged `state_poll_reschedule` row of the five nodes of one `s11a` run renders; every record of the logged run, of the replay roll-in and of both label files carries exactly the declared keys in the declared order with the declared value types at every level, and carries no node name, no absolute stamp and no negative age in the record or in its text (fifth pass; the key check used to be over the two sources' unioned paths and the leakage check did not exist); and the characters per item | that the record carries what a policy needs. It carries what the state carries, less every node name and less the decider ordinal, plus the polled peer's POSITION (third pass). Whether that is enough is ticket 11's question |
| `weight.py` | one number, reached in four printed steps from the committed value to the label's own score, with the two arms' mean scores under it, the per-stream spread, the censored share of each arm, the same four steps at p 0.4, the old fixed-arm path's value and the ticket's own estimate beside them; the generator gate; the per-pair seed prefix; the scorer against `early_score` on every one of its 40 000 simulations; the decider-call count; the provisional-timer count; and -- fourth pass -- the jackknife standard error of the step-2, step-3 and step-4 weights over the 100 seeds and the per-seed paired move between steps 2 and 3 | that the weight is right for the testbed, or for p 0.4, or for a labelled decision. Step 1 reproducing the committed value says the four steps are one quantity changing, not two measurements. The standard errors are reporting only: they do not move the fixed value and `weight.json` does not carry them |
| `specnums.py` | the label files' own numbers, recomputed from the files at every run: the two margin medians per cell, the band the "one poll or less" share lands in, the labels the weight correction moved with their directions, the state situations the cells reach with their seeds, and the render-branch counts. It refuses to count anything if a label file's sha256 is not the one `labels.sha256` records | nothing about whether a count is right. It asserts on no value: whether a cell reaches a situation or takes a branch is a property of the workload, and a script that failed when a count moved would be refusing a different workload rather than finding a fault |

## What this step cannot show

- **That a label is the testbed's answer.** A label is a score inside the replay. No run anywhere has
  been made with a decision maker that reads the state, so the port itself has no measurement behind
  it; the nearest thing is evidence/13's validation of the replay's fixed arms and two stage-1
  policies, and evidence/21's own gate, both of which read no decision state.
- **That the state is enough.** The checks are about shape and about agreement with the source, never
  about sufficiency. Whether `decision_state` plus eight resolutions carries what a policy needs is
  ticket 11's question.
- **That `in_flight` and `last_timeout_s` are right** -- the other four were checked by the second
  half, these two were not. `statecheck.py` is still shape only, and the third pass's check 1b did
  not change that: holding every sample to its level's key set and type set is a stricter SHAPE
  check, so a wrong number in any field still passes every line of it. What the second half added is
  `valuecheck.py`, which compares four of the six fields against the source's own objects on a
  roll-in run -- every origin row and `truncated` against a real `Tracker` rebuilt at the decision
  instant (474 origin rows over 316 decisions, 0 differing), and `last_poll_status` and
  `consecutive_poll_timeouts` against a real `_PollReplies` (316 rows, 0 differing, the reference
  reaching `timed_out` 48 times). **`in_flight` and `last_timeout_s` still have no reference**: a
  `Tracker` rebuilt from the store has planned no fetch, so comparing them would compare the port
  against a fixture.
- **That the situations the main cell never enters are built right.** Over the 348 port states of the
  `s11a` p 0.2 seed 1 run, which `statecheck.py` now counts and prints,
  `in_flight` is 0 at every call, `truncated` is False at every call (at most two origins per node
  against a cap of eight), `notified_max` never exceeds `hi` (0 of the 923 advertised cells),
  `last_timeout_s` is null throughout and `last_poll_status` is never "pending". The origin cap, the
  truncation flag, the sparse-extra arithmetic and the millisecond-to-second conversion are therefore
  exercised for their SHAPE only, against a `Tracker` fixture. **The second half's roll-in run reaches
  none of the five either** (0 of 316 calls, 0 of 774 advertised cells), which `valuecheck.py` prints
  as its own situation report. **The full labelling reaches two of the five, which splits the list**
  (measured over the 32 219 labelled decisions of the two label files by `specnums.py`, which is
  where every count of this bullet is read):
  - unreached by every run of any pass: `in_flight > 0` (0 of 32 219), `truncated` (0, at most two
    origins per state against a cap of eight) and `last_poll_status` "pending" (0). These three
    branches have no coverage anywhere here beyond their shape;
  - reached by the labelling and still unverified: `last_timeout_s` set (12 of 16 071 at p 0.2,
    all on seed 90; 17 of 16 148 at p 0.4, 12 on seed 36 and 5 on seed 48), where `observation`
    turned the stamp into `fetch_timeout_age_s` values of 2.1 s to 43.0 s and 0.4 s to 93.9 s;
    and `notified_max` above the advertiser's own `hi` (66 of 43 037 advertised cells at p 0.2,
    seed 90; 0 of 39 125 at p 0.4). Every count, seed and range here is `specnums.txt`'s, so the
    counts are generated from the fourth pass on; what is still missing is a REFERENCE, and no
    object anywhere in this directory says the values those fields hold are right.
- **That a decider returning a numerically equal value of another type is caught.** The guard is
  `interval not in nd.choices`, so an `int` 6 passes beside the `float` 6.0 and the emitted row then
  carries `interval_s: 6`. Harmless for a reader that keys on the number, wrong for one that formats
  the interval into a string key. No reader in this directory does, so it is left as it is and named
  here.
- **That the weight transfers.** It is this cell's indifference point between two arms. At p 0.4 the
  same construction wants a value about **1.34** times as large (139.668108 against 104.295959),
  printed in `weight.txt` and acted on nowhere. Under the first half's censored-dropping delay the
  ratio read 1.28; entering the censored records moved p 0.4 more than the main cell, because p 0.4
  censors 0.05458 of the long arm's records against 0.00400.
- **That two candidates are compared on the same randomness** (narrowed by the second half). With the
  lag streams alone they are not: `lagcheck.py` check 1b measures that the other pairs hold their lag
  VALUES still and not the events those draws serve. With the per-pair jitter stream on as well,
  check 1c finds no other pair moved at all on that one run. **What remains is the probed pair itself**
  -- it really does send a different number of polls, so its own draws land on different events -- and
  that is what the passes average and what `stability.py` measures.
- **That the labels are stable** (answered, badly; see "The stability of a label"). The agreement
  between two labellings from disjoint pass sets does not reach 0.90 at any pass count tried, so a
  measurable share of the labels the full run writes is an artefact of which passes were drawn.
  **What the third pass added is where that share sits**, not a smaller share: the agreement runs
  from 0.46 to 0.97 across the paired-t bins and from 0.51 to 0.96 across the quartiles of a run, so
  a learner can select on either -- and neither is a probability that a label is RIGHT, because both
  sides of the comparison are the same construction.
- **That a weight fixed on the main cell does not flip a label at p 0.4.** Still open. The labels at
  p 0.4 are made with the main cell's weight, as the user fixed, and the comparison that would answer
  it -- the same decisions labelled under p 0.4's own weight -- is not run.
- **That the record carries what a policy needs.** It carries what the state carries, less every
  node NAME and less the decider `ordinal`, plus the polled peer's POSITION (third pass; the second
  half withheld the polled peer and said the record could not carry it). Whether that is enough is
  ticket 11's question.
- **Anything about the live loop.** The port is on the replay only; whether a decision maker fits
  inside the agent's 100 ms loop is ticket 12's question.

## Not written here

**No learner is written**: this step produces the labels and the one road to the text, and both
learners are later tickets. `README.md` and `RUN.txt` ARE written, in this directory and in the
table below; the effort's `map.md` bullets and the ticket's own Result section are written outside
it. All four are written after the numbers above are checked and quote the generated files, never
this one. (Until the fifth pass this section said the four were not written here at all, which had
stopped being true of any of them.)

## Deliverables (this directory)

| file | what it is |
|---|---|
| `spec.md` | this file |
| `port.py` | the seed guard, the per-pair lag and jitter streams, the pass re-key, the provisional-timer count and F with its choice set, in one place each. Its engine subclasses evidence/18's, so every run of this directory goes through the same chain `gate.py` check 2 reproduces the published rows with -- and that pass-through is covered by `gate.py` check 2b, not assumed from the override; evidence/18's `resolve` override is itself a pure pass-through when no clairvoyant plan and no rule are set, which is the case for every run here |
| `gate.py` | Acceptance 1 |
| `statecheck.py` | Acceptance 2, plus `resolved` (check 4b) |
| `lagcheck.py` | Acceptance 3, plus the per-pair jitter measurement (check 1c) |
| `score.py` | the one delay and the one score, so that the weight is levelled on the quantity a label is decided by. Third pass; before it `weight.py` and `label.py` held two definitions |
| `weight.py`, `weight.txt`, `weight.json` | Acceptance 4 |
| `label.py`, `labels.txt`, `labels_summary.csv`, `labels.sha256` | Acceptance 5 |
| `labelcheck.py` | the checks the label tool carries |
| `valuecheck.py` | the state's VALUES against the source's own objects, which `statecheck.py` does not do |
| `stability.py`, `stability.txt` | Acceptance 6 |
| `observe.py`, `observe.txt` | Acceptance 7 |
| `specnums.py`, `specnums.txt` | the numbers this file states that are read off the two label files and that no other script here printed: the labelled-row median margins, the at-or-below-the-weight shares with their floating-point band, the labels the weight correction moved, the state situations the labelling reaches, the render-branch counts, and -- fifth pass -- the decisions at or after the instant a live agent stops, the retention of the t selection over all 100 seeds, and the rows whose t lands on the bar that selection cuts at with the range their rounding leaves its count. Fourth pass; it runs no simulation, so it is rerun after any edit to this file's prose |
| `README.md` | the reading of all of it in Japanese, written by hand from the generated files and the check scripts' output; no script produces it |
| `RUN.txt` | the commands in the order a rerun takes them, the wall times, what each script printed, the label files and their digests, and what was not run |

The label files themselves are **outside** the repository, under
`/home/lab_shared/semindex-learn/ev22/`, one gzipped JSON-lines file per cell; `labels.sha256` is
their digest.

The edits outside this directory are all in `../13/replay.py`: the port (the `decider` key,
`Node.decision_state`, `_Engine.ask_decider`, the two branches, the four new per-neighbour
attributes), the one-line `start_lag` refactor the lag streams need, and -- second half -- the
`resolved` key of `obs`, built in `resolve` and passed through `ask_decider`. **The third pass
changed nothing there but comments and docstrings**: the `x 1000` in the deviation-1 bound at the two
comment sites, the removal of the claim that a mutated `obs["state"]` reaches the next decision's
input, and the split of `where` into real-node context and tool-only metadata. No line of its code
moved, which is why `gate.py` check 1 still compares equal to `fd6ce55` event dict for event dict.

## What the second half changed in this file

| the first half said | the second half's correction |
|---|---|
| `obs` has exactly three keys | four: `resolved` was added |
| the F-union choice set "is not decided here" | it is decided: the union is declared, computed by F's own arithmetic, and the guard is untouched |
| aligning the jitter "would change what every existing stage does" | it is an opt-in option, default off, so no existing stage changes; `lagcheck.py` check 1c measures what it fixes |
| the F-union set "is the second half's job to check" against the "pending" bound | checked: `labelcheck.py` check 5, 3325.5 ms against 2100 ms |
| whether the pass count has to be measured before the first label "is now a decision" | it was measured first: `stability.py` ran before the full labelling and chose J by a rule written before its numbers |
| no check compares a number in `obs["state"]` | `valuecheck.py` compares four of the six fields; two still have no reference |
| Acceptance 5, 6 and 7 "are out of this half" | they are in this file now |
| `last_timeout_s` "is null in practice" and "carries no information on this cell" | it is null on the one run the first half measured, and set in 12 of the 16 071 labelled decisions at p 0.2 and 17 of the 16 148 at p 0.4, so the age it renders reaches the training data; the VALUE is still unreferenced |
| the five situations no run reaches | two of them, `last_timeout_s` and `notified_max` above the advertiser's `hi`, are reached by the full labelling; three (`in_flight`, `truncated`, "pending") are reached by nothing here |

## What the third pass changed in this file

The third pass re-measured the weight, relabelled both cells under it, and corrected eleven things
in the first half's files. It changed no arm, no cell, no definition of what a label means, and not
J = 8, and it tuned nothing after looking at a label. **Most of the eleven are checks that could not
fail where their own pass line said they could**, which is the one kind of defect a file of checks
cannot afford.

| what the earlier text said | the third pass's correction |
|---|---|
| the weight is `100.62168514446252` ms per poll | `104.29595912307323`. The old value levelled a delay that left the censored records out, which is not the score a label is decided by; `weight.txt` prints four steps from the old value to the new one and step 1 reproduces the old one exactly |
| the scorer lives in `label.py` and `weight.py` scores with `../14/after.py`'s `early_score` | one scorer, in `score.py`, which both call. That is the correction above, in code |
| at p 0.4 the same construction wants about 1.28 times the weight | about 1.34 times; entering the censored records moves p 0.4 more, because it censors 0.05458 of the long arm's records against 0.00400 |
| a deep copy of `obs["state"]` "would be about 8.3 million nested copies over `weight.py`'s runs" | one copy per decider call, and `weight.txt` prints the measured count. The estimate multiplied every simulation by one arm's call count and counted the decider-free ones |
| a mutation of `obs["state"]` would change "the published row as well as the next decision's input" | the published row only: `decision_state` allocates fresh dicts at every level and shares no mutable container with the node |
| `min(choices) * (1 - choice_jitter) > POLL_TIMEOUT_MS + TICK_PERIOD_MS` | `... * 1000.0 > ...`. The formula mixed seconds and milliseconds; every number ever plugged into it was converted correctly, so no run was wrong |
| "a learner must never read `where`" | `where` is two kinds of thing. `neighbor`, `kind` and `ts_ms` are real-node context; only the node NAME and the `ordinal` are tool-only |
| "the record does not say which peer this decision polls", in the legend and in a section titled "One thing the record cannot carry" | it does: `observation` takes the polled neighbour as a fifth input and the record carries `polled_peer`, its position. The record is 30 paths, not 29, and about 27 characters longer |
| `statecheck.py` check 1: "one extra or one missing key anywhere fails, in either direction" | only over the UNION of the samples, and with the two decision kinds merged. Check 1b holds every individual sample to its level's key set, and the kinds are compared separately |
| `statecheck.py` check 5 reads `Node.decision_state`'s source for forbidden names | it reads the `ast` of `_Engine.ask_decider` as well, which is the function that assembles `obs`, and `obs` and `obs["where"]` are held to their exact key sets on every call. The old check was a substring search, which cannot tell `nd.poll_hist` from `hist` |
| `lagcheck.py`'s "own again" verdict, cited as output | it could not come out False. It is now `hi < n and len(a) == len(b)` with the lengths printed, and on the measured run the probed pair prints False |
| `lagcheck.py` prints "348 decider calls per run" | the six runs make 348, 346, 352, 352, 348 and 346 calls; one run's count was being printed as every run's, which also hid that the per-pair jitter base is a different run from the per-node one (352 against 348) and therefore that checks 1b and 1c are not two readings of one run |
| `lagcheck.py` check 1's first-poll-offset half, controlled by check 3 | it is not controlled by anything: `start_lag` runs before any decision, so no configuration can move an offset. It is no longer a conjunct of the PASS line; the offsets are printed as a reported fact and the line gates on the moved lags alone |
| `lagcheck.py` check 2 holds if the target pair's own lag values moved | that disjunct could not fire with the streams on. It is replaced by the forced arm being on the targeted decision's emitted row |
| `gate.py` check 1 covers "every mode" | it did not cover the stage-1 fixed branch (policy, choices and rule all None); that cell is now in the list |
| `gate.py` check 3 refuses a decider beside a rule | only a rule in the run dict. The `rule=` keyword argument of `_sim_run` is a second door and is now refused too |
| `gate.py` check 4 reads `git status --porcelain` before and after | and `git diff HEAD`, because porcelain cannot see an already-dirty tracked file being overwritten |
| `stability.py` shows the larger-margin half agreeing less, with two readings and no third measurement | two breakdowns at J = 8, by set A's own paired t and by the decision's position in its run, with the bins fixed before the numbers. Both separate hard (0.46 to 0.97 by t, 0.51 to 0.96 by position), which settles that the margin's failure is the missing spread and not that it is set A's own number; the learner-usable consequence is in "Two breakdowns of the J = 8 agreement" |
| `src/semindex/engine.py:259-274` for `_expire` | `:259-282` |
| `weight.py` runs 24 000 simulations, 16 000 of them port | 40 000 and 32 000, and both totals are now derived from the sweep list rather than written out |
| `weight.py` never names the per-pair seed prefix and never runs the generator gate | both are printed, and the gate stops the script when it fails |
| `labels_summary.csv`'s 200 data rows were the earlier run's bytes beside a header the edited code emits | the whole file, `labels.txt` and `labels.sha256` are one labelling run's output end to end, which is what the relabel bought beside the new weight |
| the label counts, the tie shares and the margin medians of the superseded weight | remade: 46 of p 0.2's 16 071 labels and 53 of p 0.4's 16 148 changed (0.0029 and 0.0033), and the median margin over all decisions moved from 100.6217 to 104.2960 ms at p 0.2 and from 125.7771 to 130.3699 at p 0.4. Every row's decision, state, resolution, history and roll-in arm is byte-identical to the superseded file's, because the roll-in does not read the weight |
| the stability table's 32 shares | remeasured under the new weight: none moved by more than 0.0138, none by more than 0.0038 at J = 8, and the J = 8 disagreement on the main cell went from 0.2416 to 0.2423. No conclusion changed |
| at p 0.4 the `fetch_timeout_age_s` values run "from 0.4 s to 49.6 s" | from 0.4 s to **93.9 s**. 49.6 is the twelfth of the 17 values, not the largest; the list is the same in the superseded and the new label file, so this was a reading error and not a change |

**What a third round of review then corrected in the third pass's own work.** Three independent
verifiers read the files above against the code and the label files; every finding they returned was
reproduced here before it was acted on, and none of them moved a label or the weight.

| what the third pass's own text said | the correction |
|---|---|
| every `../13/replay.py` reference was re-verified and renumbered | every reference at or above `ask_decider`'s docstring was one line short of the statement it names -- 14 of the 16 in this directory, across `spec.md`, `statecheck.py`, `gate.py` and `label.py` -- because the docstring rewritten for finding 9 added a line inside `ask_decider` AFTER the renumbering pass, and `:1033` had never been renumbered at all. A fifteenth, the `:1006-1019` comment block, started one line later than the comment does. All of them now land on the statement, re-checked against the file line by line |
| `0.3699` of p 0.2's labels are "decided by one poll of the windowed count or less" | that share is an exact `<=` against the weight with 1797 margins sitting on it: 652 a last bit below and 1145 a last bit above. With any tolerance it is 0.4501. It is now given as an interval, because which side of the atom a margin falls on is rounding and not a property of the workload |
| `stability.py`'s t table is "over ALL decisions of a cell" | the bin loop dropped what matched no bin without a word, so the p 0.4 table covered 1621 of 1623 -- two set-A ties whose `fmean(a_j - b_j)` rounded a hair below zero. The bins are unchanged and nothing is clamped; the script now prints how many decisions the table covers and the t of anything outside it |
| the p 0.4 paired-t agreement "runs from 0.4700 to 0.9607" | `stability.txt`'s own spread line for that cell says 0.4700 to 0.9615, because the `sd = 0` bucket sits above the top t bin there. Both cells now quote the generated line. **The fifth pass splits that bucket in two and the spread line moves with it; the current range is in "What the fifth pass changed" below** |
| `t >= 3 or sd = 0` keeps 729 of p 0.2's decisions at a 0.9603 agreement rate | 700 of 729 is 0.9602. The rate was this file's arithmetic over two printed buckets; `stability.py` now prints the combined bucket itself, so the number is generated. **That retention counts unlabelled ties as kept, which the fifth pass corrects below; 729 and 0.9602 are not a retention a learner can read** |
| `notified_max > hi` holds in "0 of 1519 advertised cells" | 1519 is no population of that run. `statecheck.py` now counts all five unreached situations over its own 348 port states and prints them; the advertised cells there are 923 |
| the maximum record is "within 6 % of the median" | about 6 % with the legend and about 9 % without it, and the without-legend column is the one a caller that hoists the legend sees |
| a null `polled_peer` means "a pair that has never answered a poll is in no `advertised` map" | that is the minority case: the records that name no origin at all are the common one, and `observe.txt` prints the split. **The predicate that split them tested the wrong condition, which the fifth pass corrects below: the current three-way split is there** |
| `polled_peer` "carries nothing about that name" | it names no node, but it is not independent of the name: the marked position is the pair's rank in plain name order in 0.758 of the labelled records that name two or more peers, because the tie-break order and name order coincide here |
| the null `polled_peer` branch is listed as a branch that "is not dead" | it is also a missing attribution: in 6077 of 32 219 labelled records (0.189) the marker says nothing about which peer the decision is about, which is the gap this pass set out to close |
| a run of the main cell "has about 158 of these decisions" | about 161. 158 is the one logged seed-1 run's count; `labels.txt` measures 160.7 a run at p 0.2 and 161.5 at p 0.4 |
| `lagcheck.py` check 1's first-poll-offset half, closed with prose inside a still-gated PASS line | removed from the verdict. The offsets are printed as a reported fact and the line gates on the moved lags alone, so no conjunct of a PASS line is unable to fail |

## What the fourth pass changed in this file

The fourth pass moved no value: not the weight, not a label, not J, not the observation record. It
measured two things the earlier text asserted or left open, and it moved a class of numbers out of
this file's prose and into a script.

| what the earlier text said | the fourth pass's correction |
|---|---|
| step 3 "moved +3.001866 ms (+2.96 %)", with no account of why | the move is 0.95 times the step-2 weight's own jackknife standard error over the 100 seeds, so the two values do not separate at this cell. The keying hypothesis is UNDECIDED under a rule fixed before the measurement: the paired move of the weight is 1.17 standard errors from zero (claim 2 holds) but the per-pair keying cuts the standard error by 1.51 and not by 2.0 (claim 1 does not), because the outage instants and the record schedule vary from seed to seed as well and are identical in the two steps |
| the fixed weight was given as a point value, 104.295959 ms per poll | with a jackknife standard error of 1.685586 ms over the 100 seeds, about 1.6 % of itself. Two of the three moves BETWEEN the four steps are smaller than that (there are three moves, not four, and the third is +3.001866), and the whole distance from the committed value to the fixed one is 1.16 times STEP 2's standard error, which is the error that distance is quoted against everywhere here |
| the labelled-row median margins, the at-or-below-the-weight shares and their floating-point band, the old-against-new moved counts, the render-branch counts and the reached-situation counts "were measured over the two label files by hand" and "no check in this directory re-derives them" | `specnums.py` re-derives every one of them from the two label files at every run and writes `specnums.txt`, in about 5 seconds and with no simulation. It refuses to count anything if a label file's sha256 is not the one `labels.sha256` records |
| the old-against-new moved counts needed the superseded label files, which are kept nowhere a reader can reach | they are recomputed from the new files alone: a row's `pass_scores` and `polls` give each pass's delay back as `score - weight x polls`, and the superseded score as `delay + superseded weight x polls`. The arithmetic loses about 3e-11 ms, enough to turn an exact tie into a margin of a few 1e-12, so a reconstructed margin at or below 1e-9 ms is read back as the tie it was; the reconstruction was checked against the real superseded files row by row, on all 32 219 rows, while they are still in the third pass's scratch. The row-by-row identity of `state`, `resolved` and `history` is the one comparison that still needs them |
| nothing said how close a label's margin can sit to zero while still being a label | `label.py` calls a tie only at a margin of exactly 0.0, and 1 labelled row at p 0.2 and 4 at p 0.4 carry a label on a margin of a few 1e-12, the last bits of a mean over the 8 passes. `specnums.txt` counts them. Nothing is relabelled for it; it is named because a margin read as a preference is rounding on those five rows |

## What the fifth pass changed in this file

The fifth pass moved no value either: not the weight (`weight.json` is byte-identical), not a label,
not J, not a bin, not the observation record. Two outside readings of the fourth pass's tree were
reproduced here before anything was acted on. What they found was **one measurement counting the
wrong population and one check that could not fail**, and both had been quoted in every document.

| what the fourth pass's text said | the fifth pass's correction |
|---|---|
| `t >= 3 or sd = 0` keeps 0.4596 of p 0.2's decisions at 0.9602 agreement, handed to ticket 10 as the way to select labels | 191 of those 729 decisions TIE on set A and carry no label, so 0.4596 is not a retention a supervised learner can read. The selection is `t >= 3 or (sd = 0 and labelled)`: 538 decisions at p 0.2 (0.3392 of all, 0.3871 of the 1390 labelled) at 0.9777, and 588 at p 0.4 (0.3623, 0.3952) at 0.9643. The retention was overstated by those 191 decisions, 729 of p 0.2's 1586 claimed against the 538 a learner receives, and the agreement understated, since the unlabelled ties agree less than the labelled rows they were pooled with |
| the `sd = 0` bucket is "every pass separated the two arms by the same amount, so there is no t" | true of 66 of its 257 rows at p 0.2 and 53 of 182 at p 0.4. The rest are decisions whose passes scored the two best arms EQUALLY (`d_j = 0` throughout), which have no label at all and whose agreement is `None == None`. The bucket is two buckets, each with its own gloss, and every bucket of both breakdowns prints its set-A ties, its rows labelled on both sides and the agreement over those |
| the `sd = 0` gloss covers every decision with no t | `label.paired_t` also returns no t below 2 passes, where there is one sample and the spread does not enter it. Those decisions now have their own bucket, printed when it holds anything, so a J = 1 run cannot put a whole cell under a sentence about the sd |
| Q4's 0.9597 and 0.9680 are the agreement near the end of a run | they are, and 160 of p 0.2's 397 and 125 of p 0.4's 406 are set-A ties: every tie of a cell sits in Q3 or Q4. Over the rows labelled on both sides the last quartile reads 0.9956 of 228 and 0.9964 of 275, so the ties were pulling that figure DOWN, and what they really change is the size of the population a learner gets |
| nothing anywhere named the decisions the replay makes after a live agent has stopped | 4 of p 0.2's Q4 and 9 of p 0.4's sit at or after `t0 + 130 000 ms`, and 60 and 78 over all 100 seeds. Every one is an exact tie carrying no label, so no labelled row sits there and no relabelling would move one; `label.py` says so where the probes are labelled, `specnums.txt` counts them per cell and `stability.txt` prints them per quartile |
| `observe.py` counts "the state names no origin at all" with `peers_known == 0`, and prints a two-way split of the null marker | that predicate is also true of a state that names an origin NO peer has told it about, so those records were reported under a sentence that does not describe them. The split is three disjoint causes read off `origins_shown` and `peers_known`: 4838 name no origin at all, 598 name an origin no peer has told them about and 641 have advertisers that do not include the polled pair, where the old split read 5436 and 641; the 598 are the misfiled ones, and the logged run's 35-and-1 is really 30, 5 and 1. Every document that quoted the two-way split is corrected |
| `observe.py`'s second PASS line: "the record's key structure is the same built from the log and built from the replay" | it compared the UNION of the dotted paths of each source, which one record missing a key or holding a broken nested dict could not fail. Every record is now held to a declared schema -- key set, field order and value types, at each of the four levels -- one record at a time, over the logged run, the replay roll-in and all 32 219 label rows. The union is still printed, as a reported fact |
| the record carries no node name, no absolute stamp and no negative age | promised in the docstring and checked nowhere. A second PASS line now searches every record and every rendered text for a node name, a 10-to-13-digit stamp and a negative age, over the same population |
| `origins_omitted` is `bool(state["truncated"])` | and the function's own `MAX_ORIGINS` slice. A caller handing over nine origins with `truncated` False had them dropped while the record said none were left out. The source caps at the same eight, so this half has never fired here |
| `specnums.py` prints "so this cell gives the field no value a learner could read" beside the count of `in_flight > 0`, `truncated` and "pending" | only when the count is 0. The sentence is false of any other count, and this script is written to be rerun on other label files |
| `weight.py` prints "the +%f ms" and "inside the 2.0 standard errors claim 2 allows" | the sign comes from the number (`%+.6f`), since the move can be negative on another cell; and the band is read off `claim2`, since on the NOT SUPPORTED branch the old sentence would have asserted a move of 3 standard errors to be inside 2 |
| `port.check_seeds` guards every script that takes seeds | every CLI that takes seeds. `label.run_dict` and `label.label_run` are documented entry points for a later ticket and `port.Engine` is the constructor every run here goes through; none of the three checked, so an import could reach the measuring seeds. All three doors refuse now |
| `labelcheck.py` check 2 verifies the probed decision's own row | with `any()` over the rows matching (node, neighbour, stamp), so a duplicate row at that instant could sit beside the one it passed on. The lookup requires exactly one row, the count is printed, and the row is compared field by field |
| `statecheck.py` check 5's poisoned rebuild | could not print a FAIL. The poison's `AssertionError` left the decider, killed the script from inside `engine.go()` and took the AST half of the check with it. The decider records the exception and check 5 reports it as the fault it is. It records EVERY exception of the rebuild and not that one type, because `Poison` overrides five dunders and cannot raise the named `AssertionError` for every way of touching a history: `o.hist.copy()` raised an `AttributeError` and `o.hist * 2` a `TypeError`, and either would have put the traceback back. `Poison.__getattr__` now raises as well, so the attribute path carries the named message instead of only landing in the FAIL line |
| this file's own weight reading: "neither the step-3 move nor the correction as a whole is larger than the seed noise of the measurement it corrects" | that sentence is about the keying move alone, and `weight.txt` had already dropped it: 1.16 standard errors is larger than one, and +0.845362 ms of the distance is the same simulations rescored, which no standard error covers. Also "three of the four printed steps move it by less than that" -- there are three moves between four steps and two of them are smaller -- and the `769.9 s` wall, and the parenthetical saying the reached-situation counts come from no script in this directory when `specnums.py` is where they come from |
| "`README.md`, `RUN.txt`, the effort's `map.md` and the ticket's Result are not in this directory's scope" | `README.md` and `RUN.txt` are in this directory and in the Deliverables table, which omitted them |
| nothing anywhere said where the bar `t >= 3` falls among the values the t actually takes | it falls on a mass point of them, so the selection's row count holds only under THIS summation order. 39 rows of each cell have a t within 1e-06 of 3 and sit within 1e-13 of it, a value the statistic reaches exactly in real arithmetic, so `>=` splits them by the last bits: 19 kept at p 0.2 and 25 at p 0.4, which leaves the counts anywhere in 5677 to 5716 and 5872 to 5911 (536 to 541 and 586 to 590 over the ten measured seeds). The bar cuts a mass point of the statistic instead of a gap in it, which is a weakness of the bar and not of the arithmetic; `specnums.txt` counts those rows, prints the range and adds the 1246 and 1263 rows whose runner-up ties the third arm in mean, 2 and 5 of which cross the selection when the other arm is paired |
