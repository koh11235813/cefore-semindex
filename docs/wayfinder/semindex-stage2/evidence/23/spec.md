# evidence/23 spec: decision maker L, one forward pass per resolution, and the plan frozen before any tuned model is scored

Written 2026-11-24, **before any tuned model has been scored in the replay and before any agreement with
the labels has been computed**. Ticket `../../tickets/10-llm-system-one.md`; its `決まったこと` sections are the
user's decisions and are not reopened here. Everything a later script needs is decided in this file, with the
reason, so that no choice is left to be made after a score is seen. Where this file chooses something the
user did not decide, the line says **worker's choice** and names the alternative not taken; those are
collected again in one section near the end.

Every number quoted here comes from a generated file in this effort and the file is named: `rows.txt` and
`precheck.txt` and `smoke.txt` and `parity.txt` in this directory, `../22/weight.txt`, `../22/stability.txt`,
`../22/specnums.txt`, `../22/observe.txt`, `../22/labels.txt`, and **`RUN.txt`, which is the only source for
three numbers quoted below** (the three runs of the legend comparison, `RUN.txt:765-766`). `RUN.txt` is a
hand-written log of real runs and not a script's output, and one of those three runs has no surviving log:
`RUN.txt:688` records that it was killed by hand about 90 s in. Several numbers here are a subtraction of two
generated numbers rather than a count of their own -- 3203, 2841, the three seeds-81-to-100 class counts and
the 0.5273 baseline built from them -- and each says so on its own line. A number with no file beside it is not in this file.

What came before, in this directory: `RUN.txt` records the tuning host, the base weights, the inputs, the
byte-identical replay (`parity.txt`), the first forward passes (`smoke.txt`) and the four measurements this
plan rests on (`precheck.txt`). **Nothing tuned was kept and nothing tuned was measured there.** This file is
the step between that preparation and the first measurement.

## Question, and what the step is for

**Does a model that answers in one forward pass, tuned on hindsight labels, decide the interval of a state
poll better than a formula rule and better than a fixed arm, on the replay port, in the main cell?**

L is the first of the three decision makers of the comparison (L here, M in ticket 11, the rule F already
built). The shape is fixed by the ticket: the observable state of one decision is rendered as a short text,
the head of the answer is prefilled, **the next-token logits of three answer words are read once** and the
largest wins. No text is generated, no chain of reasoning is produced, and the tuning is a LoRA adapter over
that one position.

This step produces two tuned candidates, scores them on the replay port against five other deciders, and
chooses one to hand to ticket 12. It is the step in which the plan is frozen; the measurements it describes
have not been made.

## What this step can show

The ticket's own `示せること`, kept, and sharpened where this directory's preparation already settled something.

1. **That a read-out picks one arm from one forward pass, and returns the same arm on the same input.** Shown
   mechanically on the untuned model already (`smoke.txt` part 4: five real label rows, each pass run twice
   and bit-identical, two different rows differing by 19.1562, so the read-out reads the prompt). **That is
   the untuned model only, and the ticket's Acceptance 2 asks it of the read-out that decides**, so every
   script of this step that decides with an adapter attached repeats the same check on its own model (see
   "The repeat check" below). What is added here is the same read-out driving a simulation.
2. **L's score on the replay port, in the main cell** (p 0.2, the s6 outage shape, workload w0, store stage
   `s6b10p043`), for both candidates, beside the three fixed arms and the rule F run through the same port on
   the same per-pair random streams.
3. **The agreement with the labels on seeds 81 to 100, untuned against tuned**, with no pass or fail attached.
4. **What one decision's forward pass costs on `hpc-debian`**, median and p99, printed beside 100 ms, with no
   verdict.
5. **That seeds 121 to 160 were never used**, by what every script prints and asserts.

## What this step cannot show

The ticket's `示せないこと`, corrected by what is now known.

- **Nothing about L on the testbed.** No run of this step touches the emulator. Inserting a decision maker
  into the live loop is ticket 12's.
- **Nothing about whether one decision fits the node's 100 ms loop.** The pass is timed on an RTX PRO 5000
  Blackwell (`precheck.txt` part 1: bf16 batch 1 median 43.02 ms, p99 49.10 ms), which is not a node's
  hardware, and nothing here times the node's own rendering of the observation, its tokenizer call, or the
  rest of its loop. 43 ms sits *beside* 100 ms, not inside it.
- **The labels are not testbed truth.** They are the replay's own hindsight: "if every later decision is left
  to F, which arm is best now". Two independent 8-pass labellings of the same decisions agreed on 0.7577 of
  the decisions both of them labelled (`../22/stability.txt`, main cell, seeds 1 to 10), and the bar of 0.90
  fixed before that measurement was not reached at any J in 1, 2, 4, 8. A disagreement needs only one side to
  be wrong, so **between a half and all of 0.2423 of one labelling's labels are an artefact of which passes
  were drawn.** A model that matched the labels perfectly would match a teacher that disagrees with itself on
  a quarter of its own work.
- **Nothing about the states L visits when it is the one driving the run.** This is the largest gap in the
  step and it is named here rather than in the threats alone. The states a label row carries were produced by
  a **uniformly random** choice of arm: `../22/label.py`'s `Labeller._draw` returns
  `ARMS[int(g.random() * len(ARMS))]` at every decision of the roll-in, and `../22/labels.txt`'s own count of
  the roll-in's arm at the labelled decision -- 3.695 s 5345, 6.0 s 5327, 10.043 s 5399 -- is that uniformity
  printed. The label on top of that state is "best arm now **if F takes over afterwards**"
  (`Labeller.__call__`: after the switch "every decision of every pair, at a send and at a resolution alike,
  is F's"). So two things hold at once that neither the tuning set nor the agreement number can see past:
  the `history` column a tuned model reads carries **uniform-random** intervals during tuning and during the
  agreement measurement, while in the replay it carries **L's own** choices, which `precheck.txt` part 1d
  suggests may be near-constant (334 " short", 20 " mid", 158 " long" of 512 on the untuned model); and the
  label's optimality is conditional on an **F suffix L does not supply**, because after the decision L keeps
  deciding. The randomness differs too: the label rows come from `R.rep_seed("label:s11a", p, seed, 0)`
  (`../22/label.py:267-270`) and the evaluation from `rep_seed("pairlag:s6b10p043", 0.2, seed, rep)`.
  **Consequence, and it cuts both ways: an agreement number near the labels and an agreement number far from
  them are equally uninformative about the replay score.** See Threats 13.
- **A high agreement with the labels does not mean a good score, and this file never reads it as one.** The
  choice between the two candidates is made on the replay score alone (user, `決まったこと` 5). The
  distribution shift above is the mechanism; this line is the rule.
- **The replay is not the testbed.** `evidence/13`'s Mode A, the per-decision agreement, is on record as
  failing: 39 of 42 cells for A1, 24 for A2, 41 for A3, where all 42 were required (ticket 10's own list).
  Mode B's ±6 % band is a per-stage band and is not a bound on one decision.
- **Nothing separates reading the state from reading the clock.** The agreement of a label with a second
  labelling runs from 0.5101 in the first quarter of a run to 0.9597 in the last (`../22/stability.txt`), and
  the rendered text carries the history length and the ages of the last exchanges, so the position in the run
  is readable from the input. Nothing here measures how much of a candidate's score comes from that.
- **Nothing about a class prior being learnt instead of the state.** "Always 10.043 s" already agrees with
  0.5145 of candidate a's rows and 0.6875 of candidate b's (`rows.txt` part 2), so an agreement near those
  numbers is not evidence that anything was read.
- **Nothing about p 0.4, the other outage shapes, other workloads, or finer arms.** The p 0.4 labels are not
  used anywhere (user, `決まったこと` 12), and they were scored under the main cell's weight in any case.
- **Nothing about whether a tuned adapter changes the pass time**, until `latency.py` measures it. `RUN.txt`
  section 8 records that as untested, which is why the latency measurement is made with the adapter attached
  and not on the untuned model.
- **Nothing about whether K concurrent processes read out bit-identical logits**, until `eval.py`'s own gate
  says so. `precheck.txt` part 1b shows one second process beside an idle parent repeating all 512 x 3 bf16
  batch-1 logits exactly; part 2's K = 2 and K = 4 workers were timed and their logits were never compared
  (`RUN.txt` section 13).
- **Nothing corrects the selection bias of choosing one of two candidates on the same 20 seeds.** See Threats.

## Names and definitions

- **ROOT** = `/home/yamakawa/Development/semindex-learn` on the tuning host. **S2** =
  `docs/wayfinder/semindex-stage2` in the worktree. **The main cell** = p 0.2, workload w0, the s6 outage
  shape, store stage `s6b10p043`.
- **The arms** are the exact floats `(3.695, 6.0, 10.043)` (`../22/port.py:45`). An interval a decider returns
  must be an element of the run's `choices` by exact float identity, or the port raises
  (`../13/replay.py:876`).
- **The choice set** of every simulation of this step is `port.union_choices()`, the three arms plus every
  interval F can reach from them. F leaves the three arms as soon as it lengthens, so a run that scored F on
  the three arms alone would raise; and because the choice set only feeds the guard once a decider is set,
  **every decider of the comparison runs on the same union set, so no decider is scored on a narrower set of
  options than another.** That is a claim about the *choice set* and about nothing else. It is **not** the
  claim that the runs differ in nothing but the decider: deciders that change their mind between a send and a
  resolution draw a different number of lags from the pair's stream, so they do not see the same realised
  lags after the first divergence. The mechanism is at "The send-time convention" below and the consequence
  is Threats 14. `parity.py` already runs four of the seven deciders this way (`parity.txt`), and its
  byte-identity covers exactly those four, every one of which answers a send with the same value it answers
  the resolution with.
- **The selecting score** is `../22/score.py`'s `score_run(events, t0, weight)[0]` =
  `delay + weight x polls`, where the delay enters every censored record at its lower bound and the weight is
  **104.29595912307323 ms per poll** from `../22/weight.json` (`../22/weight.txt`, step 4). This is the
  statistic the user's `決まったこと` 5 and 10 select on.
- **The reported frontier gain** is `../20/predict.py`'s `gain = 1 - delay(decider) / interp(frontier,
  polls(decider))`, whose delay is `lat / n` with the censored records **dropped** (`score.delay_dropped`),
  read against a log-log piecewise-linear frontier through the three arm points. Two different delays, two
  different directions of "better": the score selects, the gain is reported beside it (`決まったこと` 10).
- **A run** is one (seed, replicate stream) pair. **A stream** is one of the 40 replicate streams
  (`../21/predict.py:102`, `NREP = 40`); the convention is 40 per seed and this step keeps it.
- **A label row** carries, as model input and nothing else, `state`, `resolved`, `history`, `ts_ms` (as
  `now`) and `neighbor`; as target, `label`.
- **The node rule, in the form `../13/replay.py:826-841` states it, which this file adopts rather than
  paraphrases.** That comment draws a line this spec earlier drew in the wrong place: `neighbor`, `kind` and
  `ts_ms` are real-node context and may be read, while `node` **as a name** and `ordinal` are tool-only,
  "because nothing which reaches a score may depend on a NAME or on the ordinal". So the rule is **not**
  "`node` is never read". It is:
  - **no node name and no ordinal reaches the model input or the policy.** Nothing is rendered from them, and
    no answer is a function of them. `decider.py` asserts this **of the rendered text** -- that is the whole
    reach of that assertion and it is not evidence about the policy, which is why the rule is written here
    as well as asserted there;
  - **keying a decider's own per-pair memory on the pair identity `(node, neighbor)` is real-node context**,
    because a live node knows which node it is and decides per neighbour (`engine.py`'s
    `_choose_poll_interval` takes one). `../22/label.py`'s `F.__call__` already does exactly this, and it has
    to: `run["decider"]` is **one object shared by all five Nodes** (`../13/replay.py:580`), so `neighbor`
    alone collides across nodes and would merge five nodes' memories into one. **L keys its send-time memory
    the same way, and for the same reason.**

## The input text (frozen here)

**The wording of `prefill.py` is frozen unchanged.** It is no longer provisional after this file, and
**`prefill.py`'s own two `PROVISIONAL` comments were amended to say so** (comment-only, `RUN.txt` section 14,
token streams checked equal) -- without that edit a reader of `prefill.py` was told the opposite of what this
plan says. `smoke.py` and `precheck.py` still call the wording provisional in their own text and in their
archived output; that is left alone, because those runs happened before the freeze and editing their record
would unpick it.

The instruction, 48 tokens (`smoke.txt` part 5):

```
A node decides how long to wait before polling this peer for state again.
Answer with one word: short (3.695 s), mid (6.0 s) or long (10.043 s).
```

The body is `../22/observe.py`'s `render(observation(state, resolved, history, ts_ms, neighbor), legend=True)`,
joined to the instruction by a blank line. The messages are `[user: instruction + "\n\n" + body,
assistant: PREFIX]` through `apply_chat_template(..., continue_final_message=True, enable_thinking=False)`.

The answer prefix is `Interval:` -- ending in a non-space character, because the Gemma 4 template runs the
assistant content through `| trim` and a prefix ending in a space loses it silently (`RUN.txt` section 7,
mutation e caught the template doing exactly that). The three answer words are `" short"`, `" mid"`,
`" long"` for 3.695 s, 6.0 s and 10.043 s, ids 2822, 5453, 1440, each adding exactly one token after this
prompt's own ids and all three distinct (`smoke.txt` part 3). `" 10.043"` adds 7 tokens and `" 6.0"` adds 4,
which is why the answer is a word and the seconds stay in the prompt (`smoke.txt` part 3).

**The legend line stays in** -- worker's choice. `precheck.txt` part 3 timed the two variants alternating pass
by pass: 478 tokens with the legend at a median of 43.06 ms, 392 tokens without it at 43.07 ms, so dropping
86 tokens (18.0 % of the prompt) moved the median 0.02 ms the *wrong* way, far below the row's own
median-to-p99 spread of 6.61 ms. Across three runs of that script the same comparison gave -0.69, +0.05 and
-0.02 ms (`RUN.txt:765-766`, the only file that carries the first two; **the second of those three runs has
no surviving log**, `RUN.txt:688`, so it is one recorded number and not a re-readable one), so **the
difference is below what the measurement resolves and the legend is not measured to be free either.** With no time argument on either side, the remaining argument is what `../22/observe.py` built
the legend for: it states the whole ordering of the origins and the peers including the tie-breaks, and
without it `o1` and `p1` are positions the reader cannot resolve -- an order the reader cannot see is a name by
another route. Alternative not taken: dropping it, which would save 86 tokens of prompt, buy no measured time,
and make every token count and every timing in `smoke.txt` and `precheck.txt` stale.

**The wording is otherwise unchanged** -- worker's choice. Every token count and every timing in `smoke.txt`
and `precheck.txt` rests on this exact instruction and this exact prefix; a rewording would invalidate them
for no measured gain, and nothing measured says another wording is better. Alternative not taken: removing
the three state fields that never vary (`in_flight`, `truncated`, `last_poll_status`; see Threats) from the
rendered text. Not taken because the record is shared with ML at the `observation()` level, `render` is
`evidence/22`'s frozen output, and this step may not edit outside this directory.

**Nothing in the input names a node.** The rule, and how it is enforced:

- the only **model inputs** are the five fields listed above. `obs["where"]["ordinal"]` is never read at all,
  and `obs["where"]["node"]` is read for exactly one purpose, keying the per-pair send-time memory, as "Names
  and definitions" sets out: **neither value reaches the rendered text and neither is an argument of the
  answer.** (An earlier draft of this file said flatly that both are never read, which contradicted its own
  send-time convention, since one decider object serves all five nodes and the memory cannot be keyed without
  the node.)
- the neighbour's *value* never reaches the text: `observation()` maps it to a *position*, `polled_peer`, and
  `render` prints "this decision polls p1" or, when the state names no such peer, "a peer none of these
  origins names" -- which is the case in 6077 of 32219 records, 0.189 (`../22/observe.txt`);
- origins and peers are likewise positions `o1..oN` and `p1..pN`;
- **`decider.py` asserts that the rendered text contains none of the five node names `h0`..`h4` as a
  substring**, on every call, and that assertion is shown failing on a render with a name spliced in. **That
  assertion covers the rendered text and nothing else**; the policy side of the rule is in "Names and
  definitions" above.

**The position is name-free and is not name-independent, and the difference matters here.**
`../22/observe.txt` line 70: of the 18448 labelled records that mark a position and name at least two peers,
the position is the polled pair's rank in plain advertiser-**name** order in 13992 of them, **0.758**, against
0.5 for a coin on a two-peer record. `observe.txt`'s own reading of that number travels with it: it is the
ranking's own tie-break coinciding with name order on this workload, and the name is not recoverable from the
record. **So "nothing in the input names a node" is exact about the text and is not a claim that the input
carries no name-correlated signal** -- a model that learnt the position has learnt something that agrees with
the alphabet 0.758 of the time on this workload, and nothing here measures whether that would transfer to a
workload where the two part company. See Threats 11.

## The read-out (frozen here)

**bfloat16, batch 1, everywhere a decision is produced** -- worker's choice, and the single most consequential
one in this file.

| what | dtype | batch | why |
|---|---|---|---|
| the tuning forward | bf16 base, float32 adapter | 16 | what `precheck.txt` part 4 measured |
| the held-out loss during tuning | bf16 base, float32 adapter | 1 | the same read-out the score is taken with |
| the label-agreement numbers | bf16 base, float32 adapter | 1 | the same read-out the score is taken with |
| the replay evaluation | bf16 base, float32 adapter | 1 | the read-out a node would make |

**The base stays bf16 and the adapter stays float32 at every read-out, including the untuned rows, where
there is no adapter at all.** Written out because the argument against float32 below is that reading out in
another arithmetic than the tuned one is wrong, and that argument is only checkable if the adapter's dtype at
read-out is stated. It is peft 0.22.0's own default and not a setting this file adds:
`autocast_adapter_dtype` defaults to `True` on both `get_peft_model` (`mapping_func.py:110`) and
`PeftModel.from_pretrained` (`peft_model.py:489`), and each calls the tuner's `_cast_adapter_dtype`
(`peft_model.py:1220-1223` and `:1764-1767`), which is why `precheck.txt` part 4 printed
`adapter dtype ['torch.float32']` over a bf16 base. Read on the installed build on the host, 2026-11-24; no
script of this step passes the argument, so a peft upgrade that changed the default would change the
read-out, and `tune.py` therefore **asserts the adapter's dtype is float32** rather than trusting it.

The reasons, from `precheck.txt` parts 1 and 2:

- **Batch 1 is forced in the replay and is therefore chosen everywhere.** The decider is asked for one
  interval at a time and the simulation cannot advance until it answers, so there is no batch to form. Taking
  the agreement numbers at batch 1 as well costs 2841 passes x 43.02 ms ≈ 2.0 min per model and makes the
  agreement and the score read out the same model.
- **Batching in bf16 changes decisions, and batch 1 is how that is avoided.** At batch 8 the argmax of 14 of
  512 label rows differs from the same row at batch 1 (2.7 %), with the absolute logit difference reaching
  3.3750; 162 of 512 rows (31.6 %) have a bf16 margin below that. In float32 the batch changes nothing at all
  (0 of 512, max difference 0.0013).
- **bf16 and not float32, although float32 is the faster and the batch-stable one here.** float32 is 36.73 ms
  against bf16's 43.02 at batch 1, and it removes the batching problem entirely. It is not taken because the
  tuning forward runs on a bf16 base (`precheck.txt` part 4) and reading out in float32 would score a model
  whose arithmetic is not the one that was tuned -- part 1 prices that at 13 of 512 argmax differences between
  bf16 batch 1 and float32 batch 1. float32 also costs about 10 GB more of the card (19.59 GB in use against
  9.95) and its parallel throughput was never measured, while bf16's is: one worker's footprint is 10.12 GB,
  so four fit in the 46.73 GB free, and K = 4 is measured at 75.9 passes/s (part 2). Alternative not taken:
  float32 batch 1, which a later step may revisit if a batched evaluation is ever wanted.
- **The adapter is attached and not merged.** The read-out is the call `precheck.txt` part 4 made: no
  `task_type` on the LoRA config, so plain `PeftModel.forward` passes `logits_to_keep` and `position_ids`
  through to Gemma 4's forward. Alternative not taken: `merge_and_unload`, which changes the arithmetic and
  would need its own agreement check against the unmerged read-out.
- **Parallelism, and the gate on it.** The evaluation runs K = 4 worker processes, each holding its own copy
  of the weights and its own read-out at batch 1. **Before any score is computed, one single process alone on
  the card reads out one fixed set of 64 label rows of seeds 1 to 80 at batch 1 and writes those logits to
  disk as the REFERENCE; then every worker reads out the same 64 rows and each worker's logits are compared
  for exact equality (`torch.equal`) against the reference, not only against each other.** **The reference
  pass is made by a child that exits before the K workers are spawned, and the parent holds no model copy at
  any time.** That is a card-size constraint and not a style: one copy of the weights is 10.12 GB and the
  card has 46.73 GB free (`precheck.txt` part 2), so four workers fit at 40.5 GB and a parent holding a
  fifth copy beside them would need 50.6 GB and fail. `precheck.txt:121` shows the measured K = 4 rate was
  taken with the parent at 478 MiB, i.e. holding no model, and that is the arrangement this gate keeps. `RUN.txt` section 13 names the gap in two halves -- the K = 2 and K = 4 workers' logits "were
  never compared with each other **or with batch 1**" -- and a worker-against-worker gate closes only the
  first. Four workers that agreed with each other and differed from the single-process read-out would pass
  such a gate and the score would then be read out at an arithmetic the agreement numbers and the latency
  figure are not, which is exactly what the bf16-over-float32 argument above forbids. The shape is not far
  fetched: `precheck.txt` part 2 shows the per-worker median pass rising from 42.86 ms at K = 1 to 52.5 ms at
  K = 4, i.e. a different occupancy regime. If any comparison differs, the run stops and reports, and the
  fallback is K = 1 (23.1 passes/s, measured). The gate is shown failing by moving one of the compared values
  by one unit. **`agree.py` and `latency.py` run in one process at batch 1, which is the reference
  arithmetic by construction**, so they need no gate; each states that in its own output rather than leaving
  the reader to infer it from a gate's absence.

**The repeat check** (the ticket's Acceptance 2, second half: 同じ入力に同じ腕を返すこと). `smoke.txt` part 4
has it for the **untuned** model only. So **every script of this step that produces a decision with an
adapter attached -- `tune.py`'s held-out pass, `agree.py`, `eval.py`'s gate, `latency.py` -- reads out one
fixed label row twice and prints that the three candidate logits are bit-identical and that the two passes
chose the same arm.** It is three numbers' worth of work and it is the only evidence in the step that the
tuned read-out uses no draw. **Shown failing** by perturbing one of the two stored logit triples by one unit
before the comparison, in each script's own output.

**The tie rule.** Exact ties between two candidate logits do occur: the bf16 batch-1 margin over the 512 rows
has a minimum of 0.0000 (`precheck.txt` part 1d), and even in float32 the minimum is 0.0125, which is not a
guarantee. The decision is taken by an **explicit scan over the fixed candidate order `(" short", " mid",
" long")`, keeping the first maximum**, i.e. the shortest interval wins a tie. `torch.argmax` is not used,
because this step has not read what the installed build does with equal values and a decision that depends on
that is not a decision this file fixed. **Every script that produces decisions prints how many of them were
ties**, so the rule's reach is a number and not a footnote. Alternative not taken: breaking ties to 6.0, F's
own reset neighbourhood and L's pre-first-resolution send value; not taken because it would turn a read-out
convention into a policy choice.

## The training sets

**Counted from the label file's bytes by `rows.py`, output `rows.txt`**, which reads
`/home/lab_shared/semindex-learn/ev22/labels_p0.2.jsonl.gz` after checking its sha256 against the committed
`../22/labels.sha256`, and **opens no other data file: no run store, no model, no `eval.csv`.** It is not
import-free, and the difference is worth stating because the point of the script is that a reader can
re-check its numbers: it imports `../22`'s `label` (for `paired_t` and `ARMS`, one definition rather than a
copy), `port` (for the >100 seed guard) and `stability` (for the bar the agreement was binned on), and
`label.py` reads `../22/weight.json` at its own import. Everything that costs a card, a store or a network is
absent, which is the property that matters: re-running it took 3.3 s. The plan that preceded this step quoted 4 600 rows for
candidate b from an earlier estimate and `precheck.py` used that figure as given; **the real number is 4586**
and this file uses `rows.txt`'s counts throughout.

The main cell's label file holds 16071 rows over seeds 1 to 100; 12868 of them are of seeds 1 to 80, of which
1430 (0.1111) are ties that carry no label (`rows.txt` parts 0 and 1). So seeds 81 to 100 hold
16071 - 12868 = 3203 rows, of which 14279 - 11438 = 2841 carry a label (`../22/specnums.txt` for the 14279,
`rows.txt` for the 11438).

**Candidate a** is every row of seeds 1 to 80 that carries a label: **11438 rows**, which is also what
`precheck.txt` part 0b counted independently.

**Candidate b** is the subset of a that also passes `t >= 3 or (sd = 0 and labelled)`, with `t` and `sd` from
`../22/label.py`'s `paired_t` over the row's own `pass_scores` and `passes` -- the same statistic
`../22/stability.py` binned its agreement on, imported and not re-implemented. **4586 rows**, 0.4009 of
candidate a (`rows.txt` part 2).

- **The rows at `t` exactly 3 are all kept** (user, `決まったこと` 11). The bar is implemented as
  `t >= 3 - 1e-6`, the tolerance `../22/specnums.py` measures the clump against. 31 rows of seeds 1 to 80
  have a `t` within 1e-6 of 3 -- **the seeds-1-to-80 part of the 39 rows the user's decision quotes**, which
  `../22/specnums.txt` counts over the whole p 0.2 cell and of which a bare float keeps 19. The largest
  `|t - 3|` among the 31 is 9.99201e-14, which is 9.99e-08 of the tolerance, and the nearest `|t - 3|`
  outside the tolerance is 0.00077627, which is 776 times it -- so there is room on both sides and a
  tolerance a hundred times wider or narrower selects the same rows (`rows.txt` part 3). **A bare float
  `t >= 3` keeps 4571 rows, so the tolerance keeps 15 rows the float comparison drops** by the order its sums
  happened to run in.
- **The clause removes the rounding at the bar and does not make candidate b's count fully determined.** One
  residue is left and it is `../22/specnums.txt`'s own, over seeds 1 to 100: 1246 rows have a runner-up that
  **ties the third arm** in mean, 759 of them with no `t` at all, and "nothing in the data picks which arm
  the `t` is taken against" -- the order of `ARMS` does, fixed at `../22/label.py:311`'s
  `sorted(ARMS, key=lambda a: (means[a], ARMS.index(a)))`. Taking the other arm of the tie moves **2** of
  those rows across the selection. So that many rows of b rest on this directory's arm order and not on the
  data. **Not recounted on seeds 1 to 80**: `rows.py` counts the bar and not the arm-order tie, and adding a
  second count would add a check with its own mutation for a residue `specnums.txt` already prices. See
  Threats 15.
- **A row with `sd = 0` that carries no label is refused by the `label is None` test first**, so the `sd = 0`
  arm of the rule never admits an unlabelled row. `rows.py` checks that b is a subset of a and the check is
  shown failing.
- **The post-stop rows drop out on their own.** 42 rows of seeds 1 to 80 sit at or after `ts_rel_ms` 130000,
  the instant a live agent stops, and **every one of them is a tie** (`rows.txt` part 1, checked and shown
  failing on a mutated row). A poll scheduled past that instant falls outside the scoring window, so all three
  arms score identically and the margin is exactly 0.0. So neither candidate needs a `ts_rel_ms` filter: the
  `label is not None` test already removes them, and no labelled row is lost by them being removed.

**The class shares and the baseline** (`rows.txt` part 2):

| | 3.695 s | 6.0 s | 10.043 s | "always the most frequent arm" |
|---|---|---|---|---|
| candidate a, 11438 rows | 2581 (0.2257) | 2972 (0.2598) | 5885 (0.5145) | always 10.043 s: **0.5145** |
| candidate b, 4586 rows | 714 (0.1557) | 719 (0.1568) | 3153 (0.6875) | always 10.043 s: **0.6875** |

Candidate b's filter does not keep the three classes in proportion: it keeps 0.2767 of a's 3.695 s rows and
0.5358 of its 10.043 s rows, so **b is the more skewed of the two training sets and its baseline is
correspondingly higher.** Any agreement number is read against the baseline of the set it was measured on,
and never against 1/3.

**Two words, kept apart from here on.** A candidate's **row set** is what the rule above selects: 11438 rows
for a, 4586 for b. Its **tuning set** is the part of that row set the optimiser sees, which is smaller,
because seeds 71 to 80 are held out. `precheck.txt` part 0b's "the plan's candidate a is 11438 labelled rows"
is the row set and stays true; the optimiser never sees 11438 rows.

**The held-out slice: seeds 71 to 80 do not tune.** Carved out of the tuning half, never out of seeds 81 to
100. What each candidate actually optimises on (`rows.txt` part 4):

| | tunes on seeds 1-70 | held out, seeds 71-80 | baseline on the held-out rows |
|---|---|---|---|
| candidate a | 10013 rows | 1425 rows | always 10.043 s: 0.5102 |
| candidate b | 4018 rows | 568 rows | always 10.043 s: 0.6831 |

So **candidate a's tuning set is 10013 rows and its 1425 held-out rows choose the epoch; candidate b's tuning
set is 4018 rows and its 568 held-out rows choose the epoch.**

The held-out class shares sit within 0.016 of the tuning shares for a and within 0.006 for b, so the slice is
not a differently-shaped sample of the problem.

## The tuning recipe (frozen here)

Every setting below is fixed in this file. **No setting is chosen by looking at seeds 81 to 100, and no
number from those seeds reaches the tuning at all.**

- **Base model**: `ROOT/models/gemma-4-E2B-it`, revision `4adf15748de9e1a0097b6274cee9c15836c41c37`, loaded as
  `Gemma4ForConditionalGeneration` in bfloat16, unquantised (worker's choice, recorded in the ticket's
  作業者が決めたこと, and not a decision of the user's). The text-only class is
  not usable: it loads with 541 missing and 1940 unexpected weights, i.e. a randomly initialised text tower
  (`smoke.txt` part 1).
- **LoRA**: `r = 16`, `lora_alpha = 32`, `lora_dropout = 0.0`, `bias = "none"`, `target_modules =
  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`, `exclude_modules =
  r"model\.(vision_tower|audio_tower)\..*"`, **no `task_type`** (so `PeftModel.forward` passes
  `logits_to_keep` and `position_ids` through, which is the call the read-out makes).
  **The tuning script asserts that exactly 205 modules matched, and names the 205.** Not 7 x 35 = 245: the
  last 20 of the 35 text layers are kv-shared (`num_kv_shared_layers = 20`, first shared index 15) and
  transformers 5.24.0 gives them no `k_proj` and no `v_proj` module, so the language model has q/o/gate/up/down
  35 each and k/v 15 each (`precheck.txt` part 4). The exclusion is load-bearing: 353 modules in the whole
  model end in one of the seven names, 148 of them in the vision and audio towers. Expected: 24 158 208
  trainable parameters of 5 128 455 712 (0.4711 %), adapter dtype float32, 410 adapter tensors of which 205
  `lora_B` exactly zero before the first step, and 0 base parameters carrying a gradient -- all asserted, all
  from `precheck.txt` part 4.
- **Loss**: three-way cross-entropy over the three candidate logits at the last real position, computed in
  float32 (`out.logits[:, -1, :][:, CAND_IDS].float()`), target = the index of the row's label in
  `(3.695, 6.0, 10.043)` via an exact dict lookup, so a label that is not one of the three arms raises at the
  first such row rather than being rounded into one. The soft cap of 30.0 that Gemma 4's head applies is left
  alone: `tanh` is monotone, so it does not move the ranking of the three, and both the loss and the read-out
  see the same capped logits.
- **Optimiser**: `torch.optim.AdamW` over the adapter parameters only, `lr = 2e-4`, **every other setting
  torch's own default** -- worker's choice, because `precheck.txt` part 4's step timing was measured with
  exactly that call (`precheck.py:982`) and a changed optimiser is a changed step cost. **One of those
  defaults is not visible in the call and is named here because it is a tuning setting: decoupled weight
  decay of `1e-2` is applied to the LoRA parameters** (`torch/optim/adamw.py:27` on the host's torch 2.15.0,
  read 2026-11-24; betas (0.9, 0.999), eps 1e-8, amsgrad false). It is inherited and not chosen: nothing
  here measures what it costs, and a reader who expected an unregularised adapter should know it is
  regularised. Alternative not taken: an 8-bit or paged optimiser, unnecessary at 24.95 GB peak on a 47.27 GB
  card; and `weight_decay = 0.0`, not taken because it is not the call that was timed.
- **Schedule**: constant learning rate, no warmup, no decay -- worker's choice. `precheck.txt` part 4 measured
  the step cost with no scheduler, and a schedule is two more numbers nothing here can fix. Alternative not
  taken: linear warmup plus cosine decay, which is the usual recipe and would make the epoch comparison below
  a comparison of three points on a decaying curve rather than of three equal epochs.
- **Batch size 16** -- worker's choice. `precheck.txt` part 4: 0.447 s per step at a peak of 24.95 GB
  allocated, against 0.261 s at 17.41 GB for batch 8, i.e. 0.0279 s per row against 0.0326, and 24.95 GB fits
  the 47.27 GB card with room. Alternative not taken: batch 8, which would leave room for a second process of
  our own on a card the brief requires us to find free anyway.
- **No gradient accumulation and no gradient checkpointing.** Batch 16 fits without either (`precheck.txt`
  part 4), and neither was measured.
- **No truncation.** The longest prompt over the whole cell is 519 tokens with the legend (`smoke.txt`
  part 5), so nothing is cut. The script refuses a row whose prompt exceeds 1024 tokens -- a bound no row of
  this cell reaches -- and that refusal is shown firing on a lowered bound.
- **3 epochs, with a checkpoint written after each.** The number of epochs is the one setting nothing measured
  can fix, so it is not fixed: three are run and one is kept by the rule below.
- **Seed 20261124**, one tuning run per candidate, used for the per-epoch shuffle (one permutation per epoch
  from `numpy.random.default_rng(20261124 + epoch)`) and printed. **No seed sweep** -- worker's choice: a sweep
  would be a selection, and a selection made on the held-out slice would spend that slice on a second choice.
  Alternative not taken: three seeds per candidate with the median kept, which would triple the card time and
  leave the variance over seeds measured but unreported anywhere this step reads.
- **How the kept checkpoint is chosen**: after each epoch, the three-way cross-entropy of that epoch's adapter
  on the candidate's own held-out rows (seeds 71 to 80), at the read-out above; **the epoch with the lowest
  held-out cross-entropy is kept, and a tie goes to the earlier epoch.** The rule is mechanical and is fixed
  here, so nothing is chosen after a number is seen. Cross-entropy and not held-out agreement, because the
  cross-entropy is the quantity being optimised and an agreement on 568 rows moves in steps of 0.0018.
  The two adapters not kept are deleted. **Agreement on the held-out rows is printed beside the
  cross-entropy for the reader, and is not what chooses.**

**How many tuning runs, and the wall time.** Two runs, one per candidate, three epochs each. From
`precheck.txt` part 4's 0.447 s per step at batch 16 and `rows.txt` part 4's row counts:

| | steps per epoch | one epoch | 3 epochs | held-out pass per epoch | total |
|---|---|---|---|---|---|
| candidate a, 10013 rows | 626 | 4.7 min | 14.0 min | 1425 x 43.02 ms = 1.0 min | 17.1 min |
| candidate b, 4018 rows | 252 | 1.9 min | 5.6 min | 568 x 43.02 ms = 0.4 min | 6.8 min |

**About 24 minutes of card time for the two candidates together**, plus two model loads at 2.2 s each
(`smoke.txt` part 1). The adapters are written under `ROOT/adapters/{a,b}` and **never enter the repository.**

## The evaluation on seeds 81 to 100

**Seven deciders**, every one of them through `port.sim` on a run dict from `../22/label.py`'s `run_dict`
(which carries `"seed"` and calls `port.check_seeds` itself), with `choices = list(port.union_choices())`,
`choice_jitter = 0.1` and `redecide = True`:

| decider | what it is | calls the model? |
|---|---|---|
| fixed 3.695 s | `port.const_decider(3.695)` | no |
| fixed 6.0 s | `port.const_decider(6.0)` | no |
| fixed 10.043 s | `port.const_decider(10.043)` | no |
| F | `../22/label.py`'s `F`, built fresh per simulation | no |
| untuned | the base model, this file's prompt and read-out, no adapter | yes |
| candidate a | the kept adapter of a | yes |
| candidate b | the kept adapter of b | yes |

**F and L are both built fresh for every simulation, for the same reason**: each keeps one interval per
ordered pair, so one instance reused across two runs would make the second run depend on the first
(`parity.py`'s own reason, `RUN.txt` section 4). This was earlier required of F alone, which was a defect of
this plan and not of the code: L's send-time memory (below) is the same kind of state, so a single L object
reused across the 800 runs would carry pair intervals from run N into run N+1 unguarded. **The three model
deciders therefore keep the loaded model across runs and rebuild the per-pair memory per run** -- reloading
10.2 GB of weights 800 times is not what "fresh" means here -- and `eval.py` asserts that the memory is empty
at the first decision of every simulation, shown failing on a memory seeded by hand.

**The streams.** 20 seeds (81 to 100) x 40 replicate streams = **800 runs per decider**. The per-pair stream
convention is `weight.py`'s, named here because two conventions already exist in `evidence/22` and the
comparison has to declare one:

> `base = R.rep_seed("pairlag:s6b10p043", 0.2, seed, rep)`, passed as **both** `pair_lag_seed` and
> `pair_jitter_seed`, with `sim_rng = None` so that a draw site still reaching the shared stream raises
> instead of mixing -- `../22/weight.py:149-168` (`port_run`), the prefix built at `../22/weight.py:112` from
> `../21/predict.py:117`'s `STREAM_LABEL`. **`port_run` passes the jitter seed only when its own `jitter`
> argument is true** (`return PORT.sim(run, None, base, base if jitter else None)`), so the convention quoted
> here is `weight.txt`'s steps **3 and 4** and not its steps 1 and 2, whose jitter comes from the node's own
> stream. Step 4 is the step this file selects on throughout.

`port.Engine` puts a different kind prefix on each of the two, so they are two independent streams.
`parity.txt` is the evidence that the three fixed arms and F run reproducibly on exactly this convention, on
both machines, byte for byte. **The convention not taken** is `../22/label.py:267-270`'s
`R.rep_seed("label:s11a", p, seed, 0)`, which is the labelling's own and carries no replicate.
**All seven deciders are re-run on these streams**, including the three fixed arms and F. The re-run is the
user's instruction (`決まったこと` 10: 固定の腕の折れ線は、差し込み口の上で、L と同じ乱数の流れで回し直す), so
it needs no evidence to justify it; what the evidence settles is only how much the old numbers would have
been wrong by. **An earlier draft of this file cited the wrong number for that** -- `../22/weight.txt`'s step
2 to step 3 move of +3.001866 ms is the **jitter keying** move, not a port-versus-old-path move: that file's
header says the per-pair lag streams are "on throughout" and step 3 is "step 2 with the per-pair JITTER
streams on as well", and `weight.txt` itself calls the +3.001866 undecided at these 100 seeds (0.95 of step
2's own standard error). The scout report this plan was built on made the same error, so it is corrected here
rather than inherited. The figures that do price the change:
- `../22/weight.txt`'s own old-fixed-arm-path row, "no decider, no per-pair streams", pooled and
  median-taken exactly as step 2 is: **weight 101.264917 ms per poll, -0.202 ms (-0.20 %) against step 2's**,
  and that file says "it is the port and the per-pair lag streams that this row prices". **Small**, and it is
  the honest size of the port effect on the weight;
- `../18`/`../20`'s arm numbers are a **different set of runs**: `weight.txt` recomputes the ticket's
  estimate from `../18/bound_cells.csv` at 104.187090 ms per poll and says why it is not comparable -- "seeds
  1 to 20 and 5 replicates, not 1 to 100 and 40, the arms there are the old fixed-arm path, and its delay
  leaves the censored records out". That, and not a measured move, is why `../20`'s arm points cannot stand
  in for this step's frontier.

**The send-time convention.** The port asks the decider twice per poll (`../13/replay.py:911` at the send and
`:1051` at the resolution) and only the resolution-time call is labelled. Every decider here answers a send
with the interval it already holds, and **none of the three model deciders calls the model at a send** (user,
`決まったこと` 9):

- a fixed arm returns its arm (`port.const_decider`);
- F returns its current per-pair interval and updates only at a resolution (`../22/label.py:144-170`); for a
  pair it has never seen that value is `F_MEM[0] = 3.695` (`../22/port.py:52`);
- the untuned model and both candidates return **the interval that ordered pair chose at its last resolution,
  and 6.0 before its first** (user, `決まったこと` 9), the pair being `(node, neighbor)` as above, and the
  memory rebuilt per simulation as above.

So the *rule* is shared by all seven -- "at a send, return the interval you hold" -- while the **initial value
is each decider's own**, and that is stated rather than glossed: F starts a pair at 3.695 and L starts it at
6.0.

**What this costs the comparison, which is a real and unmeasured difference between the arms and L.** When
the send-time value differs from the resolution-time value, the identity early return in `resolve` is
skipped -- the guard is `../13/replay.py:1103`, `if (nd.rule is not None or nd.redecide) and pend[4] is not
None and due == pend[4]:`, and the comment and the bare `return` are `:1104-1109` -- and **one extra
`tick_lag` is drawn from that pair's stream.** The three fixed arms never differ between a send and a
resolution (`port.const_decider` returns one constant), so **they never draw the extra lag**; F and L draw it
whenever they change their mind. Since the three arms are the frontier the reported gain is read against,
**the comparator family consumes the pair streams differently from L by construction.** `evidence/22` has
this on record as a check that did not hold, and it is a bad result: `../22/README.md` reports
`lagcheck.py`'s 1b as **NOT HELD** -- the other pairs' lag *values* do not move, but for `h2->h3` the draw
count goes from 26 to 27, the draw-kind order diverges at the 21st draw and the applied intervals diverge
from the 6th send, and that file's own sentence is
「候補どうしが同じ乱数を見ている、とはまだ言えない」. **Consequence: per-pair keying equalises the stream
keys, not the realised draws each decider sees, so the arms-versus-L reading carries an unmeasured randomness
difference on top of the decider difference.** See Threats 14.

**The two statistics.**

1. **The selecting statistic**: the mean of `score.score_run(events, t0, 104.29595912307323)[0]` over the 800
   runs of a decider -- one run is one (seed, stream) and the mean is over all of them. This is exactly the
   pooling `../22/weight.txt`'s step 4 solved the weight under, so the weight and the statistic it prices are
   the same quantity. Lower is better. **This number, and nothing else, chooses between candidate a and
   candidate b** (user, `決まったこと` 5).
2. **The reported frontier gain**, with a 95 % interval, reusing `../20/predict.py`'s `interp`, `gain`,
   `boot_index`, `ci` and `word`. Every step of the arithmetic is fixed here, line by line, because most of
   the ways to get it wrong raise nothing and print a plausible number instead.
   - **The per-seed row is `[n4, c4, lat, polls]`, in that order and no other.**
     `score.early_four(score.terms(events, t0))` returns `(n4, c4, lat, polls)`
     (`../22/score.py:89-92`) and `vals` unpacks exactly that -- `n4, c4, lat, polls = m.sum(0)` at
     `../20/predict.py:344`, returning `lat / n4`. An earlier draft of this file wrote the order
     `[lat, n, c, polls]`, which would have handed `vals` `lat` where `n4` belongs and produced a "delay" of
     `n4 / lat`, five orders of magnitude off, with `interp` then read at the wrong x. **The order is a
     silent defect and not an error, which is why it is written out here.**
   - **The pooling of a seed's 40 streams into one row is the column-wise sum**, which is what `vals`'
     `m.sum(0)` then re-sums over seeds and what makes the pooled delay `sum(lat) / sum(n4)`. It is **not**
     a mean of the 40 per-stream gains: `../20` computes a gain **per stream** and prints the **median over
     the 40** (the convention stated at `../20/predict.py:666`, the medians taken at `:725-735`), which is a
     different estimator. **This step pools instead of taking that median**, because the selecting statistic
     is a mean over the 800 runs and a gain taken as a median over streams would not be the same runs'
     summary. Mixing the two would make the printed delay and the printed gain disagree, so only one is used
     and it is named.
   - **The frontier's three points are the three fixed arms as re-run above, not `../20`'s** -- and that
     costs a re-keying the reuse does not give for free. `gain(m, name, col)` reads its frontier from
     `[... for a in ARM_STAGES]` with `ARM_STAGES = ["s10b3p695", "s10b6p0", "s10b10p043"]`
     (`../20/predict.py:115`, used at `:365-382`), and `boot_gains` likewise hardcodes
     `TESTED = ["s10t37r", "s10f"]` (`:116`, used at `:408`). **So `ana.py` stores this step's three re-run
     arms under those three `ARM_STAGES` keys and calls `gain` unchanged, and does NOT call `boot_gains`,
     whose two keys name `../20`'s two rules and not this step's four model-or-rule deciders** -- worker's
     choice: it keeps `interp` and `gain` single-sourced, at the price of three stage keys that name
     testbed stages this step never ran. `ana.py` says so in a comment where it builds the dict, because a
     reader who sees `s10b3p695` in this step's output must not read it as `../20`'s run.
   - **The resample loop is `ana.py`'s own, and it adopts `../20`'s drop-and-count rule by reference.** The
     unit is the **seed**, 2000 resamples from one `(2000, 20)` index matrix drawn once with
     `numpy.random.default_rng(20261122)` -- `BOOT_N` and `BOOT_SEED` at `../20/predict.py:97-98`, the draw
     at `:404`. **The same generator and the same seed, and that is the whole of what is shared: the matrix
     is this step's own.** `integers(0, n, size=(BOOT_N, n))` changes both the draw range and the shape with
     `n`, so at `n = 20` no element of it is an element of `../20`'s `n = 40` matrix. An earlier draft said
     it was "the one those steps drew, with 20 columns here instead of 40", which is false; what is true is
     that it is a function of 20261122 alone and so reproducible. A resample whose arms are not monotone, or
     whose decider's polls leave the arms, **yields no gain and is counted and dropped, not substituted** --
     `boot_gains`' own reason, "because substituting a value would hide how often the frontier is
     unreadable", and the dropped count is printed beside every interval.
   - **The gain can be absent, and the report says so rather than printing a number**: `gain` returns None
     when the three arm points are not monotone (more polls, less delay) or when the decider's own polls fall
     outside the arms' range, because extrapolating a frontier with no arm beyond the point would invent the
     comparison. The line then reads "no gain: the three arm points are not monotone" or "no gain: polls X
     outside [MIN, MAX]", where **MIN and MAX are the re-run arms' own poll counts on seeds 81 to 100, which
     `eval.py` prints** -- not `../22/weight.txt`'s 95.985 and 256.455, which are step 4's means over seeds 1
     to 100 and would make the message quote an interval the test did not use.
   - The 95 % interval at 20 seeds is expected to be wider than `../20`'s at 40 by roughly sqrt(2); **that is
     an inference, not a measurement, and the printed half width is what counts.**

**How the choice between a and b is made, and what "close" means.** The candidate with the **lower mean
selecting score** is chosen -- unless the two are **close**, which is fixed here as

> **`|mean score(a) - mean score(b)| < 297 ms`.**

The arithmetic, from `../22/weight.txt`: the fixed weight carries a jackknife standard error of **1.685586 ms
per poll** over the 100 seeds, which is the uncertainty that file tells a reader to carry on it; the two outer
arms' step-4 poll counts are 256.455 and 95.985 per run, whose midpoint is 176.220; and
1.685586 x 176.220 = **297.0 ms**. So the band is the amount by which re-solving the weight on another draw of
seeds would move a typical run's score. **What the band is, exactly, and nothing more: it is the part of the
comparison the weight's own uncertainty could flip.** It is not the seed-to-seed spread of a candidate's mean
score over its 800 runs -- that number is in no generated file, because no decider of this comparison has
been run yet -- and the band must not be read as a confidence interval on the difference. The seed-to-seed
spread is carried by the paired bootstrap reported below.
Read beside it, the second reading from the same file is wider: the per-stream weights range from 99.899 to
107.251, whose half range of 3.676 ms per poll is 647.8 ms at 176.220 polls. **The tighter of the two is taken
as the band**, because the jackknife standard error is the uncertainty `weight.txt` attaches to the fixed
value while the per-stream range is one experiment's own answer.

**The direction of that choice, stated correctly, because an earlier draft of this file had it backwards.**
The rule is a test of *closeness*, so a **tighter** band makes "separated" the **easier** verdict, not the
harder one: taking 297 ms rather than 647.8 ms makes this step more likely to declare a and b different and
to leave the user's relabelling clause shut. **So the "separated" branch is the weaker-evidence branch of the
two, and the band is not conservative in the direction that matters here.** Two things follow, both
pre-registered now:
- the band is **below** this file's own estimate of the uncertainty that actually decides the question. The
  paired bootstrap at 20 seeds is expected to be wide (see below), and the seed-to-seed spread of a run score
  is large: `parity.txt`'s six single-stream `const3.695` runs score 40921.49, 42754.88, 45469.25, 43591.98,
  39591.96 and 38404.80, a spread of some thousands of ms per run over six seeds. **No generated file carries
  the spread of a *mean* score over 20 seeds**, because no decider of this comparison has been run, so that
  is an order of magnitude and not a number;
- **a "separated" verdict whose gap lies inside the paired bootstrap's 95 % interval is reported as
  "separated, not resolved"**, with both the gap and the interval printed. The chosen candidate is still the
  lower one, because the user asked for a decision; what changes is that the report does not call the two
  distinguished when its own interval cannot distinguish them.
**One honest caveat, stated before the numbers exist**: a weight error shifts both candidates' scores in the
same direction and cancels to the extent their poll counts agree, so 297 ms overstates the wobble when they
poll alike. The evaluation therefore also prints `1.685586 x |polls(a) - polls(b)|`, the part that does not
cancel; it is computable only after the scores exist, so it is reported and is not the rule.
**Reported beside the band and not the rule**: the 95 % percentile interval of `score(a) - score(b)` from a
paired bootstrap over the same 20 seeds (2000 resamples, the same index matrix), and whether it contains 0.
**Its per-seed value is the mean of that seed's 40 run scores** -- named because that is the only aggregate
under which resampling 20 seed indices reproduces the 800-run mean the rule compares. It is not the rule
because at 20 seeds it would be wide enough to call almost any pair close, and because `決まったこと` 6 asks
for a decision and not an interval. **The threshold itself is a worker's choice and is flagged for the user**
in the hand-over, because what it gates -- whether the relabelling clause opens -- is the user's call and
this file's own reading is that 297 ms sits below the relevant uncertainty.

**What follows from "close"**: the relabelling clause of `決まったこと` 6 opens -- the question of a labelling
with more passes becomes live, priced by `../22/labels.txt`'s 3706.3 s on 5 workers for 8 passes over both
cells against the rise in agreement from 0.6847 at J = 1 to 0.7577 at J = 8 (`../22/stability.txt`). Both
scores are reported, neither candidate is called better, and **the candidate carried to ticket 12 is b** --
worker's choice, a tie-break convention and not a measured preference. The reason: b's filter is the one whose
agreement across two independent labellings was actually measured -- **on the bare-float cut of that filter,
which differs from b's tolerant rule by the 3 bar rows `../22/specnums.txt` counts on seeds 1 to 10; see
Threats 12** -- (0.9777 against 0.7623 over everything,
`../22/stability.txt`), and relabelling with more passes changes b's row set and not a's, so carrying b keeps
the later step's two options on one candidate. Alternative not taken: a, for its 2.49x more rows.

**The expected number of forward passes and the wall time.** A generated file does carry the count, and an
earlier draft of this file wrongly said none did: `../22/labels.txt` line 9 reads "p 0.2: 100 runs, 16071
labelled decisions (160.7 per run, min 150, max 170)", and those 16071 rows are exactly the roll-in's
**resolution-time** decisions, because `../22/label.py`'s `Labeller.__call__` captures and labels only
`kind == "state_poll_reschedule"`. So **160.7 resolution-time decisions per run (150 to 170)** is the anchor,
and 800 runs give

| | passes | at K = 4 (75.9 passes/s) | at K = 1 (23.1 passes/s) |
|---|---|---|---|
| 800 runs x 160.7, one model decider | 128 568 | 28.2 min | 92.8 min |
| the three model deciders together | 385 704 | 84.7 min | 278.3 min |

which is `precheck.txt` part 2's own "128000 passes at this rate: 28.1 min" line, reached from the data
instead of from a bracket. **It is the roll-in's count and not L's.** The roll-in draws uniformly over the
three arms, so its poll rate sits between the arms'; a candidate that leaned on " short" would poll more
often and cost more passes, one that leaned on " long" fewer. For scale, `../22/weight.txt`'s step 4 windowed
poll counts of the two outer arms are 256.455 and 95.985 per run, so a near-constant candidate could cost
roughly 1.6x or 0.6x the table above. **`eval.py` prints `engine.decider_ord` so the real number is on
record**; that attribute is a dict keyed by call kind (`../13/replay.py:589`), so **the forward-pass count is
its `"state_poll_reschedule"` entry alone**, its `"state_poll"` entry costing no pass because no model
decider calls the model at a send. The two non-model families cost no passes. The simulations themselves are
negligible beside this: `../22/weight.txt`'s 40 000 simulations took 161.4 s on 5 workers, so the 5600
simulations here are of the order of 20 s. `precheck.txt` part 2's own figures for scale: 128 000 passes take
92.2 min at K = 1, 46.7 at K = 2, 28.1 at K = 4, and K = 8 does not fit (one worker is 10.12 GB, eight need
80.95 GB against 46.73 GB free).

## The label-agreement numbers (Acceptance 3)

On **seeds 81 to 100** of the main cell: 3203 rows, of which 2841 carry a label. For each of the three model
deciders -- untuned, candidate a, candidate b -- `agree.py` prints, at the read-out fixed above:

- the share of the 2841 labelled rows whose argmax equals the label, and the same per label class;
- the 3 x 3 table of label against answer, so a decider that collapsed onto one word is visible as a column;
- the share of answers that fell to each word, beside the label shares of those 2841 rows;
- the number of exact ties resolved by the tie rule;
- **the "always the most frequent arm" baseline of those same 2841 rows**, so every agreement is read against
  it and never against 1/3. **That baseline is pre-registered here as 0.5273, before any model is read out**,
  so prediction 5 below has a bar instead of a hope. It is a subtraction of two generated files, exactly as
  the 2841 above is: `../22/labels.txt`'s whole-cell label counts minus `rows.txt` part 2's candidate a
  counts gives seeds 81 to 100 as 3.695 s 3221 - 2581 = 640, 6.0 s 3675 - 2972 = 703, 10.043 s
  7383 - 5885 = 1498, summing to the 2841, so always 10.043 s agrees with 1498 / 2841 = **0.5273**. Nothing
  was counted on seeds 81 to 100 to get it and `agree.py` recomputes it from the rows it reads, which has to
  reproduce 0.5273 or the script stops;
- the untuned and the tuned numbers side by side, which is what the ticket's Acceptance 3 asks for.

**No pass or fail is attached**, by the ticket's own instruction, and the ties that carry no label are excluded
from the denominator and counted separately.

**What agreement does not mean.** It is not a score and does not predict one: the choice between the
candidates is made on the replay score and never here (`決まったこと` 5). It is not accuracy against testbed
truth: the labels are the replay's hindsight, and two independent labellings of the same decisions agreed on
only 0.7577 (`../22/stability.txt`), so **an agreement much above that is agreement with one labelling's
noise, not with a better answer.** And it is not evidence that the state was read: 0.5145 of candidate a's
rows and 0.6875 of candidate b's are already matched by always answering 10.043 s (`rows.txt` part 2).

## The latency measurement (Acceptance 5)

`latency.py`, on `hpc-debian`, with **the chosen candidate's kept epoch attached and unmerged** -- both words
matter and "the kept adapter" carried two referents: the epoch is kept by the held-out cross-entropy rule and
the candidate is chosen by the selecting score, and this script runs after both -- the read-out fixed above
(bf16 base, float32 adapter, batch 1, single process), on **the median-length prompt of the seeds 1 to 80
rows**: 10 warm-up passes then 300 timed passes, each bracketed by `torch.cuda.synchronize()`, median and p90
and p99 and max printed **beside 100 ms, with no verdict**, as the ticket requires. The card is checked free
first, by the script and not by hand.

**Seeds 1 to 80 and not 81 to 100** -- a correction, not a preference: an earlier draft timed the selection
half's median prompt, which is a use of the selection half for something this file promised it would not be
used for, and the fix costs nothing because `precheck.txt` part 1 already timed a seeds-1-to-80 row at 478
tokens, the median of its 512 (`precheck.txt:31-33`, "seed 3 k 103, 478 tokens"). So the new figure is
comparable with the 43.02 ms on record rather than being a second prompt length as well as a second model.

Printed beside it and named as **not part of the pass**: the time to build the record and render it
(`observe.observation` + `render`) and the time to tokenize it, measured on the same host over the same rows.
A node pays those too, and the forward pass alone is not a decision.

The untuned pass is already on record at 43.02 ms median and 49.10 p99 (`precheck.txt` part 1), so the figure
this step adds is what the adapter costs on top; `RUN.txt` section 8 records that as untested.

**What it does not say about a node.** It is not a node's hardware. It does not include the node's own loop,
its tracker update, its fetch planning or its transport. It times one call per decision, which is what the
send-time convention buys; a decider that called the model at a send would pay twice. It says nothing about a
merged adapter, nothing about contention with other work on the card, and nothing about whether 100 ms is the
right budget -- the loop question stays ticket 12's.

## The proof that seeds 121 to 160 are never used (Acceptance 6)

Every script of the next steps -- `tune.py`, `agree.py`, `eval.py`, `ana.py`, `latency.py` -- does all of the
following, and prints it. **`rows.py` already does items 2, 4 and 5**, and not 1 or 3, because it starts no
simulation and reads rows of one role only; its output is cited under each item it covers.

1. **Builds every simulation through `port.sim` / `port.Engine` on a run dict that carries `"seed"`**, so the
   constructor guard at `../22/port.py:167-169` is on the one path.
2. **Passes every seed list it computes through `port.check_seeds(..., "<script>")` before using it**, and
   additionally through its own `check_seeds_tuning` (refusing anything above 80) wherever the list is a
   tuning list. `port.check_seeds` refuses above 100 and would let 81 to 100 through, so it cannot stand in
   for the tuning guard: a tuning number that quietly used a selection seed would make the choice between a
   and b circular.
3. **Prints the sorted seed set for each role** -- tuning 1 to 70, held out 71 to 80, selection 81 to 100 --
   and asserts that each role's set is inside its range and that the union's intersection with
   `range(121, 161)` is empty.
4. **Reads the `seed` column of the label rows it touches and prints its min, max and count**, so the claim
   covers the data and not only the simulations. `rows.txt` part 0 already does this: min 1, max 100, 100
   distinct, passed by `port.check_seeds`.
5. **Carries the negative test**: the guard shown refusing, in the script's own output.
   `rows.txt` part 5 shows both guards firing -- `seeds_tuning` on 81 and `port.check_seeds` on 101 -- and
   `../22/labelcheck.py:223` is the same test in `evidence/22`.
6. On the tuning host the same guards run, because `ROOT/repo` carries `evidence/22`'s code at the same
   relative paths and `PYTHONPATH=ROOT/repo/src` plus `sys.path` puts `port.py` and `label.py` in reach --
   `precheck.py` already imported and used them there (`precheck.txt` part 0b: "port.check_seeds passed
   them"). **So no process of this step filters seeds without a guard**, and no export of training rows is
   needed to carry the claim across machines.

**The one place the selection half is read before any score exists** is the class-share baseline of the
agreement section, 0.5273, which is a subtraction of two already-generated files and involves no model and no
simulation. It is named here so that the promise at "The tuning recipe" -- no *setting* is chosen by looking
at seeds 81 to 100 -- is read against a full list and not an implicit one. Nothing else of this step reads
those seeds before `eval.py` runs.

**Where the ticket's Acceptance 1 is already satisfied**, since the five sections below cover Acceptance 2,
3, 4, 5 and 6 and would otherwise leave it unaccounted for: the base weights are on `hpc-debian` at
`ROOT/models/gemma-4-E2B-it` and the licence step is written up in the ticket's own 下調べ section (the hub's
`gated: false` and `license: apache-2.0`, the unmodified Apache 2.0 terms, and that no condition attaches
unless a tuned weight is published) and in `RUN.txt` section 2 (what was fetched, from where, and its size
and digest). **This step adds nothing to Acceptance 1 and closes nothing of it**; it is listed so the six
items are accounted for in one place.

Beside that, as `RUN.txt` does: `git status --porcelain` showing only paths under `evidence/23`, and the
statement that nothing was written under `/home/lab_shared/semindex-eval` or `/home/lab_shared/semindex-learn`,
that `evalgen.generate_run` was never called, and that under an archived `debug/node_dirs` only
`*/semindex/events.jsonl` was opened.

## Predictions, written before the data

Short and honest, so that a later reader can tell a finding from a rationalisation.

1. **a against b: not separated**, i.e. the gap falls inside the 297 ms band. Both face the same
   prior-dominated three-way problem, and b's cleaner labels (0.9777 against 0.7623 agreement on the seeds
   where it was measured) buy back roughly what its 2.49x fewer rows cost.
2. **F's mean selecting score is at or below the better candidate's**, i.e. `score(F) <= min(score(a),
   score(b))`. One clause, one outcome that falsifies it. F reads the reply it has just resolved and reacts
   to it; L answers from one read of a teacher that disagrees with itself on a quarter of its own labels. (An
   earlier draft added "and if L beats F at all, by less than the gain's half width", which let the first
   clause fail without the prediction failing; that clause is dropped rather than kept as a hedge.)
3. **The tuned candidates beat the untuned model on label agreement by more than 0.05, and their mean
   selecting scores differ from the untuned model's by less than the 297 ms band.** Both halves carry a
   number, because "clearly" adjudicates nothing. The reasoning: the untuned model is nearly a constant arm
   -- over the 512 rows of `precheck.txt` part 1d it answered " short" 334 times, " long" 158 and " mid" only
   20 -- so a near-constant " short" decider should land near the three fixed arms' own level rather than far
   below it. **What that level is on seeds 81 to 100 is an output of `eval.py` and not a known constant**:
   the two outer arms' equality to every digit of a double (40954.465203126296 for both) is `../22/weight.txt`
   step 4's **arithmetic identity over the 4000 runs of seeds 1 to 100**, the weight having been solved for
   it, and it does not hold on a subset. `parity.txt` shows how far apart the arms sit per seed -- seed 100:
   3.695 s 38404.80 against 10.043 s 52836.45 -- so the prediction is read against whichever arm level
   `eval.py` prints and not against 40954.47.
4. **The frontier gain's 95 % interval contains 0 for every candidate at 20 seeds**, i.e. `word` returns
   "not-seen".
5. **Agreement with the labels in the 0.55 to 0.70 band** for a tuned candidate on seeds 81 to 100. The band
   is placed against two numbers fixed before any read-out: it starts above the pre-registered
   most-frequent-arm baseline of those rows, **0.5273** (the agreement section's own derivation), by enough
   that beating the class prior is what the lower edge tests; and it ends below the labels' own
   self-agreement of 0.7577, above which an agreement is agreement with one labelling's noise. An earlier
   draft wrote 0.55 to 0.75, whose edges sit 0.023 above the baseline and at the self-agreement, i.e. covered
   nearly the whole plausible range and could hardly fail.

**What would be surprising**, and would be reported as surprising rather than smoothed over: the untuned model
scoring better than both tuned candidates; a gap between a and b above 1000 ms; L beating F by more than the
gain interval's half width; any candidate agreeing with the labels above 0.80; the frontier gain's interval
excluding 0 in either direction at 20 seeds.

## Threats

1. **Choosing one of two candidates on the same 20 seeds biases the chosen one's reported score.** Lower is
   better here, and the rule picks the lower of two, so the winner's figure is the **minimum of two draws**
   and is biased **downward**, i.e. it flatters the chosen candidate. (An earlier draft wrote "maximum of two
   draws", which names the same bias with the sign of "better" inverted.) The bias reaches **both** reported
   statistics and not only the selecting one: the frontier gain is a different delay on the same runs
   (censored dropped, pooled per seed), so selecting on one of them biases the other through the runs they
   share. **And it reaches the step's own question.** The question is whether L beats F and the three arms;
   the figure compared against F and against the arms is the chosen candidate's, so **the L-versus-F and
   L-versus-arm readings inherit the bias and are each a best-of-two against a single comparator.** Nothing
   here corrects it; the only clean correction is a set of seeds nobody has looked at, and those are 121 to
   160, which ticket 12 holds. **The reported score of the chosen candidate is a bound on its quality in the
   favourable direction, not an estimate of it.**
2. **20 seeds give a wide interval.** The bootstrap unit is the seed, and `../20` used 40 where this step has
   20, so the half width should be about sqrt(2) wider -- an inference, not a measurement. A "not-seen" verdict
   at 20 seeds is weak evidence of no effect.
3. **The label noise.** Two independent 8-pass labellings agreed on 0.7577 of the decisions both labelled
   (`../22/stability.txt`), the 0.90 bar fixed in advance was never reached, and J = 8 was taken by the rule's
   fallback. Both candidates learn from one of those two labellings and neither of them is the truth.
4. **Three state fields never vary in any training data.** `in_flight` is never above 0, `truncated` is never
   true and `last_poll_status` is never "pending", in any of the 32219 records of both cells
   (`../22/specnums.txt`). The rendered text carries three fields that carry no information here, and a model
   tuned on this cell has never seen them move. **What a reader may therefore not conclude: no number of this
   step transfers to a condition in which any of the three varies** -- a loaded node with polls in flight, a
   truncated advertisement, a pending poll -- because the tuned model's response to those values is
   unconstrained by anything it was shown, and three constant tokens are as likely to have been ignored as
   read.
5. **The model may learn the position in the run rather than the state.** The history length, the ages of the
   last exchanges and the number of exchanges all encode how far into the run a decision is, and the label's
   own reliability moves with that position: 0.5101 agreement in the first quarter against 0.9597 in the last
   (`../22/stability.txt`). Position and label quality are confounded, and nothing in this step separates
   reading a clock from reading a state.
6. **The read-out's own noise.** Exact ties between two candidate logits occur in bf16 (margin minimum 0.0000
   over 512 rows) and the argmax of 13 of 512 rows differs between bf16 batch 1 and float32 batch 1
   (`precheck.txt` part 1), so about 2.5 % of the decisions are the dtype's and not the model's. Batch 1
   removes the batch's own 2.7 % and not this. **What that does to a score is not bounded anywhere**: 2.5 %
   of some 128 568 decisions is a few thousand decisions made by the arithmetic, and the comparison they feed
   is judged against a 297 ms band, so **a candidate's score carries a read-out component of unknown size
   relative to the band that decides between the candidates.** Both candidates read out the same way, which
   makes the component common and not cancelling, because the decisions it flips are different ones.
7. **The tuning forward is batched and the read-out is not.** At batch 16 the logits differ from the batch-1
   logits by what batch shape costs -- 14 of 512 argmax differences at batch 8 (`precheck.txt` part 1) -- so
   the gradients were computed on logits slightly unlike the ones the model will be read out at. Nothing here
   measures what that costs. **What a reader may therefore not conclude: that the model being scored is the
   model that was optimised.** The gradient steps were taken on batch-16 logits and every decision is made
   on batch-1 logits, so the kept epoch minimises a held-out cross-entropy of the read-out arithmetic over
   weights fitted in another, and the two are the same model only up to that difference.
8. **The replay is not the testbed.** `evidence/13`'s per-decision agreement failed (39 / 24 / 41 of 42 cells
   required). A score measured here screens candidates; it does not predict a testbed run.
9. **The weight itself is uncertain.** Jackknife standard error 1.685586 ms per poll over 100 seeds, with the
   per-stream weights ranging 99.899 to 107.251 (`../22/weight.txt`). The selecting statistic moves with it,
   and the 297 ms band is built from exactly that uncertainty.
10. **One labelling, one tuning seed, one wording.** Nothing in this step measures the variance over any of
    the three, so every number it produces is one draw from each. **What a reader may therefore not
    conclude: that re-running this step would reproduce its numbers, or that a difference between a and b of
    any size is a difference between the two filters rather than between two draws.** The one-seed choice is
    deliberate (a sweep would be a second selection) and its cost is exactly this.
11. **`polled_peer` is null in 0.189 of all records** (`../22/observe.txt`), where the text says "a peer none
    of these origins names". A fifth of the decisions therefore do not say which neighbour they are about,
    and the model cannot condition on the peer there. **What a reader may therefore not conclude: that L is
    a per-neighbour decider on those records.** On a fifth of its decisions the input identifies no peer, so
    whatever L answers there is a function of the node's own state and the clock alone, while the interval it
    returns is nevertheless applied to one ordered pair. And on the records that do mark a position, the
    position agrees with plain advertiser-name order 0.758 of the time (`../22/observe.txt` line 70), so a
    learnt position is not demonstrably a learnt role.
12. **Candidate b's filter was measured for agreement on seeds 1 to 10 only.** Its retention on seeds 1 to 80
    is counted (`rows.txt`), but whether the same 0.9777 agreement holds there is not measured and cannot be,
    because the file on disk is one labelling. **And the 0.9777 was measured on a slightly different set of
    rows than b is.** `../22/stability.py` bins with a bare float comparison (`if lo <= t < hi`, `:273`) and
    its selection is `buckets[T_BINS[-1]] + buckets[SD0_LABELLED]` (`:462`), so the 538 rows it measured are
    the **float cut**, while b keeps every row on the bar. `../22/specnums.txt` prices the gap on those very
    seeds: of the 5 bar rows in seeds 1 to 10, "the float comparison keeps 2 of them and drops 3", so 3 rows
    of 538 are in b and were not in the measured set. The number is small; the point is that **0.9777 is not
    an agreement measured on candidate b's own rule**, and this file leans on it twice -- once to prefer b on
    a tie, once in prediction 1.
13. **The tuning distribution is not the evaluation distribution, and this is the step's largest open
    threat.** The states carrying the labels come from a **uniformly random** roll-in
    (`../22/label.py`'s `_draw`, and `../22/labels.txt`'s own roll-in arm counts 5345 / 5327 / 5399), and the
    label is the best arm **under an F suffix** that L does not supply. In the replay, L drives the run, so
    the `history` column it reads carries its own choices and the future after its decision is its own. The
    randomness differs too (`label:s11a` against `pairlag:s6b10p043`). **What a reader may therefore not
    conclude, in either direction: nothing about the replay score follows from the agreement number**, and a
    tuned model may be conditioning on a `history` distribution it never meets again. Correcting it would
    need labels generated under L's own roll-in, i.e. a labelling that does not exist and that the user's
    `決まったこと` 4 does not ask for; it is left open and named, not fixed.
14. **The per-pair streams equalise the stream keys and not the realised draws.** A decider whose send-time
    value differs from its resolution-time value skips `resolve`'s identity early return
    (`../13/replay.py:1103-1109`) and draws one extra `tick_lag` from that pair's stream. The three fixed
    arms never differ and never draw it; F and L do. `evidence/22` records the measured consequence as a
    check that did not hold -- `../22/README.md`: `lagcheck.py`'s 1b is **NOT HELD**, `h2->h3` going from 26
    to 27 lags with the draw-kind order diverging at the 21st draw and the applied intervals at the 6th
    send -- and its own sentence is 「候補どうしが同じ乱数を見ている、とはまだ言えない」. **What a reader may
    therefore not conclude: that the difference between L's score and an arm's score is the decider's
    difference alone.** It carries an unmeasured randomness difference on top, in an unknown direction, and
    `parity.txt`'s byte-identity does not cover it, because all four deciders it reproduces answer a send
    with the value they answer the resolution with.
15. **Some of candidate b's rows rest on this directory's arm order and not on the data.**
    `../22/specnums.txt`, seeds 1 to 100: 1246 rows have a runner-up that ties the third arm in mean, "the
    order of `ARMS` and nothing in the data picks which arm the `t` is taken against"
    (`../22/label.py:311`), and taking the other arm of the tie moves **2** of them across the selection.
    The user's "keep every `t` of exactly 3" clause removes the rounding at the bar and leaves this residue
    untouched. **Not recounted on seeds 1 to 80 by any script of this directory**, so the 2 is an upper
    bound carried over from the wider cell. **The residue is not empty there**: a recount made after this
    file was reviewed, with numpy's mean in place of `statistics.fmean`, came out one row short of 4586, and
    the row is seed 12, k 70, whose 3.695 s and 6.0 s arms tie in mean to the last bit, giving `t` = 7.0
    against one and 2.497 against the other (`RUN.txt` section 16, a hand-logged run whose script is not in
    the repository). **What a reader may therefore not conclude: that b's 4586 is reproducible from the data
    alone.**

## Every worker's choice, in one place

| choice | reason | alternative not taken |
|---|---|---|
| keep the legend line | no measurable time cost (`precheck.txt` part 3); it is what makes `o1`/`p1` resolvable | drop it: 86 tokens, no measured time, loses the ordering statement |
| freeze the wording unchanged | every token count and timing in `smoke.txt`/`precheck.txt` rests on it | reword; or drop the three constant state fields (not editable here) |
| bf16 batch 1 for every decision | the dtype that was tuned; batch 1 is forced in the replay and removes the 2.7 % batch drift; K = 4 measured | float32 batch 1: faster, batch-stable, but a different arithmetic from the tuning and no parallel measurement |
| batch 16 for tuning | 0.0279 s/row against 0.0326 at batch 8, 24.95 GB of 47.27 (`precheck.txt` part 4) | batch 8, leaving card room for a second process of our own |
| AdamW, lr 2e-4, torch defaults, no schedule | the exact call `precheck.txt` part 4 timed | warmup plus cosine decay; an 8-bit optimiser |
| 3 epochs, lowest held-out cross-entropy kept, earlier epoch on a tie | the one setting nothing measured can fix, decided by a mechanical rule on tuning-half data | a fixed epoch count with no held-out slice, which would keep 1425 and 568 more rows |
| carve seeds 71 to 80 out of the tuning half | gives the epoch rule something to read that is not seeds 81 to 100 | tune on all of 1 to 80 and fix the epochs by judgement |
| one tuning seed, 20261124 | a seed sweep is a second selection and would spend the held-out slice on it | three seeds per candidate, median kept |
| tie rule: first maximum in `(" short", " mid", " long")` | deterministic, no new constant, independent of `torch.argmax`'s undocumented tie behaviour | break ties to 6.0, which would make the read-out a policy |
| K = 4 workers, gated against a single-process batch-1 reference read-out | the only measured rate above K = 2 that fits the card; a worker-against-worker gate closes only half of what `RUN.txt` section 13 says is missing | K = 1, three times the wall time; or the worker-against-worker gate alone, which four agreeing-but-wrong workers would pass |
| L built fresh per simulation, keyed on `(node, neighbor)` | L's send-time memory is the same per-pair state F's is, and `run["decider"]` is one object for all five nodes (`../13/replay.py:580`), so `neighbor` alone merges them | one L reused across the 800 runs, which makes run N+1 depend on run N; or keying on `neighbor` alone, which merges five nodes' memories |
| `ana.py` stores the re-run arms under `../20`'s three `ARM_STAGES` keys and calls `gain` unchanged; the resample loop is its own | keeps `interp` and `gain` single-sourced; `boot_gains`' own keys name `../20`'s two rules and cannot serve this step's four | re-implement `gain` in this directory, a second definition of the frontier |
| the repeat check in every script that decides with an adapter | the ticket's Acceptance 2 asks it of the read-out that decides, and `smoke.txt` part 4 is the untuned model | leave it at `smoke.txt`'s untuned evidence |
| `latency.py` times the seeds-1-to-80 median prompt | keeps the promise about the selection half, and 478 tokens is already timed (`precheck.txt` part 1) | the seeds-81-to-100 median prompt, an undeclared use of the selection half |
| `agree.py` runs after `ana.py` | the ordering removes the opportunity for a tuned model's agreement to be seen before the score decides, at no cost | `agree.py` third, which leaks nothing by construction but puts the number first |
| the 0.5273 baseline by subtracting two generated files, not by a new count | prediction 5 needs a bar fixed before any read-out, and `rows.txt` part 1's guarantee that the row list is cut at seed 80 before anything is counted is worth more than a count | a seeds-81-to-100 part in `rows.py`, which would buy the same number and spend that guarantee |
| "close" = 297 ms | 1.685586 ms/poll x 176.220 polls, both from `../22/weight.txt` | the 647.8 ms the per-stream weight range gives; or a paired bootstrap interval, reported but not the rule |
| carry b to ticket 12 if a and b are close | b's filter is the one whose cross-labelling agreement was measured, and relabelling changes b's rows and not a's | carry a, for 2.49x more rows |
| time the latency with the adapter attached, unmerged | it is the read-out being scored | `merge_and_unload`, which changes the arithmetic |
| no export of training rows to the host | `ROOT/repo` already carries `evidence/22`'s code, so the seed guards run there (`precheck.txt` part 0b) | export plus a sha256, which would add a second definition of the row filter |
| `rows.py` runs locally and reads only the label file | a reader can re-run it in seconds; no card, no store, no model | compute the counts inside `tune.py` on the host, where no reader can re-check them cheaply |

## Deliverables (this directory), and what stays out of the repository

**Before any tuned model is scored** (this step): `spec.md`, `rows.py`, `rows.txt`. Already here:
`hpc_setup.sh`, `fetch_weights.sh`, `push_inputs.sh`, `parity.py`, `parity.txt`, `prefill.py`, `smoke.py`,
`smoke.txt`, `precheck.py`, `precheck.txt`, `hpc_freeze.txt`, `RUN.txt`.

**Then, in this order:**

Every script that reads out a model or runs a simulation runs on **`hpc-debian`**, the model's own host: the
tuning because the user put it there (`決まったこと` 2 and 7), and **the replay scoring because the user put
that there too** (`決まったこと` 8: 点の計算（なぞり直し）は `hpc-debian` で、モデルの隣で回す). The column is
written out per file rather than left to the wall-time arithmetic to imply.

| file | what it is | where it runs | output |
|---|---|---|---|
| `tune.py` | the recipe above, both candidates | `hpc-debian` | `tune.txt` (the per-epoch loss, the held-out cross-entropy, the kept epoch, the 205-module assertion, the adapter-dtype assertion, the repeat check) |
| `decider.py` | L as a port decider: the send-time convention keyed on `(node, neighbor)`, the fresh-per-simulation memory, the read-out, the tie rule, the no-node-name assertion on the rendered text. Imported, no `main` | `hpc-debian` | -- |
| `eval.py` | the 7 deciders x 20 seeds x 40 streams through the port, with the reference read-out gate and the repeat check | `hpc-debian` (`決まったこと` 8) | `eval.csv` (one row per decider, seed and stream, with `score.py`'s terms), `eval.txt` |
| `ana.py` | the selecting score, the frontier gain with its 95 % interval, the a-against-b decision against the 297 ms band, the paired bootstrap | `hpc-debian` (`決まったこと` 8); it reads `eval.csv` only, so a reader can re-run it anywhere | `ana.txt` |
| `agree.py` | the label-agreement numbers on seeds 81 to 100, three model deciders, against the pre-registered 0.5273 baseline | `hpc-debian` | `agree.txt` |
| `latency.py` | the pass time with the chosen candidate's kept epoch attached, beside 100 ms | `hpc-debian` | `latency.txt` |
| `README.md`, `RUN.txt` | the result in Japanese, and the commands with their real output | local | -- |

**`agree.py` runs after `ana.py`** -- worker's choice. Every rule that a tuned model's agreement could
influence is already fixed in this file, so there is no leak by construction; running the score first removes
the opportunity as well, and costs nothing because the two read different things.

Every one of these scripts shows each of its own checks failing on a mutated input, in its own output or in
`RUN.txt`, and a check that cannot be made to fail is named as such rather than counted.

**What stays out of the repository**: the base weights (`ROOT/models`, 10.2 GB), every adapter
(`ROOT/adapters/{a,b}`, and the epochs not kept are deleted when the epoch is chosen), the label files
(`/home/lab_shared/semindex-learn/ev22/`, pinned by `../22/labels.sha256`), the host's venv, and every
intermediate log under ROOT. **No adapter or weight file enters the repository**, and the adapter kept for
ticket 12 lives under ROOT until that ticket takes it.

## Amendments after the freeze, written before any score on seeds 81 to 100 exists (2026-11-24)

Everything above this heading is the plan as frozen. The scripts were then written, both candidates were
tuned once, and the pipeline was run in its dry mode on seeds 1 to 80 (`RUN.txt` sections 17 to 19). That
work found the six things below. They are written here, and committed, before `eval.py` is started in its
real mode, so that a reader can tell what was decided with no selection-half score in hand. No setting of
the recipe, no row set, no wording, no read-out rule and no statistic above is changed by this section.

**A1. The seam check of `eval.py` reads a tuning seed in both modes.** The reviews asked for a check that
the text rendered from a label row equals the text rendered from the live engine's observation of the same
decision, and the fix placed it in `eval.py`'s first part on the first simulated seed, which is seed 81 in
a real run. That would have rendered one label row of seed 81 before any score, against the sentence above
("The one place the selection half is read before any score exists is the class-share baseline"). The seam
is about a JSON row against a live object and not about which seed, so the check is pinned to seed 1 in
both modes and the sentence stays true as written.

**A2. One thing did touch seed 81 during the build, and it is named here.** A scratch probe that is not
in the repository called `label.run_dict(0.2, 81)` once to time the call. That builds the outage timeline
and the record schedule of seed 81. No simulation was run on it, no model was read out, and no label row
was opened (`RUN.txt` section 18). The list of what read the selection half before any score is therefore
two items long: the class-share baseline, and this one call.

**A3. The tuning is run a second time under the final code, and the second run's adapters are the
candidates.** `tune.txt` of the first run names no digest of the code that wrote the adapters, and
`decider.py` was edited after it, so the first run's adapters cannot be tied to a code state. The values
do reproduce through the current code (candidate a's held-out cross-entropy to six decimals, both repeat
triples bit for bit), which is evidence about values and not a recipe. A second run is the only clean
repair, and this is the last moment it can be made without a selection-half score in hand. The rule, fixed
before the second run starts:

- `tune.py` changes in four places and in nothing that reaches a weight: it prints the sha256 of itself
  and of every module it imports from this step; it calls `decider.check_roles` instead of carrying a third
  copy of the seed ranges; it reads the weight decay off the optimiser and stops unless it is 1e-2, instead
  of printing a typed string; and it keeps each epoch's per-row held-out cross-entropy long enough to print
  the paired difference between the kept epoch and each other epoch with its standard error. The keep rule
  is untouched and the new numbers choose nothing.
- **The second run's adapters are the candidates whatever its numbers are.** No comparison of the two runs
  selects one. The first run's adapters are moved aside to `ROOT/adapters-run1` (renamed, not deleted) and
  its log is kept in this directory as `tune_run1.txt`.
- The record says whether the two runs' weight files are byte-identical. If they are, the recipe is shown
  reproducible on this host. If they are not, the differences between the two runs' held-out numbers are
  reported as the run-to-run spread of the recipe at n = 2, and every dry run is repeated on the second
  run's adapters, because every dry number on record was measured on the first run's.

**A4. Threat 16: candidate b answers differently outside its own row filter, and the replay is outside
it.** b was tuned and held out on rows its filter kept. On the first run's kept epoch b answers the 3.695 s
word on 0.2606 of its own held-out rows (148 of 568), on 0.4625 of all labelled rows of seeds 71 to 80 (659
of 1425), on 0.5963 of the rows its filter excludes (511 of 857, by subtraction), and on 0.5259 of its
simulated decisions on seeds 1 and 2 (335 of 637). It never answers the 6.0 s word anywhere it was
measured. **What a reader may therefore not conclude: that b's held-out numbers describe the decider that
is scored.** b's replay score is a score on a population closer to the complement of its filter than to
its filter. Threat 12 names the gap in b's measured agreement and Threat 13 names the `history` shift;
neither names the shift the filter itself makes.

**A5. Threat 17: candidate a's kept epoch is not distinguished from its other two.** The three held-out
cross-entropies of the first run are 0.898804, 0.895778 and 0.893928. The margin between the kept epoch
and the runner-up is 0.001850, and the standard error of the kept epoch's own mean over the same 1425 rows
is 0.015423, 8.34 times that margin. That is the unpaired scale; the paired one is what A3's fourth change
prints. The three epochs are visibly different deciders (their answers over the 1425 rows are 28/84/1313,
802/0/623 and 206/145/1074), so "candidate a" is one of three deciders the rule could not tell apart on
the quantity it uses. **What a reader may therefore not conclude: that a's score is the score of "every
labelled row" as a training set.** It is the score of one epoch of it.

**A6. The cost of the real evaluation, measured, in place of the plan's arithmetic.** A decision with an
adapter attached takes about 70 ms at batch 1 against the untuned 44 ms, so the adapter costs about 26 ms
(`latency_dry.txt`); the plan priced every decision at the 43.02 ms untuned pass. Resolution-time
decisions per run on seeds 1 and 2 were 280.0 for the untuned model, 125.8 for a and 159.2 for b, against
the plan's single anchor of 160.7. The real evaluation is therefore about 452 000 forward passes and about
2.1 h at four workers, measured at 59.2 passes per second, in place of the table's 385 704 passes and
84.7 min. The four-worker gate passed on the dry run (2304 logits bit-identical to the single-process
reference), so four workers is what the real run uses unless its own gate refuses. The single-worker
fallback was never run and its 8.0 h is an inference.

**A7. Threat 18: the recipe does not reproduce on this host, so each candidate is one draw of it.** Written
after the second tuning run and before any score on seeds 81 to 100 (`RUN.txt` sections 20 and 21). The two
runs of the same recipe on the same rows wrote different weight files for both candidates. Neither run
asked for a deterministic kernel, which is the recipe as frozen. The first printed number that differs is
candidate b's last training loss of epoch 1; the first loss is equal in both runs. What moved on the 71 to
80 slice, first run then second run (`tune_run1.txt`, `tune.txt`):

| | kept epoch | held-out cross-entropy | agreement | baseline |
|---|---|---|---|---|
| a | 3, then 2 | 0.893928, then 0.888765 | 777, then 757 of 1425 (0.5453, 0.5312) | 0.5102 |
| b | 3, then 3 | 0.622174, then 0.638168 | 408, then 406 of 568 (0.7183, 0.7148) | 0.6831 |

Candidate a's gain over its baseline went from 0.035 to 0.021 between two runs of one recipe, so the
spread of the recipe is of the size of the gain. The second run's paired figures put A5 in its right
denominator: a's kept epoch is 0.001873 below its runner-up with a paired standard error of 0.005268
(ratio 0.356), so a's epoch is still not distinguished; b's is (0.029425 with 0.010900, ratio 2.699).
As A3 fixed in advance, the second run's adapters are the candidates and nothing above selected them.
**What a reader may therefore not conclude: that a difference between a's and b's replay scores is the
difference between their row sets.** It carries one draw of tuning noise per candidate, and that noise is
not measured on the scale of the score by anything in the plan. The 297 ms rule is applied as written;
its verdict is read with this beside it. One measurement could put a number on that noise afterwards
without choosing anything: scoring the first run's two adapters (`ROOT/adapters-run1`, fixed files) on the
same seeds and streams. It is not part of this plan and is run only if the user asks for it.

**A8. What A1 and A2 do not cover.** Both are about reads of the selection half *before any score*.
`eval.py`'s last part, where its checks are shown failing on mutated inputs, takes the first simulated
seed, which is 81 in a real run: it builds that seed's run dict once more and starts one simulation that
the port refuses. That happens after every score of the run is written and reaches none of them.

**A9. A6's cost, re-measured on the second run's adapters.** Resolution-time decisions per run on seeds 1
and 2: untuned 280.0 (unchanged), a 118.5, b 147.8. The real evaluation is about 437 000 forward passes
and about 2.0 h at four workers (60.4 passes per second); the single-worker inference is 7.6 h. The
four-worker gate passed again on these adapters (2304 logits bit-identical to the reference).
