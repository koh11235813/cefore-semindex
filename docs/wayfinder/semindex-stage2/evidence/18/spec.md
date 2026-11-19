# evidence/18 spec: how much gain foresight can take over the fixed-interval frontier, and how much of it a formula already takes

Written 2026-11-19, before any number of this step was seen. Every threshold, every record set, every
search path and every reading rule below is fixed by this file and may not be changed once a number of
this step has been read.

This file replaces the draft reviewed on 2026-11-19. The review's eight findings and its scoping defect of
C are all applied here, with one partial rejection recorded under "Where the review was corrected".

## Purpose

evidence/13 to 16 compared named rules against the seven-arm fixed-interval frontier and found gains of a
few points, with intervals that span zero on 20 testbed seeds. The step after this one is a three-way
comparison of decision makers for the state-poll interval: a distilled small LLM, an ML model trained on
the logged decisions, and formula rules. This step runs on the replay alone, so that the three-way
comparison can be commissioned or cancelled with a reason.

The question this step answers is how much gain foresight can take over the fixed-interval frontier, and
how much of that gain a formula already takes.

The step produces three numbers per cell and workload, on the same axes and with the same scorer as
evidence/14 to 16.

- **C, the structural ceiling.** The delay no policy can go below, because a record cannot cross a link
  that is down and cannot be fetched before it exists. Computed from the exogenous inputs only, with no
  poll budget at all. C is an upper bound on the gain of any policy at any budget, under the assumptions
  below. It is expected to be loose. It is reported as an anchor and is never the headline.
- **V, the clairvoyant point.** A schedule computed with full knowledge of record creation times and of the
  link timeline, spending a chosen poll budget where it shortens delay most, and then run through the
  replay like any other rule. V is a lower bound on what a policy with perfect foresight could reach. V
  bounds nothing from above. A gain at V is a gain that exists. A small V leaves open that a better
  clairvoyant schedule exists.
- **F, the formula point.** The best member of an enlarged rule family, found by a search that is
  cross-fitted over the seeds. F is a lower bound on what a formula can reach. Its family is a causal
  mapping from the decision state the agent already logs, so it also serves as the computable stand-in for
  an information-limited oracle.

The decision this serves is whether it is worth training a policy for this workload. The reading rule below
turns C, V and F into one of four named outcomes, one of which is "undecided".

## What this step cannot show

- Where a learned causal policy lands between F and V. Nothing here trains anything.
- That V is reachable without foresight. V reads the future.
- How far V is from the clairvoyant optimum. That gap is not measured and no size is claimed for it.
- Anything about budgets denser than V's reachable target set (see below). At those budgets C is the only
  number this step has, and C is loose.
- Anything about workload WV through V. The reachable target set is empty there, so V is not computed
  under WV in any cell, and C alone speaks for WV.
- No number of this step has a testbed measurement behind it. The schedules V produces lie far outside
  every schedule the replay was validated on, so the 6 % fidelity of evidence/13 and 16 does not transfer
  to V. It transfers to F only for family members whose intervals stay at or above 2.2 s, which is why the
  headline F is restricted to those members.
- Nothing is claimed about workloads, outage shapes or values of p outside the cells below.

## Simulator

`../13/replay.py` and `../15/onoff.py` are imported and must not be modified. Everything this step adds
lives in this directory, in a subclass of `../15/onoff.py`'s `Engine`, which is itself a subclass of
`../13/replay.py`'s `_Engine`. The subclass changes three things and nothing else.

### 1. A deep copy of the rule dict, per engine and per node

`_Engine.__init__` (`../13/replay.py` lines 714-720) passes one and the same `rule` object into `Node(...)`
for all five nodes, so the dict is shared network wide. `../14/workload.py` lines 213-232 builds that dict
once per (donor, p, seed) for a rule-kind member and reuses the object across workloads and replicates, so
an in-place write would leak into later simulations as well.

The subclass therefore copies the dict in `__init__`, once per node, so that a write for one node's pair
cannot reach another node and cannot survive the simulation.

### 2. `resolve`, to place a poll at an arbitrary instant and to carry F's new coordinates

Setting `nd.ival[nb]` before calling the base method does not place a poll after an answered reply: the
base method overwrites `ival` from the rule, with `min(ival * m, i_max)` when the reply was empty and
`i_min` otherwise (`../13/replay.py` lines 745-759). The subclass therefore writes the target gap into the
rule dict's own fields just before calling the base method: `i_min = i_max = g`, `m` large, and `jf` 0 so
that `applied` equals the interval exactly. The base method then computes `due = max(sent_at + g*1000, t)`
and arms the timer itself, which is the behaviour evidence/15 and 16 measured.

The base method reads those fields only inside its `status == "answered"` branch. On a timeout it leaves
`nd.ival[nb]` alone and computes `applied` from the stale `ival`. The subclass therefore also writes
`nd.ival[nb] = g` on every live resolution whatever the status. A feasibility script in the scratchpad
shows what is at stake: driving one pair to a sequence of varying target gaps, with the rule dict written
and `ival` left alone, put 15 of 15 post-timeout gaps off target by more than 0.11 s; writing `ival` as
well put 0 of 13 off target.

Three further rules the subclass follows on every live resolution of every pair, so that no pair runs on
what another pair wrote.

- All five fields the step uses (`i_min`, `i_max`, `m`, `jf`, and `ival`) are written. A pair left unset
  runs on the previous pair's values. The scratchpad script shows the failure directly: with only one pair
  driven, the other pair of the same node was off its fixed 3.695 s interval on 18 of 18 gaps.
- `Node.__init__` reads `rule["i_min"]` once, for the first poll of each pair. The subclass sets that
  initial value to 10.043 s, the frontier's longest arm. A value below the 2 s interest lifetime would let
  a lost first poll hand control to the provisional timer, which is the trap recorded under "Assumptions".
- F's two new coordinates ride on the same override. On a timeout the subclass sets `nd.ival[nb]` to
  `i_timeout`, which generalises `../15/onoff.py`'s `on_timeout`. When the subclass's own per-pair counter
  of consecutive empty replies has reached `k_cap`, it writes `m = 1` into that node's copy of the dict for
  that resolution and restores the member's `m` on every other resolution. The base `consec` counter counts
  timeouts, so the empty counter is the subclass's own.

**The override is a pure pass-through** when the V flag is off, `i_timeout` is keep and `k_cap` is
unbounded. That condition is what turns the W0 and identity checks below into real checks.

### 3. `tick_lag`, for V only

The base class draws every tick lag from one `sim_rng` in event order, so changing one pair's schedule
changes the draw order and shifts other pairs' lags by up to 100 ms. evidence/13's README records the same
of its rules. V solves hop h's schedule against the realised trajectory of hop h-1, so hops below h must
stay put when hop h changes.

Under V, and only under V, the subclass draws each lag from **one cached generator per (rep_seed, node,
neighbor)**, created once and drawn in poll order. A fresh `random.Random(...)` per call would return the
same value every call, which would hand the DP a constant lag it could count on and make the placement
check pass for the wrong reason.

The flag is set by the V driver and is off in every other path, so F, the seven arms and the evidence/15
rules run on the base class's own stream, and evidence/15's byte-exact W0 check and `../13/identity.py` are
untouched. V and the arms then sit on different streams of the same distribution, which is uniform over 0
to 100 ms per poll either way. Cross-rule pairing of the lag values was already lost at the first interval
change, as evidence/13's README records.

Nothing else is overridden. The jitter draw, the clamp, the reschedule line, the fetch plan, the backoff
chain and the scorer are the base classes' own code.

## Assumptions, stated before the numbers

- **What C rests on.** A record's arrival cannot be earlier than the outage-gated zero-wait propagation of
  its creation time. This holds by construction: a packet put on a down link is lost (`delivers`), and a
  record cannot be fetched before it is advertised.
- **More polls can lengthen delay, so no bound may be built on "poll as fast as possible".** A reply that
  lands just before a link drops starts a fetch into the down link. That fetch times out after 2 s and the
  record enters the 2 / 4 / 8 / 16 s backoff chain, which keeps running after the link returns
  (`fail_fetch`). Polling harder near an outage boundary buys fetch backoff. This is a reason the ceiling
  may be low, a reason V's DP is allowed to leave gaps, and the reason no step of this family may report
  "the fastest arm" as a ceiling.
- **The feasible instant set has three parts, and the first of them is weaker than the draft claimed.**
  1. After a **delivered** poll the resolution lands at `sent + POLL_RTT_MS`, which is 1 ms, so
     `due = max(sent + g*1000, t)` leaves the target gap untouched for any `g` above 1 ms. There is no 2 s
     floor at all after an answered poll. Measured in the scratchpad: driving one pair to 0.6 s and to
     1.2 s gaps gave 0 of 123 and 0 of 65 answered-gaps off target, with a minimum gap of 0.600 s.
  2. After a **timed-out** poll the resolution lands at `sent + 2000 ms + a tick lag of up to 100 ms`, so
     the clamp pushes the next poll of that pair to about `sent + 2.0` to `2.1 s` at the earliest. V's
     candidate set therefore carries a floor of 2.1 s after a poll V expects to be lost, and no floor after
     a poll V expects to be answered. In `s5b10p043` p 0 no poll times out, so no floor applies there.
  3. The **provisional timer** takes control when the gap in force at send is below 2 s and that poll is
     then lost: the timer fires before the timeout resolves, the next poll goes out at `sent + gap`, and
     `resolve` returns on a replaced `pending`, so V's chosen instant is silently dropped. A clairvoyant DP
     places no poll inside an outage, and a tick lag of up to 100 ms can still push a poll placed just
     before a `link_down` into it. V's candidate set therefore excludes the 100 ms before every `link_down`
     of that pair, and the placement check below is what catches the residue.
- **A link_up instant is not itself feasible.** `delivers` reads the link state set by the last timeline
  event **strictly before** `t_send` (`../13/replay.py` lines 480-511), so a poll sent in the exact
  millisecond of a `link_up` is judged against the down state and is lost. Every `link_up` candidate is
  therefore `link_up + 1 ms`.
- **The hop-to-hop causality V's solve order needs holds, and it depends on `jf` 0.** `plan_fetches`
  headroom and `in_flight` are per neighbour, serving a poll never writes the holder's store, and for each
  origin the downstream node's store is contained in the upstream node's, so no reverse fetch path exists.
  The per-node `jitter_rng` is shared across a node's neighbours and would reorder under a schedule change.
  With `jf` 0 under V the drawn value is unused, so the causality survives. V therefore runs with `jf` 0,
  and this is a property the V driver establishes. The topology on its own does not give it.
- **The reported number is always the realised simulation's score.** The DP picks instants. The replay
  decides what happens.

## Bounds, in full

### C, the structural ceiling

For each record, propagate its creation time along its chain with zero discovery wait and zero fetch wait,
gated only by the link timeline: at each hop, wait until that link is up, then add the transit constant the
engine uses (`POLL_RTT_MS + FETCH_RTT_MS`). This gives `D_min(j)`, a per-record arrival floor that holds
for every policy at every budget.

**The record set is fixed across policies**, which the draft did not do. `score` takes the delay as
`lat4 / n4` over uncensored records only, so a policy that censors more than the arm scores a faster subset
and its gain can rise above a C computed on the arm's own subset. For C's table only:

- the record set is every record created in the scorer's window and at or before t0+72 s **whose floor
  reaches the receiver inside the window**, that is `created + D_min <= we` with `we` the scorer's window
  end. A record outside that set is censored under every policy including a clairvoyant one, and the step
  reports how many records it drops, per cell and workload;
- every policy's delay in C's table is recomputed from the same simulations over that same set, with a
  censored record counted at `we - created`. Record by record, `D_min(j)` is then at or below the value
  counted for every policy, so C's mean is a floor on every policy's mean over one shared set;
- the headline frontier that V and F are read against is unchanged and stays evidence/14's uncensored one.
  C's censor-inclusive line is used for C's table alone, and both lines are printed.

Reported as: at each arm of the frontier and at each rule of evidence/15 and 16, the gain
`1 - mean(D_min) / mean(D_cens(policy))`. One table. No simulation. Cost: seconds.

C is budget-free, so it is loose by exactly the amount a finite budget costs. It is here because it is
nearly free and because it is the only strict bound in the step.

### V, the clairvoyant point

Each origin's records travel one way along a chain of four directed pairs, and the two chains are disjoint:
h0's records use h1 polls h0, h2 polls h1, h3 polls h2, h4 polls h3, and h4's records use the four opposite
pairs. A node's store is not changed by the polls of the node downstream of it, so the chain is causal from
hop 1 to hop 4 and hop h's schedule moves nothing at hops below h.

That gives a solve order with no fixed point.

1. Hop 1's input is exogenous, being the record creation times and the link timeline, both known. Solve hop
   1's schedule by the DP below, simulate, and read the realised instants at which each record became
   available at the hop-1 receiver.
2. Solve hop 2 against those realised instants, simulate, read hop 2's realised instants. Then hop 3, then
   hop 4.

Both directions are solved in the same pass, because the pairs are disjoint. Four passes and four
simulations per budget target and per (cell, workload, seed, replicate).

#### The reachable target set, fixed now

V can place a poll only at a candidate instant, and its candidate set per pair is the availability instants
at the holder plus the `link_up + 1 ms` instants of that pair. Counted in the scratchpad over all 20 seeds
of every cell and workload, through `../15/onoff.py`'s own generators:

| workload | candidates per run, min over 20 seeds | median |
|---|---|---|
| W0 | 240 (p 0), 254 (p 0.2), 274 (p 0.4) | 240 / 260 / 280 |
| WO | 176 (p 0), 186 (p 0.2), 204 (p 0.4) | 188 / 204 / 228 |
| WV | 52 (p 0), 108 (p 0.2), 134 (p 0.4) | 204 / 236 / 250 |

The arms' realised polls per run are 1744, 1128, 683, 424, 256, 159 and 96 (`../14/workload_cells.csv`).

**An arm is a reachable target for V only when its polls per run is at or below 70 % of the minimum
candidate count over the 20 seeds of that cell and workload.** The minimum is used because V has to be
computable on every seed the bootstrap resamples. The 70 % is a judgment fixed now and before any number of
this step: at 100 % the DP would have to take every candidate, so V would degenerate into "poll at every
availability instant", which carries no scheduling information. V would then report the budget and would
say nothing about the foresight. For every V solve the step reports the fraction of candidates the DP actually took, so a reader
can see how tight the constraint was.

The resulting target set, fixed here:

| workload | reachable arms | V defined |
|---|---|---|
| W0, all three cells | 10.043 s (96 polls), 6.0 s (159 polls) | yes |
| WO, all three cells | 10.043 s (96 polls) | yes |
| WV, all three cells | none | no |

So V is not computed under WV, and the 0.5, 0.8, 1.359, 2.2 and 3.695 s arms have no V in any cell.
**C alone speaks for those budgets and for WV**, and C is loose there. That looseness is a limit of the
step. No reader should take it as a result.

#### The per-hop problem

Given availability instants `a_1 <= ... <= a_n` at the holder, the link timeline of that pair, and a price
`lambda` per poll, choose poll instants to minimise
`sum_j (first delivered poll at or after a_j) - a_j  +  lambda * (number of polls)`.
The candidate set is as above, each candidate pushed forward to the first feasible instant under the three
parts of the feasible instant set. The DP runs over the chosen candidates with the last chosen candidate as
its state, and it is **optimal over that candidate set**. The continuous optimum can be better, because the
feasible set of a later poll depends on whether the previous poll was delivered.

The DP knows the link timeline, so in the transition from candidate j to candidate i it asks whether a poll
at j is delivered and applies the 2.1 s floor only when it is lost. The candidate set already excludes the
down periods and the 100 ms before every `link_down`, so that floor should bind rarely. The step reports
how often it bound, per cell, workload and pair.

Measured in the scratchpad at the real sizes, about 35 candidates and 30 availability instants per pair:
0.21 ms per pair and 1.7 ms per run-solve of eight pairs. At a deliberately pessimistic 130 candidates it
is 2.2 ms per pair and 17.7 ms per run-solve. Either way the DP is below one replay simulation, which is
22 ms. The DP is still charged in the cost table and still timed on a real cell before the cell list runs,
as the check below requires.

#### What V is and is not

V is optimal per hop over the candidate set, given the realised upstream. V is not optimal for the four-hop
objective: minimising hop h's own mean wait differs from minimising the delay at hop 4, because a record
that arrives a little later at hop h may still catch the same hop h+1 poll. The realised schedule also
differs from the solved one by the tick lag and by the clamp. **V is a lower bound on the clairvoyant
optimum. V bounds nothing from above.** It is reported as "a gain this large is reachable with knowledge of
the future".

#### Budget

`lambda` is chosen by bisection, per (cell, workload) and per target, so that V's realised polls per run
land within 2 % of the target's realised polls per run. At most 8 bisection steps.

The bisection seeds are **inside the fold V is paired with and outside the fold it is scored on**, so that
V is cross-fitted exactly as F is: the V used against seeds 11-20 is bisected on seeds 1-5, and the V used
against seeds 1-10 is bisected on seeds 11-15. There is then no in-sample tuning of V's budget on any seed
it is scored on, and no leak statement is needed.

Targets the bisection fails to reach are reported as reached or not, with the realised poll count. A target
that is not reached on every seed is dropped from the reading rule for that cell and workload, and the
drop is printed.

### F, the formula point

The family is a finite parametric extension of the shape evidence/13 to 16 used, chosen so that every
member is a mapping from the decision state the agent already logs, being consecutive timeouts,
consecutive empty replies, and whether the last reply was empty.

| coordinate | values |
|---|---|
| `i_min` | the seven arms' intervals |
| `m` | 1.2, 1.5, 2, 3 |
| `i_max` | the seven arms' intervals, those at or above `i_min` |
| `i_timeout` | keep, or one of the seven arms' intervals, applied when the newest poll times out |
| `k_cap` | 1, 2, 3, unbounded: the number of consecutive empty replies after which `m` stops applying |

`jitter_frac` is 0.1 throughout, as every earlier step.

#### Two F numbers, and only one of them is read

The spec excludes gaps below 2 s from V, because they enter the provisional-timer and unresolved-poll
branches that evidence/13 lists as read from the source and never seen in a log, and the scratchpad
confirms the mechanism: driving a pair to 0.6 s and 1.2 s gaps produced zero timed-out resolutions at all,
because the provisional timer always replaced the pending poll first. F becomes the baseline of the later
three-way comparison, so F may not be allowed the branches V is forbidden.

- **F_restricted is the headline.** `i_min` and `i_timeout` range only over values at or above 2.2 s, so
  `i_min` has four values and `i_timeout` has five including keep.
- **F_open is the unrestricted search**, over all seven intervals on both coordinates. It is reported beside
  F_restricted, with its selected member and both halves' gains. **F_open never enters the reading rule.**
  It is there so that a reader can see what the restriction cost.

#### The search path, fixed now

A grid over all five coordinates does not fit the cost budget, so the search is deterministic coordinate
ascent. Start at T37r (`i_min` 3.695, `m` 2, `i_max` 10.043, `i_timeout` 3.695, `k_cap` unbounded), sweep
the coordinates in the order of the table, take the best value of each coordinate before moving to the
next, and run three full sweeps. Ties go to the value earliest in the list. The procedure needs no search
seed. One sweep is 21 evaluations for F_restricted and 30 for F_open, so about 60 and about 90 distinct
members are evaluated per fold.

F bounds from below the best mapping inside **that** family and along **that** path. It does not bound the
best formula.

#### Cross-fitting, fixed now

Seeds 1-20 are split into fold A, seeds 1-10, and fold B, seeds 11-20.

1. Select the best member on fold A by its early-record gain. Score that member on fold B.
2. Select the best member on fold B. Score that member on fold A.
3. **Pool the 20 out-of-sample per-seed values.** That pooled 20-seed value is F, and it is the only F the
   reading rule reads.

The bootstrap resamples the 20 seeds and recomputes, for each resample, the arms, the selected members and
the pooled value, so the interval carries the selection. V is paired with F on the same 20 seeds. The
`sqrt(2)` penalty a 10-seed scoring half would have carried does not arise.

The in-sample optimism is reported as the mean selection-half gain minus the pooled out-of-sample gain.
When the two folds select different members, both members are printed with their poll counts, and both
poll counts are reported, because `V_at_F` depends on which one is meant (see the reading rule).

Only 20 outage schedules exist per cell, because the link timeline comes from the stored runs'
`results.json`. The generator in `../10/replay_common.py` produces *planned* down intervals while
`../13/replay.py` builds its timeline from the measured `actual_at`, and it supplies neither `agent_start`
nor `t0_ms`, so it is not used. That limit is a property of the inputs and is stated wherever an F interval
is printed.

## Workloads and cells

As evidence/15, unchanged, so that every number of this step sits beside that step's table.

- Workloads: W0 (the logged creation times), WO (on/off regular), WV (on/off variable), generated by
  `../15/onoff.py`'s own generators, which this step imports.
- Cells: `s6b10p043` p 0.2 and p 0.4 (outages 8 to 12 s), `s5b10p043` p 0 (no outage).
- 20 seeds, Mode B, 5 replicates.

## Comparison

As `../14/spec.md` and `../15/spec.md`: gain against the seven fixed arms' frontier in log-log over
`polls_sent`, bootstrap over seeds (2000 resamples, `default_rng(20261117)`), the 1e-9 zero rule, per-axis
"outside", the censoring flag.

**The headline is the early-record gain**, over records created at or before t0+72 s, as evidence/14 and 15
fixed it. This matters more here than in earlier steps: V and F both aim at low delay and will censor close
to nothing, while the 10.043 s arm censors 3 to 5 % at p 0.4, and evidence/15 measured that this bias works
in favour of the low-censoring rule. A ceiling inflated by a scoring artefact is no ceiling. The all-record
gain is reported beside the headline, with the direction of the residual bias.

## Reading rule, fixed now

### What the thresholds are, and how big the step's own error is

evidence/16 measured a half width of about 9 points for one rule's gain with 20 testbed seeds, and about 11
points for the difference between two rules' gains. Assuming seed variance dominates and scales as
`1/sqrt(n)`, 40 seeds give `9/sqrt(2) = 6.4` for a single gain and `11/sqrt(2) = 7.8` for a difference. The
step uses **H = 6.5 points for a single gain** and **Hd = 7.8 points for a difference of two gains**. That
scaling holds only if the frontier arms are run on 40 seeds too, so "40 seeds" here means 40 seeds per
stage, the two arms the line is drawn through included.

**The replay's own error is as large as H.** evidence/16 measured the replay's fidelity on delay at 6 %
(+5.6 % on T37r in exactly the headline cell), and gain is `1 - delay/line`, so a 6 % delay error is about
6 gain points. H is 6.5 points. On top of that, this step's numbers come from 20 replay seeds, whose
intervals are wider than the 40-testbed-seed widths H and Hd were derived from.

**Where the 6 point replay margin is applied, and where it is left out.** Outcome 1 below reads V's
absolute level, so the replay's systematic error enters it in full and the outcome carries the 6 point
margin. Outcomes 2 and 3 read `V_at_F - F`, which is a paired difference of two runs of the same engine on
the same seeds, where a systematic engine error partly cancels. Those outcomes therefore carry no margin
beyond Hd. **That cancellation is asserted and is not verified for V's schedules**, which lie outside
everything evidence/16 measured, so the size of the residual systematic error in `V_at_F - F` is unknown.
A reader who wants symmetry can read outcome 3 at `Hd + 6` instead, and the step prints the lower end of
the interval so that reading is available without rerunning anything.

**The step is therefore a pointer.** It is too coarse to serve as a measurement of the headroom. It can
say "clearly large headroom" and it can
say "clearly none". Most of the middle is undecided. "Undecided" is a named outcome below, and it is a
legitimate result of the run.

### What enters the rule

Bootstrap **intervals** of the early-record gain, with the point values printed beside them. Nothing below
reads a point value alone.

- **V_max** is the largest of V's gains over the reachable target set of that cell and workload, over all
  20 seeds. Taking a maximum over several noisy gains is biased upward, which is the conservative direction
  for outcome 1 and is stated as such. The number of targets the maximum is taken over is printed.
- **V_at_F** is V's gain at F's own budget, paired with F seed by seed over the same 20 seeds. F's pooled
  realised polls per run is added to V's bisection targets for this purpose, with the same 2 % tolerance
  every other target gets. When the bisection lands between 2 % and 10 % of F's count, V_at_F is reported
  with the realised gap printed beside it. Beyond 10 % the target counts as not reached. **When F's poll count is above 70 % of the
  minimum candidate count, V cannot be placed at F's budget and `V_at_F` does not exist.** When the two
  folds select members with different poll counts, both are tried and `V_at_F` exists only if both are
  reachable.

Read per cell and workload, one at a time, with no outcome carried across them. The headline cell is
`s6b10p043` p 0.4 with W0, because that is the only cell and workload with a testbed measurement behind the
frontier (evidence/16).

### The four outcomes

1. **Clearly none.** The **upper** end of V_max's interval is at or below `H - 6` points, that is at or
   below 0.5 points. Claim: the best clairvoyant schedule this step could construct shows no gain this
   testbed can detect in this cell, and the margin is wide enough that the replay's own 6 % band cannot
   have produced it on its own. The ceiling is small only if V is close to the clairvoyant optimum, which
   this step asserts and does not show, so C is printed beside it. **For the three-way comparison:** do not
   commission it for this workload. Change the workload or the cell first. Nothing is claimed about which
   of the distilled LLM, the ML model and the formula would win, because there is nothing for any of them
   to win.
2. **The formula already takes it.** The **lower** end of V_max's interval is above H, and the **upper**
   end of `V_at_F - F`'s paired interval is at or below Hd. Claim: a formula already takes the gain that
   exists at that budget, within what the testbed could resolve. **For the three-way comparison:** the
   distilled LLM and the ML model are predicted to tie with F, within Hd. Running the comparison is then a
   test of that prediction. It would be a poor way to look for a winner, and tuning the formula is then
   the cheaper move.
3. **Clearly large headroom.** The **lower** end of V_max's interval is above H, and the **lower** end of
   `V_at_F - F`'s paired interval is above Hd. Claim: headroom exists at that budget that this formula
   family does not take, and V is its best case. **For the three-way comparison:** commission it, score
   every arm of it against F, which replaces the fixed arms as the baseline, and read `V_at_F - F` as the size of the prize
   the LLM and the ML model are competing for.
4. **Undecided.** Everything else, including each of these, which are reported by name:
   - V_max's interval straddles H, or straddles `H - 6`;
   - `V_at_F - F`'s interval straddles Hd;
   - `V_at_F` does not exist because F's budget is outside V's reachable set;
   - V is not defined for that workload, which is every cell under WV.
   **For the three-way comparison:** this step gives it no steer. Nothing is claimed about any of the three
   decision makers. The honest report is that the replay cannot separate them at this cell, and that a
   denser testbed or a tighter clairvoyant construction is what would.

### Reported in every case, as a separate line

Whether F's pooled 20-seed interval lies entirely above zero and F's point is at or above H. When it does,
the formula family already reaches a detectable gain on its own, so the later three-way comparison takes F
as its baseline. F_open's number is printed on the same line and is labelled as not entering any outcome.

## Checks required

Each check below is a gate. A check that fails stops the step at that point. The failure is then what the
step reports, and no number of the step is published.

1. **W0 byte-exact reproduction.** With the V lag flag off, `i_timeout` keep and `k_cap` unbounded, the
   seven arms, `short`, `wait`, T37, T37r and T37b through this step's subclass reproduce
   `../14/workload_cells.csv` and `../15/onoff_cells.csv` to every printed digit on W0. Nothing else is
   read before this passes.
2. **Identity through the subclass.** A member of the F family with `m` 1, `i_min` equal to `i_max`,
   `i_timeout` keep and `k_cap` unbounded reproduces the fixed arm of that interval event for event, as
   `../13/identity.py` checks for the base engine. Four intervals, three seeds.
3. **`k_cap` locality trace.** One trace with a finite `k_cap`, in which one pair of a node has reached the
   cap while another pair of the same node still multiplies its interval by `m` on its next empty reply,
   and in which no other node's pair is affected. This is the only check that exercises a finite `k_cap`,
   since checks 1 and 2 both run with `k_cap` unbounded.
4. **V placement, on answered and on timed-out polls.** For every V schedule, the realised poll instants
   are within 100 ms of the instants the DP chose, reported separately for the resolutions that followed an
   answered poll and for those that followed a timed-out poll, and separately for the cells with outages
   and the cell without. At most 1 % of V's polls may be bound by the clamp or taken by the provisional
   timer. Above that the step reports V as not placed and stops.
5. **V's lag stream is a cached generator.** Two consecutive polls of one pair under V draw different tick
   lags, and the lag sequence of a pair is unchanged when another pair's schedule changes. A constant lag
   per pair fails this check.
6. **C is at or below every record it bounds.** For every record in C's record set, in every simulation of
   this step and of evidence/15, `D_min(j)` is at or below that record's realised delay, record by record
   and never as a comparison of means. A violation means C is wrong and the step stops.
7. **The DP is timed before the cell list runs.** The DP is timed on one real (cell, workload, seed),
   namely `s6b10p043` p 0.4 W0 seed 1, and the measured per-run-solve time is printed before any cell list
   is started. The scratchpad sketch measured 1.7 ms per run-solve at the real candidate sizes and 17.7 ms
   at a pessimistic 130 candidates, against 22 ms for one replay simulation, so the expectation is that the
   DP is not the dominant cost. When the measured total exceeds 30 minutes on 12 workers, cells are dropped
   in this order and the drop is reported: p 0 WV, p 0 WO, p 0.2 WV, p 0 W0, p 0.2 WO, p 0.4 WV, p 0.2 W0.
   `s6b10p043` p 0.4 with W0 and with WO are kept in every case.
8. **No outage, the timeout coordinate is inert.** In `s5b10p043` p 0, where no poll times out, F members
   that differ only in `i_timeout` give identical numbers.
9. **Reproducibility.** Rerun gives byte-identical csv. ruff clean with the repo settings.

## Cost

Measured on this machine: one replay simulation of a rule takes 0.022 s, of the 0.5 s arm 0.043 s; 12
workers. The DP sketch takes 1.7 ms per run-solve of eight pairs at the real candidate sizes.

V's targets per cell are 2 reachable arms under W0, 1 under WO, 0 under WV, plus up to 2 F targets per
workload where F's budget is reachable. That is at most `(2 + 2) + (1 + 2) + 0 = 7` targets per cell and 21
over the three cells.

| part | simulations | DP run-solves | wall |
|---|---|---|---|
| C | 0 | 0 | seconds |
| V bisection: 21 targets x 8 steps x 4 passes x 10 seeds (5 per fold, two folds) | 6720 | 1680 | under 30 s |
| V final: 21 targets x 4 passes x 20 seeds x 5 replicates | 8400 | 2100 | about 1 minute |
| F search: (60 + 90) members x 2 folds x 10 seeds x 5 replicates x 9 cell-workloads | 135000 | 0 | about 5 minutes |
| F scoring: 2 folds x 2 variants x 10 seeds x 5 replicates x 9 | 1800 | 0 | seconds |
| The seven arms, short, wait and the evidence/15 rules: 16 x 20 seeds x 5 replicates x 9 | 14400 | 0 | about 30 s |

About 166000 simulations and about 3800 DP run-solves, which is about 7 s of DP in total. Under 15 minutes
on 12 workers.

The draft's bisection row said 30240 for factors that multiply out to 10080. The row above is recomputed
from the reachable target set and the cross-fitted bisection.

## Where the review was corrected

One claim of the review is rejected on the code.

- The review's finding 7(b) says the seven fixed arms "are rules with `m` 1 and `i_min = i_max`" and
  therefore all run through the override. Four of them do not. `../14/workload.py` lines 213-232 runs the
  four testbed arms (0.5, 1.359, 3.695, 10.043 s, `kind == "fixed"`) with `run["rule"] = None` on the
  policy path, taking `policy`, `choices` and `choice_jitter` from the target stage, and only the three
  arms evidence/14 added (0.8, 2.2, 6.0 s, `kind == "arm"`) are `m` 1 rules. `short` and `wait` are
  `kind == "policy"` and also run with `rule` None.
  The pass-through therefore holds for the four testbed arms, `short` and `wait` through `nd.rule is None`,
  and for the three added arms and the evidence/15 rules through the inert-extras clause (V flag off,
  `i_timeout` keep, `k_cap` unbounded). The draft's sentence named only the first reason and attached it to
  all seven arms. Both halves are now stated. The rest of finding 7(b), that the dict is shared by all five
  nodes, which is wider than the draft's "all neighbours of a node", is correct and is fixed by the deep
  copy in the subclass.

## Deliverables (this directory)

`spec.md` before anything runs. Then `bound.py` (the subclass, the DP, the coordinate ascent),
`bound_cells.csv`, `bound_gain.csv` (one row per cell, workload and member, with C, V, F_restricted,
F_open, the per-fold selections and the pooled out-of-sample values), `bound_v.csv` (V's chosen instants,
kept in a form a fixed-timetable poll rule could read, so that a later step can ask whether V is testable
on the testbed) and `bound.md` (tables, checks, deviations). `README.md` and `RUN.txt` are written by the
main session after review.

## Addition by the orchestrator, 2026-11-19, before any number

The revision left one case open: F_restricted may land at a budget above V's reachable set, and then
`V_at_F` does not exist and outcomes 2 and 3 cannot be reached at all. To keep those outcomes reachable, a
third formula number is computed in the same search and with the same cross-fitting:

- **F_reach** is the best member of the restricted family among the members whose polls per run on the
  selection half are at or below the reachability cap of that cell and workload (70 % of the minimum
  candidate count). It exists only where V has a reachable target set, so never under WV.
- `V_at_F` is computed at F_restricted's budget when that budget is reachable, and at F_reach's budget
  otherwise. Which of the two was used is printed beside every `V_at_F`, and when F_reach was used the
  outcome sentence says "at the budgets V can reach" and F_restricted's own gain is printed next to
  F_reach's, so that a reader sees how much the formula gains by leaving that budget range.
- The "reported in every case" line keeps using F_restricted.

Nothing else in the reading rule changes.
