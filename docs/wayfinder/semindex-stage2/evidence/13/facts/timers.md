# ev13 - WHEN a node sends a state poll

Source: `/home/lab_shared/cefore-semindex/src/semindex/` (read-only).
Logs: `/home/lab_shared/semindex-eval/<stage>/p<p>/<seed>/`.
Scripts: `check_timers.py`, `check_policy.py`, `check_misc.py` in this directory
(run with `/home/lab_shared/cefore-semindex/.venv/bin/python3 -BI`, `json` only, no semindex import).

## 0. Commit provenance

The runs record `semindex_git_commit: 7d5cd80ec06baaa825a2de0b1819a68c2d85af47`
(e.g. `s5c/p0/1/meta.json`), **not** `5365e079` as the task text says.
Both are safe for this topic:

- `git diff --stat 5365e079 HEAD -- src/semindex/` → empty.
- `git diff --stat 7d5cd80 HEAD -- src/semindex/` → only `evalagg.py`, `evalgen.py`, `evalrun.py`.

`agent.py` and `engine.py` - the only files that decide poll timing - are byte-identical
between the run commit, `5365e079` and today's HEAD (`d19a0a2`). Line numbers below are HEAD's
and are valid for the runs. For config *generation* I read `git show 7d5cd80:src/semindex/evalgen.py`,
since that file did change.

## 1. The three policies and which stage uses which

`meta.json` carries `state_poll_interval_choices_s`, `state_poll_choice_jitter_frac`,
`state_poll_policy`. Read from the first run of every stage:

| stage | choices_s | choice_jitter | policy table | mode |
|---|---|---|---|---|
| s5c | null | null | null | **stage-1 fixed** |
| s5a | [0.5, 1.359, 3.695, 10.043] | 0.1 | null | choices |
| s5b0p5 / s6b0p5 | [0.5] | 0.1 | null | choices (1 value) |
| s5b1p359 / s6b1p359 | [1.359] | 0.1 | null | choices (1 value) |
| s5b3p695 / s6b3p695 | [3.695] | 0.1 | null | choices (1 value) |
| s5b10p043 / s6b10p043 | [10.043] | 0.1 | null | choices (1 value) |
| s5short / s6short | null | null | {intervals_s: [10.043, 3.695], jitter_frac: [0.1, 0.1]} | **policy table** |
| s5wait / s6wait | null | null | {intervals_s: [10.043, 5.0], jitter_frac: [0.1, 0.0]} | **policy table** |

There is no `s6a` / `s6c` in `/home/lab_shared/semindex-eval/`; stage 6 is b*/short/wait only.

**The "fixed-interval setting" the c stage uses is not written anywhere in the config.**
`evalgen.generate_run` only writes `state_poll_interval_choices_s` /
`state_poll_choice_jitter_frac` / `state_poll_policy` when they are set
(`git show 7d5cd80:src/semindex/evalgen.py`, the `if ... is not None` block around the
`raw` dict at lines 481–498). With all three absent the agent keeps `SyncConfig`'s
defaults: `state_poll_interval_s = 2.0`, `state_poll_jitter_frac = 0.2`
(engine.py:60-61). Confirmed over all 51 022 s5c polls (`ev13c_rng.py`):
`interval_s` min 1.600018969520241, max 2.394555076121649, i.e. exactly
`2.0 * [0.8, 1.2]`; `applied_interval_s` absent on every line; `action_prob == 1.0`
on every line. And §4b reproduces the sequence bit for bit from `random.Random("h2")`.

Other defaults that matter, none of them overridden by evalgen:

- `interest_lifetime_ms = 2000` (engine.py:86) - **this is the poll timeout**.
- `max_in_flight_per_neighbor = 8` (engine.py:94), `meta.json` says `null` → default.
- `receive_timeout_ms = 100` (cefpyco_transport.py:91) - the agent loop period, i.e. the
  granularity at which a due poll is actually noticed.
- `POLL_REPLY_GRACE_S = 1.0` (engine.py:117).
- `stop_at_s = 130.0`, from `<name>.experiment.json` of every stage checked
  (`AGENT_STOP_AT_S = WINDOW_END_S = 10 + 120`, evalgen.py:54/77/116).

## 2. The loop: one tick, every neighbor, one clock reading

`Agent.step()` (agent.py:1013-1080) does: boundary check → `transport.poll()` (blocks up to
`receive_timeout_ms`) → boundary check → `now = transport.now()` → `engine.on_tick(now)`.

`SyncEngine.on_tick` (engine.py:424-437):

```
for neighbor in self._config.neighbors:
    if self._config.state_poll_policy is not None:
        self._resolve_timed_out_poll(neighbor, now)
    due_at = self._next_state_poll.get(neighbor, now)
    if now >= due_at:
        self._send_state_request(neighbor, now)
    self._send_fetch_plan(neighbor, now)
```

Consequences, all load-bearing for a replay:

- **One shared loop, one `now`.** Both neighbors of a middle node are tested against the
  *same* clock reading, so two polls that fall due in the same tick get the *identical*
  `ts_ms`. Observed: `s5a/p0/1` h2 emits `state_poll` to h1 and to h3 both at
  `ts_ms=1793766543509` (rel. t0 = +529 ms).
- **Neighbor order is config order** (`gen._neighbors_of`, gen.py:295/818 - ascending host
  index), so for h2 the h1 poll is logged before the h3 poll inside one tick.
- **There is no in-flight limit on state polls.** `max_in_flight_per_neighbor` is consumed
  only by `Tracker.plan` for *record fetches* (tracker.py:103/111/305/311); the `in_flight`
  field in the `state_poll` decision's `state` is reported from
  `Tracker.decision_state` (tracker.py:285) and never gates a poll. A neighbor due for a
  poll is always polled (engine.py:468-470).
- **No catch-up / no grid.** `_next_state_poll[neighbor]` is a single scalar. A tick that
  arrives late fires one poll, not the backlog, and the next interval is measured from the
  actual send time, not from the grid (engine.py:473-478).

## 3. First poll after agent_start

`_next_state_poll` starts empty, so `.get(neighbor, now)` returns `now` and `now >= now`
is true: **every neighbor is polled on the first `on_tick` of the agent's life**
(engine.py:434).

Measured over 20 runs per stage (`check_misc.py`), `state_poll ts_ms − agent_start ts_ms`,
first poll per neighbor:

| stage | min (ms) | max (ms) |
|---|---|---|
| s5c | 40 | 101 |
| s5a | 62 | 102 |
| s5b0p5 | 42 | 101 |
| s5b10p043 | 25 | 101 |
| s5short | 8 | 103 |
| s5wait | 87 | 103 |
| s6short | 76 | 102 |
| s6wait | 77 | 103 |
| s6b1p359 | 18 | 102 |

Bounded by one `receive_timeout_ms` (100 ms) plus the dispatch of whatever arrived: the
first `step()` blocks in `poll()` before reaching `on_tick`. Both neighbors share that
first tick (the 1st- and 2nd-neighbor ranges are the same in every stage).

Relative to `t0_ms` (the ReCefore experiment t0, not the agent's own start) the first poll
lands at 434–2187 ms, dominated by process launch, not by the timer.

## 4. Where each jitter draw happens, and which field carries it

`_send_state_request` (engine.py:439-478) - one draw per poll, at send time:

| mode | rng drawn | action fields logged |
|---|---|---|
| stage-1 fixed | `poll_jitter_rng()` once (engine.py:567) | `interval_s` only, already jittered; **no `applied_interval_s`** |
| choices | `poll_choice_rng()` picks the index (engine.py:558), `poll_choice_jitter_rng()` perturbs it (engine.py:560) | `interval_s` = the chosen value, `applied_interval_s` = after jitter |
| policy table | `poll_choice_jitter_rng()` only (engine.py:458) | `interval_s` = the table row, `applied_interval_s` = after jitter |

Arithmetic:

- stage-1: `interval = state_poll_interval_s * (1 + jitter_frac * (2u - 1))` (engine.py:567-568),
  i.e. uniform on `2.0 * [0.8, 1.2]`. `action_prob = 1.0`.
- choices: `chosen = choices[int(u_c * len)]`, `applied = chosen * (1 + j * (2u_j - 1))`
  (engine.py:558-562). `action_prob = 1/len(choices)`.
- policy: `interval, jitter_frac = policy.row(consecutive_poll_timeouts)` where
  `row(k) = intervals_s[min(k, len-1)]` (engine.py:40-43), then
  `applied = interval * (1 + jitter_frac * (2u - 1))` (`_policy_interval`, engine.py:480-489).
  `action_prob = 1.0` (engine.py:461).

Seeds (agent.py:599-637), exact constructor arguments:

- stage-1 jitter: `random.Random(node).random` - the bare node name, **no prefix, no seed**
  (agent.py:609), so it repeats identically across seeds and across `p` values.
- choice index: `random.Random(f"state_poll_interval:{seed}:{node}").random` (agent.py:625).
- choice jitter / policy jitter:
  `random.Random(f"state_poll_interval_jitter:{seed}:{node}").random` (agent.py:637).

A policy table draws **only** the `state_poll_interval_jitter` stream, once per poll.

## 4b. The rng streams, replayed

`ev13c_rng.py` re-draws these stdlib streams (no semindex import) and compares them to the
logged values, consuming them in **log order**, i.e. interleaved across both neighbors of a
node, one draw per poll:

| stage | node | matched | mismatched | not checkable (jitter_frac = 0) |
|---|---|---|---|---|
| s5short | h2 | 95 | 0 | 0 |
| s5wait | h2 | 72 | 0 | 17 |
| s6short | h2 | 104 | 0 | 0 |
| s5a | h2 | 197 | 0 | 0 |
| s5b1p359 | h2 | 555 | 0 | 0 |

For s5a the *choice* draw and the *jitter* draw are both replayed, in that order
(engine.py:558 then :560), and the chosen `interval_s` matched on all 197. This confirms:
one shared stream per node (not per neighbor), exactly one draw per poll including the
`jitter_frac == 0` rows (which must still advance the stream), and config-neighbor order
within a tick.

Stage-1's node-only seed is confirmed the same way: h2's `interval_s` sequence is byte-identical
across 4 s5c runs with different seeds and different `p` (3/3 prefix matches), and
`random.Random("h2")` predicts the first six values exactly
(1.952993044178107, 2.3651457924904804, 2.1248228982990502, 1.95450925405318,
1.6422953706481258, 2.128781268015356).

### Choices-mode rules, checked over full stages

`ev13c_rng.py`, every run of each stage: `interval_s ∈ choices`, `|applied/interval − 1| ≤ j`,
`action_prob == 1/len(choices)`.

| stage | polls checked | interval not in choices | jitter out of range | action_prob wrong |
|---|---|---|---|---|
| s5a | 134812 | 0 | 0 | 0 |
| s5b0p5 | 188769 | 0 | 0 | 0 |
| s5b3p695 | 28011 | 0 | 0 | 0 |
| s6b1p359 | 59082 | 0 | 0 | 0 |

All four s5a choice values are actually used.

**The jitter draw itself is not logged.** It is recoverable from
`applied_interval_s / interval_s` whenever `jitter_frac > 0`; with `jitter_frac == 0`
(s5wait/s6wait row 1) it is drawn but unrecoverable. A replay that must reproduce the
*stream* has to re-seed `random.Random(f"state_poll_interval_jitter:{seed}:{node}")`.

The state snapshot order differs by mode: without a policy the interval is chosen *then*
`replies.snapshot()` is taken; with a policy the snapshot comes **first**, because the row
is looked up with it (engine.py:450-461). Either way `replies.sent()` is called *after*
the decision is emitted, so `last_poll_status` / `consecutive_poll_timeouts` describe the
*previous* poll, never the one being sent (engine.py:446-448, 471-472).

## 5. Timeout of a poll, and the re-decision

A poll's timeout is `interest_lifetime_ms / 1000 = 2.0 s`, judged from the clock, not from
the transport's timeout callback: `_PollReplies._status` returns `"timed_out"` when
`poll.sent_at + lifetime_s <= now` (engine.py:249-253). The transport *does* report
state-poll deadlines to `on_fetch_timeout`, and they are deliberately ignored there
(engine.py:751-753).

Confirmed from the logs: for a `state_poll_reschedule` with `trigger="timed_out"`, the gap
`ts_ms − poll_epoch_ms` is min **2000 ms** exactly, p50 2006–2011, max 2100 (= 2000 + one
tick) - s5short n=496, s5wait n=397, s6short n=749, s6wait n=573 (`check_misc.py`,
20 runs per stage). That pins `interest_lifetime_ms = 2000` without reading the node config,
which the runs did not keep (it lived in `/tmp/semindex-evalrun-*`).

### Without a policy (s5c, s5a, all b*)

Nothing re-decides. `on_state` records the reply for bookkeeping only; the re-decision
branch is guarded by `state_poll_policy is not None` (engine.py:620-624). The timer set at
send time stands. Answered or timed out makes **no difference to when the next poll goes**.
Confirmed: `state_poll_reschedule` count is 0 in all ten non-policy stages
(`check_timers.py`).

### With a policy (short / wait)

Each poll is resolved exactly once, and the resolution re-decides the pending time
(`_redecide_next_poll`, engine.py:507-537):

- **timed_out**: detected at the *top* of `on_tick`, before the due test, so a re-decided
  time already in the past is polled in the same tick (engine.py:427-433, 491-505). Only
  the **newest** poll is resolved.
- **answered**: `on_state` → `_on_policy_reply` → resolve as `"answered"`, but only if the
  poll is unresolved **and** newest (engine.py:638-658).

The re-decision takes a **fresh snapshot** (so a possibly different table row) but
**reuses the poll's own send-time jitter draw** `poll.jitter_u` (engine.py:519-520), and
measures from the poll's own send time:

```
due_at = poll.sent_at + applied_interval
self._next_state_poll[neighbor] = max(due_at, now)     # engine.py:521-522
```

Logged as `decision kind="state_poll_reschedule"` with action
`{neighbor, poll_epoch_ms, trigger, interval_s, applied_interval_s, clamped}` and state
`{last_poll_status, consecutive_poll_timeouts}` (engine.py:524-536). `clamped` is
`due_at < now`.

Verified (`check_policy.py`, **all** runs of the four policy stages):

- `interval_s` equals `intervals_s[min(consecutive_poll_timeouts, 1)]` on every
  `state_poll` and every `state_poll_reschedule` line: **0 mismatches** out of
  7477+7443, 7039+7017, 12197+12088, 10939+10875 lines.
- the jitter `u` recovered from the reschedule equals the one recovered from its own poll
  in every case where `jitter_frac > 0`: ok 7443 / 7017(4791 recoverable) / 12088 / 10875(3613
  recoverable), **bad 0**.
- row changed between send and reschedule in 1729 / 1756 / 5234 / 4918 cases - i.e. the
  re-decision really does act on the newly learned outcome rather than one interval late.
- `poll_epoch_ms` equals the `ts_ms` of that neighbor's last `state_poll` decision in
  **7443/7443, 7017/7017, 12088/12088, 10875/10875** cases. Both are `round(now * 1000)`
  from the same `now` (engine.py:441 vs engine.py:369), so they are identical by construction.
- triggers: s5short answered 5662 / timed_out 1781; s5wait 5647 / 1370;
  s6short 6255 / 5833; s6wait 5991 / 4884.
- max observed `consecutive_poll_timeouts`: s5short 10, s5wait 7, s6short 16, s6wait 12.
  Both tables have 2 rows, so everything ≥1 uses row 1.
- **`clamped` was never true** - 0 out of 7443+7017+12088+10875 reschedules. Expected: every
  table row (10.043, 3.695, 5.0 s) is longer than the 2.0 s lifetime, so
  `sent_at + applied` is always ahead of the resolution instant. A replay must still
  implement the clamp; the stages here never exercise it.

### Late replies

A reply to a poll already resolved as `timed_out` is *processed as data* (the `state_rx`
path runs) but does **not** re-decide: the schedule the timeout set stands, and a plain
event `{"event": "state_poll_late_reply", neighbor, poll_epoch_ms}` is logged
(engine.py:648-656). A reply to an older, unresolved, non-newest poll changes nothing
(engine.py:657 requires `newest`).

Acceptance is itself conditional (`_PollReplies.replied`, engine.py:195-204): a reply is
applied only if its poll is the newest **or** `now <= sent_at + lifetime + 1.0 s`;
otherwise it is ignored entirely and the poll stays timed out.

**UNCONFIRMED from the logs:** `state_poll_late_reply` occurrences = **0** across every run
of all four policy stages. The late-reply rule is therefore source-only; nothing in these
stages exercises it. Same for `clamped` (above).

## 6. What stops polling

`stop_at_s = 130.0` in every stage's `<name>.experiment.json`. `Agent._on_receive_boundary`
(agent.py:853) returns true once `elapsed >= stop_at_s`, and `step()` takes an early return
*before* `on_tick`, so no further Interest is sent (agent.py:1013-1027, 1072-1076). ReCefore
itself runs to `duration = 135` s.

Measured max `state_poll ts_ms − t0_ms` over all runs of a stage: 129998 (s5c), 129999
(s5a), 130000 (s5b0p5), 129995, 129994, 129986, 129988, 129991, 129977, 130000, 129999,
129999, 129997, 129999 - never above 130000 (`check_timers.py`). Note `elapsed` is measured
from `t0_ms`, which is passed in on the command line and is the *experiment* t0, not the
process start.

## 7. Closed formula

Let `D_k` be the k-th `decision` line for neighbor `n` on one node, `ts(D)` its `ts_ms`,
and

```
applied(D) = D.action.applied_interval_s   if present     # choices mode, policy mode
           = D.action.interval_s           otherwise      # stage-1 fixed (s5c)
```

Then, maintaining one scalar `due[n]`:

```
on  D with kind == "state_poll":
        due[n] = ts(D) + 1000 * applied(D)

on  D with kind == "state_poll_reschedule":                # policy stages only
        due[n] = max(D.action.poll_epoch_ms + 1000 * D.action.applied_interval_s,
                     ts(D))
                 # == poll_epoch_ms + 1000*applied   unless action.clamped

next_poll_ts(n) = the ts of the first agent tick t with t >= due[n]
                  (ticks are ~receive_timeout_ms = 100 ms apart)
```

and the *content* of that next poll is itself determined by the same fields:

```
row        = min(consecutive_poll_timeouts, len(intervals_s) - 1)       # policy only
interval_s = intervals_s[row]                                           # policy
           = choices[floor(u_choice * len(choices))]                    # choices
           = 2.0                                                        # s5c
applied    = interval_s * (1 + jitter_frac_of_row * (2*u_jitter - 1))
```

## 8. Verification of the formula over full runs

`check_timers.py` replays every node of every run of a stage, predicts each `state_poll`
`ts_ms` from the previous decision lines by the rule above, and reports
`actual − predicted_due` in ms. The first poll per neighbor is excluded (it has no
predecessor); everything else is predicted.

| stage | runs | polls | predicted | min | p50 | p90 | p99 | max | in [0, 120] ms |
|---|---|---|---|---|---|---|---|---|---|
| s5c | 100 | 51022 | 50222 | −0.87 | 52.0 | 90.4 | 99.5 | 196.1 | 50095/50222 |
| s5a | 500 | 134812 | 130812 | −0.92 | 49.9 | 89.6 | 99.2 | 174.3 | 130504/130812 |
| s5b0p5 | 100 | 188769 | 187969 | −0.92 | 49.3 | 90.1 | 99.2 | 195.4 | 187345/187969 |
| s5b1p359 | 100 | 73914 | 73114 | −0.86 | 48.9 | 86.8 | 98.9 | 273.1 | 72971/73114 |
| s5b3p695 | 100 | 28011 | 27211 | −0.74 | 50.9 | 89.3 | 99.1 | 101.4 | 27132/27211 |
| s5b10p043 | 100 | 10527 | 9727 | −0.80 | 49.8 | 90.8 | 99.4 | 101.0 | 9711/9727 |
| s5short | 60 | 7477 | 6997 | −0.77 | 49.0 | 89.7 | 98.9 | 100.9 | 6978/6997 |
| s5wait | 60 | 7039 | 6559 | −0.60 | 41.4 | 88.5 | 99.1 | 100.8 | 6541/6559 |
| s6short | 80 | 12197 | 11557 | −0.67 | 50.3 | 90.3 | 99.5 | 110.4 | 11526/11557 |
| s6wait | 80 | 10939 | 10299 | −0.82 | 33.1 | 85.0 | 99.0 | 101.0 | 10286/10299 |
| s6b0p5 | 80 | 150935 | 150295 | −0.94 | 49.5 | 90.5 | 99.5 | 250.7 | 149921/150295 |
| s6b1p359 | 80 | 59082 | 58442 | −0.79 | 50.3 | 87.5 | 99.2 | 233.3 | 58345/58442 |
| s6b3p695 | 80 | 22399 | 21759 | −0.87 | 51.1 | 90.0 | 99.5 | 107.2 | 21712/21759 |
| s6b10p043 | 80 | 8420 | 7780 | −0.63 | 50.0 | 90.3 | 99.3 | 103.7 | 7763/7780 |

Agreement is **14 stages out of 14**, across 1480 runs, 786 553 predicted polls. Read it as:

- The error is a **non-negative tick-quantisation lag**, uniform on [0, 100] ms - exactly
  `receive_timeout_ms`. Median ~50 ms, p99 ~99 ms in every stage. This is the only
  systematic deviation, and a simulator reproduces it by ticking at 100 ms rather than
  firing at `due` exactly.
- Small negatives, |err| < 1 ms, 16–620 per stage (≤0.4 %): `ts_ms = round(now*1000)` loses
  sub-ms, so predicting from the rounded stamp can land a fraction of a ms after the true
  due time. Not a rule violation.
- A tail above 100 ms (127/50222 in s5c, worst 273 ms in s5b1p359): steps where the loop
  itself took longer than one `receive_timeout_ms` - dispatching a packet, a store write, a
  probe. No sign of a different rule: `_next_state_poll` is not re-armed by anything else,
  and a late tick fires exactly one poll.

s5wait and s6wait have a visibly lower p50 (41 / 33 ms) than the other stages. That is not
a different rule: their row-1 `jitter_frac` is 0, so a timed-out neighbor's reschedules land
on a tight 5.0 s grid relative to `sent_at`, and the phase relative to the 100 ms tick is no
longer uniform.

## 9. UNCONFIRMED

1. `state_poll_late_reply` never occurs (0 across all runs of s5short, s5wait, s6short,
   s6wait). The late-reply rule - processed as data, does **not** move the timer - is read
   from engine.py:638-656 only.
2. `clamped == true` never occurs (0 of 37 423 reschedules). The `max(due_at, now)` clamp is
   source-only here; it cannot fire while every table row exceeds the 2.0 s lifetime.
3. The `POLL_REPLY_GRACE_S = 1.0` acceptance window (engine.py:199) and the
   `expired_timeouts` folding (engine.py:221-230) are not separately observable in these
   logs: no event records a *rejected* reply.
4. `interest_lifetime_ms = 2000`, `receive_timeout_ms = 100` and
   `max_in_flight_per_neighbor = 8` are the code defaults plus the fact that evalgen never
   writes those keys. The per-node `{node}.json` the agents actually ran with lived under
   `/tmp/semindex-evalrun-*` and is gone. The 2000 ms figure is independently corroborated
   by the measured timeout latency (§5); the other two are not directly corroborated,
   though the 100 ms tick is strongly implied by the error distribution in §8.
5. ~~The stage-1 fixed values for s5c rest on "key absent → default".~~ **Now confirmed**
   (§1, §4b): the logged range is exactly `2.0 * [0.8, 1.2]` and `random.Random("h2")`
   reproduces the sequence value for value. Not an UNCONFIRMED item any more.
6. The run commit recorded in `meta.json` (`7d5cd80`) is not the `5365e079` named in the
   task. Timing code is identical at both, so no claim here depends on resolving it, but the
   discrepancy is unexplained.
7. `on_state` (engine.py:595-636, read in full) consults `_PollReplies.replied` **first**
   (line 621-625) and then unconditionally emits `state_rx` and folds the advertisement
   into the tracker and `_send_fetch_plan` (lines 626-636). So a reply the poll bookkeeping
   *rejects* is still fully processed as data; only the timer ignores it. Source-only:
   I did not establish from the logs that a `state_rx` with no matching outstanding poll
   actually occurs in these stages.
