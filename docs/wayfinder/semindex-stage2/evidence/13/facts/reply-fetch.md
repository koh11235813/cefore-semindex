# What a reply contains, what happens after it, and how link outages act

Topic facts for the offline replay simulator. Source: `/home/lab_shared/cefore-semindex/src/semindex/`.
Logs: `/home/lab_shared/semindex-eval/<stage>/p<p>/<seed>/`.
Scripts: `(scratch)/`
(`common.py`, `a_clock.py`, `b_polls.py`, `b2_sweep.py`, `b3_edges.py`, `c_hi.py`, `c2_sweep.py`,
`c3_extra.py`, `d_fetch.py`, `f_timeout.py`, `g_misc.py`, `h_rtt.py`, `i_fix.py`, `j_boundary.py`).

## 0. Source provenance - the files I read are the files that ran

`meta.json` records two different `semindex_git_commit` values across the stages of interest:

| stage | semindex_git_commit |
|---|---|
| s5a s5b0p5 s5b1p359 s5b3p695 s5b10p043 s5c s5short s5wait | `7d5cd80e` |
| s6b0p5 s6b1p359 s6b3p695 s6b10p043 s6short s6wait | `5365e079` |

`git diff --stat 5365e079..7d5cd80e -- src/semindex/` and `7d5cd80e..HEAD` both touch **only**
`evalagg.py`, `evalgen.py`, `evalrun.py` (the eval harness). `agent.py`, `engine.py`, `tracker.py`,
`store.py`, `holder_state.py`, `names.py`, `cefpyco_transport.py` are **byte-identical** across
`5365e079`, `7d5cd80e` and HEAD (`d19a0a2`). So every line cited below is the line that ran.

Effective config (from the per-node agent JSON in the run root, e.g.
`s5a/p0.2/1/s5a_p020_seed1/h1.json`, keys: group, neighbors, node, probes, schedule, seed,
`state_poll_*`, stop_at_s, store_synchronous, t0_ms - nothing else): no `interest_lifetime_ms`,
no `max_in_flight_per_neighbor`, no backoff override. So the `SyncConfig` defaults apply:
`interest_lifetime_ms = 2000` (engine.py:86), `max_in_flight_per_neighbor = 8` (engine.py:94),
`fetch_backoff_initial_s = 2.0`, `multiplier = 2.0`, `max_s = 16.0` (engine.py:91-93),
`state_data_expiry_s = 1.0`, `state_data_cache_time_s = 0.0` (engine.py:96-97),
`record_data_expiry_s = 3600`, `record_data_cache_time_s = 300` (engine.py:98-99),
`receive_timeout_ms = 100` (cefpyco_transport.py:91).

---

## 1. What the replying neighbour puts in the reply

**Rule (source).** `handle_interest` for a `StateName` whose holder is this node answers with
`holder_state.encode(self._store.all_progress())` (engine.py:808-812). No filtering by requester,
no per-poll state: it is the **whole store**, every origin this node holds any record for
(store.py:164-170).

**Per origin the wire body is `{"hi": int, "extra": [int...], "truncated": bool}`**
(holder_state.py:52), canonical JSON, sorted keys, ≤ 8192 bytes (holder_state.py:17-18, 29).

- **`hi` is the contiguous prefix from 1, not the max.** store.py:22-33: "`hi` is the highest seq
  held contiguously from 1 (0 if none held). `extra` is the sorted tuple of seqs held above `hi`."
  `_fold_progress` (store.py:298-321) is the only writer: a seq equal to `hi+1` extends `hi` and
  absorbs any now-contiguous extras; any other seq goes into `extra` and leaves `hi` alone.
- **`extra` is the sparse part**, capped at 64 per origin keeping the *lowest* seqs, with
  `truncated: true` when any were dropped (holder_state.py:46-52).
- **It says only what is held, never what was merely heard of** (store.py:28-30, tracker.py:8-16).
- **It is the holder's state at the instant the Interest is served**, not at poll send and not at
  reply receipt. `_serve` calls `handle_interest` synchronously inside `poll()`
  (cefpyco_transport.py:397-407) from a single-threaded loop, so the store snapshot is taken
  between the requester's send and the requester's receive.

**What the requester logs.** `state_rx` carries only the per-origin `hi`, not `extra` and not
`truncated`: `{o: e["hi"] for o, e in decoded.items()}` (engine.py:631-632). A reply that fails
`holder_state.decode` logs `state_rx` with `hi: {}` and then **returns** - no tracker update, no
fetch plan (engine.py:626-630).

### Verification (b): predict each `state_rx.hi` from the neighbour's own `events.jsonl`

Predictor: the contiguous prefix of `{seq}` over the neighbour's `record_created` (own origin) plus
`record_stored` (other origins) with `ts_ms <= t`, per origin.

42 runs, 3 per stage, all 14 stages (`c2_sweep.py`):

| | count |
|---|---|
| state_rx examined | 16815 |
| exact map match at δ = 0 ms | 16643 (98.977 %) |
| exact match for some δ ∈ [-4, +4] ms | 16810 (99.970 %) |
| no δ in ±4 ms | 5 |

Minimal |δ| needed: δ=0 → 16643, δ=-1 ms → 122, δ=-2 ms → 42, δ=-3 ms → 3. Negative δ = the
holder served slightly *before* the reply's receive stamp, which is exactly the one-way latency.

The 5 residuals are **millisecond-resolution artifacts, not rule violations**. Worked example
(`s5a/p0.1/2`, h0 ← h1, rx at t0+56.160 s, `hi {"h0": 85, "h4": 68}`): h1's own log stores
h0/85 and h0/86 both at t0+56.159 s and h0/87, h0/88 both at t0+56.160 s, so `hi` passed through
85 inside one millisecond. Checked mechanically for all 5 (`i_fix.py`): in every one, every
advertised value **is** a seq the holder had actually stored, and is bracketed by
`hi_at(t-4ms) <= obs <= hi_at(t+4ms)` per origin. Both hold 5 / 5:

```
s5a      h0<-h1 obs={'h0':85,'h4':68} lo={'h0':84,...} hi={'h0':88,...} held=True bracketed=True
s5b0p5   h3<-h2 obs={'h0':42,'h4':40} lo={'h0':40,...} hi={'h0':44,...} held=True bracketed=True
s5b1p359 h0<-h1 obs={'h0':12,'h4':10} lo={'h4':8,...}  hi={'h4':12,...} held=True bracketed=True
s5c      h2<-h3 obs={'h0':59,'h4':64} lo={'h0':56,...} hi={'h0':60,...} held=True bracketed=True
s6b0p5   h3<-h4 obs={'h0':23,'h4':28} lo={'h0':14,...} hi={'h0':28,...} held=True bracketed=True
```

The `s5b1p359` row is a *fetched* origin (h4 seen by h1), not a created batch, so the same
one-millisecond collapse happens on the receive path too. **No reply ever advertised a seq the
holder had not stored.**

Clock-skew objection is closed: all five nodes carry the **same** `agent_start.t0_ms` in every run
checked (`a_clock.py`: `t0_ms values across h0..h4: [...] (identical=True)` for all 5 runs), and they
are Mininet namespaces on one kernel clock.

### `extra` in these runs: always empty

Scanning every `advertised` row in every `state_poll` / `fetch_plan` decision across the same 42
runs (`c3_extra.py`): **137 659 rows, 0 with `notified_max > hi`**. `notified_max` is
`max(hi, max(extra))` (tracker.py:278), so `extra` was empty on every advertisement in these runs.
Consistent cause: exactly one out-of-order `record_stored` in the whole sample
(`s6b10p043`, h3, origin h0). A replay simulator for these stages may treat the reply as a pure
contiguous prefix; it must still implement `extra` to be correct in general.

**Ambiguity to flag:** `state_rx.hi == {}` happens 2379 / 16815 times. It means *either* decode
failure (engine.py:629) *or* a holder with no records yet (before t0+13 s, the first scheduled
record). The two are indistinguishable in the log. In these runs all of them are the second case
(every one occurs before the holder's first `record_created`/`record_stored`) - **UNCONFIRMED**
that no decode failure ever occurred, since nothing in the log would separate them.

---

## 2. What the polling node does with a reply

**Order inside `on_state` (engine.py:595-636):**

1. stamp `ts_ms = round(now*1000)` (engine.py:618);
2. `replies.replied(poll_epoch_ms, ...)`, and under a `state_poll_policy` re-decide the next poll
   (engine.py:621-625) - this runs for **any** reply, decode failure included;
3. decode; on failure emit `state_rx` with `hi:{}` and **return** (engine.py:626-630);
4. emit `state_rx`;
5. `tracker.observe_state(origin, neighbor, hi, extra)` per origin (engine.py:634-635) - stores the
   neighbour's `(hi, extra)` **per neighbour**, not merged (tracker.py:133-135);
6. `_send_fetch_plan(neighbor, now)` **immediately**, not at the next tick (engine.py:636).

**`_send_fetch_plan` (engine.py:571-591) + `Tracker.plan_fetches` (tracker.py:302-355):**

- `headroom = 8 - in_flight_count[neighbor]`; `headroom <= 0` ⇒ empty plan, nothing logged
  (tracker.py:311-313, engine.py:585-588). The in-flight count is **per neighbour, across all
  origins**.
- Origins are walked in `sorted(self._origins)` order (tracker.py:316, 219-221); within an origin,
  `missing(origin)` is lowest seq first (tracker.py:167-184), capped at
  `max_missing_per_origin = 1024`.
- A seq is only planned toward a neighbour that **itself advertised holding it**
  (`seq <= hi or seq in extra`, tracker.py:327-329). This is why a line topology fetches only
  "toward" the origin.
- A seq already in flight, or whose `next_attempt_at > now`, is skipped (tracker.py:331-333).
- A seq whose `failed_neighbors` contains this neighbour is skipped **if** some other advertiser
  exists outside that set (tracker.py:340-343).
- Each planned seq is marked in flight on that neighbour and the counter incremented
  (tracker.py:346-354); then one Interest per seq, name
  `ccnx:/semidx/<group>/<neighbor>/rec/<origin>/<seq>` (engine.py:589-591, names.py:105-114),
  lifetime 2000 ms.
- `fetch_plan` is logged **only when the plan is non-empty** (engine.py:585-586) - an empty plan is
  not a decision.

`on_tick` calls `_send_fetch_plan` for every neighbour on every loop iteration (~10/s), after the
state-poll check (engine.py:431-437), so a fetch that could not be planned in `on_state`
(no headroom) is retried on the next tick.

### Verification: the chain in the logs

42 runs, 3 per stage, all 14 stages (`d_fetch.py`), 40 252 `record_stored` events:

| measurement | result |
|---|---|
| `record_stored` with **no** preceding `fetch_plan` naming its exact `(origin, seq, holder)` | **0 / 40252** |
| `record_stored` whose `holder` is not the toward-origin line neighbour | **0 / 40252** |
| records needing more than one `fetch_plan` attempt | 0 in these 42 runs (419 in the 29 timeout runs, section 6) |
| `fetch_plan.fetched` list length histogram | 1 → 20039, 2 → 5, 3 → 3, 4 → 1975, 6 → 1, **8 → 1536** |
| `state_rx` immediately followed by a `fetch_plan` at the same `ts_ms` | 3523 / 16815 |

The length-8 plans are the `max_in_flight_per_neighbor` cap biting exactly; nothing exceeds 8. The
length-4 plans are the schedule's 4-records-per-2-s batches. Only 21 % of replies produce a plan at
all: most replies bring nothing new.

### Latency from `state_rx` to `record_stored`

Per stored record, measured against the nearest preceding `fetch_plan` and the nearest preceding
`state_rx` from the same holder (n = 40252 each):

| interval | min | p50 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|
| `state_rx` → `fetch_plan` | 0 | 0 | 12 | 33 | 53 | 4.0 |
| `fetch_plan` → `record_stored` | 0 | 2 | 4 | 5 | 106 | 2.5 |
| `state_rx` → `record_stored` | 0 | 4 | 14 | 37 | 106 | 6.5 |

(all ms). The `state_rx → fetch_plan` tail (up to 53 ms) is the headroom case: planned by a later
`on_tick` rather than inside `on_state`. `fetch_plan → record_stored` is the record RTT plus the
SQLite write (`PRAGMA synchronous=NORMAL`, store.py:69-76).

Related: state-poll RTT (`h_rtt.py`, 18 runs over 6 stages, each `state_rx` matched to the latest
preceding `state_poll` to that neighbour, n = 3256): min 0, **p50 1 ms**, p90 3 ms, p99 15 ms,
max 33 ms, never > 100 ms.

---

## 3. Can a record arrive by any path other than a fetch?

**No.** Source: `record_stored` is emitted only from `on_record` (engine.py:734-737), which the
transport reaches only via `_route_data` for a parsed `RecordName` (cefpyco_transport.py:453-456),
i.e. a Data answering an Interest this node sent. There is no push: `handle_interest`
(engine.py:795-824) only *serves*, and the engine never sends unsolicited Data. `record_created`
(engine.py:419) is a node's **own** origin only, fired by `Agent._drain_schedule` from the config
schedule (agent.py:861-872); in these runs only h0 and h4 have a non-empty `schedule`.

Log confirmation: 0 / 40252 `record_stored` events lacked a matching preceding `fetch_plan`
(section 2). Also `conflict` (engine.py:719) and `state_poll_late_reply` (engine.py:649) do **not
occur anywhere** in stages s5a, s5c, s6b0p5, s6b1p359, s6short, s6wait (grep over every
`events.jsonl`: 0 files).

**UNCONFIRMED:** record Data carries `cache_time = 300 s` and the name is holder-scoped
(`ccnx:/semidx/<group>/<holder>/rec/<origin>/<seq>`, names.py:9), so a repeat Interest for the same
key can be answered from a Content Store instead of the holder's agent. 419 records in the 29
timeout runs did take more than one `fetch_plan` attempt (section 6), so retries do happen. With
direct links only, the only way a CS copy of that name can exist is the mid-flight-drop case: the
Interest reached the holder, the holder's `cefnetd` cached the Data, and the Data was lost on the
way back. This is **undetectable from these artifacts** (the holder logs nothing when it serves),
and it produces the identical bytes for the identical name, so it changes no replayable event.
It matters only to a simulator that models the retry RTT, which would be shorter on a CS hit.
For state polls this is a non-issue: each poll name carries a unique `epoch_ms` segment
(names.py:102), so no two polls ever share a name and `state_data_cache_time_s = 0` is irrelevant.

---

## 4. How link outages map onto the agents' clock

**The `at` seconds are not what to use; `results.json` carries the measured time.** Each
`link_down`/`link_up` in `<name>.experiment.json` produces one row in `results.json` with
`op_type: "event"`, `scheduled_at`, `actual_at` and a wall-clock ISO `ts`. ReCefore takes
`origin = time.monotonic()` and `t0_ms = int(time.time()*1000)` on consecutive lines
(evalvalid.py:179-183), so the two share one instant.

Measured `ts_epoch_ms − (t0_ms + actual_at*1000)` (`a_clock.py`, 5 runs across 5 stages):

| run | n | min | median | max |
|---|---|---|---|---|
| s5a/p0.2/1 | 34 | 7.3 | 10.5 | 21.9 |
| s5c/p0.4/3 | 49 | 6.4 | 9.2 | 12.1 |
| s6b1p359/p0.6/1 | 59 | 7.3 | 9.7 | 29.5 |
| s6wait/p0.4/2 | 37 | 7.9 | 10.1 | 24.8 |
| s5short/p0.2/5 | 31 | 6.7 | 8.8 | 11.0 |

(all ms; every row `success: true`.) Against `scheduled_at` instead the spread is wider
(`actual_at − scheduled_at` reaches 28 ms), so a simulator must use `actual_at`, or better the
`ts` itself.

**Conclusion.** `t0_ms + actual_at*1000` is the *start* of the interface change and `ts` is its
*completion*; the true flip lies in that ≈7–30 ms window.

**Which to use, measured, not assumed** (`j_boundary.py`, same 42 runs, same predictor as section 5,
only the boundary definition changed):

| boundary | n | agree | misfits | misfit offsets to the boundary (ms) |
|---|---|---|---|---|
| `results.json` `ts` | 21573 | 21564 (99.9583 %) | 9 | −14.0 … −0.0 (all **before**) |
| `t0_ms + actual_at*1000` | 21573 | **21568 (99.9768 %)** | **5** | 0.0, 0.0, 2.0, 4.0, 6.0 (all **after**) |
| `t0_ms + scheduled_at*1000` | 21573 | 21566 (99.9676 %) | 7 | 0.0 … 9.0 |

**Use `t0_ms + actual_at*1000`.** It is the best of the three and its residual is bounded at 6 ms,
against 14 ms for `ts`. The two misfit sign patterns bracket the real flip: it is later than
`t0_ms + actual_at*1000` by a few ms and earlier than `ts` by up to 14 ms.

---

## 5. Verification (a): predict every logged `state_poll`'s outcome from the schedule alone

**Observed label** is taken from the engine itself, not inferred: the *next* `state_poll` decision
to the same neighbour reports `state.last_poll_status` for the poll before it, because
`replies.snapshot(...)` is taken before the new poll is recorded as sent (engine.py:448-457, and
`_PollReplies.snapshot` engine.py:232-246). The last poll of a run has no successor and is excluded.

**Predicted label**: link up at the poll's `ts_ms` ⇒ `answered`; link down ⇒ `timed_out` if the
next poll to that neighbour is ≥ 2000 ms later, else `pending`. Link state from `results.json`
`ts`. Nothing was tuned.

42 runs, 3 per stage, all 14 stages (`b2_sweep.py`):

```
TOTAL n=21573 agree=21564 (99.9583%) misfits=9
answered->answered 16567   pending->pending 3788   timed_out->timed_out 1209
answered->pending      5   answered->timed_out  2   pending->answered 1   timed_out->answered 1
```

Per stage, agreement is 100 % for s5b1p359, s5b3p695, s5b10p043, s5short, s6b3p695, s6b10p043,
s6short, and ≥ 99.70 % for the rest.

**All 9 misfits are the clock-window artifact of section 4.** Their send time minus the nearest
link-event `ts`, in ms: `-14.0, -5.2, -5.1, -4.5, -4.2, -3.8, -3.5, -1.0, -0.0`. Every one is a
poll sent in the last ~14 ms before a boundary row's completion stamp, i.e. the interface had
already flipped. There is no misfit anywhere else. Re-running the same predictor with the boundary
at `t0_ms + actual_at*1000` (section 4's recommendation) leaves **5** misfits, all within 6 ms of
the boundary; nothing outside the boundary window is ever mispredicted under either definition.

### Edge cases (`b3_edges.py`, 33 runs over 11 stages)

- **A poll sent while the link is down is lost outright. It is never delivered late when the link
  comes back, even well inside the Interest lifetime.** Of polls sent while down whose `link_up`
  falls within 2000 ms of the send (and whose next poll is ≥ 2000 ms later, so the label is
  decidable): **239 `timed_out`, 1 `answered`** - and that one was sent 0.4 ms before the `link_up`
  row's `ts`, i.e. the same artifact. Up-delays in the sample span 193–1750 ms and all timed out.
- **A poll in flight when the link drops still gets its reply if the reply beat the drop.** Polls
  sent while up with a `link_down` within 50 ms: answered at drop-delays 9, 22, 23, 33, 36 ms;
  `timed_out` at 1 ms and 5 ms. With an RTT p50 of 1 ms and p99 of 15 ms this is exactly the
  expected shape - the reply wins if it completes first.
- A poll sent while the link is down and *staying* down simply times out at send + 2000 ms; the
  decision machinery judges it from the clock, never from the transport callback
  (engine.py:155-162, 739-754).

---

## 6. What happens to a fetch when the link goes down

**Source.** `CefpycoTransport.send_interest` records `deadline = now + 2000 ms`
(cefpyco_transport.py:233). `_check_timeouts` runs on every `poll()` and reports every expired name
to `engine.on_fetch_timeout` (cefpyco_transport.py:458-463). That emits `fetch_timeout` and calls
`_fail_fetch` (engine.py:739-766), which calls `Tracker.on_fetch_timeout` → `_fail`
(tracker.py:390-436): the wait is the **current** backoff (not the doubled one), the next backoff
is `min(backoff*2, 16)`, the in-flight slot is released, `next_attempt_at = now + wait`, and the
neighbour is added to `failed_neighbors` (reset to just `{neighbour}` when it would cover every
advertiser). The `fetch_backoff` decision is logged only when something was actually resolved
(engine.py:786-791). A chunk fetch for a probe body never enters this table
(cefpyco_transport.py:327-337).

**Log verification** (`f_timeout.py`, 29 runs containing `fetch_timeout`, 13 stages, 1003 events):

| measurement | result |
|---|---|
| `fetch_plan` send → `fetch_timeout` | 2002–2094 ms, 0 above 2200 ms |
| `fetch_backoff` emitted at the same `ts_ms` as the `fetch_timeout` | 1003 / 1003 |
| `wait_s` histogram | 2.0 → 443, 4.0 → 331, 8.0 → 130, 16.0 → 99 (exactly 2/4/8/16) |
| retry `fetch_plan` time − (`fetch_timeout` + `wait_s`) | +1 to +2 ms, n = 979 |
| never retried | 24 (run ended) |

The 2002–2094 ms spread is the 2000 ms lifetime plus up to one `receive_timeout_ms = 100` poll
cycle - the deadline is only noticed when `poll()` next returns.

**Which fetches time out** (`g_misc.py`, same 29 runs, every planned fetch classified by link state
at its send time):

```
sent while link UP   -> stored   27579
sent while link UP   -> timeout    200
sent while link DOWN -> timeout    803   (216 of these had a link_up within the 2000 ms lifetime)
```

- All **200** "sent while up → timeout" had a `link_down` within **4.6–21.6 ms** after the send:
  the same boundary artifact. So **no fetch sent on a genuinely up link ever timed out** in this
  sample.
- All **216** fetches sent during an outage whose `link_up` fell inside the Interest lifetime still
  timed out. **A fetch Interest sent while the link is down is never delivered late**, same rule as
  for polls. There is no forwarder-side Interest retransmit.

**Where the 803 sent-while-down fetches come from** (`i_fix.py`, same 29 runs):

| | count |
|---|---|
| retry of an earlier `fetch_timeout` for the same (origin, seq, neighbour) | **560** |
| first attempt for that key, planned by `on_tick` from a stale advertisement | **243** |

So the dominant mechanism is a **retry chain, not a fresh plan**. `_fail` sets
`next_attempt_at = now + wait` (tracker.py:431) and `plan_fetches` re-sends the moment that passes
(tracker.py:332) **with no link-state check anywhere** (tracker.py:302-355 reads only
advertisements, in-flight counts and `next_attempt_at`). One fetch in flight when the link drops
therefore spawns a 2 s, 4 s, 8 s, 16 s, 16 s… chain of Interests into the dead link for as long as
the outage lasts, each one logging its own `fetch_timeout` and `fetch_backoff`. A replay simulator
must model this: the retry schedule is a pure function of the first failure time, independent of
when the link comes back.

**Why `fetch_timeout` is rare in absolute terms** (1003 across 29 selected runs; whole stages often
have none). New fetches are planned inside `on_state` (engine.py:636), at a moment when the link
has just been proven up by the reply that arrived. During an outage no `state_rx` arrives, so no
new advertisement arrives, so `missing` does not grow and almost nothing new is planned toward that
neighbour. The 243 first attempts are the residue: seqs already known missing before the drop that
had no in-flight slot (`max_in_flight_per_neighbor = 8`) until after it.

**Retries do succeed.** Of 27 579 records stored in those 29 runs, **419 took more than one
`fetch_plan` attempt**: 112 took 2, 201 took 3, 79 took 4, 27 took 5 or more.

---

## UNCONFIRMED

1. **`state_rx.hi == {}` is ambiguous** between a decode failure (engine.py:629) and an
   empty-store holder. 2379 / 16815 occurrences; all consistent with the empty-store reading, but
   the log cannot distinguish them.
2. **Content-Store-served record retries.** 419 records were fetched on a second or later attempt
   (section 6). Whether any of those were answered by a `cefnetd` Content Store rather than the
   holder's agent is undetectable from these artifacts: the holder writes no event when it serves,
   and the bytes and the name are identical either way. It changes no replayable event, only the
   retry RTT.
3. **Exact instant of the interface flip.** Bracketed by the two boundary runs in section 4: a few
   ms after `t0_ms + actual_at*1000` and up to 14 ms before the `results.json` `ts`. Not resolvable
   further from these artifacts. All 5 remaining poll misfits and all 200 "sent while up" fetch
   misfits live in that window.
4. **The 5 `state_rx` whose `hi` has no ±4 ms fit** are shown in section 1 to be sub-millisecond
   serve times (value held, and bracketed by the ±4 ms values, 5 / 5), but the sub-millisecond
   serve instant itself is not recoverable from millisecond timestamps.
5. **`extra` / `truncated` behaviour is untested by these runs** (0 / 137659 advertisements had
   non-empty `extra`). The rule is read from source only.
6. **Per-origin walk order under multiple origins beyond `sorted()`** and the
   `max_missing_per_origin = 1024` cap are never exercised here (2 origins, ≤ 120 seqs each).
