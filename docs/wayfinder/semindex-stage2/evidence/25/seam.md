# evidence/25 seam: where `src` is touched so a decider can be injected into the live agent, where L's inference runs, and the tests and line counts of both

Written 2026-11-27, **before one line of `src` has been changed, before one emulator run of this effort, and before
anything has been measured on `hpc-debian`**. This file answers Acceptance 1 of
`../../tickets/12-three-way-comparison.md` ("実機の口と推論の時間について、答えが書かれたこと（run より先）"): where `src`
is touched and how, where the inference is placed, the estimate of lines, and the form of the tests. Until it is
accepted, no emulator run of ticket 12 starts.

The user's decisions of 2026-11-22 (items 1 to 3) and 2026-11-27 (items 4 to 6) are not reopened. Where this file
chooses something no decision covers, the line says **worker's choice** and names the alternative not taken; those
are collected in section 15. Where this file departs from the design handed to it, the line says so and the
departures are collected in section 16.

Every claim about the code carries `file:line` and every one of them was read in the source during the writing of
this file, at worktree `/home/lab_shared/wt/semindex-ev25`, branch `docs/stage2-evidence-25`, HEAD `386853d`.
Paths without a prefix are repository-relative; paths beginning `../` are relative to this directory. Line numbers
are this HEAD's. Nothing was run but read-only commands (`sed -n`, `grep`, `awk`, `git show --numstat`,
`sha256sum`, `python3 -B -c` for arithmetic). No emulator, no Mininet, no test suite, no source file was touched.

## Question this document answers

**What exactly changes in `src` so that the replay's decider contract can be asked of a live node, where does each
decider's inference run, and how is it shown that the question the emulator asks the decider is the question the
replay asked?**

The second half is the one that is not a matter of plumbing. M costs 0.095 ms per decision
(`../24/latency24.txt:31-33`: 0.0176 + 0.0714 + 0.0056 at the median) and fits inside the loop. L costs 70.13 ms
(`../23/latency.txt:126`), cannot run on this machine at all (no GPU; the node interpreter has numpy and no torch,
measured by the worker 2026-11-27), and a blocking call of that size inside the agent loop is the defect ADR 0004
records as already measured and already removed once (`docs/adr/0004-stage2-policies-from-a-model-of-consecutive-decisions.md`,
Consequences, second bullet).

## What this design can show

1. **That the five existing poll modes behave as they did**, by construction (a separate branch per call site, no
   reordered shared line, and exactly one shared line newly guarded, `engine.py:529-533`, section 6.1) and by the
   guard tests of section 10, group C, which cover all five of them. The claim is behavioural and not textual: one shared line does
   change, so "not by one byte" is carried by those five tests and not by the diff.
2. **That the log of a decider run represents what the decider was handed**, field by field: the recording
   wrapper writes the four inputs it was given and the parity check of section 10, group H compares them against
   the five `../22/observe.py:312-337`'s `from_logged_rows` rebuilds from that run's own `events.jsonl`. Two keys
   of `where` are in no log and in no check. That the question is the PORT's question is argued by section 5's
   table and by the caveats under it, not by this check: both of its sides come from the same run.
3. **That the logged interval is what the decider returned**, because the engine writes the returned value into
   `action["interval_s"]` and refuses any value outside the configured choice set, the port's own rule
   (`../13/replay.py:875-879`).
4. **The cost of a deferred answer, as a count**: how many decisions were answered late, clamped by the wait,
   asked while the pair already had an answer outstanding, answered for no pending request, or never answered,
   per run (section 6.6).

## What this design cannot show

1. **That L on the emulator is L as the replay scored it.** The replay prices a decision at zero time
   (`../13/replay.py:70-71`, `POLL_RTT_MS = 1.0`, `STATE_RX_TO_PLAN_MS = 0.0`). A deferred answer leaves the send
   instant identical only under the condition stated in section 7.1; outside it, L's schedule is its own held
   interval and the replay's prediction for L does not describe the run.
2. **That the baseline stages re-measured on this `src` are comparable to ticket 07's at the decision level.** The
   re-measured stages run in rule mode or single-choice mode, and the only line of the seam either of them
   executes is the guard `if self._poll_decider is None:` on the state assignment at `engine.py:529-533`
   (section 6.1), which is false for no existing mode. So they bound testbed drift plus commit `1c3f758` plus
   that one guard, not the seam (section 13.3).
3. **Anything about numbers.** No run has been made. Every quantity here is either a quoted measurement of an
   earlier effort, arithmetic over one, or a measurement this document requires before `spec.md` is frozen.

## 0. Names

- **decision**: one call of the decider. There are two kinds and they are not symmetric.
- **send decision**: the call at `engine.py:501-546`, `where.kind == "state_poll"`, `resolved` None. Under the
  send-time convention both candidates return the interval the pair already holds and make no model call
  (`../23/decider.py:759-765`; `../24/m24.py:481-486`; pinned by an `ast` check at `../24/m24.py:506-545`).
- **resolution decision**: the call at `engine.py:636-708`, `where.kind == "state_poll_reschedule"`. This is the
  one that takes effect (`../22/port.py:152-155`) and the one that costs an inference.
- **resolution instant** (`res_now`): the `now` the resolution was reached with, `engine.py:634` for a timeout and
  `engine.py:852` for an answer. Every stamp of a resolution decision is this instant.
- **held interval**: the interval the pair's `_next_state_poll` entry was armed from at the poll's send
  (`engine.py:546`).
- **deferral**: a resolution decision whose answer is not available at `res_now`; `decide` returns `None` and the
  held arming stands.
- **collect instant** (`col_now`): the `now` of the `on_tick`, or of the one drain in `Agent.shutdown`
  (section 6.5), in which a deferred answer is taken off the client and applied.
- **late**: a deferred answer collected after the pair's next poll had already gone out.
- **overlap**: a resolution decision of a pair that is **deferred** while an earlier deferral of the same pair is
  still outstanding. The engine answers it like any other; the name exists because the second decision's
  `history` lacks the first resolution's row, which is a classified mismatch of the parity check
  (section 10, group H). It is counted at the deferral and not at the ask, which is the same thing for a decider
  that is uniformly deferring or uniformly synchronous at a resolution, and those are the only two shapes this
  design admits (section 5).
- **orphan**: an answer whose token matches no pending entry, which the engine drops and counts. It is reachable
  through the client's own reconnect (section 8.3) and not through any engine path.
- **unanswered**: a deferred answer that had not arrived when the shutdown drain ran.

## 1. Where the interval is decided today, and the one input the engine does not have

Five modes, selected by which `SyncConfig` field is set, exclusive by config validation and never by the engine
(`engine.py:33-34` and `:62-63` say "Trusted as given"):

| mode | selected by | interval source | re-decides on resolution |
|---|---|---|---|
| stage 1 jittered | none of the four keys | `engine.py:750-752` | no |
| choices | `state_poll_interval_choices_s` (`:96`) | `engine.py:731-745` | no |
| choices + redecide | the above plus `state_poll_choice_redecide` (`:113`) | send `:735`, resolution `:673-676` | yes, after the tracker fold |
| policy table | `state_poll_policy` (`:121`) | `engine.py:548-557` | yes, before the decode |
| rule | `state_poll_rule` (`:127`) | `engine.py:559-569`, `:571-596` | yes, after the decode |

Exclusivity lives in two layers and in both of them only: the agent's validators
(`agent.py:325-351`, `:355-372`, `:375-392`, `:396-443`, `:445-510`, with the combination refusals at `:351`,
`:439` and `:502`) and the experiment parser (`gen.py:508`, `:532`, `:549`, `:568`, `:616`, wired at `:780-789`).
The engine resolves a forbidden combination by branch order alone (`engine.py:511`, `:668`, `:685`).

**At a resolution the engine already assembles the port's state.** `engine.py:680-684` builds
`{**tracker.decision_state(neighbor), "last_poll_status": ..., "consecutive_poll_timeouts": ...}`, which is the
port's `obs["state"]` exactly. The uniform draw at `:674` merely sits six lines before it.

**At a send the state is built after the choice** (`engine.py:511-513` against `:529-533`). That ordering is
immaterial for both candidates, which read no state at a send, and the seam snapshots first anyway (section 6.1).

**One input is absent: `history`.** The port hands a decider the pair's last 8 resolved polls
(`../13/replay.py:1099-1102`, cap `DECIDER_HISTORY = 8` at `../13/replay.py:93`), and the engine keeps nothing
of the kind: `_Poll` holds `epoch_ms, sent_at, replied, jitter_u, resolved` (`engine.py:163-180`) and
`_PollReplies` holds those polls plus `expired_timeouts` (`engine.py:183-223`). Every field of a history row is
already in the `action` dict at `engine.py:692-699` plus `empty` at `:707` and the row's own `ts_ms` at `:432`, so
a bounded deque appended where that row is emitted is the whole addition.

## 2. The shape: a callable injected at construction, riding the redecide path

`SyncEngine.__init__` gains `poll_decider` beside the four injected callables at `engine.py:319-322`;
`SyncConfig` gains `state_poll_decider`, the spec of which object to build; `agent.build_agent` builds it through
a new module and passes it in. The mode is legal only together with `state_poll_interval_choices_s` and
`state_poll_choice_redecide = true`, which is the port's own requirement (`../13/replay.py:311-314`).

Three reasons, in order.

1. **It mirrors the port one for one.** The port's decider is an injected callable (`../13/replay.py:278-279`,
   `:316`), so "the replay and the emulator ask the same question" becomes a property of the construction and not
   a claim in a review.
2. **Riding on redecide means no gate changes.** `_redecides_after_decode()` (`engine.py:598-608`),
   `_redecides_on_resolution()` (`:610-617`), `waits_for_fold` (`:827`), the `empty` computation (`:833`) and the
   `action["empty"]` key (`:700-707`) are all already true with the knob on. **The whole of `on_state`
   (`engine.py:778-853`) is untouched by this seam.** This is item (g) of the design handed to this file, and it
   needs no code (section 16, deviation 3).
3. **A fake decider is a constructor argument in a unit test**, which is how every other injected source of
   randomness in this engine is tested (`engine.py:319-322`); none of the 92 tests in `tests/test_engine.py`
   monkeypatches a registry for a policy.

Options not taken:

- **A fifth exclusive mode of its own.** It would need its own copies of the four gates above, its own `empty`
  computation and its own place in `on_state`, and the arms it is compared against would no longer share the
  choices-mode jitter arithmetic. Rejected: more code, and the comparison against the three fixed arms would stop
  being an identity.
- **The engine resolving the decider from the config itself** (a registry inside `engine.py`). Rejected: it puts
  policy resolution in the mechanism, against `engine.py:3-5` and `:33-34`, and makes an import of numpy or torch
  reachable from `semindex.engine`.
- **Copying M into `src`.** Refused by the user's decision 4, and it would put numpy in the agent's import path
  against `pyproject.toml:6` (`dependencies = []`) and its comment at `:9-11`.

**Two sources of truth, and the invariant that binds them.** The config field says which decider the run wants;
the engine holds the object. The invariant is that one is set if and only if the other is, and it is checked in
`SyncEngine.__init__` beside the neighbour validation at `engine.py:358-359`: a config with
`state_poll_decider` set and no object injected, or an object injected with the field unset, raises at
construction. **Worker's choice**; the alternative, letting the engine branch on the config field and look the
object up, is option 2 above.

## 3. The config field

### 3.1 Shape

`SyncConfig.state_poll_decider: dict | None = None` (`engine.py` beside `:127`). Its value in a per-node config is
one JSON object with exactly four keys:

```json
{"path": "/abs/path/to/evidence/25/mdecider.py",
 "sha256": "<sha256 of mdecider.py, computed when that file exists>",
 "factory": "build",
 "args": {"coef": "/abs/path/to/evidence/24/coef_a.npy"}}
```

`path` absolute (a node's working directory is ReCefore's, not this repository's), `sha256` the hex digest of
**that** file's bytes, which is the decider script and not the coefficient file, `factory` the name of a callable
in it, `args` a JSON object passed to the factory as keyword arguments. The digest is a placeholder above because
the script is not written yet; the coefficient file has a digest of its own and it is checked inside the script
(section 8.1). **Worker's choice**: the digest of the loaded file is in
the config rather than the loaded file checking itself, so that the thing being pinned is pinned by the run
description, which `meta.json` preserves.

### 3.2 Where it is refused

In the port's own order, which checks exclusivity before sufficiency, because a rule silences the choices one
frame up and a decider beside a rule would otherwise be reported as a decider with nothing to pick from
(`../13/replay.py:303-314` says exactly this):

1. `state_poll_decider` cannot be combined with `state_poll_policy` or `state_poll_rule`.
2. It requires `state_poll_interval_choices_s`.
3. It requires `state_poll_choice_redecide` to be present and `true`.
4. Shape: an object with exactly those four keys; `path` a non-empty absolute string; `sha256` 64 lower-case hex
   characters; `factory` a Python identifier; `args` an object.
5. The choice set's smallest applied interval must clear the poll timeout plus one tick:
   `min(state_poll_interval_choices_s) * (1 - state_poll_choice_jitter_frac) * 1000 > interest_lifetime_ms +
   receive_timeout_ms`. Both right-hand terms are fields of the same `AgentConfig` the validator already reads
   (`agent.py:86` and `engine.py:128`, default 2000 and `DEFAULT_RECEIVE_TIMEOUT_MS = 100` at
   `cefpyco_transport.py:91`), so the rule names fields and not literals; the replay writes the same bound with
   its own two constants (`../13/replay.py:73`, `:76`).
   This is the port's own bound on the state the decider reads (section 5, item 1) and it is a refusal here
   because a stage that breaks it reports a `last_poll_status` the port would not report, with no test failing.
   It reads only config fields, which is why it can live beside the other four. **Worker's choice 19**; the
   alternative is a precondition in `spec.md`, which is section 17.6.

Sites, following the redecide knob's own precedent exactly:

- `agent.py`: a new `_validate_poll_decider(raw, *, have_choices, have_redecide)` placed after
  `_validate_choice_redecide` (`agent.py:375-392`), called from `validate_config` after line `:640` and before
  the seed block at `:642-651`; the key added to `_ALLOWED_KEYS` (`agent.py:139-157`) and to the refusal lists of
  the policy (`:432-439`) and the rule (`:495-502`); the validated dict put into `sync_overrides`
  (`agent.py:609-645`), which `build_sync_config` splats at `:747-755`.
- `gen.py`: `parse_state_poll_decider(raw, *, have_choices, have_redecide)` beside
  `parse_state_poll_choice_redecide` (`gen.py:549-566`), which re-checks every rule the agent enforces for the
  same reason that one gives ("so a bad value fails at generation, not after a full run"); the field added to
  `ExperimentSpec` (`gen.py:148-158`) and to `_TOP_LEVEL_OPTIONAL` (`:175-188`); parsed in `parse_experiment`
  after the redecide key (`:787-789`); written into a node's config in `_build_agent_config` only when set
  (`gen.py:897-966`, the rule stated at `:938-939`), so **every existing config stays byte for byte**.
- `evalgen.py`: the key added to `validate_settings` (`:344-403`), to `build_experiment` (`:720-859`, written into
  `raw` only when set, the rule at `:856-857`), to `generate_run` (`:861`) and to the module CLI (`:935`).
- `evalrun.py`: `--poll-decider '<json>'`, parsed by a new `evalgen.parse_poll_decider_arg` built on
  `_parse_json_arg` (`evalgen.py:303-316`), carried through `run_all` (`:900`) and `run_one` (`:689`) into
  `_build_meta` (`:581`).

### 3.3 Where it is recorded

- `meta.json`: `"state_poll_decider": <the object or null>`, recorded as given including null, which is the
  policy's and the rule's rule (`evalrun.py:636-646`) and not the redecide knob's write-only-when-on rule
  (`evalrun.py:669-673`).
- `runs.csv`: a new last column `state_poll_decider`, compact key-sorted JSON through the existing
  `_policy_cell` (`evalagg.py:1445-1450`), by the rule every column since 2026-10-26 follows ("last, so every earlier column keeps its
  position", `evalagg.py:1480-1501`), read by a new `read_meta_state_poll_decider` built exactly like
  `read_meta_state_poll_rule` (`evalagg.py:371-390`) and set beside it at `evalagg.py:1108`.

This is a deviation from the design handed to this file, which said the mode is recorded "the way the redecide
knob is": the knob is in `meta.json` only and has no `runs.csv` column (`RUNS_FIELDS`, `evalagg.py:1452-1502`;
its only mention in `evalagg.py` is a docstring at `:78`). A boolean knob can be read off the stage label; a
decider is a mode with a payload (which file, which digest), and the aggregate's own table is where a reader
compares stages. Section 16, deviation 2.

## 4. The loader, `src/semindex/decider.py`

One new module, standard library only (`hashlib`, `importlib.util`, `os`, `sys`), so `pyproject.toml:6` stays
`dependencies = []`.

```
load_poll_decider(spec: dict, *, node: str) -> object
```

1. Read `spec["path"]` as bytes, compute `hashlib.sha256`, compare against `spec["sha256"]`; on a mismatch raise
   with both digests and the path in the message. **The file is read once and the bytes that were hashed are the
   bytes that are imported** (`importlib.util.spec_from_loader` over a source loader on the read bytes), so no
   second read can slip a different file in between.
2. Import it under a fixed module name (`semindex_poll_decider`) that is not a package name of this repository, so
   a decider script cannot shadow `semindex` or anything it imports.
3. Fetch `spec["factory"]` from the module, call it with `node=node, **spec["args"]`, and return what it returns.
   Whether the returned object satisfies the protocol is not inspected: a missing `decide` raises at the first
   decision with the attribute's own name, and a protocol inspection here would be a second definition of the
   protocol.

`agent.py` imports `semindex.decider` at module scope, like every other module of the package, and calls
`load_poll_decider` inside `build_agent` (`agent.py:1239-1274`) **only when `config.sync_overrides` carries the
field**, so the decider file, and therefore numpy or a socket, is imported only in a run that asked for one.
**Worker's choice**; the alternative, importing `semindex.decider` lazily inside `build_agent`, buys nothing
because the module itself pulls in nothing new.

## 5. The decider protocol

The engine is handed an object and performs no I/O of its own (`engine.py:3-5`). Three methods:

- **`decide(obs) -> float | None`**. `obs` has the port's four keys. A returned float must be an element of
  `state_poll_interval_choices_s`, compared by `in`, the port's own check (`../13/replay.py:875-879`). `None` is
  permitted only when `obs["where"]["kind"] == "state_poll_reschedule"` and means "the answer will come later";
  `None` at a send raises. **Within one run a decider is either synchronous at every resolution or deferring at
  every resolution**, and that is a requirement of the protocol and not an observation about the two candidates:
  one that mixed the two would emit a synchronous row at one resolution and then an earlier resolution's
  deferred row with a smaller `res_now`, which is the inversion the emit site of section 6.5 refuses. M and the
  constant wrapper are the first shape, L and `delaying` the second.
- **`collect() -> list[tuple[token, float, dict]]`**. Answers that have arrived since the last call, in arrival
  order. It must never block. `token` is the value the engine passed with the deferred request,
  `(neighbor, poll_epoch_ms)`: the poll's `epoch_ms` is the pair's own identity for that poll, it is already on
  the logged row (`engine.py:694`), and it is what "still the newest poll" is tested against. It is also the key
  of the engine's pending store (section 6.4), so an answer is attributed to its own request and never to
  another poll's. `dict` is free-form and the engine does not read it; it exists so a decider can keep its own
  diagnostics. Returning the same token twice is allowed and counted as an orphan the second time; the engine
  never applies one token twice.
- **`close()`**. Called from `Agent.shutdown` (`agent.py:1195-1236`) after the single `collect()` drain of
  section 6.5 and before the final counters line.

`obs`, as the engine builds it:

```
{"state":    {**tracker.decision_state(neighbor), "last_poll_status": ..., "consecutive_poll_timeouts": ...},
 "history":  [ {ts_ms, poll_epoch_ms, trigger, empty, interval_s, applied_interval_s, clamped}, ... ],  # <= 8, oldest first
 "resolved": None | {poll_epoch_ms, trigger, empty},
 "where":    {node, neighbor, kind, ts_ms}}
```

Field by field against the port, each line verified on both sides:

| field | port | engine |
|---|---|---|
| `state.in_flight` | `../13/replay.py:381` | `tracker.py:242-249`, through `decision_state` `tracker.py:264-309` |
| `state.last_timeout_s` | `../13/replay.py:386-387` | `tracker.py:251-262`; epoch seconds against the replay's sim-ms, and `../22/observe.py:218-219` only ever subtracts it from `now` |
| `state.truncated` | `../13/replay.py:388` | `tracker.py:306`, same cap 8 (`../13/replay.py:92`) |
| `state.origins[*].hi` / `.missing` / `.advertised[*].{hi,notified_max}` | `../13/replay.py:365-378` | `tracker.py:291-305`, `missing_total` at `tracker.py:205-236` |
| `state.last_poll_status`, `state.consecutive_poll_timeouts` | `../13/replay.py:390-391`, written at a resolution by `../13/replay.py:1020-1021` | `engine.py:284-298`; **equal only under the bound and the two divergences below** |
| `history` | `../13/replay.py:1098-1102` | **new**, section 6.2 |
| `resolved.{poll_epoch_ms,trigger,empty}` | `../13/replay.py:1049-1050` | `engine.py:694-695` and the `empty` of `:833` |
| `where.node`, `where.neighbor`, `where.kind`, `where.ts_ms` | `../13/replay.py:872-873` | `self._config.node`, the loop variable, the call site, `round(now * 1000)` (`engine.py:432`) |
| `where.ordinal` | `../13/replay.py:861-862` | **absent** |

**The two state fields are not the same function on the two sides, and the table's row is an equality only
inside a stated boundary.** The engine judges the newest poll's status off the clock and folds dropped polls
into a counter (`engine.py:284-298`, `:300-308`, `:259-282`); the port writes a status at a resolution only, plus
`"pending"` at a send (`../13/replay.py:731`). Three things have to hold, and the first is a config
precondition while the other two are measured divergences that no test can remove:

1. **A poll must not be superseded before it resolves.** The port states the bound twice and says that outside
   it the two sides report different statuses (`../13/replay.py:726-730` and `:1010-1019`):
   `min(choices) * (1 - state_poll_choice_jitter_frac) * 1000 > interest_lifetime_ms + receive_timeout_ms`, which
   on the port's side is `POLL_TIMEOUT_MS + TICK_PERIOD_MS` (`../13/replay.py:73`, `:76`). For the
   three arms with jitter 0.1 and a 2000 ms lifetime that is `3.695 * 0.9 * 1000 = 3325.5 > 2100`, computed in
   this session. **This is a precondition of the whole comparison and not a property of the seam**: a decider
   stage configured with a shorter arm breaks the row above with no test failing. It reads only config fields, so
   it belongs in the agent's own validation, as rule 5 of section 3.2 (**worker's choice**; the alternative is a
   line in `spec.md`, which is a document and not a refusal). The open point is section 17.6.
2. **A reply that arrives after its poll timed out rewrites both fields, and the port cannot.**
   `_PollReplies.replied` never looks at `poll.resolved`: an accepted reply to the newest poll sets
   `poll.replied = True` and clears `expired_timeouts` (`engine.py:248-255`), after which `_status` reports
   `"answered"` (`engine.py:302-304`). The reply is otherwise inert, because `_on_policy_reply` takes the
   `state_poll_late_reply` branch instead of re-deciding (`engine.py:890-898`) and the port has no such event
   (`../13/replay.py:995-998`). The window is real: a timed-out poll stays the newest until the pair's next send,
   which is at least `3.3255 - 2.1 = 1.2255 s` away, and a reply to the newest poll is accepted however late
   (`engine.py:252-255`, pinned by `tests/test_engine.py:1129`). Every later decision of that pair is then asked
   with `last_poll_status_answered` where the replay would have `last_poll_status_timed_out`, and with a reset
   `consecutive_poll_timeouts`; both are cells of M's vector (`../24/features.py:145-147`, the four values at
   `:85`). **The
   measure is the per-run count of `state_poll_late_reply` events**, which the engine already writes and which
   the comparison script of section 9 reports per stage beside the decision counts. It falls on L and on M alike,
   because it is a property of the engine and not of the decider, so it biases the emulator against the replay
   and not L against M.
3. **A reply that cannot be decoded resolves its poll as answered with `empty` unknown.**
   `engine.py:812-818` calls the resolution with `empty=None` after a `HolderStateError`, so the pair
   (`trigger="answered"`, `empty=None`) exists on the emulator, sets `res_empty_none` beside
   `res_trigger_answered` (`../24/features.py:140-143`, the cells declared at `:242-244`) and renders as
   "reply unknown" in the text L reads (`../22/observe.py:269-272`). The
   port's `resolve` is only ever reached with a decoded reply, so the combination is unreachable there. It is
   countable from the log with no new counter: such a reply writes `state_rx` with `hi == {}` (`engine.py:813`),
   and the comparison script reports those decisions per stage. The repository already treats this as a resolved
   question of its own (`tests/test_engine.py:1151`), which is why it is a caveat here and not a defect.

`where.ordinal` counts the replay's decider calls across all five nodes, which no node can know, and neither
candidate reads it: `../23/decider.py:727` says it is "never read at all", and
`../24/features.py:388-397`'s `record_from_port_obs` reads only `where["ts_ms"]` and `where["neighbor"]`, with
`../24/features.py:737-738` checking that the record is invariant to the node name and the ordinal. **A decider
that reads `obs["where"]["ordinal"]` raises `KeyError` on the emulator.** That is the intended behaviour: a
decider whose answer depends on the ordinal is not a decider a node could run.

**Mutation.** The `state` dict handed to `decide` is the very dict `_emit_decision` publishes
(`engine.py:397-438`), and it is handed over by reference, carrying the port's contract
(`../13/replay.py:847-859`) rather than a copy. Three reasons: `tracker.decision_state` allocates a fresh dict at
every level on every call (`tracker.py:291-309`), so a mutation corrupts exactly one logged row and cannot reach
the next decision; a deep copy is needed to guard anything at all, since `origins` is nested, and it would cost
one nested copy per decision inside the loop; the history, which is a list the engine keeps for the rest of the
run, **is** copied per call for precisely the opposite reason, which is the port's split as well
(`../13/replay.py:865-867`). **Worker's choice**; the alternative is `copy.deepcopy` on the state at both call
sites, about 2 lines and the copy cost, and it can be adopted later without changing any other part of the seam.

## 6. The engine changes

### 6.1 At a send

A new branch **before** the existing test at `engine.py:511`, and one guard on the shared assignment at
`engine.py:529-533`:

```
if self._poll_decider is not None:
    last_poll_status, consecutive_poll_timeouts = replies.snapshot(now, lifetime_s)
    state = {**self._tracker.decision_state(neighbor),
             "last_poll_status": last_poll_status,
             "consecutive_poll_timeouts": consecutive_poll_timeouts}
    interval = self._decide(neighbor, "state_poll", now, state, None)    # None is refused here
    jitter_u = self._poll_choice_jitter_rng()
    applied_interval = interval * (1 + self._config.state_poll_choice_jitter_frac * (2 * jitter_u - 1))
    action = {"neighbor": neighbor, "interval_s": interval, "applied_interval_s": applied_interval}
    action_prob = 1.0
    self._poll_held[neighbor] = applied_interval      # section 6.4 needs it; _Poll keeps no interval
elif self._config.state_poll_policy is None and self._config.state_poll_rule is None:
    ...unchanged...
else:
    ...unchanged...
if self._poll_decider is None:          # the one shared line this seam guards
    state = {...}                       # engine.py:529-533, unchanged inside the guard
```

**`engine.py:529-533` is not inside the if/else at `:511-528`: it runs unconditionally for all five existing
modes** (read in this session), so the branch above cannot simply "skip" it. Two options, and neither leaves the
shared text untouched:

- **Guard it** with `if self._poll_decider is None:`. One shared line gains a condition that is true in every
  existing mode, and no shared line moves and no draw moves. **Taken.**
- **Leave it unguarded** and let it rebuild the state after the decision. Rejected, and it is the trap worth
  naming: `tracker.decision_state` allocates a fresh dict on every call (`tracker.py:291-309`), so the row would
  carry a different object from the one the decider was handed, which contradicts section 5's mutation contract
  and test A3, and is not what the port does -- the port builds the state once, before the decision, and emits
  that very dict (`../13/replay.py:704-718`).

So the user's "the four existing modes do not change by one byte" is carried by the guard tests of
section 10, group C and not by the diff: one shared line does change. The shared tail from `engine.py:534` onward (the
`_emit_decision`, the `replies.sent` and the arming at `:546`) runs unchanged, and `self._poll_held[neighbor]` is
written beside that arming because it is the same quantity the arming uses.

- **`_choose_poll_interval` is not touched.** Its draw at `engine.py:735` is the thing being replaced, and
  leaving the method alone is what keeps the five existing modes' draws where they are.
- **The choice stream is not drawn**, the port's own rule ("the decider replaces the draw and the draw only",
  `../13/replay.py:905-910`). `poll_choice_rng` and `poll_choice_jitter_rng` are separate `random.Random`
  instances (`agent.py:717-744`), so not drawing the first leaves every value of the second bit-identical.
- **The jitter draw stays exactly where it was**, one per send, in the same place in the stream
  (`engine.py:737`).
- Snapshotting before the decision changes no value: `_choose_poll_interval` in choices mode reads neither
  `replies` nor the tracker, and `replies.snapshot` only expires stale polls (`engine.py:284-298`).
- `action_prob` is 1.0, not the `1 / len(choices)` of `engine.py:743`. The decider is deterministic
  (`../24/latency24.txt:13` and `:22` are M's bit-identity check and its mutation; `../23/latency.txt:84-90` is
  L's), so the probability the engine gave this action is 1. `1 / len` would record a probability no decision of
  the run had. No module of `src` divides by the field today: `evalagg.py:1283` copies it onto a decision row
  and `:1947-1948` reports its median and quartiles, which is exactly why the recorded value has to be the true
  one -- a later off-policy reader has only this column to go by.

### 6.2 The history

`self._poll_hist: dict[str, collections.deque] = {}`, `maxlen = 8`, created next to `self._poll_replies`
(`engine.py:378`), **appended only in decider mode** and appended exactly where a `state_poll_reschedule` row is
emitted, from that row's own `action` plus its `ts_ms`:

```
{"ts_ms": <the row's ts_ms>, "poll_epoch_ms": ..., "trigger": ..., "empty": ...,
 "interval_s": ..., "applied_interval_s": ..., "clamped": ...}
```

Two rules, both load-bearing:

- **The append is outside the `log_decisions` gate.** `_emit_decision` is a no-op with logging off
  (`engine.py:427-428`) and the draws are deliberately made outside it (`engine.py:726-729`, `:671-672`). If the
  history were appended inside it, turning logging off would change M's input and therefore which Interests go
  out when. Three existing tests are this property for the draws (`tests/test_engine.py:555`, `:1636`, `:2247`).
- **The append happens after the decision it belongs to has been asked**, so a decision is never in its own
  history, and in emission order, which is the order `../22/observe.py:312-337` rebuilds from the log. The port
  appends in the same place and says why it must be before its own early return
  (`../13/replay.py:1093-1102`).

**Two differences from the port's history, both of them on the deferred path, and both of them the parity
check's business and not a repairable defect.** The port appends for **every** resolution, deliberately before
its own early return (`../13/replay.py:1093-1102`), because every resolution has an answer at the instant it is
asked. Here the row exists only once the answer does:

- **A resolution whose answer never arrives appends nothing**, so it is in no later decision's history. The port
  would have it. The measure is the `unanswered` count of section 6.6.
- **A decision asked between a deferral and its answer sees a history shorter by that outstanding row**
  (the overlap of section 0), while `from_logged_rows` rebuilds the same decision's history from file order,
  where the earlier resolution's row is already present. The two therefore disagree on exactly those decisions,
  which is why section 10, group H classifies a mismatch instead of only reporting one, and why section 6.6
  counts overlaps.

### 6.3 At a resolution

A new branch **before** `engine.py:668`. This ordering is not cosmetic: decider mode requires
`state_poll_choice_redecide = true`, so the test at `:668` is also true, and a decider branch placed after it
would never run and L and M would silently poll by the uniform draw.

```
if self._poll_decider is not None:
    state = {**self._tracker.decision_state(neighbor),        # the five lines of :680-684, duplicated
             "last_poll_status": last_poll_status,
             "consecutive_poll_timeouts": consecutive_poll_timeouts}
    resolved = {"poll_epoch_ms": poll.epoch_ms, "trigger": trigger, "empty": empty}
    answer = self._decide(neighbor, "state_poll_reschedule", now, state, resolved)
    if answer is None:
        self._defer(neighbor, poll, trigger, empty, now, state, resolved)
        return
    interval, applied_interval = answer, answer * (1 + <choice jitter> * (2 * poll.jitter_u - 1))
    action_prob = 1.0
elif self._config.state_poll_choice_redecide:
    ...unchanged...
```

The five state lines are duplicated rather than hoisted out of `:680-684`, because hoisting them would move the
uniform draw at `:674` relative to a line the existing modes execute. Duplication is five lines; a moved draw is
a changed run.

**On `None` the method returns after `_defer`**: the shared tail from `engine.py:690` runs only with a float, so
on a deferral nothing is armed, no `action` is built and no row is emitted.

With a float answer the arithmetic is today's, line for line: `due_at = poll.sent_at + applied_interval`,
`self._next_state_poll[neighbor] = max(due_at, now)` (`engine.py:690-691`), the `action` of `:692-699` with
`clamped = due_at < now`, `action["empty"] = empty` (`:700-707`), the row emitted at `:708`, and then the history
append of section 6.2.

`poll.resolved = trigger` (`engine.py:661`) is set at `res_now` **whether or not the answer is deferred**.
Otherwise `_resolve_timed_out_poll` (`engine.py:619-634`) would resolve the same poll again on the next tick, and
a reply arriving afterwards would take neither of `_on_policy_reply`'s branches (`engine.py:890-900`) and would
not even be logged as `state_poll_late_reply`.

### 6.4 The pending entry

**Keyed by the token, not by the pair**, so an answer is applied with its own poll's send stamp and send draw:

```
self._poll_pending[(neighbor, poll.epoch_ms)] = {
    "res_now": now, "sent_at": poll.sent_at, "jitter_u": poll.jitter_u,
    "trigger": trigger, "empty": empty, "state": state,
    "held_applied_interval_s": self._poll_held[neighbor]}
```

The poll's own `sent_at` and `jitter_u` are copied into the entry rather than read back from `replies.polls`
later, because the poll may have been dropped from that list by then (`engine.py:259-282`).
`held_applied_interval_s` is the applied interval the pair's `_next_state_poll` was armed from at this poll's
send, which is the time that actually runs if the answer comes too late; it is read from `self._poll_held`
(section 6.1) because `_Poll` holds `epoch_ms, sent_at, replied, jitter_u, resolved` and no interval at all
(`engine.py:163-180`) and `_next_state_poll` holds an instant (`engine.py:374`).

**Why not one entry per pair.** A per-pair dict is what the port keeps (`../13/replay.py:720`), and it is sound
there only because the port's bound makes a second resolution of a pair impossible while the first is still
unresolved (`../13/replay.py:1010-1019`, quoted in section 5). Here a second resolution of the same pair with
the first answer still outstanding is reachable, and a per-pair dict would then silently apply the first answer
with the **second** poll's `sent_at` and `jitter_u`: `due_at = sent_at + applied` (`engine.py:690`) would refer
to a different poll. Two resolutions of one pair are at least `3.3255 - 2.1 = 1.2255 s` apart (section 7.2), so
any answer slower than that from its own `res_now` overlaps -- inside the band the design's own budget admits it
may reach. **Worker's choice**: the overlap is counted and not refused. Refusing a second deferral would mean
answering it synchronously, which is the one thing L cannot do.

The store is bounded by the run's own resolution count and nothing is evicted from it, so at shutdown its size
**is** the `unanswered` count (section 6.6). A deferral of a pair that already has an outstanding entry adds a
second entry and increments `overlap`.

### 6.5 Collecting answers

At the start of `on_tick`, before the neighbour loop at `engine.py:493`, and once more in `Agent.shutdown`
(below), and only in decider mode:

```
for token, interval, _info in self._poll_decider.collect():
    ...apply...
```

For each answer, with `col_now` the `on_tick` argument:

0. **The entry is looked up by the token and consumed.** `pend = self._poll_pending.pop(token, None)`. If it is
   absent -- an answer for a poll that never deferred, or a second answer for one already applied, which the
   client's own reconnect can produce (section 8.3) -- the answer is **dropped, counted as `orphan`, and nothing
   is emitted**. One token is never applied twice, so the match key of the parity check stays unique and
   `applied` cannot double-count.
1. **Still the pair's newest poll** (`replies.polls[-1].epoch_ms == token[1]`; the newest poll is never dropped,
   `engine.py:259-262`): `applied = interval * (1 + j * (2 * pend["jitter_u"] - 1))`,
   `due_at = pend["sent_at"] + applied`, `self._next_state_poll[neighbor] = max(due_at, col_now)`. `late = False`.
2. **No longer the newest poll**: the next poll has already gone out with the pair's held interval. The schedule
   is **not** touched and `late = True`. Under an overlap this is the case the earlier of the two answers takes
   by construction, because a second resolution of a pair exists only after a second poll was sent.

`clamped = due_at < pend["res_now"]` **in both cases**, which is the port's own field: the port computes it
against the resolution instant, which is also the row's stamp (`../13/replay.py:1073-1075`), and so does the
synchronous path (`engine.py:690-698`). Computing it against `col_now` instead would make the field a statement
about an instant that is not the row's `ts_ms`, and would move the `x{k}_clamped` cell of each of M's eight
exchange slots (declared at `../24/features.py:256-262`, read back at `:359`) and the clamped marker in the text
L reads (`../22/observe.py:302`). The quantity the wait adds is separate and is a count only:
`clamped_by_wait = (not clamped) and due_at < col_now`, in case 1 alone, since case 2 arms nothing.

Either way the `state_poll_reschedule` row is emitted then, **carrying `pend["state"]`, the snapshot taken at
`res_now`, and stamped `ts_ms = pend["res_now"]`**, and the history row is appended from it. The state is the
entry's and is not rebuilt at `col_now`: `from_logged_rows` hands `e["state"]` and `e["ts_ms"]` over together
and `observation` ages every stamp against that one `now` (`../22/observe.py:206-219`), and `evalagg.py:1266`
computes `last_timeout_age_s` as `event["ts_ms"] / 1000.0 - last_timeout_s`, which is a time only while the
stamp and the state come from one instant. A state rebuilt at `col_now` would break both and the parity check
would report it as the emulator handing the decider something the log does not represent, which would be the
wrong diagnosis.

**One drain in `Agent.shutdown`.** `Agent.step` returns **before** `on_tick` as soon as the receive boundary
reports a stop (`agent.py:1175-1181`, with `_on_receive_boundary` setting the flag at `agent.py:960-962`),
`run()` is `while not self._stop_requested: self.step()` then `self.shutdown()` (`agent.py:1191-1193`), and
`shutdown()` calls `abort_fetch`, `_settle_probe_fetch`, `_emit_pending_snapshots`, `_drain_engine_events` and
the `counters` write and **never `on_tick`** (`agent.py:1217-1236`). Without a drain there, every pair's last
deferral is lost by construction: it would be counted `unanswered` and emit no row although its answer had
arrived, which is section 13.3's own gate firing on the stop boundary rather than on the server. So `shutdown`
calls `collect()` once, before `_drain_engine_events()` at `agent.py:1220` and therefore before the
`decider_counts` line and the `counters` line. An answer applied there arms a timer nothing will fire, which is
harmless; the row and the counts are the point. Its `ts_ms` is `res_now`, inside the measurement window, so
`evalagg.py:1212`'s window filter keeps it and `_check_last_line` (`evalvalid.py:372-379`) still holds because
`counters` stays last. **What remains lost**: an answer that arrives after that one drain. That is the only
residue, and it is what `unanswered` counts.

Two consequences to state plainly:

- **`ts_ms` is not monotone over the file** in decider mode. It is the resolution instant, which is what keeps
  `../22/observe.py:157-235`'s ages identical on both sides, and it is what `evalagg.py:1212` filters the window
  by, so a decision made in the window counts in the window whenever its answer arrives before the drain above.
  Nothing in `evalvalid.py` reads a decision row at all: a grep for `interval|state_poll|decision` over that
  module, run in this session, hits four comment lines about the record-schedule slack (`:71`, `:76`, `:78`,
  `:530`) and nothing else.
  **Within one ordered pair it must stay monotone, and that is an assertion and not a hope.** `age` does not
  clamp -- "Negative is possible in principle and is not clamped" (`../22/observe.py:112-118`) -- and
  `leak_faults` fails a record that holds a negative age (`../22/observe.py:442-451`, the pass line at `:638`),
  so one out-of-order pair of rows would fail any later labelling of these runs rather than merely look odd. A
  single FIFO queue per node (section 8.4) is what makes it hold, and the apply site asserts
  `res_now >= the pair's last emitted res_now` and **raises** if it does not, by the same reading as section 6.7:
  a run whose rows cannot be read is better stopped than averaged in. **Worker's choice**; the alternative is to
  count the inversion and carry it, which leaves a file that the labelling cannot read.
- **A late row's `applied_interval_s` is a time that was not waited.** `interval_s` must be the decider's answer,
  or Acceptance 5 cannot be read off the row, so the time that actually ran goes in a separate key
  `held_applied_interval_s`, present only on a late row. The history, and therefore M's next input, still carries
  the answer's `applied_interval_s` as `waited_s` (`../22/observe.py:208`), which on a late row is wrong. The
  defence is that the count of late rows is reported, is tied to the replay side's own count of the same failure
  (section 6.6) and is expected to be zero; the threat is numbered 2 in section 14.

### 6.6 What the engine counts

`self._decider_counts`, six integers, reported by a new `SyncEngine.decider_counts()` and written by
`Agent.shutdown` (`agent.py:1195-1236`) as one `decider_counts` event **after** the drain of section 6.5 and
**before** the final `counters` line, only in decider mode:

- `deferred`: resolution decisions that returned `None`.
- `applied`: deferred answers applied to the schedule (case 1 above).
- `late`: deferred answers collected after the pair's next poll had gone out (case 2).
- `clamped_by_wait`: answers of case 1 for which `due_at >= res_now` but `due_at < col_now`, that is, the arming
  landed in the past only because of the wait. A zero-latency decider would not have clamped there.
- `overlap`: resolution decisions deferred while the same pair already had an outstanding entry. These are the
  decisions whose `history` is shorter than the one `from_logged_rows` rebuilds for the same row (section 6.2),
  so this count is what tells a classified parity mismatch from a defect.
- `orphan`: answers whose token matched no pending entry, dropped (section 6.5, case 0).

`unanswered` **is written**, as `len(self._poll_pending)` after the drain: nothing is ever evicted from that
store, so its size is exactly the resolutions whose answers never came. It also equals
`deferred - applied - late`, and writing it rather than leaving the subtraction to the reader is what lets the
gate of section 13.3 be read off one line.

**The replay side already counts the same failure as `late`, and the expectation of zero is tied to its measured
value.** `../22/port.py:152-158` defines `provisional_fired` as "the count of send-time provisional timers that
fired before their own poll had resolved", incremented at `../22/port.py:259-261`, and `labelcheck.py:134-136`
treats any non-zero count in an F run as a check failure of that step and not a footnote. Its measured value is
**0** over every simulation of three scripts of that directory (`../22/labels.txt:27`, `../22/weight.txt:113`
over 40000 simulations of which 32000 are port simulations, `../22/stability.txt:87`). So "`late` is expected to
be zero" is not a hope about the emulator: it is the quantity the replay side measured at zero with a decision
priced at zero time, and `late` is what the same quantity becomes once the answer costs something.

The line is written before the `counters` line because `evalvalid._check_counters` requires exactly one
`counters` line per node (`evalvalid.py:266-308`) and `_check_last_line` requires it last (`:372-379`); a
separate event is chosen over six more keys on the `counters` line because `_COUNTER_FIELDS`
(`evalvalid.py:315`) is the set the control-traffic metric subtracts and `_check_counter_decrease` compares, and
growing that line would put new keys inside a check that exists to compare two of them. **Worker's choice.**

### 6.7 Failure

- **An interval outside the choices raises**, where the engine would otherwise arm it, the port's own rule and
  message shape (`../13/replay.py:875-879`). A loud invalid run is better than an arm that is counted as an arm
  it is not.
- **`None` at a send raises.**
- **An exception raised inside `decide` or `collect` is not caught**, and it reaches `run()` by **three**
  different paths, because the three call sites are not on one road:
  1. a resolution reached from a reply goes `cefpyco_transport._route_data` (`:440-456`, which does not guard the
     `on_state` call), `_dispatch` (`:378-395`), `poll()` (`:237-277`), then `Agent.step` (`agent.py:1178`);
  2. a resolution reached from a timeout, and the `collect()` loop, are inside `on_tick`, which `Agent.step`
     calls directly (`agent.py:1183`) and not through the transport at all;
  3. the shutdown drain is inside `Agent.shutdown` (`agent.py:1217-1236`), reached from `run()`
     (`agent.py:1193`).

  In all three the node dies. Justification against `evalvalid`: the run is then flagged by
  `counters_count:{node}` (`evalvalid.py:302-303`) and `last_line_not_counters:{node}` (`:378-379`), because the
  exception propagates out of `run()`, `shutdown()` never runs and no `counters` line is written; so the run is
  excluded and reported rather than averaged in. Two flags that do **not** carry this argument, and the design
  says so rather than listing them: `agent_lives:{node}` fires on `lives != 1` (`evalvalid.py:334-335`) and a
  node that started once and then died wrote exactly one `agent_start` (`agent.py:849`), so it cannot fire
  on a single-life crash; and `traceback:{node}` greps `semindex_agent.log` (`evalvalid.py:502-512`), which
  `evalvalid.py:188-190` describes as stdout-only redirection with nothing in it to read, while a traceback out
  of `agent.main` goes to stderr -- whether the runner merges stderr into that file is outside this worktree and
  is listed as unverified in section 17.7. A logged fallback interval would be worse in exactly the way
  Acceptance 5 cares about: the run would pass the three-arm check while part of its decisions were not the
  decider's. The contrast with `cefpyco_transport._serve`, which does swallow engine exceptions (`:397-414`), is
  deliberate and the comment there says why: that guard protects the loop that serves **other** neighbours from
  one bad incoming Interest, and no decider call is on that path.
- **An answer for an unknown token is not a failure.** It is dropped and counted as `orphan` (section 6.5,
  case 0), because the one mechanism that produces it is the client's own reconnect and a run with a handful of
  them is still a run whose rows are each the answer of one decision.
- **A transport failure of the L client is the client's business, not the engine's.** An absent socket, a closed
  connection or a full send buffer is turned by the client into a deferral that never answers (section 8.3), which
  is a counted state, so one socket hiccup does not void a 160 s run. Only a bug in the decider reaches the
  engine.
- **A deferred answer that never arrives is not an error.** The held interval keeps firing and the counts of
  section 6.6 say how often.

## 7. Why deferred answers for L, and not a wait inside the loop

### 7.1 The argument, with the source

1. **The loop is one thread and it has one receive call.** `Agent.step` is
   `_on_receive_boundary -> transport.poll() -> on_tick -> _drain_schedule -> _drain_probes -> flush`
   (`agent.py:1175-1187`), `run()` is `while not self._stop_requested: self.step()` (`:1191-1193`), and
   `poll()` blocks in cefpyco's `receive()` for up to `DEFAULT_RECEIVE_TIMEOUT_MS = 100`
   (`cefpyco_transport.py:88-91`, `:237-277`). The module docstring states why there can be only one call site and
   only one thread: cefpyco's `receive()` holds the GIL and reads into one process-wide static buffer
   (`cefpyco_transport.py:71-76`). **A thread inside the node process is therefore not an option**, and a 70 ms
   call inside the loop stops everything the loop does, the node's own duty as a holder included
   (`cefpyco_transport.py:386-395` to `_serve` at `:397-414`).
2. **ADR 0004 has already measured this exact shape.** Its Consequences record that h2's probe fetched its body
   synchronously inside the loop, so `on_tick` stopped meanwhile and 3.9 to 8.8 % of h2's polls went out 1.0 to
   1.2 s late, unevenly across arms, and that the fix was to stop blocking. A synchronous decider would be worse
   than that probe: during `fetch_content` other Data and Interests were still processed, while inside a decider
   call nothing is.
3. **The schedule is measured from the poll's own send time**, `due_at = poll.sent_at + applied_interval`,
   armed at `max(due_at, now)` (`engine.py:690-691`). So an answer that arrives before the provisional timer
   fires produces the identical send instant a zero-latency answer would.
4. **The identity holds under two conditions, and they are measured against two different intervals.** Let
   `applied` be the **answer's** applied interval and `held_applied` the one the pair's timer was armed from at
   this poll's send (`engine.py:546`). The send instant equals the zero-latency decider's if and only if
   - `col_now <= poll.sent_at + applied` -- the arming is not clamped by the wait alone (the `clamped_by_wait`
     count), measured against the **answer's** interval; and
   - `col_now < poll.sent_at + held_applied` -- the pair's next poll has not already gone out (the `late`
     count), measured against the **held** interval, because that is the one the provisional timer carries.

   The two differ precisely when the decider changes its mind, which is the quantity M counts as `mind_changes`
   (`../24/m24.py:494-497`), and the binding one is whichever interval is shorter. Both are zero if and only if
   every answer beats the margins below.
5. **Waiting would fall on L alone.** M's 0.095 ms is three orders below the U[0, 100] ms tick lag the replay
   already draws (`../13/replay.py:76`, `:645`) and R is arithmetic (`../22/port.py:55-68`). So a loop stall
   biases the L-against-M difference, which is precisely the quantity ticket 12 item 2 exists to measure.

### 7.2 The margins

Two margins per resolution, one per clause of item 4. `res_now - sent_at` is about the RTT, 1 ms, on an answered
resolution (`../13/replay.py:70`) and in `[2.0, 2.1]` on a timed-out one (the clock rule at `engine.py:633` plus
at most one tick); the pessimistic end, 2.1, is used throughout below.

| clause | measured against | margin | at the 3.695 arm with -10 % jitter, timed out |
|---|---|---|---|
| `late` | the **held** applied interval | `held_applied - (res_now - sent_at)` | `3.3255 - 2.1 = 1.2255 s` |
| `clamped_by_wait` | the **answer's** applied interval | `applied - (res_now - sent_at)` | `3.3255 - 2.1 = 1.2255 s` |

One more 100 ms must come off, because an answer is taken off the client in `on_tick` and the loop reaches
`on_tick` at most one receive period later. **The hard budget for one round trip, node to answer, is therefore
`3.695 * 0.9 - 2.1 - 0.1 = 1.1255 s`**, at the tightest arm and the tightest resolution. The number survives the
split of item 4 only because the minimum over the arms, 3.695, is the same in both roles; the two clauses are not
the same quantity on any single decision. At the other arms, held or answered, the timed-out margins are
`6.0 * 0.9 - 2.1 = 3.3 s` and `10.043 * 0.9 - 2.1 = 6.9387 s`, and a reader computing one decision's margin must
take the **held** interval for the `late` clause: a 10.043 s answer after a 3.695 s held interval has 1.2255 s
and not 6.9387 s.

Against that budget: the queue of one serialised 70.13 ms server with 8 directed pairs in the run is at worst
`8 * 70.13 = 561 ms`, and one node's own burst is at most three resolutions in one iteration (one answered reply,
since `poll()` dispatches at most one packet, `cefpyco_transport.py:237-238`, plus one timed-out poll per
neighbour, `engine.py:493-495` with `_resolve_timed_out_poll` taking only `replies.polls[-1]` at `:632`), that is
`3 * 70.13 = 210 ms`. Both are inside 1.1255 s with room, and neither has been measured through the real
transport.

### 7.3 What must still be measured, and the number that reopens this

Before `spec.md` is frozen:

1. **The round trip from inside a Mininet node namespace to the answer**, p50 and p99, under five nodes'
   concurrent load at the measured decision rate, through the real relay (section 8.4). The worker's measurement
   of 2026-11-27 covers only a local unix socket with no model behind it (2000 round trips of 2000 bytes, median
   0.018 ms, p99 0.028 ms, max 0.452 ms).
2. **L's forward pass on the card that will serve it**, with `../23/latency.txt:31-36`'s read-out (bf16 base,
   float32 adapter attached and not merged, batch 1), under that concurrent load, reported as p99.
   `../23/latency.txt:165-178` is explicit that the existing figure says nothing about contention.
3. **The cold first call.** `../23/latency.txt:35-36` timed 300 warm passes after 10 warm-up passes; a cold pass
   is unmeasured and must land before t0, not in the window.

**Reconsideration number.** If the measured p99 of (1) under load exceeds **half the hard budget, 0.56 s**, or if
any single observation exceeds 1.1255 s, the deferred design is reopened before any L run. The two candidate
replacements, in order: pin the answer to the resolution with a bounded wait only when a pair's previous answer is
still outstanding (section 14, threat 6); or run L's stage at the two long arms' margins only, which changes the
comparison and would need the user's decision.

M and the constant deciders answer synchronously and return a float, so none of this applies to them.

## 8. The deciders, in `evidence/25/` and not in `src`

### 8.1 M

`mdecider.py`: a factory `build(node, coef)` that checks `coef`'s sha256 against a constant in the file, calls
`m24.M.from_file(coef)` (`../24/m24.py:466-470`) and returns a shim whose `decide` is `M.__call__`, whose
`collect` returns `[]` and whose `close` does nothing. **This is the one thing M does adapt**: the engine calls
`decide`, and `M` is a callable object (`../24/m24.py:477-505`). It adapts nothing else, because the engine's
`obs` is the port's `obs`: `M.__call__` reads `where["node"]`, `where["kind"]`, and through
`FE.record_from_port_obs` only `where["ts_ms"]` and `where["neighbor"]` (`../24/features.py:388-397`).

The file that runs on the emulator is the file that was scored in the replay: `../24/m24.py` and
`../24/coef_a.npy` (digest `bbf5a5c7abf1c6fa476c728b3855ecdcab601c580452a2101cf4203e560b8ebf`, computed with
`sha256sum` in this session and equal to `../24/latency24.txt:12`, which is candidate a's line; `:14` is
candidate b's `coef_b.npy`), imported by `mdecider.py`, which is itself pinned by the config's own digest.

### 8.2 L, the client

`ldecider.py`, a factory `build(node, socket_path, send_start)` returning a client that:

- keeps `mem`, the pair's last answer, so the send-time convention is client side and is therefore fresh per node
  process and per run, which is what `../23/decider.py:717-721` requires of one L instance per simulation;
- at a send returns `mem.get((node, neighbor), send_start)` and makes no request. `send_start` is a **required**
  factory argument and not a literal in the file: its value is `../22/port.py:45`'s `ARMS[1]`, which is what both
  candidates return before a pair has resolved (`../23/decider.py:62-65` and `:765`; `../24/m24.py:39-40` and
  `:483`), pinned on the replay side by the `ast` check at `../24/m24.py:506-545`. It is a config argument rather
  than `from port import ARMS` because `../22/port.py:23-24` imports `replay` and `bound` at module scope, and
  the L client is the one script of this directory that must stay as small as a socket and a JSON encoder inside
  a node process. Two things check it instead: the engine refuses any value outside the configured choices
  (section 6.7), so a wrong `send_start` raises on the pair's first send rather than quietly polling at it; and
  `parity.py`, which runs in the analysis venv and imports `port` anyway, asserts
  `send_start == PORT.ARMS[1]` before the sweep. **This is the precondition `ARMS[1] in
  state_poll_interval_choices_s`**, which section 17.4 leaves to `spec.md` together with the arms themselves;
- **a late answer therefore also moves the next send row.** `mem` is written when the answer arrives, so a send
  that went out while the answer was outstanding carries the pair's previous value and the send after it carries
  the answer. In the replay `L.mem[pair]` is written inside the resolution call (`../23/decider.py:777`), so
  there the next send row always carries the answer. Threat 2 names this;
- at a resolution writes one request and returns `None`;
- in `collect` reads whatever complete lines are available, updates `mem` from each answer and yields
  `(token, interval, {"logits": [...]})`.

`mem` is updated from **the answer received, whether or not it was applied to the schedule**. The alternative,
updating only on an applied answer, would make the client's send-time value disagree with the row the log carries
for that resolution. Either choice is wrong in some case; this one keeps the client's state equal to the
decisions the log shows. **Worker's choice**, and it is why a late answer still moves the next send row's
`interval_s`.

### 8.3 L, the wire and the client's failures

One `AF_UNIX` stream socket; one JSON object per line, UTF-8, `\n` terminated, no embedded newline because
`json.dumps` escapes them. Request and response:

```
{"token": ["h1", 1763978000123], "node": "h2", "neighbor": "h1", "kind": "state_poll_reschedule",
 "ts_ms": 1763978002131, "state": {...}, "resolved": {...}, "history": [...]}
{"token": ["h1", 1763978000123], "interval_s": 3.695, "logits": [..., ..., ...]}
```

The token round-trips so a response can never be attributed to another decision. The socket is non-blocking: the
request is appended to a bounded outbound buffer and flushed in `collect` and in `decide`; **a blocking
`sendall` would hand the loop back exactly the stall this design exists to avoid** if the server ever stopped
reading.

Client failures, all turned into a deferral that never answers and all counted by the client and reported by
`close()` into a side file of this directory:

- socket absent or `ECONNREFUSED` at construction: the client is built, counts every decision as unreachable, and
  returns `None` at every resolution. The run then polls by its held intervals and `deferred - applied - late` is
  the whole decision count, so it is detectable and the run is discarded.
- `EAGAIN` with a full buffer: the request is dropped and counted.
- `ECONNRESET` or EOF: one reconnect attempt per `collect`, outstanding requests counted as lost. A reconnect
  that re-sent a request, or a server that answered one twice, would deliver a duplicate answer; the token
  round-trips precisely so the engine can recognise it, and the engine drops and counts it as `orphan`
  (section 6.5, case 0).

**A crash loses these counts.** `close()` is the only place that writes the client's side file and it is reached
only from `Agent.shutdown`, which an exception skips (section 6.7), so the one run where the unreachable, dropped
and lost counts would explain the failure is the run that does not write them. What survives there is the
engine's own picture: the missing `counters` line says the node died, and the rows it did write say which
decisions were answered.

### 8.4 L, the server and the relay

**The server**: one process on `hpc-debian`, holding one model, serving every node of the run. Five model copies
are not an option (`../23/decider.py:719-720` says the replay side assumes the opposite, and this machine has no
card at all). It loops over its listening socket and its connections with `select`, reads whole lines, and
processes **one request at a time, batch 1, in a single FIFO queue over all connections**. Three reasons: batch 1
is the read-out the score, the agreement and the latency of ticket 10 were all measured at, and batching in bf16
changes the argmax of 14 of 512 label rows (`../23/decider.py:657-663`); a single FIFO queue never reorders one
node's own requests, which is the condition section 6.5's ordering rests on; and per-node fairness would make one
node's answer depend on another node's load in a way no node can see.

**The server holds one `L` and calls it**, exactly as ticket 10's own scoring did
(`../23/eval.py:488`: `D.L(read_three, tok)`), with a `where` of
`{"node": <the requesting node>, "neighbor": ..., "kind": "state_poll_reschedule", "ts_ms": ...}` built from the
request. So one decision on the server is `L.__call__` (`../23/decider.py:759-778`) and therefore
`text_of` (`:602-613`) -> `prompt_of` (`:616-618`) -> `read_three`, which is `readout` (`:657-666`) and
tokenises with `P.ids_of` before `forward_three` (`:621-654`) -> `pick` (`:669-680`) ->
**`check_pick` (`:277-294`)** -> `ARMS[index]`, plus the tie count. Nothing is re-derived, which is this
repository's own standard for a second definition of what a decider sees
(`../24/features.py:379-382`: "it is imported rather than imitated").

This reverses a worker's choice, and the two things that choice was defending against are answered:

- **`L.mem` on the server is written and never read.** `mem.get` is only on the send branch
  (`../23/decider.py:763-765`) and the server never sees a send, so the send-time convention still has exactly
  one home, the client of section 8.2. One instance serves all five nodes, keyed on `(node, neighbor)`, which is
  what `../23/decider.py:723-725` says the pair key exists for.
- **Two steps would otherwise have been re-derived, not reused.** `forward_three` takes token ids and not a
  prompt, so a server calling it after `prompt_of` would have had no tokenisation at all; and `check_pick` is
  the statement that an exact tie is resolved by the first maximum, i.e. that the shortest interval wins, on a
  read-out whose "bf16 batch-1 margin over 512 label rows has a minimum of 0.0000"
  (`../23/decider.py:283-285`). Dropping it would have dropped the one assertion the replay ran on every single
  decision.

One reach to state: `check_no_node_name`, which `text_of` runs on every call, tests the five names of
`NODE_NAMES = ("h0", "h1", "h2", "h3", "h4")` (`../23/decider.py:69`, `:181-191`). A stage whose node names are
not a subset of those five keeps that assertion green by having nothing to find, so the names of the decider
stages are the same five or the constant is extended with them.

**The relay**: `ssh -N -L <local.sock>:<remote.sock> hpc-debian`, a local unix socket forwarded to the server's
unix socket, which gives the node-facing end with no new code and no open port. Two constraints from the worker's
measurements of 2026-11-27: a node reaches a unix socket on the shared filesystem but not the host's TCP
loopback, and an `AF_UNIX` path longer than 108 bytes cannot be opened, so the local socket lives at a short
fixed path (for example `/tmp/sx25/l.sock`) and not under the run directory. **The same 108-byte limit applies to
the remote end**, which `ssh -N -L local.sock:remote.sock` binds on `hpc-debian`, so the server's own path is
fixed and short there too and is not derived from a run directory either. **Nothing about the forward has been
measured**; it is measurement 1 of section 7.3. Whether a Mininet host sees the same `/tmp` as the forwarding
process is part of that measurement and not an assumption of this document (section 17.7).

Server and relay are started by a script of this directory before the sweep and are not expressed in the ReCefore
YAML: ADR 0001 requires that anything the agent needs at run time be expressible through the generic testbed
features, and what the agent needs here is one socket path in its own config, which `gen.py:897-966` already
carries. The script checks that the socket answers a probe request before each stage and after it; a run whose
counts show any unanswered decision is reported with the count and excluded.

### 8.5 The constant, recording and delaying wrappers

`fakes.py` of this directory, three wrappers:

- `const(interval)`: one arm at every call, synchronous (the identity test and the smoke run).
- `recording(inner)`: writes **the four inputs themselves**, one JSON line per decision (`state`, `resolved`,
  `history`, `where`, the returned interval and the token), and delegates to `inner`. The inputs and not a digest
  of them, because the parity check compares field by field and a digest can only say that two things differ.
  It wraps whichever decider the run uses, so an M or an L run can be checked without a second kind of run.
- `delaying(inner, latency_s)`: answers `inner`'s decisions after a latency the test or the script sets, through
  `decide` returning `None` and `collect` releasing them. This is the only way to exercise the deferred path, the
  overlap and the orphan without a model, and it is what group E and the second parity run use.

## 9. What is logged per decision

In decider mode the `state_poll_reschedule` row's `action` carries, beside the six fields of `engine.py:692-699`
and the `empty` of `:707`:

| key | meaning |
|---|---|
| `decided_at_ms` | `round(col_now * 1000)`, the instant the answer was applied; equal to `ts_ms` for a synchronous decider |
| `late` | the answer arrived after the pair's next poll had gone out; the schedule was not moved |
| `held_applied_interval_s` | present only when `late` is true: the applied interval that actually ran, from the pair's send-time arming (`engine.py:546`) |

`clamped` keeps the port's own meaning on every row, late rows included: `due_at < res_now`, a fact about the
resolution instant, which is also the row's `ts_ms` (`../13/replay.py:1073-1075`, `engine.py:690-698`). It is
therefore comparable against the replay's field and against the three fixed-arm stages' rows without a caveat.
`clamped_by_wait` is a count and not a row field: whether the wait is what pushed an arming into the past is the
comparison `ts_ms < decided_at_ms <= ts_ms + applied_interval_s`, which the row's reader can make but which no
single row can be asked for on its own.

Nothing else. In particular **no `obs` digest and no decider identity are added to the row.** The digest belongs
to the recording wrapper's own side file (section 8.5), which is one side of the parity check and the engine's log
is the other; and the decider's identity, its path and its sha256 are in `meta.json` and `runs.csv` per run
(section 3.3), so a per-row copy would be the same fact 160 times per run. One schema serves L and M: an M row
carries `decided_at_ms == ts_ms` and `late: false`.

**Rows of the five existing modes gain nothing.** The three keys above are written only when the engine holds a
decider, which is the rule `engine.py:700-707` already follows for `empty`.

**The Acceptance 5 check**, which `evalvalid` cannot do today (it checks preconditions only, `evalvalid.py:1-9`,
and nothing about intervals): a new script of this directory reads every run's `events.jsonl` and asserts, per
stage, that the set of `action["interval_s"]` over the `state_poll_reschedule` rows is a subset of that stage's
own choice set. **The second half of the assertion needs a second witness and only the validity runs have one.**
"Every row's `interval_s` equals the value the decider returned" cannot be read off the log alone: the log is the
only record of what was returned, so comparing the two would be comparing a value with itself, and the engine's
own guard (section 6.7) is what enforces it at the instant of the decision. So the script asserts it where there
is an independent record: on the one validity run per decider stage (section 13.2 item 4), which is run through
the recording wrapper of section 8.5, the script compares the wrapper's side file against the rows decision by
decision. Over the 280-run sweep the claim rests on the engine's refusal plus those validity runs, and the check
says so instead of reporting a tautology as a pass. Two scope statements belong in the check and in `spec.md`:

- **The three-arm claim is per stage.** It holds for the L and M stages and for the three fixed-arm stages. It
  does **not** hold for the F stage. F is a `--poll-rule` and a `StatePollRule` has **five** fields
  (`engine.py:66-70`), all five of which the stage has to name: `i_min_s` 3.695, `factor` 1.5, `i_max_s` 10.043,
  `jitter_frac` 0.1 and `on_timeout` `"reset"`, which are the five values ticket 07 pinned and recorded as its
  own pass line 1 (`../../tickets/07-baseline-heldout.md:219`, restated at `../20/results.md:57`). The two the
  design of the arms cannot leave out are the last two: `on_timeout` must be `"reset"`, because the labelling's F
  returns `i_timeout = F_MEM[3] = 3.695 = i_min` on anything but an answer (`../22/port.py:52`, `:65-67`), and
  `"keep"` is the one thing the repository's two configured rules disagree about (`engine.py:52-55`);
  `jitter_frac` must be the fraction the other stages use, or the five stages' applied intervals stop being
  comparable. F's reachable intervals are then 3.695, 5.5425, 8.31375 and 10.043 by its own arithmetic
  (`../22/port.py:55-68`), a subset of the six values `port.union_choices()` declares (`../22/port.py:71-90`) and
  not the three arms. The check computes them and compares no literal: F's second value is
  5.5424999999999995 and no rounded literal equals it (`../20/results.md:37`).
- **The check reads the reschedule rows, not the send rows.** Under the send-time convention a `state_poll` row's
  `interval_s` is the interval the pair already held, not a decision.

## 10. The tests

Forms, with the existing test each one follows. Names are proposals; those under group C are existing tests to
keep or to extend and their current lines are given.

**A. The engine hands over the port's `obs`** (`tests/test_engine.py`)
1. `test_a_decider_is_handed_the_port_s_own_state_fields_at_a_send_and_at_a_resolution`: a recording fake receives,
   at both call sites, the six state keys in order, `resolved`'s three fields (None at a send) and `where`'s four
   keys; the engine-side twin of `test_redecide_row_carries_exactly_the_send_time_row_s_state_fields`
   (`tests/test_engine.py:2219`).
2. `test_the_decider_s_history_is_the_pair_s_last_eight_resolutions_oldest_first`: over twelve resolutions, the
   history has the seven fields of section 6.2, is capped at 8, is oldest first, never contains the decision being
   made, and is per pair on a two-neighbour engine.
3. `test_a_decider_mutating_the_state_corrupts_only_its_own_row`: the port's contract as an assertion
   (`../13/replay.py:847-859`). It is the one test that pins worker's choice 5 rather than a property of the
   seam: adopting `copy.deepcopy` later would require rewriting this test, and nothing else.

**B. The identity** (`tests/test_engine.py`)
4. `test_a_constant_decider_polls_exactly_like_the_same_single_choice`: the pattern of
   `test_a_one_row_policy_polls_exactly_like_the_same_single_choice` (`tests/test_engine.py:1527`): a
   `FakeNetwork` run with replies, a partition and a heal, comparing the Interest names of a decider engine
   (three choices, redecide on, `const(x)`) against a choices engine with the single choice `x`. They must be
   equal, because the two draw the same jitter stream and the choice stream is a separate instance
   (`agent.py:717-744`). This is the acceptance-grade test of the whole seam.

**C. The five existing modes behave as they did** (`tests/test_engine.py`, extend in place). **Five, not four**:
section 1's table has five rows, and the fifth, choices plus redecide, is the mode both insertions sit in front
of, so it is the one whose guard matters most and the one a reader of this list must be able to find.
5. `test_without_a_policy_nothing_new_runs` (`:1602`, parametrised `stage1`/`choices`): add the assertion that
   `on_tick` calls nothing of the decider path and that no history is kept.
6. `test_choices_without_redecide_never_reschedule` (`:2076`), `test_a_flat_rule_polls_exactly_like_the_same_one_row_policy`
   (`:1983`) and `test_a_one_row_policy_polls_exactly_like_the_same_single_choice` (`:1527`): unchanged, and they
   must still pass, which is what "not by one byte" means operationally.
7. **The redecide mode's own guards**, which the seam's two branches are inserted in front of and which pin the
   neighbourhood of `engine.py:511` and `:668`: `test_redecide_draws_a_new_choice_on_an_answered_poll` (`:2097`),
   `test_redecide_row_state_shows_what_the_reply_advertised_but_not_its_fetches` (`:2146`),
   `test_redecide_draws_a_new_choice_on_a_timed_out_poll_and_clamps` (`:2178`),
   `test_redecide_row_carries_exactly_the_send_time_row_s_state_fields` (`:2219`),
   `test_every_redecided_interval_is_one_of_the_choices_at_one_over_len` (`:2233`) and
   `test_redecide_draws_once_per_resolution_with_logging_off_as_well` (`:2247`): unchanged, and they must still
   pass. Items 5, 6 and 7 together are ten existing tests over the five modes, and they are what the opening
   claim rests on and what would fail if the guard on `engine.py:529-533` (section 6.1) were written wrong.
8. `test_redecide_draws_once_per_resolution_with_logging_off_as_well` (`:2247`) gets a decider twin:
   `test_a_decider_run_is_the_same_run_with_logging_off`, which also pins that the history is appended with
   logging off (section 6.2).

**D. Refusals** (`tests/test_engine.py`)
9. `test_an_interval_outside_the_choices_raises`, and `test_none_at_a_send_raises`.
10. `test_an_engine_with_a_decider_config_and_no_object_refuses_to_construct`, and the converse (section 2's
    invariant).
11. `test_a_non_monotone_resolution_stamp_within_one_pair_raises` (section 6.5): two answers released out of
    order for one pair, and the emit site refuses rather than writing a row whose age would be negative
    (`../22/observe.py:442-451`).

**E. The deferred path** (`tests/test_engine.py`, with `delaying` of section 8.5, whose answers the test releases)
12. `test_an_answer_collected_before_the_provisional_timer_gives_the_synchronous_send_instant`: the same engine
    driven twice, once with a synchronous constant decider and once with a deferring one answered before the arm,
    must send the same Interest names at the same times.
13. `test_an_answer_collected_after_the_next_poll_is_marked_late_and_moves_nothing`: `late` true,
    `held_applied_interval_s` present, `_next_state_poll` unchanged, the row still stamped with `res_now`, and
    `clamped` computed against `res_now` and not against `col_now`.
14. `test_an_answer_that_never_comes_leaves_the_held_interval_firing`: the pair keeps polling at the held
    interval and `decider_counts` reports `deferred` and `unanswered` without `applied` or `late`.
15. `test_an_answer_whose_due_instant_has_passed_clamps_and_is_counted_as_clamped_by_wait`, with `clamped` false
    on the same row, which is what separates the two quantities.
16. `test_two_outstanding_answers_for_one_pair_are_each_applied_with_their_own_poll_s_stamp`: two deferrals of
    one pair, answered in order; the second is applied and the first is `late`, each row carries its own
    `poll_epoch_ms`, `ts_ms` and `applied_interval_s`, and `overlap` is 1. This is the test a per-pair pending
    store fails (section 6.4).
17. `test_an_answer_for_an_unknown_token_is_dropped_and_counted`, and the same for a token already applied:
    no second row for one `poll_epoch_ms`, `applied` unchanged, `orphan` incremented.
18. `test_the_last_deferral_of_a_run_is_collected_in_shutdown`: a stop boundary reached with an answer already
    available; the row is written, `unanswered` is 0, the `decider_counts` line precedes the `counters` line and
    `counters` is still the last line. The shutdown-order assertion follows
    `test_a_counters_snapshot_captured_mid_probe_is_not_lost_when_that_probe_is_what_triggers_stop`
    (`tests/test_agent.py:910`) and
    `test_stop_at_s_ends_the_loop_exactly_as_sigterm_would_after_the_last_due_step` (`:1341`); it lives in
    `tests/test_agent.py`, because `shutdown` is the agent's.
19. `test_a_deferred_resolution_is_resolved_once`: `poll.resolved` is set at `res_now`, the next tick does not
    resolve it again, and a reply afterwards is logged as `state_poll_late_reply` (`engine.py:890-898`).

**F. The config chain**, each following the redecide knob's own test, which is named beside it
20. `tests/test_agent.py`: the decider key refused without choices, without redecide, beside a policy, beside a
    rule, with a malformed object, and with a choice set whose smallest applied interval breaks the bound of
    section 5 item 1, after `test_validate_config_rejects_the_redecide_knob_without_choices_or_as_a_non_boolean`
    (`tests/test_agent.py:2197`); and carried into `sync_overrides`, after
    `test_validate_config_carries_the_redecide_knob_and_defaults_it_to_off` (`:2213`).
21. `tests/test_gen.py`: reaches every agent config with the choices, the redecide key and the seed, after
    `test_the_redecide_knob_reaches_every_agent_config_with_the_choices_and_the_seed` (`:802`); writes no key when
    unset, after `test_the_redecide_knob_off_writes_no_key_at_all` (`:814`); a written key passes the agent's own
    validation, after `:823`; `parse_experiment` refuses it without choices, after `:834`.
22. `tests/test_evalgen.py`: generates the bytes it did before the key when unset, after
    `test_the_redecide_knob_off_generates_the_bytes_it_did_before_the_key` (`:1597`); reaches every agent config
    when given, after `:1606`.
23. `tests/test_evalrun.py`: `run_one` generates with and records the decider, after
    `test_run_one_generates_with_and_records_the_redecide_knob` (`:1992`); `main` passes it and None when not
    given, after `:2035`; `main` refuses it before any run, after `:2052`; `run_all` hands every run the same
    object, after `:2071`. Plus the new `runs.csv` column in `tests/test_evalagg.py`, following that file's
    `state_poll_rule` column test (`tests/test_evalagg.py:2418-2449`,
    `test_read_meta_state_poll_rule_reads_the_rule_off_evalruns_own_meta_json` at `:2446`).

**G. The loader** (`tests/test_decider.py`, new)
24. `test_a_wrong_sha256_refuses_to_start`, naming both digests.
25. `test_no_decider_run_touches_the_loader`: after a `build_agent` with the field absent, `load_poll_decider`
    (monkeypatched to raise) was not called and `sys.modules` holds no `semindex_poll_decider`. The obvious
    wording, "a file that raises on import, with the field absent, and `build_agent` succeeds", is the one to
    avoid: with the field absent there is no path for anything to import (the spec carries the path, section 4),
    so no implementation of this seam could fail it. `numpy`'s absence from `sys.modules` is deliberately **not**
    asserted here: that dict is shared across a pytest session, so any earlier test that imported numpy would
    fail this one for a reason that is not the seam. If that property is wanted it is an assertion made in a
    subprocess, and the thing it would really be measuring is section 13.2 item 1, the import cost.
26. `test_the_bytes_that_were_hashed_are_the_bytes_that_run`.
27. `test_the_factory_receives_the_node_name_and_the_configured_args`.

**H. On the emulator** (a script of this directory, `parity.py`, not pytest)

28. **The parity check.** For each decider run named below, compare decision by decision the four inputs the
    recording wrapper of section 8.5 wrote at a **resolution** against the five
    `../22/observe.py:312-337`'s `from_logged_rows` yields from that same node's `events.jsonl`:
    `state` against `e["state"]`, `resolved` against `{k: action[k] for k in RESOLVED_FIELDS}`, `history`
    against the rebuilt list, and `where["ts_ms"]` against `e["ts_ms"]`, by exact equality, field by field.
    - **Not through `observation()`.** `from_logged_rows` yields a five-tuple and never an `obs`
      (`../22/observe.py:333-334`), so a digest of an `obs` and a digest of a record are different objects and
      could not be equal for any run. Putting both sides through `observation()` instead would compare, but
      lossily: ages are rounded to 0.1 s and intervals to 0.001 s (`../22/observe.py:91-92`, `:112-118`), origin
      and advertiser names become positions (`../22/observe.py:121-154`), and anything past `MAX_ORIGINS`,
      `MAX_PEERS` or `MAX_EXCHANGES` is dropped (`:93-95`), so a 50 ms stamp skew or a wrong origin key would
      pass. The comparison is therefore on the raw inputs, and `record_from_port_obs` against
      `records_from_logged_rows` (`../24/features.py:376-398`) is run **beside** it as the second half, because
      that pair of functions is what M actually reads.
    - **Both directions, named.** A recorded decision with no logged row, and a logged row with no recorded
      decision, are each reported; neither is silently skipped.
    - **The two keys of `where` no log carries** are `node` and `kind`, and they are checked against the only
      thing there is to check them against: the wrapper's file is one node's, so `where["node"]` must equal the
      node whose `events.jsonl` is being read, and `where["kind"]` must equal `e["kind"]`. Both are free and both
      are asserted. What that cannot show is that the engine's `node` is the node the testbed thinks it is; that
      rests on `self._config.node`, which is the same field every other row of the file is written from.
    - **Scope**: resolutions only. `from_logged_rows` yields nothing for a send row, so the send decisions of a
      run are not covered at all.
    - **Every mismatch is classified before it is a defect**, from the run's own log, read before the gate:
      an `overlap` or `unanswered` count above zero in `decider_counts` (section 6.6) explains a missing or
      short history exactly (section 6.2); a `state_poll_late_reply` event for the pair explains a
      `last_poll_status` or `consecutive_poll_timeouts` difference (section 5 item 2); a `state_rx` with
      `hi == {}` explains an `empty` of null beside an `answered` trigger (section 5 item 3); two `agent_start`
      lines explain a history that crosses a restart while the wrapper's file starts empty. What is left after
      those four is a **defect**, and only a defect stops the run plan.
    - **What a defect would mean**: that the emulator hands the decider something the log does not represent,
      and therefore that M, whose coefficients were fitted on exactly those logged rows, is being asked a
      different question on the emulator than the one it was trained and scored on.
    - **What it cannot show.** Both sides come from the same run and are built from the same `action` at the same
      instant (section 6.2), so this is the engine checked against its own log and not against the port. That
      the question is the port's question is section 5's table and its three caveats.
29. **Three runs, not one.** (a) the M smoke run of section 13.1, which is synchronous and is the only one that
    exercises `record_from_port_obs` and `vector` end to end -- a `const` decider reads no field of `obs`, so a
    run with one could not raise on a missing or unexpected field; (b) one run with `delaying` at an injected
    latency above a pair's margin, which is the only way to reach the deferred path, the overlap class and the
    `late` row; (c) the L stage's own validity run (section 13.2 item 4). A check that never ran against a
    deferring decider would never exercise the one property it exists to catch.
30. One `cefore-run-tests` smoke scenario with `const(6.0)` to prove the mode starts, stops and validates.

## 11. The estimate

Basis, the three closest precedent commits, each read in this session with `git show --numstat`:

| commit | what it was | src + | src - | tests + | tests/src |
|---|---|---|---|---|---|
| `1c3f758` | ticket 08's redecide knob: engine branch plus the whole config chain plus the aggregate | 279 | 62 | 598 | 2.14 |
| `c3ee2f0` | the rule, engine only | 200 | 31 | 389 | 1.95 |
| `5e840b9` | the rule's config chain, engine untouched | 253 | 25 | 456 | 1.80 |

Per file, with the comment density this repository's `CLAUDE.md` requires counted in:

| file | added | changed | basis |
|---|---|---|---|
| `src/semindex/engine.py`, the synchronous seam | 120 to 160 | 10 to 20 | config field and docstring 15, constructor argument and the invariant 15, send branch 25, resolution branch 25, `_decider_obs` and `_decide` 40, history deque 20 |
| `src/semindex/engine.py`, the deferral on top | 110 to 150 | 5 to 15 | token-keyed pending store 25, collect and apply 50, the six counts and `unanswered` 20, the per-pair monotone assertion 5, docstrings 30. New arithmetic, so priced against `c3ee2f0`'s 181 engine lines rather than against the seam |
| `src/semindex/decider.py` (new) | 120 to 160 | 0 | one function, the digest check and the import, with the docstrings this repository asks for |
| `src/semindex/agent.py` | 80 to 105 | 5 to 15 | `5e840b9`'s 87/8 for the rule's agent half, plus the `build_agent` call, the shutdown drain and the `decider_counts` line |
| `src/semindex/gen.py` | 50 to 70 | 0 to 5 | `5e840b9`'s 77 |
| `src/semindex/evalgen.py` | 45 to 65 | 5 to 15 | `5e840b9`'s 63/13 |
| `src/semindex/evalrun.py` | 30 to 45 | 2 to 8 | `5e840b9`'s 26/4 |
| `src/semindex/evalagg.py` | 30 to 50 | 5 to 12 | one reader and one column. The precedent is `b7c1b4a` ("report the state poll rule and the re-decided interval in the aggregate"), which added `read_meta_state_poll_rule` and the `state_poll_rule` column at +50/-11 in `evalagg.py` and +132/-5 in its tests, read in this session; `1c3f758`'s 3/3 was a comment only and is not a basis for this row |
| **src total** | **585 to 805** | **32 to 90** | |
| **tests total** | **1050 to 1700** | | the precedents' ratio 1.8 to 2.1 on the src total; groups B, E and H are the heavy ones |

The dominant uncertainty is the deferral's share of `engine.py` and the test groups E and H, neither of which has
a precedent in this repository. **If the implementation lands above the top of these ranges, that is a signal that
the seam has grown a second mechanism and the design is to be re-read, not the estimate.**

Separately, the scripts of `evidence/25/`, which are not `src` and carry no test suite of their own:

| script | lines |
|---|---|
| `mdecider.py` (M's factory and shim) | 60 to 90 |
| `ldecider.py` (L's client) | 150 to 200 |
| `lserver.py` (the model server) | 180 to 240 |
| `fakes.py` (the constant, recording and delaying wrappers) | 60 to 90 |
| `parity.py` (test group H, checks 28 and 29, including the mismatch classification) | 160 to 220 |
| `arms.py` (the Acceptance 5 check of section 9) | 80 to 120 |
| `relay.sh` (the ssh forward and its probe) | 20 to 40 |

## 12. Whether this needs an ADR

**Yes.** Three of the four tests the repository applies to an ADR are met. It is hard to reverse: the
`state_poll_reschedule` row's meaning changes in decider mode (its `ts_ms` is no longer the instant it was
written, although `clamped` still measures against it, and `applied_interval_s` on a late row is not what was
waited), and every later reader of these logs
inherits that. It is surprising without context: a config field that names a file by path and runs it is code
execution by configuration. And it is a real trade-off, stated in section 7: a deferred answer is a different
decision rule in a countable set of cases, chosen over a loop stall that ADR 0004 already removed once.

ADR 0004's own line is the precedent: "Such a rule looks at whether the reply brought anything new, which the
policies of this ADR may not look at, so the stepwise model of this ADR does not cover it and its effect is read
from runs." A decider reading the whole six-field state plus an 8-row history goes further again, so the same
sentence is owed. The text is in appendix A, in this repository's ADR format, for the main session to file as
`docs/adr/0005-*.md`.

## 13. The order of work

### 13.1 Built and tested first

1. The synchronous seam alone: the config field, the loader, the two call sites, the history, the guard. With it,
   test groups A, B, C, D, F and G, and the `const` smoke run (group H check 30). **M's stage could run on this
   much**, which is the reason to build it first: it separates the seam's own risk from the deferral's.
2. The parity check, group H check 28, on the **M** smoke run of check 29 (a), with `recording(M)`. A `const`
   decider reads no field of `obs`, so it would check the plumbing and not that M can read the emulator's `obs`
   at all. Nothing proceeds past a mismatch classified as a defect.
3. The deferral, with test group E, and then the parity check again on run (b) of check 29, with
   `recording(delaying(M, latency))`. This is where the overlap and the `late` row are first exercised.
4. The L client, the server and the relay, with the measurements of section 7.3, and the parity check on L's own
   validity run, which is run (c) of check 29.

### 13.2 Measured before `spec.md` is frozen

In this order, because each one can stop the next:

1. **The import cost of `mdecider.py` in a node process**, against the 3 s startup margin
   (`gen.py:63`, enforced at `:793-797`): `m24` imports numpy and, through `charges`, `features` and `port`, also
   `../13/replay.py` and `../18/bound.py`. None of those four has module-level work beyond constants, imports and
   `sys.path` edits (checked in this session: no top-level call at column 0, and `__main__` guards at
   `../13/replay.py:1982`, `../18/bound.py:2588`, `../24/features.py:1050`, `../24/charges.py:1318`), but the cost
   is **not measured** and five node processes pay it at once on six cores.
2. **M's per-decision cost in the node process**, p99 not median, to confirm `../24/latency24.txt`'s 0.095 ms
   survives the move from the analysis interpreter into a live loop.
3. **Section 7.3's three measurements** for L.
4. **One full-length validity run per decider stage**, through `recording`, checked with `evalvalid`, with the
   parity check and with the Acceptance 5 script, before the 280-run sweep starts.

### 13.3 What sends the design back

- A group H check 28 mismatch that the classification of that check leaves as a **defect**. A mismatch explained
  by a non-zero `overlap` or `unanswered` count, by a `state_poll_late_reply`, by an undecodable reply or by a
  restart is a caveat of this document and not a verdict on the seam: it is reported with its count and the
  affected decisions, and the run plan continues. The two verdicts are deliberately not the same, because the
  first is a statement about the engine and the second about the network and the server.
- Any of group C's ten existing tests failing.
- A p99 round trip above 0.56 s, or any observation above 1.1255 s (section 7.3).
- `decider_counts` showing any `late` or `unanswered` decision in a validity run. This gate is readable only
  because of the shutdown drain of section 6.5: without it every pair's last deferral would be `unanswered` and
  the gate would fire on the stop boundary rather than on the server.
- `decider_counts` showing any `orphan`, which is a client bug and not a latency, or any `overlap` in a validity
  run, which says the measured round trip is above a pair's resolution period and therefore that section 7.3's
  reconsideration number has already been passed.
- An import cost that pushes a node's first record past the 1.5 s `RECORD_SCHEDULE_SLACK_S`
  (`evalvalid.py:83`, applied at `:430-436`) or its final counters line past the 1.0 s window of
  `counters_timing` (`evalvalid.py:266-308`).

**One thing the re-measured baselines cannot do.** The five baseline stages run in rule mode (F, T37r) or
single-choice mode (the three arms), and the only line of this seam any of them executes is the one guard of
section 6.1, which is false for all five. They bound testbed drift plus commit `1c3f758` plus that guard between
ticket 07's version `d009cfb` and this one; they do not validate the seam.
The arithmetic that differs between a baseline stage and a decider stage, exactly:

| | rule mode | choices mode, decider |
|---|---|---|
| send-time order | snapshot before the choice (`engine.py:514-518`) | snapshot before the choice (section 6.1) |
| jitter fraction | `rule.jitter_frac` (`:569`, `:689`) | `state_poll_choice_jitter_frac` (`:675`, `:738`) |
| resolution point on an answer | before the tracker fold (`:834-840`) | after the fold (`:843-852`) |
| reschedule row's state | two fields (`:667`) | the full state (`:680-684`) |
| interval source | the per-neighbour running value (`:571-596`) | the decider's return |

## 14. Threats

1. **The replay prices a decision at zero time.** `../13/replay.py:70-71` and its U[0, 100] ms tick lag
   (`:76`, `:645`) are the whole of its loop model, and 70.13 ms per resolution is 1.90 %, 1.17 % and 0.70 % of
   the three arms (arithmetic over `../23/latency.txt:126`). Acceptance 8's band is 6 % and the existing rules
   already consume 3.7 to 4.4 % of it. **What a reader may therefore not conclude: that an L-minus-M difference
   of the size of that bias is a difference between the deciders.** It is inside the budget the replay's own
   prediction error already occupies, and it falls on L alone.
2. **A late answer silently turns L into its held interval, feeds its own history a time that was not waited, and
   moves the next send row as well.** The schedule is the held interval's, the row's `interval_s` is the answer,
   and the history row that every later decision of that pair reads carries the answer's `applied_interval_s` as
   `waited_s` (`../22/observe.py:208`). The third part is the client's: `mem` is written when the answer arrives
   (section 8.2), so the send that went out while it was outstanding carries the pair's previous value and the
   send after it carries the answer, where the replay writes `L.mem[pair]` inside the resolution call
   (`../23/decider.py:777`) and its next send row always carries the answer. So a reader comparing send rows
   between the replay and the emulator sees a one-poll lag on exactly the late decisions. **What a reader may
   therefore not conclude: that a run with a non-zero `late` count measures L.** It measures L on the decisions
   that were answered in time and the held interval on the rest, in the schedule, in the history and in the next
   send row, and the counts of section 6.6 are the only place that is visible.
3. **One card, five node processes, one queue.** The queue arithmetic of section 7.2 (561 ms worst case) assumes a
   serialised server at the measured median, no other load on the card, and the measured decision rate of
   `../24/rows24.txt`. **What a reader may therefore not conclude: that the margin of section 7.2 is a measured
   margin.** It is arithmetic over one warm, uncontended, single-process measurement
   (`../23/latency.txt:165-178` says that measurement says nothing about contention).
4. **The baselines and the decider stages share a `src` version but not a mode.** Section 13.3's table lists the
   five arithmetic differences. The one part of the seam the baselines do cover is the guard of section 6.1,
   because they run through it. **What a reader may therefore not conclude: that the re-measured baselines show
   the seam changed nothing.** They show that the three arms and the two rules behave on this version as they did
   on `d009cfb`, which is a statement about the version and about one guard, not about the seam.
5. **The history the engine keeps is not the history the replay kept, it is the history the log shows.** The port
   rebuilds its own from its own resolutions (`../13/replay.py:1093-1102`) and `from_logged_rows` rebuilds it from
   logged rows in file order (`../22/observe.py:321-337`). The engine's is appended in emission order, which on
   the deferred path is later than the decision instant. The three agree exactly when every answer arrives before
   the pair's next resolution. With an answer still outstanding when the next resolution is asked, the engine's
   history lacks a row that `from_logged_rows` later has, and with an answer that never arrives it lacks one the
   port would have had. **What a reader may therefore not conclude: that a run whose parity check reported no
   defect had the port's history on every decision.** The two cases are counted as `overlap` and `unanswered`
   (section 6.6) and the check classifies a mismatch against those counts rather than failing on it
   (section 10, check 28), so a run with a non-zero count is a run on which some decisions were asked with a
   shorter history than the replay's, and only the counts say how many.
6. **The engine never waits, so a pathological server degrades the decider rather than the run.** The option not
   taken was a bounded wait when a pair's previous answer is still outstanding, which would keep the history
   complete at the cost of a loop stall in exactly the case the design exists to avoid. **What a reader may
   therefore not conclude: that a run whose counts are all zero proves the server was fast.** It proves that every
   answer beat its own margin, which is the only thing the counts can say.
7. **A decider file loaded by path is code execution by configuration.** Who can set it: whoever can write the
   per-node JSON config or pass `--poll-decider`, which is whoever can already start the sweep, and that account
   can already run arbitrary code as root through the ReCefore scenario (`CLAUDE.md`: every script requires root
   for Mininet's namespaces; `gen.py:851-872` launches the node processes). The digest pins which file ran, which
   is what a later reader needs. **What a reader may therefore not conclude: that the digest makes the mechanism
   safe outside this testbed.** It makes the run reproducible; it is not a trust boundary, and the repository's
   own warning that this baseline's validation never means authentication (`engine.py:11-13`) applies here too.
8. **The parity check covers three runs, and both of its sides come from one run each.** Resolutions only, one
   node's rows at a time, and the recorded inputs and the logged rows are built from the same `action` at the
   same instant (section 6.2), so the check compares the engine against its own log. **What a reader may
   therefore not conclude: that a passing check shows the emulator asks the replay's question.** It shows that
   the log of a decider run represents what that decider was handed. The replay's question is section 5's table
   and its three caveats, and a 280-run version of the check is affordable only as a sampled re-run of the
   script per stage, which `spec.md` should decide.
9. **An unanswered decision is a missing row, and Acceptance 8 counts rows.** A resolution whose answer never
   arrives emits no `state_poll_reschedule` row at all, so the window's decision count drops below what the
   replay predicts, and Acceptance 8 reads exactly that count. The shutdown drain of section 6.5 removes the one
   cause of this that is the design's own doing and not the server's; what is left is a genuine loss.
   **What a reader may therefore not conclude: that a decision-count miss in Acceptance 8 is a timing
   disagreement.** On an L stage it may be a decision that was never logged, and the `decider_counts` line is
   what separates the two.
10. **The emulator's own state differs from the replay's in two ways no test can remove, and they fall on L and M
    alike.** A reply arriving after its poll timed out rewrites `last_poll_status` and
    `consecutive_poll_timeouts` where the port writes them once (`engine.py:248-255`, `:302-304` against
    `../13/replay.py:1020-1021`), and an undecodable reply resolves its poll as answered with `empty` unknown, a
    combination the port cannot produce (`engine.py:812-818`). Both are cells of M's vector and words of L's text
    (section 5, items 2 and 3). **What a reader may therefore not conclude: that a difference between the
    replay's prediction and the emulator's run is a property of the decider.** Part of it is a property of the
    live engine's reply bookkeeping, measured by the per-run `state_poll_late_reply` count and the undecodable
    replies, and it biases the emulator against the replay for every stage of the comparison at once.

## 15. Every worker's choice, in one place

1. The config field holds the spec and the engine holds the object, with the invariant checked at construction
   (section 2). Not taken: the engine resolving the decider itself.
2. The digest of the loaded file lives in the run description, not in the file (section 3.1).
3. A `runs.csv` column for the decider, following the policy and the rule, not the redecide knob (section 3.3).
4. `semindex.decider` imported at module scope; the decider file imported only when the field is set
   (section 4). Not taken: a lazy import of the module.
5. The state handed over by reference under the port's contract, the history copied per call (section 5). Not
   taken: `copy.deepcopy` of the state.
6. `action_prob` 1.0 on both rows in decider mode (section 6.1). Not taken: `1 / len(choices)`.
7. The six counts in their own `decider_counts` event before the `counters` line (section 6.6). Not taken: six
   more keys on the `counters` line.
8. An exception from the decider kills the node (section 6.7). Not taken: a logged fallback interval.
9. `held_applied_interval_s` as a separate key on a late row, with `interval_s` kept as the answer (section 6.5).
   Not taken: `interval_s` as the interval that ran, which would make Acceptance 5 unreadable.
10. The L client's `mem` updated from every answer received, applied or not (section 8.2).
11. One FIFO queue over all connections on the server, batch 1 (section 8.4). Not taken: per-node fairness, or
    batching.
12. The server holds one `L` and calls `L.__call__` with a synthesized resolution-time `where` (section 8.4).
    Not taken: calling `text_of`, `prompt_of`, the read-out and `pick` directly, which would have left the
    tokenisation and `check_pick` re-derived or absent.
13. No `obs` digest and no decider identity on the logged row; the recording wrapper writes the four inputs
    themselves and not a digest of them (sections 8.5 and 9). Not taken: a digest, which can only say that two
    things differ.
14. The synchronous seam is built, tested and smoke-run before the deferral (section 13.1).
15. The pending store is keyed by the token and an overlap is counted, not refused (section 6.4). Not taken: one
    entry per pair, which the port can afford and this engine cannot; and not taken: refusing a second deferral,
    which would mean answering it synchronously.
16. `engine.py:529-533` is guarded rather than duplicated into the existing branches (section 6.1). Not taken:
    rebuilding the state after the decision, which would hand the row a different object from the decider's.
17. `send_start` is a factory argument of the L client, checked by the engine's own choice-set refusal and by
    `parity.py`, rather than imported from `../22/port.py` (section 8.2). Not taken: importing the constant,
    which would pull `replay` and `bound` into every node process.
18. An out-of-order resolution stamp within one pair raises (section 6.5). Not taken: counting the inversion,
    which leaves a file the labelling cannot read.
19. The bound of section 5 item 1 is enforced in the agent's config validation. Not taken: a line in `spec.md`.

## 16. Deviations from the brief

1. **The deferred path has three outcomes, not two.** The design handed to this file described an answer as
   either applied by today's arithmetic or late. A third exists: an answer applied while its poll is still the
   newest, but after `poll.sent_at + applied_interval` has already passed, which clamps to `col_now`
   (`engine.py:691`) where a zero-latency decider would not have clamped. It happens exactly when the answer is a
   shorter arm than the held interval. It is counted as `clamped_by_wait` (section 6.6) and the identity condition
   is stated with both clauses (section 7.1 item 4).
2. **`runs.csv` does not record the redecide knob**, so "the mode is recorded in `meta.json` and `runs.csv` the way
   the redecide knob is" could not be carried out as written. The knob is in `meta.json` only
   (`evalrun.py:669-673`) and `RUNS_FIELDS` has no column for it (`evalagg.py:1452-1502`). This file follows the
   policy's and the rule's precedent instead and adds a column, with the reason in section 3.3.
3. **Item (g) of the brief needs no code.** "The `empty` flag of a reply is computed in decider mode" is already
   true: decider mode requires `state_poll_choice_redecide`, so `_redecides_after_decode` (`engine.py:598-608`),
   `_redecides_on_resolution` (`:610-617`), `waits_for_fold` (`:827`), the `empty` computation (`:833`) and the
   `action["empty"]` key (`:700-707`) all fire as they do today. The consequence is the opposite of a saving and
   is recorded as a trap: the test at `engine.py:668` is **also** true in decider mode, so the decider branch must
   be placed before it (section 6.3).
4. **The estimate is larger than the scout's.** The scout priced the engine at 120 to 160 lines for a synchronous
   seam. The deferral is new arithmetic of about the rule's size, so section 11 prices the engine as two numbers
   and the src total lands at 585 to 805 added rather than 300 to 380.
5. **The protocol is three methods, so M is wrapped.** The brief says M "adapts nothing". M is a callable object
   (`../24/m24.py:477-505`), not an object with `decide`, so `mdecider.py` provides a four-line shim. Nothing of
   the observation is adapted, which is what the brief's sentence is about.
6. **The brief's `where` is four keys, not five.** `where.ordinal` is deliberately absent and a decider reading it
   raises (section 5). The brief did not say which of the port's `where` keys the engine provides.
7. **The pending state is keyed by the token, not by the pair.** The brief's item 3 proposes the neighbour and
   the poll's epoch as the token and the brief does not say how the engine stores it; the first draft of this
   document kept one entry per ordered pair, as the port does, and that cannot represent two outstanding answers
   of one pair, which this engine can reach (section 6.4). With it come two counts the brief does not name,
   `overlap` and `orphan`, and a rule for an answer whose entry is gone.
8. **`collect()` is drained once in `Agent.shutdown` as well as at the start of `on_tick`.** The brief's item
   4(f) says "at the start of every `on_tick`". `Agent.step` returns before `on_tick` at the stop boundary and
   `shutdown` never ticks (`agent.py:1175-1181`, `:1217-1236`), so the brief's wording alone loses every pair's
   last deferral and would make the design's own gate fire on the boundary rather than on the server
   (section 6.5).
9. **`clamped` is measured against the resolution instant on every row, late rows included.** The first draft of
   this document measured it against the collect instant, which made the field a statement about an instant that
   is not the row's stamp. Measuring it against `res_now` is a **return** to the brief's item 4(d), "exactly
   today's redecide arithmetic", and it keeps the field comparable with the port's (`../13/replay.py:1073-1075`)
   and with the fixed-arm stages' rows. The quantity the wait adds stays a count, `clamped_by_wait`
   (sections 6.5 and 9).
10. **The server calls `L.__call__` and does not re-derive the read-out.** This is also a return to the brief's
    item 6, "`evidence/23/decider.py`'s own class L and read-out, unchanged", which the first draft of this
    document had narrowed to the four module-level functions. Calling them would have skipped the tokenisation,
    because `forward_three` takes ids and the one-call wrapper is `readout` (`../23/decider.py:657-666`), and
    dropped `check_pick`, the tie rule the replay asserted on every decision (section 8.4).
11. **One shared line of the send path is guarded.** The brief's item 4(c) says the decision "replaces the draw
    from the choice stream and nothing else", and the first draft of this document claimed no shared line moves.
    `engine.py:529-533` runs unconditionally for all five modes, so the state assignment has to be guarded; the
    claim is now behavioural and is carried by group C's ten existing tests over the five modes (section 6.1).

## 17. What this document does not decide

1. **The share of unanswered, late or overlapping decisions at which a run is excluded.** This document requires
   zero in a validity run and reports the counts; the threshold for the sweep, and what to do with a stage that
   has one bad run, belong to `spec.md`.
2. **Whether the parity check of group H is repeated per stage or per run** (threat 8). This document fixes its
   three runs (check 29) and not its cadence over the sweep.
3. **Whether `evidence/25`'s scripts run under the node venv or the analysis venv** for the server, which depends
   on what `hpc-debian` has installed and is a measurement, not a choice.
4. **The stage names, the run counts and the choice set.** `spec.md`'s, after the measurements of section 13.2.
   Two preconditions the arms have to satisfy are fixed here and not the choice: `ARMS[1]` must be an element of
   `state_poll_interval_choices_s` (section 8.2), and the smallest applied interval must clear the bound of
   section 5 item 1.
5. **Whether the ADR of appendix A is filed as 0005**, and its final wording, which is the main session's and the
   user's.
6. **Whether the bound of section 5 item 1 is a refusal or a note.** This document puts it in the agent's
   config validation (worker's choice 19), which makes a stage that breaks it unlaunchable; the alternative is a
   precondition in `spec.md`, which makes it a reader's responsibility. The main session or the user decides.
7. **Three things outside this worktree**, which no statement here depends on: whether ReCefore's runner merges a
   node's stderr into `semindex_agent.log`, which is what the `traceback:{node}` flag of section 6.7 would need;
   whether a Mininet host sees the same `/tmp` as the ssh forwarding process (section 8.4); and everything about
   `hpc-debian`, the forward and the card, which is section 7.3.

## 18. What the first emulator runs changed (2026-11-27)

This section was written after the seam ran on the emulator for the first time. It records what those runs showed
to be wrong in the sections above, where the scripts deliberately differ from them, and the numbers sections 7.3
and 13.2 asked for. Nothing above was edited: a reader comparing the two sees what the design said before it met
the emulator.

Every run below is one run of learning seed 1 (p 0.2, k 4, `w0`, `--disconnect-duration-s 8 12`) in a store of
its own, `/home/lab_shared/t25store`, with the record directories in `/home/lab_shared/t25rec`. No run of seeds
121 to 160 exists yet. All were valid with no flag.

| stage | what ran | `src` commit |
|---|---|---|
| `t25b6` | one fixed arm, 6.0 s | `f7a1a31` |
| `t25c6` | a constant decider returning 6.0, through the recorder | `f7a1a31` |
| `t25m` | M, through the recorder | `f7a1a31` |
| `t25md`, `t25md1`, `t25md45` | M deferred by 0.07 s, 1.0 s and 4.5 s | `f7a1a31` |
| `t25n45` | M deferred by 4.5 s again | `1bb5ab1` |
| `t25l` | L, through the relay and the recorder | `1bb5ab1` |
| `t25m2` | M again, after the scripts were corrected (18.4) | `b70f5e8` |
| `t25l2` | L again, after the scripts were corrected (18.4) | `b70f5e8` |

### 18.1 Statements above that the runs showed to be wrong

1. **Section 5 item 3 cannot be counted the way it says.** It says an undecodable reply "writes `state_rx` with
   `hi == {}`" and is "countable from the log with no new counter". A valid reply from a neighbour that holds
   nothing yet writes the same row: `t25c6` has two to six such rows per node and not one reschedule row with
   `trigger` answered and `empty` null. The count has to be taken from the reschedule rows (answered with `empty`
   null), which is what `parity.py` prints per node.
2. **Section 13.3's fourth bullet fires on the end of the run.** It sends the design back on any `unanswered`
   decision of a validity run and says the shutdown drain is what makes that readable. The drain collects once and
   does not wait, so a resolution asked less than the decider's own latency before the stop stays pending: its
   answer has not arrived yet when the drain looks. The loop tick does not add to that window; it adds only to
   the wait of an answer released while the loop is running. In `t25md45` a node
   whose last resolution came 0.6 s before the stop left two unanswered. The user decided (ticket, decision 9) to
   leave the run's behaviour alone and to record, per unanswered resolution, when it was asked: the
   `decider_counts` row now carries `unanswered_asked` (commit `1bb5ab1`). In `t25n45` the three unanswered
   resolutions were asked 1.3 s, 1.7 s and 3.4 s before the row was written, all inside the 4.5 s delay. Where the
   line between "the run ended" and "the server did not answer" falls is `spec.md`'s.

### 18.2 Where the scripts differ from the sections above, on purpose

1. **`parity.py` excuses less than check 28 allows.** Check 28 lists a `state_poll_late_reply`, an undecodable
   reply and a restart as explanations of a mismatch. Within one run the `state` and `resolved` a decider is
   handed are the very values its row logs, taken at the same instant, so no event of the log can make the two
   sides differ; a difference in either is reported as a defect. The first two are still counted per node,
   because those counts are what section 5's caveats need. A second `agent_start` line excuses nothing either:
   no stage of this comparison restarts an agent, and an excuse keyed on one extra line would cover every history
   difference of the node, those before the restart included.
2. **A history is checked against the instant of the ask before it is checked against the log's row order.**
   The engine appends an exchange to a pair's history when its row is written, which for a deferred answer is
   when the answer is applied. So the history at an ask is "the pair's rows already written", and it equals
   "the pair's earlier rows in file order" only when no two resolutions of the pair overlap. `parity.py` rebuilds
   the first from `decided_at_ms` and treats any difference as a defect, with no excuse; a difference from the
   row-order rebuild, which is what M was fitted on, is then the `overlap` caveat and nothing else, and at most
   as many decisions of a node are excused that way as the node counted overlaps. In `t25md45` the 189 recorded
   histories equal the as-of-ask rebuild in all 189 cases, the six resolutions with no logged row included, with
   72 overlaps.
3. **An unanswered resolution is matched by name.** With `unanswered_asked` in the log, a recorded resolution
   with no logged row is excused only if the row names that resolution, and a named one that was never recorded
   or has a row is a defect. On a log written before the key existed the excuse is a budget of `unanswered` many.
4. **`arms.py` reads send rows as well as reschedule rows.** Section 9 says the check reads the reschedule rows.
   A fixed-arm stage writes no reschedule row at all (`t25b6` has none), so that check would pass such a stage
   without reading one interval. Membership is checked on both kinds, and so is the comparison with what the
   decider returned: the recorder holds the value returned at every send, which joins to the send row on the
   neighbour and the instant. Given the jitter fraction, the check also requires every applied interval to lie
   inside its arm's jitter band, which nothing else in either check reads.
5. **An out-of-choices answer raises even when its token matches no pending entry.** Section 6.7 says such an
   answer is counted and dropped. The engine checks the value before it looks the token up, and that is kept: a
   decider that returns a value outside its own choice set is wrong whichever answer it happens on.
6. **The client's factory is `build(node, socket_path, send_start, stats_dir, record_dir=None)`**, two arguments
   more than section 8.2: the counts of section 8.3 need a place outside the node's own directory, and a validity
   run needs the recorder. Its memory is keyed by the neighbour alone, which is the pair for one client per node.
7. **The local socket is `/home/lab_shared/t25sock/l.sock` and not under `/tmp`** (section 8.4). A node reaches a
   socket on the shared file system; whether a node sees this machine's `/tmp` was never tested, which leaves the
   second item of section 17.7 open and unneeded.
8. **M's entry file pins thirteen files, not one.** Section 8.1 pins `mdecider.py` through the config's digest.
   Importing M loads twelve evidence files, so `mdecider.py` refuses to start unless each of them and the
   coefficient file hash to a table in it. Seven of the thirteen digests are the ones M's scoring run recorded,
   and an eighth, `22/observe.py`, is the one L's scoring recorded. For the other five no recorded digest exists;
   they have not changed in git since before M's scoring commit, and what their pins enforce is that every run of
   the comparison loads the same bytes. `lserver.py` does the same
   for L, plus the adapter's two files, the three library versions and the archived logits of the repeat row.

### 18.3 What sections 7.3 and 13.2 asked to be measured

**M (section 13.2 items 1 and 2).** Loading `mdecider.py` in a node process, five at once: 0.150 to 0.241 s over
the five runs that load it, against the 3 s margin. One resolution-time decision inside the live loop, per run:
median 0.25 to 0.33 ms, p99 0.45 to 0.58 ms, largest 0.76 ms. The constant decider consumed no draw of the engine's stream: `t25c6` and
`t25b6` sent the same number of polls on every pair, and 148 of their 174 send rows carry the same applied
interval to the last digit.

**The deferred path, without a model.**

| delay | asked | applied | late | unanswered | overlap | clamped by the wait | wait the engine saw, ms (min, median, max) |
|---|---|---|---|---|---|---|---|
| 0.07 s | 148 | 148 | 0 | 0 | 0 | 0 | 70, 101, 122 |
| 1.0 s | 149 | 149 | 0 | 0 | 0 | 0 | 1001, 1004, 1095 |
| 4.5 s | 189 | 99 | 84 | 6 | 72 | 25 | 4502, 4517, 4596 |

The wait the engine sees is the decider's latency plus the time to the next loop pass, up to one tick. The 4.5 s
row is `t25md45`; its rerun on the later commit, `t25n45`, asked 188, applied 103, with 82 late, 3 unanswered, 72
overlaps and 28 clamped by the wait. Both are far outside any margin of section 7.2 and exist to reach the late
and the overlapping answer: neither raised anything and both stayed valid.

**L (section 7.3).** The server on the GPU machine, `torch 2.15.0+cu132`, `transformers 5.24.0`, `peft 0.22.0`,
the adapter's two digests as the tuning and the scoring recorded them, the repeat row read out as
`[18.5, 17.75, 19.25]` twice at start-up.

| quantity | value |
|---|---|
| cold first pass at start-up, three starts | 376.6, 386.7, 373.4 ms |
| warm pass at start-up | 69.7, 71.3, 68.7 ms |
| replay of `t25m`'s 151 resolutions from five clients at their recorded instants, from the host through the relay: round trip median, p99, largest | 72.3, 488.8, 564.6 ms |
| the same with the first second left out (143) | 72.3, 148.0, 212.6 ms |
| the same replay at four times the rate | 72.6, 498.9, 568.5 ms |
| the server's own compute time in that replay: median, p99, largest | 70.4, 78.6, 80.1 ms |
| `t25l`, the wait the engine saw (146): min, median, p99, largest | 75, 101, 609, 705 ms |
| `t25l` with each node's first second left out (138): median, p99, largest | 101, 201, 212 ms |
| `t25l`, the client's `decide` at a resolution: median, largest | 0.05, 0.24 ms |
| `t25l`: applied, late, unanswered, overlap, orphan, clamped by the wait | 146, 0, 0, 0, 0, 0 |
| `t25l2`, the wait the engine saw (148): min, median, p99, largest | 72, 101, 508, 602 ms |
| `t25l2` with each node's first second left out (140): median, p99, largest | 101, 201, 210 ms |
| `t25l2`: applied, late, unanswered, overlap, orphan, clamped by the wait | 148, 0, 0, 0, 0, 0 |

**The slow answers are one event, the start of the run.** The first polls of all eight ordered pairs are sent
within 19 ms of each other in `t25l` and all eight are resolved within 25 ms of the first send, the server answers
one request at a time, and the eight answers therefore come back as a staircase of about 70 ms a step. That is
where the replay's p99 and largest value come from, and both of `t25l`'s waits above 0.56 s.

**What bounds the staircase is a condition, not the number of pairs.** The server's queue holds one request per
resolution still unanswered, and the engine admits a second one for a pair whose first is outstanding (it counts
it as `overlap`). The queue stays at eight or fewer as long as every answer returns before its pair's next
resolution, which is at least 3.3255 s after the previous send: eight requests at the largest compute time seen
in these runs, 102.5 ms, is 0.82 s. A server slower than about 0.4 s a request would let a pair ask twice, and
the staircase could then grow. The run reports this itself: `overlap` is zero in `t25l`.

**Read against section 7.3's reconsideration number.** The number is on the round trip: a p99 above 0.56 s, or
any observation above 1.1255 s. The round trip's p99 is 488.8 ms and nothing is above 1.1255 s, so by its own
wording the design is not reopened. Two things have to be said beside that. The margin is thin, and it is thin
because of the staircase alone. And the quantity that decides whether an answer is late is the wait the engine
sees, which adds up to one tick to the round trip and whose p99 in `t25l` is 609 ms, above 0.56 s. Neither of the
two waits above 0.56 s can make an answer late: they belong to a pair's first resolution, when the pair holds
6.0 s and its next poll is at least 5.4 s after the send. The user decided (ticket, decision 10) that the margin
is measured on the round trip and that the design is not reopened. The deadline an answer has to meet is the
pair's next send, 1.1255 s away at the least; 0.56 s is a guide set at half of that and not the deadline itself.
What each run reports is the number of answers that missed the deadline (`late`) and of those the wait alone
pushed past it (`clamped_by_wait`); what a run with either does to the comparison is `spec.md`'s.

**Not measured.** The round trip from inside a node with nothing between the client and the clock: `t25l` gives
the engine's wait, which includes the tick, and the replay ran from the host. A run in which the relay or the
server dies part way. A reply the agent cannot decode, and a restart of an agent, on the emulator: neither
occurred in these runs. That a node's stderr reaches `semindex_agent.log` (section 17.7, first item) is read from
ReCefore's runner, which sends both streams to the log file when one is given; no node has raised in a run yet.

### 18.4 What an independent review changed, and what it left

Six reviewers, one question each, read the scripts, the source commit and this section; every finding was then
handed to a second reader whose task was to refute it. Of 63 findings, 15 stood as written, 46 stood once an
overstatement was removed and 2 were refuted. No number of this section failed to reproduce from the raw files.
Three explanations did, and are corrected above: the width of the window at the stop (18.1 item 2), what bounds
the staircase (18.3), and whether a link cut bears on the unanswered count (the ticket, decision 9).

**Changed in the scripts.** The server no longer ends on one line it cannot decode or parse. It hashes the pinned
files before importing them, reads no bytecode cache (the GPU machine's checkout carries caches from the earlier
scoring runs, and a cache is used whenever its header matches the source), takes the repeat row's expected logits
from the tuning archive instead of a typed constant, logs the logits it read out, refuses a request that is not a
resolution, and removes a stale socket only if it is a socket. The client validates a token and an interval
instead of coercing them and can no longer raise out of one bad line. `parity.py` and `arms.py` fail on a node
with no log, on a record file for a node the run does not have, on a decider run with no `decider_counts` row and
on a count of logged rows that differs from applied plus late; a second `agent_start` excuses nothing; the
overlap excuse is a budget; send rows are compared with what the decider returned; a second release of one token
must be matched by the engine's `orphan` count; and, given the jitter fraction, every applied interval must lie
inside its arm's band. `t25m2` and `t25l2` are the reruns after these changes, and all nine decider runs pass both
checks.

**Left as they are**, because the second reader found each unreachable with this client and this server, and a
guard for a failure that cannot occur is code that has to be kept true for nothing:

- A server that streams answers nobody asked for makes the client's memory grow and one loop pass long.
- A peer that stops reading without closing leaves the client connected and its requests dropped.
- More than about a thousand simultaneous connections end the server.
- A node built before the forward exists counts every decision unreachable for the whole run; the launcher
  probes the server through the forward before each run instead.
- The server keeps one `L` for its lifetime; its per-pair memory is written and never read, because no client
  sends a send-time request and the server now refuses one.
- A file that changes between its hash and its import is not caught.

**Not shown by either check**, and so not to be read into a pass: that every pair kept being polled to the end of
the run; that the interval a late row says was held is the one that ran; and anything about a log and a record
altered consistently with each other. The first belongs in `spec.md` if the comparison needs it.

## Appendix A: the ADR text, in this repository's format

> # A state-poll interval may come from a decider injected into the agent
>
> Stage 2 compares three ways of deciding a state poll's interval: a tuned language model (L), a policy learned
> from the logs (M) and a fixed formula rule (F). The replay this comparison was scored on asks a decider one
> question per decision, as an injected callable handed the decision's whole observable state
> (`docs/wayfinder/semindex-stage2/evidence/13/replay.py`'s `ask_decider`). The agent had no such door: its
> interval came from whichever of four exclusive config fields was set, and three of those four read almost none
> of the state the decision logs. The agent now accepts a decider object at construction, selected by one config
> field that names a file, its sha256 and a factory in it, and rides on the existing choices-plus-re-decision
> mode, replacing the uniform draw and nothing else. The five existing modes behave as they did, one shared line
> of the send path is newly guarded, and tests pin the behaviour rather than the diff.
> The decider's own code stays outside this repository's `src`: the file scored in the replay is the file the
> emulator loads, so `dependencies = []` holds and a Mininet host imports numpy or a socket client only in a run
> that asked for one.
>
> Such a decider reads the whole six-field decision state and, in addition, a bounded history of its pair's last
> eight resolved polls, which the agent did not keep before and which only a decider reads. ADR 0004 restricts
> its own policies to the newest poll's status and the count of consecutive timeouts, and already records that
> the 2026-11-18 rule went beyond that restriction. A decider goes further again: the stepwise model of ADR 0004
> does not cover it, and its effect is read from runs only.
>
> One decider cannot run inside the agent loop. L costs about 70 ms per decision on a datacenter card, the loop
> is a single thread with one cefpyco receive call, and a synchronous call of that size inside it is the defect
> ADR 0004's Consequences record as measured and removed once. So a decider may answer later: it returns nothing
> at the resolution, the interval the pair already holds keeps the next poll armed, and the answer is applied
> when it arrives. Because a re-decided poll is scheduled from its own send time, an answer that arrives before
> that provisional timer fires produces the identical send instant a zero-latency answer would. The engine keeps
> one pending entry per outstanding request, keyed by the neighbour and the poll's epoch, so an answer is applied
> with its own poll's send stamp and send draw even when two of one pair's answers are outstanding at once, and
> an answer for no pending request is dropped. The answers that arrive too late, that clamp only because of the
> wait, that were asked while the pair already had one outstanding, that match no request and that never arrive
> are counted per node and written into the run's own log. The queue is drained once more when the node stops,
> because the loop's last step returns before its tick.
>
> ## Considered Options
>
> - **A fifth exclusive config mode of its own.** Rejected: it would duplicate the four resolution gates the
>   re-decision path already provides, and the three fixed arms it is compared against would stop sharing the
>   choices-mode jitter arithmetic, so the identity between a constant decider and a single-choice run would no
>   longer hold.
> - **The decider's code inside `src`.** Rejected: it would put numpy, and for one candidate a model client, into
>   every Mininet host's import path, and the file scored in the replay would no longer be the file the emulator
>   runs.
> - **A synchronous call to a model service, with a timeout and a fallback interval.** Rejected: it blocks the
>   same loop, now with five nodes' queueing added, and a fallback interval that fires silently replaces the
>   decider on the rows the comparison is read from.
> - **Pre-expanding the decider into a table keyed on the logged state.** Not available: both candidates read a
>   261-cell feature vector or its rendered text with an eight-row history, of which 93.5 % of inputs are distinct
>   (`evidence/24/features.txt:86-92`), so there is nothing to tabulate. Only the scalar state a `state_poll_policy`
>   reads is enumerable, and that is explicitly not what these deciders read.
> - **Waiting for a pending answer when the next resolution of the same pair arrives.** Kept as the contingency if
>   the measured round trip approaches the margin: it keeps the pair's history complete at the cost of a stall in
>   the one case the deferral exists to avoid.
>
> ## Consequences
>
> - A config field naming a file by path, with its digest, is code execution by configuration. It is acceptable
>   here because the account that can write a node's config or pass the run flag already runs the testbed as root
>   through Mininet; the digest exists to say which file ran, not to make the mechanism safe elsewhere.
> - In decider mode a `state_poll_reschedule` row's `ts_ms` is the resolution instant, which can be earlier than
>   the instant the row was written, and the row carries `decided_at_ms`, `late` and, when late,
>   `held_applied_interval_s`. `clamped` keeps its meaning, a comparison against that same resolution instant, so
>   it stays the field the replay and the other stages carry. A row of any other mode gains nothing. Rows of a
>   run with a non-zero late count describe a schedule the decider did not set, and only the per-node count makes
>   that visible. Within one ordered pair the stamps must stay in order, and the agent refuses to write a row
>   that would break it, because an out-of-order pair of rows makes a negative age in every later reading of them.
> - The agent now keeps a bounded per-pair history of resolved polls, in decider mode only. It is appended outside
>   the decision-logging gate, because the history is an input to the next decision and logging must not change
>   behaviour. On the deferred path it is appended when the answer arrives, so a decision asked while an answer
>   is outstanding is asked with a history shorter by that row, and a decision whose answer never arrives is in
>   no history at all. Both are counted.
> - The two state fields that describe the newest poll are not the replay's function of the same name. A reply
>   that arrives after its poll timed out marks that poll answered and clears the timeout count, which the replay
>   cannot do, and an undecodable reply resolves a poll as answered with the reply's content unknown, which the
>   replay has no state for. Both are read by every decider and both are counted from the run's own log.
> - The deferred path is a different decision rule in a countable set of cases. A run is read as the decider's
>   only together with its counts of late, clamped-by-wait, overlapping, orphaned and unanswered decisions.
