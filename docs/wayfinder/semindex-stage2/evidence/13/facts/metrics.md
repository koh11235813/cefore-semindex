# The two yardsticks for a replay: bytes and delay (facts from source + logs)

Scope: the 14 stages `s5a s5b0p5 s5b1p359 s5b3p695 s5b10p043 s5c s5short s5wait s6b0p5 s6b1p359
s6b3p695 s6b10p043 s6short s6wait` under `/home/lab_shared/semindex-eval/`.
Source read read-only at `/home/lab_shared/cefore-semindex/src/semindex/`.
Scripts (json only for log reads, no `semindex` import):
`(scratch)/verify.py`
and `.../ev13/cells.py`.

## 0. Commit state

- `git -C /home/lab_shared/cefore-semindex log -1` → `d19a0a21` (2026-11-16). `5365e079` is an
  ancestor of HEAD and `git diff --stat 5365e079..HEAD -- src/semindex/` is **empty**: every source
  file cited below is byte-identical to what the s6 runs used.
- The runs were **not** all made at `5365e079`. `meta.json` records the commit per run:
  `s5short/p0.2/1` → `semindex_git_commit 7d5cd80e…`, `s6short/p0.5/2` → `5365e079…`; both
  `semindex_git_dirty false`, `recefore_git_commit 25ad4e94…`, `k = 4`. This matches
  `evidence/09/README.md:7` (s5* at 7d5cd80) and `map.md:103` (s6* at 5365e07).
- Between the two: `49f6d68` + `5365e07`, touching only `evalagg.py`, `evalgen.py`, `evalrun.py`
  (the second disconnect shape). `agent.py`, `engine.py`, `tracker.py`, `names.py`,
  `holder_state.py`, `transport.py`, `cefpyco_transport.py` are unchanged across 7d5cd80..HEAD,
  so **s5 and s6 runs were produced by the same agent/engine code**; only the generator and the
  aggregator differ. (Verified by the diffstat above, which lists no other file.)

## 1. Bytes

### 1.1 What is on the wire for one state poll

- Request: one Interest, no payload. Name
  `f"{PREFIX}/{group}/{holder}/sv/{requester}/{epoch_ms}"` with `PREFIX = "ccnx:/semidx"`
  (`src/semindex/names.py:102`, `names.py:24`). `group` is the run's group string, e.g.
  `s5short_p020_seed1` (from `<run_dir>/<name>/h0.json`, key `group`). `epoch_ms` is the poll's
  own send time in ms, so **the request's byte count grows with the group name and with the digit
  width of the epoch** - 54 B for `s5short_p020_seed1` (matches `evidence/09/handcheck.txt`'s
  hand check of 116 polls), 53 B for `s5wait_p000_seed5`, 56 B for `s5b10p043_p040_seed3`.
- Reply (only if the holder answers): one Data under **the same name**, plus a payload
  `holder_state.encode(progress)` - canonical JSON, sorted keys, no whitespace
  (`holder_state.py:27-29`), one entry per origin the holder holds anything of:
  `{"<origin>":{"extra":[...],"hi":N,"truncated":bool}}` (`holder_state.py:46-52`).
  So the reply **does** grow with the number of origins and **does** carry the sparse `extra`
  list (capped at `MAX_EXTRA_PER_ORIGIN = 64` per origin, whole thing capped at
  `MAX_ENCODED_BYTES = 8192`, `holder_state.py:17-18`, extras dropped origin by origin if needed,
  `holder_state.py:58-66`). An empty state encodes as `{}` = 2 B.
- Byte totals per poll, as the evidence counts them (`evidence/05/attrib.py:266,274`):
  `int_B = len(state name)` for every poll; `rep_B += len(state name) + len(payload)` for every
  reply, attributed to the latest poll at or before the reply's rx ts. A **timed-out poll costs
  the request name only**; an answered poll costs `name + name + payload`.
  This is the requester's view: a reply the holder sent but the requester never received is
  charged as a timed-out poll.
- The payload length is **not in events.jsonl** - `state_rx` logs only `hi`
  (`{"event":"state_rx","hi":{},"neighbor":"h0","ts_ms":…}`). `attrib.py:207-226` (`reply_recon`)
  reconstructs it by replaying every node's `record_stored`/`record_created` timeline through
  `holder_state.encode`. A simulator must model the payload, it cannot read it back.

### 1.2 The constants

- `INT_B = 50.9` B - the mean state-request name length of `s4a` seeds 1–100, i.e. a name rebased
  to the reference group length `REF_G = 14.92` (`evidence/09/replay.py:26`,
  `evidence/09/ana.py:9,27`). It is a *normalized* name cost, not any one run's 53/54/56 B.
- `REP_B = {0.0: 132.5, 0.1: 131.7, 0.2: 130.4, 0.3: 128.2, 0.4: 125.5}` B - mean
  (reply name + payload) per answered poll, measured on `s4a` (`evidence/09/replay.py:27`).
  **There is no value for p 0.5 / 0.6**, so `B_const` and `empty_B` do not exist for 12 of the
  42 cells (`evidence/11/README.md:25,148`). Measured per-reply bytes also differ by arm
  (138.4 B at 0.5 s vs 127.1–127.3 B at 10.043 s, `evidence/09/README.md:126`), which `REP_B`
  flattens.
- `B_const = (polls · INT_B + answered polls · REP_B[p]) / record-hops`, pooled over a cell's runs
  before dividing (`evidence/09/ana.py:11,59`; `evidence/11/emptypoll.py:86-90,402`).
  `polls = Σ_h ngap_h`, `answered = Σ_h (ngap_h − nnorep_h)`, `record-hops = Σ_h nu_h`
  (uncensored record-hops, `evidence/05/attrib.py:301-302,404`).
- `empty_B` (column `empty_B_per_rh_no_new`) `= rx_no_new · (INT_B + REP_B[p]) / record-hops`:
  the same constant price charged to the answered polls whose reply held nothing new up to `hi`
  (`evidence/11/emptypoll.py:90-95,415`). It is a **perfect-foresight reference**, not an
  achievable saving: `rx_no_new` ignores the neighbor's sparse `extra`, and 9 such replies over
  all cells did start a fetch above `hi` (`evidence/11/README.md:94,119`).
  `empty_B_per_rh_hi_unchanged / _after_complete / _no_new_rest / _why_*` split the same quantity
  by other poll classes at the same unit price (`emptypoll.py:415-421`).
  `excess_B_per_rh = B_const(policy) − B_const(10.043 arm, same p, same seeds)`;
  `ratio_no_new_over_excess = empty_B / excess_B` (`emptypoll.py:424-427`).

### 1.3 Formula, and how it relates to the counters

For one run, over the window `[t0+10 s, t0+130 s]`, both edges inclusive:

```
state-poll bytes (measured)  = Σ_polls len(request name)            # int_B
                             + Σ_replies (len(name) + len(payload)) # rep_B
state-poll bytes (B_const)   = n_polls · 50.9 + n_answered · REP_B[p]
                             = n_answered · (INT_B + REP_B[p]) + n_timed_out · INT_B
                               with n_timed_out = ngap − answered = Σ_h nnorep_h
B_const per record-hop       = that / Σ_h nu_h
```

`n_polls = ngap = polls_sent` = `decision` lines of kind `state_poll` in the window - the engine
logs exactly one per Interest sent, and a re-decision is logged as `state_poll_reschedule`
instead (`evalagg.py:990-1001`). `n_answered = ngap − nnorep` is attrib's "latest poll ≤ reply ts"
match, which is **not identical** to counting `state_rx` lines: it differs in exactly
7 of the 42 cells (s6b0p5 p0.2 −2, s6b1p359 p0.2 −1, s6wait p0.2 −1, s5short p0 −1,
s5b10p043 p0.2 −1, s6b0p5 p0.5 −1, s6b0p5 p0.6 −4), and `ngap − polls_sent` is 0
in all 42.
2026-11-17 correction: this said "5 of the 42 cells" and listed five, read off
`cells.csv`'s `perrun_answered_minus_state_rx`. That column is **blank for the twelve
s6 cells at p 0.5 / 0.6** (evidence/11 fills it only where it computed a `B_const`),
so reading the count off it silently skipped 12 of the 42 cells. Recomputed from
`evidence/09|10/perrun.csv` Σ_h (ngap_h − nnorep_h) against `evidence/11/cells.csv`
`state_rx` for all 42 cells, the count is 7 and the largest difference is −4, not −2. A late reply can also be logged separately as `state_poll_late_reply`.

**The `counters` lines are not state-poll bytes.** `control_bytes_sent/received` count the whole
sync protocol: state Interests *and* record-fetch Interests (`cefpyco_transport.py:232`), every
Interest this node serves (`:405`), every Data it serves as holder (`:438`) and every sync Data it
receives (`:452`). Only the content-body chunk fetch is excluded (`:333`, module docstring
`:47-63`). In-window counters are `final counters − counters_snapshot`, life-aware
(`evalagg.py:717-795`). Measured on 6 runs (see below), state polls are **3.5–50.6 %** of
`control_bytes_sent`; the residual is 282–289 kB in every run regardless of poll count, i.e. the
240 in-window records × 4 hops of record Interests and record Data. A replay scored on bytes must
therefore be scored on the *state-poll* model, not on the counters.

### 1.4 Verification

`verify.py` (json only) on 6 runs from 5 stages / both outage shapes
(`s5short p0.2 s1`, `s5wait p0 s5`, `s5b10p043 p0.4 s3`, `s6short p0.5 s2`, `s6b0p5 p0.2 s4`,
`s6wait p0.6 s7`), against `evidence/09/perrun.csv` and `evidence/10/perrun.csv`:

| quantity | agreement |
|---|---|
| `polls_sent` (state_poll rows in window) | 6/6 exact |
| `state_rx` (state_rx rows in window) | 6/6 exact |
| `int_B` = Σ len(request name) | 6/6 exact |

Residual against the counters, per run (`int_B + rep_B` from perrun.csv vs in-window
`control_bytes_sent`):

| run | int_B+rep_B | ctl_sent | residual | state-poll share |
|---|---|---|---|---|
| s5short p0.2 s1 | 16 863 | 305 907 | 289 044 | 5.5 % |
| s5wait p0 s5 | 17 325 | 304 629 | 287 304 | 5.7 % |
| s5b10p043 p0.4 s3 | 10 380 | 293 598 | 283 218 | 3.5 % |
| s6short p0.5 s2 | 16 000 | 305 198 | 289 198 | 5.2 % |
| s6b0p5 p0.2 s4 | 294 087 | 581 344 | 287 257 | 50.6 % |
| s6wait p0.6 s7 | 13 021 | 295 375 | 282 354 | 4.4 % |

`cells.py`: `B_const` recomputed from `perrun.csv` by the formula above reproduces the published
`live_main.csv` value in **43/43** cells that have a `REP_B` (max |diff| 5.68e-14 B - the same
figure `evidence/11/README.md:137` reports).

## 2. Delay

### 2.1 The metric the evidence reports for these stages

Name: **4-hop discovery delay** (`delay` in `evidence/09|10/live_main.csv`, `seq_model.direct(S,4)`,
`evidence/09/ana.py:5`). Definition:

- A **discovery pair** is (origin, seq, node) for every node other than the origin
  (`evalagg.py:1102-1130`). `hop_distance = |index(node) − index(origin)|` on the line
  h0-h1-h2-h3-h4, so the 4-hop pairs are exactly origin h0 → node h4 and origin h4 → node h0.
- A record counts only if its `record_created` ts is inside `[t0+10 s, t0+130 s]`, both edges
  inclusive (`evalagg.py:1082-1084`, window constants `evalgen.WINDOW_START_S/WINDOW_END_S`).
- `latency_ms = earliest record_stored ts at the node − record_created ts at the origin`.
- **Censored** iff the node has no `record_stored` for that record, or the earliest one is after
  the window end (`evalagg.py:1111`). A store exactly at the window end is uncensored.
  Censored pairs carry `follow_up_ms = window_end − created` and no latency.
- `delay` = Σ latency over **uncensored** 4-hop pairs ÷ their count, pooled over the cell's runs.
  `delay_seedmean` = mean over seeds of each seed's own pooled value (the hull/frontier
  convention). `cens4` = censored 4-hop pairs / all 4-hop pairs.
- **Unfinished records are dropped from `delay`,** so a cell with heavy censoring looks faster
  than it is: the 10.043 s arm loses 6.58 % of its 4-hop pairs at p 0.4 and 22 %/32 % at p 0.5/0.6
  (`evidence/09/README.md:92`, `map.md:105`), and the evidence says to discount those cells.
- A second, censoring-aware family exists in evalagg: Kaplan–Meier quantiles over the same pairs
  (`survival.csv`, `summary.csv` median/q1/q3, `evalagg.py:41-49,566-612`). The s5/s6 policy
  tables do **not** use it; they use the uncensored mean above.
- Reward ingredients, per `map.md:126-129`: reward = weighted combination of **delay** and
  **control traffic**, with delay weighted by hop count, the hop weight applied by the aggregator
  (the agent does not know hop count); name-importance weighting is explicitly out of scope.
  No reward formula is fixed yet.

### 2.2 Where the numbers already live

| level | file | columns |
|---|---|---|
| per cell, s5 | `evidence/09/live_main.csv` | `delay`, `delay_seedmean`, `cens4`, `B_const`, `B_norm`, `polls`, `waste`, with `_se` |
| per cell, s6 | `evidence/10/live_main.csv` | same |
| per run | `evidence/09/perrun.csv`, `evidence/10/perrun.csv` | `n4, lat4, c4` (and `n1..3`), `nu_h, intB_h, repB_h, ngap_h, nrep_h, nnorep_h`, `polls_sent, state_rx` |
| per pair | evalagg `<stage>/discovery.csv` (rebuilt, not stored: ~47 MB) | one row per (run, origin, seq, node): `hop`, `latency_ms`, `censored` |
| per cell, polls | `evidence/11/cells.csv`, `evidence/12/cells.csv` | `polls_sent, state_rx, rx_no_new, …, B_const, empty_B_per_rh_*, excess_B_per_rh` |
| per run, raw | `<run>/results.json`, `<run>/meta.json` | link/record schedule outcomes, validity verdict |

`results.json` does **not** hold the delay; it holds the scenario outcome records that
`evalvalid` checks (agents started, put succeeded, records on schedule, link events fired -
`evalvalid.py:1-30`). `meta.json.validity.valid` is the run's pass flag.

### 2.3 Recipe from events.jsonl alone

1. `t0` = the `t0_ms` of any `agent_start` line (all nodes agree; this is evalagg's first and
   winning rung, `evalagg.py:445-480`). Window `[t0+10 000, t0+130 000]` ms.
2. `created[(origin, seq)]` = the `record_created` ts, read from whichever node's file holds it.
   Only the origin ever logs `record_created` for its own records, so each key is written at most
   once in the whole run (`analyze.py:72-84`, used by `evalagg.py:1081`). Keep only those inside
   the window.
3. `stored[(origin, seq, node)]` = **earliest** `record_stored` ts at that node (first store wins,
   `evalagg.py:1095-1098`).
4. For each kept record, take the node 4 hops away (h4 for origin h0, h0 for origin h4):
   censored if missing or `ts > window_end`; else latency = `ts − created`.
5. `delay = Σ latency / count`, `cens4 = censored / (uncensored + censored)` - pooled over the
   cell's runs, not averaged per run.

### 2.4 Verification

`verify.py` on the same 6 runs reproduces `perrun.csv` exactly:

| quantity | agreement |
|---|---|
| `n4` (uncensored 4-hop pairs) | 6/6 exact |
| `c4` (censored) | 6/6 exact |
| `lat4` (ms, summed) | 6/6 exact |

`cells.py` pools `perrun.csv` to cell level: `Σlat4/Σn4` reproduces the published `delay` in
**55/55** cells of `live_main.csv` (max |diff| 0 ms), `Σc4/(Σn4+Σc4)` reproduces `cens4`
(max |diff| 0).

Caveat carried from the source: `perrun.csv`'s `n4/c4` count a pair as censored when *either*
discovery.csv or attrib's own hop chain is censored (`attrib.py:431-437`); the two disagree 0
times here (`live_main.csv` column `censor_disagree` is 0 in every cell), which is why the
json-only recipe matches exactly.

## 3. The cells

`evidence/11/cells.csv` (and `evidence/12/cells.csv`) hold **42 cells = 6 stages × 3 p (s5) +
6 stages × 4 p (s6)**, 20 runs each = 840 runs:

- s5, p 0 / 0.2 / 0.4 (18 cells): `s5b0p5`, `s5b1p359`, `s5b3p695`, `s5b10p043`, `s5short`,
  `s5wait`.
- s6, p 0.2 / 0.4 / 0.5 / 0.6 (24 cells): `s6b0p5`, `s6b1p359`, `s6b3p695`, `s6b10p043`,
  `s6short`, `s6wait`.
- Of these, 30 have a `B_const` (the 18 s5 cells + the 12 s6 cells at p 0.2/0.4); the 12 s6 cells
  at p 0.5/0.6 have none, since `REP_B` has no s4a value there.

Store inventory for all 14 stages (seed directories, DONE marker, `meta.json.validity.valid`),
all contiguous from seed 1, all DONE, all valid:

| stage | p values | seeds per cell | cells |
|---|---|---|---|
| `s5a` | 0, 0.1, 0.2, 0.3, 0.4 | 100 | 5 |
| `s5b0p5`, `s5b1p359`, `s5b3p695`, `s5b10p043` | 0, 0.1, 0.2, 0.3, 0.4 | 20 | 5 each (20) |
| `s5c` | 0, 0.1, 0.2, 0.3, 0.4 | 20 | 5 |
| `s5short`, `s5wait` | 0, 0.2, 0.4 | 20 | 3 each (6) |
| `s6b0p5`, `s6b1p359`, `s6b3p695`, `s6b10p043` | 0.2, 0.4, 0.5, 0.6 | 20 | 4 each (16) |
| `s6short`, `s6wait` | 0.2, 0.4, 0.5, 0.6 | 20 | 4 each (8) |

Total 60 cells / 1600 runs in the store; the 42 metric cells above are the subset the s5/s6
evidence scores. Cells in the store but outside the 42: all of `s5a` (5 cells, 500 runs) and
`s5c` (5 cells, 100 runs), and the s5 arms' p 0.1 / 0.3 (8 cells, 160 runs).
`evidence/09|10/perrun.csv` cover 55 cells (the 42 plus `s5c` and the s5 arms' p 0.1/0.3);
`s5a` has no perrun row.

### Cross-check of the two READMEs the task named

- `evidence/10/README.md:52` states that for the s6 stages the delay, censoring, byte and waste
  definitions are **the same as `evidence/09/`** (mean over uncensored 4-hop pairs, seed
  bootstrap SE, bytes per record-hop with names rebased to 14.92 chars), and `:75` that [4] is
  the same hull difference. Nothing there contradicts §1–2. It adds one more per-seed file worth
  knowing: `replay_calib_perseed.csv` (`:191`), which holds replay, not live, per-seed delays.
  `:96-97` is the explicit warning that censoring biases both the delay *and* the bytes of the
  10.043 s arm at p 0.5/0.6 (it loses 5.7 % / 8.3 % of its record-hops from the denominator).
- `evidence/12/README.md:9,15,17` defines `k` = how many empty replies from the same neighbor
  came immediately before, and `seen_new` = whether any non-empty reply from that neighbor ever
  arrived; both are traced from the start of the file, with the same window `[t0+10 s, t0+130 s]`
  for counting. Its byte column is not a new price: `:117` states it is
  `hits / empty replies × evidence/11/cells.csv's empty_B_per_rh_no_new ÷ B_const`, i.e. the same
  `INT_B + REP_B[p]` unit. `:118,122` repeat that these shares are **not** saved bytes.

## 4. UNCONFIRMED

1. **Reply payload length.** Never logged; `state_rx` carries `hi` only, and `extra` is invisible
   there (`evidence/11/README.md:147`). The per-reply byte figures (127–140 B) come from
   `attrib.reply_recon`'s reconstruction, which itself over-includes records stored between serve
   and the requester's processing (`attrib.py:214-236`). I did not re-derive it.
2. **`REP_B` for p 0.5 / 0.6.** No value exists; any replay scored in bytes at those p must either
   measure a new constant or report bytes only in poll counts.
3. **Why per-reply bytes differ by arm** (138.4 B at 0.5 s vs 127 B at 10.043 s) is marked an
   estimate in `evidence/09/README.md:126` and I did not check it.
4. **Which record-fetch traffic makes the 282–289 kB counter residual.** I showed it is nearly
   constant across runs with very different poll counts, consistent with 240 records × 4 hops,
   but I did not itemize record Interests vs record Data from the logs.
5. **`nu_h` (record-hops)** was taken from `perrun.csv`; I did not recompute attrib's uncensored
   record-hop denominator from events.jsonl (it needs the per-hop chain, not just the 4-hop pair).
6. **KM medians** in evalagg's `summary.csv`/`survival.csv` were read in the source only
   (`evalagg.py:41-49,566-612`); I did not verify a stored KM value from the logs.
7. **Verification breadth.** Per-run checks used 6 runs of 840; cell-level checks used all 55
   perrun cells. Stages `s5a` and `s5c` were not checked per run.
