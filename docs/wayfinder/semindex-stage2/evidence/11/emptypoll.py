"""Count answered state polls that brought nothing new, three ways, in the live s5*/s6* semindex runs.

Read-only analysis of existing runs. Writes perpair.csv, cells.csv, tables.md and mismatch_examples.txt
next to this script; reads only the run store, evalagg's polls.csv for s6, evidence/09 and evidence/10
perrun.csv (bytes) and live_main.csv (the B_const cross-check), and the semindex source (for evalagg's
t0/window derivation and file reader).

Definitions (as fixed before the counting, except the two sentences marked "2026-11-12 correction"):

Unit: one directed pair (node, neighbor) per run; aggregate to cells afterwards. Window: events with
ts_ms in [t0+10000, t0+130000] inclusive (as polls.csv).
- polls_sent: `decision` lines with kind `state_poll` for that neighbor in the window.
- state_rx: `state_rx` lines from that neighbor in the window. Late replies (`state_poll_late_reply`)
  also produce a state_rx; keep them in, and report late_replies as its own column.
- For each state_rx in the window:
  - rx_no_fetch: the next line after this state_rx line in the same node's events.jsonl is NOT a
    `decision` of kind `fetch_plan` with action.neighbor equal to this state_rx's neighbor and the same
    ts_ms. Pair by line order (on_tick also emits fetch_plan decisions when a backoff expires or a slot
    frees, so a same-ts search would mis-pair). This is what the engine actually did.
  - rx_no_new: for every origin o in the reply's `hi` (an empty hi counts as no_new), every seq
    1..hi[o] is already held by the node at that moment. Holdings = `record_created` (the origin's own
    records) plus `record_stored` lines that appear BEFORE this state_rx line in the file. It ignores
    the neighbor's sparse `extra` holdings because state_rx logs only hi.
    2026-11-12 correction: the task called this "ground truth that the policy cannot see". Three
    things have to be kept apart. (1) At the moment the reply arrives the node can compute it: it
    knows what it holds, and the engine plans its fetches from exactly this comparison (rx_no_fetch
    below). (2) One `state_poll` decision state does NOT keep that result: the fields that bear on
    emptiness are the node's own hi and missing per origin and each neighbor's advertised hi and
    notified_max as they are at decision time (tracker.decision_state; the logged state also has
    in_flight, last_timeout_s, truncated, last_poll_status and consecutive_poll_timeouts, none of
    which records whether the reply was empty), so a reply that brought 36 new seqs which were all
    fetched before the next decision looks the same there as a reply that brought nothing.
    (3) Two consecutive decision states for the same neighbor do show it, nearly always: on the
    line only the neighbor's side origin can be new, so the node's own hi for that origin
    advancing between the two states tracks whether the reply in between was non-empty
    (recheck.py's seq_* counts: never after an empty reply, 59032 of 59109 after a non-empty one,
    all cells). That relies on the side-origin rule and on the fetches landing before the next
    decision. A policy that reads one state only needs the result (or a streak count) added to the
    state. (4) No policy can see whether the NEXT reply will be empty.
  - rx_hi_unchanged: the reply's hi equals, origin by origin (a missing origin counts as 0), the hi of
    the previous state_rx from the same neighbor in the same run (the previous one may be before the
    window; the first reply of a run is compared with all zeros).
    The task called this "the signal a policy could observe". Consecutive decision states do show
    whether the neighbor's advertised hi changed, but a state holds the latest advertisement only, so
    with two replies between two decisions the per-reply value counted here cannot be recovered. It
    looks only at the neighbor's advertisement and not at what the node held when the reply arrived,
    so it is narrower than rx_no_new.
  - rx_after_complete: the node already holds every record of both origins (check from record_created
    how many each origin made; expected 120 = 30 generations x k 4, generated from about t0+13 s to
    t0+71 s) at that moment. It is a subset of rx_no_new; it separates "the burst of record generation
    is over and this node has everything" from emptiness during propagation.
  - Mismatch: count of state_rx where rx_no_fetch != rx_no_new, split into the two directions.
- Added beyond the task's list: rx_hi_unchanged_but_new = rx_hi_unchanged and not rx_no_new (the
  advertisement did not change while the node still lacks a seq <= hi).

Why empty (the four why_* columns; every rx_no_new falls in exactly one, first match wins): the topology is
the line h0-h1-h2-h3-h4 with the two origins h0 and h4, so from a given neighbour only the origin on that
neighbour's side can be new. For node hn receiving a reply from hm the side origin is h0 if m < n else h4.
  a. why_all_complete: the node holds every record of both origins. Identical to rx_after_complete, and the
     per-row equality is checked, so this column is the subset proof of "the generation burst is over".
  b. why_side_complete: the node holds all records of the side origin (count from record_created, as above),
     but not everything.
  c. why_side_none: the reply's hi for the side origin is 0 or missing, i.e. nothing of that origin has
     reached the neighbour yet, so the neighbour has nothing to give on the only axis that could be new.
  d. why_during: the rest. The neighbour has some of the side origin, the node already holds all of 1..hi,
     and more is still to come. It includes replies after generation has ended while records are still on
     their way, so it is not a time slice of the generation burst.
The run log and tables.md print the per-row checks a + b + c + d == rx_no_new and a == rx_after_complete.
- Shares: rx_* / state_rx, and state_rx / polls_sent. Seed-level means with a bootstrap CI over seeds
  (seed as the unit, 2000 resamples, fixed RNG seed); also pooled sums.

Mismatch causes (diagnostic only, columns mm_*): the engine's fetch bookkeeping is replayed from the
same file -- a `fetch_plan` marks each fetched (origin, seq) in flight on its neighbor, a
`record_stored` releases it, a `fetch_backoff` decision releases it and blocks retries until
ts_ms + wait_s*1000 (tracker.plan_fetches / _release / _fail). For "new but no fetch" the first
matching reason wins: cap (in-flight count to this neighbor >= 8, the default
max_in_flight_per_neighbor), in_flight (every missing seq <= hi is in flight already), backoff (every
missing seq is in flight or still backing off, at least one backing off), other. For "no new but
fetched" the reason is extra when every fetched seq is above the reply's hi for its origin (the
neighbor's sparse extra, which state_rx does not log), else other.

Environment: AGG (required, evalagg output with <stage>/polls.csv for the s6 stages), STORE (default
/home/lab_shared/semindex-eval). Needs PYTHONPATH=<repo>/src; set PYTHONDONTWRITEBYTECODE=1 so that
importing semindex leaves nothing behind in the checkout.

Bytes (p 0, 0.2, 0.4 only; REP_B has no s4a value for p 0.5/0.6): evidence/09 ana.py's constant-bytes
model, B_const = (polls * INT_B + answered polls * REP_B[p]) / record-hops, where, as ana.per_seed_cols
does, polls = sum over hops of perrun.csv ngap, answered polls = sum of (ngap - nnorep), record-hops =
sum of nu1..nu4 (evidence/05 attrib's uncensored record-hops), summed over the cell's runs before
dividing. empty_B_per_rh = sum over runs of rx_no_new * (INT_B + REP_B[p]) / record-hops: the bytes, in
this constant model, of the answered polls whose reply held nothing new up to hi. It is a reference value
for a policy with perfect foresight, not what any real rule can reach, and not exactly "what could be
skipped without changing delay": rx_no_new ignores the neighbor's sparse extra, and 9 such replies (all
cells together, mm_fbnn_extra) did start a fetch above hi. Skipping those would move that fetch. The same
unit price splits empty_B over the four why_* reasons.
"""

import csv
import json
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np
from semindex import evalagg, evalgen

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(HERE, os.pardir)
STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")
AGG6 = os.environ["AGG"]
PERRUN = {"s5": f"{EV}/09/perrun.csv", "s6": f"{EV}/10/perrun.csv"}
OUT = HERE

INT_B = 50.9
REP_B = {0.0: 132.5, 0.2: 130.4, 0.4: 125.5}
MAX_IN_FLIGHT = 8  # tracker.DEFAULT_MAX_IN_FLIGHT_PER_NEIGHBOR; every s5/s6 meta.json leaves it null
KINDS = ("b0p5", "b1p359", "b3p695", "b10p043", "short", "wait")
CELLS = {"s6": (0.2, 0.4, 0.5, 0.6), "s5": (0.0, 0.2, 0.4)}
SEEDS = range(1, 21)
BOOT = 2000
RNG_SEED = 20261112

WHY = ("why_all_complete", "why_side_complete", "why_side_none", "why_during")
COUNTS = ("polls_sent", "state_rx", "late_replies", "rx_no_fetch", "rx_no_new", "rx_hi_unchanged",
          "rx_after_complete", "rx_hi_unchanged_but_new", *WHY, "mm_nofetch_but_new", "mm_fetch_but_nonew",
          "mm_nfbn_cap", "mm_nfbn_in_flight", "mm_nfbn_backoff", "mm_nfbn_other",
          "mm_fbnn_extra", "mm_fbnn_other")
RX = ("rx_no_fetch", "rx_no_new", "rx_hi_unchanged", "rx_after_complete")
# The 8 directed pairs of the line h0-h1-h2-h3-h4, written node<-neighbour, for the per-direction table.
DIRS = [("h0", "h1"), ("h1", "h0"), ("h1", "h2"), ("h2", "h1"),
        ("h2", "h3"), ("h3", "h2"), ("h3", "h4"), ("h4", "h3")]
DIR_STAGES = ("s6b10p043", "s6short", "s6wait")
DIR_PS = (0.2, 0.4)


def side_origin(node, neighbor):
    """The only origin whose records can be new when `node` hears from `neighbor` on the line topology.

    Records flow along h0-h1-h2-h3-h4 from the two ends, so a neighbour closer to h0 can only ever hold
    h0's records that this node lacks, and a neighbour closer to h4 only h4's. Breaks if the topology is
    not that line.
    """
    return "h0" if int(neighbor[1:]) < int(node[1:]) else "h4"


def _node_events_with_lines(run_dir):
    """evalagg's own reader (file order, blank lines skipped), so line order is what evalagg sees."""
    return evalagg.discover_node_events(run_dir)


def _window(run_dir, nodes):
    """evalagg._analyze_run_metrics's window: derive_t0_ms over the run's events (+ results.json)."""
    results = evalagg.read_results(run_dir) if os.path.isfile(os.path.join(run_dir, "results.json")) else None
    t0 = evalagg.derive_t0_ms(nodes, results).t0_ms
    return t0 + evalgen.WINDOW_START_S * 1000.0, t0 + evalgen.WINDOW_END_S * 1000.0


def analyze_node(node, events, lo, hi_edge, created_count, examples):
    """Per-neighbor counts for one node's file; `examples` collects the first quoted mismatch per reason."""
    rows = defaultdict(lambda: dict.fromkeys(COUNTS, 0))
    held = defaultdict(set)  # origin -> held seqs, as of the current line
    prev_hi = {}  # neighbor -> previous state_rx's hi (window-independent)
    fl = {}  # (origin, seq) -> in-flight neighbor
    flc = defaultdict(int)  # neighbor -> in-flight count
    nxt = {}  # (origin, seq) -> backoff end ts_ms
    for i, ev in enumerate(events):
        name = ev.get("event")
        ts = ev.get("ts_ms")
        if name == "record_created" and ev["origin"] == node:
            held[ev["origin"]].add(ev["seq"])
        elif name == "record_stored":
            key = (ev["origin"], ev["seq"])
            held[key[0]].add(key[1])
            if key in fl:
                flc[fl.pop(key)] -= 1
            nxt.pop(key, None)
        elif name == "decision" and ev["kind"] == "fetch_plan":
            nb = ev["action"]["neighbor"]
            for o, s in ev["action"]["fetched"]:
                fl[(o, s)] = nb
                flc[nb] += 1
                nxt.pop((o, s), None)
        elif name == "decision" and ev["kind"] == "fetch_backoff":
            a = ev["action"]
            key = (a["origin"], a["seq"])
            if fl.get(key) == a["neighbor"]:
                flc[fl.pop(key)] -= 1
            nxt[key] = ts + a["wait_s"] * 1000.0
        in_win = ts is not None and lo <= ts <= hi_edge
        if name == "decision" and ev["kind"] == "state_poll" and in_win:
            rows[ev["action"]["neighbor"]]["polls_sent"] += 1
        elif name == "state_poll_late_reply" and in_win:
            rows[ev["neighbor"]]["late_replies"] += 1
        elif name == "state_rx":
            nb, hi = ev["neighbor"], ev["hi"]
            before = prev_hi.get(nb, {})
            prev_hi[nb] = hi
            if not in_win:
                continue
            r = rows[nb]
            r["state_rx"] += 1
            nx = events[i + 1] if i + 1 < len(events) else {}
            fetched = (nx.get("event") == "decision" and nx.get("kind") == "fetch_plan"
                       and nx["action"]["neighbor"] == nb and nx["ts_ms"] == ts)
            missing = [(o, s) for o, h in hi.items() for s in range(1, h + 1) if s not in held[o]]
            no_new = not missing
            unchanged = all(hi.get(o, 0) == before.get(o, 0) for o in set(hi) | set(before))
            complete = all(len(held[o]) >= n and all(s in held[o] for s in range(1, n + 1))
                           for o, n in created_count.items())
            r["rx_no_fetch"] += not fetched
            r["rx_no_new"] += no_new
            r["rx_hi_unchanged"] += unchanged
            r["rx_after_complete"] += complete
            r["rx_hi_unchanged_but_new"] += unchanged and not no_new
            if no_new:
                # Why was it empty? First match wins, in this order, so that the four columns partition
                # rx_no_new: "everything is in" before "this side is in" before "this side has not started".
                side = side_origin(node, nb)
                n_side = created_count[side]
                if complete:
                    why = "all_complete"
                elif len(held[side]) >= n_side and all(s in held[side] for s in range(1, n_side + 1)):
                    why = "side_complete"
                elif not hi.get(side, 0):
                    why = "side_none"
                else:
                    why = "during"
                r[f"why_{why}"] += 1
            if not fetched and not no_new:
                r["mm_nofetch_but_new"] += 1
                if flc[nb] >= MAX_IN_FLIGHT:
                    why = "cap"
                elif all(k in fl for k in missing):
                    why = "in_flight"
                elif all(k in fl or nxt.get(k, -1) > ts for k in missing):
                    why = "backoff"
                else:
                    why = "other"
                r[f"mm_nfbn_{why}"] += 1
                examples.setdefault(f"nofetch_but_new/{why}",
                                    (node, i, ev, nx, sorted(missing)[:10], len(missing), flc[nb]))
            elif fetched and no_new:
                r["mm_fetch_but_nonew"] += 1
                above = all(s > hi.get(o, 0) for o, s in nx["action"]["fetched"])
                why = "extra" if above else "other"
                r[f"mm_fbnn_{why}"] += 1
                examples.setdefault(f"fetch_but_nonew/{why}", (node, i, ev, nx, [], 0, flc[nb]))
    return rows


def analyze_run(job):
    stage, p, seed, run_dir = job
    nodes = _node_events_with_lines(run_dir)
    lo, hi_edge = _window(run_dir, nodes)
    created_count = defaultdict(int)
    for events in nodes.values():
        for ev in events:
            if ev.get("event") == "record_created":
                created_count[ev["origin"]] += 1
    examples = {}
    out = []
    for node, events in nodes.items():
        for nb, r in analyze_node(node, events, lo, hi_edge, dict(created_count), examples).items():
            out.append(dict(stage=stage, p=p, seed=seed, node=node, neighbor=nb, **r))
    out.sort(key=lambda r: (r["node"], r["neighbor"]))
    return out, dict(created_count), {k: (stage, p, seed, *v) for k, v in examples.items()}


def jobs():
    js = []
    for fam, ps in CELLS.items():
        for kind in KINDS:
            st = f"{fam}{kind}"
            for ref in evalagg.discover_runs(STORE, st):
                if ref.p in ps and ref.seed in SEEDS:
                    js.append((st, ref.p, ref.seed, ref.run_dir))
    return js


def check_refs(rows):
    """polls_sent/state_rx/late_replies against polls.csv (s6, per pair) and evidence/09 perrun.csv (s5, per run)."""
    msgs = []
    mine = {(r["stage"], r["p"], r["seed"], r["node"], r["neighbor"]): r for r in rows}
    n6 = bad6 = 0
    for kind in KINDS:
        st = f"s6{kind}"
        seen = set()
        with open(f"{AGG6}/{st}/polls.csv") as f:
            for q in csv.DictReader(f):
                key = (st, float(q["p"]), int(q["seed"]), q["node"], q["neighbor"])
                seen.add(key)
                r = mine.get(key)
                n6 += 1
                got = (r["polls_sent"], r["state_rx"], r["late_replies"]) if r else None
                want = (int(q["polls_sent"]), int(q["state_rx"]), int(q["late_replies"]))
                if got != want:
                    bad6 += 1
                    msgs.append(f"s6 mismatch {key}: mine {got} ref {want}")
        extra = [k for k in mine if k[0] == st and k not in seen
                 and any(mine[k][c] for c in ("polls_sent", "state_rx", "late_replies"))]
        for k in extra:
            bad6 += 1
            msgs.append(f"s6 pair missing from polls.csv: {k}")
    per_run = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        k = (r["stage"], r["p"], r["seed"])
        per_run[k][0] += r["polls_sent"]
        per_run[k][1] += r["state_rx"]
        per_run[k][2] += r["late_replies"]
    n5 = bad5 = 0
    with open(PERRUN["s5"]) as f:
        for q in csv.DictReader(f):
            if float(q["p"]) not in CELLS["s5"] or not q["stage"].startswith("s5") or q["stage"] == "s5c":
                continue
            k = (q["stage"], float(q["p"]), int(q["seed"]))
            n5 += 1
            want = [int(q["polls_sent"]), int(q["state_rx"]), int(q["late"])]
            if per_run.get(k) != want:
                bad5 += 1
                msgs.append(f"s5 mismatch {k}: mine {per_run.get(k)} ref {want}")
    head = (f"reference check: s6 {n6} polls.csv pairs, {bad6} mismatched; "
            f"s5 {n5} perrun.csv runs, {bad5} mismatched (polls_sent, state_rx, late_replies)")
    return [head, *msgs]


def check_bconst(cell_rows):
    """B_const recomputed here vs the value evidence/09 and evidence/10 ana.py already published."""
    want = {}
    for fam, d in (("s5", "09"), ("s6", "10")):
        with open(f"{EV}/{d}/live_main.csv") as fh:
            for q in csv.DictReader(fh):
                want[(q["stage"], float(q["p"]))] = float(q["B_const"])
    diffs = [(r["stage"], r["p"], abs(r["B_const"] - want[(r["stage"], r["p"])]))
             for r in cell_rows if "B_const" in r and (r["stage"], r["p"]) in want]
    worst = max((d for _, _, d in diffs), default=float("nan"))
    missing = [(r["stage"], r["p"]) for r in cell_rows
               if "B_const" in r and (r["stage"], r["p"]) not in want]
    return (f"B_const check: {len(diffs)} of the {sum('B_const' in r for r in cell_rows)} bytes cells are in "
            f"evidence/09 and evidence/10 live_main.csv, max abs diff {worst:.3g}"
            + (f"; not published there: {missing}" if missing else ""))


def boot_ci(per_seed, rng):
    """Seed-level mean of a per-seed share and its 2.5/97.5 % bootstrap interval over seeds."""
    x = np.array([v for v in per_seed if not np.isnan(v)])
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_perrun():
    pr = {}
    for path in PERRUN.values():
        with open(path) as f:
            for q in csv.DictReader(f):
                h = (1, 2, 3, 4)
                ngap = sum(float(q[f"ngap{i}"]) for i in h)
                pr[(q["stage"], float(q["p"]), int(q["seed"]))] = {
                    "rh": sum(float(q[f"nu{i}"]) for i in h), "polls": ngap,
                    "ans": ngap - sum(float(q[f"nnorep{i}"]) for i in h), "polls_sent": int(q["polls_sent"])}
    return pr


def cells(rows, pr):
    rng = np.random.default_rng(RNG_SEED)
    # The why_* shares draw from their own generator: sharing `rng` would shift every later cell's CI of
    # the pre-existing columns, which must stay comparable with the earlier runs of this script.
    rng_why = np.random.default_rng(RNG_SEED + 1)
    run = defaultdict(lambda: dict.fromkeys(COUNTS, 0))
    for r in rows:
        acc = run[(r["stage"], r["p"], r["seed"])]
        for c in COUNTS:
            acc[c] += r[c]
    out = []
    for fam, ps in CELLS.items():
        for p in ps:
            arm = f"{fam}b10p043"
            for kind in KINDS:
                st = f"{fam}{kind}"
                seeds = sorted(s for (a, b, s) in run if a == st and b == p)
                tot = {c: sum(run[(st, p, s)][c] for s in seeds) for c in COUNTS}
                row = dict(stage=st, p=p, n_runs=len(seeds), **tot)
                for c in RX:
                    row[f"{c}_share_pooled"] = tot[c] / tot["state_rx"]
                    m, a, b = boot_ci([run[(st, p, s)][c] / run[(st, p, s)]["state_rx"] for s in seeds], rng)
                    row[f"{c}_share_seedmean"], row[f"{c}_share_ci_lo"], row[f"{c}_share_ci_hi"] = m, a, b
                row["rx_per_poll_pooled"] = tot["state_rx"] / tot["polls_sent"]
                m, a, b = boot_ci([run[(st, p, s)]["state_rx"] / run[(st, p, s)]["polls_sent"] for s in seeds], rng)
                row["rx_per_poll_seedmean"], row["rx_per_poll_ci_lo"], row["rx_per_poll_ci_hi"] = m, a, b
                for c in WHY:
                    row[f"{c}_share_pooled"] = tot[c] / tot["state_rx"]
                    m, a, b = boot_ci([run[(st, p, s)][c] / run[(st, p, s)]["state_rx"] for s in seeds],
                                      rng_why)
                    row[f"{c}_share_seedmean"], row[f"{c}_share_ci_lo"], row[f"{c}_share_ci_hi"] = m, a, b
                row["why_sum_minus_no_new"] = sum(tot[c] for c in WHY) - tot["rx_no_new"]
                row["why_all_complete_minus_after_complete"] = tot["why_all_complete"] - tot["rx_after_complete"]
                if p in REP_B:
                    def bconst(stage, p=p, seeds=seeds):
                        q = [pr[(stage, p, s)] for s in seeds]
                        return (sum(x["polls"] for x in q) * INT_B + sum(x["ans"] for x in q) * REP_B[p]) / sum(
                            x["rh"] for x in q)
                    rh = sum(pr[(st, p, s)]["rh"] for s in seeds)
                    unit = INT_B + REP_B[p]
                    row["record_hops"] = rh
                    row["perrun_ngap_minus_polls_sent"] = sum(pr[(st, p, s)]["polls"] - pr[(st, p, s)]["polls_sent"]
                                                              for s in seeds)
                    # B_const's "answered polls" (ngap - nnorep, from the 4-hop discovery rows) and state_rx
                    # (polls.csv's reply lines) are two different counts of the same thing; how far apart they
                    # are bounds the error of pricing an empty reply at REP_B.
                    row["perrun_answered_minus_state_rx"] = sum(pr[(st, p, s)]["ans"] for s in seeds) - tot["state_rx"]
                    row["B_const"] = bconst(st)
                    row["B_const_arm10p043"] = bconst(arm)
                    row["empty_B_per_rh_no_new"] = tot["rx_no_new"] * unit / rh
                    row["empty_B_per_rh_hi_unchanged"] = tot["rx_hi_unchanged"] * unit / rh
                    row["empty_B_per_rh_after_complete"] = tot["rx_after_complete"] * unit / rh
                    row["empty_B_per_rh_no_new_rest"] = (tot["rx_no_new"] - tot["rx_after_complete"]) * unit / rh
                    for c in WHY:
                        row[f"empty_B_per_rh_{c}"] = tot[c] * unit / rh
                    # What share of the cell's whole state-poll byte bill the hard-to-predict part is.
                    row["during_share_of_B_const"] = row["empty_B_per_rh_why_during"] / row["B_const"]
                    if kind in ("short", "wait"):
                        ex = row["B_const"] - row["B_const_arm10p043"]
                        row["excess_B_per_rh"] = ex
                        row["ratio_no_new_over_excess"] = row["empty_B_per_rh_no_new"] / ex
                        row["ratio_hi_unchanged_over_excess"] = row["empty_B_per_rh_hi_unchanged"] / ex
                out.append(row)
    return out


def f(x, nd=3):
    return "-" if x is None or np.isnan(x) else f"{x:.{nd}f}"


LEGEND = """Legend (definitions from emptypoll.py's docstring; every count is in the window [t0+10000, t0+130000] ms
inclusive, t0 and window from evalagg.derive_t0_ms / evalgen.WINDOW_START_S, WINDOW_END_S, as polls.csv):

- polls_sent: `decision` lines with kind `state_poll` for that neighbor in the window.
- state_rx: `state_rx` lines from that neighbor in the window (late replies included; late = `state_poll_late_reply` lines).
- rx_no_fetch: the line right after the state_rx is NOT a `fetch_plan` decision for that neighbor at the same ts_ms (what the engine did).
- rx_no_new: every seq 1..hi[o] of every origin o in the reply's hi was already held (record_created + earlier record_stored lines); empty hi counts. The node can compute this when the reply arrives; one state_poll decision state does not keep it, two consecutive ones nearly always show it; no policy knows it before sending (see the docstring's 2026-11-12 correction). Ignores the neighbor's sparse extra.
- rx_hi_unchanged: the reply's hi equals the previous state_rx's hi from the same neighbor in the same run, origin by origin (missing = 0; first reply vs all zeros). Uses the neighbor's advertisement only, not what the node held; consecutive decision states show whether the advertisement changed, which is not always this per-reply value.
- rx_after_complete: the node already held every record of both origins (counts from record_created). Subset of rx_no_new.
- why_*: of the rx_no_new replies, why nothing was new; first match wins, so the four partition rx_no_new. Side origin = the
  origin on the neighbour's side of the line h0-h1-h2-h3-h4 (h0 if the neighbour's index is lower, else h4); only it can be new.
  all_complete = the node holds every record of both origins (equal to rx_after_complete, checked per row);
  side_complete = it holds all of the side origin but not everything; side_none = the reply's hi for the side origin is 0 or
  missing (that origin has not reached the neighbour yet); during = the rest (the neighbour has some of the side origin, the
  node holds all of 1..hi, more is still to come; it includes replies after generation has ended while records are still on their way).
- hi_unch_but_new (added, not in the task's list): rx_hi_unchanged and not rx_no_new.
- mm nofetch_but_new: rx_no_fetch and not rx_no_new (reasons cap / in_flight / backoff / other, replayed bookkeeping, see docstring); mm fetch_but_nonew: fetched and rx_no_new (reason extra = every fetched seq above hi).
- share = rx_* / state_rx (why_* too); rx/poll = state_rx / polls_sent. "seed mean [CI]" = mean over the 20 seeds of each
  seed's own ratio, 2.5-97.5 % percentile bootstrap over seeds (2000 resamples, numpy default_rng(20261112), and
  default_rng(20261113) for the why_* shares, which draw from their own stream so that the other columns' CIs do not move);
  "pooled" = sum / sum over the cell.
- B_const = (sum ngap * 50.9 + sum (ngap - nnorep) * REP_B[p]) / sum nu (evidence/09 ana.py's model; ngap, nnorep, nu from
  evidence/09 or evidence/10 perrun.csv, summed over the four hops and the cell's runs). B_arm = the same for the p's fixed 10.043 s arm.
- empty_B (no_new) = sum rx_no_new * (50.9 + REP_B[p]) / sum nu; empty_B (hi_unch.) uses rx_hi_unchanged instead;
  after_complete / rest split empty_B (no_new) into rx_after_complete and rx_no_new - rx_after_complete; the why_* columns
  split the same empty_B four ways at the same unit price, and during/B_const is that part of the cell's whole poll bill.
  **empty_B is a reference value for perfect foresight (skipping exactly the answered polls that brought nothing up to hi,
  saving their Interest and reply), not reachable by any real rule; 9 of those replies (all cells) started a fetch from the
  neighbor's sparse extra, so it is not exactly "skippable without changing delay".**
- excess_B = B_const(policy) - B_const(10.043 arm, same p, same seeds); ratio = empty_B / excess_B.
  At p 0 no poll fails, so short/wait send the same polls as the arm and excess_B is ~0 (within noise, sign
  arbitrary): the p 0 ratios are not meaningful.
- B_const here is checked against evidence/09 and evidence/10 live_main.csv's B_const (the line under the bytes table).
- REP_B (s4a reply bytes per answered poll) = {0: 132.5, 0.2: 130.4, 0.4: 125.5}; it has no s4a value for p 0.5 / 0.6,
  so those cells get counts and shares only (evidence/10 measured its own REP_B there; not used here, by instruction)."""


def dir_rows(rows):
    """Pooled why_* counts per directed pair (node<-neighbour) for the three stages of the per-direction table."""
    acc = defaultdict(lambda: dict.fromkeys(("state_rx", "rx_no_new", *WHY), 0))
    for r in rows:
        if r["stage"] in DIR_STAGES and r["p"] in DIR_PS:
            a = acc[(r["stage"], r["p"], r["node"], r["neighbor"])]
            for c in a:
                a[c] += r[c]
    out = []
    for st in DIR_STAGES:
        for p in DIR_PS:
            for node, nb in DIRS:
                a = acc[(st, p, node, nb)]
                out.append((st, p, node, nb, side_origin(node, nb), a))
    return out


def tables(cell_rows, refmsg, mm_counts, rows):
    L = ["# Empty answered state polls (s5*, s6*)", "", refmsg[0], "", LEGEND, ""]
    L += ["## Counts and shares (per cell, 20 seeds)", "",
          ("| stage | p | polls_sent | state_rx | late | rx/poll seed mean [CI] | no_fetch share seed mean [CI] | "
          "no_new share seed mean [CI] | hi_unch. share seed mean [CI] | after_complete share seed mean [CI] | "
          "pooled no_fetch / no_new / hi_unch. / after_compl. |"),
          "|---|---|---|---|---|---|---|---|---|---|---|"]

    def ci(r, c):
        return f"{f(r[c + '_seedmean'])} [{f(r[c + '_ci_lo'])}, {f(r[c + '_ci_hi'])}]"

    for r in cell_rows:
        pooled = " / ".join(f(r[f"{c}_share_pooled"]) for c in RX)
        L.append(f"| {r['stage']} | {r['p']} | {r['polls_sent']} | {r['state_rx']} | {r['late_replies']} | "
                 f"{ci(r, 'rx_per_poll')} | {ci(r, 'rx_no_fetch_share')} | {ci(r, 'rx_no_new_share')} | "
                 f"{ci(r, 'rx_hi_unchanged_share')} | {ci(r, 'rx_after_complete_share')} | {pooled} |")
    L += ["", "## Why the answered poll brought nothing new (per cell, shares of state_rx)", "",
          ("| stage | p | state_rx | no_new | all_complete seed mean [CI] | side_complete seed mean [CI] | "
           "side_none seed mean [CI] | during seed mean [CI] | pooled all_c. / side_c. / side_none / during |"),
          "|---|---|---|---|---|---|---|---|---|"]
    for r in cell_rows:
        pooled = " / ".join(f(r[f"{c}_share_pooled"]) for c in WHY)
        L.append(f"| {r['stage']} | {r['p']} | {r['state_rx']} | {r['rx_no_new']} | "
                 + " | ".join(ci(r, f"{c}_share") for c in WHY) + f" | {pooled} |")
    bad_sum = [f"{r['stage']} p{r['p']}: {r['why_sum_minus_no_new']:+d}" for r in cell_rows
               if r["why_sum_minus_no_new"]]
    bad_ac = [f"{r['stage']} p{r['p']}: {r['why_all_complete_minus_after_complete']:+d}" for r in cell_rows
              if r["why_all_complete_minus_after_complete"]]
    L += ["", "Checks over the 42 cells: sum(why_*) - rx_no_new = "
          + (", ".join(bad_sum) if bad_sum else "0 everywhere")
          + "; why_all_complete - rx_after_complete = " + (", ".join(bad_ac) if bad_ac else "0 everywhere")
          + ". The same two checks hold per perpair.csv row (see the run log)."]
    L += ["", "## Why empty, per direction (pooled over the 20 seeds; node<-neighbour)", "",
          "| stage | p | node<-neighbour | side origin | state_rx | no_new | all_complete | side_complete | "
          "side_none | during |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for st, p, node, nb, side, a in dir_rows(rows):
        n = a["state_rx"]
        sh = " | ".join(f(a[c] / n) if n else "-" for c in ("rx_no_new", *WHY))
        L.append(f"| {st} | {p} | {node}<-{nb} | {side} | {n} | {sh} |")
    L += ["", "## Mismatch between rx_no_fetch and rx_no_new (pooled counts per cell)", "",
          ("| stage | p | state_rx | nofetch_but_new | cap | in_flight | backoff | other | fetch_but_nonew | extra | other "
          "| hi_unch_but_new |"),
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in cell_rows:
        L.append(f"| {r['stage']} | {r['p']} | {r['state_rx']} | {r['mm_nofetch_but_new']} | {r['mm_nfbn_cap']} | "
                 f"{r['mm_nfbn_in_flight']} | {r['mm_nfbn_backoff']} | {r['mm_nfbn_other']} | "
                 f"{r['mm_fetch_but_nonew']} | {r['mm_fbnn_extra']} | {r['mm_fbnn_other']} | {r['rx_hi_unchanged_but_new']} |")
    L.append("")
    L.append("All cells: " + ", ".join(f"{k} {v}" for k, v in mm_counts.items()))
    L += ["", "## Bytes per record-hop (p 0, 0.2, 0.4)", "",
          ("| stage | p | B_const | B_arm 10.043 | excess_B | empty_B no_new | of which after_complete | rest | "
          "empty_B hi_unch. | ratio no_new/excess | ratio hi_unch./excess | B all_c. | B side_c. | B side_none | "
          "B during | during/B_const |"),
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in cell_rows:
        if "B_const" not in r:
            continue
        L.append(f"| {r['stage']} | {r['p']} | {f(r['B_const'], 2)} | {f(r['B_const_arm10p043'], 2)} | "
                 f"{f(r.get('excess_B_per_rh'), 2)} | {f(r['empty_B_per_rh_no_new'], 2)} | "
                 f"{f(r['empty_B_per_rh_after_complete'], 2)} | {f(r['empty_B_per_rh_no_new_rest'], 2)} | "
                 f"{f(r['empty_B_per_rh_hi_unchanged'], 2)} | {f(r.get('ratio_no_new_over_excess'), 2)} | "
                 f"{f(r.get('ratio_hi_unchanged_over_excess'), 2)} | "
                 + " | ".join(f(r[f"empty_B_per_rh_{c}"], 2) for c in WHY)
                 + f" | {f(r['during_share_of_B_const'])} |")
    gap = [r for r in cell_rows if r.get("perrun_ngap_minus_polls_sent")]
    nb = len([r for r in cell_rows if "B_const" in r])
    L += ["", f"perrun.csv sum(ngap) - polls_sent (B_const's polls vs polls.csv's), over the {nb} cells with bytes: "
          + (", ".join(f"{r['stage']} p{r['p']}: {r['perrun_ngap_minus_polls_sent']:+.0f}" for r in gap) or "0 in every one")]
    L += ["", check_bconst(cell_rows)]
    ans = [r for r in cell_rows if r.get("perrun_answered_minus_state_rx")]
    L += ["", f"perrun.csv answered polls (ngap - nnorep) - state_rx, over the {nb} cells with bytes: "
          + (", ".join(f"{r['stage']} p{r['p']}: {r['perrun_answered_minus_state_rx']:+.0f}" for r in ans)
             + f"; 0 in the other {nb - len(ans)}" if ans else "0 in every cell")]
    return "\n".join(L) + "\n"


def main():
    js = jobs()
    with mp.get_context("fork").Pool(4) as pool:
        res = pool.map(analyze_run, js, chunksize=4)
    rows, examples, created = [], {}, defaultdict(int)
    for (out, cc, ex), job in zip(res, js):
        rows += out
        for k, v in ex.items():
            examples.setdefault(k, v)
        for o, n in cc.items():
            created[(job[0][:2], o, n)] += 1
    rows.sort(key=lambda r: (r["stage"], r["p"], r["seed"], r["node"], r["neighbor"]))
    bad_order = [r for r in rows if r["rx_after_complete"] > r["rx_no_new"]]
    # The why_* split must partition rx_no_new row by row, and its first branch must be rx_after_complete.
    bad_why = [r for r in rows if sum(r[c] for c in WHY) != r["rx_no_new"]]
    bad_why_ac = [r for r in rows if r["why_all_complete"] != r["rx_after_complete"]]
    with open(f"{OUT}/perpair.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, ["stage", "p", "seed", "node", "neighbor", *COUNTS])
        w.writeheader()
        w.writerows(rows)
    refmsg = check_refs(rows)
    cr = cells(rows, load_perrun())
    with open(f"{OUT}/cells.csv", "w", newline="") as fo:
        keys = []
        for r in cr:
            keys += [k for k in r if k not in keys]
        w = csv.DictWriter(fo, keys)
        w.writeheader()
        w.writerows(cr)
    mm = {c: sum(r[c] for r in rows) for c in COUNTS if c.startswith("mm_") or c == "rx_hi_unchanged_but_new"}
    with open(f"{OUT}/tables.md", "w") as fo:
        fo.write(tables(cr, refmsg, mm, rows))
    with open(f"{OUT}/mismatch_examples.txt", "w") as fo:
        for k in sorted(examples):
            st, p, seed, node, i, ev, nx, miss, nmiss, flc = examples[k]
            fo.write(f"## {k}: {st} p{p} seed{seed} {node} line index {i} (0-based, blank lines skipped); "
                     f"missing<=hi {nmiss} first {miss}; in-flight to neighbor {flc}\n"
                     f"{json.dumps(ev, sort_keys=True)}\n{json.dumps(nx, sort_keys=True)}\n\n")
    print("\n".join(refmsg[:20]))
    print("runs", len(js), "pairs", len(rows), "after_complete>no_new rows", len(bad_order))
    print("why split: rows with sum(why_*) != rx_no_new", len(bad_why),
          "rows with why_all_complete != rx_after_complete", len(bad_why_ac))
    print("record_created counts (family, origin, count): runs", dict(sorted(created.items())))


if __name__ == "__main__":
    main()
