"""How well does a streak of empty replies predict that the next reply is empty too?

Read-only re-reading of the 840 s5*/s6* runs that evidence/11 counted (the 42 (stage, p) cells of
../11/perpair.csv). No new runs. Writes perpair.csv, cells.csv and tables.md next to this script.

Definitions, fixed before counting:

Unit: one directed pair (node, neighbor) per run. Window: ts_ms in [t0+10000, t0+130000] inclusive,
t0 = the `t0_ms` of the node's own `agent_start` line.
- Holdings at a line: the (origin, seq) of every `record_created` and `record_stored` line above it in
  the same node's events.jsonl.
- A `state_rx` line from the neighbor is EMPTY -- evidence/11's rx_no_new -- when every seq 1..hi[o] of
  every origin o in its `hi` is already held. An empty `hi` counts as EMPTY.
- Reply sequence of a pair: ALL `state_rx` lines from that neighbor in file order, starting at the top of
  the file. Replies before the window are history: they are not counted, but they do feed k and seen_new,
  so a streak that started before t0+10 s is not restarted at the window edge.
- For each reply, read BEFORE the reply itself updates the state:
  - k = the number of consecutive EMPTY replies from that neighbor immediately before it (0 when the
    previous reply was non-empty, and 0 when there is no previous reply);
  - seen_new = 1 when some earlier reply from that neighbor was non-empty, else 0.
  k = 0 together with seen_new = 0 is therefore "no earlier reply at all", which is a different situation
  from "the last reply brought something", and the two are kept apart everywhere.
- Counted: replies inside the window only. k bins: 0, 1, 2, 3, 4, 5_9, 10_19, 20p.

What this is NOT. The numbers describe replies that actually arrived in the logged runs. A rule that
skips or delays polls changes which replies exist afterwards: after a skipped poll there is no reply, so
k stops growing, and a flagged non-empty reply is a record that is found later instead. recall and
hit_B_share are therefore NOT bytes saved, and miss_share is NOT a delay. They say how much of the
emptiness a streak count flags, and how often the flag is wrong, on the trajectories that the rules
actually in force produced.

Cross-check (printed, and fatal when it fails): per perpair.csv row, the 16 n_* columns must sum to the
`state_rx` and the 16 e_* columns to the `rx_no_new` of the same row of ../11/perpair.csv. Both are the
same quantity counted a second way, so any disagreement is a bug here, not a finding.

Environment: STORE (default /home/lab_shared/semindex-eval), read-only. Deliberately imports nothing from
semindex, so the checkout gets no bytecode and no PYTHONPATH is needed.
"""

import csv
import json
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV11 = os.path.join(HERE, os.pardir, "11")
STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")

WIN_LO_MS = 10000
WIN_HI_MS = 130000
NODES = ("h0", "h1", "h2", "h3", "h4")
BOOT = 2000
RNG_SEED = 20261116

# k bins, as label -> lowest k that falls in it. The order here is the column and table order; the
# lower edges double as the thresholds of the "k >= K" rules, which is why K in (1, 2, 3, 5) needs no
# finer counting than these bins.
BINS = (("0", 0), ("1", 1), ("2", 2), ("3", 3), ("4", 4), ("5_9", 5), ("10_19", 10), ("20p", 20))
BIN_LABELS = tuple(b for b, _ in BINS)
KS = (1, 2, 3, 5)
PAIR_COLS = tuple(f"{x}_s{s}_k{b}" for s in (0, 1) for b in BIN_LABELS for x in ("n", "e"))

# The six cells the task wants split by seen_new: the three policies whose polls are rare enough that a
# streak is worth anything, at the two loss rates that have byte references in ../11/cells.csv.
SPLIT_STAGES = ("s6b10p043", "s6short", "s6wait")
SPLIT_PS = (0.2, 0.4)


def bin_of(k):
    """The k bin label for a streak length; bins are closed on the left, so the last match wins."""
    label = BIN_LABELS[0]
    for b, lo in BINS:
        if k >= lo:
            label = b
    return label


def read_events(path):
    """The node's events.jsonl in file order, blank lines skipped (evidence/11's reader did the same)."""
    out = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def analyze_node(job):
    """Per-neighbor 32 counts for one node's file.

    Returns (key, {neighbor: {col: count}}). The streak state runs over the whole file while only
    in-window replies are counted, which is the one subtlety of the unit: history feeds k but is not
    itself a trial.
    """
    stage, p, seed, node, path = job
    events = read_events(path)
    starts = [e for e in events if e.get("event") == "agent_start"]
    if len(starts) != 1:
        raise SystemExit(f"{path}: expected exactly one agent_start line, found {len(starts)}")
    t0 = starts[0]["t0_ms"]
    lo, hi_edge = t0 + WIN_LO_MS, t0 + WIN_HI_MS

    rows = defaultdict(lambda: dict.fromkeys(PAIR_COLS, 0))
    held = defaultdict(set)  # origin -> seqs held as of the current line
    streak = defaultdict(int)  # neighbor -> consecutive EMPTY replies so far
    seen_new = defaultdict(int)  # neighbor -> 1 once a non-empty reply has arrived from it
    for ev in events:
        name = ev.get("event")
        if name in ("record_created", "record_stored"):
            held[ev["origin"]].add(ev["seq"])
        elif name == "state_rx":
            nb, hi = ev["neighbor"], ev["hi"]
            empty = all(s in held[o] for o, h in hi.items() for s in range(1, h + 1))
            ts = ev["ts_ms"]
            if lo <= ts <= hi_edge:
                col = f"s{seen_new[nb]}_k{bin_of(streak[nb])}"
                r = rows[nb]
                r[f"n_{col}"] += 1
                r[f"e_{col}"] += empty
            streak[nb] = streak[nb] + 1 if empty else 0
            seen_new[nb] |= not empty
    return (stage, p, seed, node), t0, {nb: dict(r) for nb, r in rows.items()}


def ev11_rows():
    """The 6720 (stage, p, seed, node, neighbor) keys of ../11/perpair.csv with its two check columns.

    The key set is taken from there rather than from the logs so that a pair that polled but never got
    an answer inside the window still appears, as an all-zero row, instead of silently vanishing.
    """
    out = {}
    with open(os.path.join(EV11, "perpair.csv")) as fh:
        for q in csv.DictReader(fh):
            key = (q["stage"], float(q["p"]), int(q["seed"]), q["node"], q["neighbor"])
            out[key] = (int(q["state_rx"]), int(q["rx_no_new"]))
    return out


def ev11_bytes():
    """B_const and empty_B_per_rh_no_new per cell from ../11/cells.csv; absent for p 0.5 / 0.6."""
    out = {}
    with open(os.path.join(EV11, "cells.csv")) as fh:
        for q in csv.DictReader(fh):
            if q["B_const"] and q["empty_B_per_rh_no_new"]:
                out[(q["stage"], float(q["p"]))] = (float(q["B_const"]),
                                                    float(q["empty_B_per_rh_no_new"]))
    return out


def jobs(keys):
    """One job per node file, in sorted order so that Pool.map's result order is reproducible."""
    runs = sorted({k[:3] for k in keys})
    js = []
    for stage, p, seed in runs:
        # p 0.0 is stored as "p0": the directory name is the shortest float repr, not the csv's "0.0".
        run_dir = os.path.join(STORE, stage, f"p{p:g}", str(seed))
        for node in NODES:
            js.append((stage, p, seed, node,
                       os.path.join(run_dir, "debug", "node_dirs", node, "semindex", "events.jsonl")))
    return js


def boot_ci(per_seed, rng):
    """Seed-level mean of a per-seed ratio and its 2.5/97.5 % bootstrap interval over seeds.

    Same convention as ../11/emptypoll.py: seed is the resampling unit, 2000 resamples. Seeds whose
    denominator was 0 arrive as nan and are dropped, so a bin that only some seeds reached is averaged
    over the seeds that reached it (their number is reported as _nseeds).
    """
    x = np.array([v for v in per_seed if not np.isnan(v)])
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    idx = rng.integers(0, len(x), size=(BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(x)


def ratio(a, b):
    """a / b, or nan when b is 0, so that an unreachable bin is dropped from a mean instead of faked."""
    return a / b if b else float("nan")


def rule_counts(acc, s=None):
    """hit/miss per K for "predict EMPTY iff k >= K", plus the two denominators of recall/miss_share.

    `s` restricts to one seen_new value; None pools both. Every K is a bin lower edge, so the rule is a
    sum of whole bins and needs no counts beyond the 32 per-pair columns.
    """
    ss = (0, 1) if s is None else (s,)
    n_tot = sum(acc[f"n_s{t}_k{b}"] for t in ss for b in BIN_LABELS)
    e_tot = sum(acc[f"e_s{t}_k{b}"] for t in ss for b in BIN_LABELS)
    out = {"n": n_tot, "e": e_tot}
    for K in KS:
        bs = [b for b, lo in BINS if lo >= K]
        hit = sum(acc[f"e_s{t}_k{b}"] for t in ss for b in bs)
        n_ge = sum(acc[f"n_s{t}_k{b}"] for t in ss for b in bs)
        out[K] = (hit, n_ge - hit)
    return out


def cell_rows(per_seed_acc, bytes_ref):
    """One row per (stage, p): pooled counts, per-seed means with CIs, and the rule read-outs.

    The rng is drawn in one fixed order (cells sorted, then the 8 bins, then K x the three ratios) so
    that the CIs are reproducible for the same data. A ratio that no seed can form (an empty k bin) takes
    no draw, so data that fills such a bin moves the later CIs of the run.
    """
    rng = np.random.default_rng(RNG_SEED)
    cells = sorted({k[:2] for k in per_seed_acc})
    out = []
    for stage, p in cells:
        seeds = sorted(s for (a, b, s) in per_seed_acc if (a, b) == (stage, p))
        accs = [per_seed_acc[(stage, p, s)] for s in seeds]
        tot = {c: sum(a[c] for a in accs) for c in PAIR_COLS}
        row = {"stage": stage, "p": p, "n_runs": len(seeds)}
        for b in BIN_LABELS:
            n = tot[f"n_s0_k{b}"] + tot[f"n_s1_k{b}"]
            e = tot[f"e_s0_k{b}"] + tot[f"e_s1_k{b}"]
            row[f"n_k{b}"], row[f"e_k{b}"] = n, e
            row[f"p_empty_k{b}_pooled"] = ratio(e, n)
            m, clo, chi, ns = boot_ci(
                [ratio(a[f"e_s0_k{b}"] + a[f"e_s1_k{b}"], a[f"n_s0_k{b}"] + a[f"n_s1_k{b}"])
                 for a in accs], rng)
            row[f"p_empty_k{b}_seedmean"] = m
            row[f"p_empty_k{b}_ci_lo"], row[f"p_empty_k{b}_ci_hi"] = clo, chi
            row[f"p_empty_k{b}_nseeds"] = ns
        for s in (0, 1):
            for b in BIN_LABELS:
                n, e = tot[f"n_s{s}_k{b}"], tot[f"e_s{s}_k{b}"]
                row[f"n_s{s}_k{b}"], row[f"e_s{s}_k{b}"] = n, e
                row[f"p_empty_s{s}_k{b}_pooled"] = ratio(e, n)
        pooled = rule_counts(tot)
        row["state_rx"], row["rx_no_new"] = pooled["n"], pooled["e"]
        row["p_empty_pooled"] = ratio(pooled["e"], pooled["n"])
        per_seed = [rule_counts(a) for a in accs]
        bc = bytes_ref.get((stage, p))
        for K in KS:
            hit, miss = pooled[K]
            row[f"hit_K{K}"], row[f"miss_K{K}"] = hit, miss
            defs = {"precision": (hit, hit + miss),
                    "recall": (hit, pooled["e"]),
                    "miss_share": (miss, pooled["n"] - pooled["e"])}
            for name, (num, den) in defs.items():
                row[f"{name}_K{K}_pooled"] = ratio(num, den)
                vals = []
                for q in per_seed:
                    h, m_ = q[K]
                    vals.append(ratio(*{"precision": (h, h + m_),
                                        "recall": (h, q["e"]),
                                        "miss_share": (m_, q["n"] - q["e"])}[name]))
                sm, clo, chi, ns = boot_ci(vals, rng)
                row[f"{name}_K{K}_seedmean"] = sm
                row[f"{name}_K{K}_ci_lo"], row[f"{name}_K{K}_ci_hi"] = clo, chi
                if name == "precision":
                    # Only precision can lose seeds (a seed with no reply at k >= K); recall and
                    # miss_share divide by totals that every seed has. Kept so that a mean over a
                    # handful of seeds is visible as such.
                    row[f"precision_K{K}_nseeds"] = ns
            if bc is not None:
                # Every answered poll costs the same in evidence/09's constant-bytes model, so the bytes
                # of the flagged empty replies scale with their count: recall x (empty bytes / all bytes).
                row[f"hit_B_share_K{K}"] = row[f"recall_K{K}_pooled"] * bc[1] / bc[0]
        s1 = rule_counts(tot, s=1)
        for K in KS:
            hit, miss = s1[K]
            row[f"hit_s1_K{K}"], row[f"miss_s1_K{K}"] = hit, miss
            row[f"precision_s1_K{K}_pooled"] = ratio(hit, hit + miss)
            row[f"recall_s1_K{K}_pooled"] = ratio(hit, s1["e"])
            row[f"miss_share_s1_K{K}_pooled"] = ratio(miss, s1["n"] - s1["e"])
        out.append(row)
    return out


def cell_fields():
    """cells.csv column order; built once so that a cell without bytes leaves those columns empty."""
    f = ["stage", "p", "n_runs"]
    for b in BIN_LABELS:
        f += [f"n_k{b}", f"e_k{b}", f"p_empty_k{b}_pooled", f"p_empty_k{b}_seedmean",
              f"p_empty_k{b}_ci_lo", f"p_empty_k{b}_ci_hi", f"p_empty_k{b}_nseeds"]
    for s in (0, 1):
        for b in BIN_LABELS:
            f += [f"n_s{s}_k{b}", f"e_s{s}_k{b}", f"p_empty_s{s}_k{b}_pooled"]
    f += ["state_rx", "rx_no_new", "p_empty_pooled"]
    for K in KS:
        f += [f"hit_K{K}", f"miss_K{K}"]
        for name in ("precision", "recall", "miss_share"):
            f += [f"{name}_K{K}_pooled", f"{name}_K{K}_seedmean",
                  f"{name}_K{K}_ci_lo", f"{name}_K{K}_ci_hi"]
        f += [f"precision_K{K}_nseeds", f"hit_B_share_K{K}", f"hit_s1_K{K}", f"miss_s1_K{K}",
              f"precision_s1_K{K}_pooled", f"recall_s1_K{K}_pooled", f"miss_share_s1_K{K}_pooled"]
    return f


def fmt(x, nd=3):
    """A table cell: a fixed-point number, or `-` for a ratio with no denominator."""
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


LEGEND = """Legend (the definitions as fixed before the counting; see streak.py's docstring):

- Unit: one directed pair (node, neighbor) per run. Window: ts_ms in [t0+10000, t0+130000] inclusive,
  t0 = the `t0_ms` of the node's own `agent_start` line.
- Holdings at a line: the (origin, seq) of every `record_created` and `record_stored` line above it.
- EMPTY reply = evidence/11's rx_no_new: a `state_rx` line from the neighbor where every seq 1..hi[o] of
  every origin o in its `hi` is already held (an empty `hi` counts as EMPTY).
- Reply sequence of a pair: ALL `state_rx` lines from that neighbor in file order from the top of the
  file. Replies before the window are history -- they are not counted, but they do feed k and seen_new.
- k = the number of consecutive EMPTY replies from that neighbor immediately before this one (0 when the
  previous reply was non-empty or there is no previous reply). seen_new = 1 when some earlier reply from
  that neighbor was non-empty. k = 0 with seen_new = 0 means "no earlier reply at all".
- k bins: 0, 1, 2, 3, 4, 5_9, 10_19, 20p. P(empty | k bin) = EMPTY replies / replies in the bin.
- Rule read-out for a threshold K: "predict EMPTY iff k >= K". hit = EMPTY replies with k >= K,
  miss = non-empty replies with k >= K, precision = hit / (hit + miss) = P(empty | k >= K),
  recall = hit / all EMPTY replies, miss_share = miss / all non-empty replies.
  hit_B_share = recall (pooled) x empty_B_per_rh_no_new / B_const from ../11/cells.csv, i.e. the flagged
  empties' share of the cell's whole state-poll byte bill under evidence/09's constant-bytes model
  (the same unit price for every answered poll); empty where that csv has no bytes (p 0.5 / 0.6).
- The `_s1_` read-outs (Table D, "within seen_new=1") apply the same rule to the seen_new = 1 replies
  only, and BOTH denominators are restricted to them: recall_s1 = hit_s1 / EMPTY replies with
  seen_new = 1, miss_share_s1 = miss_s1 / non-empty replies with seen_new = 1. They are not shares of
  the cell's totals; for those divide hit_s1 by rx_no_new and miss_s1 by state_rx - rx_no_new.
- "seed mean [CI]" = the mean over the cell's seeds of each seed's own ratio, with a 2.5-97.5 %
  percentile bootstrap over seeds (2000 resamples, numpy default_rng(20261116)), the same way
  ../11/emptypoll.py's boot_ci does it. Seeds with a zero denominator are left out of the mean; for
  precision the number of seeds left in is printed ("seeds"), because a seed with no reply at k >= K
  has no precision (K = 5 rests on a few seeds in the 10-second cells).
  "pooled" = sum / sum over the cell's 20 runs.
- Cross-check against ../11/perpair.csv: per row the n_* columns sum to its `state_rx` and the e_*
  columns to its `rx_no_new`.

**What this is NOT.** The numbers describe replies that actually arrived in the logged runs. A rule that
skips or delays polls changes which replies exist afterwards: after a skipped poll there is no reply, so
k stops growing, and a flagged non-empty reply is a record that is found later instead. recall and
hit_B_share are therefore NOT bytes saved, and miss_share is NOT a delay. They say how much of the
emptiness a streak count flags, and how often the flag is wrong, on the trajectories that the rules
actually in force produced."""


def tables(crs, checkline):
    """tables.md: the Legend, then A (P(empty | k bin)), B (rule read-outs), C and D (the six cells)."""
    L = ["# Does a streak of empty replies predict the next one? (s5*, s6*)", "", checkline, "",
         LEGEND, ""]
    L += ["## Table A: P(empty | k bin), pooled over the cell's 20 runs", "",
          "Each entry is `share (n replies)`; `-` when the bin is empty.", "",
          "| stage | p | " + " | ".join(f"k {b}" for b in BIN_LABELS) + " |",
          "|---|---|" + "---|" * len(BIN_LABELS)]
    for r in crs:
        L.append(f"| {r['stage']} | {r['p']} | " + " | ".join(
            f"{fmt(r[f'p_empty_k{b}_pooled'])} ({r[f'n_k{b}']})" for b in BIN_LABELS) + " |")
    L += ["", "## Table B: rule read-outs, \"predict EMPTY iff k >= K\" (seen_new pooled)", "",
          "| stage | p | K | precision seed mean [CI] | seeds | recall seed mean [CI] "
          "| miss_share seed mean [CI] "
          "| precision pooled | recall pooled | miss_share pooled | hit | miss | hit_B_share |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in crs:
        for K in KS:
            ci = " | ".join(f"{fmt(r[f'{c}_K{K}_seedmean'])} [{fmt(r[f'{c}_K{K}_ci_lo'])}, "
                            f"{fmt(r[f'{c}_K{K}_ci_hi'])}]"
                            + (f" | {r[f'precision_K{K}_nseeds']}" if c == "precision" else "")
                            for c in ("precision", "recall", "miss_share"))
            pooled = " | ".join(fmt(r[f"{c}_K{K}_pooled"]) for c in ("precision", "recall", "miss_share"))
            L.append(f"| {r['stage']} | {r['p']} | {K} | {ci} | {pooled} | {r[f'hit_K{K}']} | "
                     f"{r[f'miss_K{K}']} | {fmt(r.get(f'hit_B_share_K{K}'))} |")
    six = [r for r in crs if r["stage"] in SPLIT_STAGES and r["p"] in SPLIT_PS]
    L += ["", "## Table C: Table A split by seen_new (the six cells)", "",
          "seen_new = 0: no earlier reply from that neighbor was non-empty (with k = 0 this is the very "
          "first reply). seen_new = 1: the data from that neighbor had already started.", "",
          "| stage | p | seen_new | " + " | ".join(f"k {b}" for b in BIN_LABELS) + " |",
          "|---|---|---|" + "---|" * len(BIN_LABELS)]
    for r in six:
        for s in (0, 1):
            L.append(f"| {r['stage']} | {r['p']} | {s} | " + " | ".join(
                f"{fmt(r[f'p_empty_s{s}_k{b}_pooled'])} ({r[f'n_s{s}_k{b}']})"
                for b in BIN_LABELS) + " |")
    L += ["", "## Table D: where the misses are (the six cells)", "",
          "miss_K = non-empty replies the rule flags, split by seen_new. The last three columns are the "
          "`_s1_` read-outs: the rule applied to the seen_new = 1 replies only, with both denominators "
          "restricted to them (see the Legend).", "",
          "| stage | p | K | miss seen_new=0 | miss seen_new=1 | miss total | precision within seen_new=1 | "
          "recall within seen_new=1 | miss_share within seen_new=1 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in six:
        for K in KS:
            m1 = r[f"miss_s1_K{K}"]
            L.append(f"| {r['stage']} | {r['p']} | {K} | {r[f'miss_K{K}'] - m1} | {m1} | "
                     f"{r[f'miss_K{K}']} | {fmt(r[f'precision_s1_K{K}_pooled'])} | "
                     f"{fmt(r[f'recall_s1_K{K}_pooled'])} | {fmt(r[f'miss_share_s1_K{K}_pooled'])} |")
    return "\n".join(L) + "\n"


def main():
    """Read the 4200 node files, write perpair.csv, verify it against evidence/11, then cells and tables."""
    ref = ev11_rows()
    js = jobs(ref)
    with mp.get_context("fork").Pool(16) as pool:
        res = pool.map(analyze_node, js, chunksize=8)

    # One t0 per run: the window would mean different things per node otherwise, and evidence/11 derived
    # a single run-level t0, so a disagreement here would silently break the cross-check.
    t0s = defaultdict(set)
    seen = {}
    for (stage, p, seed, node), t0, per_nb in res:
        t0s[(stage, p, seed)].add(t0)
        for nb, counts in per_nb.items():
            seen[(stage, p, seed, node, nb)] = counts
    bad_t0 = sorted(k for k, v in t0s.items() if len(v) != 1)
    if bad_t0:
        raise SystemExit(f"nodes disagree about t0_ms in {len(bad_t0)} runs: {bad_t0[:5]}")
    extra = sorted(k for k in seen if k not in ref)
    if extra:
        raise SystemExit(f"{len(extra)} pairs answered that ../11/perpair.csv does not list: {extra[:5]}")

    rows = []
    for key in sorted(ref):
        counts = seen.get(key, dict.fromkeys(PAIR_COLS, 0))
        stage, p, seed, node, nb = key
        rows.append(dict(stage=stage, p=p, seed=seed, node=node, neighbor=nb, **counts))
    with open(os.path.join(HERE, "perpair.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, ["stage", "p", "seed", "node", "neighbor", *PAIR_COLS])
        w.writeheader()
        w.writerows(rows)

    bad_n = bad_e = 0
    for r in rows:
        want_n, want_e = ref[(r["stage"], r["p"], r["seed"], r["node"], r["neighbor"])]
        bad_n += sum(r[c] for c in PAIR_COLS if c[0] == "n") != want_n
        bad_e += sum(r[c] for c in PAIR_COLS if c[0] == "e") != want_e
    checkline = (f"check against ../11/perpair.csv over {len(rows)} rows: {bad_n} rows where sum(n_*) != "
                 f"state_rx, {bad_e} rows where sum(e_*) != rx_no_new")
    print(checkline)
    print("node files read", len(js), "cells", len({k[:2] for k in ref}), "pairs", len(rows))
    if bad_n or bad_e:
        # perpair.csv is already on disk, so the failing rows can be diffed; stop before publishing
        # cells.csv and tables.md, which would otherwise look like checked numbers.
        raise SystemExit("cross-check failed; cells.csv and tables.md not written")

    acc = defaultdict(lambda: dict.fromkeys(PAIR_COLS, 0))
    for r in rows:
        a = acc[(r["stage"], r["p"], r["seed"])]
        for c in PAIR_COLS:
            a[c] += r[c]
    crs = cell_rows(dict(acc), ev11_bytes())
    with open(os.path.join(HERE, "cells.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, cell_fields())
        w.writeheader()
        w.writerows(crs)
    with open(os.path.join(HERE, "tables.md"), "w") as fo:
        fo.write(tables(crs, checkline))


if __name__ == "__main__":
    main()
