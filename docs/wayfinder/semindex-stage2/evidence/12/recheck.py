"""Independent recount of streak.py's per-pair counts, written without its code.

Why this exists: streak.py's own check only ties its row sums to evidence/11 (state_rx, rx_no_new). A slip
in how the streak length k or seen_new is carried from one reply to the next would leave those sums intact.
This script shares no code with streak.py or the semindex package: it reads each node's events.jsonl with
the json module only and recounts all 32 count columns of every perpair.csv row.

Definitions (restated so that the recount does not depend on streak.py's docstring):

- Window: ts_ms in [t0+10000, t0+130000] inclusive, t0 = `t0_ms` of the node's `agent_start` line.
- Holdings at a line: (origin, seq) of every `record_created` and `record_stored` line above it.
- A `state_rx` reply is EMPTY when every seq 1..hi[o] of every origin o in its `hi` is held.
- Per (node, neighbor), over ALL replies from that neighbor in file order (replies before the window are
  history): k = the number of consecutive EMPTY replies immediately before this one, seen_new = whether an
  earlier reply from that neighbor was non-empty. Only replies inside the window are counted.
- k bins: 0, 1, 2, 3, 4, 5_9, 10_19, 20p. Columns n_s{seen_new}_k{bin} (replies), e_s{seen_new}_k{bin}
  (EMPTY ones).

Reported but not compared (perpair.csv has no such columns): what a flagged non-empty reply carried.
A record (origin, seq) is "first sighted" by a node at the first reply, from any neighbor, whose hi covers
it while the node does not hold it; each record is counted once per node. For K in (1, 2, 3, 5):
  sight_total      records first sighted in a reply inside the window
  sight_K{K}       of them, first sighted in a reply with k >= K (a reply the rule "predict EMPTY iff
                   k >= K" would have called empty)
  sight_s1_total   records first sighted in a reply with seen_new = 1 (after the first data from that
                   neighbor)
  sight_s1_K{K}    of them, first sighted in a reply with k >= K
It measures the size of a wrong flag in records. It is still not a delay: how late a flagged record would
be found depends on the rule that replaces the poll.

Usage (in this directory): python3 recheck.py > recheck.txt
Reads perpair.csv and, for the set of rows only, ../11/perpair.csv.
Environment: STORE (default /home/lab_shared/semindex-eval).
"""

import csv
import glob
import json
import multiprocessing as mp
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")

BINS = ("0", "1", "2", "3", "4", "5_9", "10_19", "20p")
KS = (1, 2, 3, 5)
COLS = tuple(f"{x}_s{s}_k{b}" for s in (0, 1) for b in BINS for x in ("n", "e"))
EXTRA = ("sight_total",) + tuple(f"sight_K{k}" for k in KS) + ("sight_s1_total",) + tuple(f"sight_s1_K{k}" for k in KS)


def bin_of(k):
    if k <= 4:
        return str(k)
    if k <= 9:
        return "5_9"
    return "10_19" if k <= 19 else "20p"


def count_node(path):
    """Recount one node's events.jsonl; returns (path, {neighbor: {column: n}})."""
    with open(path) as f:
        lines = [json.loads(x) for x in f if x.strip()]
    t0 = next(e["t0_ms"] for e in lines if e.get("event") == "agent_start")
    lo, hi_edge = t0 + 10000, t0 + 130000
    held = set()
    sighted = set()  # (origin, seq) already seen as new in some reply, whatever the neighbor
    streak = {}  # neighbor -> consecutive EMPTY replies so far
    seen_new = {}  # neighbor -> a non-empty reply has arrived
    out = {}
    for e in lines:
        ev = e.get("event")
        if ev in ("record_created", "record_stored"):
            held.add((e["origin"], e["seq"]))
            continue
        if ev != "state_rx":
            continue
        nb = e["neighbor"]
        new = [(o, s) for o, v in e["hi"].items() for s in range(1, v + 1) if (o, s) not in held]
        fresh = [x for x in new if x not in sighted]
        sighted.update(fresh)
        k, s = streak.get(nb, 0), seen_new.get(nb, 0)
        # History is updated before the window test: replies before the window feed k and seen_new.
        streak[nb] = 0 if new else k + 1
        seen_new[nb] = 1 if new else s
        if not lo <= e["ts_ms"] <= hi_edge:
            continue
        c = out.setdefault(nb, dict.fromkeys(COLS + EXTRA, 0))
        b = bin_of(k)
        c[f"n_s{s}_k{b}"] += 1
        c[f"e_s{s}_k{b}"] += not new
        c["sight_total"] += len(fresh)
        c["sight_s1_total"] += len(fresh) * s
        for kk in KS:
            if k >= kk:
                c[f"sight_K{kk}"] += len(fresh)
                c[f"sight_s1_K{kk}"] += len(fresh) * s
    return path, out


def main():
    with open(f"{HERE}/perpair.csv", newline="") as f:
        ref_rows = list(csv.DictReader(f))
    absent = [c for c in COLS if c not in ref_rows[0]]
    if absent:
        raise SystemExit(f"perpair.csv lacks columns: {absent}")
    ref = {(r["stage"], r["p"], r["seed"], r["node"], r["neighbor"]): r for r in ref_rows}
    # perpair.csv is streak.py's own output, so a pair it dropped would be missing from the recount too.
    # Tie the row set to evidence/11's table, which was produced and rechecked separately.
    with open(f"{HERE}/../11/perpair.csv", newline="") as f:
        ev11 = {(r["stage"], r["p"], r["seed"], r["node"], r["neighbor"]) for r in csv.DictReader(f)}
    if set(ref) != ev11:
        raise SystemExit(f"perpair.csv rows differ from ../11/perpair.csv: {len(set(ref) ^ ev11)} keys")
    # Recount only the (stage, p) cells streak.py reports; the store holds more stages and p values.
    cells = {(r["stage"], r["p"]) for r in ref_rows}

    def cell_of(path):
        stage, pdir = path[len(STORE) + 1 :].split("/")[:2]
        return stage, str(float(pdir[1:]))

    paths = sorted(glob.glob(f"{STORE}/s[56]*/p*/*/debug/node_dirs/h*/semindex/events.jsonl"))
    paths = [p for p in paths if cell_of(p) in cells]
    with mp.Pool(16) as pool:
        results = pool.map(count_node, paths, chunksize=8)

    seen, diffs, tot = set(), [], {}
    for path, per_nb in results:
        stage, pdir, seed = path[len(STORE) + 1 :].split("/")[:3]
        node = "h" + path.split("/node_dirs/h")[1].split("/")[0]
        for nb, c in per_nb.items():
            key = (stage, str(float(pdir[1:])), seed, node, nb)
            seen.add(key)
            r = ref.get(key)
            if r is None:
                diffs.append((key, "row missing in perpair.csv"))
                continue
            for col in COLS:
                if c[col] != int(r[col]):
                    diffs.append((key, f"{col}: recount {c[col]} vs perpair.csv {r[col]}"))
            t = tot.setdefault((stage, key[1]), dict.fromkeys(COLS + EXTRA, 0))
            for col, v in c.items():
                t[col] += v
    missing = sorted(set(ref) - seen)

    print(f"cells: {len(cells)}; node files read: {len(paths)}; pairs recounted: {len(seen)}; perpair.csv rows: {len(ref)}")
    print(f"perpair.csv rows not recounted: {len(missing)}")
    print(f"perpair.csv row keys equal to ../11/perpair.csv: yes ({len(ev11)})")
    print(f"columns compared per pair: {len(COLS)}")
    print(f"differences: {len(diffs)}")
    for d in diffs[:40]:
        print("  ", d)
    for m in missing[:10]:
        print("   not recounted:", m)
    print()
    # Cell totals in a short form: replies and EMPTY ones per k bin with seen_new pooled, then the extras.
    print("recounted cell totals (stage, p, then n/e per k bin " + " ".join(BINS) + ", then " + ", ".join(EXTRA) + ")")
    for (stage, p), t in sorted(tot.items()):
        per_bin = [f"{t[f'n_s0_k{b}'] + t[f'n_s1_k{b}']}/{t[f'e_s0_k{b}'] + t[f'e_s1_k{b}']}" for b in BINS]
        print(stage, p, *per_bin, *(t[k] for k in EXTRA))


if __name__ == "__main__":
    main()
