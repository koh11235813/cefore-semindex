"""Independent recount of emptypoll.py's per-pair counts, written without its code.

Why this exists: emptypoll.py goes through semindex.evalagg for the window and the file reader. A slip
there would be invisible in its own output. This script shares no code with emptypoll.py or the semindex
package: it reads each node's events.jsonl with the json module only, takes t0 from the node's own
`agent_start` line, and recounts the ten columns defined below for every (stage, p, seed, node, neighbor)
row of perpair.csv.

Not recounted here: late_replies, rx_hi_unchanged_but_new, the mm_* columns, and everything in cells.csv
that is derived (shares, CIs, bytes). The mm_nfbn_* reason split is the one part of emptypoll.py that
replays the engine's fetch bookkeeping, and it has no independent recount; its cross-checks are
mm_nfbn_other == mm_fbnn_other == 0 and the per-pair agreement of rx_no_fetch and rx_no_new here.

Reported but not compared (perpair.csv has no such columns):

- during_after_gen: the `during` replies that arrived later than t0+72000 ms, i.e. after the last record
  was created (generation runs from t0+13 s to about t0+71.1 s). It shows how much of `during` is
  propagation after the end of generation.
- seq_*: what two consecutive `state_poll` decision states for the same neighbor show about the reply
  that arrived between them. One decision state does not record whether the last reply brought anything
  new, but each carries the node's own contiguous hi per origin (`state.origins[o].hi`). For every
  in-window state_rx, take the last state_poll decision to that neighbor above it and the first one below
  it, and ask whether the node's own hi for the SIDE origin is higher in the later state:
    seq_new_adv / seq_new_noadv     the reply brought something new (not rx_no_new); hi advanced / did not
    seq_empty_adv / seq_empty_noadv the reply was rx_no_new; hi advanced / did not
    seq_multi                       more than one reply from that neighbor between the two decisions
    seq_no_next                     no decision to that neighbor above or below the reply
  The six sum to state_rx. It relies on the same side-origin rule as the why_* split (line topology).

Definitions (restated here so that the recount does not depend on emptypoll.py's docstring):

- Window: ts_ms in [t0+10000, t0+130000] inclusive, t0 = the `t0_ms` field of the node's `agent_start`
  line (the same value on every node of a run).
- Holdings at a line: (origin, seq) of every `record_created` and `record_stored` line above it.
- polls_sent: `decision` lines of kind `state_poll` for the neighbor.
- For each `state_rx` line from the neighbor:
  - rx_no_fetch: the next line is not a `fetch_plan` decision for the same neighbor at the same ts_ms.
  - rx_no_new: every seq 1..hi[o] of every origin o in the reply's `hi` is held.
  - rx_hi_unchanged: `hi` equals the previous state_rx's `hi` from that neighbor (missing origin = 0;
    the previous reply may lie before the window; the first reply is compared with all zeros).
  - rx_after_complete: the node holds 120 records of h0 and 120 of h4.
  - Why an rx_no_new reply was empty, first match wins. Only the origin on the neighbor's side of the
    line can be new (h0 if the neighbor's index is lower than the node's, else h4):
    all_complete (= rx_after_complete), side_complete (all 120 of the side origin held), side_none
    (the reply's hi for the side origin is 0 or missing), during (the rest).

Usage (in this directory): python3 recheck.py > recheck.txt
Environment: STORE (default /home/lab_shared/semindex-eval).
"""

import csv
import glob
import json
import multiprocessing as mp
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")

# perpair.csv column -> this script's counter. The four why-empty names are filled from the header.
BASE_COLS = ("polls_sent", "state_rx", "rx_no_fetch", "rx_no_new", "rx_hi_unchanged", "rx_after_complete")
WHY = ("all_complete", "side_complete", "side_none", "during")
EXTRA = (  # reported only, see the docstring
    "during_after_gen",
    "seq_new_adv",
    "seq_new_noadv",
    "seq_empty_adv",
    "seq_empty_noadv",
    "seq_multi",
    "seq_no_next",
)
GEN_END_MS = 72000  # the last record_created of any run is at t0 + 71.1 s
N_PER_ORIGIN = 120  # 30 generations x k 4; emptypoll.py's run log reports 120 for every run


def count_node(path):
    """Recount one node's events.jsonl; returns {neighbor: {counter: n}}."""
    with open(path) as f:
        lines = [json.loads(x) for x in f if x.strip()]
    t0 = next(e["t0_ms"] for e in lines if e.get("event") == "agent_start")
    lo, hi_edge = t0 + 10000, t0 + 130000
    node = int(path.split("/node_dirs/h")[1].split("/")[0])
    held = {"h0": set(), "h4": set()}
    prev = {}
    out = {}
    # For the seq_* counters, per neighbor: the own-hi snapshot of the last state_poll decision, and the
    # in-window replies seen since then as (was the reply empty, was it counted in the window).
    last_poll_hi = {}
    since_poll = {}

    def own_hi(decision):
        return {o: v["hi"] for o, v in decision["state"]["origins"].items()}

    def settle(nb, now_hi):
        """Tally the replies from `nb` that arrived between the previous decision and this one."""
        replies = since_poll.pop(nb, [])
        before = last_poll_hi.get(nb)
        side = "h0" if int(nb[1:]) < node else "h4"
        for empty, in_window in replies:
            if not in_window:
                continue
            c = out[nb]
            if before is None or now_hi is None:
                c["seq_no_next"] += 1
            elif len(replies) > 1:
                c["seq_multi"] += 1
            else:
                adv = now_hi.get(side, 0) > before.get(side, 0)
                c[("seq_empty_" if empty else "seq_new_") + ("adv" if adv else "noadv")] += 1

    for i, e in enumerate(lines):
        ev = e.get("event")
        if ev in ("record_created", "record_stored"):
            held[e["origin"]].add(e["seq"])
            continue
        if ev == "decision" and e["kind"] == "state_poll":
            nb_polled = e["action"]["neighbor"]
            settle(nb_polled, own_hi(e))
            last_poll_hi[nb_polled] = own_hi(e)
            if lo <= e["ts_ms"] <= hi_edge:
                c = out.setdefault(nb_polled, dict.fromkeys(BASE_COLS + WHY + EXTRA, 0))
                c["polls_sent"] += 1
        if ev != "state_rx":
            continue
        nb, adv = e["neighbor"], e["hi"]
        last = prev.get(nb, {})
        prev[nb] = adv  # updated before the window test: the previous reply may lie outside the window
        is_new = any(s not in held[o] for o, v in adv.items() for s in range(1, v + 1))
        in_window = lo <= e["ts_ms"] <= hi_edge
        since_poll.setdefault(nb, []).append((not is_new, in_window))
        if not in_window:
            continue
        c = out.setdefault(nb, dict.fromkeys(BASE_COLS + WHY + EXTRA, 0))
        c["state_rx"] += 1
        nxt = lines[i + 1] if i + 1 < len(lines) else {}
        fetched = (
            nxt.get("event") == "decision"
            and nxt.get("kind") == "fetch_plan"
            and nxt["action"]["neighbor"] == nb
            and nxt["ts_ms"] == e["ts_ms"]
        )
        c["rx_no_fetch"] += not fetched
        c["rx_hi_unchanged"] += all(adv.get(o, 0) == last.get(o, 0) for o in set(adv) | set(last))
        complete = len(held["h0"]) == N_PER_ORIGIN and len(held["h4"]) == N_PER_ORIGIN
        c["rx_after_complete"] += complete
        if is_new:
            continue
        c["rx_no_new"] += 1
        side = "h0" if int(nb[1:]) < node else "h4"
        if complete:
            c["all_complete"] += 1
        elif len(held[side]) == N_PER_ORIGIN:
            c["side_complete"] += 1
        elif adv.get(side, 0) == 0:
            c["side_none"] += 1
        else:
            c["during"] += 1
            c["during_after_gen"] += e["ts_ms"] > t0 + GEN_END_MS
    for nb in list(since_poll):
        settle(nb, None)  # replies with no later decision to that neighbor
    return path, out


def main():
    with open(f"{HERE}/perpair.csv", newline="") as f:
        ref_rows = list(csv.DictReader(f))
    header = ref_rows[0].keys()
    # emptypoll.py names the why-empty columns itself; find each by its suffix so that a rename there
    # does not silently drop the comparison (a missing column stops the script).
    colmap = {c: c for c in BASE_COLS}
    for w in WHY:
        hits = [h for h in header if h.endswith(w)]
        if len(hits) != 1:
            raise SystemExit(f"perpair.csv: expected one column ending in {w}, found {hits}")
        colmap[w] = hits[0]
    ref = {(r["stage"], r["p"], r["seed"], r["node"], r["neighbor"]): r for r in ref_rows}

    paths = sorted(glob.glob(f"{STORE}/s[56]*/p*/*/debug/node_dirs/h*/semindex/events.jsonl"))
    # Recount only the (stage, p) cells emptypoll.py reports. The store holds more (s5a, s5c, s6smoke and
    # the fixed arms' p 0.1 / 0.3 runs), and a recount of those has no perpair.csv row to be compared with.
    cells = {(r["stage"], r["p"]) for r in ref_rows}

    def cell_of(path):
        stage, pdir = path[len(STORE) + 1 :].split("/")[:2]
        return stage, str(float(pdir[1:]))

    paths = [p for p in paths if cell_of(p) in cells]
    with mp.Pool(16) as pool:
        results = pool.map(count_node, paths, chunksize=8)

    seen, diffs = set(), []
    tot = {}
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
            for mine, theirs in colmap.items():
                if c[mine] != int(r[theirs]):
                    diffs.append((key, f"{mine}: recount {c[mine]} vs perpair.csv {r[theirs]}"))
            t = tot.setdefault((stage, key[1]), dict.fromkeys(BASE_COLS + WHY + EXTRA, 0))
            for k, v in c.items():
                t[k] += v
    missing = sorted(set(ref) - seen)

    print(f"cells: {len(cells)}; node files read: {len(paths)}; pairs recounted: {len(seen)}; perpair.csv rows: {len(ref)}")
    print(f"perpair.csv rows not recounted: {len(missing)}")
    print(f"columns compared per pair: {', '.join(colmap.values())}")
    print(f"differences: {len(diffs)}")
    for d in diffs[:40]:
        print("  ", d)
    for m in missing[:10]:
        print("   not recounted:", m)
    print()
    print("recounted cell totals (stage, p, then " + ", ".join(BASE_COLS + WHY + EXTRA) + ")")
    for (stage, p), t in sorted(tot.items()):
        print(stage, p, *(t[k] for k in BASE_COLS + WHY + EXTRA))


if __name__ == "__main__":
    main()
