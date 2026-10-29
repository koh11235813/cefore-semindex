"""Summarise the Q24 check: per stage, per-record handling interval, slot wait, receive wait by outstanding.

Reuses evidence/05's fetchsplit.one (slot wait vs. receive wait per record-hop) and the
commit_time.py interval (record_stored -> next fetch_plan to the same holder, same step).
Usage: micro24_an.py STAGE:P [STAGE:P ...]   (from the repo root or anywhere; reads the store read-only)
"""
import json, glob, os, statistics, collections, sys
from multiprocessing import Pool
# fetchsplit.py lives in the sibling evidence/05 directory; import it from there unchanged.
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "05")))
import fetchsplit
ST = "/home/lab_shared/semindex-eval"
def handling(stage, p):
    """record_stored -> next fetch_plan (same holder) gaps in ms, as commit_time.py measures them."""
    out = []
    for f in glob.glob(f"{ST}/{stage}/p{p}/*/debug/node_dirs/h*/semindex/events.jsonl"):
        ev = [json.loads(l) for l in open(f)]
        for a, b in zip(ev, ev[1:]):
            if a["event"] == "record_stored" and b.get("kind") == "fetch_plan" and b["action"]["neighbor"] == a["holder"]:
                d = b["ts_ms"] - a["ts_ms"]
                if 0 <= d < 2000:
                    out.append(d)
    return sorted(out)
def arrival_gaps(stage, p):
    """Gaps (ms) between consecutive record_stored lines from the same holder at one node, under 2 s.

    With several fetches outstanding, the next Data is already waiting, so this gap is the
    fetcher's handling time of the previous record; it exists even where no fetch_plan follows.
    """
    out = []
    for f in glob.glob(f"{ST}/{stage}/p{p}/*/debug/node_dirs/h*/semindex/events.jsonl"):
        last = {}
        for l in open(f):
            e = json.loads(l)
            if e["event"] == "record_stored":
                t0 = last.get(e["holder"])
                if t0 is not None and e["ts_ms"] - t0 < 2000:
                    out.append(e["ts_ms"] - t0)
                last[e["holder"]] = e["ts_ms"]
    return sorted(out)
def q(xs, f):
    """Nearest-rank quantile."""
    return xs[int(f * (len(xs) - 1))] if xs else None
if __name__ == "__main__":
    for arg in sys.argv[1:]:
        stage, p = arg.split(":")
        h = handling(stage, p)
        g = arrival_gaps(stage, p)
        seeds = sorted(int(x.split("/")[-1]) for x in glob.glob(f"{ST}/{stage}/p{p}/*") if x.split("/")[-1].isdigit())
        with Pool(6) as pool:
            rows = [r for rs in pool.map(fetchsplit.one, [(ST, stage, p, s) for s in seeds]) for r in rs]
        by = collections.defaultdict(list)
        for _, rt, k in rows:
            by[fetchsplit.bucket(k)].append(rt)
        cells = " | ".join(f"{b}: {statistics.median(by[b]):.0f}ms (n={len(by[b])})" if by[b] else f"{b}: -" for b in ("1", "2-4", "5-8", "9+"))
        sw = statistics.mean(r[0] for r in rows) if rows else float("nan")
        rw = statistics.mean(r[1] for r in rows) if rows else float("nan")
        print(f"{stage} p{p} handling n={len(h)} p10={q(h,.1)} median={q(h,.5)} p90={q(h,.9)} "
              f"p99={q(h,.99)} max={h[-1] if h else None} | "
              f"arrival gap n={len(g)} p10={q(g,.1)} median={q(g,.5)} | "
              f"fetch n={len(rows)} slot_wait={sw:.0f}ms receive_wait={rw:.0f}ms | receive wait by outstanding: {cells}")
