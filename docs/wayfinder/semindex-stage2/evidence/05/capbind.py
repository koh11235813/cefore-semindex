"""Does the in-flight cap make big reply batches slow, or does the upstream's per-Data service?

Batch = records newly covered by one state_rx (hi only); fetch = n's record_stored - that state_rx.
Prints one line per stage/p: per batch-size bucket, count, mean fetch and median per-record service.
Read-only over the store.

Usage: capbind.py [--store DIR]   (default /home/lab_shared/semindex-eval)
"""
import argparse, json, sys, statistics, collections
from multiprocessing import Pool
ST = "/home/lab_shared/semindex-eval"
POS = {f"h{i}": i for i in range(5)}
def ev(d, n):
    """Node `n`'s semindex event log in run dir `d`, as a list of event dicts."""
    return [json.loads(l) for l in open(f"{d}/debug/node_dirs/{n}/semindex/events.jsonl")]
def one(args):
    """(batch size, mean fetch, max fetch) per covering state_rx over one run; [] if invalid or missing.

    The store path rides in `args` so workers do not depend on how the pool starts them.
    """
    store, stage, p, seed = args
    d = f"{store}/{stage}/p{p}/{seed}"
    try:
        if not json.load(open(f"{d}/meta.json"))["validity"]["valid"]:
            return []
    except FileNotFoundError:
        return []
    E = {n: ev(d, n) for n in POS}
    out = []
    for o in ("h0", "h4"):
        step = 1 if o == "h0" else -1
        for hop in range(1, 5):
            n = f"h{POS[o] + step * hop}"; m = f"h{POS[o] + step * (hop - 1)}"
            stored = {e["seq"]: e["ts_ms"] for e in E[n] if e.get("event") == "record_stored" and e["origin"] == o}
            last = 0
            for e in E[n]:
                if e.get("event") != "state_rx" or e["neighbor"] != m:
                    continue
                h = e["hi"].get(o, 0)
                if h <= last:
                    continue
                new = [s for s in range(last + 1, h + 1) if s in stored and stored[s] >= e["ts_ms"]]
                b = h - last; last = h
                if not new:
                    continue
                f = [stored[s] - e["ts_ms"] for s in new]
                out.append((b, statistics.mean(f), max(f)))
    return out
def bucket(b):
    """Batch-size bucket label; edges sit at the default in-flight cap (8) and its double."""
    return "1-4" if b <= 4 else "5-8" if b <= 8 else "9-16" if b <= 16 else "17+"
def report(label, jobs):
    """Pool all batches of `jobs` and print one summary line per batch-size bucket under `label`."""
    with Pool(6) as pool:
        rows = [r for rs in pool.map(one, jobs) for r in rs]
    acc = collections.defaultdict(list)
    for b, fm, fx in rows:
        acc[bucket(b)].append((b, fm, fx))
    cells = []
    for k in ("1-4", "5-8", "9-16", "17+"):
        v = acc.get(k, [])
        if not v:
            cells.append(f"{k}: -"); continue
        # per-Data service estimate = last record's fetch / batch size
        svc = statistics.median(fx / b for b, fm, fx in v)
        cells.append(f"{k}: n={len(v)} mean_fetch={statistics.mean(fm for _, fm, _ in v):.0f}ms svc/rec={svc:.0f}ms")
    print(label, " | ".join(cells))
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=ST)
    a = ap.parse_args()
    for stage in ["s2b0p5", "s2b10p043", "s2a"]:
        seeds = range(1, 101) if stage == "s2a" else range(1, 21)
        report(f"{stage} p0", [(a.store, stage, 0, s) for s in seeds])
    for stage in ["t2k4", "t2c32k4"]:
        report(f"{stage} p0.4", [(a.store, stage, 0.4, s) for s in range(1, 4)])
