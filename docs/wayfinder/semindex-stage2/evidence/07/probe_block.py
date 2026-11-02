"""How long h2's scheduled probes block its agent loop: probe_retrieved.ts - probe_discovered.ts, per stage x p.

A probe with no match writes only probe_discovered (no content fetch, ~0 ms block).
Share = summed block time / measurement window (120 s) per run, averaged.
Usage: probe_block.py <eval_root> stage [stage ...]
"""
import glob, json, statistics, sys, collections
root = sys.argv[1]
for stage in sys.argv[2:]:
    by_p = collections.defaultdict(list); share = collections.defaultdict(list); nomatch = collections.Counter(); fails = collections.Counter()
    for f in glob.glob(f"{root}/{stage}/*/*/debug/node_dirs/h2/semindex/events.jsonl"):
        p = f.split("/")[-7]
        ev = [json.loads(l) for l in open(f)]
        tot = 0
        for i, e in enumerate(ev):
            if e["event"] == "probe_discovered":
                if not e["matches"]:
                    nomatch[p] += 1; continue
                r = next((x for x in ev[i + 1:] if x["event"] == "probe_retrieved"), None)
                if r is None:
                    continue
                d = r["ts_ms"] - e["ts_ms"]; by_p[p].append(d); tot += d
                if not r.get("success", True):
                    fails[p] += 1
        share[p].append(tot / 120000)
    for p in sorted(by_p):
        xs = sorted(by_p[p]); q = lambda a: xs[int(a * (len(xs) - 1))]
        print(f"{stage} {p}: fetches={len(xs)} nomatch={nomatch[p]} failed={fails[p]} block ms median={q(.5)} p90={q(.9)} max={xs[-1]} mean_share_of_window={statistics.mean(share[p])*100:.1f}%")
