"""List in-step processing intervals (commit_time.py's N1-N0) >= a threshold, with run, node, time and record index.

The interval is a record_stored line followed directly by a fetch_plan to the same holder (the
evidence/05 commit_time.py rule; intervals of 2 s or more are left out, as there). `t+` is seconds
since the node's first event line (agent_start's write time, a few hundred ms after t0).
`probe_block` says whether the interval overlaps one of this node's matched-probe blocks
([probe_discovered, next probe_retrieved], the probe_extract.py rule; only h2 runs probes), so a
stall from the synchronous probe fetch can be told apart from one with another cause.
Usage: slow_steps.py <store> <threshold_ms> stage [stage ...] > slow_steps.txt
"""
import glob, json, sys
root, thr = sys.argv[1], int(sys.argv[2])
for stage in sys.argv[3:]:
    for path in sorted(glob.glob(f"{root}/{stage}/*/*/debug/node_dirs/h*/semindex/events.jsonl")):
        # .../<stage>/p<p>/<seed>/debug/node_dirs/<node>/semindex/events.jsonl
        # 2026-11-02 fix: p and seed were read one level too far right (seed printed as "debug").
        parts = path.split("/")
        pdir, seed, node = parts[-7], parts[-6], parts[-3]
        ev = [json.loads(l) for l in open(path)]
        t0 = ev[0]["ts_ms"]
        blocks = []
        for i, e in enumerate(ev):
            if e["event"] == "probe_discovered" and e["matches"]:
                r = next((x for x in ev[i + 1:] if x["event"] in ("probe_retrieved", "probe_discovered")), None)
                if r is not None and r["event"] == "probe_retrieved":
                    blocks.append((e["ts_ms"], r["ts_ms"]))
        nstored = 0
        for i in range(len(ev) - 1):
            a, b = ev[i], ev[i + 1]
            if a["event"] == "record_stored":
                nstored += 1
            if a["event"] == "record_stored" and b["event"] == "decision" and b["kind"] == "fetch_plan" \
                    and b["action"]["neighbor"] == a["holder"]:
                d = b["ts_ms"] - a["ts_ms"]
                if thr <= d < 2000:
                    inb = any(b0 <= b["ts_ms"] and b1 >= a["ts_ms"] for b0, b1 in blocks)
                    print(f"{stage} {pdir} seed={seed} {node} d={d}ms at={a['ts_ms']} t+{(a['ts_ms']-t0)/1000:.1f}s "
                          f"stored#{nstored} probe_block={'yes' if inb else 'no'}")
