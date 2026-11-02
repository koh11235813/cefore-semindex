"""Independent spot checks of ope05 (written separately from attrib.py).

(a) B stages' mean latency 4 hops from the origin at p = 0, from evalagg's discovery.csv;
(b) hop-1 poll wait / chosen interval from raw events; (c) fetch_plan sizes at h2/h3, B 0.5 s
vs the mixed stage. Reads the store and the evalagg dir read-only.

Usage: indep_check.py --agg DIR [--store DIR] [--prefix s2]   (store default /home/lab_shared/semindex-eval;
  --prefix: stage-name family, s2 = FULL main run, s4 = NORMAL re-collection)
"""
import argparse, csv, json, os, statistics, collections, sys
ap = argparse.ArgumentParser()
ap.add_argument("--agg", required=True, help="evalagg output dir holding <stage>/runs.csv and discovery.csv")
ap.add_argument("--store", default="/home/lab_shared/semindex-eval")
# 2026-11-02: stage names come from --prefix so the same code reads the NORMAL re-collection (s4*).
ap.add_argument("--prefix", default="s2", help="stage-name family (s2 = FULL main run, s4 = NORMAL)")
_a = ap.parse_args()
P = _a.prefix
ST = _a.store
AGG = _a.agg
def ev(stage, p, seed, node):
    """Node `node`'s semindex event log for one run, as a list of event dicts."""
    f = f"{ST}/{stage}/p{p}/{seed}/debug/node_dirs/{node}/semindex/events.jsonl"
    return [json.loads(l) for l in open(f)]
def valid(stage, p, seed):
    """Whether evalrun judged the run valid (invalid runs are excluded from every check)."""
    return json.load(open(f"{ST}/{stage}/p{p}/{seed}/meta.json"))["validity"]["valid"]
# (a) hop-4 mean latency, uncensored, valid runs, from discovery.csv
for stage in [P + "b0p5", P + "b1p359", P + "b3p695", P + "b10p043"]:
    vs = {r["seed"] for r in csv.DictReader(open(f"{AGG}/{stage}/runs.csv")) if r["p"] in ("0.0", "0") and r["valid"] == "valid"}
    lat = [float(r["latency_ms"]) for r in csv.DictReader(open(f"{AGG}/{stage}/discovery.csv"))
           if r["p"] in ("0.0", "0") and r["seed"] in vs and r["hop_distance"] == "4" and r["censored"] in ("False", "false", "0", "")]
    print("(a)", stage, "p0 hop4 mean", round(statistics.mean(lat)), "n", len(lat))
# (b) hop-1 wait / interval, B 1.359 s and 0.5 s at p0, hi-only (origin holdings are contiguous)
for stage, iv in [(P + "b1p359", 1.359), (P + "b0p5", 0.5)]:
    waits = []
    for seed in range(1, 21):
        if not valid(stage, 0, seed): continue
        for n, m in [("h1", "h0"), ("h3", "h4")]:
            arr = {e["seq"]: e["ts_ms"] for e in ev(stage, 0, seed, m) if e.get("event") == "record_created" and e["origin"] == m}
            rx = [(e["ts_ms"], e["hi"].get(m, 0)) for e in ev(stage, 0, seed, n) if e.get("event") == "state_rx" and e["neighbor"] == m]
            t0 = min(arr.values())
            for s, a in arr.items():
                if a < t0 + 10000 or a > t0 + 100000: continue
                l = next((t for t, h in rx if t > a and h >= s), None)
                if l is not None: waits.append(l - a)
    print("(b)", stage, "hop1 wait/interval", round(statistics.mean(waits) / 1000 / iv, 3), "n", len(waits))
# (c) fetch_plan sizes at hops 2-4 (n=h2,h3 toward h1/h2...), p0: A stage vs B 0.5 s
for stage, seeds in [(P + "b0p5", range(1, 21)), (P + "a", range(1, 41))]:
    sizes = []
    for seed in seeds:
        if not valid(stage, 0, seed): continue
        for n in ["h2", "h3"]:
            sizes += [len(e["action"]["fetched"]) for e in ev(stage, 0, seed, n) if e.get("kind") == "fetch_plan"]
    c = collections.Counter(sizes)
    print("(c)", stage, "fetch_plan size mean", round(statistics.mean(sizes), 2), "share==8", round(c[8] / len(sizes), 3), "n", len(sizes))
