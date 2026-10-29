"""Split each record-hop's fetch time into the in-flight-slot wait and the round trip.

fetch = n's record_stored - learned (first state_rx from the upstream m whose hi covers the
record, as in capbind.py). slot wait = first fetch_plan of n toward m that lists the record -
learned: time spent waiting for room under the per-neighbour in-flight cap. round trip = stored -
that fetch_plan: the Interest's time out and back through the upstream. The round trip is also
split by how many fetches toward m were outstanding when that Interest went out (fetch_plan
state.in_flight + its position in the plan), which shows whether the upstream answers queued
Interests one after another. Read-only over the store. Usage: fetchsplit.py [--store DIR]
"""
import argparse, json, statistics, collections
from multiprocessing import Pool
POS = {f"h{i}": i for i in range(5)}
def ev(d, n):
    """Node n's semindex events for run dir d."""
    return [json.loads(l) for l in open(f"{d}/debug/node_dirs/{n}/semindex/events.jsonl")]
def one(args):
    """(slot_wait_ms, round_trip_ms, outstanding_at_send) per record-hop of one valid run."""
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
            plan = {}
            for e in E[n]:
                if e.get("kind") == "fetch_plan" and e["action"]["neighbor"] == m:
                    for j, (oo, s) in enumerate(e["action"]["fetched"]):
                        if oo == o and s not in plan:
                            plan[s] = (e["ts_ms"], e["state"]["in_flight"] + j + 1)
            learned = {}
            for e in E[n]:
                if e.get("event") == "state_rx" and e["neighbor"] == m:
                    for s in range(1, e["hi"].get(o, 0) + 1):
                        learned.setdefault(s, e["ts_ms"])
            for s, ts in stored.items():
                if s in plan and s in learned and plan[s][0] >= learned[s]:
                    out.append((plan[s][0] - learned[s], ts - plan[s][0], plan[s][1]))
    return out
def bucket(k):
    """Outstanding-count bucket label."""
    return "1" if k == 1 else "2-4" if k <= 4 else "5-8" if k <= 8 else "9+"
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/home/lab_shared/semindex-eval")
    a = ap.parse_args()
    for stage, p, seeds in [("s2b0p5", 0, 20), ("s2b1p359", 0, 20), ("s2b3p695", 0, 20), ("s2b10p043", 0, 20),
                            ("s2a", 0, 100), ("t2k4", 0.4, 3), ("t2c32k4", 0.4, 3)]:
        with Pool(6) as pool:
            rows = [r for rs in pool.map(one, [(a.store, stage, p, s) for s in range(1, seeds + 1)]) for r in rs]
        q = statistics.mean(r[0] for r in rows); t = statistics.mean(r[1] for r in rows)
        by = collections.defaultdict(list)
        for _, rt, k in rows:
            by[bucket(k)].append(rt)
        cells = " | ".join(f"{b}: {statistics.median(by[b]):.0f}ms (n={len(by[b])})" if by[b] else f"{b}: -"
                           for b in ("1", "2-4", "5-8", "9+"))
        print(f"{stage} p{p} n={len(rows)} slot_wait_mean={q:.0f}ms round_trip_mean={t:.0f}ms "
              f"slot_wait_share={q / (q + t):.2f} | round trip median by outstanding at send: {cells}")
