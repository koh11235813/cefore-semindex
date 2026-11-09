"""Independent re-count of three phase claims for s5short / s5wait (written separately from phase.py).

1. Largest number of retry polls (the state_poll after a timed_out reschedule of the same pair),
   over all 8 directed pairs of a run, falling inside any 100 ms span (= within +/-50 ms of each other).
2. Retries sent within 0.5 s after that link's link_up, and how many of those timed out.
3. First state_poll of each node relative to t0 (evalagg.derive_t0_ms).
Window: [t0 + 10 s, t0 + 130 s] for 1 and 2.
"""
import collections
import glob
import json

from semindex import evalagg
from semindex.analyze import discover_node_events
from semindex.check import read_results

STORE = "/home/lab_shared/semindex-eval"
POS = {f"h{i}": i for i in range(5)}


def run(run_dir):
    nodes = discover_node_events(run_dir)
    t0 = evalagg.derive_t0_ms(nodes, read_results(run_dir)).t0_ms
    exp = json.load(open(glob.glob(f"{run_dir}/*.experiment.json")[0]))
    ups = collections.defaultdict(list)
    for e in exp["events"]:
        if e["type"] == "link_up":
            ups[min(int(n) for n in e["nodes"])].append(t0 + e["at"] * 1000.0)
    retries = []  # (ts_ms, link, outcome_timed_out)
    first = []
    for n, evs in nodes.items():
        polls = collections.defaultdict(list)
        resch = {}
        for e in evs:
            if e.get("event") != "decision":
                continue
            if e["kind"] == "state_poll":
                polls[e["action"]["neighbor"]].append(e["ts_ms"])
            elif e["kind"] == "state_poll_reschedule":
                a = e["action"]
                resch[(a["neighbor"], a["poll_epoch_ms"])] = a["trigger"]
        allp = [t for v in polls.values() for t in v]
        if allp:
            first.append(min(allp) - t0)
        for m, ts in polls.items():
            ts.sort()
            link = min(POS[n], POS[m])
            for i in range(1, len(ts)):
                if resch.get((m, ts[i - 1])) == "timed_out":
                    retries.append((ts[i], link, resch.get((m, ts[i]))))
    win = [r for r in retries if t0 + 10000 <= r[0] <= t0 + 130000]
    times = sorted(r[0] for r in win)
    best = 0
    j = 0
    for i in range(len(times)):
        while times[i] - times[j] > 100.0:
            j += 1
        best = max(best, i - j + 1)
    after_up = [r for r in win if any(0 <= r[0] - u < 500.0 for u in ups[r[1]])]
    failed_after_up = sum(1 for r in after_up if r[2] == "timed_out")
    return best, len(win), len(after_up), failed_after_up, first


def main():
    for stage in ("s5short", "s5wait"):
        for p in ("0.2", "0.4"):
            mx, nret, nup, nfail, firsts = 0, 0, 0, 0, []
            for rd in sorted(glob.glob(f"{STORE}/{stage}/p{p}/*")):
                b, n_, u, f, fi = run(rd)
                mx = max(mx, b)
                nret += n_
                nup += u
                nfail += f
                firsts += fi
            firsts.sort()
            print(f"{stage} p{p}: retries {nret}, max retries within 100 ms {mx}, "
                  f"within 0.5 s after link_up {nup} ({nup / max(nret, 1):.3f}), of which timed out {nfail}, "
                  f"first poll - t0 ms min/median/max {firsts[0]:.0f}/{firsts[len(firsts) // 2]:.0f}/{firsts[-1]:.0f}")


if __name__ == "__main__":
    main()
