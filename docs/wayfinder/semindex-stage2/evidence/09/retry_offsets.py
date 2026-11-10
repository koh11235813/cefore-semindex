"""For retries sent within 0.5 s after link_up that timed out: ms since the nominal link_up (t0 + at)."""
import glob
import retry_recount
import collections, json
from semindex import evalagg
from semindex.analyze import discover_node_events
from semindex.check import read_results
for stage in ("s5short", "s5wait"):
    for rd in sorted(glob.glob(f"/home/lab_shared/semindex-eval/{stage}/p0.4/*")):
        nodes = discover_node_events(rd)
        t0 = evalagg.derive_t0_ms(nodes, read_results(rd)).t0_ms
        exp = json.load(open(glob.glob(f"{rd}/*.experiment.json")[0]))
        ups = collections.defaultdict(list)
        for e in exp["events"]:
            if e["type"] == "link_up":
                ups[min(e["nodes"])].append(t0 + e["at"] * 1000.0)
        for n, evs in nodes.items():
            polls = collections.defaultdict(list); resch = {}
            for e in evs:
                if e.get("event") != "decision": continue
                if e["kind"] == "state_poll": polls[e["action"]["neighbor"]].append(e["ts_ms"])
                elif e["kind"] == "state_poll_reschedule": resch[(e["action"]["neighbor"], e["action"]["poll_epoch_ms"])] = e["action"]["trigger"]
            for m, ts in polls.items():
                ts.sort(); link = min(retry_recount.POS[n], retry_recount.POS[m])
                for i in range(1, len(ts)):
                    if resch.get((m, ts[i-1])) == "timed_out" and resch.get((m, ts[i])) == "timed_out":
                        for u in ups[link]:
                            if 0 <= ts[i] - u < 500 and t0 + 10000 <= ts[i] <= t0 + 130000:
                                print(stage, rd.split("/")[-1], n, "->", m, f"{ts[i]-u:.1f} ms after nominal link_up")
