"""Hand check of the byte accounting on s5short p0.2 seed 1: rebuild every in-window state name from the raw events."""
import json
import os
from semindex import evalagg, names
from semindex.analyze import discover_node_events
from semindex.check import read_results
rd = os.environ.get("STORE", "/home/lab_shared/semindex-eval") + "/s5short/p0.2/1"
nodes = discover_node_events(rd)
t0 = evalagg.derive_t0_ms(nodes, read_results(rd)).t0_ms
ws, we = t0 + 10000, t0 + 130000
group = json.load(open(f"{rd}/s5short_p020_seed1/h0.json"))["group"]
L, npoll, lens = 0, 0, set()
for n, evs in nodes.items():
    for e in evs:
        if e.get("event") == "decision" and e["kind"] == "state_poll" and ws <= e["ts_ms"] <= we:
            s = names.build_state_name(group, e["action"]["neighbor"], n, int(e["ts_ms"]))
            L += len(s); npoll += 1; lens.add(len(s))
print("group", group, len(group), "in-window polls", npoll, "name bytes", L, "distinct name lengths", lens)
print("example", s)
print("s4a equivalent", names.build_state_name("s4a_p020_seed1", "h0", "h1", int(t0)), len(names.build_state_name("s4a_p020_seed1", "h0", "h1", int(t0))))
