"""Three checks of the s6 numbers written separately from perrun.py / ana.py / retry.py / replay_*.py.

1. 4-hop delay and censored share straight from evalagg's discovery.csv (rows with hop_distance 4, pooled
   over seeds and both directions), against live_main.csv's `delay` and `cens4`, for every (stage, p) there.
2. Retries and their down share with an own pairing from the raw `decision` events: a retry is the first
   state_poll to a neighbour after a state_poll_reschedule of that neighbour whose trigger is timed_out and whose
   poll_epoch_ms is the latest state_poll to it; it is "down" if the experiment's link between the two nodes is
   down at its send time (t0 from evalagg). Against retry_table.csv's retry_per_run and down_share, for every
   policy (stage, p) there. Its timed-out share matches the reschedule whose poll_epoch_ms is the retry's ts_ms;
   retries without one are counted apart. 7 cells count 1-4 retries more (of 491-1980) than retry_table.csv,
   whose extraction is evidence/09 phase.retries_of: it takes the last reschedule at or before the poll's own ms
   (bisect_right), so a retry whose reply came back within the same ms is matched to its own `answered`
   reschedule and dropped (s5short p 0.4 seed 16, h2->h1 at 1793853807390). The down share moves by <= 0.0014.
3. replay.simulate called directly (not through replay_common.py) for s6wait (SlotBlind(10.043, 5.0)) and the
   retuned wait-form pick (SlotBlind(10.043, 2.75)) at p 0.4 on seeds 1-20, 10 reps each with the paired
   seeding, against replay_retune.csv's test_delay of the `best` and `cap` rows.

Usage: AGG=<evalagg output dir> PYTHONPATH=<repo>/src python3 recheck.py > recheck.txt
"""
import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict

from semindex import evalagg
from semindex.analyze import discover_node_events
from semindex.check import read_results

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "09"))
import replay as R  # noqa: E402

AGG = os.environ["AGG"]
STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")


def check_discovery():
    print("## 1. 4-hop delay and cens4 from discovery.csv vs live_main.csv")
    live = {(r["stage"], r["p"]): r for r in csv.DictReader(open(f"{HERE}/live_main.csv"))}
    stages = sorted({s for s, _ in live})
    bad = 0
    for st in stages:
        acc = defaultdict(lambda: [0.0, 0, 0])
        with open(f"{AGG}/{st}/discovery.csv", newline="") as f:
            for r in csv.DictReader(f):
                if r["hop_distance"] != "4":
                    continue
                a = acc[r["p"]]
                if r["censored"] == "false":
                    a[0] += float(r["latency_ms"])
                    a[1] += 1
                else:
                    a[2] += 1
        for p, (s, n, c) in sorted(acc.items()):
            L = live[(st, p)]
            d, cz = s / n, c / (n + c)
            ok = abs(d - float(L["delay"])) < 1e-6 and abs(cz - float(L["cens4"])) < 1e-12
            bad += not ok
            print(f"{st} p {p}: delay {d:.1f} (live_main {float(L['delay']):.1f}) cens4 {100 * cz:.2f} % "
                  f"(live_main {100 * float(L['cens4']):.2f} %) {'ok' if ok else 'MISMATCH'}")
    print(f"cells {sum(1 for _ in live)}, mismatches {bad}\n")


def retries(run):
    """(n retries in window, n sent while down, n timed out, n without an outcome) for one run."""
    t0 = evalagg.derive_t0_ms(discover_node_events(run), read_results(run)).t0_ms
    exp = json.load(open(glob.glob(f"{run}/*.experiment.json")[0]))
    down = defaultdict(list)
    for e in sorted((e for e in exp["events"] if e["type"] in ("link_down", "link_up")), key=lambda e: e["at"]):
        k = frozenset(e["nodes"])
        if e["type"] == "link_down":
            down[k].append([e["at"], 1e9])
        else:
            down[k][-1][1] = e["at"]
    n = nd = nto = nunk = 0
    for f in glob.glob(f"{run}/debug/node_dirs/h*/semindex/events.jsonl"):
        me = int(f.split("/node_dirs/h")[1].split("/")[0])
        ev = [json.loads(line) for line in open(f)]
        ev = sorted((e for e in ev if e.get("event") == "decision" and e.get("kind") in ("state_poll", "state_poll_reschedule")),
                    key=lambda e: e["ts_ms"])
        outcome = {(e["action"]["neighbor"], e["action"]["poll_epoch_ms"]): e["action"]["trigger"]
                   for e in ev if e["kind"] == "state_poll_reschedule"}
        pending, last = {}, {}
        for e in ev:
            nb = e["action"]["neighbor"]
            if e["kind"] == "state_poll_reschedule":
                # only a timeout of the latest poll to this neighbour makes the next poll a retry
                pending[nb] = e["action"]["trigger"] == "timed_out" and e["action"]["poll_epoch_ms"] == last.get(nb)
                continue
            last[nb] = e["ts_ms"]
            if pending.pop(nb, False):
                t = (e["ts_ms"] - t0) / 1000
                if not 10 <= t <= 130:
                    continue
                n += 1
                nd += any(a <= t < b for a, b in down[frozenset((me, int(nb[1:])))])
                o = outcome.get((nb, e["ts_ms"]))
                nto += o == "timed_out"
                nunk += o is None
    return n, nd, nto, nunk


def check_retries():
    print("## 2. Retries and down share from raw decision events vs retry_table.csv")
    tab = {(r["stage"], r["p"]): r for r in csv.DictReader(open(f"{HERE}/retry_table.csv"))}
    bad = 0
    for (st, p), r in sorted(tab.items()):
        if not st.endswith(("short", "wait")):
            continue
        tot = [0, 0, 0, 0]
        for run in sorted(glob.glob(f"{STORE}/{st}/p{p}/*/")):
            tot = [a + b for a, b in zip(tot, retries(run.rstrip("/")))]
        nruns = len(glob.glob(f"{STORE}/{st}/p{p}/*/"))
        per, ds = tot[0] / nruns, tot[1] / tot[0]
        ok = abs(per - float(r["retry_per_run"])) < 1e-9 and abs(ds - float(r["down_share"])) < 1e-9
        bad += not ok
        print(f"{st} p {p}: retries/run {per:.2f} (table {float(r['retry_per_run']):.2f}) down {ds:.4f} "
              f"(table {float(r['down_share']):.4f}) timed out {tot[2] / tot[0]:.4f} + no outcome {tot[3] / tot[0]:.4f} "
              f"(table timed out {float(r['timed_out_share']):.4f}) {'ok' if ok else 'differs'}")
    print(f"differing cells {bad} (same-ms reply quirk of phase.retries_of, see the docstring)\n")


def check_replay():
    print("## 3. replay.simulate called directly: wait-form at p 0.4, seeds 1-20, 10 paired reps")
    p = 0.4
    means = {}
    for w in (5.0, 2.75):
        vals = []
        for s in range(1, 21):
            links, rec = R.load_schedule(p, s, "s6b0p5")
            links = [R.Link(iv) for iv in links]
            ds = []
            for rep in range(10):
                rng = random.Random(hash((10.043, w) + (p, s, rep)) & 0xffffffff)
                d = R.simulate(R.SlotBlind(10.043, wait=w), links, rec, p, rng)[0]
                if d == d:
                    ds.append(d * 1000)
            vals.append(sum(ds) / len(ds))
        means[w] = sum(vals) / len(vals)
    rows = {r["pick"]: r for r in csv.DictReader(open(f"{HERE}/replay_retune.csv")) if r["family"] == "wait" and r["p"] == "0.4"}
    print(f"SlotBlind(10.043, 5.0): {means[5.0]:.1f} ms (replay_retune.csv cap row {float(rows['cap']['test_delay']):.1f})")
    print(f"SlotBlind(10.043, 2.75): {means[2.75]:.1f} ms (replay_retune.csv best row {float(rows['best']['test_delay']):.1f})")
    print(f"delta {means[2.75] - means[5.0]:.1f} ms (replay_retune.csv ddelay {float(rows['best']['ddelay']):.1f})")


if __name__ == "__main__":
    check_discovery()
    check_retries()
    check_replay()
