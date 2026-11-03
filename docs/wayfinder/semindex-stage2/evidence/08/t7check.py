"""Sanity checks for the t7 policy trial runs (re-decision on answer/timeout).

Per stage and p, over every (node, neighbor) of every run:
- how many state_poll rows got 0 / 1 / 2+ state_poll_reschedule rows (expect 1, except the last poll of a life)
- triggers, clamps, late replies
- the table value and applied interval chosen after each trigger
- gap from a resolved poll's send to the next send, minus the re-decided applied interval
  (the loop's own delay; expect tens of ms)
Reads only events.jsonl under each run's debug/node_dirs/<host>/semindex/.
Usage: python3 -I t7check.py <store> <stage> [<stage> ...]
"""

import collections
import glob
import json
import os
import statistics
import sys


def load(path):
    """Every JSON line of one events.jsonl."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def quantiles(xs):
    """n, min, p50, p90, p99 and max of `xs` as one line."""
    if not xs:
        return "n=0"
    xs = sorted(xs)

    def q(f):
        return xs[min(len(xs) - 1, int(f * len(xs)))]

    return f"n={len(xs)} min={xs[0]:.3f} p50={q(0.5):.3f} p90={q(0.9):.3f} p99={q(0.99):.3f} max={xs[-1]:.3f}"


def main(store, stages):
    """Print the checks for every p of every stage (see the module docstring)."""
    for stage in stages:
        for pdir in sorted(glob.glob(os.path.join(store, stage, "p*"))):
            res = collections.Counter()
            chosen = collections.defaultdict(list)  # trigger -> (interval_s, applied)
            excess = collections.defaultdict(list)  # trigger -> gap - applied (s)
            runs = 0
            for run in sorted(glob.glob(os.path.join(pdir, "*"))):
                if not os.path.exists(os.path.join(run, "DONE")):
                    continue
                runs += 1
                for ev_path in sorted(glob.glob(os.path.join(run, "debug/node_dirs/h*/semindex/events.jsonl"))):
                    events = load(ev_path)
                    polls = collections.defaultdict(list)  # neighbor -> [ts_ms]
                    resch = collections.defaultdict(dict)  # neighbor -> epoch -> [rows]
                    for e in events:
                        if e.get("event") == "decision" and e.get("kind") == "state_poll":
                            polls[e["action"]["neighbor"]].append(e["ts_ms"])
                        elif e.get("event") == "decision" and e.get("kind") == "state_poll_reschedule":
                            a = e["action"]
                            resch[a["neighbor"]].setdefault(a["poll_epoch_ms"], []).append(a)
                        elif e.get("event") == "state_poll_late_reply":
                            res["late_reply"] += 1
                    for nb, sends in polls.items():
                        for i, ts in enumerate(sends):
                            rows = resch[nb].get(ts, [])
                            res[f"resolved_{min(len(rows), 2)}"] += 1
                            if len(rows) == 0 and i != len(sends) - 1:
                                res["unresolved_not_last"] += 1
                            for a in rows:
                                res[a["trigger"]] += 1
                                res["clamped"] += a["clamped"]
                                chosen[a["trigger"]].append((a["interval_s"], round(a["applied_interval_s"], 3)))
                                if i + 1 < len(sends):
                                    gap = (sends[i + 1] - ts) / 1000.0
                                    excess[a["trigger"]].append(gap - a["applied_interval_s"])
                        orphan = set(resch[nb]) - set(sends)
                        res["reschedule_without_poll"] += len(orphan)
            print(f"== {stage} {os.path.basename(pdir)} runs={runs}")
            print("   ", dict(sorted(res.items())))
            for trig, vals in sorted(chosen.items()):
                iv = collections.Counter(v[0] for v in vals)
                ap = [v[1] for v in vals]
                print(f"    {trig}: table={dict(iv)} applied: {quantiles(ap)}")
            for trig, xs in sorted(excess.items()):
                print(f"    {trig}: gap-applied (s): {quantiles(xs)} mean={statistics.mean(xs):.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
