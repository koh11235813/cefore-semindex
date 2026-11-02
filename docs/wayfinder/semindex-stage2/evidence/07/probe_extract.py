"""Extract, per valid run, h2's probe blocks, every node's state-poll schedule, and every record-hop.

Rerun (from the worktree root, ~1-2 min, read-only over the run store):
  PYTHONPATH=src /home/lab_shared/cefore-semindex/.venv/bin/python3 docs/wayfinder/semindex-stage2/evidence/07/probe_extract.py \\
    --store /home/lab_shared/semindex-eval --out <scratch>/extract.pkl.gz > probe_extract.txt

Output (--out, ~13 MB gzip pickle; keep it outside the repo, probe_analyze*.py read it):
  list of per-run dicts: stage, p, seed, ws, we (window ms), blocks, polls, hops, links, chk.
stdout: run / probe totals and the h2 events stamped inside a block (probe_extract.txt).
Stage names are fixed to the NORMAL re-collection (s4*).

Definitions (mirroring docs/wayfinder/semindex-stage2/evidence/05/attrib.py):
  - block = [probe_discovered.ts, next probe_retrieved.ts] for a probe with >=1 match. Both are
    EventWriter write-time stamps (agent.py _run_probe), so the interval brackets the
    transport.fetch_content call only (not the local query, not the post-return tail).
  - record-hop (o, s, h): m = hop h-1 node, n = hop h node along the line from origin o;
    arrival = record_created at o (h=1) or first record_stored at m; end = first record_stored
    at n, censored when missing or > window end. A hop whose upstream is censored is dropped
    (counted), as attrib.py does. Only records created inside the window are used.
  - gap governing a hop = n's state_poll toward m with the latest ts strictly before arrival
    (attrib.py's bisect_left(pts, arrival) - 1 rule).
"""
from __future__ import annotations

import argparse
import bisect
import glob
import gzip
import json
import multiprocessing as mp
import os
import pickle
from collections import Counter

from semindex import evalagg, evalgen
from semindex.analyze import discover_node_events
from semindex.check import read_results

STAGES = ["s4c", "s4b0p5", "s4b1p359", "s4b3p695", "s4b10p043", "s4a"]
# The stage-1 default policy logs the jittered interval itself as interval_s; label its arm 2.0.
# (attrib.py keys this on the literal "s2c"; this script names its own stage-1 stage.)
STAGE1 = {"s4c"}
POS = {f"h{i}": i for i in range(5)}
NODE = {i: f"h{i}" for i in range(5)}


def extract(args):
    """One run -> compact dict; None-free except censored hop ends (stored as None)."""
    stage, p, seed, run_dir = args
    meta = json.load(open(f"{run_dir}/meta.json"))
    chk = Counter()
    if not meta["validity"]["valid"]:
        return dict(stage=stage, p=p, seed=seed, valid=False)
    nodes = discover_node_events(run_dir)
    t0 = evalagg.derive_t0_ms(nodes, read_results(run_dir))
    ws = t0.t0_ms + evalgen.WINDOW_START_S * 1000.0
    we = t0.t0_ms + evalgen.WINDOW_END_S * 1000.0
    for n, evs in nodes.items():
        if sum(1 for e in evs if e.get("event") == "agent_start") > 1:
            chk["multi_life_nodes"] += 1

    # ---- h2 probe blocks -------------------------------------------------------------
    ev2 = nodes["h2"]
    blocks = []
    for i, e in enumerate(ev2):
        if e.get("event") != "probe_discovered":
            continue
        chk["probes"] += 1
        if not e["matches"]:
            chk["probes_nomatch"] += 1
            continue
        # The retrieved line is the next probe_* line (check.py relies on this adjacency).
        r = None
        for x in ev2[i + 1:]:
            if x.get("event") == "probe_retrieved":
                r = x
                break
            if x.get("event") == "probe_discovered":
                break
        if r is None:
            chk["probes_unmatched"] += 1
            continue
        blocks.append((e["ts_ms"], r["ts_ms"], bool(r["success"]), r["bytes"], r["origin"]))
    blocks.sort()

    # ---- state polls of every node toward every neighbor -------------------------------
    polls = {}
    created, stored = {}, {}
    inblock = Counter()
    bs = [b[0] for b in blocks]
    for n, evs in nodes.items():
        for e in evs:
            k = e.get("event")
            if k == "record_created":
                created[(e["origin"], e["seq"])] = e["ts_ms"]
            elif k == "record_stored":
                key = (e["origin"], e["seq"], n)
                if key not in stored or e["ts_ms"] < stored[key]:
                    stored[key] = e["ts_ms"]
            elif k == "decision" and e["kind"] == "state_poll":
                a = e["action"]
                arm = 2.0 if stage in STAGE1 else a["interval_s"]
                polls.setdefault((n, a["neighbor"]), []).append(
                    (e["ts_ms"], arm, a.get("applied_interval_s", a["interval_s"])))
            # Empirical check of what h2 still does while inside fetch_content.
            if n == "h2" and k not in ("probe_discovered", "probe_retrieved"):
                j = bisect.bisect_right(bs, e["ts_ms"]) - 1
                if j >= 0 and blocks[j][0] < e["ts_ms"] < blocks[j][1]:
                    inblock[(k, e.get("kind"))] += 1
    for v in polls.values():
        v.sort()

    # ---- record-hops ------------------------------------------------------------------
    hops = []  # (o, s, h, m, n, arrival, end_or_None, created, gov_poll_idx)
    for (o, s), c in created.items():
        if not (ws <= c <= we):
            continue
        po = POS[o]
        d = 1 if po == 0 else -1
        for h in range(1, 5):
            n, m = NODE[po + d * h], NODE[po + d * (h - 1)]
            arrival = c if m == o else stored.get((o, s, m))
            if arrival is None or arrival > we:
                chk["hop_upstream_censored"] += 1
                break
            sn = stored.get((o, s, n))
            end = sn if (sn is not None and sn <= we) else None
            pts = polls.get((n, m), [])
            gi = bisect.bisect_left(pts, (arrival,)) - 1
            hops.append((o, s, h, m, n, arrival, end, c, gi))
            if end is None:
                break
    # ---- link down intervals (the disconnect schedule, `at` is seconds after t0) ----------
    # p is a per-link, per-5s-slot down probability, so a failed probe body fetch and a slow
    # record-hop share a cause (a down link); analyses condition on this schedule.
    exp = json.load(open(glob.glob(f"{run_dir}/*.experiment.json")[0]))
    down, links = {}, []
    for e in sorted(exp["events"], key=lambda e: e["at"]):
        if e["type"] not in ("link_down", "link_up"):
            continue
        key = tuple(sorted(e["nodes"]))
        at = t0.t0_ms + e["at"] * 1000.0
        if e["type"] == "link_down":
            down[key] = at
        elif key in down:
            links.append((key, down.pop(key), at))
    for key, at in down.items():
        links.append((key, at, float("inf")))
    return dict(stage=stage, p=p, seed=seed, valid=True, ws=ws, we=we, blocks=blocks,
                polls=polls, hops=hops, links=links, chk=dict(chk), inblock=dict(inblock))


def main():
    """Extract every run of the s4 stages in parallel, write the pickle, print the totals."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/home/lab_shared/semindex-eval")
    ap.add_argument("--out", required=True, help="pickle path (.pkl.gz); not committed, ~13 MB")
    a = ap.parse_args()
    jobs = []
    for stage in STAGES:
        for rd in sorted(glob.glob(f"{a.store}/{stage}/p*/*/")):
            rd = rd.rstrip("/")
            p = float(rd.split("/")[-2][1:])
            jobs.append((stage, p, int(rd.split("/")[-1]), rd))
    with mp.Pool(6) as pool:
        out = pool.map(extract, jobs, chunksize=4)
    with gzip.open(a.out, "wb") as f:
        pickle.dump(out, f)
    tot = Counter()
    ib = Counter()
    for r in out:
        tot["runs"] += 1
        tot["valid"] += int(r["valid"])
        for k, v in r.get("chk", {}).items():
            tot[k] += v
        for k, v in r.get("inblock", {}).items():
            ib[k] += v
    print("totals:", dict(tot))
    print("h2 events stamped strictly inside a block (event, kind):", dict(ib))


if __name__ == "__main__":
    main()
