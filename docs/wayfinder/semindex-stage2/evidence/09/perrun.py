"""Per-run live metrics for the semindex stage-2 policy runs (s5*): one CSV row per (stage, p, seed).

Reuses evidence/05 attrib.attribute_run unchanged (seq=False: none of the sequential pieces are reported).
Per run: 4-hop pair sums (latencies are evalagg discovery.csv values, as in seq_model.direct), per-hop
uncensored record-hops and in-window poll bytes pooled over the provisional "arm" labels, the
counts needed to rebase names to s4a's group length, and polls.csv counts.
"""
import csv
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict

import attrib
import run_all
from semindex import evalagg

# The evalagg output (<AGG>/<stage>/discovery.csv, polls.csv) is ~47 MB and rebuilt from the store, so it is
# not kept here; the README gives the evalagg command.
STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")
AGG = os.environ["AGG"]
OUT = os.path.dirname(os.path.abspath(__file__))
STAGES = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "s5short", "s5wait", "s5b0p5", "s5b1p359", "s5b3p695", "s5b10p043", "s5c"]
CHK = ("exact_n", "exact_within1", "exact_maxerr", "exact_censor_disagree", "exact_missing_in_discovery",
       "runs_without_discovery_rows", "stored_without_learn", "hop_no_gap_before", "hop_gap_outside_window",
       "reply_before_any_poll", "neg_fetch")


def work(job):
    stage, p, seed, run_dir = job
    res = attrib.attribute_run(stage, p, seed, run_dir, run_all.DISC.get((stage, p, seed)), c_stage=(stage == "s5c"))
    group = json.load(open(f"{attrib.run_name_dir(run_dir)}/h0.json"))["group"]
    row = dict(stage=stage, p=p, seed=seed, group_len=len(group))
    pr = res["pair"]
    for H in (1, 2, 3, 4):
        row[f"n{H}"] = pr.get(("n", H), 0)
        row[f"lat{H}"] = pr.get(("lat", H), 0.0)
        row[f"c{H}"] = pr.get(("c", H), 0)
    # hop key layout (attrib.stratum): 0 stage, 1 p, 2 arm, 3 hop, ...; pool over arm and the state strata
    hu = defaultdict(lambda: defaultdict(float))
    for k, v in res["hop"].items():
        for f, x in v.items():
            hu[k[3]][f] += x
    du = defaultdict(lambda: defaultdict(float))
    for k, v in res["dec"].items():
        for f, x in v.items():
            du[k[3]][f] += x
    for h in (1, 2, 3, 4):
        row[f"nu{h}"] = hu[h].get("n_u", 0)
        row[f"nc{h}"] = hu[h].get("n_c", 0)
        row[f"intB{h}"] = du[h].get("int_B", 0)
        row[f"repB{h}"] = du[h].get("rep_B", 0)
        row[f"ngap{h}"] = du[h].get("n_gap", 0)
        row[f"nrep{h}"] = du[h].get("n_rep", 0)
        row[f"nnorep{h}"] = du[h].get("n_norep", 0)
    for c in CHK:
        row[c] = res["chk"].get(c, 0)
    return row


def main():
    jobs = []
    for st in STAGES:
        run_all.load_disc(st, AGG)
        for r in evalagg.discover_runs(STORE, st):
            jobs.append((st, r.p, r.seed, r.run_dir))
    print("jobs", len(jobs), file=sys.stderr, flush=True)
    with mp.get_context("fork").Pool(3) as pool:
        rows = pool.map(work, jobs, chunksize=2)
    rows.sort(key=lambda r: (r["stage"], r["p"], r["seed"]))
    # polls.csv per run (8 directed pairs summed)
    for r in rows:
        r.update(dict(polls_sent=0, state_rx=0, resched_ans=0, resched_to=0, clamped=0, late=0, n_pollrows=0))
    idx = {(r["stage"], r["p"], r["seed"]): r for r in rows}
    for st in STAGES:
        with open(f"{AGG}/{st}/polls.csv") as f:
            for q in csv.DictReader(f):
                r = idx.get((st, float(q["p"]), int(q["seed"])))
                if r is None:
                    continue
                r["polls_sent"] += int(q["polls_sent"]); r["state_rx"] += int(q["state_rx"])
                r["resched_ans"] += int(q["reschedule_answered"]); r["resched_to"] += int(q["reschedule_timed_out"])
                r["clamped"] += int(q["reschedule_clamped"]); r["late"] += int(q["late_replies"]); r["n_pollrows"] += 1
    name = "perrun.csv" if len(sys.argv) <= 1 else f"perrun_{sys.argv[1].replace(',', '_')}.csv"
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", name, len(rows), file=sys.stderr)


if __name__ == "__main__":
    main()
