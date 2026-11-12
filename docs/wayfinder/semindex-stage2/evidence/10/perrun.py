"""Per-run live metrics for the s6* runs (8-12 s uniform outages): one CSV row per (stage, p, seed).

The per-run computation is evidence/09/perrun.py's `work`, imported unchanged (it calls evidence/05
attrib.attribute_run with seq=False). The columns are therefore exactly evidence/09 perrun.csv's:
4-hop pair sums from evalagg discovery.csv, per-hop uncensored record-hops and in-window poll bytes,
the run's group length (to rebase names to s4a's 14.92), the sanity counters, and polls.csv counts
(8 directed pairs summed, window [t0 + 10 s, t0 + 130 s]).

Only main() is new: 09's main writes into evidence/09, so the polls.csv fold is repeated here with the
same column names (ana.per_seed_cols reads them) and the output goes to this directory.

Environment: AGG (required, evalagg output with <stage>/discovery.csv and polls.csv), STORE (default
/home/lab_shared/semindex-eval). Needs PYTHONPATH=<repo>/src:../05:../07 as evidence/09 does.
Prints the sanity counters summed per stage to stderr.
"""
import csv
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "09"))
# 09/perrun.py reads AGG from the environment and sys.argv[1] (its stage list) at import; this script
# takes no arguments so that its STAGES default is harmless, and AGG must be set before this line.
import perrun as P9  # noqa: E402
import run_all  # noqa: E402
from semindex import evalagg  # noqa: E402

STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")
AGG = os.environ["AGG"]
STAGES = ["s6short", "s6wait", "s6b0p5", "s6b1p359", "s6b3p695", "s6b10p043"]
POLL_COLS = ("polls_sent", "state_rx", "resched_ans", "resched_to", "clamped", "late", "n_pollrows")


def main():
    jobs = []
    for st in STAGES:
        # DISC must be filled before the fork: workers inherit it instead of re-reading discovery.csv
        run_all.load_disc(st, AGG)
        for r in evalagg.discover_runs(STORE, st):
            jobs.append((st, r.p, r.seed, r.run_dir))
    print("jobs", len(jobs), file=sys.stderr, flush=True)
    with mp.get_context("fork").Pool(3) as pool:
        rows = pool.map(P9.work, jobs, chunksize=2)
    rows.sort(key=lambda r: (r["stage"], r["p"], r["seed"]))
    for r in rows:
        r.update(dict.fromkeys(POLL_COLS, 0))
    idx = {(r["stage"], r["p"], r["seed"]): r for r in rows}
    for st in STAGES:
        with open(f"{AGG}/{st}/polls.csv") as f:
            for q in csv.DictReader(f):
                r = idx.get((st, float(q["p"]), int(q["seed"])))
                if r is None:
                    continue
                r["polls_sent"] += int(q["polls_sent"])
                r["state_rx"] += int(q["state_rx"])
                r["resched_ans"] += int(q["reschedule_answered"])
                r["resched_to"] += int(q["reschedule_timed_out"])
                r["clamped"] += int(q["reschedule_clamped"])
                r["late"] += int(q["late_replies"])
                r["n_pollrows"] += 1
    with open(f"{HERE}/perrun.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote perrun.csv", len(rows), file=sys.stderr)
    # sanity counters per stage (exact_maxerr is a max, the rest are counts)
    for st in STAGES:
        rs = [r for r in rows if r["stage"] == st]
        parts = [f"{c}={max(r[c] for r in rs) if c == 'exact_maxerr' else sum(r[c] for r in rs)}" for c in P9.CHK]
        print(st, f"runs={len(rs)}", f"n_pollrows={sum(r['n_pollrows'] for r in rs)}", *parts, file=sys.stderr)


if __name__ == "__main__":
    main()
