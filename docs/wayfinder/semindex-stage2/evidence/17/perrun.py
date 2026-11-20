"""Per-run live metrics for the s8* runs of spec.md: one CSV row per (stage, p, seed).

The per-run computation is evidence/09/perrun.py's `work`, imported unchanged, exactly as
evidence/10/perrun.py and evidence/16/perrun.py import it, so this directory's perrun.csv has the
same columns as theirs and one loader reads all three: 4-hop pair sums from evalagg discovery.csv,
per-hop uncensored record-hops and in-window poll bytes, the run's group length, the sanity
counters, and polls.csv counts (8 directed pairs summed, window [t0 + 10 s, t0 + 130 s]).

Only main() differs from evidence/16's: the stage list is the five s8 stages. A run without a DONE
marker is left out of the jobs, because evalagg.discover_runs returns every seed directory under a
stage whether or not the run inside it finished, an unfinished one has no meta.json yet, and `work`
would die on it. The per-stage line on stderr says how many directories exist and how many of them
are DONE, so a partial stage is visible in the output of the run itself.

Environment: AGG (required, evalagg output with <stage>/discovery.csv and polls.csv), STORE
(default /home/lab_shared/semindex-eval). Needs PYTHONPATH=<repo>/src:../05:../07 and must be run
from this directory, as evidence/09, 10 and 16 are.
"""
import csv
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "09"))
# 09/perrun.py reads AGG from the environment and sys.argv[1] (its stage list) at import; this
# script takes no arguments so that its STAGES default is harmless, and AGG must be set before
# this line.
import perrun as P9  # noqa: E402
import run_all  # noqa: E402
from semindex import evalagg  # noqa: E402

STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")
AGG = os.environ["AGG"]
STAGES = ["s8t37", "s8t37r", "s8b3p695", "s8b10p043", "s8b6p0"]
POLL_COLS = ("polls_sent", "state_rx", "resched_ans", "resched_to", "clamped", "late", "n_pollrows")


def main():
    jobs = []
    for st in STAGES:
        # DISC must be filled before the fork: workers inherit it instead of re-reading discovery.csv
        run_all.load_disc(st, AGG)
        refs = evalagg.discover_runs(STORE, st)
        done = [r for r in refs if evalagg.run_has_done_marker(r.run_dir)]
        print(st, f"dirs={len(refs)}", f"done={len(done)}", file=sys.stderr, flush=True)
        for r in done:
            jobs.append((st, r.p, r.seed, r.run_dir))
    print("jobs", len(jobs), file=sys.stderr, flush=True)
    with mp.get_context("fork").Pool(3) as pool:
        rows = pool.map(P9.work, jobs, chunksize=2)
    rows.sort(key=lambda r: (r["stage"], r["p"], r["seed"]))
    for r in rows:
        r.update(dict.fromkeys(POLL_COLS, 0))
    idx = {(r["stage"], r["p"], r["seed"]): r for r in rows}
    for st in STAGES:
        path = f"{AGG}/{st}/polls.csv"
        if not os.path.exists(path):
            # A stage whose runs have not started has no evalagg output at all; the stderr line
            # above already reported it as dirs=0, so there is nothing to fold here.
            continue
        with open(path) as f:
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
    # An AGG built before a run finished has no discovery.csv or polls.csv rows for it, and
    # attrib.attribute_run then skips that run's pair sums instead of failing: the run would sit in
    # perrun.csv with lat4 = n4 = polls_sent = 0 and deflate its stage's polls per run while leaving
    # its delay alone. That is a silent corruption of the measurement, so it stops the script here.
    # The missing polls.csv row is not the condition tested: a flagged run has none of those either,
    # and reporting that belongs to line 1 of the analysis.
    stale = [(r["stage"], r["seed"]) for r in rows if r["runs_without_discovery_rows"]]
    if stale:
        print("AGG is older than the store; rebuild it for these runs:", stale, file=sys.stderr)
        raise SystemExit(1)
    with open(f"{HERE}/perrun.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote perrun.csv", len(rows), file=sys.stderr)
    # sanity counters per stage (exact_maxerr is a max, the rest are counts)
    for st in STAGES:
        rs = [r for r in rows if r["stage"] == st]
        if not rs:
            continue
        parts = [f"{c}={max(r[c] for r in rs) if c == 'exact_maxerr' else sum(r[c] for r in rs)}" for c in P9.CHK]
        print(st, f"runs={len(rs)}", f"n_pollrows={sum(r['n_pollrows'] for r in rs)}", *parts, file=sys.stderr)


if __name__ == "__main__":
    main()
