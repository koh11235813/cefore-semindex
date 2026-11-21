"""Per-run live metrics for the s9* runs of spec.md: one CSV row per (stage, p, seed).

The all-record half of each row is evidence/09/perrun.py's `work`, imported unchanged exactly as
evidence/10, 16 and 17 import it, so the columns evidence/16 and 17 carry are here under the same
names and one loader reads all of them: 4-hop pair sums from evalagg discovery.csv, per-hop
uncensored record-hops and in-window poll bytes, the run's group length, the sanity counters, and
polls.csv counts (8 directed pairs summed, window [t0 + 10 s, t0 + 130 s]).

What this step adds, because spec.md scores these loads twice:

    hn4, hlat4, hc4    the 4-hop sums over the HEADLINE records only, which spec.md fixes as the
                       records whose scheduled creation time is at or before t0+72.0 s. The
                       selection is the experiment description's `schedule[host][seq-1]["at_s"]
                       <= 72.0`, read against discovery.csv's `(origin, seq, node)` rows, which is
                       spec.md's own wording of the cut. The selector is the scheduled time, so that
                       the testbed and the replay put the same records on each side of the boundary;
                       a record is created a median 91 ms after it was due, and a selection on the
                       logged time would move the boundary by that lag on one side only.
    dn4, dlat4, dc4    the same 4-hop sums with no cut, computed from the same discovery rows. They
                       exist to be compared against `n4`, `lat4` and `c4`, which attrib builds from
                       its own censoring chain. A run where the two disagree would have the two
                       scorings of spec.md resting on two different censoring rules, so the count of
                       such runs is printed on stderr and belongs in the analysis.
    sched_all          how many records the schedule lays down in this run, both origins together.
    sched_head         how many of them have `at_s` at or before 72.0 s.
    sched_regen_diff   how many of the run's two record origins hold instants that the committed
                       generator does not draw again for this run's (host, seed, k, workload).
                       0 means both origins agree instant by instant and in order. spec.md's note
                       under pass line 5 leans on the testbed and the replay having run the same
                       instants, and the replay's instants are that same draw, so this column is
                       what carries that statement run by run.
    head_missing_disc  headline records with no discovery row at all, and
    all_missing_disc   the same over every scheduled record. Both are expected to be zero; a
                       nonzero count means a record the schedule names never appeared in the
                       aggregation, which would deflate that run's counts silently.
    workload           `meta.json` -> `workload`, so the analysis reads the load off the run itself.

A run without a DONE marker is left out of the jobs, because evalagg.discover_runs returns every
seed directory under a stage whether or not the run inside it finished, an unfinished one has no
meta.json yet, and `work` would die on it. The per-stage line on stderr says how many directories
exist and how many of them are DONE, so a partial stage is visible in the output of the run itself.

Environment: AGG (required, evalagg output with <stage>/discovery.csv and polls.csv), STORE
(default /home/lab_shared/semindex-eval). Needs PYTHONPATH=<repo>/src:../05:../07 and must be run
from this directory, as evidence/09, 10, 16 and 17 are.
"""
import csv
import glob
import json
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
from semindex import evalagg, evalgen  # noqa: E402

STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")
AGG = os.environ["AGG"]
# The three gate stages first, then the five stages of each load, which is the order main9.sh ran
# them in and the order the analysis reports them in.
STAGES = ["s9w0_b3p695", "s9w0_b10p043", "s9w0_t37r",
          "s9wo_b3p695", "s9wo_b6p0", "s9wo_b10p043", "s9wo_t37", "s9wo_t37r",
          "s9wv_b3p695", "s9wv_b6p0", "s9wv_b10p043", "s9wv_t37", "s9wv_t37r"]
POLL_COLS = ("polls_sent", "state_rx", "resched_ans", "resched_to", "clamped", "late", "n_pollrows")
CUT_S = 72.0
# The two record origins and the node 4 hops away from each. The line is h0 .. h4, so a record of
# h0 is 4 hops away once it is stored at h4 and the other way round. 4 hops is the distance every
# delay in spec.md is measured at.
FAR = {"h0": "h4", "h4": "h0"}
HEAD_COLS = ("hn4", "hlat4", "hc4", "dn4", "dlat4", "dc4", "sched_all", "sched_head",
             "sched_regen_diff", "head_missing_disc", "all_missing_disc")


def schedule_at_s(run_dir):
    """{(origin, seq): at_s} of one run, from its experiment description.

    The description is the file the testbed was handed, so reading the cut out of it is what makes
    the headline set of a measured run and the headline set of the replay the same set of records:
    predict.py read `schedule[host][i]["at_s"]` out of this same file for its own prediction.
    `seq` runs from 1 per origin in schedule order, which is the numbering the agents log.
    """
    hits = glob.glob(os.path.join(run_dir, "*.experiment.json"))
    if len(hits) != 1:
        raise RuntimeError(f"{run_dir}: expected one *.experiment.json, found {hits}")
    with open(hits[0]) as f:
        sched = json.load(f)["schedule"]
    return {(f"h{host}", i): float(e["at_s"])
            for host, entries in sched.items()
            for i, e in enumerate(entries, 1)}


def regen_diff(at_s, seed, k, workload):
    """How many of the two record origins hold instants the committed generator does not redraw.

    The left hand side is the run's own experiment description, the file the testbed was handed,
    read by `schedule_at_s`. The right hand side is a fresh call of the generator's schedule
    builder for the same (host, seed, k, workload). That builder is where
    `evalgen.build_experiment` gets the `schedule` block of every description it writes, and
    predict.py's replay instants come from descriptions written the same way, so the two sides
    being equal is what says the testbed and the replay ran the same instants. The comparison is
    on the whole ordered list, so a repeated or dropped tick and the k records sharing one tick are
    all covered.

    A run with a nonzero count would put the prediction of pass line 5 and the measurement on two
    different loads, which is why the count sits in every row and its total is printed on stderr.
    """
    bad = 0
    for host in evalgen.RECORD_ORIGIN_HOSTS:
        stored = [t for key, t in sorted(at_s.items()) if key[0] == f"h{host}"]
        drawn = [float(e["at_s"]) for e in evalgen._record_schedule(host, seed, k, workload)]
        if stored != drawn:
            bad += 1
    return bad


def four_hop_sums(disc, at_s, cut):
    """(n, lat ms, c, missing) of the 4-hop pairs whose scheduled creation is at or before `cut`.

    `disc` is the run's {(origin, seq, node): latency ms or None} from discovery.csv, where None
    means the record was not stored at that node inside the window. `cut` of None takes every
    scheduled record. A record with no discovery row is counted in `missing` and left out of the
    sums, so a missing row cannot pass as a delivered record or as a censored one.
    """
    n = c = missing = 0
    lat = 0.0
    for (origin, seq), t in at_s.items():
        if cut is not None and t > cut:
            continue
        if (origin, seq, FAR[origin]) not in disc:
            missing += 1
            continue
        dl = disc[(origin, seq, FAR[origin])]
        if dl is None:
            c += 1
        else:
            n += 1
            lat += dl
    return n, lat, c, missing


def work(job):
    """One run's row: evidence/09's columns, then this step's headline and schedule columns."""
    stage, p, seed, run_dir = job
    row = P9.work(job)
    disc = run_all.DISC.get((stage, p, seed)) or {}
    at_s = schedule_at_s(run_dir)
    hn, hlat, hc, hmiss = four_hop_sums(disc, at_s, CUT_S)
    dn, dlat, dc, dmiss = four_hop_sums(disc, at_s, None)
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    row["workload"] = meta["workload"]
    # k and the load come off the run's own meta.json, so the redraw is asked for the settings this
    # run was actually generated with and not for the sweep's nominal ones.
    row.update(hn4=hn, hlat4=hlat, hc4=hc, dn4=dn, dlat4=dlat, dc4=dc,
               sched_all=len(at_s), sched_head=sum(1 for t in at_s.values() if t <= CUT_S),
               sched_regen_diff=regen_diff(at_s, seed, int(meta["k"]), meta["workload"]),
               head_missing_disc=hmiss, all_missing_disc=dmiss)
    return row


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
        rows = pool.map(work, jobs, chunksize=2)
    rows.sort(key=lambda r: (STAGES.index(r["stage"]), r["p"], r["seed"]))
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
    # and reporting that belongs to line 2 of the analysis.
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
    # Whether the two scorings rest on the same censoring rule. `n4`, `lat4` and `c4` come from
    # attrib's own chain through discovery.csv, `dn4`, `dlat4` and `dc4` from the discovery rows of
    # every scheduled record. A run where they differ is reported here by name, because the
    # headline sums are built the second way and the all-record sums the first.
    diff = [(r["stage"], r["seed"]) for r in rows
            if (r["dn4"], round(r["dlat4"], 3), r["dc4"]) != (r["n4"], round(r["lat4"], 3), r["c4"])]
    print("runs where the discovery 4-hop sums differ from attrib's:", len(diff), diff[:10],
          file=sys.stderr)
    for c in ("sched_regen_diff", "head_missing_disc", "all_missing_disc"):
        print(f"{c} over all runs:", sum(r[c] for r in rows), file=sys.stderr)


if __name__ == "__main__":
    main()
