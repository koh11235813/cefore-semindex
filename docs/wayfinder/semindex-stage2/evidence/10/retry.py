"""Does the retry after a failed state poll land inside the same outage? s6 (8-12 s outages) vs s5 (5 s slots).

Hypothesis tested: after a failure the policies wait 3.695 s (short) or 5 s (wait) from the failed poll's send
time, which is much shorter than an 8-12 s outage, so on s6 the retry is sent while the link is still down and
fails again more often than on s5's 5 s slots.

Definitions (times in ms, relative to t0 = evalagg.derive_t0_ms, window [t0 + 10 s, t0 + 130 s]):
  retry      policy stages: the state_poll that follows a timed_out state_poll_reschedule of the same
             (node, neighbour) whose poll_epoch_ms is the previous poll; fixed arms (no reschedule rows): the poll
             whose state.last_poll_status is timed_out, i.e. the poll sent right after a timed-out poll. Both are
             evidence/09/phase.py's retries_of, imported unchanged. Only retries sent inside the window count.
  outcome    policy: the trigger of the reschedule row naming the poll; fixed arm: last_poll_status of the same
             pair's next poll (its interval, 3.695 s, exceeds the 2 s Interest lifetime).
  down       the link between the two nodes is down at the send time by the nominal schedule in
             *.experiment.json (t0 + at * 1000). Down intervals of one link that touch (next down <= last up)
             are merged into one outage episode; an interval with no link_up stays open to the run's end.
  residual   for a retry sent while down: next link_up - send time (open episodes are counted, not averaged).
  disagree   retries that are down but answered, and retries that are up but timed out.
  episode    one merged outage of a link, started inside the window, seen from one directed pair:
             polls_in   = that pair's polls sent while it is down (0, 1, 2, 3, 4+), the failures the schedule implies;
             streak     = measured maximal runs of consecutive timed-out polls of a pair (first poll in the window),
                          binned 1, 2, 3, 4+. A streak includes the first failure, so it is retries + 1 when the
                          retry chain is not broken.
No 5 s slot or slot phase is used: s6 has no slots, and s5 is described by its own link events the same way.

Environment: STORE (default /home/lab_shared/semindex-eval). Needs PYTHONPATH=<repo>/src (phase.py adds
evidence/04 itself). Writes retry_table.csv here and replaces the "## Retry" section at the end of tables.md
(run ana.py first).
"""
import bisect
import csv
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "09"))
import phase  # noqa: E402
from semindex import evalagg  # noqa: E402
from semindex.analyze import discover_node_events  # noqa: E402
from semindex.check import read_results  # noqa: E402

STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")
JOBS = [("s6short", (0.2, 0.4, 0.5, 0.6)), ("s6wait", (0.2, 0.4, 0.5, 0.6)),
        ("s5short", (0.2, 0.4)), ("s5wait", (0.2, 0.4)),
        ("s6b3p695", (0.2, 0.4, 0.5, 0.6)), ("s5b3p695", (0.2, 0.4))]
SECTION = "## Retry"


def outages(exp, t0):
    """{frozenset(hA, hB): [(down_ms, up_ms or inf)]} from experiment.json, touching intervals merged."""
    ev = defaultdict(list)
    for e in exp["events"]:
        if e["type"] in ("link_down", "link_up"):
            a, b = e["nodes"]
            ev[frozenset((f"h{a}", f"h{b}"))].append((t0 + e["at"] * 1000.0, e["type"]))
    out = {}
    for k, lst in ev.items():
        downs = sorted(t for t, ty in lst if ty == "link_down")
        ups = sorted(t for t, ty in lst if ty == "link_up")
        ivs = []
        for s in downs:
            j = bisect.bisect_right(ups, s)  # an up at the same instant as this down closes the previous interval
            e = ups[j] if j < len(ups) else math.inf
            if ivs and s <= ivs[-1][1]:
                ivs[-1] = (ivs[-1][0], max(ivs[-1][1], e))
            else:
                ivs.append((s, e))
        out[k] = ivs
    return out


def which(ivs, t):
    """The interval of `ivs` containing instant t, or None."""
    for s, e in ivs:
        if s <= t < e:
            return (s, e)
    return None


def analyze(job):
    stage, p, seed, run_dir = job
    nodes = discover_node_events(run_dir)
    t0 = evalagg.derive_t0_ms(nodes, read_results(run_dir)).t0_ms
    exp = json.load(open(glob.glob(f"{run_dir}/*.experiment.json")[0]))
    policy = bool(json.load(open(f"{run_dir}/meta.json")).get("state_poll_policy"))
    polls, resch = defaultdict(list), defaultdict(list)
    for n, evs in nodes.items():
        for e in evs:
            if e.get("event") != "decision":
                continue
            a = e["action"]
            if e["kind"] == "state_poll":
                polls[(n, a["neighbor"])].append((e["ts_ms"], e["state"].get("last_poll_status")))
            elif e["kind"] == "state_poll_reschedule":
                resch[(n, a["neighbor"])].append((e["ts_ms"], a["poll_epoch_ms"], a["trigger"]))
    for v in list(polls.values()) + list(resch.values()):
        v.sort()
    ws, we = t0 + phase.WS, t0 + phase.WE
    links = outages(exp, t0)
    rr = phase.retries_of(polls, resch, policy)
    trig = {(k, ep): tr for k, rs in resch.items() for _, ep, tr in rs}
    out = dict(stage=stage, p=p, seed=seed, retries=[], polls_in=Counter(), streak=Counter(),
               n_polls=0, n_polls_down=0, gaps=[])
    for k, pl in polls.items():
        ivs = links.get(frozenset(k), [])
        ts = [t for t, _ in pl]
        if policy:
            oc = [trig.get((k, t)) for t in ts]
        else:
            oc = [pl[i + 1][1] if i + 1 < len(pl) else None for i in range(len(pl))]
        oc_at = dict(zip(ts, oc))
        for t in ts:
            if ws <= t <= we:
                out["n_polls"] += 1
                out["n_polls_down"] += which(ivs, t) is not None
        for t in rr.get(k, []):
            if not ws <= t <= we:
                continue
            iv = which(ivs, t)
            i = bisect.bisect_left(ts, t)
            out["gaps"].append(t - ts[i - 1])
            out["retries"].append((iv is not None, (iv[1] - t) if iv else None, oc_at[t]))
        for s, e in ivs:
            if ws <= s < we:
                c = bisect.bisect_left(ts, e) - bisect.bisect_left(ts, s)
                out["polls_in"][min(c, 4)] += 1
        run = 0
        for i, (t, o) in enumerate(zip(ts, oc)):
            if o == "timed_out":
                if run == 0:
                    first = t
                run += 1
            if o != "timed_out" or i == len(ts) - 1:
                if run and ws <= first <= we:
                    out["streak"][min(run, 4)] += 1
                run = 0
    return out


def pool(res):
    g = defaultdict(list)
    for r in res:
        g[(r["stage"], r["p"])].append(r)
    rows = []
    for (st, p), rs in g.items():
        R = [x for r in rs for x in r["retries"]]
        n = len(R)
        down = [x for x in R if x[0]]
        res_ms = [x[1] for x in down if x[1] != math.inf]
        to = sum(1 for x in R if x[2] == "timed_out")
        known = sum(1 for x in R if x[2] is not None)
        pin = sum((r["polls_in"] for r in rs), Counter())
        stk = sum((r["streak"] for r in rs), Counter())
        npe = sum(pin.values())
        nst = sum(stk.values())
        gaps = [x for r in rs for x in r["gaps"]]
        rows.append(dict(
            stage=st, p=p, n_runs=len(rs), n_retry=n, retry_per_run=n / len(rs),
            retry_gap_med_ms=statistics.median(gaps) if gaps else None,
            down_share=len(down) / n if n else None,
            timed_out_share=to / known if known else None, outcome_unknown=n - known,
            down_answered=sum(1 for x in R if x[0] and x[2] == "answered"),
            up_timed_out=sum(1 for x in R if not x[0] and x[2] == "timed_out"),
            resid_mean_ms=statistics.mean(res_ms) if res_ms else None,
            resid_med_ms=statistics.median(res_ms) if res_ms else None,
            resid_open=len(down) - len(res_ms),
            poll_down_share=sum(r["n_polls_down"] for r in rs) / sum(r["n_polls"] for r in rs),
            episodes=npe, **{f"ep_polls_in_{c}": pin[c] / npe if npe else None for c in range(5)},
            ep_polls_in_mean=sum(c * v for c, v in pin.items()) / npe if npe else None,
            streaks=nst, **{f"streak_{c}": stk[c] / nst if nst else None for c in range(1, 5)},
        ))
    order = {st: i for i, (st, _) in enumerate(JOBS)}
    rows.sort(key=lambda r: (order[r["stage"]], r["p"]))
    return rows


def fmt(x, nd):
    return "-" if x is None else f"{x:.{nd}f}"


def table(rows):
    L = [SECTION + ": does the retry land inside the same outage?", "",
         "Definitions in retry.py's docstring. Policies (short, wait): the poll after a timed-out reschedule; "
         "b3p695: the poll after a timed-out poll. down = link down at send by experiment.json; to = measured "
         "timed out; disagree = down-but-answered / up-but-timed-out. resid = time from the retry to the link_up "
         "(retries sent while down). polls in episode = a directed pair's polls sent inside one merged outage "
         "(0/1/2/3/4+); streak = measured runs of consecutive timed-out polls (1/2/3/4+). all-poll down = share of "
         "all in-window polls sent while down (about p).", "",
         "| stage | p | retries/run | gap med ms | down | to | disagree | resid mean / med ms (open) | all-poll down "
         "| polls in episode 0/1/2/3/4+ (mean, n) | streak 1/2/3/4+ (n) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pe = "/".join(fmt(r[f"ep_polls_in_{c}"], 2) for c in range(5))
        sk = "/".join(fmt(r[f"streak_{c}"], 2) for c in range(1, 5))
        L.append(f"| {r['stage']} | {r['p']} | {r['retry_per_run']:.1f} | {fmt(r['retry_gap_med_ms'], 0)} | "
                 f"{fmt(r['down_share'], 3)} | {fmt(r['timed_out_share'], 3)} | {r['down_answered']}/{r['up_timed_out']} | "
                 f"{fmt(r['resid_mean_ms'], 0)} / {fmt(r['resid_med_ms'], 0)} ({r['resid_open']}) | "
                 f"{r['poll_down_share']:.3f} | {pe} ({fmt(r['ep_polls_in_mean'], 2)}, {r['episodes']}) | "
                 f"{sk} ({r['streaks']}) |")
    return L


def main():
    jobs = []
    for st, ps in JOBS:
        for r in evalagg.discover_runs(STORE, st):
            if r.p in ps:
                jobs.append((st, r.p, r.seed, r.run_dir))
    print("jobs", len(jobs), file=sys.stderr, flush=True)
    with get_context("fork").Pool(3) as pl:
        res = pl.map(analyze, jobs, chunksize=4)
    rows = pool(res)
    with open(f"{HERE}/retry_table.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    L = table(rows)
    tpath = f"{HERE}/tables.md"
    old = open(tpath).read() if os.path.exists(tpath) else ""
    i = old.find("\n" + SECTION)
    head = old if i < 0 else old[:i + 1]
    open(tpath, "w").write(head.rstrip("\n") + "\n\n" + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
