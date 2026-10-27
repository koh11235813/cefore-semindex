"""Per-neighbor in-flight cap check: does the fetch side saturate the cap, and
what do the fetch timeouts look like when it is raised?

Read-only over the store. For each stage (default: t2k4 t2c32k4 t2k8 t2c32k8)
at p=0.4, pooled over every seed directory, over the whole run (not only the
measurement window; in the trial the plan counts came out equal to the
windowed counts in k_table.csv):

- plans: fetch_plan decisions; capped: share with
  state.in_flight + len(action.fetched) >= cap, where cap is the run's
  meta.json max_in_flight_per_neighbor (absent/null = the agent default 8).
  A plan that fills the cap is one the cap cut short, which is what decides
  whether the cap, not k, limits how much is fetched at once.
- in_flight quartiles before each plan.
- fetch_timeout events split by whether the node-neighbor link was down at
  any point in the 2 s (the Interest lifetime) before the timeout. Timeouts
  with the link up for the whole 2 s are the ones the link cannot explain:
  the upstream was reachable but did not answer within the lifetime.
- send-to-timeout delay: from the latest fetch_plan to that neighbor that
  carried the (origin, seq) to the fetch_timeout (min / max), to confirm the
  timeouts are lifetime expiries and not something else.
- records stored, and the median gap between consecutive record_stored on
  h1-h3 (the relaying nodes), counting only gaps under 300 ms so the idle
  time between generation bursts does not swamp the per-Data service time.

Run:
  PYTHONPATH=<worktree>/src /home/lab_shared/cefore-semindex/.venv/bin/python3 \
    capcheck.py [--store /home/lab_shared/semindex-eval] [stage ...]
"""
import argparse, bisect, glob, json, os, statistics, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import breakdown2  # noqa: E402

DEFAULT_CAP = 8          # agent default max_in_flight_per_neighbor
LIFETIME_MS = 2000.0     # fetch Interest lifetime; the window checked for a down link
GAP_LIMIT_MS = 300.0     # longer gaps are idle time between bursts, not service time
RELAYS = ("h1", "h2", "h3")


def quartiles(v):
    """(Q1, median, Q3) of a non-empty list."""
    if len(v) < 2:
        return (v[0], v[0], v[0])
    q = statistics.quantiles(v, n=4, method="inclusive")
    return (q[0], q[1], q[2])


def check_stage(store, stage, p_dir="p0.4"):
    """One stage's pooled numbers as a dict (see module docstring)."""
    plans = capped = stored = 0
    in_flight = []
    to_up = to_down = 0
    send_to_timeout = []
    gaps = []
    caps = set()
    for rd in sorted(glob.glob(f"{store}/{stage}/{p_dir}/*/")):
        meta = json.load(open(f"{rd}/meta.json"))
        cap = meta.get("max_in_flight_per_neighbor") or DEFAULT_CAP
        caps.add(cap)
        links = breakdown2.link_intervals(json.load(open(f"{rd}/results.json")))
        nodes = breakdown2.load_events(rd)
        for n, evs in nodes.items():
            sent = defaultdict(list)   # (nbr, origin, seq) -> fetch_plan ts, in log order
            st_ts = []
            for e in evs:
                k = e["event"]
                if k == "decision" and e["kind"] == "fetch_plan":
                    f = e["action"]["fetched"]
                    fl = e["state"]["in_flight"]
                    plans += 1
                    in_flight.append(fl)
                    capped += fl + len(f) >= cap
                    for o, s in f:
                        sent[(e["action"]["neighbor"], o, s)].append(e["ts_ms"])
                elif k == "fetch_timeout":
                    t = e["ts_ms"]
                    ivs = links.get(frozenset((n, e["neighbor"])), [])
                    if breakdown2.overlap(ivs, t - LIFETIME_MS, t) > 0:
                        to_down += 1
                    else:
                        to_up += 1
                    ts = sent.get((e["neighbor"], e["origin"], e["seq"]), [])
                    i = bisect.bisect_right(ts, t)
                    if i:
                        send_to_timeout.append(t - ts[i - 1])
                elif k == "record_stored":
                    stored += 1
                    st_ts.append(e["ts_ms"])
            if n in RELAYS:
                st_ts.sort()
                gaps += [b - a for a, b in zip(st_ts, st_ts[1:]) if b - a < GAP_LIMIT_MS]
    q1, med, q3 = quartiles(in_flight)
    return dict(
        stage=stage, cap="/".join(str(c) for c in sorted(caps)), plans=plans,
        capped_share=capped / plans if plans else None,
        in_flight_q1=q1, in_flight_med=med, in_flight_q3=q3,
        timeouts=to_up + to_down, timeouts_link_up_2s=to_up, timeouts_link_down=to_down,
        send_to_timeout_min_ms=min(send_to_timeout) if send_to_timeout else None,
        send_to_timeout_max_ms=max(send_to_timeout) if send_to_timeout else None,
        records_stored=stored, relay_gap_med_ms=statistics.median(gaps) if gaps else None,
    )


def main():
    """Print a Markdown table with one row per stage of the measures check_stage returns."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="/home/lab_shared/semindex-eval")
    ap.add_argument("stages", nargs="*", default=["t2k4", "t2c32k4", "t2k8", "t2c32k8"])
    a = ap.parse_args()
    rows = [check_stage(a.store, s) for s in a.stages]
    cols = list(rows[0].keys())
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join("-" if r[c] is None else (f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]))
                                for c in cols) + " |")


if __name__ == "__main__":
    main()
