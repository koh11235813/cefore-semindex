"""h2's probe blocks and how they delay h2's state polls: block table, Q3 and Q4b.

Rerun (from the worktree root, seconds; needs probe_extract.py's pickle first):
  /home/lab_shared/cefore-semindex/.venv/bin/python3 docs/wayfinder/semindex-stage2/evidence/07/probe_analyze.py \\
    --pkl <scratch>/extract.pkl.gz > probe_analyze.txt

The latency questions (Q1 per-hop excess, Q2 block time on completed 4-hop pairs, which is
descriptive and not an upper bound, Q4a by arm) are in
probe_analyze2.py, conditioned on the link schedule. 2026-11-02: an unconditioned first pass of
those three was dropped from this file, because p is a per-link, per-5-s-slot down probability
and a failed probe fetch shares its cause (a down link) with a slow hop, which confounded it.

Block = [probe_discovered.ts, probe_retrieved.ts] of a matched probe at h2 (see probe_extract.py).
While inside, h2's loop never runs on_tick (no state poll, no tick-driven fetch/retry), but
packets are still dispatched: h2 serves neighbors, handles replies and on_state fetch plans.

Block table, per stage x p: matched-probe count, success share, duration quantiles of successful
and failed fetches, summed block time / 120 s window (mean over runs), unmatched probes.

Q3: gaps (t_i, t_{i+1}] of n's polls toward m with t_i in the window [t0 + 10 s, t0 + 130 s]
(the evalagg / evidence/05 attrib window; t0 from semindex.evalagg.derive_t0_ms) and a next poll
logged, n in h1/h2/h3, both neighbors pooled. Strata:
  nob = no real block intersects the gap; cont = intersects but due_i outside every block;
  due = due_i = t_i + applied_i inside a block. gap - applied in ms (share %, mean, p90).
  h1/h3 are tested against h2's blocks as a placebo (they are never blocked themselves).
  The `due` share of h2 is the delayed-poll share; its denominator is h2's in-window gaps.

Q4b: Q3 split by arm (B stage = its arm; s4a = the gap's own poll's interval_s; s4c = stage-1
default policy, labelled 2.0).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.normpath(os.path.dirname(os.path.abspath(__file__))))
from probe_common import PS, STAGES, Blocks, load_valid, mean, q  # noqa: E402


def main():
    """Print the block table, Q3 and Q4b."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="probe_extract.py --out")
    R = load_valid(ap.parse_args().pkl)
    out = []
    P = out.append

    # ---------------------------------------------------------------- block descriptives
    P("== Blocks (h2 matched probes): per stage x p ==")
    P("stage      p    n_blk  ok%   dur_ok_med dur_ok_p90  dur_fail_med  blocked%_of_120s  nomatch")
    for st in STAGES:
        for p in PS:
            rs = [r for r in R if r["stage"] == st and r["p"] == p]
            bl = [b for r in rs for b in r["blocks"]]
            ok = [b[1] - b[0] for b in bl if b[2]]
            fl = [b[1] - b[0] for b in bl if not b[2]]
            share = mean([sum(b[1] - b[0] for b in r["blocks"]) / 120000.0 for r in rs])
            nm = sum(r["chk"].get("probes_nomatch", 0) for r in rs)
            P(f"{st:10s} {p:.1f} {len(bl):6d} {100*len(ok)/max(1,len(bl)):5.1f} {q(ok,.5):9.0f} {q(ok,.9):9.0f} "
              f"{q(fl,.5):12.0f} {100*share:12.1f} {nm:8d}")

    # ---------------------------------------------------------------- Q3 / Q4b
    g = defaultdict(lambda: {"nob": [], "cont": [], "due": []})
    for r in R:
        st, p = r["stage"], r["p"]
        real = Blocks([(a, b) for a, b, *_ in r["blocks"]])
        for (n, m), lst in r["polls"].items():
            if n not in ("h1", "h2", "h3"):
                continue
            for i in range(len(lst) - 1):
                t, arm, ap_s = lst[i]
                if not (r["ws"] <= t <= r["we"]):
                    continue
                t1 = lst[i + 1][0]
                dg = (t1 - t) - ap_s * 1000.0
                due = t + ap_s * 1000.0
                if real.contains(due):
                    s = "due"
                elif real.intersects(t, t1):
                    s = "cont"
                else:
                    s = "nob"
                for key in [(st, p, n, "all"), (st, "pooled", n, "all"), (st, p, n, f"arm={arm}"),
                            (st, "pooled", n, f"arm={arm}")]:
                    g[key][s].append(dg)

    def q3row(d):
        N = sum(len(v) for v in d.values())
        cells = []
        for s in ("nob", "cont", "due"):
            v = d[s]
            cells.append(f"{100*len(v)/N:5.1f} {mean(v):6.0f} {q(v,.9):6.0f}")
        allv = d["nob"] + d["cont"] + d["due"]
        return f"{N:7d}  " + "   ".join(cells) + f"   {mean(allv):6.0f}"

    P("\n== Q3: state-poll gap minus applied interval (ms) at h1/h2/h3, by block stratum ==")
    P("(nob: no block in gap; cont: block in gap but poll due outside it; due: poll due inside a block.")
    P(" columns per stratum: share%, mean, p90; then overall mean)")
    for st in STAGES:
        for p in PS + ["pooled"]:
            P(f"-- {st} p={p}")
            P("   node    gaps    nob: %  mean  p90    cont: %  mean  p90    due: %  mean  p90    all_mean")
            for n in ("h1", "h2", "h3"):
                key = (st, p, n, "all")
                if key in g:
                    P(f"   {n:5s}" + q3row(g[key]))

    P("\n== Q4b: h2 gap minus applied by arm (h2 only; h1/h3 for contrast) ==")
    for st in STAGES:
        arms = sorted({k[3] for k in g if k[0] == st and k[3] != "all"}, key=lambda a: float(a[4:]))
        for p in PS + ["pooled"]:
            P(f"-- {st} p={p}")
            P("   node  arm           gaps    nob: %  mean  p90    cont: %  mean  p90    due: %  mean  p90    all_mean")
            for n in ("h2", "h1", "h3"):
                for a in arms:
                    key = (st, p, n, a)
                    if key in g:
                        P(f"   {n:5s} {a:11s}" + q3row(g[key]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
