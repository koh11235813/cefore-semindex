"""How much h2's probe blocks lengthen record-hops: Q1, Q4a, Q2 and the tail, link-conditioned.

Rerun (from the worktree root, ~30 s; needs probe_extract.py's pickle first):
  /home/lab_shared/cefore-semindex/.venv/bin/python3 docs/wayfinder/semindex-stage2/evidence/07/probe_analyze2.py \\
    --pkl <scratch>/extract.pkl.gz > probe_analyze2.txt

Why conditioned: p is a per-link, per-5-s-slot *down* probability (evalgen._link_events), and
probes fire on slot boundaries. 3769 of 4012 failed probe fetches (the 2-s blocks) had the
body path (h0-h1/h1-h2) down at block start + 1 s; no successful fetch did. So "exposed to a
long block" and "the hop's own link was down" share a cause, and an unconditioned
E-vs-notE contrast on the h0 side is confounded by link state (a first, unconditioned pass was
dropped for that reason). Here every contrast is taken only over hops whose own link m-n is up
over an outcome-free window:
  - into-h2 hop (n = h2): link up over [arrival, due_i + 3 s]; exposed (E) iff h2's next poll
    toward m was due inside a block (due_i = t_i + applied_i of the governing poll).
    `pred` = mean over E hops of t_{i+1} - max(due_i, arrival): the extra wait the block-end
    poll implies if nothing else changed (mechanistic prediction of mean_E - mean_notE).
  - any other hop: link up over [arrival, arrival + 6 s]; exposed iff a block meets
    (arrival, arrival + 3 s] (a fixed window, so not length-biased).
  - placebo: same rules against blocks shifted +4 s and +6 s (burst-phase preserving);
    exc = (E - notE)_real - mean placebo (E - notE); SE = run-clustered bootstrap (300).
Q2 (per 4-hop pair, all four hops stored in-window, no link conditioning):
  dAttr = attributable poll delay on the into-h2 hop (as `pred`, 0 when not exposed);
  dAttrUp = same, counted only when link h2-m was up at due_i (a poll sent into a down link
  would have timed out anyway); ovl2 = overlap of the into-h2 hop interval with blocks
  (covers poll delay and deferred tick-driven fetches/retries); ovl23 = hops 2+3 overlap.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.normpath(os.path.dirname(os.path.abspath(__file__))))
from probe_common import PS, SHIFTS, STAGES, Blocks, hop_class, load_valid, mean  # noqa: E402

NB = 300
POS = {f"h{i}": i for i in range(5)}


def link_up(r, a, b, x, y):
    """True iff link a-b (node names) has no scheduled down interval meeting [x, y]."""
    key = tuple(sorted((POS[a], POS[b])))
    return not any(k == key and d0 <= y and d1 >= x for k, d0, d1 in r["links"])


def main():
    """Print Q1, Q4a, Q2 and the tail table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="probe_extract.py --out")
    R = load_valid(ap.parse_args().pkl)
    out = []
    P = out.append
    zero = lambda: [0, 0.0, 0, 0.0]  # noqa: E731
    q1 = defaultdict(lambda: defaultdict(lambda: {"real": zero(), 4000.0: zero(), 6000.0: zero(),
                                                   "pred": [0, 0.0], "Eok": [0, 0.0], "Efail": [0, 0.0],
                                                   "cond": [0, 0]}))
    q2 = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for ri, r in enumerate(R):
        st, p = r["stage"], r["p"]
        real = Blocks([(a, b) for a, b, *_ in r["blocks"]])
        okmap = {(a, b): ok for a, b, ok, *_ in r["blocks"]}
        pl = {S: Blocks([(a + S, b + S) for a, b, *_ in r["blocks"]]) for S in SHIFTS}
        bypair = defaultdict(dict)
        for hop in r["hops"]:
            o, s, h, m, n, arr, end, c, gi = hop
            bypair[(o, s)][h] = hop
            if end is None:
                continue
            lat = end - arr
            polls = r["polls"].get((n, m), [])
            if n == "h2":
                if gi < 0 or gi + 1 >= len(polls):
                    continue
                due = polls[gi][0] + polls[gi][2] * 1000.0
                cond = link_up(r, m, n, arr, due + 3000.0)
                expo = {lab: blk.contains(due) for lab, blk in [("real", real)] + list(pl.items())}
                pred = max(0.0, polls[gi + 1][0] - max(due, arr))
            else:
                cond = link_up(r, m, n, arr, arr + 6000.0)
                expo = {lab: blk.intersects(arr, arr + 3000.0) for lab, blk in [("real", real)] + list(pl.items())}
                pred = None
            arm = polls[gi][1] if gi >= 0 else None
            for key in [(st, p, hop_class(h, o), "all"), (st, "pooled", hop_class(h, o), "all"),
                        (st, p, hop_class(h, o), f"arm={arm}"), (st, "pooled", hop_class(h, o), f"arm={arm}")]:
                cell = q1[key][ri]
                cell["cond"][0] += 1
                if not cond:
                    continue
                cell["cond"][1] += 1
                for lab, e in expo.items():
                    acc = cell[lab]
                    if e:
                        acc[0] += 1; acc[1] += lat
                    else:
                        acc[2] += 1; acc[3] += lat
                if expo["real"] and pred is not None:
                    cell["pred"][0] += 1; cell["pred"][1] += pred
                    j = [iv for iv in real.iv if iv[0] <= due <= iv[1]][0]
                    sub = cell["Eok" if okmap[j] else "Efail"]
                    sub[0] += 1; sub[1] += lat
        for (o, s), hs in bypair.items():
            if len(hs) < 4 or any(hs[h][6] is None for h in range(1, 5)):
                continue
            lat4 = sum(hs[h][6] - hs[h][5] for h in range(1, 5))
            _o, _s, _h, m, n, arr, end, c, gi = hs[2]
            polls = r["polls"].get((n, m), [])
            da = dau = 0.0
            if gi >= 0 and gi + 1 < len(polls):
                due = polls[gi][0] + polls[gi][2] * 1000.0
                if real.contains(due):
                    da = max(0.0, polls[gi + 1][0] - max(due, arr))
                    dau = da if link_up(r, m, n, due, due) else 0.0
            ov2 = real.overlap(hs[2][5], hs[2][6])
            ov3 = real.overlap(hs[3][5], hs[3][6])
            for key in [(st, p), (st, "pooled")]:
                a = q2[key]
                a[0] += 1; a[1] += lat4; a[2] += da; a[3] += dau; a[4] += ov2; a[5] += ov2 + ov3

    rng = random.Random(28)

    def diff(cells, lab):
        nE = sum(c[lab][0] for c in cells); sE = sum(c[lab][1] for c in cells)
        nN = sum(c[lab][2] for c in cells); sN = sum(c[lab][3] for c in cells)
        if not nE or not nN:
            return float("nan"), nE, nN, float("nan"), float("nan")
        return sE / nE - sN / nN, nE, nN, sE / nE, sN / nN

    def exc_of(cells):
        return diff(cells, "real")[0] - mean([diff(cells, S)[0] for S in SHIFTS])

    def row(key):
        cells = list(q1[key].values())
        d, nE, nN, mE, mN = diff(cells, "real")
        pls = [diff(cells, S)[0] for S in SHIFTS]
        exc = d - mean(pls)
        bs = [v for v in (exc_of([cells[rng.randrange(len(cells))] for _ in cells]) for _ in range(NB)) if v == v]
        se = (sum((x - mean(bs)) ** 2 for x in bs) / (len(bs) - 1)) ** 0.5 if len(bs) > 2 else float("nan")
        share = nE / (nE + nN) if nE + nN else float("nan")
        ncond = sum(c["cond"][1] for c in cells); nall = sum(c["cond"][0] for c in cells)
        pr = sum(c["pred"][1] for c in cells) / max(1, sum(c["pred"][0] for c in cells))
        eok = [sum(c["Eok"][i] for c in cells) for i in (0, 1)]
        efl = [sum(c["Efail"][i] for c in cells) for i in (0, 1)]
        sub = (f"  predUB={pr:5.0f}  rawE_ok(n={eok[0]})={eok[1]/eok[0] if eok[0] else float('nan'):6.0f}"
               f"  rawE_fail(n={efl[0]})={efl[1]/efl[0] if efl[0] else float('nan'):6.0f}") if key[2].startswith("into") else ""
        # The placebo is only trusted where its two shifts agree; otherwise burst/poll phase
        # structure dominates and `exc` is not identified.
        flag = "ok " if (pls[0] == pls[0] and pls[1] == pls[1] and se == se and abs(pls[0] - pls[1]) <= se) else "NID"
        return (f"{nall:6d} {100*ncond/max(1,nall):5.1f} {100*share:5.1f} {mN:7.0f} {mE:7.0f} {d:7.0f} "
                f"{pls[0]:7.0f} {pls[1]:7.0f} {exc:7.0f} {se:6.0f} {share*exc:7.1f} {flag}" + sub)

    hdr = "  n_all cond%  E%    mean_N  mean_E  E-N    pl+4s   pl+6s    exc     SE  per-hop id"
    classes = [f"{rl}:{o}" for rl in ("into_h2", "out_h2", "first", "last") for o in ("h0", "h4")]
    P("== Q1 (link-conditioned): latency excess per hop class, ms ==")
    P("(cond% = share of uncensored hops whose own link passes the up-window; E%/means/contrasts are among those.")
    P(" id: ok = |pl+4s - pl+6s| <= SE, NID = placebos disagree, exc not identified. predUB = mean over E hops of")
    P(" t_{i+1} - max(due, arrival): upper bound on E-N from the poll delay alone. rawE_ok/rawE_fail = raw mean")
    P(" latency of E hops whose due fell in a successful / failed (2 s) block, NOT placebo-corrected.)")
    for st in STAGES:
        for p in PS + ["pooled"]:
            P(f"-- {st} p={p}")
            P(f"   {'class':11s}" + hdr)
            for cl in classes:
                if (st, p, cl, "all") in q1:
                    P(f"   {cl:11s}" + row((st, p, cl, "all")))

    P("\n== Q4a (link-conditioned): into-h2 hop by arm of the governing h2 poll ==")
    for st in STAGES:
        arms = sorted({k[3] for k in q1 if k[0] == st and k[3] not in ("all", "arm=None")}, key=lambda a: float(a[4:]))
        for p in PS + ["pooled"]:
            P(f"-- {st} p={p}")
            P(f"   {'class':11s} {'arm':11s}" + hdr)
            for cl in ("into_h2:h0", "into_h2:h4"):
                for a in arms:
                    if (st, p, cl, a) in q1:
                        P(f"   {cl:11s} {a:11s}" + row((st, p, cl, a)))

    P("\n== Q2: probe contribution bound on mean 4-hop latency (ms per 4-hop pair; % of lat4) ==")
    P("stage      p        pairs   lat4    dAttr   %     dAttrUp  %     ovl2    %     ovl23   %")
    for st in STAGES:
        for p in PS + ["pooled"]:
            a = q2.get((st, p))
            if not a:
                continue
            n, l4 = a[0], a[1] / a[0]
            v = [x / n for x in a[2:]]
            P(f"{st:10s} {str(p):7s} {n:7d} {l4:7.0f} " + " ".join(f"{x:7.0f} {100*x/l4:5.1f}" for x in v))
    # ---- how far h2's no-tick period runs past the probe_retrieved stamp -------------------
    P("\n== Tail: h2 polls due inside a block, sent at t_{i+1}; t_{i+1} - block end (retrieved ts), ms ==")
    P("stage      n      mean  p50  p90  max   (<0 would mean a poll inside the stamped block)")
    for st in STAGES:
        tails = []
        for r in R:
            if r["stage"] != st:
                continue
            real = Blocks([(a, b) for a, b, *_ in r["blocks"]])
            for (n, m), lst in r["polls"].items():
                if n != "h2":
                    continue
                for i in range(len(lst) - 1):
                    due = lst[i][0] + lst[i][2] * 1000.0
                    if r["ws"] <= lst[i][0] <= r["we"] and real.contains(due):
                        j = [iv for iv in real.iv if iv[0] <= due <= iv[1]][0]
                        tails.append(lst[i + 1][0] - j[1])
        tails.sort()
        if tails:
            P(f"{st:10s} {len(tails):5d} {mean(tails):6.0f} {tails[len(tails)//2]:4.0f} "
              f"{tails[int(.9*(len(tails)-1))]:4.0f} {tails[-1]:5.0f}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
