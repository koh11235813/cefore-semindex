"""Can the A stage's per-decision data predict "every node always uses arm x" (the B stages)?

Three predictors of B(x)'s per-hop wait + fetch, all from A's record-hops (uncensored only, as in
evidence/05), compared end-to-end with B's measured discovery latency (pairs, `pair` sums):

  current  evidence/05's rule: the whole wait of a record-hop goes to decision i, the decision whose
           gap (t_i, t_{i+1}] holds the record's arrival at the upstream m.
  seg      time-segment re-attribution (per-decision reward): each ms goes to the decision whose gap
           holds it -- arrival..t_{i+1} to i, the full gap of each failed poll k (i < k < j) to k,
           and the final reply RTT t_j..learned plus the fetch to j (j = latest poll <= learned).
           Predictor: (sum of charges to arm-x decisions) / (record-hops arriving in arm-x gaps).
           For a B stage every decision is x, so it equals the per-record-hop wait+fetch exactly.
  chain    sequential composition: in-gap wait, q1 = P(poll i+1 does not yield), the continuation
           probability c after a failed arm-x poll, the gap G after a failed arm-x poll, the reply
           RTT after a failure, and the post-reply delay when nothing failed, all measured in A, then
           composed for an all-x schedule. Two continuation models: `step` (c by failed-poll index
           1/2/3+) and `elapsed` (c by time since t_{i+1}, the 5 s slot outage being time-based).
  is       per-decision importance sampling: keep A record-hops whose failed polls i+1..j-1 all
           chose x, weighted 4^(j-i-1) (each spanned decision chose x with logged prob 0.25; decision
           i is conditioned on, decision j's arm cannot change `learned`). Unbiased for B's post
           part, but the weight explodes for short arms (a 10 s outage spans ~20 0.5 s polls).

Uncertainty: bootstrap over runs within (stage, p) (`--reps`, default 200); B direct's SE is the same
bootstrap over B runs. Censoring flags are copied from evidence/07 predicted_vs_B.csv (`unreliable`).

s4-specific: the stage names (s4a, s4b*) are fixed below, so this reads only the NORMAL
re-collection; the FULL main run (s2*) would need its own stage map.

Rerun (from the worktree root; <AGG> = the evalagg output dir of the s4 stages):
  PYTHONPATH=src:docs/wayfinder/semindex-stage2/evidence/05 \\
    /home/lab_shared/cefore-semindex/.venv/bin/python3 docs/wayfinder/semindex-stage2/evidence/07/seq_model.py \\
    --store /home/lab_shared/semindex-eval --agg <AGG>
Needs predicted_vs_B.csv in --out first (`run_all.py --prefix s4 --out <same dir>`): it supplies
the censoring flags and is the reference for the `current` identity check.
Outputs (into --out, default this script's directory): seq_vs_B.csv, seq_pieces.csv, seq_model.md.
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np

import attrib
import run_all
from run_all import ARMS, PS, fmt, ratio, write_csv
from semindex import evalagg

HERE = os.path.dirname(os.path.abspath(__file__))
B_STAGES = {"s4b0p5": 0.5, "s4b1p359": 1.359, "s4b3p695": 3.695, "s4b10p043": 10.043}
A_STAGE = "s4a"
HOPS = (1, 2, 3, 4)


def work(args):
    """Pool task: one run's opt-in sequential sums and pair sums (DISC inherited through fork)."""
    stage, p, seed, run_dir = args
    res = attrib.attribute_run(stage, p, seed, run_dir, run_all.DISC.get((stage, p, seed)), seq=True)
    return stage, p, seed, res["seq"], res["pair"]


class RunMatrix:
    """Runs of one (stage, p) as a dense (runs x features) matrix, so a bootstrap is one matmul.

    Features are (seq key, field) pairs plus ("pair", k) for the e2e pair sums.
    """

    def __init__(self, runs):
        self.seeds = [r[0] for r in runs]
        cols = {}
        for _seed, sq, pr in runs:
            for k, v in sq.items():
                for f in v:
                    cols.setdefault((k, f), len(cols))
            for k in pr:
                cols.setdefault(("pair", k), len(cols))
        self.cols = cols
        M = np.zeros((len(runs), len(cols)))
        for r, (_seed, sq, pr) in enumerate(runs):
            for k, v in sq.items():
                for f, x in v.items():
                    M[r, cols[(k, f)]] = x
            for k, x in pr.items():
                M[r, cols[("pair", k)]] = x
        self.M = M

    def sums(self, w=None):
        """Feature sums with run multiplicities `w` (None = each run once) as a lookup object."""
        v = self.M.sum(0) if w is None else w @ self.M
        return Sums(self.cols, v)


class Sums:
    """Read access to summed features: s.get(key, field) -> float (0 when absent)."""

    def __init__(self, cols, v):
        self.cols, self.v = cols, v

    def get(self, key, field):
        i = self.cols.get((key, field))
        return float(self.v[i]) if i is not None else 0.0

    def keys(self, tag):
        """Distinct seq keys with this leading tag."""
        return {k for (k, _f) in self.cols if isinstance(k, tuple) and k and k[0] == tag}


def pieces(S, x):
    """Model pieces for arm x from summed features S (pooled where noted)."""
    out = {}
    for h in HOPS:
        k = ("rh", x, h)
        n = S.get(k, "n")
        nf1 = S.get(k, "n_f1")
        out[h] = dict(n=n, ingap=ratio(S.get(k, "ingap"), n), post=ratio(S.get(k, "post"), n),
                      fetch=ratio(S.get(k, "fetch"), n), q1=ratio(nf1, n),
                      post_nf=ratio(S.get(k, "post_nf"), n - nf1), rtt_f=ratio(S.get(k, "rtt_f"), nf1),
                      own=ratio(S.get(("seg", x, h), "own"), n),
                      carried=ratio(S.get(("seg", x, h), "carried"), n),
                      final_fetch=ratio(S.get(("seg", x, h), "final") + S.get(("seg", x, h), "fetch"), n),
                      # the record's own reply RTT + fetch, for the variant that keeps them with decision i
                      own_rtt_fetch=ratio(S.get(k, "post_nf") + S.get(k, "rtt_f") + S.get(k, "fetch"), n),
                      is_post=ratio(S.get(k, "is_post"), n),
                      # IS weight mass on failure spans: E_A[w 1{nf>=1}] = P_B(nf>=1), so dividing by
                      # the observed failure count gives ~1 when the all-x spans B needs show up in A;
                      # well below 1 = the long all-x spans are missing from the sample.
                      is_fail_mass=ratio(S.get(k, "is_w") - (n - nf1), nf1))
    # continuation and post-failure gap pooled over hops (one link process per hop, same rule)
    st = defaultdict(lambda: [0.0, 0.0, 0.0])
    for kk in S.keys("step"):
        if kk[1] == x:
            a = st[kk[3]]
            a[0] += S.get(kk, "n"); a[1] += S.get(kk, "cont"); a[2] += S.get(kk, "gap")
    se = defaultdict(lambda: [0.0, 0.0])
    for kk in S.keys("stepe"):
        if kk[1] == x:
            a = se[kk[3]]
            a[0] += S.get(kk, "n"); a[1] += S.get(kk, "cont")
    nst = sum(a[0] for a in st.values())
    out["G"] = ratio(sum(a[2] for a in st.values()), nst)
    out["c_step"] = {s: ratio(st[s][1], st[s][0]) for s in attrib.SEQ_SBUCKETS}
    out["n_step"] = {s: st[s][0] for s in attrib.SEQ_SBUCKETS}
    out["c_elapsed"] = {b: ratio(se[b][1], se[b][0]) for b in attrib.SEQ_EBINS_S}
    out["n_elapsed"] = {b: se[b][0] for b in attrib.SEQ_EBINS_S}
    # decision level: P(fail) and P(fail | previous poll failed), arm of the earlier poll; slow-reply flag
    n0 = sum(S.get(("trans", x, h, 0), "n") for h in HOPS); f0 = sum(S.get(("trans", x, h, 0), "next_fail") for h in HOPS)
    n1 = sum(S.get(("trans", x, h, 1), "n") for h in HOPS); f1 = sum(S.get(("trans", x, h, 1), "next_fail") for h in HOPS)
    out["dec_pfail"] = ratio(n1, n0 + n1)
    out["dec_c"] = ratio(f1, n1)
    out["dec_c_after_ok"] = ratio(f0, n0)
    nd = [sum(S.get(("dec", x, h, b), "n") for h in HOPS) for b in (0, 1, 2)]
    out["dec_nrep2_share"] = ratio(nd[2], sum(nd))
    return out


# A continuation estimate from fewer failed polls, or one at 1.0 (no recovery seen), borrows the
# previous step/bin's value. Counts are record-weighted (one failed poll counts once per waiting
# record), so a bin of ~150 can be one or two outages; c = 1.0 there means "unidentified", not
# "never recovers" (2026-11-02: the 0.5 s arm at p = 0.1 had c = 1.0 over 164 counts at 5-10 s).
MIN_N = 30


def chain_failed_time(G, c_of_step):
    """E[sum of failed-poll gaps | poll i+1 failed] = G * sum_s prod_{r<s} c(r), c(r) given per step."""
    tot, surv = 0.0, 1.0
    for s in range(1, 200000):
        tot += surv
        c = c_of_step(s)
        if c != c:  # no failed poll observed at all (p = 0): nothing to continue
            c = 0.0
        if c >= 1.0:  # no recovery observed: the model is undefined
            return float("nan")
        surv *= c
        if surv < 1e-12:
            break
    return G * tot


def estimates(S, x):
    """Per-hop wait+fetch for arm x under each predictor, from summed features S."""
    P = pieces(S, x)
    G = P["G"]
    if G != G:  # no failed arm-x poll at all (p = 0): fall back to the arm itself (inferred, not measured)
        G = x * 1000.0
    def pooled(c, n, keys):
        """c by key with thin keys (< MIN_N failed polls) replaced by the previous key's value."""
        out, last = {}, float("nan")
        for kk in keys:
            if n[kk] >= MIN_N and c[kk] < 1.0:
                last = c[kk]
            out[kk] = last
        return out

    cs = pooled(P["c_step"], P["n_step"], attrib.SEQ_SBUCKETS)
    ce = pooled(P["c_elapsed"], P["n_elapsed"], attrib.SEQ_EBINS_S)

    def c_step(s):
        return cs[min(s, attrib.SEQ_SBUCKETS[-1])]

    def c_el(s):
        e = (s - 1) * G  # elapsed since t_{i+1} at the s-th failed poll on an all-x grid
        return ce[max(bb for bb in attrib.SEQ_EBINS_S if e >= bb * 1000.0)]

    Ef_step = chain_failed_time(G, c_step) if G == G else float("nan")
    Ef_el = chain_failed_time(G, c_el) if G == G else float("nan")
    out = defaultdict(dict)
    for h in HOPS:
        ph = P[h]
        out["current"][h] = ph["ingap"] + ph["post"] + ph["fetch"]
        out["seg"][h] = ph["own"] + ph["carried"] + ph["final_fetch"]
        # variant: only the full gaps of failed polls move; reply RTT + fetch stay with decision i
        out["seg_rtt_i"][h] = ph["own"] + ph["carried"] + ph["own_rtt_fetch"]
        for name, Ef in (("chain_step", Ef_step), ("chain_elapsed", Ef_el)):
            q1 = ph["q1"]
            post = (1 - q1) * ph["post_nf"] + (q1 * (Ef + ph["rtt_f"]) if q1 else 0.0)
            out[name][h] = ph["ingap"] + post + ph["fetch"]
        out["is"][h] = ph["ingap"] + ph["is_post"] + ph["fetch"]
    return out, P


def e2e(perhop, H=4):
    """End-to-end at H hops = sum of per-hop means (as evidence/05 predicted_vs_B)."""
    return sum(perhop[h] for h in range(1, H + 1))


def direct(S, H=4):
    """Measured end-to-end mean latency of uncensored (origin, seq, node) pairs at H hops."""
    return ratio(S.get("pair", ("lat", H)), S.get("pair", ("n", H)))


def boot(rm, fn, reps, rng):
    """Bootstrap SE of each value of fn(Sums) -> {name: float} over runs of `rm`; NaN replicates dropped."""
    n = len(rm.seeds)
    vals = defaultdict(list)
    for _ in range(reps):
        w = rng.multinomial(n, [1.0 / n] * n).astype(float)
        for k, v in fn(rm.sums(w)).items():
            if v == v:
                vals[k].append(v)
    return defaultdict(lambda: float("nan"),
                       {k: float(np.std(v, ddof=1)) for k, v in vals.items() if len(v) > 1})


def main():
    """Attribute s4a and the s4b* stages with seq=True, then write seq_vs_B / seq_pieces / seq_model.md."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=run_all.STORE)
    ap.add_argument("--agg", required=True, help="evalagg output dir holding <stage>/discovery.csv")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    jobs = []
    for st in [A_STAGE] + list(B_STAGES):
        run_all.load_disc(st, a.agg)
        for r in evalagg.discover_runs(a.store, st):
            jobs.append((st, r.p, r.seed, r.run_dir))
    runs = defaultdict(list)  # (stage, p) -> [(seed, seq, pair)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        for st, p, seed, sq, pr in pool.imap_unordered(work, jobs, chunksize=4):
            runs[(st, p)].append((seed, sq, pr))
    for v in runs.values():
        v.sort(key=lambda r: r[0])
    rms = {k: RunMatrix(v) for k, v in runs.items()}
    # A restricted to B's seeds: link schedules depend on (seed, p) only, so these runs saw the very
    # outage schedules the B runs saw -- the strongest available check that a piece transfers.
    bseeds = {s for (st, _p), v in runs.items() if st in B_STAGES for s, _q, _r in v}
    rms_a_matched = {p: RunMatrix([r for r in runs[(A_STAGE, p)] if r[0] in bseeds]) for p in PS}

    cur = {}
    # 2026-11-02 fix: read the reference from --out; reading it from HERE paired a scratch --out
    # with the committed predicted_vs_B.csv, so a rerun elsewhere checked against the wrong file.
    with open(f"{a.out}/predicted_vs_B.csv") as f:
        for r in csv.DictReader(f):
            cur[(float(r["arm_s"]), float(r["p"]), int(r["hop"]))] = r

    rng = np.random.default_rng(20261102)
    vs_rows, piece_rows = [], []
    names = ("current", "seg", "seg_rtt_i", "chain_step", "chain_elapsed", "is")
    for x in ARMS:
        bst = {v: k for k, v in B_STAGES.items()}[x]
        for p in PS:
            SA, SB = rms[(A_STAGE, p)].sums(), rms[(bst, p)].sums()
            estA, PA = estimates(SA, x)
            estB, PB = estimates(SB, x)
            _estM, PM = estimates(rms_a_matched[p].sums(), x)
            seA = boot(rms[(A_STAGE, p)], lambda S: {nm: e2e(estimates(S, x)[0][nm], 4) for nm in names},
                       a.reps, rng)
            seB = boot(rms[(bst, p)], lambda S: {H: direct(S, H) for H in HOPS}, a.reps, rng)
            for H in HOPS:
                bd = direct(SB, H)
                row = dict(arm_s=x, p=p, hop=H, b_direct_ms=bd, b_direct_se=seB[H])
                for nm in names:
                    v = e2e(estA[nm], H)
                    row[f"{nm}_ms"] = v
                    row[f"{nm}_se"] = seA[nm] if H == 4 else float("nan")  # SE only at 4 hops
                    row[f"{nm}_relerr"] = ratio(v - bd, bd)
                # structure check: the same estimator fed B's own pieces (seg/current/is on B are
                # identities; chain on B tests only the continuation model)
                for nm in names:
                    row[f"b_self_{nm}_ms"] = e2e(estB[nm], H)
                row["b_self_chain_step_relerr"] = ratio(row["b_self_chain_step_ms"] - bd, bd)
                row["b_self_chain_elapsed_relerr"] = ratio(row["b_self_chain_elapsed_ms"] - bd, bd)
                row["is_fail_mass_hop1_4"] = "/".join(f"{PA[h]['is_fail_mass']:.2f}" for h in HOPS)
                for nm in names:
                    # z = (pred - B direct) / sqrt(SE_pred^2 + SE_B^2), 4 hops only
                    row[f"{nm}_z"] = ((row[f"{nm}_ms"] - bd) / math.hypot(row[f"{nm}_se"], seB[H])
                                      if H == 4 else float("nan"))
                c = cur.get((x, p, H), {})
                row["evidence07_pred_e2e_ms"] = float(c.get("pred_e2e_ms", "nan") or "nan")
                row["evidence07_relerr_direct"] = float(c.get("relerr_e2e_vs_direct", "nan") or "nan")
                row["unreliable"] = int(c.get("unreliable", 0) or 0)
                row["a_e2e_cens"] = float(c.get("e2e_censored_share_a", "nan") or "nan")
                row["b_e2e_cens"] = float(c.get("e2e_censored_share_b", "nan") or "nan")
                vs_rows.append(row)
            for side, PP in (("A", PA), ("A_seeds_of_B", PM), ("B", PB)):
                hp = [PP[h] for h in HOPS]
                ntot = sum(q["n"] for q in hp)
                piece_rows.append(dict(
                    arm_s=x, p=p, side=side, n_recordhops=int(ntot),
                    ingap_ms=sum(q["ingap"] * q["n"] for q in hp) / ntot,
                    post_ms=sum(q["post"] * q["n"] for q in hp) / ntot,
                    q1=sum(q["q1"] * q["n"] for q in hp) / ntot,
                    post_nofail_ms=np.nanmean([q["post_nf"] for q in hp]),
                    rtt_after_fail_ms=np.nanmean([q["rtt_f"] for q in hp]),
                    G_after_fail_ms=PP["G"],
                    c_step1=PP["c_step"][1], c_step2=PP["c_step"][2], c_step3p=PP["c_step"][3],
                    n_step1=int(PP["n_step"][1]), n_step2=int(PP["n_step"][2]), n_step3p=int(PP["n_step"][3]),
                    **{f"c_el_{b:g}s": PP["c_elapsed"][b] for b in attrib.SEQ_EBINS_S},
                    **{f"n_el_{b:g}s": int(PP["n_elapsed"][b]) for b in attrib.SEQ_EBINS_S},
                    seg_own_ms=sum(q["own"] * q["n"] for q in hp) / ntot,
                    seg_carried_ms=sum(q["carried"] * q["n"] for q in hp) / ntot,
                    seg_final_fetch_ms=sum(q["final_fetch"] * q["n"] for q in hp) / ntot,
                    dec_pfail=PP["dec_pfail"], dec_c_after_fail=PP["dec_c"], dec_c_after_ok=PP["dec_c_after_ok"],
                    dec_nrep2_share=PP["dec_nrep2_share"],
                ))
    write_csv(f"{a.out}/seq_vs_B.csv", vs_rows)
    write_csv(f"{a.out}/seq_pieces.csv", piece_rows)

    # B identity of `seg` (and `current`): fed B's own data both must equal B's per-hop wait+fetch
    # sum (evidence/07 b_e2e_sum_of_hop_means_ms). Second identity: `current` from A must reproduce
    # evidence/07's pred_e2e_ms (same rule, same data). Both reported as the worst relative gap.
    worst_b = worst_cur = 0.0
    for r in vs_rows:
        c = cur[(r["arm_s"], r["p"], r["hop"])]
        bs = float(c["b_e2e_sum_of_hop_means_ms"])
        worst_b = max(worst_b, abs(r["b_self_seg_ms"] - bs) / bs, abs(r["b_self_current_ms"] - bs) / bs)
        worst_cur = max(worst_cur, abs(r["current_ms"] - r["evidence07_pred_e2e_ms"]) / r["evidence07_pred_e2e_ms"])
    write_md(f"{a.out}/seq_model.md", vs_rows, piece_rows, worst_b, worst_cur, names)
    print(f"runs={len(jobs)} B-identity worst relgap={worst_b:.3g} current-vs-evidence07 worst relgap={worst_cur:.3g}")


def pc(x):
    """Signed whole-percent text; NaN as '-'."""
    return "-" if x != x else f"{100 * x:+.0f}%"


def g(x, nd=2):
    """Fixed-point text with nd decimals; NaN as '-'."""
    return "-" if x != x else f"{x:.{nd}f}"


def write_md(path, vs_rows, piece_rows, worst_b, worst_cur, names):
    """Markdown tables straight from the rows just written."""
    L = ["# Sequential attribution vs B (seq_model.py)", "",
         f"- B identity: `seg` and `current` fed each B stage's own data vs evidence/07 "
         f"`b_e2e_sum_of_hop_means_ms`, worst relative gap {worst_b:.2g}.",
         f"- `current` from s4a vs evidence/07 `pred_e2e_ms`, worst relative gap {worst_cur:.2g}.", "",
         "## 4-hop end-to-end: predicted from s4a vs B direct", "",
         "ms ± bootstrap SE over runs; in brackets the error vs B direct and z = (pred - B) / sqrt(SE_pred^2 + SE_B^2).", "",
         "| arm s | p | B direct | " + " | ".join(names) + " | A / B 4-hop cens % | flag |",
         "|---|---|---|" + "---|" * len(names) + "---|---|"]
    for r in vs_rows:
        if r["hop"] != 4:
            continue
        cells = " | ".join(f"{r[nm + '_ms']:.0f} ± {r[nm + '_se']:.0f} ({pc(r[nm + '_relerr'])}, z {g(r[nm + '_z'], 1)})"
                           for nm in names)
        L.append(f"| {r['arm_s']} | {r['p']} | {r['b_direct_ms']:.0f} ± {r['b_direct_se']:.0f} | {cells} "
                 f"| {100 * r['a_e2e_cens']:.2f} / {100 * r['b_e2e_cens']:.2f} | {'UNRELIABLE' if r['unreliable'] else ''} |")
    L += ["", "## Diagnostics at 4 hops", "",
          "IS failure-span weight mass per hop (about 1 when A holds the all-x spans B needs); chain fed B's own "
          "pieces (chain step is an identity by construction there, up to pooling G over hops; chain elapsed is not).", "",
          "| arm s | p | IS fail mass h1/h2/h3/h4 | B-self chain step | B-self chain elapsed |", "|---|---|---|---|---|"]
    for r in vs_rows:
        if r["hop"] == 4:
            L.append(f"| {r['arm_s']} | {r['p']} | {r['is_fail_mass_hop1_4']} | {pc(r['b_self_chain_step_relerr'])} "
                     f"| {pc(r['b_self_chain_elapsed_relerr'])} |")
    # 2026-11-02: control bytes are charged per poll, not per ms of wait, so no re-attribution here
    # moves them; their transfer rests on the per-poll failure rate not depending on the arm.
    L += ["", "## Control bytes", "",
          "No variant here moves bytes: interest and reply bytes are charged per poll, not per ms of wait, so the "
          "'always x' prediction is evidence/07 `relerr_poll_B` unchanged. The assumption it rests on, that a poll's "
          "failure rate does not depend on its arm given p, is the `dec P(fail)` column below.", ""]
    L += ["", "## Pieces (record-hop weighted over hops 1-4)", "",
          "q1 = P(poll i+1 does not yield | arrival in an arm-x gap); c = P(next poll also fails | this failed poll "
          "chose x), by failed-poll index (step) or by time since t_{i+1} (el; bins below 30 polls borrow the previous "
          "one in the model); G = gap after a failed arm-x poll; seg own / carried / final+fetch = analysis-1 charges "
          "to arm-x decisions per arm-x arrival; dec = decision level (any record).", "",
          "| arm s | p | side | rh | q1 | G ms | c step 1/2/3+ (n) | c el 0/2/5/10/20 s | RTT after fail ms "
          "| post no-fail ms | in-gap ms | seg own / carried / final+fetch | dec P(fail) | dec P(fail after fail) "
          "| dec n_rep>=2 % |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in piece_rows:
        cel = "/".join(g(r[f"c_el_{b:g}s"]) for b in attrib.SEQ_EBINS_S)
        cst = "/".join(g(r[k]) for k in ("c_step1", "c_step2", "c_step3p"))
        nst = "/".join(str(r[k]) for k in ("n_step1", "n_step2", "n_step3p"))
        L.append(f"| {r['arm_s']} | {r['p']} | {r['side']} | {r['n_recordhops']} | {r['q1']:.3f} "
                 f"| {g(r['G_after_fail_ms'], 0)} | {cst} ({nst}) | {cel} | {g(r['rtt_after_fail_ms'], 1)} "
                 f"| {g(r['post_nofail_ms'], 1)} | {g(r['ingap_ms'], 0)} "
                 f"| {r['seg_own_ms']:.0f} / {r['seg_carried_ms']:.0f} / {r['seg_final_fetch_ms']:.0f} "
                 f"| {r['dec_pfail']:.3f} | {g(r['dec_c_after_fail'])} | {100 * r['dec_nrep2_share']:.2f} |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
