"""Run attrib.attribute_run over every run of the given stages and write the ope05 CSVs.

Usage: run_all.py [--stages s2b0p5,...] [--out DIR] [--store DIR] [--agg DIR] [--workers N]
  --store: evalrun store holding <stage>/p<p>/<seed>/ run dirs (read-only).
  --agg:   evalagg output dir holding <stage>/discovery.csv (used for the exactness check).
Outputs (in --out): per_hop_B.csv, predicted_vs_B.csv, state_buckets.csv, checks.csv, fetch_by_inflight.csv,
plus raw_sums.json (merged accumulators; lets tables be re-derived without re-reading the store).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import multiprocessing as mp
import os
import time
from collections import defaultdict

import attrib
from semindex import evalagg

# Defaults for --store / --agg: where the committed CSVs were produced from.
STORE = "/home/lab_shared/semindex-eval"
AGG = "/tmp/claude-1001/-home-lab-shared-cefore-emu-sslab/4677c249-e96f-46cd-a827-e50302f7cbe5/scratchpad/main-agg"
B_STAGES = {"s2b0p5": 0.5, "s2b1p359": 1.359, "s2b3p695": 3.695, "s2b10p043": 10.043, "s2c": 2.0}
ARMS = [0.5, 1.359, 3.695, 10.043]
PS = [0.0, 0.1, 0.2, 0.3, 0.4]
CENSOR_FLAG = 0.02  # a side whose censored share of pairs/hops exceeds this is flagged unreliable

DISC = {}  # (stage, p, seed) -> {(o, s, n): latency or None}; filled before fork, inherited by workers


def load_disc(stage, agg=AGG):
    """evalagg discovery.csv for `stage` (under `agg`) into DISC (p parsed as float on both sides)."""
    with open(f"{agg}/{stage}/discovery.csv") as f:
        for r in csv.DictReader(f):
            key = (stage, float(r["p"]), int(r["seed"]))
            DISC.setdefault(key, {})[(r["origin"], int(r["seq"]), r["node"])] = (
                None if r["censored"] == "true" else float(r["latency_ms"]))


def work(args):
    """Pool task: attribute one run, tagged with its stage and p so the parent can stratify the merge.

    Needs no paths of its own: the run dir travels in `args` and DISC is inherited through fork.
    """
    stage, p, seed, run_dir = args
    return stage, p, attrib.attribute_run(stage, p, seed, run_dir, DISC.get((stage, p, seed)))


def merge(dst, src, maxkeys=("exact_maxerr",)):
    """Fold one run's accumulator into the running total: sums add, except `maxkeys` which keep the max."""
    for k, v in src.items():
        if isinstance(v, dict):
            d = dst.setdefault(k, {})
            for f, x in v.items():
                d[f] = d.get(f, 0) + x
        elif k in maxkeys:
            dst[k] = max(dst.get(k, 0), v)
        else:
            dst[k] = dst.get(k, 0) + v


def collapse(acc, keep):
    """Sum an accumulator over every key dimension not in `keep` (indices into the stratum tuple)."""
    out = defaultdict(lambda: defaultdict(float))
    for k, v in acc.items():
        kk = tuple(k[i] for i in keep)
        for f, x in v.items():
            out[kk][f] += x
    return out


def ratio(a, b):
    """a/b, with NaN (written as an empty cell) where there is nothing to divide by."""
    return a / b if b else float("nan")


def fmt(x):
    """CSV cell text: floats to 6 significant digits, NaN as empty; everything else unchanged."""
    if isinstance(x, float):
        return "" if math.isnan(x) else f"{x:.6g}"
    return x


def write_csv(path, rows):
    """Write dict rows to `path`; column order is the first row's key order."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})


# stratum tuple layout (attrib.stratum): 0 stage, 1 p, 2 arm, 3 hop, 4 status, 5 ctob, 6 prev_arm, 7 infl0


# (tag, stratum indices kept, numerator fields) for run-clustered SEs
RUNSE = [("bucket_wait", (0, 1, 4, 5, 2), ("wait_u",)), ("bucket_wait_pooled", (0, 4, 5, 2), ("wait_u",)),
         ("hop_wait", (0, 1, 2, 3), ("wait_u",)), ("hop_wf", (0, 1, 2, 3), ("wait_u", "fetch_u"))]


def ratio_se(d):
    """Delta-method SE of sum(W)/sum(N) with runs as clusters: sqrt(sum_r (W_r - R N_r)^2) / sum(N).

    `d` is {run_idx: (W, N)}; runs absent from `d` contribute (0, 0) and so drop out of the sum.
    """
    lst = list(d.values())
    W = sum(w for w, _ in lst)
    N = sum(n for _, n in lst)
    if not N or len(lst) < 2:
        return float("nan")
    R = W / N
    return math.sqrt(sum((w - R * n) ** 2 for w, n in lst) * len(lst) / (len(lst) - 1)) / N


def e2e_se(per_hop_dicts):
    """SE of sum_h W_h/N_h, the same runs feeding every hop (run-clustered delta method).

    Each element is {run_idx: (W, N)} for one hop; a run's influence is summed across hops, so the
    positive within-run correlation between hops is kept rather than assumed away.
    """
    runs = defaultdict(float)
    for d in per_hop_dicts:
        W = sum(w for w, _ in d.values())
        N = sum(n for _, n in d.values())
        if not N:
            return float("nan")
        R = W / N
        for idx, (w, n) in d.items():
            runs[idx] += (w - R * n) / N
    vals = list(runs.values())
    if len(vals) < 2:
        return float("nan")
    return math.sqrt(sum(v * v for v in vals) * len(vals) / (len(vals) - 1))


def hop_metrics(h, d):
    """Per-record and per-time metrics from one collapsed (hop sums, decision sums) pair."""
    n_u, n_c = h.get("n_u", 0), h.get("n_c", 0)
    poll_B = d.get("int_B", 0) + d.get("rep_B", 0)
    return dict(
        n_gaps=int(d.get("n_gap", 0)), n_recordhops=int(n_u), n_recordhops_censored=int(n_c),
        censored_share=ratio(n_c, n_u + n_c),
        wait_ms=ratio(h.get("wait_u", 0), n_u), fetch_ms=ratio(h.get("fetch_u", 0), n_u),
        # in-gap = arrival -> end of the credited gap (or learned, if earlier); post = learned - gap end
        # (reply RTT/processing of poll i+1, plus spillover when poll i+1 did not cover).
        ingap_wait_ms=ratio(h.get("wait_u", 0) - h.get("post_u", 0), n_u), post_ms=ratio(h.get("post_u", 0), n_u),
        batch_mean=ratio(h.get("batch_u", 0), n_u),
        wait_fetch_ms=ratio(h.get("wait_u", 0) + h.get("fetch_u", 0), n_u),
        spill_share_of_wait=ratio(h.get("spill_u", 0), h.get("wait_u", 0)),
        spill_hop_share=ratio(h.get("n_spill", 0), n_u),
        poll_B_per_recordhop=ratio(poll_B, n_u), fetch_B_per_recordhop=ratio(h.get("fbytes_u", 0), n_u),
        backlog_rate=ratio(h.get("waitcap_all", 0), d.get("gap_ms", 0)),
        gaps_per_s=ratio(d.get("n_gap", 0), d.get("gap_ms", 0) / 1000.0),
        noreply_share=ratio(d.get("n_norep", 0), d.get("n_gap", 0)),
    )


def main():
    """Attribute every run of the requested stages, merge, and write the ope05 CSVs plus raw sums."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="s2b0p5,s2b1p359,s2b3p695,s2b10p043,s2c,s2a")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--agg", default=AGG)
    a = ap.parse_args()
    stages = a.stages.split(",")
    t_start = time.time()
    jobs = []
    for st in stages:
        load_disc(st, a.agg)
        for r in evalagg.discover_runs(a.store, st):
            jobs.append((st, r.p, r.seed, r.run_dir))
    hop, dec, pair, gapdiff = {}, {}, defaultdict(dict), defaultdict(lambda: [0, 0.0, 0.0])
    chk = defaultdict(dict)
    perrun = {tag: defaultdict(dict) for tag, _k, _f in RUNSE}  # tag -> key -> {run_idx: (W, N)}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for run_idx, (st, pp_, res) in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            merge(hop, res["hop"])
            merge(dec, res["dec"])
            merge(chk[st], res["chk"])
            for k, v in res["pair"].items():
                pair[(st, pp_)][k] = pair[(st, pp_)].get(k, 0) + v
            # per-run (W, N) for run-clustered SEs of ratio estimators (records in one gap share
            # their wait, and runs share link schedules, so record-level SEs would be far too small)
            for tag, keep, fields in RUNSE:
                tmp = defaultdict(lambda: [0.0, 0.0])
                for k, v in res["hop"].items():
                    if "n_u" not in v:
                        continue
                    kk = tuple(k[i] for i in keep)
                    tmp[kk][0] += sum(v.get(f_, 0) for f_ in fields)
                    tmp[kk][1] += v["n_u"]
                for kk, wn in tmp.items():
                    perrun[tag][kk][run_idx] = tuple(wn)
            for arm, (n, s, mx) in res["gapdiff"].items():
                g = gapdiff[(st, arm)]
                g[0] += n
                g[1] += s
                g[2] = max(g[2], mx)
    elapsed = time.time() - t_start

    with gzip.open(f"{a.out}/raw_sums.json.gz", "wt") as f:
        json.dump({"hop": [[list(k), v] for k, v in hop.items()], "dec": [[list(k), v] for k, v in dec.items()],
                   "chk": chk, "pair": [[list(s), [[list(k), v] for k, v in d.items()]] for s, d in pair.items()],
                   "elapsed_s": elapsed}, f)

    # ---- checks.csv ----------------------------------------------------------------
    rows = []
    for st in stages:
        c = chk[st]
        base = {k: v for k, v in c.items() if "|" not in k}
        for k in sorted(base):
            rows.append(dict(stage=st, p="all", check=k, value=base[k]))
        rule_tot = defaultdict(float)
        for k, v in c.items():
            if "|" in k:
                rk, pp = k.split("|")
                rows.append(dict(stage=st, p=pp, check=rk, value=v))
                rule_tot[pp] += v
        for pp, tot in rule_tot.items():
            for k, v in c.items():
                if k.endswith("|" + pp):
                    rows.append(dict(stage=st, p=pp, check=k.split("|")[0] + "_share", value=v / tot))
        n = c.get("exact_n", 0)
        rows.append(dict(stage=st, p="all", check="exact_share_within_1ms", value=ratio(c.get("exact_within1", 0), n)))
    # spillover + censoring by p (all strata summed)
    hs = collapse(hop, (0, 1))
    for (st, pp), h in sorted(hs.items()):
        rows.append(dict(stage=st, p=pp, check="spill_share_of_wait", value=ratio(h["spill_u"], h["wait_u"])))
        rows.append(dict(stage=st, p=pp, check="post_gap_share_of_wait", value=ratio(h["post_u"], h["wait_u"])))
        rows.append(dict(stage=st, p=pp, check="hops_uncensored", value=h["n_u"]))
        rows.append(dict(stage=st, p=pp, check="hops_censored", value=h.get("n_c", 0)))
    for (st, arm), (n, s, mx) in sorted(gapdiff.items()):
        rows.append(dict(stage=st, p="all", check=f"gap_minus_applied_mean_ms|arm={arm}", value=ratio(s, n)))
        rows.append(dict(stage=st, p="all", check=f"gap_minus_applied_maxabs_ms|arm={arm}", value=mx))
    write_csv(f"{a.out}/checks.csv", rows)

    # ---- per_hop_B.csv (B stages + s2c), per p x hop ---------------------------------
    hB = collapse(hop, (0, 1, 2, 3))
    dB = collapse(dec, (0, 1, 2, 3))
    rows = []
    for st in stages:
        if st not in B_STAGES:
            continue
        for pp in PS:
            for hp in range(1, 5):
                k = (st, pp, B_STAGES[st], hp)
                if k not in hB:
                    continue
                m = hop_metrics(hB[k], dB.get(k, {}))
                pr = pair[(st, pp)]
                rows.append(dict(stage=st, chosen_s=B_STAGES[st], p=pp, hop=hp, **m,
                                 wait_over_chosen=m["wait_ms"] / (B_STAGES[st] * 1000.0),
                                 e2e_pairs=int(pr.get(("n", hp), 0)), e2e_pairs_censored=int(pr.get(("c", hp), 0)),
                                 e2e_mean_ms=ratio(pr.get(("lat", hp), 0), pr.get(("n", hp), 0))))
    if rows:
        write_csv(f"{a.out}/per_hop_B.csv", rows)

    # ---- predicted_vs_B.csv ----------------------------------------------------------
    if "s2a" in stages:
        hA = collapse(hop, (0, 1, 2, 3))
        dA = collapse(dec, (0, 1, 2, 3))
        bstage = {v: k for k, v in B_STAGES.items() if k != "s2c"}
        rows = []
        for arm in ARMS:
            for pp in PS:
                cum_a = cum_b = 0.0
                cens = {"a": [0, 0], "b": [0, 0]}
                for hp in range(1, 5):
                    ka, kb = ("s2a", pp, arm, hp), (bstage[arm], pp, arm, hp)
                    ma = hop_metrics(hA.get(ka, {}), dA.get(ka, {}))
                    mb = hop_metrics(hB.get(kb, {}), dB.get(kb, {}))
                    cum_a += ma["wait_fetch_ms"]
                    cum_b += mb["wait_fetch_ms"]
                    for side, mm in (("a", ma), ("b", mb)):
                        cens[side][0] += mm["n_recordhops_censored"]
                        cens[side][1] += mm["n_recordhops"] + mm["n_recordhops_censored"]
                    ca, cb = ratio(*cens["a"]), ratio(*cens["b"])
                    # 2026-10-29 review fix: the pooled hop share above under-counts censoring at
                    # this hop distance (a record that never reached the upstream drops out of the
                    # downstream hops' counts), so the end-to-end cell is also judged by the share of
                    # (origin, seq, node) pairs at exactly this distance that were censored.
                    ea = ratio(pair[("s2a", pp)].get(("c", hp), 0),
                               pair[("s2a", pp)].get(("n", hp), 0) + pair[("s2a", pp)].get(("c", hp), 0))
                    eb = ratio(pair[(bstage[arm], pp)].get(("c", hp), 0),
                               pair[(bstage[arm], pp)].get(("n", hp), 0) + pair[(bstage[arm], pp)].get(("c", hp), 0))
                    rows.append(dict(
                        arm_s=arm, p=pp, hop=hp,
                        a_n_recordhops=ma["n_recordhops"], a_censored=ma["n_recordhops_censored"],
                        a_wait_ms=ma["wait_ms"], a_fetch_ms=ma["fetch_ms"], a_wait_fetch_ms=ma["wait_fetch_ms"],
                        a_ingap_wait_ms=ma["ingap_wait_ms"], a_post_ms=ma["post_ms"],
                        a_batch=ma["batch_mean"],
                        a_backlog_rate=ma["backlog_rate"], a_gaps_per_s=ma["gaps_per_s"],
                        a_spill_share=ma["spill_share_of_wait"], a_poll_B_per_rh=ma["poll_B_per_recordhop"],
                        a_fetch_B_per_rh=ma["fetch_B_per_recordhop"],
                        b_n_recordhops=mb["n_recordhops"], b_censored=mb["n_recordhops_censored"],
                        b_wait_ms=mb["wait_ms"], b_fetch_ms=mb["fetch_ms"], b_wait_fetch_ms=mb["wait_fetch_ms"],
                        b_ingap_wait_ms=mb["ingap_wait_ms"], b_post_ms=mb["post_ms"],
                        b_batch=mb["batch_mean"],
                        b_backlog_rate=mb["backlog_rate"], b_gaps_per_s=mb["gaps_per_s"],
                        b_spill_share=mb["spill_share_of_wait"], b_poll_B_per_rh=mb["poll_B_per_recordhop"],
                        b_fetch_B_per_rh=mb["fetch_B_per_recordhop"],
                        a_wait_se=ratio_se(perrun["hop_wait"].get(ka, {})),
                        b_wait_se=ratio_se(perrun["hop_wait"].get(kb, {})),
                        a_wait_fetch_se=ratio_se(perrun["hop_wf"].get(ka, {})),
                        b_wait_fetch_se=ratio_se(perrun["hop_wf"].get(kb, {})),
                        relerr_wait=ratio(ma["wait_ms"] - mb["wait_ms"], mb["wait_ms"]),
                        relerr_ingap_wait=ratio(ma["ingap_wait_ms"] - mb["ingap_wait_ms"], mb["ingap_wait_ms"]),
                        relerr_fetch=ratio(ma["fetch_ms"] - mb["fetch_ms"], mb["fetch_ms"]),
                        relerr_poll_B=ratio(ma["poll_B_per_recordhop"] - mb["poll_B_per_recordhop"],
                                            mb["poll_B_per_recordhop"]),
                        pred_e2e_ms=cum_a,
                        pred_e2e_se=e2e_se([perrun["hop_wf"].get(("s2a", pp, arm, x), {}) for x in range(1, hp + 1)]),
                        b_e2e_sum_se=e2e_se([perrun["hop_wf"].get((bstage[arm], pp, arm, x), {})
                                             for x in range(1, hp + 1)]), b_e2e_sum_of_hop_means_ms=cum_b,
                        b_e2e_direct_ms=ratio(pair[(bstage[arm], pp)].get(("lat", hp), 0),
                                              pair[(bstage[arm], pp)].get(("n", hp), 0)),
                        b_e2e_pairs=int(pair[(bstage[arm], pp)].get(("n", hp), 0)),
                        b_e2e_pairs_censored=int(pair[(bstage[arm], pp)].get(("c", hp), 0)),
                        relerr_e2e_vs_sum=ratio(cum_a - cum_b, cum_b),
                        relerr_e2e_vs_direct=ratio(cum_a - ratio(pair[(bstage[arm], pp)].get(("lat", hp), 0),
                                                         pair[(bstage[arm], pp)].get(("n", hp), 0)),
                                                   ratio(pair[(bstage[arm], pp)].get(("lat", hp), 0),
                                                         pair[(bstage[arm], pp)].get(("n", hp), 0))),
                        cum_censored_share_a=ca, cum_censored_share_b=cb,
                        unreliable=int(ca > CENSOR_FLAG or cb > CENSOR_FLAG
                                       or ea > CENSOR_FLAG or eb > CENSOR_FLAG),
                        a_e2e_pairs=int(pair[("s2a", pp)].get(("n", hp), 0)),
                        a_e2e_pairs_censored=int(pair[("s2a", pp)].get(("c", hp), 0)),
                        e2e_censored_share_a=ea, e2e_censored_share_b=eb,
                    ))
        write_csv(f"{a.out}/predicted_vs_B.csv", rows)

        # ---- state_buckets.csv: bucket (status x ctob) x p (incl. pooled) x arm ----------
        hS = collapse(hop, (0, 1, 4, 5, 2))
        dS = collapse(dec, (0, 1, 4, 5, 2))
        dPrev = collapse(dec, (0, 1, 4, 5, 6))
        hSp = collapse(hop, (0, 4, 5, 2))
        dSp = collapse(dec, (0, 4, 5, 2))
        dPrevp = collapse(dec, (0, 4, 5, 6))
        rows = []
        buckets = sorted({(k[2], k[3]) for k in dS if k[0] == "s2a"})
        for pp in PS + ["all"]:
            for (stt, cb) in buckets:
                ms = {}
                for arm in ARMS:
                    if pp == "all":
                        h, d = hSp.get(("s2a", stt, cb, arm), {}), dSp.get(("s2a", stt, cb, arm), {})
                    else:
                        h, d = hS.get(("s2a", pp, stt, cb, arm), {}), dS.get(("s2a", pp, stt, cb, arm), {})
                    ms[arm] = hop_metrics(h, d)
                if pp == "all":
                    prev = {pa: dPrevp.get(("s2a", stt, cb, pa), {}).get("n_gap", 0) for pa in ARMS + ["none"]}
                else:
                    prev = {pa: dPrev.get(("s2a", pp, stt, cb, pa), {}).get("n_gap", 0) for pa in ARMS + ["none"]}
                tot_prev = sum(prev.values())
                ranks = {}
                for lam in (0.0, 1.0, 10.0):
                    sc = {arm: ms[arm]["wait_ms"] + lam * ms[arm]["poll_B_per_recordhop"] for arm in ARMS
                          if ms[arm]["n_recordhops"] > 0}
                    ranks[lam] = ">".join(str(x) for x in sorted(sc, key=sc.get))  # best first
                for arm in ARMS:
                    m = ms[arm]
                    rows.append(dict(
                        p=pp, last_poll_status=stt, cto_bucket=cb, arm_s=arm,
                        n_gaps=m["n_gaps"], n_recordhops=m["n_recordhops"],
                        n_recordhops_censored=m["n_recordhops_censored"],
                        wait_ms=m["wait_ms"],
                        wait_se=ratio_se(perrun["bucket_wait_pooled" if pp == "all" else "bucket_wait"].get(
                            ("s2a", stt, cb, arm) if pp == "all" else ("s2a", pp, stt, cb, arm), {})),
                        fetch_ms=m["fetch_ms"], spill_share_of_wait=m["spill_share_of_wait"],
                        poll_B_per_recordhop=m["poll_B_per_recordhop"], backlog_rate=m["backlog_rate"], gaps_per_s=m["gaps_per_s"],
                        noreply_share=m["noreply_share"],
                        **{f"prev_arm_share_{pa}": ratio(prev[pa], tot_prev) for pa in ARMS + ["none"]},
                        rank_wait=ranks[0.0], rank_l1=ranks[1.0], rank_l10=ranks[10.0],
                    ))
        write_csv(f"{a.out}/state_buckets.csv", rows)

        # fetch split by in_flight at decision (confound check), s2a + B, per arm x p
        hF = collapse(hop, (0, 1, 2, 7))
        rows = []
        for (st, pp, arm, infl0), h in sorted(hF.items(), key=lambda kv: str(kv[0])):
            rows.append(dict(stage=st, p=pp, arm_s=arm, in_flight_zero=int(infl0), n_recordhops=int(h.get("n_u", 0)),
                             wait_ms=ratio(h.get("wait_u", 0), h.get("n_u", 0)),
                             fetch_ms=ratio(h.get("fetch_u", 0), h.get("n_u", 0)),
                             batch_mean=ratio(h.get("batch_u", 0), h.get("n_u", 0))))
        write_csv(f"{a.out}/fetch_by_inflight.csv", rows)
    print(f"runs={len(jobs)} elapsed_s={elapsed:.1f}")


if __name__ == "__main__":
    main()
