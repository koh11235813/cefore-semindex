"""Tables for the s5 policy runs from perrun.csv (live) and replay_perseed.csv (replay); writes CSVs, prints markdown.

Live definitions (per (stage, p), seeds as the bootstrap unit, 200 multinomial reps):
  delay      = seq_model.direct(S, 4): sum of discovery.csv latency over uncensored 4-hop pairs / their count
  delay_seedmean = mean over seeds of each seed's direct value (frontier.py's convention; used for the [4] hull)
  cens4      = censored 4-hop pairs / all 4-hop pairs
  B_raw      = in-window state-request + reply bytes (attrib int_B + rep_B) / uncensored record-hops, pooled over hops
  B_hopmean  = mean over hops of the per-hop ratio (evidence/05 per_hop_B poll_B_per_recordhop, as validate.py averages)
  B_norm     = B_raw with every state name rebased to group length 14.92 (s4a, seeds 1-100: what makes INT_B = 50.9)
  B_s4aseed  = rebased to the same seed's s4a group (len(stage) - 3 shorter per name)
  B_const    = (polls * 50.9 + answered polls * REP_B[p]) / record-hops (validate2.py's "corrected"; drops live payload)
  polls      = polls.csv polls_sent per run (8 directed pairs, window [t0+10, t0+130] s inclusive)
  waste      = per-run mean of 1 - state_rx/polls_sent (the replay's convention); waste_pooled = 1 - sum rx / sum polls
"""
import csv
import math
import os
from collections import defaultdict

import numpy as np

import replay as R  # copied unchanged into this directory, see the README

OUT = os.path.dirname(os.path.abspath(__file__))
EV07 = os.path.join(OUT, os.pardir, "07")
VC = OUT  # policy_table.csv, the published replay table, is copied here unchanged
REF_G = 14 + 0.90 + 0.02  # s4a_p0x0_seedN over seeds 1-100: 9 one-digit, 90 two-digit, 1 three-digit seeds
ARM = {"s5b0p5": 0.5, "s5b1p359": 1.359, "s5b3p695": 3.695, "s5b10p043": 10.043}
S4B = {"s5b0p5": "s4b0p5", "s5b1p359": "s4b1p359", "s5b3p695": "s4b3p695", "s5b10p043": "s4b10p043"}
LIVE2REP = {"s5short": "shorten (10.043, 3.695)", "s5wait": "slotblind 10.043", "s5b0p5": "fixed 0.5",
            "s5b1p359": "fixed 1.359", "s5b3p695": "fixed 3.695", "s5b10p043": "fixed 10.043"}
H = (1, 2, 3, 4)
REPS = 200


def load_live():
    g = defaultdict(list)
    for r in csv.DictReader(open(f"{OUT}/perrun.csv")):
        st = r["stage"]
        d = {k: float(v) for k, v in r.items() if k != "stage"}
        d["stage"] = st
        g[(st, d["p"])].append(d)
    for v in g.values():
        v.sort(key=lambda r: r["seed"])
    return g


def per_seed_cols(runs, stage, p):
    """Matrix of per-seed sums the ratio statistics are built from (rows = seeds)."""
    dl = len(stage) - 3
    cols = []
    for r in runs:
        nu = [r[f"nu{h}"] for h in H]
        pb = [r[f"intB{h}"] + r[f"repB{h}"] for h in H]
        nn = [r[f"ngap{h}"] + r[f"nrep{h}"] for h in H]   # names on the wire: one per Interest, one per reply
        dg = r["group_len"] - REF_G
        ans = sum(r[f"ngap{h}"] - r[f"nnorep{h}"] for h in H)
        cols.append([r["lat4"], r["n4"], r["c4"], sum(nu), sum(pb), sum(pb) - dg * sum(nn), sum(pb) - dl * sum(nn),
                     sum(r[f"ngap{h}"] for h in H) * R.INT_B + ans * R.REP_B[p], r["polls_sent"], r["state_rx"],
                     sum(r[f"ngap{h}"] for h in H), *nu, *pb, *[pb[i] - dg * nn[i] for i in range(4)]])
    return np.array(cols)


def stats(M, w):
    """All live statistics from the seed matrix with weights w."""
    s = w @ M
    lat, n, c, nu, pb, pbn, pbs, pbc, pol, rx, ngap = s[:11]
    nuh, pbh, pbnh = s[11:15], s[15:19], s[19:23]
    seed_delay = M[:, 0] / M[:, 1]
    seed_waste = 1 - M[:, 9] / M[:, 8]
    ok = ~np.isnan(seed_delay)
    return dict(delay=lat / n, delay_seedmean=np.sum(w[ok] * seed_delay[ok]) / np.sum(w[ok]), cens4=c / (n + c),
                B_raw=pb / nu, B_norm=pbn / nu, B_s4aseed=pbs / nu, B_const=pbc / nu,
                B_hopmean=np.mean(pbh / nuh), B_hopmean_norm=np.mean(pbnh / nuh),
                polls=pol / w.sum(), waste=np.sum(w * seed_waste) / w.sum(), waste_pooled=1 - rx / pol,
                polls_per_rh=ngap / nu)


def boot(M, reps, rng):
    n = len(M)
    base = stats(M, np.ones(n))
    vals = defaultdict(list)
    for _ in range(reps):
        w = rng.multinomial(n, [1 / n] * n).astype(float)
        for k, v in stats(M, w).items():
            vals[k].append(v)
    return base, {k: float(np.nanstd(v, ddof=1)) for k, v in vals.items()}


def hull_delay(pts, b):
    """frontier.py's hull_delay, copied unchanged: lower convex envelope of (bytes, delay) at b."""
    pts = sorted(pts)
    hull = []
    for q in pts:
        while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (q[0] - hull[-2][0]) >= (q[1] - hull[-2][1]) * (hull[-1][0] - hull[-2][0]):
            hull.pop()
        hull.append(q)
    xs = [h[0] for h in hull]
    if b < xs[0] or b > xs[-1]:
        return np.nan
    return float(np.interp(b, xs, [h[1] for h in hull]))


def f(x, nd=0):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def main():
    live = load_live()
    rng = np.random.default_rng(20261105)
    L = []
    main_rows = {}
    mats = {}
    for (st, p), runs in sorted(live.items()):
        M = per_seed_cols(runs, st, p)
        mats[(st, p)] = (M, [r["seed"] for r in runs])
        base, se = boot(M, REPS, rng)
        extra = dict(resched_ans=np.mean([r["resched_ans"] for r in runs]), resched_to=np.mean([r["resched_to"] for r in runs]),
                     clamped=sum(r["clamped"] for r in runs), late=sum(r["late"] for r in runs),
                     censor_disagree=sum(r["exact_censor_disagree"] for r in runs),
                     within1=sum(r["exact_within1"] for r in runs) / max(sum(r["exact_n"] for r in runs), 1),
                     n_runs=len(runs))
        row = dict(stage=st, p=p, **{k: base[k] for k in base}, **{k + "_se": se[k] for k in se}, **extra)
        main_rows[(st, p)] = row
    keys = list(next(iter(main_rows.values())).keys())
    with open(f"{OUT}/live_main.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, keys)
        w.writeheader()
        w.writerows(main_rows.values())

    order = ["s5short", "s5wait", "s5b0p5", "s5b1p359", "s5b3p695", "s5b10p043", "s5c"]
    L += ["## Main table (live)", "",
          "| stage | p | 4-hop delay ms ± SE | seed-mean | cens4 % | B/rh raw | B/rh norm ± SE | B/rh s4a-seed | B/rh const | B hop-mean raw | polls/run ± SE | polls/rh | waste run-mean ± SE | waste pooled | resched ans/to per run | clamped | late |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for st in order:
        for p in (0.0, 0.1, 0.2, 0.3, 0.4):
            r = main_rows.get((st, p))
            if not r:
                continue
            L.append(f"| {st} | {p} | {f(r['delay'])} ± {f(r['delay_se'])} | {f(r['delay_seedmean'])} | {f(100*r['cens4'],2)} | {f(r['B_raw'],1)} | {f(r['B_norm'],1)} ± {f(r['B_norm_se'],1)} | {f(r['B_s4aseed'],1)} | {f(r['B_const'],1)} | {f(r['B_hopmean'],1)} | {f(r['polls'],1)} ± {f(r['polls_se'],1)} | {f(r['polls_per_rh'],4)} | {f(r['waste'],3)} ± {f(r['waste_se'],3)} | {f(r['waste_pooled'],3)} | {f(r['resched_ans'],1)}/{f(r['resched_to'],1)} | {int(r['clamped'])} | {int(r['late'])} |")

    # ---- calibration: s5b* vs s4b* (evidence/07)
    s4 = {(float(r["arm_s"]), float(r["p"])): r for r in csv.DictReader(open(f"{EV07}/seq_vs_B.csv")) if r["hop"] == "4"}
    ph = defaultdict(list)
    for r in csv.DictReader(open(f"{EV07}/per_hop_B.csv")):
        ph[(r["stage"], float(r["p"]))].append(float(r["poll_B_per_recordhop"]))
    L += ["", "## Calibration: s5b* (code 7d5cd80) vs s4b* (evidence/07)", "",
          "| arm | p | s5b delay ± SE | s4b b_direct ± SE | diff ms | rel | z | s5b B hop-mean raw | s4b B hop-mean raw |", "|---|---|---|---|---|---|---|---|---|"]
    cal = []
    for st, x in ARM.items():
        for p in (0.0, 0.1, 0.2, 0.3, 0.4):
            r = main_rows[(st, p)]
            q = s4[(x, p)]
            bd, bse = float(q["b_direct_ms"]), float(q["b_direct_se"])
            d = r["delay"] - bd
            z = d / math.hypot(r["delay_se"], bse)
            b4 = float(np.mean(ph[(S4B[st], p)]))
            # s4b names rebased the same way: group len(s4bXX) + 9 + seed digits; 2026-11-05 s4b seeds 1-20 (mean 1.55 digits)
            cal.append(dict(arm=x, p=p, s5b=r["delay"], s5b_se=r["delay_se"], s4b=bd, s4b_se=bse, diff=d, rel=d / bd, z=z,
                            s5b_B_hopmean=r["B_hopmean"], s4b_B_hopmean=b4))
            L.append(f"| {x} | {p} | {f(r['delay'])} ± {f(r['delay_se'])} | {f(bd)} ± {f(bse)} | {d:+.0f} | {100*d/bd:+.1f}% | {z:+.1f} | {f(r['B_hopmean'],1)} | {f(b4,1)} |")
    with open(f"{OUT}/calibration_s5b_vs_s4b.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, cal[0].keys())
        w.writeheader()
        w.writerows(cal)

    # ---- replay
    rp = defaultdict(dict)
    for r in csv.DictReader(open(f"{OUT}/replay_perseed.csv")):
        rp[(r["set"], r["policy"], float(r["p"]))][int(r["seed"])] = {k: float(r[k]) for k in ("delay_ms", "bytes_rh", "waste", "polls", "cens")}
    pub = {}
    for r in csv.DictReader(open(f"{VC}/policy_table.csv")):
        if r["group"] in ("fixed", "2b-shorten", "2b-engine", "4b-slotblind"):
            pub[(r["name"], float(r["p"]))] = r
    L += ["", "## Replay: published (policy_table.csv, s4a seeds 1-100) vs our re-run of the same set-up (pub-rerun)", "",
          "| policy | p | published D ± se | B/rh | waste | cens | [4] | [c] | rerun D | rerun B/rh | rerun waste | rerun polls/run |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ["shorten (10.043, 3.695)", "shorten-engine (10.043, 3.695)", "slotblind 10.043", "fixed 10.043", "fixed 3.695", "fixed 5.0"]:
        for p in (0.0, 0.2, 0.4):
            q = pub.get((name, p))
            rr = rp[("pub", name, p)]
            mm = {k: np.nanmean([v[k] for v in rr.values()]) for k in ("delay_ms", "bytes_rh", "waste", "polls")}
            if q is None:
                L.append(f"| {name} | {p} | not published | | | | | | {f(mm['delay_ms'])} | {f(mm['bytes_rh'],1)} | {f(mm['waste'],3)} | {f(mm['polls'],1)} |")
            else:
                L.append(f"| {name} | {p} | {f(float(q['delay_ms']))} ± {f(float(q['se_ms']))} | {f(float(q['bytes_rh']),1)} | {f(float(q['waste']),3)} | {f(float(q['censored']),4)} | {q['d4_ms'] and f(float(q['d4_ms']))} | {q['dc_ms'] and f(float(q['dc_ms']))} | {f(mm['delay_ms'])} | {f(mm['bytes_rh'],1)} | {f(mm['waste'],3)} | {f(mm['polls'],1)} |")

    # paired comparison
    L += ["", "## Live vs replay", "",
          "paired = replay on the same (p, seed) schedules, seeds 1-20, 10 reps; z uses the SE of the per-seed difference "
          "(seed-level, n = 20). vs published: z for delay only = (live - pub) / sqrt(SE_live^2 + se_pub^2). "
          "Live delay here is the per-seed mean (the replay's convention); bytes = B_norm.", "",
          "| live stage | p | metric | live | paired replay | live-paired | rel | z(paired) | published | live-pub | rel | z(pub) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    cmp_rows = []
    for st in ["s5short", "s5wait", "s5b10p043", "s5b3p695", "s5b1p359", "s5b0p5"]:
        name = LIVE2REP[st]
        for p in (0.0, 0.2, 0.4):
            M, seeds = mats[(st, p)]
            rr = rp[("paired", name, p)]
            lv = {"delay": M[:, 0] / M[:, 1], "bytes": M[:, 5] / M[:, 3], "polls": M[:, 8], "waste": 1 - M[:, 9] / M[:, 8]}
            rv = {"delay": np.array([rr[int(s)]["delay_ms"] for s in seeds]), "bytes": np.array([rr[int(s)]["bytes_rh"] for s in seeds]),
                  "polls": np.array([rr[int(s)]["polls"] for s in seeds]), "waste": np.array([rr[int(s)]["waste"] for s in seeds])}
            q = pub.get((name, p))
            pubv = {"delay": float(q["delay_ms"]), "bytes": float(q["bytes_rh"]), "waste": float(q["waste"])} if q else {}
            pubse = float(q["se_ms"]) if q else float("nan")
            pr = rp[("pub", name, p)]
            pubv["polls"] = np.mean([v["polls"] for v in pr.values()])  # not published: our re-run of the published set-up
            for met in ("delay", "bytes", "polls", "waste"):
                a, b = lv[met], rv[met]
                ok = ~np.isnan(a) & ~np.isnan(b)
                dd = a[ok] - b[ok]
                se_d = np.std(dd, ddof=1) / math.sqrt(len(dd))
                z = dd.mean() / se_d if se_d > 0 else float("nan")
                lm, rm = np.nanmean(a), np.nanmean(b)
                pv = pubv.get(met, float("nan"))
                zp = float("nan")
                if met == "delay":
                    lse = np.nanstd(a, ddof=1) / math.sqrt(np.sum(~np.isnan(a)))
                    zp = (lm - pv) / math.hypot(lse, pubse)
                nd = {"delay": 0, "bytes": 2, "polls": 1, "waste": 3}[met]
                cmp_rows.append(dict(stage=st, p=p, metric=met, live=lm, paired=rm, diff=dd.mean(), rel=dd.mean() / rm, z_paired=z,
                                     published=pv, diff_pub=lm - pv, rel_pub=(lm - pv) / pv if pv == pv and pv else float("nan"), z_pub=zp))
                c = cmp_rows[-1]
                L.append(f"| {st} | {p} | {met} | {f(lm,nd)} | {f(rm,nd)} | {c['diff']:+.{nd}f} | {'-' if not np.isfinite(c['rel']) else format(100*c['rel'], '+.1f') + '%'} | {f(z,1)} | {f(pv,nd)}{'*' if met=='polls' else ''} | {f(c['diff_pub'],nd)} | {'-' if c['rel_pub']!=c['rel_pub'] else f'{100*c[chr(114)+chr(101)+chr(108)+chr(95)+chr(112)+chr(117)+chr(98)]:+.1f}%'} | {f(zp,1)} |")
    with open(f"{OUT}/live_vs_replay.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, cmp_rows[0].keys())
        w.writeheader()
        w.writerows(cmp_rows)

    # ---- [4] live and paired replay
    L += ["", "## [4]: policy delay minus lower hull of the 4 fixed arms at the policy's bytes (ms), seed bootstrap 400", "",
          "| policy | p | live [4] (CI95) | live D seed-mean | live B norm | hull x-range | paired-replay [4] seeds 1-20 (CI95) | published [4] (CI95) | published [c] (CI95) |", "|---|---|---|---|---|---|---|---|---|"]
    fr_rows = []
    brng = np.random.default_rng(5)
    for st in ("s5short", "s5wait"):
        for p in (0.0, 0.2, 0.4):
            # live: per-seed (delay, bytes) for the policy and each arm, paired by seed (link schedules are seed-only)
            Mp, sp = mats[(st, p)]
            arms = {a: mats[(a, p)] for a in ARM}
            for a, (_m, sa) in arms.items():
                assert sa == sp, (a, p)
            def vals(M):
                return np.c_[M[:, 0] / M[:, 1], M[:, 5] / M[:, 3]]
            V = {"pol": vals(Mp), **{a: vals(m) for a, (m, _s) in arms.items()}}
            rr = {k: rp[("paired", n, p)] for k, n in [("pol", LIVE2REP[st])] + [(a, LIVE2REP[a]) for a in ARM]}
            VR = {k: np.array([[rr[k][int(s)]["delay_ms"], rr[k][int(s)]["bytes_rh"]] for s in sp]) for k in rr}

            def d4(VV, w):
                def mean(x):
                    ok = ~np.isnan(x[:, 0])
                    return np.sum(x[ok] * w[ok, None], 0) / np.sum(w[ok])
                pts = [tuple(mean(VV[a])[[1, 0]]) for a in ARM]
                m = mean(VV["pol"])
                return m[0] - hull_delay(pts, m[1]), m, pts

            n = len(sp)
            out = {}
            for lab, VV in (("live", V), ("replay", VR)):
                d0, m, pts = d4(VV, np.ones(n))
                bs = []
                for _ in range(400):
                    w = brng.multinomial(n, [1 / n] * n).astype(float)
                    v = d4(VV, w)[0]
                    if v == v:
                        bs.append(v)
                lo, hi = (np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if len(bs) > 320 else (np.nan, np.nan)
                out[lab] = (d0, lo, hi, m, pts, len(bs))
            q = pub.get((LIVE2REP[st], p))
            d0, lo, hi, m, pts, nb = out["live"]
            r0, rlo, rhi, *_ = out["replay"]
            xs = sorted(x for x, _ in pts)
            pubs = (f"{f(float(q['d4_ms']))} ({f(float(q['d4_lo']))}, {f(float(q['d4_hi']))})", f"{f(float(q['dc_ms']))} ({f(float(q['dc_lo']))}, {f(float(q['dc_hi']))})") if q and q["d4_ms"] else ("not published", "not published")
            L.append(f"| {st} | {p} | {f(d0)} ({f(lo)}, {f(hi)}) | {f(m[0])} | {f(m[1],2)} | {xs[0]:.1f}-{xs[-1]:.1f} | {f(r0)} ({f(rlo)}, {f(rhi)}) | {pubs[0]} | {pubs[1]} |")
            fr_rows.append(dict(stage=st, p=p, live_d4=d0, live_lo=lo, live_hi=hi, live_D=m[0], live_B=m[1], n_boot_ok=nb,
                                replay_d4=r0, replay_lo=rlo, replay_hi=rhi,
                                arm_pts_live=";".join(f"{x:.2f}:{y:.0f}" for x, y in pts),
                                arm_pts_replay=";".join(f"{x:.2f}:{y:.0f}" for x, y in out["replay"][4])))
    with open(f"{OUT}/front4.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fr_rows[0].keys())
        w.writeheader()
        w.writerows(fr_rows)
    for r in fr_rows:
        L.append(f"- {r['stage']} p {r['p']}: arms (B norm : D ms) live {r['arm_pts_live']} | replay {r['arm_pts_replay']}")
    open(f"{OUT}/tables.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
