"""Live tables for the s6* runs (8-12 s uniform outages) from perrun.csv; writes live_main.csv, front4.csv, tables.md.

Every statistic is evidence/09/ana.py's, imported unchanged (per_seed_cols, stats via boot, hull_delay, REPS = 200):
  delay      = sum of discovery.csv latency over uncensored 4-hop pairs / their count (pairs pooled over seeds)
  delay_seedmean = mean over seeds of each seed's 4-hop delay (the [4] hull uses this convention)
  cens4      = censored 4-hop pairs / all 4-hop pairs
  B_norm     = in-window state-request + reply bytes per uncensored record-hop, every state name rebased to
               s4a's mean group length 14.92 (what makes replay's INT_B = 50.9)
  B_const    = (polls * INT_B + answered polls * REP_B[p]) / record-hops, the replay's byte model
  polls      = polls.csv polls_sent per run (8 directed pairs, window [t0 + 10 s, t0 + 130 s])
  waste      = per-run mean of 1 - state_rx / polls_sent
  ± values are seed-bootstrap SEs (multinomial over the 20 seeds, 200 reps).
[4] = policy delay (seed mean) minus the lower convex hull of the 4 fixed arms' (B_norm, delay seed mean) read at
the policy's B_norm, seeds paired (the link schedule depends on (p, seed) only); CI95 from 400 seed bootstraps.

Bootstrap order (so p 0.2 / 0.4 reproduce the interim table built on p 0.2 / 0.4 alone): the main table draws
from one default_rng(20261105) stream, first over the p 0.2 / 0.4 cells in sorted (stage, p) order, then over the
p 0.5 / 0.6 cells in the same order; [4] draws from one default_rng(5) stream in the order
(short, wait) x (0.2, 0.4), then (short, wait) x (0.5, 0.6).

REP_B: replay.REP_B (s4a's reply bytes per answered poll) stops at p 0.4, and per_seed_cols indexes it for
B_const. The reply bytes per answered poll are measured here per p over the six s6 stages pooled, with names
rebased to 14.92 as evidence/09 recheck.py does (rep_B - (group_len - 14.92) * answered, answered = n_gap -
n_norep); the measured values are put into REP_B for p 0.5 / 0.6 only. p 0.2 / 0.4 keep s4a's values.

Censoring: cens4 is reported on every cell together with the censored 4-hop pair count, the runs with any
censored 4-hop pair (c4 > 0) and the runs with any censored pair at any hop distance. A censored pair drops out
of `delay`, and censored pairs are the slow ones, so a heavily censored cell's delay is biased low.

Environment: none needed (reads perrun.csv here and evidence/09 live_main.csv / front4.csv for the s5 columns).
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV09 = os.path.join(HERE, os.pardir, "09")
sys.path.insert(0, EV09)
import ana as A  # noqa: E402
import replay as R  # noqa: E402  (the same module object ana.per_seed_cols reads REP_B from)

H = (1, 2, 3, 4)
KINDS = ("short", "wait", "b0p5", "b1p359", "b3p695", "b10p043")
S6ARM = {"s6b0p5": 0.5, "s6b1p359": 1.359, "s6b3p695": 3.695, "s6b10p043": 10.043}
PGROUPS = ((0.2, 0.4), (0.5, 0.6))
PS = PGROUPS[0] + PGROUPS[1]
f = A.f


def load_perrun():
    """perrun.csv rows grouped by (stage, p), each group sorted by seed, values as floats (as 09's load_live)."""
    g = defaultdict(list)
    for r in csv.DictReader(open(f"{HERE}/perrun.csv")):
        d = {k: float(v) for k, v in r.items() if k != "stage"}
        d["stage"] = r["stage"]
        g[(r["stage"], d["p"])].append(d)
    for v in g.values():
        v.sort(key=lambda r: r["seed"])
    return g


def reply_bytes(runs):
    """(rebased, raw, answered) reply bytes per answered poll pooled over `runs` (recheck.py's formula)."""
    reb = raw = ans = 0.0
    for r in runs:
        dg = r["group_len"] - A.REF_G
        for h in H:
            a = r[f"ngap{h}"] - r[f"nnorep{h}"]
            raw += r[f"repB{h}"]
            reb += r[f"repB{h}"] - dg * a
            ans += a
    return reb / ans, raw / ans, ans


def main():
    g = load_perrun()
    L = []

    # ---- REP_B calibration (must precede per_seed_cols for p 0.5 / 0.6)
    rep = {}
    for p in PS:
        rep[p] = reply_bytes([r for kind in KINDS for r in g[(f"s6{kind}", p)]])
        rep.update({(f"s6{k}", p): reply_bytes(g[(f"s6{k}", p)]) for k in KINDS})
    s4a_rep = {p: R.REP_B.get(p) for p in PS}
    for p in PGROUPS[1]:
        assert p not in R.REP_B, p
        R.REP_B[p] = rep[p][0]
    L += ["## Reply bytes per answered poll (REP_B calibration)", "",
          "Measured over the six s6 stages pooled, names rebased to 14.92 (raw in brackets). s4a is replay.REP_B. "
          "Used in B_const: s4a's value at p 0.2 / 0.4, the measured value at p 0.5 / 0.6.", "",
          "| p | s6 measured rebased (raw) | s4a REP_B | " + " | ".join(f"s6{k}" for k in KINDS) + " |",
          "|---|---|---|" + "---|" * len(KINDS)]
    for p in PS:
        L.append(f"| {p} | {rep[p][0]:.1f} ({rep[p][1]:.1f}) | {f(s4a_rep[p], 1)} | "
                 + " | ".join(f"{rep[(f's6{k}', p)][0]:.1f}" for k in KINDS) + " |")

    # ---- main table
    rng = np.random.default_rng(20261105)
    main_rows, mats = {}, {}
    for grp in PGROUPS:
        for (st, p), runs in sorted(g.items()):
            if p not in grp:
                continue
            M = A.per_seed_cols(runs, st, p)
            mats[(st, p)] = (M, [r["seed"] for r in runs])
            base, se = A.boot(M, A.REPS, rng)
            extra = dict(
                cens4_pairs=sum(r["c4"] for r in runs), pairs4=sum(r["c4"] + r["n4"] for r in runs),
                runs_cens4=sum(1 for r in runs if r["c4"] > 0),
                runs_cens_any=sum(1 for r in runs if any(r[f"c{h}"] > 0 for h in H)),
                seeds_no_uncens4=sum(1 for r in runs if r["n4"] == 0),
                resched_ans=np.mean([r["resched_ans"] for r in runs]), resched_to=np.mean([r["resched_to"] for r in runs]),
                clamped=sum(r["clamped"] for r in runs), late=sum(r["late"] for r in runs),
                censor_disagree=sum(r["exact_censor_disagree"] for r in runs),
                within1=sum(r["exact_within1"] for r in runs) / max(sum(r["exact_n"] for r in runs), 1),
                rep_b_used=R.REP_B[p], n_runs=len(runs))
            main_rows[(st, p)] = dict(stage=st, p=p, **base, **{k + "_se": v for k, v in se.items()}, **extra)
    rows = sorted(main_rows.values(), key=lambda r: (r["stage"], r["p"]))
    with open(f"{HERE}/live_main.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    s5 = {(r["stage"], float(r["p"])): r for r in csv.DictReader(open(f"{EV09}/live_main.csv"))}
    L += ["", "## Main table (live, s6 = 8-12 s outages; s5 = 5 s slots from evidence/09 live_main.csv)", "",
          "| stage | p | 4-hop delay ms ± SE | seed-mean | cens4 % | B/rh norm ± SE | polls/run ± SE | waste ± SE "
          "| s5 delay ± SE | s5 cens4 % | s5 B/rh norm | s5 polls/run | s5 waste |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        for p in PS:
            r = main_rows[(f"s6{kind}", p)]
            o = s5.get((f"s5{kind}", p))
            s5c = (f"{f(float(o['delay']))} ± {f(float(o['delay_se']))}", f(100 * float(o["cens4"]), 2),
                   f(float(o["B_norm"]), 1), f(float(o["polls"]), 1), f(float(o["waste"]), 3)) if o else ("-",) * 5
            L.append(f"| s6{kind} | {p} | {f(r['delay'])} ± {f(r['delay_se'])} | {f(r['delay_seedmean'])} | "
                     f"{f(100 * r['cens4'], 2)} | {f(r['B_norm'], 1)} ± {f(r['B_norm_se'], 1)} | "
                     f"{f(r['polls'], 1)} ± {f(r['polls_se'], 1)} | {f(r['waste'], 3)} ± {f(r['waste_se'], 3)} | "
                     + " | ".join(s5c) + " |" + f" n={r['n_runs']}" * (r["n_runs"] != 20))

    L += ["", "## Censoring counts", "",
          "cens4 pairs = censored 4-hop (h0<->h4) pairs over all 4-hop pairs, 20 seeds pooled; runs cens4 = runs with "
          "at least one censored 4-hop pair; runs cens any = runs with a censored pair at any hop distance; "
          "seeds w/o delay = seeds whose 4-hop pairs are all censored (dropped from the seed mean).", "",
          "| stage | p | cens4 % | cens4 pairs | runs cens4 | runs cens any | seeds w/o delay |", "|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        for p in PS:
            r = main_rows[(f"s6{kind}", p)]
            L.append(f"| s6{kind} | {p} | {f(100 * r['cens4'], 2)} | {int(r['cens4_pairs'])}/{int(r['pairs4'])} | "
                     f"{r['runs_cens4']} | {r['runs_cens_any']} | {r['seeds_no_uncens4']} |")

    # ---- [4]
    f4 = {(r["stage"], float(r["p"])): r for r in csv.DictReader(open(f"{EV09}/front4.csv"))}
    brng = np.random.default_rng(5)
    fr_rows = []
    for grp in PGROUPS:
        for kind in ("short", "wait"):
            for p in grp:
                st = f"s6{kind}"
                Mp, sp = mats[(st, p)]
                arms = {a: mats[(a, p)] for a in S6ARM}
                for a, (_m, sa) in arms.items():
                    assert sa == sp, (a, p)
                V = {"pol": np.c_[Mp[:, 0] / Mp[:, 1], Mp[:, 5] / Mp[:, 3]],
                     **{a: np.c_[m[:, 0] / m[:, 1], m[:, 5] / m[:, 3]] for a, (m, _s) in arms.items()}}

                def d4(w, V=V):
                    def mean(x):
                        ok = ~np.isnan(x[:, 0])
                        return np.sum(x[ok] * w[ok, None], 0) / np.sum(w[ok])
                    pts = [tuple(mean(V[a])[[1, 0]]) for a in S6ARM]
                    m = mean(V["pol"])
                    return m[0] - A.hull_delay(pts, m[1]), m, pts

                n = len(sp)
                d0, m, pts = d4(np.ones(n))
                # weight of each arm's delay in the hull value at the policy's bytes (d hull / d arm delay):
                # a bias of delta ms in an arm's delay moves [4] by -weight * delta
                hw = []
                for i in range(len(pts)):
                    q = [(x, y + (1.0 if j == i else 0.0)) for j, (x, y) in enumerate(pts)]
                    hw.append(A.hull_delay(q, m[1]) - A.hull_delay(pts, m[1]))
                bs = []
                for _ in range(400):
                    w = brng.multinomial(n, [1 / n] * n).astype(float)
                    v = d4(w)[0]
                    if v == v:
                        bs.append(v)
                lo, hi = (np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if len(bs) > 320 else (np.nan, np.nan)
                q = f4.get((f"s5{kind}", p))
                fr_rows.append(dict(stage=st, p=p, live_d4=d0, live_lo=lo, live_hi=hi, live_D=m[0], live_B=m[1],
                                    n_boot_ok=len(bs), pol_cens4=main_rows[(st, p)]["cens4"],
                                    arm_pts_live=";".join(f"{x:.2f}:{y:.0f}" for x, y in pts),
                                    hull_w=";".join(f"{x:.2f}" for x in hw),
                                    arm_cens4=";".join(f"{100 * main_rows[(a, p)]['cens4']:.2f}" for a in S6ARM),
                                    s5_d4=float(q["live_d4"]) if q else np.nan, s5_lo=float(q["live_lo"]) if q else np.nan,
                                    s5_hi=float(q["live_hi"]) if q else np.nan))
    fr_rows.sort(key=lambda r: (r["stage"], r["p"]))
    with open(f"{HERE}/front4.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, list(fr_rows[0].keys()))
        w.writeheader()
        w.writerows(fr_rows)
    L += ["", "## [4]: policy delay minus lower hull of the 4 fixed arms at the policy's bytes (ms), seed bootstrap 400", "",
          "Arm points are (B/rh norm : 4-hop delay seed mean ms) for 0.5 / 1.359 / 3.695 / 10.043 s; arm cens4 % in the same order.", "",
          "| policy | p | s6 [4] (CI95) | boots ok | s6 D | s6 B | policy cens4 % | s5 [4] (CI95) | arm points | arm cens4 % |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in fr_rows:
        s5d = f"{f(r['s5_d4'])} ({f(r['s5_lo'])}, {f(r['s5_hi'])})" if r["s5_d4"] == r["s5_d4"] else "-"
        L.append(f"| {r['stage']} | {r['p']} | {f(r['live_d4'])} ({f(r['live_lo'])}, {f(r['live_hi'])}) | {r['n_boot_ok']} | "
                 f"{f(r['live_D'])} | {f(r['live_B'], 2)} | {f(100 * r['pol_cens4'], 2)} | {s5d} | "
                 f"{r['arm_pts_live'].replace(';', '; ')} | {r['arm_cens4'].replace(';', ' / ')} |")
    L += ["", "Hull weights (d hull / d arm delay at the policy's bytes) with the arms' cens4: an arm's censored pairs "
          "drop out of its delay, so a censored arm's point sits too low and pulls the hull down by weight x bias, "
          "which makes [4] less negative; the policy's own censoring biases its D low and [4] more negative.", ""]
    for r in fr_rows:
        ws = [float(x) for x in r["hull_w"].split(";")]
        cs = r["arm_cens4"].split(";")
        terms = " + ".join(f"{w:.2f} x {a[2:]} (cens4 {c} %)" for w, a, c in zip(ws, S6ARM, cs) if w > 0)
        L.append(f"- {r['stage']} p {r['p']}: hull at B {r['live_B']:.2f} = {terms}; policy cens4 {100 * r['pol_cens4']:.2f} %")
    open(f"{HERE}/tables.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
