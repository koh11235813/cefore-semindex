"""Paired replay of the six unchanged s6 rules on the live schedules, and its bias against the live runs.

Replay side (replay_common): each stage's own stored schedule, p 0.2/0.4/0.5/0.6, seeds 1-20, REPS = 10 jitter draws
per seed with replay_runs.py's seeding (numeric tag + (p, seed, rep)). Each stage's schedule is asserted equal to
s6b0p5's for the same (p, seed). Bytes are NaN at p 0.5/0.6 (REP_B has no value there, see replay_common).

Live side, per run from the evalagg output in $AGG/<stage>/ (discovery.csv and polls.csv have CRLF endings):
  delay_ms  mean latency_ms over discovery.csv rows with hop_distance 4 and censored false, pooled over the run's
            4-hop pairs (seq_model.direct per seed, as evidence/09's per-seed delay)
  cens      censored 4-hop rows / all 4-hop rows (the replay's cens counts unfinished 4-hop record walks instead)
  polls     sum of polls_sent over the run's 8 directed pairs (polls.csv; window [t0+10, t0+130] s)
  waste     1 - sum state_rx / sum polls_sent of the run (per-run, then the seed mean: the replay's convention)
  Live bytes are not computed here (they come from live_main.csv of the live half of evidence/10).
Comparison per (stage, p, metric): diff = mean over seeds of (replay - live); diff_se = SD of 2000 seed-bootstrap
means of the per-seed differences; z = diff / diff_se; rel = diff / live mean. Seeds with a NaN on either side are
dropped for that metric.
Before anything else every stage must have 80 rows in runs.csv, all valid, covering 4 p x seeds 1-20.
Outputs: replay_calib_perseed.csv (replay per (stage, p, seed)), replay_calib.csv (the comparison),
replay_calib_front4.csv (replay [4] of the two unchanged rules, see front4), markdown on stdout.
"""
import csv
import multiprocessing as mp
import os
import sys
from collections import defaultdict

import numpy as np

import replay_common as C

OUT = C.HERE
AGG = os.environ.get("AGG", "")
PS = (0.2, 0.4, 0.5, 0.6)
SEEDS = range(1, 21)
RULES = {k: (pol, tag) for k, pol, tag in C.arm_rules() + C.unchanged_rules()}
METRICS = ("delay_ms", "waste", "polls", "cens")


def check_agg():
    """Exit naming what is missing unless every stage's runs.csv has the 80 valid runs."""
    bad = []
    for st in RULES:
        f = f"{AGG}/{st}/runs.csv"
        if not os.path.exists(f):
            bad.append(f"{st}: no runs.csv")
            continue
        rows = list(csv.DictReader(open(f, newline="")))
        got = {(float(r["p"]), int(r["seed"])) for r in rows if r["valid"] == "valid"}
        want = {(p, s) for p in PS for s in SEEDS}
        if len(rows) != 80 or got != want:
            bad.append(f"{st}: {len(rows)} rows, missing valid {sorted(want - got)[:5]}")
    if bad:
        sys.exit("AGG not ready: " + "; ".join(bad))


def live(st):
    """(p, seed) -> live per-run values of one stage."""
    lat, n4, c4 = defaultdict(float), defaultdict(int), defaultdict(int)
    for r in csv.DictReader(open(f"{AGG}/{st}/discovery.csv", newline="")):
        if r["hop_distance"] != "4":
            continue
        k = (float(r["p"]), int(r["seed"]))
        if r["censored"] == "true":
            c4[k] += 1
        else:
            assert r["censored"] == "false", r
            lat[k] += float(r["latency_ms"])
            n4[k] += 1
    pol, rx, npair = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in csv.DictReader(open(f"{AGG}/{st}/polls.csv", newline="")):
        k = (float(r["p"]), int(r["seed"]))
        pol[k] += int(r["polls_sent"])
        rx[k] += int(r["state_rx"])
        npair[k] += 1
    out = {}
    for p in PS:
        for s in SEEDS:
            k = (p, s)
            assert npair[k] == 8, (st, k, npair[k])
            out[k] = dict(delay_ms=lat[k] / n4[k] if n4[k] else float("nan"), cens=c4[k] / (n4[k] + c4[k]),
                          polls=pol[k], waste=1 - rx[k] / pol[k])
    return out


def task(t):
    """Replay of one stage's rule on its own stored schedule (asserted equal to s6b0p5's)."""
    st, p, s = t
    links, rec = C.stored_schedule(p, s, st)
    assert (links, rec) == C.stored_schedule(p, s), (st, p, s)
    return st, p, s, C.per_seed(*RULES[st], links, rec, p, s)


def s5_bias():
    """evidence/09's live_vs_replay.csv flipped to replay - live; its SE is SD(diff)/sqrt(n) (= diff / z_paired)."""
    out = {}
    for r in csv.DictReader(open(f"{C.EV09}/live_vs_replay.csv")):
        if r["stage"] in ("s5short", "s5wait") and r["metric"] in ("delay", "polls", "waste"):
            d, z = float(r["diff"]), float(r["z_paired"])
            out[(r["stage"], float(r["p"]), r["metric"])] = (-d, abs(d / z) if z == z and z else float("nan"))
    return out


def main():
    check_agg()
    with mp.get_context("fork").Pool(3) as pool:
        res = pool.map(task, [(st, p, s) for st in RULES for p in PS for s in SEEDS], chunksize=4)
    rp = {(st, p, s): v for st, p, s, v in res}
    with open(f"{OUT}/replay_calib_perseed.csv", "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["stage", "p", "seed", "delay_ms", "bytes_rh", "waste", "polls", "cens"])
        for (st, p, s), v in rp.items():
            w.writerow([st, p, s, v["delay_ms"], v["bytes_rh"], v["waste"], v["polls"], v["cens"]])
    rng = np.random.default_rng(20261112)
    rows = []
    for st in RULES:
        lv = live(st)
        for p in PS:
            for met in METRICS:
                a = np.array([rp[(st, p, s)][met] for s in SEEDS])
                b = np.array([lv[(p, s)][met] for s in SEEDS])
                ok = ~np.isnan(a) & ~np.isnan(b)
                d = a[ok] - b[ok]
                bs = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(1)
                se = float(np.std(bs, ddof=1))
                rows.append(dict(stage=st, p=p, metric=met, replay=float(np.mean(a[ok])), live=float(np.mean(b[ok])),
                                 diff=float(d.mean()), diff_se=se, z=float(d.mean() / se) if se > 0 else float("nan"),
                                 rel=float(d.mean() / np.mean(b[ok])) if np.mean(b[ok]) else float("nan"), n=int(ok.sum())))
            rows.append(dict(stage=st, p=p, metric="bytes_rh", replay=float(np.mean([rp[(st, p, s)]["bytes_rh"] for s in SEEDS])),
                             live=float("nan"), diff=float("nan"), diff_se=float("nan"), z=float("nan"), rel=float("nan"), n=20))
    with open(f"{OUT}/replay_calib.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # markdown
    R = {(r["stage"], r["p"], r["metric"]): r for r in rows}
    print("| stage | p | replay delay ms | live delay ms | replay - live (SE) | rel | replay B/rh | replay waste | live waste | diff (SE) | replay polls | live polls | diff (SE) | replay cens | live cens |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for st in RULES:
        for p in PS:
            d, wv, pl, c, b = (R[(st, p, m)] for m in ("delay_ms", "waste", "polls", "cens", "bytes_rh"))
            bb = "-" if b["replay"] != b["replay"] else f"{b['replay']:.1f}"
            print(f"| {st} | {p} | {d['replay']:.0f} | {d['live']:.0f} | {d['diff']:+.0f} ({d['diff_se']:.0f}) | {100*d['rel']:+.1f}% | {bb} | "
                  f"{wv['replay']:.3f} | {wv['live']:.3f} | {wv['diff']:+.3f} ({wv['diff_se']:.3f}) | {pl['replay']:.1f} | {pl['live']:.1f} | "
                  f"{pl['diff']:+.1f} ({pl['diff_se']:.1f}) | {100*c['replay']:.2f}% | {100*c['live']:.2f}% |")
    s5 = s5_bias()
    print()
    print("| rule | p | metric | s6 replay - live (SE) | s5 replay - live (SE) |")
    print("|---|---|---|---|---|")
    for st, s5n in (("s6short", "s5short"), ("s6wait", "s5wait")):
        for p in (0.2, 0.4, 0.5, 0.6):
            for met, m5, nd in (("delay_ms", "delay", 0), ("polls", "polls", 1), ("waste", "waste", 3)):
                r = R[(st, p, met)]
                b5 = s5.get((s5n, p, m5))
                t5 = f"{b5[0]:+.{nd}f} ({b5[1]:.{nd}f})" if b5 else "no s5 stage"
                print(f"| {st} | {p} | {m5} | {r['diff']:+.{nd}f} ({r['diff_se']:.{nd}f}) | {t5} |")
    front4(rp)


def front4(rp):
    """Replay [4] of the two unchanged rules on seeds 1-20 and the paired s6wait - s6short difference.

    The map's readings 1 and 2 turn on these: whether both rules stay below the fixed arms' line, and whether the
    exact 5 s wait loses its gain against the jittered 3.695 s wait on this shape. CI95 from 400 seed-bootstrap reps
    whose weights are shared by both rules and the four arms (ana.py's convention). Writes replay_calib_front4.csv.
    """
    W = C.boot_weights(len(SEEDS), 400, 20261112)
    one = np.ones(len(SEEDS))
    M = {(st, p): np.array([[rp[(st, p, s)]["delay_ms"], rp[(st, p, s)]["bytes_rh"]] for s in SEEDS])
         for st in RULES for p in (0.2, 0.4)}
    rows = []
    print()
    print("| p | s6short [4] (CI95) | s6wait [4] (CI95) | wait - short [4] (CI95) | wait - short delay ms (CI95) |")
    print("|---|---|---|---|---|")
    for p in (0.2, 0.4):
        arms = [M[(a, p)] for a in C.ARMS]

        def stats(w):
            ds, dw = C.d4(M[("s6short", p)], arms, w), C.d4(M[("s6wait", p)], arms, w)
            return ds, dw, dw - ds, C.seed_mean(M[("s6wait", p)], w)[0] - C.seed_mean(M[("s6short", p)], w)[0]

        v0 = stats(one)
        bs = np.array([stats(w) for w in W])
        ci = [C.ci95(bs[:, j]) for j in range(4)]
        names = ("short_d4", "wait_d4", "wait_minus_short_d4", "wait_minus_short_delay")
        row = dict(p=p)
        for j, nm in enumerate(names):
            row.update({nm: v0[j], nm + "_lo": ci[j][0], nm + "_hi": ci[j][1]})
        rows.append(row)
        print(f"| {p} | " + " | ".join(f"{v0[j]:+.0f} ({ci[j][0]:.0f}, {ci[j][1]:.0f})" for j in range(4)) + " |")
    with open(f"{OUT}/replay_calib_front4.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
