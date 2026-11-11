"""Retune only the post-failure wait on the new outage shape, in the replay (a model, not a measurement).

Procedure (fixed before computing; the map's rule: only run live if the retuned rule is ~2 s faster):
  Families (post-answer interval 10.043 s jittered +/-10 % in all of them):
    short  R.Table((10.043, w))        wait jittered +/-10 %      w in 0.5..12 step 0.25   unchanged rule: s6short
    wait   R.SlotBlind(10.043, wait=w) wait NOT jittered          w in 0.5..12 step 0.25   unchanged rule: s6wait
    two    R.Table((10.043, w1, w2))   w1 after 1 failure, w2 after 2+, both jittered; w1, w2 in 0.5..12 step 0.5
                                                                                           unchanged rule: s6short
    So short-form and wait-form at the same w are different rules (jittered vs exact wait); each family is compared
    with its own unchanged rule only: s6short's wait is jittered, s6wait's is not.
  Training: schedules regenerated with the s6 generator for seeds 21-120 at p 0.2 and 0.4, REPS jitter draws each
    (replay_common.per_seed). The four fixed arms are evaluated on the same seeds.
    [4] = seed-mean delay - lower hull of the 4 arms' seed-mean (bytes, delay) at the rule's seed-mean bytes (ana.py);
    NaN (skipped) outside the arms' bytes range. Pick per (family, p): smallest training [4].
    Bytes-capped pick: smallest training seed-mean delay among candidates whose training bytes <= the unchanged rule's.
  Test: the stored live schedules, seeds 1-20, paired replay. delta X = X(pick) - X(unchanged rule), X in [4], delay,
    bytes, all seed means; CI95 = percentiles of 400 seed-bootstrap reps whose weights are shared by the pick, the
    unchanged rule and the four arms within a rep. R.Oracle(10.043) on the same seeds for scale.
  Gate per (family, p): delta[4] < -2000 ms, its CI95 excludes 0, and the unchanged rule's replay - live delay bias
    (replay_calib.csv) is well under 2 s, operationalised here as |bias| + 1.96 SE < 2000 ms.
Training REPS (10) were not fixed by the procedure; 10 is replay_runs.py's paired count.
Outputs: replay_retune_train.csv (every candidate's training seed means and [4]), replay_retune.csv (picks on test).
"""
import csv
import multiprocessing as mp
import os
import sys

import numpy as np

import replay_common as C
from replay_common import R

OUT = C.HERE
PS = (0.2, 0.4)
TRAIN = range(21, 121)
TEST = range(1, 21)
GRID1 = [round(0.5 + 0.25 * i, 3) for i in range(47)]   # 0.5 .. 12.0
GRID2 = [round(0.5 + 0.5 * i, 3) for i in range(24)]    # 0.5 .. 12.0
REF = {"short": "s6short", "wait": "s6wait", "two": "s6short"}
B10 = C.B10


def make(key):
    """(policy, tag) of a candidate key; tags are numeric (see replay_common's seeding note)."""
    fam = key[0]
    if fam == "short":
        return R.Table((B10, key[1])), (B10, key[1])
    if fam == "wait":
        return R.SlotBlind(B10, wait=key[1]), (B10, key[1])
    if fam == "two":
        return R.Table((B10, key[1], key[2])), (B10, key[1], key[2])
    if fam == "oracle":
        return R.Oracle(B10), (B10, -1.0)
    named = {k: (pol, tag) for k, pol, tag in C.arm_rules() + C.unchanged_rules()}
    return named[key[1]]


BASE = [("arm", k) for k in C.ARMS] + [("unch", "s6short"), ("unch", "s6wait")]
CANDS = [("short", w) for w in GRID1] + [("wait", w) for w in GRID1] + [("two", a, b) for a in GRID2 for b in GRID2]
POLS = {k: make(k) for k in BASE + CANDS}


def train_task(t):
    """All rules on one regenerated training schedule."""
    p, s = t
    links, rec = C.gen_schedule(p, s)
    links = [R.Link(iv) for iv in links]
    return p, s, {k: C.per_seed(*POLS[k], links, rec, p, s) for k in BASE + CANDS}


def test_task(t):
    """Given rules on one stored live schedule."""
    p, s, keys = t
    links, rec = C.stored_schedule(p, s)
    links = [R.Link(iv) for iv in links]
    return p, s, {k: C.per_seed(*make(k), links, rec, p, s) for k in keys}


def mat(res, p, seeds, key):
    """(n_seeds, 2) array of (delay_ms, bytes_rh) for one rule."""
    return np.array([[res[(p, s)][key]["delay_ms"], res[(p, s)][key]["bytes_rh"]] for s in seeds])


def load_bias():
    """(stage, p) -> (replay - live delay mean, bootstrap SE) from replay_calib.csv, if it has been written."""
    f = f"{OUT}/replay_calib.csv"
    if not os.path.exists(f):
        return {}
    return {(r["stage"], float(r["p"])): (float(r["diff"]), float(r["diff_se"]))
            for r in csv.DictReader(open(f)) if r["metric"] == "delay_ms"}


def flat(rows, best):
    """Candidates within 300 ms of the best training [4]: a short text for the report."""
    near = [r for r in rows if r["d4"] == r["d4"] and r["d4"] <= best + 300]
    if rows[0]["family"] == "two":
        w1 = sorted({r["w1"] for r in near})
        w2 = sorted({r["w2"] for r in near})
        return f"{len(near)} of {len(rows)} (w1 {w1[0]}-{w1[-1]}, w2 {w2[0]}-{w2[-1]})"
    ws = sorted(r["w1"] for r in near)
    idx = [GRID1.index(w) for w in ws]
    contig = idx == list(range(idx[0], idx[-1] + 1))
    return f"w {ws[0]}-{ws[-1]} ({len(ws)} points, {'contiguous' if contig else 'NOT contiguous: ' + ' '.join(map(str, ws))})"


def main():
    C.check_loader()
    with mp.get_context("fork").Pool(3) as pool:
        tr = {(p, s): v for p, s, v in pool.map(train_task, [(p, s) for p in PS for s in TRAIN], chunksize=4)}
    one = np.ones(len(TRAIN))
    train_rows, picks = [], {}
    for p in PS:
        arms = [mat(tr, p, TRAIN, ("arm", a)) for a in C.ARMS]
        ref_b = {u: C.seed_mean(mat(tr, p, TRAIN, ("unch", u)), one)[1] for u in ("s6short", "s6wait")}
        for k in BASE + CANDS:
            M = mat(tr, p, TRAIN, k)
            m = C.seed_mean(M, one)
            train_rows.append(dict(p=p, family=k[0], w1=k[1] if k[0] in ("short", "wait", "two") else "",
                                   w2=k[2] if k[0] == "two" else "", name=k[1] if k[0] in ("arm", "unch") else "",
                                   delay_ms=m[0], bytes_rh=m[1], d4=C.d4(M, arms, one),
                                   waste=np.mean([tr[(p, s)][k]["waste"] for s in TRAIN]),
                                   polls=np.mean([tr[(p, s)][k]["polls"] for s in TRAIN])))
        for fam in ("short", "wait", "two"):
            rows = [r for r in train_rows if r["p"] == p and r["family"] == fam]
            ok = [r for r in rows if r["d4"] == r["d4"]]
            best = min(ok, key=lambda r: r["d4"])
            unch = next(r for r in train_rows if r["p"] == p and r["name"] == REF[fam])
            capped = [r for r in rows if r["bytes_rh"] <= ref_b[REF[fam]]]
            cap = min(capped, key=lambda r: r["delay_ms"]) if capped else None
            picks[(fam, p)] = dict(best=best, cap=cap, unch=unch, flat=flat(rows, best["d4"]), n_skipped=len(rows) - len(ok))
    with open(f"{OUT}/replay_retune_train.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, train_rows[0].keys())
        w.writeheader()
        w.writerows(train_rows)

    def key_of(r):
        return (r["family"], r["w1"]) if r["family"] != "two" else ("two", r["w1"], r["w2"])

    test_keys = {p: set(BASE) | {("oracle", B10)} for p in PS}
    for (fam, p), d in picks.items():
        test_keys[p].add(key_of(d["best"]))
        if d["cap"]:
            test_keys[p].add(key_of(d["cap"]))
    with mp.get_context("fork").Pool(3) as pool:
        te = {(p, s): v for p, s, v in pool.map(test_task, [(p, s, sorted(test_keys[p], key=str)) for p in PS for s in TEST])}

    # cross-check: the unchanged rules and arms on the test seeds equal replay_calib's per-seed rows (same seeding)
    fcal = f"{OUT}/replay_calib_perseed.csv"
    if os.path.exists(fcal):
        n = 0
        for r in csv.DictReader(open(fcal)):
            p, s = float(r["p"]), int(r["seed"])
            k = ("unch", r["stage"]) if r["stage"] in ("s6short", "s6wait") else ("arm", r["stage"])
            if (p, s) in te:
                assert abs(te[(p, s)][k]["delay_ms"] - float(r["delay_ms"])) < 1e-6, (k, p, s)
                n += 1
        print(f"cross-check vs replay_calib_perseed.csv: {n} per-seed rows equal", file=sys.stderr)

    bias = load_bias()
    W = C.boot_weights(len(TEST), 400, 20261112)
    one = np.ones(len(TEST))
    out = []
    for (fam, p), d in sorted(picks.items(), key=lambda x: (x[0][1], x[0][0])):
        arms = [mat(te, p, TEST, ("arm", a)) for a in C.ARMS]
        U = mat(te, p, TEST, ("unch", REF[fam]))
        bz = bias.get((REF[fam], p))
        bias_ok = None if bz is None else abs(bz[0]) + 1.96 * bz[1] < 2000
        for kind in ("best", "cap"):
            r = d[kind]
            if r is None:
                out.append(dict(family=fam, p=p, pick=kind, note="no candidate under the bytes cap"))
                continue
            P = mat(te, p, TEST, key_of(r))

            def deltas(w):
                mp_, mu = C.seed_mean(P, w), C.seed_mean(U, w)
                return C.d4(P, arms, w) - C.d4(U, arms, w), mp_[0] - mu[0], mp_[1] - mu[1]

            d0 = deltas(one)
            bs = np.array([deltas(w) for w in W])
            ci = [C.ci95(bs[:, j]) for j in range(3)]
            row = dict(family=fam, p=p, pick=kind, w1=r["w1"], w2=r["w2"], unchanged=REF[fam],
                       train_d4=r["d4"], train_delay=r["delay_ms"], train_bytes=r["bytes_rh"],
                       train_unch_d4=d["unch"]["d4"], train_unch_delay=d["unch"]["delay_ms"],
                       train_unch_bytes=d["unch"]["bytes_rh"], train_flat_300ms=d["flat"] if kind == "best" else "",
                       train_n_nan_d4=d["n_skipped"],
                       test_d4=C.d4(P, arms, one), test_unch_d4=C.d4(U, arms, one),
                       test_delay=C.seed_mean(P, one)[0], test_bytes=C.seed_mean(P, one)[1],
                       ddelta4=d0[0], ddelta4_lo=ci[0][0], ddelta4_hi=ci[0][1],
                       ddelay=d0[1], ddelay_lo=ci[1][0], ddelay_hi=ci[1][1],
                       dbytes=d0[2], dbytes_lo=ci[2][0], dbytes_hi=ci[2][1],
                       bias_ms=bz[0] if bz else float("nan"), bias_se=bz[1] if bz else float("nan"))
            if kind == "best":
                c1 = d0[0] < -2000
                c2 = ci[0][1] < 0 or ci[0][0] > 0
                row.update(gate_d4_lt_m2000=c1, gate_ci_excl0=c2, gate_bias_ok=bias_ok,
                           gate="pending (no calibration)" if bias_ok is None else ("pass" if c1 and c2 and bias_ok else "fail"))
            else:
                # the capped pick's own gate uses delta delay (its bytes are <= the unchanged rule's on training)
                c1 = d0[1] < -2000
                c2 = ci[1][1] < 0 or ci[1][0] > 0
                row.update(gate_d4_lt_m2000=c1, gate_ci_excl0=c2, gate_bias_ok=bias_ok,
                           gate="pending (no calibration)" if bias_ok is None else ("pass" if c1 and c2 and bias_ok else "fail"))
            out.append(row)
    for p in PS:
        arms = [mat(te, p, TEST, ("arm", a)) for a in C.ARMS]
        orc = mat(te, p, TEST, ("oracle", B10))
        for u in ("s6short", "s6wait"):
            U = mat(te, p, TEST, ("unch", u))
            bs = np.array([C.seed_mean(orc, w)[0] - C.seed_mean(U, w)[0] for w in W])
            lo, hi = C.ci95(bs)
            out.append(dict(family="oracle", p=p, pick="bound", w1="", w2="", unchanged=u,
                            test_d4=C.d4(orc, arms, one), test_unch_d4=C.d4(U, arms, one),
                            test_delay=C.seed_mean(orc, one)[0], test_bytes=C.seed_mean(orc, one)[1],
                            ddelay=C.seed_mean(orc, one)[0] - C.seed_mean(U, one)[0], ddelay_lo=lo, ddelay_hi=hi,
                            dbytes=C.seed_mean(orc, one)[1] - C.seed_mean(U, one)[1]))
        for a in C.ARMS:
            m = C.seed_mean(mat(te, p, TEST, ("arm", a)), one)
            out.append(dict(family="arm", p=p, pick=a, test_delay=m[0], test_bytes=m[1]))
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    with open(f"{OUT}/replay_retune.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, keys)
        w.writeheader()
        w.writerows(out)
    print("rows", len(out), file=sys.stderr)


if __name__ == "__main__":
    main()
