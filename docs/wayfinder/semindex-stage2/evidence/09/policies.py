"""Evaluate fixed and outcome-dependent poll policies in the replay over the 100 logged s4a schedules per p."""
import itertools
import multiprocessing as mp
import pickle
import random
import sys

import numpy as np

import dp
import replay as R

PS = (0.0, 0.2, 0.4)
SEEDS = list(range(1, 101))
A = R.ARMS
GRID = (0.5, 0.7, 1.0, 1.359, 1.8, 2.5, 3.695, 5.0, 7.0, 10.043, 15.0)


def named():
    """(name, group, policy) for the rules asked for; `engine` copies use today's information timing."""
    out = [(f"fixed {x}", "fixed", R.Fixed(x)) for x in GRID]
    tabs = [(.5, 1.359), (.5, 3.695), (.5, 10.043), (1.359, 3.695), (1.359, 10.043), (3.695, 10.043),
            (.5, 1.359, 3.695, 10.043), (1.359, 3.695, 10.043)]
    for t in tabs:
        out.append((f"lengthen {t}", "2-lengthen", R.Table(t)))
        out.append((f"lengthen-engine {t}", "2-engine", R.Table(t, ideal=False)))
    for t in [(1.359, .5), (3.695, .5), (10.043, .5), (3.695, 1.359), (10.043, 1.359), (10.043, 3.695)]:
        out.append((f"shorten {t}", "2b-shorten", R.Table(t)))
        out.append((f"shorten-engine {t}", "2b-engine", R.Table(t, ideal=False)))
    for b in (.5, 1.359, 3.695):
        out.append((f"backoff {b}", "3-backoff", R.Backoff(b)))
        out.append((f"backoff-engine {b}", "3-engine", R.Backoff(b, ideal=False)))
    for b in A:
        out.append((f"slotclock {b}", "4a-slotclock", R.SlotClock(b)))
        out.append((f"slotclock.6 {b}", "4a-slotclock", R.SlotClock(b, eps=0.6)))
        out.append((f"slotblind {b}", "4b-slotblind", R.SlotBlind(b)))
        out.append((f"oracle {b}", "oracle", R.Oracle(b)))
    return out


def grid():
    """All arm tables for s = 0, 1, 2, 3+ (contains every step-only rule over the four arms)."""
    return [(f"table {t}", "table", R.Table(t)) for t in itertools.product(A, repeat=4)]


SCHED = {}


def eval_policy(task):
    """Per-seed (delay s, B/rh, waste, censored records share, polls) for one policy at one p, `reps` jitter draws each."""
    name, group, pol, p, reps = task
    out = np.full((len(SEEDS), 4), np.nan)
    for si, s in enumerate(SEEDS):
        links, rec = SCHED[(p, s)]
        ds, bs, ws, cs = [], [], [], []
        total = len(rec["0"]) + len(rec["4"])
        for rep in range(reps):
            rng = random.Random(1000003 * s + 7919 * rep + int(p * 10))
            d, b, w, c, _ = R.simulate(pol, links, rec, p, rng)
            ds.append(d)
            bs.append(b)
            ws.append(w)
            cs.append(1 - c / total)
        out[si] = (np.nanmean(ds) if np.any(~np.isnan(ds)) else np.nan, np.mean(bs), np.mean(ws), np.mean(cs))
    return name, group, p, out


def main():
    """Named rules, the 256 tables and the DP folds; everything is pickled for frontier.py."""
    for p in PS:
        for s in SEEDS:
            links, rec = R.load_schedule(p, s)
            SCHED[(p, s)] = ([R.Link(iv) for iv in links], rec)
    dp.load_runs()
    tasks = [(n, g, pol, p, 4) for n, g, pol in named() for p in PS]
    tasks += [(n, g, pol, p, 1) for n, g, pol in grid() for p in (0.2, 0.4)]
    dp_meta = []
    lams = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    for p in (0.2, 0.4):
        for fold, seeds in enumerate(({s for s in SEEDS if s % 2}, {s for s in SEEDS if not s % 2})):
            for variant in ("step", "elapsed"):
                c, G = dp.tables(p, seeds, variant)
                for lam in lams:
                    act, g = dp.solve(c, G, p, lam, variant)
                    model_age, model_rate = dp.predict(act, c, G, p, variant)
                    nk = len(act)
                    ages = (0, 10, 20, 50, 100, 200)
                    summ = {k: [act[k][a] for a in ages] for k in range(nk)}
                    dp_meta.append(dict(p=p, fold=fold, variant=variant, lam=lam, model_age=model_age, model_rate=model_rate,
                                        act=summ, c=c, G=G))
                    name = f"dp-{variant} f{fold} lam{lam}"
                    el = variant == "elapsed"
                    pol = R.DP(act, 4, dp.DAGE, elapsed=el)
                    pol2 = R.DP(act, 4, dp.DAGE, ideal=False, elapsed=el)
                    tasks.append((name, "5-dp", pol, p, 4))
                    tasks.append((name + " engine", "5-engine", pol2, p, 4))
            print("dp done", p, fold, file=sys.stderr, flush=True)
    pickle.dump(dp_meta, open("dp_meta.pkl", "wb"))
    print("tasks", len(tasks), file=sys.stderr, flush=True)
    with mp.get_context("fork").Pool(3) as pool:
        res = pool.map(eval_policy, tasks, chunksize=2)
    pickle.dump(res, open("policy_res.pkl", "wb"))
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
