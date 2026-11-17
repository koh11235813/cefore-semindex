"""Two measurements added AFTER the results of trickle.py were seen (not in spec2.md).

Why they exist:

(1) extra_arms. trickle.py scores a rule against a chord through four fixed arms. A review showed the chord
    lies above the real fixed-interval curve where that curve is convex in log-log, which flatters a rule
    whose poll count falls mid-segment. T37 is the only variant with positive gains, all mid-segment. The
    identity test shows a rule with m = 1 and I_min = I_max = X is the fixed arm X, so fixed arms at
    intervals straddling T37's poll count are simulated here and T37 is scored against them instead.
    The idea and the interval lists come from the review (review2-compare); this is a rerun by the main
    session with its own code.

(2) gaps. The first write-up asserted that the rule loses because it sits at its longest interval when the
    next record lands. This measures it: per rule, how long the node had gone without asking a neighbor when
    a reply brought something new (gap before a non-empty reply) and when it brought nothing (gap before an
    EMPTY reply), how many polls were sent after the node already held all 240 records, and how uneven the
    spacing is (mean gap against the gap met at a random instant). The assertion predicts a longer gap
    before non-empty replies; an uneven-spacing cost predicts a length-weighted gap well above the mean.

Both use Mode B on the donor inputs, the same replicate seeds as trickle.py. Output: after.md (tables).
Usage: python3 after.py
"""
import math
import multiprocessing as mp
import os
import random

import replay as R
import trickle as T

HERE = os.path.dirname(os.path.abspath(__file__))
JF = T.JF
EXTRA = {
    ("s5b10p043", 0.0): [5.0, 5.5, 5.8, 6.0, 6.5],
    ("s5b10p043", 0.2): [5.5, 6.0, 6.2, 6.5, 7.0],
    ("s6b10p043", 0.2): [5.5, 6.0, 6.2, 6.5, 7.0],
}
GAP_CELLS = [("s6b10p043", 0.2), ("s6b10p043", 0.4), ("s5b10p043", 0.0)]
GAP_RULES = ["fix3.695", "fix10.043", "short", "T05", "T136", "T37"]
SPEC = {n: (k, s) for n, k, s in T.RULE_SPECS}


def run_for(donor_run, name):
    """The donor's exogenous inputs with rule `name` (a trickle.py name or a float = fixed interval)."""
    run = dict(donor_run)
    if isinstance(name, float):
        run["policy"] = run["choices"] = None
        run["rule"] = dict(i_min=name, m=1.0, i_max=name, jf=JF)
        return run
    kind, spec = SPEC[name]
    if kind == "rule":
        run["policy"] = run["choices"] = None
        run["rule"] = spec
    else:
        tgt = R.load_run(donor_run["stage"][:2] + spec, donor_run["p"], donor_run["seed"])
        for k in ("policy", "choices", "choice_jitter"):
            run[k] = tgt[k]
        run["rule"] = None
    return run


def gaps(events, t0):
    """Per run: [n_nonempty, sum gap ms, n_empty, sum gap ms, polls in window, polls after complete,
    sum of gaps ms, sum of squared gaps ms^2, number of gaps] (the last three over every pair of consecutive polls to one
    neighbor inside the window, answered or not)."""
    ws, we = t0 + R.WINDOW_START_MS, t0 + R.WINDOW_END_MS
    out = [0, 0.0, 0, 0.0, 0, 0, 0.0, 0.0, 0]
    for rows in events.values():
        held, prev, cur = set(), {}, {}
        for e in rows:
            ev = e.get("event")
            if ev in ("record_created", "record_stored"):
                held.add((e["origin"], e["seq"]))
            elif ev == "decision" and e.get("kind") == "state_poll":
                nb = e["action"]["neighbor"]
                prev[nb], cur[nb] = cur.get(nb), e["ts_ms"]
                if ws <= e["ts_ms"] <= we:
                    out[4] += 1
                    out[5] += len(held) >= 240
                    if prev[nb] is not None and prev[nb] >= ws:
                        g = cur[nb] - prev[nb]
                        out[6] += g
                        out[7] += g * g
                        out[8] += 1
            elif ev == "state_rx":
                nb = e["neighbor"]
                if prev.get(nb) is None or not ws <= e["ts_ms"] <= we:
                    continue
                new = any((o, s) not in held for o, v in e["hi"].items() for s in range(1, v + 1))
                g = cur[nb] - prev[nb]
                if new:
                    out[0] += 1
                    out[1] += g
                else:
                    out[2] += 1
                    out[3] += g
    return out


def job(arg):
    donor, p, seed = arg
    d = R.load_run(donor, p, seed)
    res = {}
    names = list(T.FIXED) + ["T37"] + EXTRA.get((donor, p), [])
    if (donor, p) in EXTRA:
        for name in names:
            run = run_for(d, name)
            res[("x", name)] = [
                T.tup(R.score(R._sim_run(run, "B", sim_rng=random.Random(
                    R.rep_seed(f"trickle:{donor}", p, seed, rep)), rule=run.get("rule")), run["t0"], p))
                for rep in range(T.REPS)]
    if (donor, p) in GAP_CELLS:
        for name in GAP_RULES:
            run = run_for(d, name)
            ev = R._sim_run(run, "B", sim_rng=random.Random(R.rep_seed(f"trickle:{donor}", p, seed, 0)),
                            rule=run.get("rule"))
            res[("g", name)] = gaps(ev, run["t0"])
    return donor, p, res


def interp(pts, x):
    """log-log piecewise-linear y at x through pts [(x, y)], None outside."""
    pts = sorted(pts)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + w * (math.log(y1) - math.log(y0)))
    return None


def main():
    cells = sorted(set(EXTRA) | set(GAP_CELLS))
    with mp.Pool(6) as pool:
        res = pool.map(job, [(d, p, s) for d, p in cells for s in T.SEEDS])
    L = ["# after.md: two measurements added after the results were seen (see after.py)", ""]
    L += ["## 1. T37 against real fixed arms near its own poll count", "",
          "| donor | p | arm | polls_sent | delay ms | cens4 |", "|---|---|---|---|---|---|"]
    summ = []
    for cell in sorted(EXTRA):
        rr = [r[2] for r in res if (r[0], r[1]) == cell]
        val = {}
        for name in list(T.FIXED) + EXTRA[cell] + ["T37"]:
            c = T.pooled([x for r in rr for x in r[("x", name)]], cell[1])
            val[name] = c
            label = f"fixed {name} s" if isinstance(name, float) else name
            L.append(f"| {cell[0]} | {cell[1]} | {label} | {c['polls_sent']:.1f} | {c['delay']:.0f} | {c['cens4']:.4f} |")
        t = val["T37"]
        chord = interp([(val[n]["polls_sent"], val[n]["delay"]) for n in T.FIXED], t["polls_sent"])
        real = interp([(val[n]["polls_sent"], val[n]["delay"]) for n in EXTRA[cell]], t["polls_sent"])
        summ.append(f"| {cell[0]} | {cell[1]} | {t['polls_sent']:.1f} | {t['delay']:.0f} | {chord:.0f} | "
                    f"{100 * (1 - t['delay'] / chord):+.1f} % | "
                    + (f"{real:.0f} | {100 * (1 - t['delay'] / real):+.1f} % |" if real else "outside | |"))
    L += ["", "| donor | p | T37 polls | T37 delay ms | chord delay ms | gain vs chord | real-arm delay ms | gain vs real arms |",
          "|---|---|---|---|---|---|---|---|"] + summ
    L += ["", "## 2. How long the node had not asked when a reply came (replicate 0, 20 seeds)", "",
          "gap = time between the poll that was answered and the poll before it to the same neighbor.",
          "mean gap = sum(g) / n over consecutive polls to one neighbor; gap met at a random instant = "
          "sum(g^2) / sum(g), the length-weighted mean (an event landing at a random time falls into a "
          "gap with probability proportional to its length). Equal for evenly spaced polls.", "",
          "| donor | p | rule | polls in window | non-empty replies | gap before non-empty s | EMPTY replies | gap before EMPTY s | polls after the node held all 240 | mean gap s | gap met at a random instant s |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in GAP_CELLS:
        rr = [r[2] for r in res if (r[0], r[1]) == cell]
        for name in GAP_RULES:
            s = [sum(r[("g", name)][i] for r in rr) for i in range(9)]
            k = len(rr)
            L.append(f"| {cell[0]} | {cell[1]} | {name} | {s[4] / k:.1f} | {s[0] / k:.1f} | {s[1] / s[0] / 1000:.2f} | "
                     f"{s[2] / k:.1f} | {s[3] / s[2] / 1000:.2f} | {s[5] / k:.1f} | "
                     f"{s[6] / s[8] / 1000:.2f} | {s[7] / s[6] / 1000:.2f} |")
    open(f"{HERE}/after.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
