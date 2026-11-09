"""Sensitivity of the live [4] to the counting choices: bytes axis (raw / name-rebased / s4a-constant per poll)
and delay aggregation (per-seed mean vs pair-pooled direct). Same seed bootstrap as ana.py (400 reps)."""
import numpy as np

import ana

live = ana.load_live()
rng = np.random.default_rng(5)
BCOL = {"B_raw": 4, "B_norm": 5, "B_const": 7}
print("| policy | p | bytes axis | delay agg | policy B | arms B (0.5/1.359/3.695/10.043) | [4] ms (CI95) | boot ok |")
print("|---|---|---|---|---|---|---|---|")
for st in ("s5short", "s5wait"):
    for p in (0.0, 0.2, 0.4):
        M = {k: ana.per_seed_cols(live[(k, p)], k, p) for k in [st] + list(ana.ARM)}
        for bname, bc in BCOL.items():
            for agg in ("seedmean", "pooled"):
                def d4(w):
                    def mean(X):
                        if agg == "pooled":
                            return np.array([w @ X[:, 0] / (w @ X[:, 1]), w @ X[:, bc] / (w @ X[:, 3])])
                        d = X[:, 0] / X[:, 1]; b = X[:, bc] / X[:, 3]; ok = ~np.isnan(d)
                        return np.array([np.sum(w[ok] * d[ok]) / np.sum(w[ok]), np.sum(w[ok] * b[ok]) / np.sum(w[ok])])
                    pts = [tuple(mean(M[a])[[1, 0]]) for a in ana.ARM]
                    m = mean(M[st])
                    return m[0] - ana.hull_delay(pts, m[1]), m, pts
                n = 20
                d0, m, pts = d4(np.ones(n))
                bs = [v for v in (d4(rng.multinomial(n, [1 / n] * n).astype(float))[0] for _ in range(400)) if v == v]
                ci = f"({np.percentile(bs, 2.5):.0f}, {np.percentile(bs, 97.5):.0f})" if len(bs) > 320 else "(-, -)"
                print(f"| {st} | {p} | {bname} | {agg} | {m[1]:.2f} | {'/'.join(f'{x:.1f}' for x, _ in pts)} | {d0:.0f} {ci} | {len(bs)} |")
