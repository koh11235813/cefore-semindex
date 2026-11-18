"""Three quantities added after results.md was read. They are outside spec.md and carry no pass line.

results.md's line 2 came out "not seen" while the point estimate had the predicted sign, so the question
that follows is which part of the comparison is uncertain: the two rules against each other, or each rule
against the line through the fixed arms. This script reports, with the bootstrap of ana.py (same seeds
drawn, same generator seed), the interval of each rule's own gain, the interval of the plain ratio of the
two rules' delays and polls, and the share of resamples whose gain difference is above zero.

Reads perrun.csv here and ../10/perrun.csv through ana.py's functions, writes after.md. No PYTHONPATH.
"""
import os

import ana
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    here = ana.load_perrun(f"{HERE}/perrun.csv")
    ten = ana.load_perrun(f"{ana.EV10}/perrun.csv")
    seeds = list(range(1, ana.SEEDS_EXPECTED + 1))
    stages = ana.ARMS + ana.RULES
    mats = {s: ana.matrix({k[1]: v for k, v in (ten if s in ana.ARMS else here).items() if k[0] == s}, seeds)
            for s in stages}
    idx = np.random.default_rng(ana.BOOT_SEED).integers(0, len(seeds), size=(ana.BOOT_REPS, len(seeds)))
    gains = {r: [] for r in ana.RULES}
    delay_ratio, polls_ratio = [], []
    for row in idx:
        sub = {s: mats[s][row] for s in stages}
        for r in ana.RULES:
            gains[r].append(ana.gain_of(sub[r], [sub[a] for a in ana.ARMS]))
        keep, reset = ana.pooled(sub["s7t37"]), ana.pooled(sub["s7t37r"])
        delay_ratio.append(reset[0] / keep[0] - 1)
        polls_ratio.append(reset[1] / keep[1] - 1)
    # ana.py found no resample with a rule outside the arms; a None here would mean the two scripts disagree.
    assert all(g is not None for r in ana.RULES for g in gains[r])
    keep, reset = ana.pooled(mats["s7t37"]), ana.pooled(mats["s7t37r"])
    diff = np.array(gains["s7t37r"]) - np.array(gains["s7t37"])

    def ci(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return f"{100 * lo:+.1f} to {100 * hi:+.1f}"

    L = ["# evidence/16: added after the results were read", "",
         "Outside spec.md, no pass line. Same bootstrap as ana.py (2000 resamples of seeds 1 to 20, "
         f"`default_rng({ana.BOOT_SEED})`, all stages from the same resampled seeds). Values in percent.", "",
         "| quantity | point | 95 % interval |", "|---|---|---|"]
    for r in ana.RULES:
        g = ana.gain_of(mats[r], [mats[a] for a in ana.ARMS])
        L.append(f"| gain of {r} against the line through the two fixed arms | {100 * g:+.1f} | {ci(gains[r])} |")
    L += [f"| delay of s7t37r relative to s7t37 | {100 * (reset[0] / keep[0] - 1):+.1f} | {ci(delay_ratio)} |",
          f"| polls per run of s7t37r relative to s7t37 | {100 * (reset[1] / keep[1] - 1):+.1f} | {ci(polls_ratio)} |",
          "", f"Share of resamples with gain difference s7t37r minus s7t37 above zero: {100 * (diff > 0).mean():.1f} %.",
          "", "Per-seed delay in seconds (`lat4 / n4`), seeds 1 to 20:", ""]
    for s in stages:
        L.append(f"- {s}: " + " ".join(f"{x:.1f}" for x in mats[s][:, 0] / mats[s][:, 1] / 1000))
    open(f"{HERE}/after.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
