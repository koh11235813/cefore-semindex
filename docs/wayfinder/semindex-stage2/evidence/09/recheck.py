"""Two checks of the s5 numbers that do not go through attrib.py (written separately from perrun.py and ana.py).

1. 4-hop delay and censored share straight from evalagg's discovery.csv (rows with hop_distance 4, pooled
   over seeds and both directions), against live_main.csv's `delay` and `cens4`, for every (stage, p) there.
2. Reply bytes per answered state poll, with each reply's state name rebased to s4a's mean group length
   (14.92, as ana.py does), per (stage, p) from perrun.csv. replay.py uses one REP_B per p for every arm,
   so this shows how far the live reply size moves with the arm.

Usage: AGG=<evalagg output dir> python3 recheck.py > recheck.txt
"""
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
AGG = os.environ["AGG"]
REF_G = 14 + 0.90 + 0.02  # same constant as ana.py
STAGES = ("s5short", "s5wait", "s5b0p5", "s5b1p359", "s5b3p695", "s5b10p043", "s5c")


def recount_4hop():
    """Print one line per (stage, p) of live_main.csv and the number of cells that match."""
    live = {(r["stage"], float(r["p"])): r for r in csv.DictReader(open(f"{HERE}/live_main.csv"))}
    n = ok = 0
    for st in STAGES:
        acc = defaultdict(lambda: [0, 0, 0])  # p -> [latency sum, uncensored, censored]
        for r in csv.DictReader(open(f"{AGG}/{st}/discovery.csv")):
            if r["hop_distance"] != "4":
                continue
            a = acc[float(r["p"])]
            if r["censored"] == "true":
                a[2] += 1
            else:
                a[0] += int(r["latency_ms"])
                a[1] += 1
        for p, (s, u, c) in sorted(acc.items()):
            if (st, p) not in live:
                continue
            d, cf = s / u, c / (u + c)
            row = live[(st, p)]
            same = abs(d - float(row["delay"])) < 0.5 and abs(cf - float(row["cens4"])) < 1e-9
            n += 1
            ok += same
            print(f"{st} p{p:g}: delay {d:.1f} ms (live_main {float(row['delay']):.1f}), "
                  f"censored {cf:.4f} (live_main {float(row['cens4']):.4f}) {'same' if same else 'DIFF'}")
    print(f"4-hop recount: {ok}/{n} cells match")


def reply_bytes():
    """Print the name-rebased reply bytes per answered poll for each (stage, p) in perrun.csv."""
    acc = defaultdict(lambda: [0.0, 0.0])
    for r in csv.DictReader(open(f"{HERE}/perrun.csv")):
        dg = float(r["group_len"]) - REF_G
        a = acc[(r["stage"], float(r["p"]))]
        for h in (1, 2, 3, 4):
            answered = float(r[f"ngap{h}"]) - float(r[f"nnorep{h}"])
            a[0] += float(r[f"repB{h}"]) - dg * answered
            a[1] += answered
    for (st, p), (b, n) in sorted(acc.items()):
        print(f"{st} p{p:g}: reply bytes per answered poll (names rebased) {b / n:.1f}")


if __name__ == "__main__":
    recount_4hop()
    reply_bytes()
