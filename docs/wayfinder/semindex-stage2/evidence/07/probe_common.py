"""Shared pieces of probe_analyze.py and probe_analyze2.py: stage list, block intervals, small stats.

The pickle these read is written by probe_extract.py (see its docstring for every field).
"""
from __future__ import annotations

import bisect
import gzip
import pickle

STAGES = ["s4c", "s4b0p5", "s4b1p359", "s4b3p695", "s4b10p043", "s4a"]
PS = [0.0, 0.1, 0.2, 0.3, 0.4]
# Placebo shifts of h2's block list. Records are created in bursts every 2 s and probes fire on
# burst instants, so exposure is phase-locked to bursts; 2-s multiples keep that phase while
# moving the block away from the real ones (probes repeat every 10 s, so +4/+6 s never land on one).
SHIFTS = [4000.0, 6000.0]


def load_valid(path):
    """probe_extract.py's per-run dicts, valid runs only."""
    return [r for r in pickle.load(gzip.open(path)) if r["valid"]]


class Blocks:
    """Sorted, non-overlapping block intervals with point / interval queries."""

    def __init__(self, ivs):
        self.iv = sorted(ivs)
        self.s = [a for a, _ in self.iv]

    def contains(self, t):
        j = bisect.bisect_right(self.s, t) - 1
        return j >= 0 and self.iv[j][0] <= t <= self.iv[j][1]

    def intersects(self, a, b):
        """Any block meets the open-closed interval (a, b]."""
        j = bisect.bisect_right(self.s, b) - 1  # last block starting at or before b
        return j >= 0 and self.iv[j][1] > a

    def overlap(self, a, b):
        tot = 0.0
        j = bisect.bisect_right(self.s, b) - 1
        while j >= 0 and self.iv[j][1] > a:
            tot += max(0.0, min(b, self.iv[j][1]) - max(a, self.iv[j][0]))
            j -= 1
        return tot


def q(xs, f):
    """Lower-index quantile f of xs (no interpolation); NaN when empty."""
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[int(f * (len(xs) - 1))]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def hop_class(h, o):
    """Label a hop by its role relative to h2 (into / out of / not touching) and its origin."""
    role = {1: "first", 2: "into_h2", 3: "out_h2", 4: "last"}[h]
    return f"{role}:{o}"
