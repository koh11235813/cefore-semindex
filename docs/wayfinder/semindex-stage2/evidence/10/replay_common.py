"""Shared pieces of the evidence/10 replay scripts (replay_calib.py, replay_retune.py).

The replay model is evidence/09/replay.py (a verbatim copy of an earlier project's value-check), imported from
there and never edited; `hull_delay` is imported from evidence/09/ana.py for the same reason.

Rule mapping from the live s6 stages to replay rules (all `ideal` information mode; the live engine re-decides
on answer or timeout, which evidence/09's README matches to the replay's ideal form):
  s6b0p5 / s6b1p359 / s6b3p695 / s6b10p043   R.Fixed(x), every interval jittered +/-10 %
  s6short  {"intervals_s": [10.043, 3.695], "jitter_frac": [0.1, 0.1]}   R.Table((10.043, 3.695)): wait jittered
  s6wait   {"intervals_s": [10.043, 5.0],   "jitter_frac": [0.1, 0.0]}   R.SlotBlind(10.043, wait=5.0): wait NOT jittered

Definitions (per run = one (p, seed) schedule, one jitter draw; then the mean over REPS draws per seed):
  delay_ms  mean over records that crossed all 4 hops inside the window of (4th-hop stored time - creation - 0.09 s)
  bytes_rh  (polls * INT_B + answered polls * REP_B[p]) / (4 * record-hops)   (replay.simulate's B/rh)
  waste     1 - answered polls / polls, polls sent in [10, 130] s on the 8 directed pairs
  polls     polls per run on the 8 directed pairs in the window
  cens      1 - completed 4-hop record walks / generation times (replay.simulate's cnt / total)

Seeding (replay_runs.py's paired convention): random.Random(hash(tag + (p, seed, rep)) & 0xffffffff) where tag is
the rule's numeric interval tuple. Only numbers go into the tag: str hashes are salted per process, so a string
would make every run draw different jitter.
"""
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV09 = os.path.join(HERE, os.pardir, "09")
# The worktree's own src: evalgen is used only to regenerate training schedules (seeds 21-120); the stored runs were
# made by semindex 5365e07, and schedule_from_experiment is asserted equal to the stored files on seeds 1-3.
SRC = os.path.abspath(os.path.join(HERE, *[os.pardir] * 5, "src"))
# EV09 goes BEFORE this directory: evidence/10 has its own ana.py (live half), and `import ana` must get evidence/09's.
sys.path[:0] = [EV09, SRC]

import replay as R  # noqa: E402  evidence/09/replay.py, unchanged
from ana import hull_delay  # noqa: E402  evidence/09/ana.py's copy of frontier.py's hull_delay

# REP_B (reply bytes per answered poll, measured on s4a) only has p <= 0.4. replay.simulate reads REP_B[p] for its
# bytes term unconditionally, so p 0.5/0.6 would raise KeyError. NaN here makes their bytes NaN while delay, waste,
# polls and cens are computed as usual. This mutates the imported dict in this process only; replay.py is untouched.
R.REP_B.setdefault(0.5, float("nan"))
R.REP_B.setdefault(0.6, float("nan"))

STAGE_SCHED = "s6b0p5"  # link + record schedules are identical across the six s6 stages for a (p, seed)
REPS = 10
ARMS = {"s6b0p5": 0.5, "s6b1p359": 1.359, "s6b3p695": 3.695, "s6b10p043": 10.043}
B10 = 10.043  # post-answer interval of both unchanged rules and of every retune candidate


def arm_rules():
    """(key, policy, tag) for the four fixed arms."""
    return [(st, R.Fixed(x), (x,)) for st, x in ARMS.items()]


def unchanged_rules():
    """(key, policy, tag) for the two rules that ran live unchanged from s5; tags as replay_runs.py."""
    return [("s6short", R.Table((B10, 3.695)), (B10, 3.695)),
            ("s6wait", R.SlotBlind(B10, wait=5.0), (B10, 5.0))]


def schedule_from_experiment(d):
    """(down intervals per link 0..3, record times per origin) from an experiment dict, mirroring R.load_schedule."""
    links = [[] for _ in range(4)]
    cur = {}
    for e in sorted((e for e in d["events"] if e["type"] in ("link_down", "link_up")), key=lambda e: e["at"]):
        k = min(e["nodes"])
        if e["type"] == "link_down":
            cur[k] = e["at"]
        else:
            links[k].append((cur.pop(k), e["at"]))
    for k, a in cur.items():
        links[k].append((a, 1e9))
    rec = {host: sorted({x["at_s"] for x in d["schedule"][host]}) for host in ("0", "4")}
    return links, rec


def gen_schedule(p, seed):
    """Schedule of an s6 run that was never run (training seeds): regenerated with the s6 generator settings."""
    from semindex.evalgen import build_experiment
    d = build_experiment(p, seed, STAGE_SCHED, "/nonexistent", k=4, store_synchronous="NORMAL",
                         disconnect_duration_s=(8.0, 12.0), state_poll_interval_choices_s=[0.5],
                         state_poll_choice_jitter_frac=0.1)
    return schedule_from_experiment(d)


def stored_schedule(p, seed, stage=STAGE_SCHED):
    """Schedule of a stored s6 run (the live seeds 1-20). R.load_schedule's default stage is s4a: always pass s6."""
    return R.load_schedule(p, seed, stage)


def check_loader():
    """Assert the regenerated schedules equal the stored ones on seeds 1-3 at p 0.2/0.4 (the loader's only check)."""
    for p in (0.2, 0.4):
        for s in (1, 2, 3):
            assert gen_schedule(p, s) == stored_schedule(p, s), (p, s)


def per_seed(pol, tag, links, rec, p, seed, reps=REPS):
    """Means over `reps` jitter draws of one rule on one schedule (delay over draws with a defined delay)."""
    L = [R.Link(iv) if not isinstance(iv, R.Link) else iv for iv in links]
    total = len(rec["0"]) + len(rec["4"])
    d, b, w, n, c = [], [], [], [], []
    for rep in range(reps):
        rng = random.Random(hash(tuple(tag) + (p, seed, rep)) & 0xffffffff)
        dd, bb, ww, cc, npoll = R.simulate(pol, L, rec, p, rng)
        if dd == dd:
            d.append(dd * 1000)
        b.append(bb)
        w.append(ww)
        n.append(npoll)
        c.append(1 - cc / total)
    m = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    return dict(delay_ms=m(d), bytes_rh=m(b), waste=m(w), polls=m(n), cens=m(c))


def seed_mean(x, w):
    """ana.py d4's mean: rows with NaN delay (column 0) are dropped from every column; w are bootstrap weights."""
    ok = ~np.isnan(x[:, 0])
    return np.sum(x[ok] * w[ok, None], 0) / np.sum(w[ok])


def d4(pol, arms, w):
    """[4] = rule's seed-mean delay - lower hull of the arms' seed-mean (bytes, delay) at the rule's bytes (ms).

    pol and each arms[k] are (n_seeds, 2) arrays of (delay_ms, bytes_rh). NaN when the rule's bytes fall outside
    the arms' bytes range (hull_delay's convention).
    """
    pts = [tuple(seed_mean(a, w)[[1, 0]]) for a in arms]
    m = seed_mean(pol, w)
    return m[0] - hull_delay(pts, m[1])


def boot_weights(n, reps, seed):
    """Multinomial seed-resampling weights, one row per bootstrap rep; shared by everything compared in a rep."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, [1 / n] * n, size=reps).astype(float)


def ci95(v):
    """Percentile CI over the finite bootstrap values of 400 reps (NaN pair unless more than 320 are finite, as ana.py)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) <= 320:
        return math.nan, math.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
