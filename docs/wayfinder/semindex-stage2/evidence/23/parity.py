#!/usr/bin/env python3
"""Does one port simulation give the same bytes on the tuning host as on this machine?

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B parity.py

    --seeds 1,20,...   the seeds to run        (default: 1,20,21,80,81,100)
    --env              print the interpreter and library versions and stop

WHY THIS EXISTS
  The decision maker L has to run next to the model, i.e. on the tuning host, which means
  the replay runs there and not here.  Every number this effort has filed was produced
  here, so before anything is measured there, the two machines have to be shown to produce
  the same run bit for bit.  The output is deliberately free of paths, versions and times:
  the whole file is meant to be compared with `cmp`, so anything in it that differs
  BECAUSE the machines differ would hide the thing being checked.

WHAT ONE LINE SAYS
  seed, decider, the sha256 of the canonical JSON of the whole event dict, and
  `score.py`'s own terms for that run.  The digest is the strong half: two runs that agree
  on the score can still have made different decisions, while two runs that agree on the
  digest made the same run.

WHICH SEEDS, AND WHY THESE SIX
  1 and 20 are the ends of the range that HAS a logged run (`REAL_SEEDS[0.2]`), so they
  exercise the store reader.  21, 80, 81 and 100 have no run of their own: they borrow a
  donor's clock and build their outage timeline and record schedule with
  `semindex.evalgen` inside the process.  So both ways a run dict can come about are
  covered, and a parity failure cannot hide in the one this file happened not to use.
  21/80 are training seeds and 81/100 selection seeds, which is not what the check needs
  but is what the seed roles are, so the printed seed set also shows the two roles.

WHAT THIS CANNOT SHOW
  That L itself is reproducible.  No model runs here: the deciders are the three constant
  arms and the formula rule F, i.e. everything in the comparison EXCEPT L.  A forward pass
  on a GPU is not bit-reproducible across machines in general, and nothing here claims it.
"""
import argparse
import hashlib
import json
import os
import sys

# 2026-11-24: this script measured 2774 % cpu for a one-seed run on the 96-core tuning
# host -- numpy's BLAS opens one thread per core at import.  That host is shared and has
# no scheduler, and the brief caps this work at 8 processes, which a 96-thread pool
# breaks without anyone asking for it.  Set BEFORE `port` is imported below, because
# OpenBLAS and OpenMP read these once, when the library loads.  A cap and not a pin of
# the result: the replay does no float reduction that a thread count could reorder, and
# the capped run was shown to give the same bytes as the uncapped one.
for _name, _value in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
                      ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_name, _value)

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
sys.path.insert(0, EV22)
import port as PORT  # noqa: E402
import score as SCORE  # noqa: E402
import label as LABEL  # noqa: E402  F, and the run dict with the union choice set

R = PORT.R

MAIN_P = 0.2
DEFAULT_SEEDS = (1, 20, 21, 80, 81, 100)
# 2026-11-24: there is no second guard here against the held-out measuring seeds
# (121-160).  An earlier version had one, and it could not fire: `port.check_seeds`
# refuses every seed above `port.MAX_SEED` = 100, so a held-out seed never reached it and
# the branch and its printed intersection were constants.  A check that cannot fail is
# worse than no check, because it reads as evidence.  `port.check_seeds` below is the
# guard, and the mutated run `--seeds 1,121` is where it is shown refusing.

# 2026-11-24, the pair-stream seed convention: `weight.py`'s own, which is
# `R.rep_seed(f"pairlag:{STREAM_LABEL}", p, seed, rep)` with `STREAM_LABEL` the main
# cell's donor stage, used as BOTH `pair_lag_seed` and `pair_jitter_seed` (weight.py's
# `port_run` with `jitter` true, which is also what `label.py` runs).  The two streams get
# different kind prefixes inside `port.Engine._seed`, so one seed string still gives two
# independent streams.  The other convention in the directory, `label.py`'s
# `R.rep_seed("label:s11a", p, seed, 0)`, is NOT used here: a parity check has to pick one,
# and the comparison of candidates against the arms is the weight's.
LAG_PREFIX = f"pairlag:{LABEL.PR.STREAM_LABEL}"
REP = 0                                 # one stream per seed is enough to compare bytes


def pair_seed(p, seed):
    return R.rep_seed(LAG_PREFIX, p, seed, REP)


def deciders():
    """(name, factory) for every decider in the comparison except L.

    A FACTORY and not an object: `label.F` carries one interval per ordered pair and is
    mutated as the run goes, so one instance across two simulations would make the second
    run depend on the first.  The three constant arms do not care, but they are built the
    same way so that nothing here has two ways of making a decider.
    """
    out = [(f"const{arm:.3f}", (lambda a=arm: PORT.const_decider(a))) for arm in PORT.ARMS]
    out.append(("F", LABEL.F))
    return out


def one(seed, make_decider):
    """One simulation: (sha256 of the canonical event JSON, score terms)."""
    # `label.run_dict` and not a hand-built dict: it is what puts the union choice set, the
    # choice jitter and `redecide` on the run, and it is where the seed guard sits on this
    # path.  Every decider in the comparison therefore runs on the SAME choice set, which
    # F needs (it leaves the three arms as soon as it lengthens) and the arms do not.
    run = LABEL.run_dict(MAIN_P, seed)
    run = dict(run, decider=make_decider())
    base = pair_seed(MAIN_P, seed)
    events, _engine = PORT.sim(run, None, base, base)
    # `sort_keys` and the tightest separators, so that the digest is a function of the
    # content and not of the dict order the engine happened to build.  `allow_nan=False`
    # on purpose: a NaN would serialise as the non-JSON token `NaN`, which two json
    # versions need not spell the same way, and a NaN in an event is a defect to see rather
    # than to hash.
    blob = json.dumps(events, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode()).hexdigest(), SCORE.terms(events, run["t0"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--env", action="store_true")
    args = ap.parse_args()

    if args.env:
        # Printed by a separate switch and never into the compared output: the two machines
        # are KNOWN to differ here (python 3.12 against 3.13), and the point of the check
        # is that the run does not.
        import numpy
        print("python", sys.version.split()[0])
        print("numpy", numpy.__version__)
        print("weight", repr(LABEL.WEIGHT))
        return 0

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    # The guard, on the seed list this script computed and not only inside the engine.
    PORT.check_seeds(seeds, "parity.py")

    print("choices " + " ".join(repr(c) for c in PORT.union_choices()))
    print("seeds " + ",".join(str(s) for s in seeds))
    for seed in seeds:
        for name, make in deciders():
            sha, t = one(seed, make)
            n4, c4, lat, cens, polls = t
            print(f"seed {seed:3d} {name:10s} {sha} "
                  f"n4 {n4} c4 {c4} lat {lat!r} cens {cens!r} polls {polls} "
                  f"delay {SCORE.delay(t)!r} "
                  f"score {SCORE.delay(t) + LABEL.WEIGHT * polls!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
