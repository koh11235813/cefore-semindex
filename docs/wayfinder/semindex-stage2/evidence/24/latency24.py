#!/usr/bin/env python3
"""Time one decision of M in three parts on CPU beside 100 ms, for both candidates.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B latency24.py

Reads the first 1200 distinct `state_poll_reschedule` decisions of p 0.2 seeds 1 to 8
in log order (precheck.py part 9's exact row set).  No verdict is attached.  Every
non-timing line sits above `TIME_MARKER` so deterministic prefixes match with `cmp`.
"""
import hashlib
import os
import sys
import time

for _name, _value in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
                      ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_name, _value)

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if EV22 not in sys.path:
    sys.path.insert(0, EV22)

import charges as CH   # noqa: E402
import features as FE  # noqa: E402
import m24             # noqa: E402
import observe as OBS  # noqa: E402
import port as PORT    # noqa: E402

TIMING_CALLS = 1200
TIMING_SEEDS = tuple(range(1, 9))
TICK_MS = 100.0
TIME_MARKER = ">>> TIMINGS BELOW THIS LINE DO NOT REPRODUCE BYTE FOR BYTE <<<"


class Fail(Exception):
    """A check that did not hold."""


def say(text=""):
    """Print one line of output."""
    print(text, flush=True)


def head(title):
    """Print a section heading."""
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


def file_sha256(path):
    """Hex sha256 digest of a file."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def quant(values, q):
    """Order statistic matching precheck.py."""
    ordered = sorted(values)
    return ordered[min(int(q * (len(ordered) - 1) + 0.5), len(ordered) - 1)]


def time_calls(fn, items):
    """Return list of per-call wall times in milliseconds, unbatched."""
    out = []
    for item in items:
        t0 = time.perf_counter()
        fn(item)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def main():
    """Run deterministic checks, repeat checks, mutations, and CPU timings."""
    head("0  SCOPE, DIGESTS AND DETERMINISTIC CHECKS")
    say("row set         first %d distinct decisions of p 0.2 seeds %d to %d in log order"
        % (TIMING_CALLS, TIMING_SEEDS[0], TIMING_SEEDS[-1]))
    say("code digests (sha256):")
    for fname in ("latency24.py", "m24.py", "features.py", "charges.py"):
        say("  %-14s %s" % (fname, file_sha256(os.path.join(HERE, fname))))

    inputs, opened = [], []
    for seed in TIMING_SEEDS:
        events, _t0 = CH.load_run("0.2", seed)
        opened.append(seed)
        for node in sorted(events):
            inputs.extend(OBS.from_logged_rows(events[node]))
        if len(inputs) >= TIMING_CALLS:
            break
    if len(inputs) < TIMING_CALLS:
        raise Fail("only %d decisions found, expected >= %d" % (len(inputs), TIMING_CALLS))
    inputs = inputs[:TIMING_CALLS]
    say("opened seeds    %s, kept %d decisions" % (opened, len(inputs)))

    decs = {}
    sample_obs = m24.first_logged_resched_obs("0.2", 1, max_seed=20)
    for cname in ("a", "b"):
        cpath = os.path.join(HERE, "coef_%s.npy" % cname)
        say("model %-2s        %s  sha256 %s" % (cname, os.path.basename(cpath), file_sha256(cpath)))
        dec = m24.M.from_file(cpath)
        vals, arm = m24.repeat_check(dec, sample_obs, "latency24 %s" % cname)
        say("  repeat check %-2s: %s -> %.3f s" % (cname, [round(v, 6) for v in vals], arm))
        decs[cname] = dec

    head("1  EVERY CHECK SHOWN FAILING ON A MUTATED INPUT")
    fired = []

    def show_mut(label, fn):
        try:
            fn()
        except (Fail, CH.Fail, m24.Fail, SystemExit) as exc:
            say("  %-32s FIRES: %s" % (label, exc))
            return 1
        say("  %-32s DID NOT FIRE" % label)
        return 0

    fired.append(show_mut("charges.load_run > 20", lambda: CH.load_run("0.2", 21)))
    fired.append(show_mut("port.check_seeds > 100", lambda: PORT.check_seeds([101], "mut/101")))
    fired.append(show_mut("repeat check mutated",
                          lambda: m24.repeat_check(decs["a"], sample_obs, "mut/rep", mutate=True)))
    say()
    say("%d of %d mutations produced a failure" % (sum(fired), len(fired)))

    head("2  CPU TIMINGS IN THREE PARTS (NO VERDICT)")
    say(TIME_MARKER)
    records = [OBS.observation(st, res, hist, now, nb) for nb, st, res, hist, now in inputs]
    vectors = [FE.vector(r) for r in records]

    t_obs = time_calls(lambda got: OBS.observation(got[1], got[2], got[3], got[4], got[0]), inputs)
    t_vec = time_calls(FE.vector, records)

    say("  %-26s %7s %10s %10s %10s %10s"
        % ("part", "calls", "median ms", "p99 ms", "max ms", "the tick"))
    say("  %-26s %7d %10.4f %10.4f %10.4f %10.1f"
        % ("observe.observation", len(t_obs), quant(t_obs, 0.5), quant(t_obs, 0.99), max(t_obs), TICK_MS))
    say("  %-26s %7d %10.4f %10.4f %10.4f %10.1f"
        % ("features.vector", len(t_vec), quant(t_vec, 0.5), quant(t_vec, 0.99), max(t_vec), TICK_MS))

    for cname in ("a", "b"):
        dec = decs[cname]
        chosen = []
        t_dec = time_calls(lambda v, d=dec: chosen.append(m24.ARMS[m24.pick_min(d._values(v))[0]]), vectors)
        bad = [a for a in chosen if a not in m24.ARMS]
        if bad:
            raise Fail("cand %s returned arm outside ARMS: %r" % (cname, bad[0]))
        say("  %-26s %7d %10.4f %10.4f %10.4f %10.1f"
            % ("cand %s: 3 values+argmin" % cname, len(t_dec),
               quant(t_dec, 0.5), quant(t_dec, 0.99), max(t_dec), TICK_MS))
    say()
    say("end of latency24.py")
    return 0 if sum(fired) == len(fired) else 1


if __name__ == "__main__":
    sys.exit(main())
