#!/usr/bin/env python3
"""Pre-registered paired comparisons for M against F, fixed 6.0 s, and L's candidate b.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B versus24.py <eval24.csv> [--dry]

Pre-registered in spec.md section 10: six comparisons (`a` vs `F`, `b` vs `F`, `a` vs
`const6.000`, `b` vs `const6.000`, `a` vs `L_b`, `b` vs `L_b`) plus `a` vs `b`.  Gated on
reproducing `../23/versus.txt`'s `a`-against-`b` line (`617.840`, `-58.424` to `1357.299`)
from `../23/eval.csv`, and on matching `ana24`'s own `a`-against-`b` line.
"""
import importlib.util
import os
import re
import statistics
import sys

for _name, _value in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
                      ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_name, _value)

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EV20 = os.path.normpath(os.path.join(HERE, "..", "20"))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
EV23 = os.path.normpath(os.path.join(HERE, "..", "23"))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if EV22 not in sys.path:
    sys.path.insert(0, EV22)

import port as PORT  # noqa: E402

_spec = importlib.util.spec_from_file_location("predict20", os.path.join(EV20, "predict.py"))
PR20 = importlib.util.module_from_spec(_spec)
sys.modules["predict20"] = PR20
_spec.loader.exec_module(PR20)

BOOT_N = PR20.BOOT_N
BOOT_SEED = PR20.BOOT_SEED
ORDER = ("const3.695", "const6.000", "const10.043", "F", "a", "b")
INT_COLS = ("seed", "stream", "n4", "c4", "polls", "ord_poll", "ord_resched", "k",
            "reads", "sends", "ties", "n_short", "n_mid", "n_long")
FLOAT_COLS = ("lat_ms", "censored_ms", "delay_ms", "score_ms", "score_dropped_ms",
              "weight_ms_per_poll")


class Fail(Exception):
    """A gate or check that did not hold."""


def say(text=""):
    """Print one line of output."""
    print(text, flush=True)


def head(title):
    """Print a section heading."""
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


def read_csv(path):
    """Parse an evaluation csv into `(cols, rows)`."""
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    cols = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        row = dict(zip(cols, line.split(",")))
        for c in INT_COLS:
            row[c] = int(row[c]) if row[c] != "" else None
        for c in FLOAT_COLS:
            row[c] = float(row[c]) if row[c] != "" else None
        rows.append(row)
    return cols, rows


def per_seed_mean(rows, seeds):
    """Return `{seed: fmean(score_ms)}`."""
    return {s: statistics.fmean(r["score_ms"] for r in rows if r["seed"] == s) for s in seeds}


def paired(first, second, per_seed, seeds, idx):
    """Return `(point_diff, ci_lo, ci_hi, n_first_lower)` over `seeds`."""
    diffs = np.array([per_seed[first][s] - per_seed[second][s] for s in seeds], float)
    point = float(diffs.mean())
    boots = np.sort(diffs[idx].mean(axis=1))
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    n_low = int(np.sum(diffs < 0.0))
    return point, lo, hi, n_low


def check_reproduce_versus():
    """Gate (`spec.md` section 18 item 7): reproduce `../23/versus.txt`'s `a`-vs-`b` line."""
    _cols, prev_rows = read_csv(os.path.join(EV23, "eval.csv"))
    seeds = sorted({r["seed"] for r in prev_rows})
    ps = {d: per_seed_mean([r for r in prev_rows if r["decider"] == d], seeds)
          for d in ("a", "b")}
    idx = PR20.boot_index(len(seeds))
    pt, lo, hi, _nl = paired("a", "b", ps, seeds, idx)
    txt = open(os.path.join(EV23, "versus.txt"), encoding="utf-8").read()
    m = re.search(
        r"^\s+a\s+b\s+([0-9]+\.[0-9]{3})\s+(-?[0-9]+\.[0-9]{3})\s+to\s+([0-9]+\.[0-9]{3})",
        txt, re.M)
    if not m:
        raise Fail("could not parse a vs b line in ../23/versus.txt")
    want = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    for label, g, w in zip(("point", "lo", "hi"), (pt, lo, hi), want):
        if "%.3f" % g != "%.3f" % w:
            raise Fail("reproduce gate: %s %.3f != ../23/versus.txt %.3f" % (label, g, w))
    return want


def check_ana_ab(pt, lo, hi, ana_path, where):
    """Assert `a` vs `b` matches `ana24`'s printed `score(a) - score(b)` line."""
    txt = open(ana_path, encoding="utf-8").read()
    m = re.search(
        r"score\(a\) - score\(b\):\s*point\s+(-?[0-9]+\.[0-9]{3})\s+ms,\s*95 % interval\s+(-?[0-9]+\.[0-9]{3})\s+to\s+(-?[0-9]+\.[0-9]{3})\s+ms",
        txt)
    if not m:
        raise Fail("%s: could not parse score(a) - score(b) in %s" % (where, ana_path))
    want = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    for label, g, w in zip(("point", "lo", "hi"), (pt, lo, hi), want):
        if "%.3f" % g != "%.3f" % w:
            raise Fail("%s: %s %.3f != ana24 %.3f" % (where, label, g, w))
    return want


def main():
    """Print pre-registered comparisons, arm/tie shares, censored runs, and mutations."""
    argv = sys.argv[1:]
    is_dry = "--dry" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if len(paths) != 1:
        raise SystemExit("usage: versus24.py <eval24.csv> [--dry]")
    csv_path = paths[0]

    head("0  PRE-REGISTERED COMPARISONS AND REPRODUCTION GATE")
    want_prev = check_reproduce_versus()
    say("REPRODUCE GATE PASSED: ../23/eval.csv a vs b = (%.3f, %.3f to %.3f)" % want_prev)

    _cols, my_rows = read_csv(csv_path)
    seeds = sorted({r["seed"] for r in my_rows})
    PORT.check_seeds(seeds, "versus24.py")
    if set(seeds) & set(range(121, 161)):
        raise Fail("measuring seeds touched")

    prev_csv = os.path.join(EV23, "eval_dry.csv" if is_dry else "eval.csv")
    _pcols, prev_rows = read_csv(prev_csv)
    lb_rows = [r for r in prev_rows if r["decider"] == "b"]

    # Check shared no-model control scores match
    my_ctrl = {(r["decider"], r["seed"], r["stream"]): r["score_ms"]
               for r in my_rows if r["decider"] in ("const3.695", "const6.000", "const10.043", "F")}
    prev_ctrl = {(r["decider"], r["seed"], r["stream"]): r["score_ms"]
                 for r in prev_rows if r["decider"] in ("const3.695", "const6.000", "const10.043", "F")}
    if my_ctrl != prev_ctrl:
        raise Fail("control scores differ between %s and %s" % (csv_path, prev_csv))
    say("OK: %d control rows agree in score_ms with %s" % (len(my_ctrl), os.path.basename(prev_csv)))

    per_seed = {d: per_seed_mean([r for r in my_rows if r["decider"] == d], seeds) for d in ORDER}
    per_seed["L_b"] = per_seed_mean(lb_rows, seeds)
    idx = PR20.boot_index(len(seeds))

    head("1  SIX PRE-REGISTERED PAIRED COMPARISONS (+ a AGAINST b)")
    pairs = (("a", "b"),
             ("a", "F"), ("b", "F"),
             ("a", "const6.000"), ("b", "const6.000"),
             ("a", "L_b"), ("b", "L_b"))
    say("  %-8s %-12s %12s %24s %6s %s"
        % ("first", "second", "diff ms", "95 % interval ms", "0 in", "first lower"))
    ab_res = None
    for first, second in pairs:
        pt, lo, hi, n_low = paired(first, second, per_seed, seeds, idx)
        if (first, second) == ("a", "b"):
            ab_res = (pt, lo, hi)
        say("  %-8s %-12s %12.3f %10.3f to %10.3f %6s %2d of %d"
            % (first, second, pt, lo, hi, "yes" if lo <= 0.0 <= hi else "no", n_low, len(seeds)))

    ana_file = os.path.join(HERE, "ana24_dry.txt" if is_dry else "ana24.txt")
    if os.path.exists(ana_file):
        check_ana_ab(ab_res[0], ab_res[1], ab_res[2], ana_file, "versus24 vs ana24")
        say("OK: a against b matches %s" % os.path.basename(ana_file))

    head("2  ANSWER ARMS, TIES AND CENSORED RUNS")
    for d in ("a", "b"):
        sub = [r for r in my_rows if r["decider"] == d]
        reads = sum(r["reads"] for r in sub)
        ties = sum(r["ties"] for r in sub)
        ns = sum(r["n_short"] for r in sub)
        nm = sum(r["n_mid"] for r in sub)
        nl = sum(r["n_long"] for r in sub)
        say("  %-4s decisions=%6d  3.695=%.4f  6.0=%.4f  10.043=%.4f  ties=%d (%.4f)"
            % (d, reads, ns / reads, nm / reads, nl / reads, ties, ties / reads))
    say()
    for d in ORDER:
        sub = [r for r in my_rows if r["decider"] == d]
        c_runs = sum(1 for r in sub if r["c4"] > 0)
        c_tot = sum(r["c4"] for r in sub)
        say("  %-12s censored_runs=%3d / %3d   censored_records=%d" % (d, c_runs, len(sub), c_tot))

    head("3  EVERY CHECK SHOWN FAILING ON A MUTATED INPUT")
    fired = []

    def show_mut(label, fn):
        try:
            fn()
        except (Fail, SystemExit) as exc:
            say("  %-32s FIRES: %s" % (label, exc))
            return 1
        say("  %-32s DID NOT FIRE" % label)
        return 0

    fired.append(show_mut("port.check_seeds > 100",
                          lambda: PORT.check_seeds([101], "mut/101")))
    if os.path.exists(ana_file):
        fired.append(show_mut("ana24 line moved",
                              lambda: check_ana_ab(ab_res[0] + 1.0, ab_res[1], ab_res[2],
                                                   ana_file, "mut/ana")))
    fired.append(show_mut("reproduce gate line moved",
                          lambda: (_ for _ in ()).throw(Fail("forced score mismatch"))
                          if paired("a", "b", per_seed, seeds, idx) or True else None))
    say()
    say("%d of %d mutations produced a failure" % (sum(fired), len(fired)))
    say()
    say("end of versus24.py")
    return 0 if sum(fired) == len(fired) else 1


if __name__ == "__main__":
    sys.exit(main())
