#!/usr/bin/env python3
"""Selecting score, frontier gain, and a-against-b verdict from eval24.csv.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B ana24.py <eval.csv> [--dry]

Reads the named csv, `../22/weight.json`, and (for its mandatory reproduction gate)
`../23/eval.csv` and `../23/ana.txt`.  Prints no clock time and no host path, so a
second run is byte-identical.
"""
import importlib.util
import json
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

ARM_STAGES = PR20.ARM_STAGES
BOOT_N = PR20.BOOT_N
BOOT_SEED = PR20.BOOT_SEED
COL = 0

with open(os.path.join(EV22, "weight.json"), encoding="utf-8") as _fh:
    WEIGHT = json.load(_fh)["weight_ms_per_poll"]

JACK_SE = 1.685586
POLL_MID = 176.220
BAND_MS = 297.0

ARM_NAMES = ("const3.695", "const6.000", "const10.043")
MODEL_NAMES = ("a", "b")
ORDER = ARM_NAMES + ("F",) + MODEL_NAMES
ARM_KEY = dict(zip(ARM_NAMES, ARM_STAGES))

INT_COLS = ("seed", "stream", "n4", "c4", "polls", "ord_poll", "ord_resched", "k",
            "reads", "sends", "ties", "n_short", "n_mid", "n_long")
FLOAT_COLS = ("lat_ms", "censored_ms", "delay_ms", "score_ms", "score_dropped_ms",
              "weight_ms_per_poll")
DRY_GRID = (range(1, 3), range(0, 2))
REAL_GRID = (range(81, 101), range(0, 40))


class Fail(Exception):
    """A check or archive reproduction gate that did not hold."""


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
    """Parse an evaluation csv into `(cols, rows)` keeping empty cells as None."""
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


def check_roles(roles, want, where):
    """Assert every role's seeds sit in range and none meets 121..160."""
    union = sorted({int(s) for seeds in roles.values() for s in seeds})
    met = sorted(set(union) & set(range(121, 161)))
    if met:
        raise Fail("%s: seed set meets 121-160: %s" % (where, met))
    for role, seeds in sorted(roles.items()):
        lo, hi = want[role]
        bad = sorted(int(s) for s in seeds if not lo <= int(s) <= hi)
        if bad:
            raise Fail("%s: %s has %s outside %d-%d" % (where, role, bad[:4], lo, hi))
    PORT.check_seeds(union, where)
    return union


def check_one_digest(rows, column, where):
    """Assert every row carries the exact same digest string in `column`."""
    got = sorted({r[column] for r in rows})
    if len(got) != 1:
        raise Fail("%s: %d distinct %s values: %s" % (where, len(got), column, got[:3]))
    return got[0]


def check_weight(rows, want, where):
    """Assert every row was scored at `want`."""
    bad = sorted({r["weight_ms_per_poll"] for r in rows if r["weight_ms_per_poll"] != want})
    if bad:
        raise Fail("%s: wrong weight %r against %r" % (where, bad[0], want))
    return want


def check_score(rows, weight, where):
    """Recompute `delay` and `score` bit-identically from row terms."""
    for r in rows:
        delay = (r["lat_ms"] + r["censored_ms"]) / (r["n4"] + r["c4"])
        want = delay + weight * r["polls"]
        if want != r["score_ms"] or delay != r["delay_ms"]:
            raise Fail("%s: %s seed %d stream %d score mismatch" % (where, r["decider"], r["seed"], r["stream"]))
    return len(rows)


def check_grid(keys, seeds, streams, where):
    """Assert `(seed, stream)` pairs equal the full declared Cartesian product."""
    want = sorted((s, r) for s in seeds for r in streams)
    got = sorted(keys)
    if got != want:
        raise Fail("%s: got %d of %d grid pairs" % (where, len(got), len(want)))
    return len(want)


def check_complete(rows, deciders, keys, where):
    """Assert one row per `(decider, seed, stream)`."""
    want = sorted((d, s, r) for d in deciders for s, r in keys)
    got = sorted((r["decider"], r["seed"], r["stream"]) for r in rows)
    if got != want:
        raise Fail("%s: complete grid mismatch (%d vs %d)" % (where, len(got), len(want)))
    return len(got)


def check_k(rows, want_k, where):
    """Assert `k == want_k` across all rows."""
    got = sorted({r["k"] for r in rows})
    if got != [want_k]:
        raise Fail("%s: k=%s, expected [%d]" % (where, got, want_k))
    return got[0]


def per_seed(rows, seeds):
    """Return `(len(seeds), 4)` array of `[n4, c4, lat, polls]` summed per seed."""
    by = {s: [0.0, 0.0, 0.0, 0.0] for s in seeds}
    for r in rows:
        acc = by[r["seed"]]
        acc[0] += r["n4"]
        acc[1] += r["c4"]
        acc[2] += r["lat_ms"]
        acc[3] += r["polls"]
    if len({len([r for r in rows if r["seed"] == s]) for s in seeds}) != 1:
        raise Fail("per_seed: unequal run counts across seeds")
    return np.array([by[s] for s in seeds], float)


def check_row_order(row, n4, c4, lat, polls, where):
    """Assert row order `[n4, c4, lat, polls]` and round-trip through `PR20.vals`."""
    want = [n4, c4, lat, polls]
    if list(row) != want:
        raise Fail("%s: row %r != %r" % (where, list(row), want))
    delay, per, cens = PR20.vals(np.array([row], float))
    if delay != lat / n4 or per != polls or cens != c4 / (n4 + c4):
        raise Fail("%s: vals readback mismatch" % where)
    return want


def check_scale(m_dict, scaled_dict, key, where, tol=1e-9):
    """Assert frontier gain is invariant under scaling all stages by the stream count."""
    here = PR20.gain(m_dict, key, COL)
    there = PR20.gain(scaled_dict, key, COL)
    if (here is None) != (there is None):
        raise Fail("%s: scale mismatch %r vs %r" % (where, here, there))
    if here is not None and abs(here - there) > tol:
        raise Fail("%s: gain moved %.9f -> %.9f" % (where, here, there))
    return here


def check_boot(idx, n, where):
    """Assert resample matrix shape `(2000, n)` and index bounds."""
    if tuple(idx.shape) != (BOOT_N, n):
        raise Fail("%s: shape %r != (%d, %d)" % (where, tuple(idx.shape), BOOT_N, n))
    if int(idx.min()) < 0 or int(idx.max()) >= n:
        raise Fail("%s: index out of range" % where)
    return idx.shape


def compute_gain_with_ci(m_dict, key, idx):
    """Return `(point, lo_pct, hi_pct, hw_pct, dropped, word)` using drop-and-count."""
    point = PR20.gain(m_dict, key, COL)
    if point is None:
        return None, None, None, None, BOOT_N, "unreadable"
    vals = []
    dropped = 0
    for row_idx in idx:
        sub = {k: {COL: v[COL][row_idx]} for k, v in m_dict.items()}
        g = PR20.gain(sub, key, COL)
        if g is None:
            dropped += 1
        else:
            vals.append(g)
    lo, hi, hw = PR20.ci(vals)
    return point, lo, hi, hw, dropped, PR20.word(lo, hi)


def check_reproduce_prev():
    """Gate (`spec.md` section 18 item 7): reproduce `../23/ana.txt`'s 7 means and 4 gains
    from `../23/eval.csv`.
    """
    prev_csv = os.path.join(EV23, "eval.csv")
    prev_txt = os.path.join(EV23, "ana.txt")
    _cols, rows = read_csv(prev_csv)
    seeds = sorted({r["seed"] for r in rows})

    txt = open(prev_txt, encoding="utf-8").read()
    want_means = {}
    for m in re.finditer(r"^\s+(const3\.695|const6\.000|const10\.043|F|untuned|a|b)\s+([0-9]+\.[0-9]{3})\s+",
                         txt, re.M):
        want_means[m.group(1)] = float(m.group(2))
    if len(want_means) != 7:
        raise Fail("reproduce gate: parsed %d means from ../23/ana.txt, expected 7" % len(want_means))

    for dec, want_m in want_means.items():
        got_m = statistics.fmean(r["score_ms"] for r in rows if r["decider"] == dec)
        if "%.3f" % got_m != "%.3f" % want_m:
            raise Fail("reproduce gate: %s mean %.3f != ../23/ana.txt %.3f" % (dec, got_m, want_m))

    want_gains = {}
    for m in re.finditer(
            r"^\s+(F|untuned|a|b)\s+(-?[0-9]+\.[0-9]{4})\s+(-?[0-9]+\.[0-9]{3})\s+%\s+to\s+(-?[0-9]+\.[0-9]{3})\s+%",
            txt, re.M):
        want_gains[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    if len(want_gains) != 4:
        raise Fail("reproduce gate: parsed %d gains from ../23/ana.txt, expected 4" % len(want_gains))

    m_prev = {}
    prev_order = ARM_NAMES + ("F", "untuned", "a", "b")
    for dec in prev_order:
        drows = [r for r in rows if r["decider"] == dec]
        m_prev[ARM_KEY.get(dec, dec)] = {COL: per_seed(drows, seeds)}
    idx = PR20.boot_index(len(seeds))
    for dec, (wg, wlo, whi) in want_gains.items():
        g, lo, hi, _hw, _dr, _w = compute_gain_with_ci(m_prev, dec, idx)
        if "%.4f" % g != "%.4f" % wg or "%.3f" % lo != "%.3f" % wlo or "%.3f" % hi != "%.3f" % whi:
            raise Fail("reproduce gate: %s gain (%.4f, %.3f, %.3f) != ../23/ana.txt (%.4f, %.3f, %.3f)"
                       % (dec, g, lo, hi, wg, wlo, whi))
    return len(want_means), len(want_gains)


def main():
    """Analyse `eval24.csv` (or `eval24_dry.csv`) and run every check and mutation."""
    argv = sys.argv[1:]
    is_dry = "--dry" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if len(paths) != 1:
        raise SystemExit("usage: ana24.py <eval.csv> [--dry]")
    csv_path = paths[0]

    head("0  WHAT THIS IS, AND ARCHIVE REPRODUCTION GATE")
    if is_dry:
        say("DRY RUN on training seeds 1 and 2 (bootstrap degenerate; band verdict not read).")
    else:
        say("REAL RUN on selection seeds 81 to 100 of the main cell.")
    n_m, n_g = check_reproduce_prev()
    say("REPRODUCE GATE PASSED: reproduced %d means and %d gains (+95%% CI) from ../23/eval.csv"
        % (n_m, n_g))

    head("1  THE CSV, DIGESTS, GRID AND SEED ROLES")
    cols, rows = read_csv(csv_path)
    seeds = sorted({r["seed"] for r in rows})
    streams = sorted({r["stream"] for r in rows})
    keys = sorted({(r["seed"], r["stream"]) for r in rows})
    decs = sorted({r["decider"] for r in rows})
    if sorted(decs) != sorted(ORDER):
        raise Fail("unexpected deciders %r != %r" % (decs, sorted(ORDER)))
    say("rows            %d" % len(rows))
    say("columns         %s" % ",".join(cols))
    say("deciders        %s" % ", ".join(ORDER))
    say("seed column     min %d max %d, %d distinct" % (seeds[0], seeds[-1], len(seeds)))
    say("stream column   min %d max %d, %d distinct" % (streams[0], streams[-1], len(streams)))
    say("script digest   %s" % check_one_digest(rows, "script_sha", "csv"))
    for nm in ORDER:
        sub = [r for r in rows if r["decider"] == nm]
        say("  %-12s %4d rows, model digest %s"
            % (nm, len(sub), check_one_digest(sub, "model_sha", nm)))
    check_weight(rows, WEIGHT, "csv")
    check_score(rows, WEIGHT, "csv")
    g_seeds, g_streams = DRY_GRID if is_dry else REAL_GRID
    check_grid(keys, g_seeds, g_streams, "csv")
    check_complete(rows, ORDER, keys, "csv")
    check_k(rows, 1, "csv")
    roles = {"fit": list(range(1, 71)), "held out": list(range(71, 81)),
             "selection": list(range(81, 101)), "scored": seeds}
    want_r = {"fit": (1, 70), "held out": (71, 80), "selection": (81, 100),
              "scored": (1, 80) if is_dry else (81, 100)}
    check_roles(roles, want_r, "ana24")
    say("OK: weight, score recomputation, grid, completeness, k=1 and seed roles all pass")

    head("2  THE SELECTING STATISTIC: MEAN SCORE OVER RUNS")
    mean, polls_run, per_seed_sc = {}, {}, {}
    say("  %-12s %14s %16s %15s   %s"
        % ("decider", "mean score ms", "mean polls/run", "mean delay ms", "ties / decisions"))
    for nm in ORDER:
        sub = [r for r in rows if r["decider"] == nm]
        mean[nm] = statistics.fmean(r["score_ms"] for r in sub)
        polls_run[nm] = statistics.fmean(r["polls"] for r in sub)
        delay_m = statistics.fmean(r["delay_ms"] for r in sub)
        per_seed_sc[nm] = {s: statistics.fmean(r["score_ms"] for r in sub if r["seed"] == s)
                           for s in seeds}
        ties = sum(r["ties"] for r in sub) if sub[0]["ties"] is not None else None
        reads = sum(r["reads"] for r in sub) if sub[0]["reads"] is not None else None
        say("  %-12s %14.3f %16.3f %15.3f   %s"
            % (nm, mean[nm], polls_run[nm], delay_m,
               "--" if ties is None else "%d / %d" % (ties, reads)))
    say()
    say("ranked, lower first: %s"
        % ", ".join("%s %.1f" % (n, mean[n]) for n in sorted(ORDER, key=lambda n: mean[n])))

    head("3  THE REPORTED FRONTIER GAIN, WITH ITS 95 % INTERVAL")
    m_dict = {}
    for nm in ORDER:
        sub = [r for r in rows if r["decider"] == nm]
        m_dict[ARM_KEY.get(nm, nm)] = {COL: per_seed(sub, seeds)}
    first_arm = [r for r in rows if r["decider"] == ARM_NAMES[0] and r["seed"] == seeds[0]]
    check_row_order(m_dict[ARM_STAGES[0]][COL][0],
                    sum(r["n4"] for r in first_arm), sum(r["c4"] for r in first_arm),
                    sum(r["lat_ms"] for r in first_arm), sum(r["polls"] for r in first_arm),
                    "first row check")
    scaled = {k: {COL: v[COL] / len(streams)} for k, v in m_dict.items()}
    for k_arm in ARM_STAGES:
        dl, pl, cs = PR20.vals(m_dict[k_arm][COL])
        say("  %-14s delay %10.3f ms   polls %10.3f/seed (%8.3f/run)   censored %.4f"
            % (k_arm, dl, pl, pl / len(streams), cs))
    idx = PR20.boot_index(len(seeds))
    check_boot(idx, len(seeds), "ana24 boot")
    say()
    say("  %-12s %9s  %-22s %11s %9s  %s"
        % ("decider", "gain", "95 % interval", "half width", "dropped", "word"))
    for nm in ("F",) + MODEL_NAMES:
        check_scale(m_dict, scaled, nm, "scale %s" % nm)
        g, lo, hi, hw, dropped, word = compute_gain_with_ci(m_dict, nm, idx)
        if g is None:
            say("  %-12s no gain (%s)" % (nm, word))
        else:
            say("  %-12s %9.4f  %8.3f %% to %8.3f %%  %8.3f %% %9d  %s"
                % (nm, g, lo, hi, hw, dropped, word))

    head("4  HOW THE CHOICE BETWEEN a AND b IS MADE")
    ma, mb = mean["a"], mean["b"]
    gap = abs(ma - mb)
    winner = "a" if ma < mb else "b"
    diffs = np.array([per_seed_sc["a"][s] - per_seed_sc["b"][s] for s in seeds], float)
    boot_diffs = np.sort(diffs[idx].mean(axis=1))
    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))
    contains_zero = ci_lo <= 0.0 <= ci_hi
    pt_ab = float(ma - mb)
    non_cancel = JACK_SE * abs(polls_run["a"] - polls_run["b"])

    say("mean score      a %.3f ms, b %.3f ms" % (ma, mb))
    say("gap             %.3f ms, lower being %s" % (gap, winner))
    say("band            %.3f ms (%.6f x %.3f = %.4f ms)" % (BAND_MS, JACK_SE, POLL_MID, JACK_SE * POLL_MID))
    say("non-cancelling  %.6f x |%.3f - %.3f| = %.3f ms"
        % (JACK_SE, polls_run["a"], polls_run["b"], non_cancel))
    say("score(a) - score(b): point %.3f ms, 95 %% interval %.3f to %.3f ms, contains 0: %s"
        % (pt_ab, ci_lo, ci_hi, contains_zero))
    if gap < BAND_MS:
        verdict = "CLOSE"
    elif contains_zero:
        verdict = "SEPARATED, NOT RESOLVED"
    else:
        verdict = "SEPARATED"
    say("VERDICT: %s" % verdict)
    say("CHOSEN CANDIDATE: %s" % winner)

    head("5  EVERY CHECK SHOWN FAILING ON A MUTATED INPUT")
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
    fired.append(show_mut("roles/measuring",
                          lambda: check_roles({"bad": [121]}, {"bad": (1, 160)}, "mut/121")))
    fired.append(show_mut("weight moved",
                          lambda: check_weight([dict(rows[0], weight_ms_per_poll=100.0)], WEIGHT, "mut/wt")))
    fired.append(show_mut("score recomputation moved",
                          lambda: check_score([dict(rows[0], polls=rows[0]["polls"] + 1)], WEIGHT, "mut/sc")))
    fired.append(show_mut("complete/missing row",
                          lambda: check_complete(rows[1:], ORDER, keys, "mut/comp")))
    fired.append(show_mut("grid/missing pair",
                          lambda: check_grid(keys[1:], g_seeds, g_streams, "mut/grid")))
    fired.append(show_mut("row order [lat, n, c, polls]",
                          lambda: check_row_order([1000.0, 240.0, 0.0, 100.0],
                                                  240.0, 0.0, 1000.0, 100.0, "mut/ord")))
    bad_scaled = dict(scaled)
    bad_scaled[ARM_STAGES[0]] = {COL: scaled[ARM_STAGES[0]][COL] * 2.0}
    fired.append(show_mut("scale one stage alone",
                          lambda: check_scale(m_dict, bad_scaled, "F", "mut/scale")))
    fired.append(show_mut("boot/wrong shape",
                          lambda: check_boot(idx, len(seeds) + 1, "mut/boot")))
    say()
    say("%d of %d mutations produced a failure" % (sum(fired), len(fired)))
    say()
    say("end of ana24.py")
    return 0 if sum(fired) == len(fired) else 1


if __name__ == "__main__":
    sys.exit(main())
