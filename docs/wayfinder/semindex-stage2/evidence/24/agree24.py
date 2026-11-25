#!/usr/bin/env python3
"""Agreement of M's two candidates with the hindsight labels on seeds 81 to 100.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B agree24.py [--dry]

Carries its own label-file reader (reading the path and sha256 from `features.label_file`)
so `features.label_rows`'s cap at seed 80 stays intact for every other script of this
directory.  Reports agreement beside the pre-registered 0.5273 baseline with NO pass or
fail verdict.  Prints no clock time or machine path, so two runs are byte-identical.
"""
import gzip
import hashlib
import json
import os
import sys

for _name, _value in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
                      ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_name, _value)

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if EV22 not in sys.path:
    sys.path.insert(0, EV22)

import features as FE  # noqa: E402
import m24             # noqa: E402
import port as PORT    # noqa: E402

ARMS = PORT.ARMS
ARM_INDEX = m24.ARM_INDEX
REAL_ROWS, REAL_TOP, REAL_BASELINE = 2841, 1498, 0.5273
DRY_ROWS, DRY_TOP = 1425, 727


class Fail(Exception):
    """A check or slice invariant that did not hold."""


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


def check_slice_seeds(seeds, lo, hi, where):
    """Refuse any seed outside `[lo, hi]` and call `port.check_seeds`."""
    bad = sorted({float(s) for s in seeds
                  if float(s) != int(float(s)) or not lo <= int(float(s)) <= hi})
    if bad:
        raise Fail("%s: seeds outside %d..%d: %s" % (where, lo, hi, bad[:8]))
    PORT.check_seeds(seeds, where)
    return sorted({int(s) for s in seeds})


def read_label_slice(lo, hi, want_sha_override=None):
    """Own 8-line label reader keeping `features.label_rows`'s <=80 guard intact."""
    path, want_sha = FE.label_file()
    if want_sha_override is not None:
        want_sha = want_sha_override
    with open(path, "rb") as fh:
        raw = fh.read()
    got_sha = hashlib.sha256(raw).hexdigest()
    if got_sha != want_sha:
        raise Fail("%s sha256 %s != %s" % (os.path.basename(path), got_sha, want_sha))
    all_rows = [json.loads(ln) for ln in gzip.decompress(raw).splitlines() if ln.strip()]
    PORT.check_seeds([r["seed"] for r in all_rows], "agree24.read_label_slice")
    kept = [r for r in all_rows if lo <= int(r["seed"]) <= hi]
    check_slice_seeds([r["seed"] for r in kept], lo, hi, "agree24 slice")
    labelled = [r for r in kept if r["label"] is not None]
    return all_rows, kept, labelled, got_sha


def check_baseline(top_count, total, want_top, want_share, where):
    """Assert most-frequent-arm baseline matches the pre-registered value."""
    if want_top is not None and top_count != want_top:
        raise Fail("%s: top_count %d != %d" % (where, top_count, want_top))
    share = top_count / float(total)
    if want_share is not None and round(share, 4) != want_share:
        raise Fail("%s: baseline %.6f != %.4f" % (where, share, want_share))
    return share


def check_table(table, total, where):
    """Assert 3x3 table sums to `total`."""
    got = sum(sum(r) for r in table)
    if got != total:
        raise Fail("%s: table sum %d != %d" % (where, got, total))
    return got


def eval_candidate(cname, labelled, sample_obs):
    """Evaluate `coef_<cname>.npy` on `labelled` rows."""
    cpath = os.path.join(HERE, "coef_%s.npy" % cname)
    m_sha = file_sha256(cpath)
    dec = m24.M.from_file(cpath)
    vals, arm = m24.repeat_check(dec, sample_obs, "agree24 %s" % cname)
    say("-- candidate %s (coef sha256 %s) --" % (cname, m_sha))
    say("  repeat check (p0.2 seed 1 first resched): %s -> %.3f s" % ([round(v, 6) for v in vals], arm))

    table = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    ties = 0
    for row in labelled:
        rec = FE.record_from_label_row(row)
        vec = FE.vector(rec)
        three = dec._values(vec)
        pred_idx, is_tied = m24.pick_min(three)
        if is_tied:
            ties += 1
        lbl_idx = ARM_INDEX[float(row["label"])]
        table[lbl_idx][pred_idx] += 1

    total = check_table(table, len(labelled), "agree24 %s" % cname)
    agree = sum(table[i][i] for i in range(3))
    say("  agreement: %d / %d = %.6f (no pass or fail attached)" % (agree, total, agree / float(total)))
    say("  exact ties resolved by first-minimum rule: %d / %d" % (ties, total))
    say("  label \\ answer     %8s %8s %8s | %7s %s"
        % ("3.695", "6.000", "10.043", "total", "per class"))
    for i, arm_val in enumerate(ARMS):
        row_sum = sum(table[i])
        say("  %-16.3f %8d %8d %8d | %7d %.4f"
            % (arm_val, table[i][0], table[i][1], table[i][2],
               row_sum, table[i][i] / float(row_sum)))
    ans_totals = [sum(table[i][j] for i in range(3)) for j in range(3)]
    say("  %-16s %8d %8d %8d | %7d" % ("answers", ans_totals[0], ans_totals[1], ans_totals[2], total))
    say("  %-16s %8.4f %8.4f %8.4f"
        % ("answer shares", ans_totals[0] / total, ans_totals[1] / total, ans_totals[2] / total))
    return agree, total


def main():
    """Run agreement measurement and mutations."""
    is_dry = "--dry" in sys.argv[1:]
    lo, hi = (71, 80) if is_dry else (81, 100)
    want_n = DRY_ROWS if is_dry else REAL_ROWS
    want_top = DRY_TOP if is_dry else REAL_TOP
    want_base = None if is_dry else REAL_BASELINE

    head("0  SCOPE AND CODE DIGESTS")
    say("mode            %s (seeds %d to %d, no verdict)" % ("dry" if is_dry else "real", lo, hi))
    for fname in ("agree24.py", "m24.py", "features.py", "charges.py"):
        say("  %-14s %s" % (fname, file_sha256(os.path.join(HERE, fname))))

    head("1  SLICE, LABEL COUNTS AND BASELINE")
    all_rows, kept, labelled, got_sha = read_label_slice(lo, hi)
    all_seeds = sorted({int(r["seed"]) for r in all_rows})
    kept_seeds = sorted({int(r["seed"]) for r in kept})
    say("label file      %s (sha256 %s)" % (FE.LABEL_CELL, got_sha))
    say("all rows        %d, seeds %d to %d (%d distinct)"
        % (len(all_rows), all_seeds[0], all_seeds[-1], len(all_seeds)))
    say("slice rows      %d (seeds %d to %d, %d distinct), labelled=%d, ties=%d"
        % (len(kept), kept_seeds[0], kept_seeds[-1], len(kept_seeds),
           len(labelled), len(kept) - len(labelled)))
    if len(labelled) != want_n:
        raise Fail("labelled count %d != %d" % (len(labelled), want_n))

    lbl_counts = {arm: 0 for arm in ARMS}
    for r in labelled:
        lbl_counts[float(r["label"])] += 1
    for arm in ARMS:
        say("  label %6.3f s: %5d (%.4f)" % (arm, lbl_counts[arm], lbl_counts[arm] / len(labelled)))
    top_count = max(lbl_counts.values())
    base = check_baseline(top_count, len(labelled), want_top, want_base, "agree24 baseline")
    say("always 10.043 s baseline: %d / %d = %.4f" % (top_count, len(labelled), base))

    head("2  CANDIDATES A AND B ON THE LABELLED SLICE")
    sample_obs = m24.first_logged_resched_obs("0.2", 1, max_seed=20)
    for cname in ("a", "b"):
        eval_candidate(cname, labelled, sample_obs)

    head("3  EVERY CHECK SHOWN FAILING ON A MUTATED INPUT")
    fired = []

    def show_mut(label, fn):
        try:
            fn()
        except (Fail, m24.Fail, SystemExit) as exc:
            say("  %-32s FIRES: %s" % (label, exc))
            return 1
        say("  %-32s DID NOT FIRE" % label)
        return 0

    fired.append(show_mut("port.check_seeds > 100", lambda: PORT.check_seeds([101], "mut/101")))
    fired.append(show_mut("slice guard outside range",
                          lambda: check_slice_seeds([lo - 1], lo, hi, "mut/slice")))
    fired.append(show_mut("label file sha256 mismatch",
                          lambda: read_label_slice(lo, hi, want_sha_override="0" * 64)))
    fired.append(show_mut("baseline count mismatch",
                          lambda: check_baseline(top_count + 1, len(labelled), want_top, want_base, "mut/base")))
    fired.append(show_mut("table sum mismatch",
                          lambda: check_table([[1, 0, 0], [0, 0, 0], [0, 0, 0]], len(labelled), "mut/tab")))
    say()
    say("%d of %d mutations produced a failure" % (sum(fired), len(fired)))
    say()
    say("end of agree24.py")
    return 0 if sum(fired) == len(fired) else 1


if __name__ == "__main__":
    sys.exit(main())
