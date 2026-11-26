#!/usr/bin/env python3
"""Script-built inventory of every file reading selection seeds 81 to 100 before M.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B inventory.py

Scans `../22`, `../23` and `.` (this directory):
1. Parses the `seed` column of every `*.csv` and prints distinct seeds in 81 to 100, row
   counts and distinct deciders;
2. Scans every `*.txt` and `*.md` for decimal tokens in 81 to 100 bounded by non-digits
   and for the phrases `selection`, `81 to 100`, `81-100` (prints evidence only; makes no
   assertion on prose matches);
3. Reconciles the previous evaluation's nine deciders from the parsed archives (`../23/eval.csv`,
   `../23/spread.txt`, `../23/parity.txt`, `../23/agree.txt`) rather than typed literals.
"""
import collections
import glob
import hashlib
import os
import re
import sys

for _name, _value in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
                      ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_name, _value)

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
EV23 = os.path.normpath(os.path.join(HERE, "..", "23"))
sys.path.insert(0, HERE)
sys.path.insert(0, EV22)

import port as PORT  # noqa: E402

SEL_LO, SEL_HI = 81, 100
TOKEN_RE = re.compile(r"(?<![0-9])(8[1-9]|9[0-9]|100)(?![0-9])")
PHRASE_RE = re.compile(r"(?:selection|81 to 100|81-100)")


class Fail(Exception):
    """A check that did not hold."""


def say(text=""):
    """Print one line of stdout."""
    print(text, flush=True)


def head(title):
    """Print a section header."""
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


def file_sha256(path):
    """Hex sha256 of a file."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def parse_csv_sel(path):
    """Return `(distinct_sel_seeds, sel_row_count, distinct_deciders, all_rows)` for a csv."""
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    if not lines:
        raise Fail("%s: empty csv" % path)
    cols = lines[0].split(",")
    if "seed" not in cols:
        return [], 0, [], []
    rows = [dict(zip(cols, ln.split(","))) for ln in lines[1:]]
    seeds_all = [int(r["seed"]) for r in rows]
    PORT.check_seeds(seeds_all, path)
    sel_rows = [r for r in rows if SEL_LO <= int(r["seed"]) <= SEL_HI]
    sel_seeds = sorted({int(r["seed"]) for r in sel_rows})
    decs = sorted({r["decider"] for r in sel_rows if "decider" in r})
    return sel_seeds, len(sel_rows), decs, rows


def scan_text_file(path):
    """Return `[(lineno, token_hits, phrase_hits, text)]` for matching lines."""
    hits = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            ln = raw.rstrip("\n")
            toks = TOKEN_RE.findall(ln)
            phs = PHRASE_RE.findall(ln)
            if toks or phs:
                hits.append((lineno, toks, phs, ln))
    return hits


def parse_parity_sel(path):
    """Parse `../23/parity.txt` and count single-stream runs for seeds 81 to 100."""
    sel_lines = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            m = re.match(r"^\s*seed\s+(\d+)\s+(\S+)\s+[0-9a-f]{64}\b", ln)
            if m and SEL_LO <= int(m.group(1)) <= SEL_HI:
                sel_lines.append((int(m.group(1)), m.group(2)))
    return sel_lines


def parse_spread_adapters(path):
    """Parse `../23/spread.txt` part 1 model digests."""
    adapters = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            m = re.match(r"^\s*(a|b)\s+(first run|second run)\s+([0-9a-f]{64})\s*$", ln)
            if m:
                adapters.append((m.group(1), m.group(2), m.group(3)))
    return adapters


def parse_agree_baseline(path):
    """Parse labelled row count and baseline from `../23/agree.txt`."""
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            m = re.search(r"agrees with\s+(\d+)/(\d+)\s*=\s*([0-9.]+)", ln)
            if m:
                return int(m.group(1)), int(m.group(2)), float(m.group(3))
    raise Fail("%s: could not parse baseline line" % path)


def show_mutation(label, fn):
    """Run `fn()` and verify it raises `Fail` or `SystemExit`."""
    try:
        fn()
    except (Fail, SystemExit) as exc:
        say("  %-30s FIRES: %s" % (label, exc))
        return 1
    say("  %-30s DID NOT FIRE" % label)
    return 0


def main():
    """Build and print the complete selection-seed inventory."""
    head("0  SCOPE AND CODE DIGEST")
    say("directories     ../22, ../23, .")
    say("selection range %d to %d" % (SEL_LO, SEL_HI))
    say("inventory.py    sha256 %s" % file_sha256(os.path.abspath(__file__)))

    dirs = (("../22", EV22), ("../23", EV23), (".", HERE))

    head("1  CSV FILES: PARSED SEED COLUMNS (81 TO 100)")
    csv_records = {}
    for label, dpath in dirs:
        for path in sorted(glob.glob(os.path.join(dpath, "*.csv"))):
            rel = os.path.join(label, os.path.basename(path))
            sel_seeds, count, decs, rows = parse_csv_sel(path)
            csv_records[rel] = (sel_seeds, count, decs, rows)
            say("  %-28s rows_81_100=%5d  seeds=%s  deciders=%s"
                % (rel, count,
                   "%d..%d (%d)" % (sel_seeds[0], sel_seeds[-1], len(sel_seeds)) if sel_seeds else "none",
                   decs if decs else "none"))

    head("2  TXT AND MD FILES: TOKEN AND PHRASE MATCHES (EVIDENCE ONLY, NO ASSERTION)")
    self_out = os.path.abspath(os.path.join(HERE, "inventory.txt"))
    for label, dpath in dirs:
        for ext in ("*.txt", "*.md"):
            for path in sorted(glob.glob(os.path.join(dpath, ext))):
                if os.path.abspath(path) == self_out:
                    continue
                rel = os.path.join(label, os.path.basename(path))
                hits = scan_text_file(path)
                tok_n = sum(len(h[1]) for h in hits)
                phs_n = sum(len(h[2]) for h in hits)
                say("  %-28s matching_lines=%4d  token_hits=%4d  phrase_hits=%4d"
                    % (rel, len(hits), tok_n, phs_n))
                for lineno, toks, phs, ln in hits:
                    say("    %s:%d [tok=%s ph=%s] %s" % (rel, lineno, toks, phs, ln[:120]))

    head("3  RECONCILIATION OF PREVIOUS EVALUATION COUNTS (FROM PARSED ARCHIVES)")
    eval_rows = csv_records["../23/eval.csv"][3]
    by_dec = collections.Counter(r["decider"] for r in eval_rows)
    no_model = ("const3.695", "const6.000", "const10.043", "F", "untuned")
    adapter_decs = ("a", "b")
    no_model_n = sum(by_dec[d] for d in no_model)
    adapter_n = sum(by_dec[d] for d in adapter_decs)
    say("../23/eval.csv: %d total rows (%d deciders)" % (len(eval_rows), len(by_dec)))
    say("  no-model deciders %s: %d rows" % (list(no_model), no_model_n))
    say("  adapter deciders  %s: %d rows" % (list(adapter_decs), adapter_n))

    eval_run1_rows = csv_records["../23/eval_run1.csv"][3]
    say("../23/eval_run1.csv: %d total rows (%d deciders)"
        % (len(eval_run1_rows), len({r["decider"] for r in eval_run1_rows})))

    adapters = parse_spread_adapters(os.path.join(EV23, "spread.txt"))
    run1_adapters = [a for a in adapters if a[1] == "first run"]
    run2_adapters = [a for a in adapters if a[1] == "second run"]
    eval_shas = {r["decider"]: r["model_sha"] for r in eval_rows if r["decider"] in ("a", "b")}
    for cand, _kind, sha in run2_adapters:
        if eval_shas.get(cand) != sha:
            raise Fail("spread.txt second run %s sha %s != eval.csv %s" % (cand, sha, eval_shas.get(cand)))
    say("../23/spread.txt: %d adapter rows (%d first run, %d second run; run2 matches eval.csv)"
        % (len(adapters), len(run1_adapters), len(run2_adapters)))
    say("  distinct deciders on seeds 81-100: %d (eval.csv) + %d (spread.txt run1) = %d"
        % (len(by_dec), len(run1_adapters), len(by_dec) + len(run1_adapters)))

    parity_sel = parse_parity_sel(os.path.join(EV23, "parity.txt"))
    say("../23/parity.txt: %d single-stream run(s) on seeds 81-100: %s"
        % (len(parity_sel), sorted({s for s, _d in parity_sel})))

    top_n, lab_n, base = parse_agree_baseline(os.path.join(EV23, "agree.txt"))
    say("../23/agree.txt: %d/%d = %.4f pre-registered baseline" % (top_n, lab_n, base))
    say("from ../23/spec.md A2 (unarchived scratch probe): one timing call to label.run_dict(0.2, 81)")

    head("4  EVERY CHECK SHOWN FAILING ON A MUTATED INPUT")
    fired = []
    fired.append(show_mutation(
        "port.check_seeds > 100",
        lambda: PORT.check_seeds([101], "mutated")))
    tmp_empty = os.path.join(HERE, ".tmp_inv_empty.csv")
    try:
        open(tmp_empty, "w", encoding="utf-8").write("")
        fired.append(show_mutation("empty csv", lambda: parse_csv_sel(tmp_empty)))
    finally:
        if os.path.exists(tmp_empty):
            os.remove(tmp_empty)
    fired.append(show_mutation(
        "agree baseline parse missing",
        lambda: parse_agree_baseline(os.path.join(HERE, "charges.txt"))))
    say()
    say("%d of %d mutations produced a failure" % (sum(fired), len(fired)))
    say()
    say("end of inventory.py")
    return 0 if sum(fired) == len(fired) else 1


if __name__ == "__main__":
    sys.exit(main())
