#!/usr/bin/env python3
"""Acceptance 1: the decider port changes no answer the replay already gives.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B gate.py

Prints one PASS/FAIL line per check and exits non-zero if any failed.

WHAT IT CHECKS
  1. With no decider the edited ../13/replay.py is the replay as it stood before the
     port, event dict for event dict: a fixed choices arm, a policy stage, a formula rule
     on a donor run, three choices with the re-decision off, three choices with it on,
     and the logged `s11a` stage (whose own meta.json turns the re-decision on), each in
     Mode A and Mode B.  The port is worthless if it moves a stage that is already
     measured against the testbed, and the `start_lag` split this step made in `go` is
     exactly the kind of refactor that silently reorders a draw.
  2. The main cell (p 0.2, workload W0, donor s6b10p043) reproduces every printed column
     of ../18/bound_cells.csv and ../15/onoff_cells.csv, through evidence/18's own job
     and row builders, i.e. through the subclass chain every later step simulates with.
     On any difference the gate prints the columns and exits non-zero; the construction
     is ../19/predict.py:399-419's -- an adjusted control would check nothing.
     What "printing no value" means here: the SIMULATED and ARCHIVED strings of each
     differing column are printed, as the precedent prints its own got/want pairs, and
     what the precedent suppresses on a failure is its downstream prediction.  This gate
     derives no number of its own, so there is nothing to suppress; only the exit code
     and the printed difference carry the result.
  2b. The port's own engine is inert with the lag streams off: `port.Engine` with
     `pair_lag_seed=None` emits the same event dicts as evidence/18's `B.Engine`, in both
     modes.  Check 2 runs evidence/18's job builder, which simulates with `B.Engine`, so
     without this line the claim that every run of this directory goes through the chain
     check 2 reproduces the published rows with would rest on inspecting the overrides --
     and weight.py's "old fixed-arm path" row, which is `port.Engine` with the streams
     off, would be a number from an engine nothing had compared.
  3. Beyond Acceptance 1: the port's own refusals.  A decider without choices, without
     the re-decision, beside a rule or beside a policy is refused at Node construction,
     and an interval outside the choice set raises when it is returned.  A port that
     accepted any of these would run a sweep on a branch nobody chose.
  4. Nothing committed was overwritten.  `git status --porcelain` is read before and
     after the work above and must be unchanged: these scripts read csv files that are
     the published record of two earlier steps.

WHAT IT CANNOT SHOW
  Nothing here is a measurement.  Check 1 and check 2 are this replay against itself and
  against its own archived output, so they say the port is inert, never that the testbed
  polls at these instants.  No run anywhere has been made with a decider in the live
  loop.

BASE_REV names the revision check 1 reads ../13/replay.py from.  The default is the last
commit before the port; a default of HEAD would carry the port as soon as it was
committed and check 1 would compare the file with itself, which the gate also refuses
outright.
"""
import importlib.util
import multiprocessing as mp
import os
import random
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import port as PORT  # noqa: E402

R = PORT.R
B = PORT.B
ONOFF = B.O                             # evidence/15, for its csv cell formatting

BASE_REV = os.environ.get("BASE_REV", "fd6ce55")
REL = "docs/wayfinder/semindex-stage2/evidence/13/replay.py"

CHOICES = [3.695, 6.0, 10.043]          # the s11a arms
JF = 0.1                                # the s11a choice jitter
RULE = {"i_min": 3.695, "m": 2.0, "i_max": 10.043, "jf": 0.1}

# (label, stage, p, rule=, run-dict overrides, whether to drop the `redecide` key).
# Dropping the key is what reaches the engine's own `run.get("redecide", False)` default,
# which is the only thing keeping the frozen scripts of evidence/18 and evidence/20 -- all
# of which build their run dicts by hand, from before the key existed -- off the
# re-decision path.  A wrong default would otherwise stay invisible here.
OFF_CELLS = [
    ("fixed choices arm", "s6b3p695", 0.4, None, {}, True),
    ("policy stage", "s6short", 0.2, None, {}, True),
    ("rule on a donor", "s6b10p043", 0.4, RULE, {}, True),
    ("three choices, redecide off", "s6b10p043", 0.4, None,
     {"policy": None, "choices": list(CHOICES), "choice_jitter": JF, "redecide": False}, False),
    ("three choices, redecide on", "s6b10p043", 0.4, None,
     {"policy": None, "choices": list(CHOICES), "choice_jitter": JF, "redecide": True}, False),
    ("logged redecide stage", "s11a", 0.2, None, {}, False),
]
OFF_SEEDS = PORT.check_seeds((1, 2), "gate.py check 1")

MAIN_CELL = ("s6b10p043", 0.2)
MAIN_WL = "W0"
CELL_SEEDS = PORT.check_seeds(B.SEEDS, "gate.py check 2")
WORKERS = 5                             # the task's ceiling on a six-core machine


def flat(events):
    """The event stream as comparable tuples, node by node in name order.

    `repr` rather than the value itself so that a float, a list and a dict all compare by
    their written form: two events are equal here exactly when the two replays would
    write the same line into events.jsonl.
    """
    out = []
    for node in sorted(events):
        for e in events[node]:
            out.append((node, tuple(sorted((k, repr(v)) for k, v in e.items()))))
    return out


def first_difference(left, right):
    """The first index where two flattened streams differ, with both rows."""
    for i, (u, v) in enumerate(zip(left, right)):
        if u != v:
            return f"    at {i}\n      base {u}\n      edit {v}"
    return f"    lengths {len(left)} and {len(right)}"


def load_base():
    """The replay as of BASE_REV, imported as a second module object.

    Written into a temporary directory and loaded by path: importing it under its own
    name would hand back the edited module that is already in sys.modules, and the check
    would then compare a module with itself.
    """
    root = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, check=True).stdout.strip()
    src = subprocess.run(["git", "-C", root, "show", f"{BASE_REV}:{REL}"],
                         capture_output=True, text=True, check=True).stdout
    with open(os.path.join(PORT.EV13, "replay.py"), encoding="utf-8") as fh:
        if fh.read() == src:
            return None
    tmp = tempfile.TemporaryDirectory()
    path = os.path.join(tmp.name, "replay_base.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("replay_base", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["replay_base"] = module
    spec.loader.exec_module(module)
    module._gate_tmp = tmp          # keep the directory alive as long as the module is
    return module


def say(ok, text):
    """One check's line; returns 1 when it failed, so main can total the failures."""
    print(f"{'PASS' if ok else 'FAIL'} {text}", flush=True)
    return 0 if ok else 1


def check_off(base):
    """Check 1: with no decider the edited replay equals BASE_REV's, on every mode."""
    bad = 0
    for label, stage, p, rule, over, drop in OFF_CELLS:
        for mode in ("A", "B"):
            diffs, n_events = [], 0
            for seed in OFF_SEEDS:
                run, brun = R.load_run(stage, p, seed), base.load_run(stage, p, seed)
                for d in (run, brun):
                    d.update(over)
                    if drop:
                        d.pop("redecide")
                if mode == "A":
                    got = R._sim_run(run, "A", lags=R.logged_lags(run), rule=rule)
                    want = base._sim_run(brun, "A", lags=base.logged_lags(brun), rule=rule)
                else:
                    seeded = R.rep_seed(stage, p, seed, 0)
                    got = R._sim_run(run, "B", sim_rng=random.Random(seeded), rule=rule)
                    want = base._sim_run(brun, "B", sim_rng=random.Random(seeded), rule=rule)
                left, right = flat(want), flat(got)
                n_events += len(right)
                if left != right:
                    diffs.append((seed, first_difference(left, right)))
            bad += say(not diffs, f"no decider, {label:<28} {stage:<10} p {p} seeds "
                                  f"{OFF_SEEDS[0]} to {OFF_SEEDS[-1]} mode {mode}: "
                                  f"{n_events} events, every dict equal to {BASE_REV}'s")
            for seed, text in diffs:
                print(f"  seed {seed}\n{text}")
    return bad


def main_cell_rows(pool):
    """The main cell's W0 rows, built by evidence/18's own job and row builders.

    `job_base` and `cell_row` are called rather than copied, and `sim` inside them is
    evidence/18's Engine on evidence/15's Engine on the edited `_Engine`, which is the
    chain every later step simulates with.  Restricted to the one cell and the one
    workload the main cell is: evidence/18's own check 1 walks all three cells and three
    workloads and takes twenty times as long for an answer about the same rows.
    """
    donor, p = MAIN_CELL
    res = pool.map(B.job_base, [(donor, p, s, (MAIN_WL,)) for s in CELL_SEEDS], chunksize=1)
    per_seed, records = {}, {}
    for d, pp, seed, out in res:
        for key, v in out.items():
            if key[0] == MAIN_WL:
                per_seed[(key[1], seed)] = v
            elif key[0] == "records":
                records[seed] = v
    rec = [records[s] for s in CELL_SEEDS]
    return [B.cell_row(MAIN_WL, donor, p, name, kind,
                       [per_seed[(name, s)] for s in CELL_SEEDS], rec)
            for name, kind, _spec in B.RULE_SPECS]


def check_cells(rows):
    """Check 2: those rows against the two published cells files, column by column."""
    bad, checked, missing = [], 0, []
    for path in (os.path.join(PORT.EV18, "bound_cells.csv"),
                 os.path.join(os.path.normpath(os.path.join(HERE, "..", "15")),
                              "onoff_cells.csv")):
        ref, cols = B.read_cells(path)
        for r in rows:
            key = (r["workload"], r["donor"], r["p"], r["rule"])
            want = ref.get(key)
            if want is None:
                missing.append((os.path.basename(path), key))
                continue
            checked += 1
            # 2026-11-23: the absent column is a FAILURE, not a skip.  The precedent this
            # comparison is copied from drops a csv column the built row does not carry,
            # which happens to be nothing today (both headers are exactly the 24 keys
            # `cell_row` returns) but would silently turn a column that stopped being
            # emitted into one fewer column compared, under a pass line that claims every
            # printed column.
            gone = [c for c in cols if c not in r]
            if gone:
                bad.append((os.path.basename(path), key,
                            {"columns the built row does not carry": gone}))
                continue
            diffs = {c: (ONOFF.csv_str(r[c]), want[c]) for c in cols
                     if ONOFF.csv_str(r[c]) != want[c]}
            if diffs:
                bad.append((os.path.basename(path), key, diffs))
    out = say(not bad and not missing,
              f"main cell {MAIN_CELL[0]} p {MAIN_CELL[1]} {MAIN_WL}, seeds "
              f"{CELL_SEEDS[0]} to {CELL_SEEDS[-1]}, {len(B.RULE_SPECS)} rules: "
              f"{checked} (rule x file) rows identical on every printed column of "
              "../18/bound_cells.csv and ../15/onoff_cells.csv")
    for f, key, diffs in bad:
        print(f"  {f} {key}: {diffs}")
    for f, key in missing:
        print(f"  {f}: no row for {key}")
    return out


def check_passthrough():
    """Check 2b: `port.Engine` with the lag streams off IS evidence/18's engine.

    2026-11-23: the subclass overrides three methods, and two of them return
    `super()`'s value unchanged when `pair_lag_seed` is None while the third only
    increments a counter before delegating.  That is a claim about the source; this is
    the claim measured.  Both engines are handed the same run dict and a `random.Random`
    on the same seed string, so a draw the subclass added or dropped shows up as a
    differing event dict, not as a differing random stream.
    """
    bad, fired = 0, 0
    for stage, p in ((MAIN_CELL[0], MAIN_CELL[1]), ("s11a", 0.2)):
        for mode in ("A", "B"):
            diffs, n_events = [], 0
            for seed in OFF_SEEDS:
                run = R.load_run(stage, p, seed)
                seeded = R.rep_seed(f"passthrough:{stage}", p, seed, 0)
                lags = R.logged_lags(run) if mode == "A" else None
                want = flat(B.Engine(
                    run, mode, lags,
                    None if mode == "A" else random.Random(seeded)).go())
                engine = PORT.Engine(
                    run, mode, lags,
                    None if mode == "A" else random.Random(seeded), None, None)
                got = flat(engine.go())
                # Counted so that the provisional-timer count covers this script's own
                # port runs; weight.py's number speaks only for weight.py's simulations.
                fired += engine.provisional_fired
                n_events += len(got)
                if want != got:
                    diffs.append((seed, first_difference(want, got)))
            bad += say(not diffs,
                       f"port.Engine with pair_lag_seed None equals ../18's Engine: "
                       f"{stage:<10} p {p} seeds {OFF_SEEDS[0]} to {OFF_SEEDS[-1]} mode "
                       f"{mode}: {n_events} events, every dict equal")
            for seed, text in diffs:
                print(f"  seed {seed}\n{text}")
    print(f"  provisional timers that fired before their own poll had resolved, over the "
          f"port runs of this check: {fired}")
    return bad


def check_refusals():
    """Check 3: what the port refuses.  Beyond Acceptance 1, but part of its contract."""
    run = R.load_run("s6b10p043", 0.4, OFF_SEEDS[0])
    base = dict(run, policy=None, rule=None, choices=list(CHOICES), choice_jitter=JF,
                redecide=True, decider=PORT.const_decider(CHOICES[0]))
    cases = [
        ("no choices to pick from", dict(base, choices=None)),
        ("the re-decision off", dict(base, redecide=False)),
        ("a policy beside it", dict(base, policy={"intervals_s": [2.0],
                                                  "jitter_frac": [0.1]})),
        ("a rule beside it", dict(base, rule=RULE)),
    ]
    bad = 0
    for label, bad_run in cases:
        try:
            R._sim_run(bad_run, "A", lags=R.logged_lags(bad_run))
        except ValueError:
            bad += say(True, f"refused at construction: a decider with {label}")
        else:
            bad += say(False, f"NOT refused: a decider with {label}")
    try:
        R._sim_run(dict(base, decider=PORT.const_decider(4.2)), "A",
                   lags=R.logged_lags(base))
    except ValueError:
        bad += say(True, "refused when returned: an interval outside the choice set")
    else:
        bad += say(False, "NOT refused: an interval outside the choice set")
    return bad


def git_status():
    """`git status --porcelain` of the whole worktree, as one string."""
    root = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return subprocess.run(["git", "-C", root, "status", "--porcelain"],
                          capture_output=True, text=True, check=True).stdout


def main():
    before = git_status()
    base = load_base()
    if base is None:
        print(f"FAIL base source: {BASE_REV}:{REL} already carries the decider port, so "
              "check 1 would compare the file with itself. Re-run with BASE_REV set to "
              "the revision before the port.")
        return 1
    print(f"Base for check 1: {BASE_REV}:{REL}\n")
    t = time.time()
    bad = check_off(base)
    bad += check_passthrough()
    bad += check_refusals()
    with mp.Pool(WORKERS) as pool:
        rows = main_cell_rows(pool)
    bad += check_cells(rows)
    after = git_status()
    bad += say(before == after,
               "nothing tracked was written: `git status --porcelain` unchanged across "
               "this run")
    if before != after:
        print(f"  before\n{before}  after\n{after}")
    print(f"\nwall {time.time() - t:.1f} s")
    print(f"{'every check passed' if not bad else str(bad) + ' CHECK(S) FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
