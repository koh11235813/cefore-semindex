#!/usr/bin/env python3
"""Post hoc sensitivity: does candidate a read the state, or the position in the run?

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B drift24.py > drift24.txt

THIS IS NOT IN THE FROZEN PLAN AND IT SELECTS NOTHING.  A cross-verification of this
step found that candidate a's fit is dominated by a within-run drift: a double-digit
count of live cells whose correlation with the decision's own instant (`t_rel_ms`)
exceeds 0.5 in absolute value, over candidate a's fit rows, carry most of the fitted
rate's gap from the logged rate.  This script asks one question about it: if those cells
are neutralised (held
at their fit-row mean, so M cannot read them), does the replay score move, and by how
much, on training seeds the shipped fit already saw.  This is a sensitivity check of
the shipped fit's object, not a second measurement of M on new data and not a
re-selection between candidates a and b, which stays fixed at a (ana24.txt's VERDICT).

HOW THE NEUTRALISED MODEL IS BUILT.  The fit rows of candidate a (seeds 1 to 70, p 0.2,
`rows24.build_all()`) are copied nowhere: the drifting cells (named below, by this
script's own count and not by a literal) are replaced by their own column mean in every
row's `vec` AND in every entry of that row's own `chain` list, in
place.  This is necessary and not cosmetic: `precheck.table_of_run` makes a row and its
own position in its block's `chain` the SAME dict object (`got["chain"] = seq` and
`out.extend(seq)`), so a row's current state and the successor state its own next row
reads are two entries of one shared structure, and `m24.build_system`'s instrument-minus-
successor regressor needs both neutralised the same way or the construction compares a
neutralised current state against an un-neutralised successor, which an earlier scratch
version of this check did by mistake (copying the row dict but not its shared `chain`)
and which is why this script neutralises by walking `chain` explicitly rather than by
copying rows.  The neutralised coefficients are then the ordinary k=0 LSTD-Q solve
(`m24.build_system` / `solve_system` at `m24.LAMBDA_REF`, the shipped candidate a's own
penalty) on that neutralised row set, exactly the computation `learn24.py` performed on
the un-neutralised rows to ship `coef_a.npy`.

WHAT THE REPLAY COMPARES.  Four deciders are run through `eval24.one_run`, the committed
replay runner, unmodified, on seeds 1 to 20 (a subset of the 70 fit seeds and disjoint
from the 81-100 selection seeds) at all 40 replicate streams: the shipped `coef_a.npy`,
the neutralised `coef_a_neutral.npy` this script writes, the formula rule F
(`eval24.py`'s own `LBL.F()`), and the fixed 6.0 s arm (`eval24.py`'s own
`PORT.const_decider`), both of the latter two read off `eval24.py`'s module namespace and
not re-implemented.  The per-seed mean score and the paired bootstrap interval are
`versus24.py`'s and `spread.py`'s own convention: `per_seed_mean`, then
`predict20.boot_index(n_seeds)` resampled over seeds with the stream mean as one seed's
value.

The two M deciders are additionally run as a thin `m24.M` subclass that logs every
resolution decision's own timestamp and answer (see `RecordingM`), so that this script
can show, from the SAME replay that scores the four deciders and without simulating the
ports a second time, whether a model's chosen interval moves with the position in the
run.

WHAT THIS CANNOT SAY.  Seeds 1 to 20 are inside the 70 seeds candidate a's own fit
already read, so a difference here is a difference WITHIN the data the fit was built on,
not a generalisation claim; the held seeds (71 to 80) and the selection seeds (81 to 100)
are untouched by this script and say nothing about them.  It is a sensitivity of the
shipped fit's own object to a minority of its 261 input cells, nothing more.  And
removing a cell that is a within-run function of the clock removes whatever state
information that cell ALSO carried; a worse score after removing it does not by itself
show that the shipped M was reading the clock rather than the state, and an equal score
would not have shown the opposite either -- see section 9.
"""
import collections
import hashlib
import json
import os
import statistics
import sys
import tempfile

# The thread caps before numpy is imported, as every other script of this directory
# pins them: a BLAS library opens one thread per core at load time, and nothing here is
# a float reduction a thread count could reorder.
for _name, _value in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
                      ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_name, _value)

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EV20 = os.path.normpath(os.path.join(HERE, "..", "20"))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if EV22 not in sys.path:
    sys.path.insert(0, EV22)

import importlib.util    # noqa: E402

import eval24 as EV24    # noqa: E402  the committed replay runner, called and not edited
import features as FE    # noqa: E402  FE.NAMES: the 261 cell names, by position
import m24               # noqa: E402
import rows24 as R24     # noqa: E402

PORT = EV24.PORT

# `../20/predict.py` by path and under its own module name, exactly as `versus24.py` and
# `../23/spread.py` load it: two files in this tree are named `predict.py`, and importing
# either by its bare name would make which one loads depend on `sys.path` order.
_spec = importlib.util.spec_from_file_location("predict20", os.path.join(EV20, "predict.py"))
PR20 = importlib.util.module_from_spec(_spec)
sys.modules["predict20"] = PR20
_spec.loader.exec_module(PR20)

# Worker's choice: the cross-verification's own threshold for naming a cell "drifting",
# restated here and not reopened.  This script recomputes the cell list from the rows
# rather than naming it, so the count printed below is read and not assumed.
DRIFT_THRESH = 0.5
LAMBDA = m24.LAMBDA_REF                  # candidate a's own shipped penalty, 1e-4
GATE_TOL = 1e-6                          # the shipped-fit reproducibility gate's tolerance
TRAIN_SEEDS = tuple(range(1, 21))        # the replay seeds this step is allowed to touch
NREP = EV24.NREP                         # 40 replicate streams, same as eval24.py's grid
COEF_PATH = os.path.join(HERE, "coef_a_neutral.npy")
JSON_PATH = os.path.join(HERE, "coef_a_neutral.json")
CSV_PATH = os.path.join(HERE, "drift24.csv")

# The 30 extra per-run columns of drift24.csv: a count of resolutions per (decile of
# decision time, chosen interval), so that a verifier can recompute section 7's shares
# from the archived table without re-simulating anything.  `mind_changes` and
# `res_decisions` are this run's own slice of `count_mind_changes`'s two totals (see
# `run_replay`), so that section 5's mind-change share is also recomputable from the csv
# alone and not only from this script's own accumulator.
ARM_LABELS = tuple("%.3f" % arm for arm in PORT.ARMS)
DECILE_ARM_COLS = tuple("d%d_%s" % (d, lbl) for d in range(10) for lbl in ARM_LABELS)
CSV_COLUMNS = ("decider", "seed", "stream", "score_ms", "polls", "delay_ms", "c4",
              "n_short", "n_mid", "n_long", "mind_changes", "res_decisions") + DECILE_ARM_COLS


class Fail(Exception):
    """A check, gate, or seed-range guard that did not hold."""


def say(text=""):
    """Print one line of output."""
    print(text, flush=True)


def head(title):
    """Print a section heading in this directory's own shape."""
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


def file_sha256(path):
    """Return the hex sha256 digest of a file."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def check_fit_seed_range(seeds, where, want=None):
    """Refuse any seed outside candidate a's own fit range, 1 to 70 inclusive.

    This is a narrower check than anything `rows24.py` or `m24.py` already runs; it
    exists so that a mistake in THIS script's own row selection is caught here rather
    than relying on `rows24.build_all` alone to have been called correctly.  With `want`
    (an iterable of the exact seeds expected), also refuses a seed set that stays inside
    1 to 70 but drops or adds one of them, which the range check alone cannot see.
    """
    lo, hi = min(R24.FIT_SEEDS), max(R24.FIT_SEEDS)
    bad = sorted({int(s) for s in seeds if not lo <= int(s) <= hi})
    if bad:
        raise Fail("%s: fit seeds must be %d to %d, refused %s" % (where, lo, hi, bad[:8]))
    got = sorted({int(s) for s in seeds})
    if want is not None and set(got) != set(want):
        raise Fail("%s: fit seed set %s != expected %s" % (where, got[:4], sorted(want)[:4]))
    return got


def check_replay_in_fit(seeds, where):
    """Assert every replay seed is also one of candidate a's own 1-70 fit seeds.

    Seeds 1-20 are a subset of 1-70 by construction, so this can only ever fire on a
    seed set this script itself never produces; it exists so the claim printed in
    section 5 is asserted rather than merely implied by `check_replay_seed_range`'s
    narrower 1-20 bound, and so a mutated seed outside 1-70 (e.g. 71) is shown refused
    here specifically, not only by the fit-side range check.  Returns the count of
    `seeds` found in `R24.FIT_SEEDS`.
    """
    seeds = {int(s) for s in seeds}
    in_fit = seeds & set(R24.FIT_SEEDS)
    if len(in_fit) != len(seeds):
        raise Fail("%s: %d of %d replay seeds are not in candidate a's own 1-70 fit seeds"
                   % (where, len(seeds) - len(in_fit), len(seeds)))
    return len(in_fit)


def check_replay_seed_range(seeds, where):
    """Refuse any seed outside the 1 to 20 replay range this step is allowed to touch.

    Refuses above 100 and the 121-160 measuring range explicitly and first, the order
    `../23/spread.py`'s own seed-role check uses and for the same reason: 121-160 is
    already above 100, so testing the narrower 1-20 bound alone would never show the
    measuring range being refused on an input the narrower bound does not already catch.
    """
    bad_measuring = sorted(set(int(s) for s in seeds) & set(range(121, 161)))
    if bad_measuring:
        raise Fail("%s: seed set meets the measuring seeds 121-160: %s" % (where, bad_measuring))
    PORT.check_seeds(seeds, where)
    bad = sorted({int(s) for s in seeds if not 1 <= int(s) <= 20})
    if bad:
        raise Fail("%s: replay seeds for this step must be 1 to 20, refused %s" % (where, bad[:8]))
    return sorted({int(s) for s in seeds})


def within_corr(y, t, runs):
    """Pooled within-run (per-seed centred) Pearson correlation of `y` against `t`.

    Centres `t` and `y` separately within each seed's own rows, then pools the centred
    values across seeds and correlates the pool.  This is a correlation of DEVIATIONS
    from each run's own mean, so it cannot be driven by a difference in level between
    runs; it can only be driven by a trend WITHIN a run, which is the drift this script
    is built to find.
    """
    uniq = np.unique(runs)
    tt = np.concatenate([t[runs == g] - t[runs == g].mean() for g in uniq])
    yy = np.concatenate([y[runs == g] - y[runs == g].mean() for g in uniq])
    if yy.std() == 0.0:
        return 0.0
    return float(np.corrcoef(tt, yy)[0, 1])


def within_slope(y, t, runs):
    """Pooled within-run OLS slope of `y` on `t` (deviations from each run's own mean).

    Same per-seed centring as `within_corr`.  The slope is `sum(tt*yy) / sum(tt*tt)`
    over the pooled, centred values: `y`'s units per `t`'s units, so for `y` a Q-value
    in score ms and `t` the decision time in seconds, this is a rate in score ms per
    second of position within a run.
    """
    uniq = np.unique(runs)
    tt = np.concatenate([t[runs == g] - t[runs == g].mean() for g in uniq])
    yy = np.concatenate([y[runs == g] - y[runs == g].mean() for g in uniq])
    denom = float(np.sum(tt * tt))
    if denom == 0.0:
        return 0.0
    return float(np.sum(tt * yy) / denom)


def decile_cuts(values):
    """Return the 9 cutoffs splitting `values` into ten pooled deciles (10th..90th pct)."""
    return np.quantile(np.asarray(values, dtype=np.float64), [0.1 * k for k in range(1, 10)])


def decile_bin(values, cuts):
    """Return each value's decile bin index (0..9) against the fixed `cuts`."""
    return np.searchsorted(cuts, np.asarray(values, dtype=np.float64), side="right")


def neutralise_in_place(rows, drift_idx, colmean):
    """Replace `vec[drift_idx]` by `colmean` in every reachable record, in place.

    Walks each row AND every entry of that row's own `chain` (see the module docstring
    for why the two are not the same thing to skip), deduplicated by the id of the
    RECORD dict itself, not of its `vec` array.  2026-11-26 fix: an earlier version
    deduplicated on `id(rec["vec"])`, but this loop rebinds `rec["vec"]` to a freshly
    allocated array on a record's first visit, which drops the last reference to the
    original array; CPython can then reuse that freed address for the replacement
    array, so a record's second visit sometimes tests an id the allocator happened to
    recycle rather than a fact about the data, making the returned count depend on
    malloc behaviour instead of being reproducible across interpreters.  Deduplicating
    on the record's own id is stable because a record dict is never replaced, only
    mutated.  Returns the number of distinct records touched.
    """
    seen = set()
    touched = 0
    for r in rows:
        for rec in (r, *r.get("chain", ())):
            if id(rec) in seen:
                continue
            seen.add(id(rec))
            v = rec.get("vec")
            if v is None:
                continue
            arr = np.array(v, dtype=np.float64)   # copy: never alias colmean's own buffer
            arr[drift_idx] = colmean
            rec["vec"] = arr
            touched += 1
    return touched


def check_shipped_gate(file_w, resolve_w, tol, where):
    """Assert the shipped file's flat weight vector equals the local re-solve within `tol`."""
    diff = float(np.max(np.abs(np.asarray(file_w).reshape(-1) - np.asarray(resolve_w))))
    if diff > tol:
        raise Fail("%s: max |shipped file - local re-solve| %.3e > tol %.3e"
                   % (where, diff, tol))
    return diff


def check_greedy_agreement(gi_a, gi_b, where):
    """Assert two greedy-index arrays agree on every row (the shipped-fit reproducibility gate)."""
    gi_a = np.asarray(gi_a)
    gi_b = np.asarray(gi_b)
    n_diff = int((gi_a != gi_b).sum())
    if n_diff:
        raise Fail("%s: %d of %d rows disagree between the two greedy-index arrays"
                   % (where, n_diff, len(gi_a)))
    return 0


def per_seed_mean(rows, seeds):
    """{seed: the mean score_ms of that seed's runs}, `versus24.py`'s own arithmetic."""
    return {s: statistics.fmean(r["score_ms"] for r in rows if r["seed"] == s) for s in seeds}


def paired(first, second, per_seed, seeds, idx):
    """(point difference, lower, upper, count where first < second) over `seeds`.

    `first` minus `second`: a negative point means the first decider scores lower
    (better, since lower score is better throughout this directory).  The interval is
    the 95 % percentile interval of `idx`, a paired bootstrap over seeds, the matrix
    `predict20.boot_index` draws and `versus24.py` / `spread.py` already use.
    """
    diff = np.array([per_seed[first][s] - per_seed[second][s] for s in seeds], dtype=float)
    boot = np.array([diff[row].mean() for row in idx], dtype=float)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return float(diff.mean()), lo, hi, int((diff < 0.0).sum())


class RecordingM(m24.M):
    """`m24.M` that also logs every resolution decision's own timestamp and answer.

    `__call__` defers to `m24.M.__call__` first and unconditionally, then appends to its
    own `decisions` list; it never changes what is returned or what `m24.M`'s own state
    (`mem`, `reads`, `arms`, ...) holds, so a run played through `RecordingM` scores
    exactly as the same run played through plain `m24.M` would, and `eval24.one_run`'s
    own checks (`FirstCallEmpty`, the engine counter cross-check) are unaffected.  Used
    so that section 7's decile table is read off the SAME replay that scores the four
    deciders, instead of a second simulation pass.
    """

    def __init__(self, w_blocks, mu, sigma):
        """Initialise as `m24.M` and add an empty per-instance decision log."""
        super().__init__(w_blocks, mu, sigma)
        self.decisions = []   # [(ts_ms, arm), ...] appended at every resolution

    def __call__(self, obs):
        """Delegate to `m24.M.__call__`, then log the resolution's own time and answer."""
        arm = super().__call__(obs)
        w = obs["where"]
        if w["kind"] == "state_poll_reschedule":
            self.decisions.append((float(w["ts_ms"]), arm))
        return arm


def run_replay(name, seeds, maker):
    """Run `eval24.one_run` for `name` over `seeds` x `NREP` streams.

    Calls the committed `eval24.one_run` unmodified, with this script's own `extra_stats`
    accumulator in `eval24.main`'s own shape, so the replay mechanics (the `FirstCallEmpty`
    fresh-memory assertion, the engine counter cross-check, the per-pair lag-draw count)
    are exactly the ones `eval24.py` itself runs on the selection seeds.

    Returns `(rows, stats, decisions)`.  Each row also carries this one run's own
    `mind_changes` and `res_decisions` (the delta of `extra_stats[name]`'s two running
    totals across this single call to `one_run`), so the per-decider mind-change share
    this script prints is recomputable from `drift24.csv` alone, not only from this
    accumulator.  `decisions` is `[]` unless the decider instance `maker()` builds for a
    given run exposes a `.decisions` list (only `RecordingM`, above); when it does, each
    entry becomes `(seed, rep, t_rel_ms, arm)`, with `t_rel_ms` measured against that
    seed's own `EV24.LBL.run_dict` `t0` -- the same clock `SCORE.terms` uses to score the
    run, read off the dict the harness already builds for every seed and not re-derived
    by a second simulation.
    """
    extra_stats = {name: {"fresh_checked": 0, "fresh_unchecked": 0,
                          "mind_changes": 0, "res_decisions": 0,
                          "lag_draws": collections.Counter()}}
    rows = []
    decisions = []
    t0_cache = {}
    for seed in seeds:
        if seed not in t0_cache:
            t0_cache[seed] = float(EV24.LBL.run_dict(EV24.MAIN_P, seed)["t0"])
        for rep in range(NREP):
            holder = {}

            def wrapped_maker(holder=holder):
                """Build the decider via `maker()` and also stash it in `holder`."""
                inner = maker()
                holder["inner"] = inner
                return inner

            mc_before = extra_stats[name]["mind_changes"]
            rt_before = extra_stats[name]["res_decisions"]
            row = EV24.one_run(name, seed, rep, wrapped_maker, extra_stats)
            row["mind_changes"] = extra_stats[name]["mind_changes"] - mc_before
            row["res_decisions"] = extra_stats[name]["res_decisions"] - rt_before
            rows.append(row)
            inner = holder.get("inner")
            for ts_ms, arm in getattr(inner, "decisions", ()):
                decisions.append((seed, rep, ts_ms - t0_cache[seed], arm))
    return rows, extra_stats[name], decisions


def check_malformed_coef():
    """Exercise `m24.load_coef`'s shape guard on a 4-row array.

    Written to a fresh `tempfile` path rather than anywhere in this directory, because
    only the files named in this effort's rules may be created here.  The path is
    random per run, so it is stripped out of the raised message before re-raising: this
    archive prints no machine path, and a message that varied with the temp directory's
    random name would make two honest runs of this script differ byte for byte.
    """
    bad_arr = np.zeros((4, m24.BLOCK), dtype=np.float64)
    fd, bad_path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    np.save(bad_path, bad_arr)
    try:
        m24.load_coef(bad_path)
    except m24.Fail as exc:
        raise Fail(str(exc).replace(bad_path, "<malformed coefficient file>")) from None
    finally:
        os.remove(bad_path)


def show_mutation(label, fn):
    """Run `fn()` and confirm it raises `Fail`, `m24.Fail`, `R24.Fail`, or `SystemExit`."""
    try:
        fn()
    except (Fail, m24.Fail, R24.Fail, SystemExit) as exc:
        say("  %-32s FIRES: %s" % (label, exc))
        return 1
    say("  %-32s DID NOT FIRE" % label)
    return 0


def csv_value(v):
    """Format one csv cell: `repr` for a float, `str` otherwise (`\"\"` stays `\"\"`)."""
    return repr(v) if isinstance(v, float) else str(v)


def write_drift_csv(path, rows):
    """Write `rows` (dicts keyed by `CSV_COLUMNS`) to `path`, header line first."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(CSV_COLUMNS) + "\n")
        for row in rows:
            fh.write(",".join(csv_value(row[c]) for c in CSV_COLUMNS) + "\n")


def main():
    """Build the neutralised model, replay four deciders, and report every statistic."""
    say("NOT in the frozen plan: this script and its appended RUN.txt section were")
    say("written after every score of M (seeds 81 to 100) was already seen.")
    say("It selects nothing -- the candidate that goes forward is candidate a, as")
    say("shipped (ana24.txt's VERDICT); this script does not reopen that choice.")
    say("The replay below runs on seeds 1 to 20, which are TRAINING seeds: candidate")
    say("a's own fit read their emulator logs, so every comparison in this file is an")
    say("IN-SAMPLE comparison, not a generalisation claim.")
    say("The 0.5 threshold on |within-run corr(decision time)|, used below to name a")
    say("cell 'drifting', was chosen by a verifier after the fact; it is restated here")
    say("and not tuned by this script.")

    head("0  SCOPE, CODE DIGESTS, AND THE TWO SEED RANGES THIS SCRIPT MAY TOUCH")
    say("fit seeds (candidate a, read via rows24.build_all): %d to %d, p 0.2 only"
        % (min(R24.FIT_SEEDS), max(R24.FIT_SEEDS)))
    say("replay seeds (this script's own comparison):        %d to %d, %d streams each"
        % (min(TRAIN_SEEDS), max(TRAIN_SEEDS), NREP))
    say("code digests (sha256):")
    for fname in ("drift24.py", "m24.py", "rows24.py", "eval24.py", "features.py", "charges.py"):
        say("  %-14s %s" % (fname, file_sha256(os.path.join(HERE, fname))))
    say("  %-14s %s" % ("coef_a.npy", file_sha256(os.path.join(HERE, "coef_a.npy"))))

    head("1  CANDIDATE A'S FIT ROWS, AND THE WITHIN-RUN CORRELATION OF EACH CELL")
    data = R24.build_all()
    fit = data["a"]["fit"]
    fit_seeds = check_fit_seed_range({r["seed"] for r in fit}, "candidate a fit rows",
                                     want=R24.FIT_SEEDS)
    say("fit rows: %d, seeds %d to %d (%d distinct)"
        % (len(fit), fit_seeds[0], fit_seeds[-1], len(fit_seeds)))

    x = np.stack([r["vec"] for r in fit])          # an independent copy: stacking copies data
    t = np.array([float(r["t_rel_ms"]) for r in fit]) / 1000.0
    c = np.array([float(r["c"]) for r in fit])
    tau = np.array([float(r["tau"]) for r in fit])
    runs = np.array([r["seed"] for r in fit])
    a_logged_idx = np.array([m24.ARM_INDEX[r["arm"]] for r in fit], dtype=np.int64)
    r_fit = float(c.sum() / tau.sum())
    say("logged rate over the fit rows, R_fit = sum(c)/sum(tau) = %.4f score ms/s" % r_fit)

    cors = np.array([within_corr(x[:, j], t, runs) for j in range(x.shape[1])])
    drift_idx = np.where(np.abs(cors) > DRIFT_THRESH)[0]
    say("cells with |within-run corr(decision time)| > %.1f: %d of %d"
        % (DRIFT_THRESH, len(drift_idx), x.shape[1]))
    say("  max |corr| over all cells: %.4f  (at cell index %d, %s)"
        % (float(np.max(np.abs(cors))), int(np.argmax(np.abs(cors))),
           FE.NAMES[int(np.argmax(np.abs(cors)))]))
    say("  the %d drifting cells, by name and index, with their own within-run"
        % len(drift_idx))
    say("  correlation against decision time (t_rel_ms):")
    for j in drift_idx:
        say("    [%3d] %-28s corr = %+.4f" % (int(j), FE.NAMES[int(j)], float(cors[j])))
    colmean = x[:, drift_idx].mean(axis=0)

    head("2  THE SHIPPED FIT, RE-SOLVED LOCALLY, AS A REPRODUCIBILITY GATE")
    mu, sigma = m24.standardiser(x)
    msys0, rhs0, nr0 = m24.build_system(fit, mu, sigma, LAMBDA, next_mode="logged",
                                        two_rates=False, check=True, where="drift24 base")
    w0, rhos0, st0 = m24.solve_system(msys0, rhs0, nr0, check_diagnostics=True)
    m24.check_telescoped_identity(fit, w0, rhos0, mu, sigma, False, "drift24 base")
    w_blocks_a, mu_a, sigma_a = m24.load_coef(os.path.join(HERE, "coef_a.npy"))
    max_diff = float(np.max(np.abs(w_blocks_a.reshape(-1) - w0)))
    say("rho_0 (re-solved) = %.4f   gap vs R_fit = %+.4f   cond = %.3e"
        % (rhos0[0], rhos0[0] - r_fit, st0["cond"]))
    say("max |shipped coef_a.npy - this re-solve| = %.3e  (max|w| = %.4f)"
        % (max_diff, float(np.max(np.abs(w0)))))
    say("mu/sigma identical to the shipped file's own: %s"
        % (bool(np.array_equal(mu_a, mu) and np.array_equal(sigma_a, sigma))))

    check_shipped_gate(w_blocks_a, w0, GATE_TOL, "drift24 gate: coefficients")
    gi_shipped_file = m24.greedy_indices(m24.q_matrix(x, w_blocks_a, mu_a, sigma_a))
    gi_shipped_resolve = m24.greedy_indices(
        m24.q_matrix(x, w0.reshape(3, m24.BLOCK), mu, sigma))
    check_greedy_agreement(gi_shipped_file, gi_shipped_resolve, "drift24 gate: greedy answers")
    say("GATE PASSED: coefficients agree within %.0e, and the greedy answer on every one"
        % GATE_TOL)
    say("  of the %d fit rows agrees between the shipped file and the local re-solve."
        % len(fit))

    # The Q-value of the arm the LOG actually took, under the shipped weights, used for
    # the within-run slope below.  `a_logged_idx` is a fact of the log and is unaffected
    # by neutralising `vec`, so it is computed once here (before neutralisation) and
    # reused for both the shipped and the neutralised slope.
    q_shipped_full = m24.q_matrix(x, w_blocks_a, mu_a, sigma_a)
    q_shipped_logged = q_shipped_full[np.arange(len(fit)), a_logged_idx]
    slope_shipped = within_slope(q_shipped_logged, t, runs)
    corr_shipped_q = within_corr(q_shipped_logged, t, runs)

    head("3  THE NEUTRALISED FIT: %d CELLS HELD AT THEIR FIT-ROW MEAN, ROWS AND CHAIN ALIKE"
        % len(drift_idx))
    touched = neutralise_in_place(fit, drift_idx, colmean)
    say("records neutralised (rows and chain entries, deduplicated by record identity): %d"
        % touched)
    x2 = np.stack([r["vec"] for r in fit])
    mu2, sigma2 = m24.standardiser(x2)
    msys2, rhs2, nr2 = m24.build_system(fit, mu2, sigma2, LAMBDA, next_mode="logged",
                                        two_rates=False, check=True, where="drift24 neutral")
    w2, rhos2, st2 = m24.solve_system(msys2, rhs2, nr2, check_diagnostics=True)
    m24.check_telescoped_identity(fit, w2, rhos2, mu2, sigma2, False, "drift24 neutral")
    say("rho_2 (neutralised) = %.4f   gap vs R_fit = %+.4f   cond = %.3e"
        % (rhos2[0], rhos2[0] - r_fit, st2["cond"]))

    q0 = m24.q_matrix(x, w_blocks_a, mu_a, sigma_a)
    gi0 = m24.greedy_indices(q0)
    q2 = m24.q_matrix(x2, w2, mu2, sigma2)
    gi2 = m24.greedy_indices(q2)
    n_diff = int((gi0 != gi2).sum())
    say("argmin disagreement, shipped vs neutralised, on the %d fit rows: %d (%.4f)"
        % (len(fit), n_diff, n_diff / float(len(fit))))

    q_neutral_logged = q2[np.arange(len(fit)), a_logged_idx]
    slope_neutral = within_slope(q_neutral_logged, t, runs)
    corr_neutral_q = within_corr(q_neutral_logged, t, runs)
    say()
    say("within-run slope of the fitted Q of the LOGGED interval on decision time")
    say("(score ms per second within a run; a_logged is a fact of the log, unaffected")
    say("by neutralising vec):")
    say("  shipped     (w_blocks_a) slope = %+.4f  within-run corr = %+.4f"
        % (slope_shipped, corr_shipped_q))
    say("  neutralised (w2)         slope = %+.4f  within-run corr = %+.4f"
        % (slope_neutral, corr_neutral_q))

    npy_sha = m24.save_coef(COEF_PATH, w2, mu2, sigma2)
    with open(JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump({
            "candidate": "a_neutral",
            "built_from": "a",
            "drift_threshold": DRIFT_THRESH,
            "drift_cell_count": int(len(drift_idx)),
            "drift_cell_indices": [int(i) for i in drift_idx],
            "drift_cell_names": [FE.NAMES[int(i)] for i in drift_idx],
            "lambda": LAMBDA,
            "fit_rows": int(len(fit)),
            "r_fit": r_fit,
            "rho_base": rhos0[0],
            "rho_neutral": rhos2[0],
            "cond_neutral": st2["cond"],
            "argmin_disagreement_vs_shipped": n_diff,
            "argmin_disagreement_fraction": n_diff / float(len(fit)),
            "within_run_slope_q_logged_shipped": slope_shipped,
            "within_run_slope_q_logged_neutral": slope_neutral,
            "npy_sha256": npy_sha,
        }, fh, indent=2, sort_keys=True)
        fh.write("\n")
    say()
    say("wrote coef_a_neutral.npy (sha256 %s) and coef_a_neutral.json" % npy_sha)

    head("4  FIT-ROW ANSWER SHARES: OVERALL, AND BY DECILE OF DECISION TIME")
    say("deciles are the 9 pooled cutoffs of all %d fit rows' own decision time"
        % len(fit))
    say("(t_rel_ms / 1000, in seconds); the SAME cutoffs are applied to both models, so")
    say("the shipped and neutralised rows below are comparable decile for decile.")
    cuts_fit = decile_cuts(t)
    bins_fit = decile_bin(t, cuts_fit)
    say("decile cutoffs (s): %s" % ", ".join("%.3f" % c for c in cuts_fit))
    say()
    say("overall shares, n=%d fit rows:" % len(fit))
    say("  %-12s %8s %8s %8s" % ("", ARM_LABELS[0], ARM_LABELS[1], ARM_LABELS[2]))
    for label, gi in (("shipped", gi0), ("neutralised", gi2)):
        shares = [float((gi == i).sum()) / len(gi) for i in range(3)]
        say("  %-12s %8.4f %8.4f %8.4f" % (label, shares[0], shares[1], shares[2]))
    say()
    say("by decile of decision time:")
    say("  %-4s %6s   %-26s   %-26s"
        % ("dec", "n", "shipped " + "/".join(ARM_LABELS), "neutral " + "/".join(ARM_LABELS)))
    for d in range(10):
        mask = bins_fit == d
        n = int(mask.sum())
        s_sh = [float((gi0[mask] == i).sum()) / n if n else 0.0 for i in range(3)]
        s_nt = [float((gi2[mask] == i).sum()) / n if n else 0.0 for i in range(3)]
        say("  %-4d %6d   %6.4f %6.4f %6.4f   %6.4f %6.4f %6.4f"
            % (d, n, s_sh[0], s_sh[1], s_sh[2], s_nt[0], s_nt[1], s_nt[2]))

    head("5  THE REPLAY: FOUR DECIDERS, SEEDS 1 TO 20 x %d STREAMS" % NREP)
    replay_seeds = check_replay_seed_range(TRAIN_SEEDS, "drift24 replay")
    say("replay seeds: %d to %d (%d distinct), %d streams each, %d runs per decider"
        % (replay_seeds[0], replay_seeds[-1], len(replay_seeds), NREP,
           len(replay_seeds) * NREP))
    n_measuring = len(set(replay_seeds) & set(range(121, 161)))
    n_in_fit = check_replay_in_fit(replay_seeds, "drift24 replay")
    say("seed guards: port.check_seeds (>100 refused) passed; meets 121-160 in %d seeds;"
        % n_measuring)
    say("             %d of %d replay seeds are also in candidate a's own 1-70 fit seeds"
        % (n_in_fit, len(replay_seeds)))

    w_blocks_n, mu_n, sigma_n = m24.load_coef(COEF_PATH)

    def make_shipped():
        """Fresh shipped decider, reloaded from `coef_a.npy` for every run."""
        return RecordingM(w_blocks_a, mu_a, sigma_a)

    def make_neutral():
        """Fresh neutralised decider, reloaded from `coef_a_neutral.npy` for every run."""
        return RecordingM(w_blocks_n, mu_n, sigma_n)

    def make_formula():
        """`eval24.py`'s own formula-rule decider F, read off its module and not redefined."""
        return EV24.LBL.F()

    def make_fixed():
        """`eval24.py`'s own fixed 6.0 s decider, read off its module and not redefined."""
        return EV24.PORT.const_decider(EV24.ARMS[1])

    FIXED_NAME = EV24.ARM_NAMES[1]
    DECIDERS = (("a", make_shipped), ("a_neutral", make_neutral),
               ("F", make_formula), (FIXED_NAME, make_fixed))

    replay_results = {}
    all_decisions = {}
    for name, maker in DECIDERS:
        rows, stats, decisions = run_replay(name, replay_seeds, maker)
        replay_results[name] = (rows, stats)
        all_decisions[name] = decisions

    say()
    say("  %-12s %6s %12s %10s %10s %14s %12s"
        % ("decider", "runs", "mean_score", "mean_polls", "mean_delay",
           "censored_runs", "mind_chg_share"))
    for name, _maker in DECIDERS:
        rows, stats = replay_results[name]
        n = len(rows)
        mean_score = statistics.fmean(r["score_ms"] for r in rows)
        mean_polls = statistics.fmean(r["polls"] for r in rows)
        mean_delay = statistics.fmean(r["delay_ms"] for r in rows)
        censored = sum(1 for r in rows if r["c4"] > 0)
        rt = stats["res_decisions"]
        mc = stats["mind_changes"]
        mc_share = mc / float(rt) if rt else 0.0
        say("  %-12s %6d %12.3f %10.3f %10.3f %7d / %-4d %12.4f"
            % (name, n, mean_score, mean_polls, mean_delay, censored, n, mc_share))

    say()
    say("interval shares over resolutions, the two M's only (F and the fixed arm do not")
    say("choose among the three by argmin, so eval24.one_run leaves their reads/arm-count")
    say("fields blank):")
    for name in ("a", "a_neutral"):
        rows, _stats = replay_results[name]
        reads = sum(r["reads"] for r in rows)
        ns = sum(r["n_short"] for r in rows)
        nm = sum(r["n_mid"] for r in rows)
        nl = sum(r["n_long"] for r in rows)
        say("  %-10s reads=%6d  %s=%.4f  %s=%.4f  %s=%.4f"
            % (name, reads, ARM_LABELS[0], ns / reads, ARM_LABELS[1], nm / reads,
               ARM_LABELS[2], nl / reads))

    # Pooled decision-time cutoffs for section 7's decile table, drawn from BOTH M's
    # resolutions together so the two models' rows below are cut at the same instants.
    decisions_by_run = {name: collections.defaultdict(list) for name in ("a", "a_neutral")}
    pooled_t_sec = []
    for name in ("a", "a_neutral"):
        for seed, rep, t_rel_ms, arm in all_decisions[name]:
            t_sec = t_rel_ms / 1000.0
            decisions_by_run[name][(seed, rep)].append((t_sec, arm))
            pooled_t_sec.append(t_sec)
    pooled_t_sec = np.array(pooled_t_sec, dtype=np.float64)
    cuts_replay = decile_cuts(pooled_t_sec)

    say()
    say("archiving the per-run table ...")
    csv_rows = []
    for name, _maker in DECIDERS:
        rows, _stats = replay_results[name]
        for r in rows:
            base = {"decider": name, "seed": r["seed"], "stream": r["stream"],
                    "score_ms": r["score_ms"], "polls": r["polls"], "delay_ms": r["delay_ms"],
                    "c4": r["c4"], "n_short": r["n_short"], "n_mid": r["n_mid"],
                    "n_long": r["n_long"], "mind_changes": r["mind_changes"],
                    "res_decisions": r["res_decisions"]}
            if name in ("a", "a_neutral"):
                for col in DECILE_ARM_COLS:
                    base[col] = 0
                decs = decisions_by_run[name].get((r["seed"], r["stream"]), ())
                for t_sec, arm in decs:
                    d = int(decile_bin(np.array([t_sec]), cuts_replay)[0])
                    base["d%d_%.3f" % (d, arm)] += 1
            else:
                for col in DECILE_ARM_COLS:
                    base[col] = ""
            csv_rows.append(base)
    write_drift_csv(CSV_PATH, csv_rows)
    csv_sha = file_sha256(CSV_PATH)
    say("wrote drift24.csv (%d rows, sha256 %s)" % (len(csv_rows), csv_sha))

    head("6  PAIRED BOOTSTRAP DIFFERENCES OVER SEEDS 1 TO 20")
    per_seed = {name: per_seed_mean(replay_results[name][0], replay_seeds)
               for name, _ in DECIDERS}
    idx = PR20.boot_index(len(replay_seeds))
    comparisons = (
        ("a_neutral", "a"),
        ("a", "F"),
        ("a_neutral", "F"),
        ("a", FIXED_NAME),
        ("a_neutral", FIXED_NAME),
    )
    say("  %-12s %-12s %12s %10s %10s %6s %s"
        % ("first", "second", "diff ms", "lo 95%", "hi 95%", "0 in", "first lower"))
    comp_results = {}
    for first, second in comparisons:
        pt, lo, hi, n_low = paired(first, second, per_seed, replay_seeds, idx)
        comp_results[(first, second)] = (pt, lo, hi, n_low)
        say("  %-12s %-12s %12.3f %10.3f %10.3f %6s %2d of %d"
            % (first, second, pt, lo, hi, "yes" if lo <= 0.0 <= hi else "no",
               n_low, len(replay_seeds)))

    pt, lo, hi, n_low = comp_results[("a_neutral", "a")]
    say()
    say("HEADLINE: score(a_neutral) - score(a), seeds 1 to 20, paired by seed:")
    say("  point %.3f ms   95%% interval %.3f to %.3f ms   0 in interval: %s"
        % (pt, lo, hi, "yes" if lo <= 0.0 <= hi else "no"))
    say("  a_neutral scores lower (better) on %d of %d seeds" % (n_low, len(replay_seeds)))
    say("  (negative point / negative interval = neutralised scores lower = better;")
    say("   positive = neutralised scores higher = worse; the sign and the interval are")
    say("   printed above and are not characterised further here)")

    head("7  REPLAY DECISIONS BY DECILE OF DECISION TIME WITHIN THE RUN")
    say("decision time is measured against the run's OWN start: each resolution's own")
    say("obs['where']['ts_ms'] minus that seed's t0 from EV24.LBL.run_dict -- the same")
    say("clock SCORE.terms uses to score the run, read off the dict the harness already")
    say("builds for every seed and not re-derived by a second simulation.")
    say()
    say("decile cutoffs are pooled over shipped a's AND the neutralised model's own")
    say("resolutions together, so the two rows below are cut at the same instants and")
    say("are directly comparable to each other.")
    say("decision-time range seen (s), pooled over both M's: min %.3f  max %.3f"
        % (float(pooled_t_sec.min()), float(pooled_t_sec.max())))
    say("decile cutoffs (s): %s" % ", ".join("%.3f" % c for c in cuts_replay))
    say("These are shares of what each model CHOSE; whether a bigger or steeper move")
    say("here reflects reading the clock or reading some other state cannot be told")
    say("apart by this table alone -- see section 9.  One scalar per model, computed")
    say("and not read off the table by eye, follows each one: the within-run slope and")
    say("correlation of the chosen interval itself (seconds) against decision time,")
    say("with (seed, stream) as the run key.")
    for name in ("a", "a_neutral"):
        arr = all_decisions[name]
        arr_t_sec = np.array([t_rel_ms / 1000.0 for _s, _r, t_rel_ms, _a in arr])
        arr_a_idx = np.array([m24.ARM_INDEX[arm] for _s, _r, _t, arm in arr], dtype=np.int64)
        arr_arm_val = np.array([arm for _s, _r, _t, arm in arr], dtype=np.float64)
        arr_run_id = np.array([seed * 100 + rep for seed, rep, _t, _a in arr], dtype=np.int64)
        bins = decile_bin(arr_t_sec, cuts_replay)
        say()
        say("%s (%d resolutions, pooled over %d seeds x %d streams):"
            % (name, len(arr), len(replay_seeds), NREP))
        say("  %-4s %8s %8s %8s %8s" % ("dec", "n", ARM_LABELS[0], ARM_LABELS[1], ARM_LABELS[2]))
        for d in range(10):
            mask = bins == d
            n = int(mask.sum())
            shares = [float((arr_a_idx[mask] == i).sum()) / n if n else 0.0 for i in range(3)]
            say("  %-4d %8d %8.4f %8.4f %8.4f" % (d, n, shares[0], shares[1], shares[2]))
        slope_arm = within_slope(arr_arm_val, arr_t_sec, arr_run_id)
        corr_arm = within_corr(arr_arm_val, arr_t_sec, arr_run_id)
        say("  within-run slope of the chosen interval (s) on decision time: %+.6f s/s"
            % slope_arm)
        say("  within-run corr of the same pair:                             %+.4f"
            % corr_arm)

    head("8  EVERY CHECK SHOWN FAILING ON A MUTATED INPUT")
    fired = []
    fired.append(show_mutation(
        "fit seed 71 (just above range)",
        lambda: check_fit_seed_range([71], "mut/fit-71")))
    fired.append(show_mutation(
        "fit seed 0",
        lambda: check_fit_seed_range([0], "mut/fit-0")))
    fired.append(show_mutation(
        "fit seed set missing seed 1",
        lambda: check_fit_seed_range([s for s in R24.FIT_SEEDS if s != 1],
                                      "mut/fit-drop-1", want=R24.FIT_SEEDS)))
    fired.append(show_mutation(
        "learn seed 81 (above the 80-seed log cap)",
        lambda: R24.check_learn_seeds([81], "mut/learn-81")))
    fired.append(show_mutation(
        "replay seed 21 (just above range)",
        lambda: check_replay_seed_range([21], "mut/replay-21")))
    fired.append(show_mutation(
        "replay seed 101",
        lambda: check_replay_seed_range([101], "mut/replay-101")))
    fired.append(show_mutation(
        "replay seed 121 (measuring range)",
        lambda: check_replay_seed_range([121], "mut/replay-121")))
    fired.append(show_mutation(
        "replay seed 71 (outside candidate a's own fit seeds)",
        lambda: check_replay_in_fit([71], "mut/replay-not-fit")))
    bad_w_blocks = w_blocks_n.copy()
    bad_w_blocks[0, 0] += 1.0e5          # a large shift so the intercept move cannot
    fired.append(show_mutation(          # cancel across the ~560 blocks by coincidence
        "tampered weight breaks Check A",
        lambda: m24.check_telescoped_identity(
            fit, bad_w_blocks.reshape(-1), rhos2, mu2, sigma2, False, "mut/checka")))
    fired.append(show_mutation("malformed coefficient shape", check_malformed_coef))
    fired.append(show_mutation(
        "shipped gate: coefficient off by 1e-5",
        lambda: check_shipped_gate(w_blocks_a, w0 + 1.0e-5, GATE_TOL, "mut/gate-coef")))
    bad_gi = gi_shipped_resolve.copy()
    bad_gi[0] = (bad_gi[0] + 1) % 3
    fired.append(show_mutation(
        "shipped gate: one greedy answer flipped",
        lambda: check_greedy_agreement(gi_shipped_file, bad_gi, "mut/gate-greedy")))
    say()
    say("%d of %d mutations produced a failure" % (sum(fired), len(fired)))

    head("9  WHAT THIS DOES NOT SHOW")
    say("- NOTHING ABOUT GENERALISATION.  Seeds 1 to 20 are inside the 70 seeds candidate")
    say("  a's own fit already read; this is a within-fit sensitivity, not a held-out or")
    say("  selection-half measurement.  The held seeds (71-80) and the selection seeds")
    say("  (81-100) are untouched by this script -- nothing here is about them.")
    say("- NO RE-SELECTION.  Candidate a stays the chosen candidate (ana24.txt's VERDICT);")
    say("  this script does not compare candidate b and does not reopen that choice.")
    say("- ONE THRESHOLD.  The 0.5 cutoff on |within-run corr(decision time)| is the")
    say("  cross-verification's own after-the-fact choice, restated here and not")
    say("  re-tuned; no other threshold is tried.")
    say("- WHAT A SCORE CHANGE CANNOT SETTLE.  The 40 neutralised cells are held at their")
    say("  fit-row mean, which removes whatever STATE information they also carried, not")
    say("  only the clock-position information the within-run correlation flags.  A")
    say("  worse replay score after neutralising them does not by itself show that the")
    say("  shipped M was reading the clock rather than the state those same cells also")
    say("  describe; an EQUAL score would not have shown the opposite either, because")
    say("  the neutralised model could have picked up the lost state signal through the")
    say("  221 cells left untouched.  Section 7 reports, for each model, the decile")
    say("  shares of its own chosen interval and the within-run slope and correlation")
    say("  of that chosen interval against decision time; these numbers are printed as")
    say("  computed and are NOT characterised here as one model 'reading the clock'")
    say("  more or less than the other -- that reading is left to whoever compares the")
    say("  two printed slopes, not asserted by this script.")
    say("- THE EMULATOR-TO-REPLAY GAP.  Every number in this file comes from")
    say("  `eval24.one_run`'s replay port on logged-style observations; nothing here")
    say("  touches whether the replay port's own dynamics match the live Mininet")
    say("  emulator those logs were recorded from.")
    say("- NOTHING ABOUT WHY a difference, if any, has the sign it has beyond what is")
    say("  printed with its interval above; nothing further is inferred from it.")
    say("- Nothing about candidate b, p 0.4, the held or selection seeds, or any further")
    say("  neutralisation threshold.")
    say()
    say("end of drift24.py")
    return 0 if sum(fired) == len(fired) else 1


if __name__ == "__main__":
    sys.exit(main())
