#!/usr/bin/env python3
"""The prediction for the s9 on/off workload testbed runs, archived before those runs.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src EV19_GEN_DIR=<scratch dir> \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B predict.py > predict.txt

  PYTHONPATH must point at this worktree's `src`: the interpreter's venv has another
  checkout of the package installed, and the assert below stops the script if that is
  the one that got imported. EV19_GEN_DIR is where the generated experiment
  descriptions are written; without it they land in a directory under $TMPDIR.

WHY THIS EXISTS
  evidence/13 compared the replay with a measurement on the default load only. The
  on/off numbers of evidence/14 and 15 have no measurement behind them, so the first
  thing the s9 runs buy is a check of the replay on a load it has never been held to
  (ticket point (a), spec.md pass line 5). A check is only worth something if the
  numbers it is held to were written down first, so every quantity pass line 5 reads is
  printed here, before the first run.

WHY THE CREATION TIMES ARE NOT DRAWN HERE
  evidence/15 drew its own on/off times from the seed string "ev15:<name>:<p>:<seed>:
  <origin>". The committed generator draws from "ev19:<NAME>:<seed>:h<host>", without
  the p, so its instants are a different draw and evidence/15's gain values cannot be
  used as a prediction for these runs. This script therefore asks the worktree's own
  generator for the experiment description each (workload, seed) run will use, reads
  `schedule[host][i]["at_s"]` out of the written file and converts that to the replay's
  `created` list. What the comparison afterwards can then state is that the replay and
  the testbed ran the same instants. Whether those instants are the shape WO and WV are
  defined to have is a separate question, which the generator's own tests answer.

WHAT IT COMPUTES
  Per workload (wo, wv) and per stage (three fixed arms, T37, T37r), 40 replicate
  streams of 20 single runs. One stream is one experiment, the condition the testbed
  has: one run per seed, 20 seeds. Printed per stage: the headline delay (records whose
  scheduled creation is at or before t0+72 s, uncensored mean, 4 hops), the polls per
  run over the whole window and the censored share on the headline set, each as the
  median over the 40 experiments with the plus or minus 6 % band of pass line 5 and the
  per-experiment minimum and maximum; the all-record delay and censored share beside
  them. Then, per workload: the gain of T37 and of T37r against the three-arm log-log
  frontier with a seed-bootstrap interval and that interval's half width, the T37r minus
  T37 difference, how many of the 40 experiments put each gain interval entirely above,
  across or entirely below zero, and the direct T37r/T37 ratios in delay and polls with
  their half widths.

THE W0 CONTROL
  Before any on/off number is printed, the same code path is run on the default load and
  must reproduce ../16/predict.txt's archived T37 and T37r medians (34.627 s / 142.50
  and 25.160 s / 180.15) to the digits those were printed at. A mismatch means the
  engine, the scorer or the replicate stream has moved since that prediction was
  archived, which would make everything below incomparable with it, so the script stops
  and prints the difference instead of printing a prediction.

WHAT IT CANNOT SHOW
  Every number here is simulated. The replay's agreement with a measurement has only
  ever been checked on the default load, which is the thing pass line 5 is for; a band
  printed here says what the replay expects the testbed to measure, and is itself no
  measurement.

The seed strings, the RNG constructors, the bootstrap size, the replicate count and the
worker count are fixed on purpose: changing any of them moves the printed numbers, and a
prediction that moves after it was archived is worth nothing.
"""
import json
import math
import multiprocessing as mp
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Insert 13, 14, 15 in that order, so the front of sys.path ends up 15, 14, 13.
# All three of evidence/13, 14 and 15 hold an `after.py` and the one wanted is 14's
# headline scorer, so `onoff` is imported FIRST on purpose: importing it re-inserts
# evidence/13 and then evidence/14 at the front, which is what makes `after` resolve to
# evidence/14's file and `replay` to evidence/13's. The asserts below are what actually
# holds that resolution down.
for _d in ("13", "14", "15"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", _d)))
import onoff as O  # noqa: E402  evidence/15: the engine, the rule table, sim_run
import after as A  # noqa: E402  evidence/14: the headline (early-record) scorer
import replay as R  # noqa: E402  evidence/13: the run store, the scorer, the seeds
import workload as W  # noqa: E402  evidence/14: check_created

# realpath rather than the raw __file__: evidence/15 inserts its own ".." entries, so replay
# arrives spelled ".../14/../13/replay.py" even though it is evidence/13's file.
assert os.path.realpath(O.__file__).startswith(os.path.realpath(os.path.join(HERE, "..", "15"))), O.__file__
assert os.path.realpath(R.__file__).startswith(os.path.realpath(os.path.join(HERE, "..", "13"))), R.__file__
assert os.path.realpath(A.__file__).startswith(os.path.realpath(os.path.join(HERE, "..", "14"))), A.__file__
assert os.path.realpath(W.__file__).startswith(os.path.realpath(os.path.join(HERE, "..", "14"))), W.__file__

from semindex import evalgen as G  # noqa: E402  the worktree's own generator

# The generator must be THIS worktree's file. The interpreter is a venv that carries
# another checkout of the package, and that checkout's `_onoff_ticks` could draw from
# another seed string, which would make every number below a prediction for instants the
# testbed never schedules. The path is derived from this file's own location rather than
# written out, so it still holds after the branch is merged and the evidence moves.
_SRC = os.path.realpath(os.path.join(HERE, "..", "..", "..", "..", "..", "src"))
assert os.path.realpath(G.__file__).startswith(_SRC), (G.__file__, _SRC)

DONOR = "s6b10p043"                 # the logged runs the outages, t0 and topology come from
P = 0.4                             # the cell the s9 sweep runs
SEEDS = range(1, 21)                # 20 seeds, as the sweep
K = 4                               # records per generation time, as the sweep
DISCONNECT = (8.0, 12.0)            # the outage range, as the sweep
NREP = 40                           # replicate streams, i.e. 40 simulated experiments
WORKERS = 12                        # the task's ceiling
BOOT_N = 2000                       # seed bootstrap resamples, as the post-run analysis
BOOT_SEED = 20261117                # fixed, so the printed bounds are reproducible
BAND = 0.06                         # pass line 5: within plus/minus 6 %
CUT_S = A.CUT_MS / 1000.0           # the headline cut, t0+72 s

WORKLOADS = ["wo", "wv"]
# The rule names are evidence/15's, so the rule table and the engine are used exactly as
# that step uses them. The stage names are the sweep's, so the printed block of a stage
# can be read against that stage's measured row without a lookup table.
RULES = ["fix3.695", "fix6.0", "fix10.043", "T37", "T37r"]
ARMS = ["fix3.695", "fix6.0", "fix10.043"]
STAGE_SUFFIX = {"fix3.695": "b3p695", "fix6.0": "b6p0", "fix10.043": "b10p043",
                "T37": "t37", "T37r": "t37r"}
TESTED = ["T37", "T37r"]
SPEC = {n: (k, s) for n, k, s in O.RULE_SPECS}

# ../16/predict.txt's archived medians. The control below must land on these strings.
W0_EXPECT = {"T37": ("34.627", "142.50"), "T37r": ("25.160", "180.15")}

# Filled in the parent process before the pool is created, so every worker inherits the
# same creation times: {(workload, seed): [(ts_ms, origin, seq), ...]}. "w0" maps to None
# and means the donor run's own logged stamps (see `created_for`).
CREATED: dict[tuple[str, int], list | None] = {}


def gen_dir():
    """Where the generated experiment descriptions are written.

    Outside the repository: these are inputs to the prediction rather than evidence, and the
    generator writes a ReCefore yaml and a config per node beside each description.
    """
    d = os.environ.get("EV19_GEN_DIR") or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "ev19-predict-exp")
    return d


def generate_created(workload, seed):
    """The replay's `created` list for one (workload, seed), via the committed generator.

    The generator is asked for the whole run, written to disk, and the schedule is read
    back out of the written file rather than out of the returned dict: the file is the
    one the testbed reads, so reading it is what makes a later file-against-file
    comparison of the two sides meaningful.

    `at_s` is converted the way evidence/15 converts its own instants,
    `round(t0 + at_s*1000)`, and the same way the replay's scorer expects: whole
    milliseconds. The two origin hosts 0 and 4 become "h0" and "h4", and `seq` runs 1..N
    per origin in schedule order, which is time order because the generator appends the
    k records of a tick together in increasing `at_s`.

    The stage name and the poll settings are passed because the generator wants a
    complete run, and they are the sweep's first arm for this workload. Neither enters
    `schedule`: the instants are a function of (seed, k, workload) and the attrs of
    (seed, host), so one generated stage per (workload, seed) covers all five stages of
    that workload.
    """
    outdir = os.path.join(gen_dir(), workload)
    paths = G.generate_run(
        P, seed, f"s9{workload}_b3p695", outdir,
        k=K, workload=workload, store_synchronous="NORMAL",
        state_poll_interval_choices_s=[3.695], state_poll_choice_jitter_frac=0.1,
        disconnect_duration_s=DISCONNECT,
    )
    with open(paths[0], encoding="utf-8") as fh:
        raw = json.load(fh)
    t0 = R.load_run(DONOR, P, seed)["t0"]
    out = []
    for host in G.RECORD_ORIGIN_HOSTS:
        entries = raw["schedule"][str(host)]
        assert entries, (workload, seed, host)
        at = [e["at_s"] for e in entries]
        assert at == sorted(at), (workload, seed, host)   # seq order must be time order
        out += [(float(round(t0 + s * 1000.0)), f"h{host}", i)
                for i, s in enumerate(at, 1)]
    out.sort()                      # load_run sorts its own list the same way
    W.check_created(out, t0, (workload, seed))
    return out


def created_for(workload, seed, donor_run):
    """The creation times one simulated run sees.

    "w0" is the control and means the donor run's own logged stamps, which is what
    evidence/15 calls W0 and what ../16/predict.py fed: the archived medians this control
    reproduces were produced from those stamps. The generated w0 schedule would NOT
    reproduce them, because a logged stamp carries the lag between the instant a record
    was due and the instant the agent made it (median 91 ms), so feeding the scheduled
    instants would move every delay by that lag.
    """
    if workload == "w0":
        return donor_run["created"]
    return CREATED[(workload, seed)]


def build_run(donor_run, created, name, seed):
    """The simulator input for one stage, built from the donor run of this seed.

    A fixed arm has no rule of its own. Where a logged run of that arm exists its
    policy, choices and jitter are copied in and the rule is cleared; the 6.0 s arm has
    no logged run, so it is simulated as a rule whose two bounds are equal, which is how
    evidence/14 and 15 have always simulated it. A tested rule is the other way round:
    the logged schedule is cleared so the rule alone decides when a poll goes out.

    Only `created`, the poll settings and the rule are replaced. The outage timeline,
    `agent_start`, t0 and the topology stay the donor run's own, which is what makes the
    stages of one workload a comparison on one set of outages.
    """
    kind, spec = SPEC[name]
    run = dict(donor_run)
    run["created"] = created
    if kind in ("fixed", "policy"):
        tgt = R.load_run(DONOR[:2] + spec, P, seed)
        for key in ("policy", "choices", "choice_jitter"):
            run[key] = tgt[key]
        run["rule"] = None
    else:
        run["policy"] = None
        run["choices"] = None
        run["rule"] = dict(spec) if kind == "rule" else dict(i_min=spec, m=1.0, i_max=spec, jf=O.T.JF)
    return run


def job(arg):
    """Simulate every stage of one (workload, seed) NREP times, two scored rows each.

    The replicate seed string is "power:<donor>", ../16/predict.py's own string. It is
    kept rather than renamed because the W0 control has to land on that step's archived
    digits, and the digits depend on the stream: a renamed stream would turn the control
    into a different quantity that nothing can be compared with.

    Each simulation is scored twice from the same event log, never simulated twice: the
    all-record score over the whole window, and the headline score over the records whose
    creation is at or before t0+72 s. In the replay a record is created at exactly the
    instant the schedule names, so the headline scorer's `created <= t0+72 s` cut selects
    exactly the records the testbed will select by `at_s <= 72.0`, the boundary record at
    `at_s` 72.0 included in both.
    """
    workload, seed = arg
    donor_run = R.load_run(DONOR, P, seed)
    created = created_for(workload, seed, donor_run)
    out = {}
    for name in RULES:
        run = build_run(donor_run, created, name, seed)
        rows = []
        for rep in range(NREP):
            sim_rng = random.Random(R.rep_seed(f"power:{DONOR}", P, seed, rep))
            ev = O.sim_run(run, sim_rng)
            s = R.score(ev, donor_run["t0"], P)
            rows.append(((s["n4"], s["c4"], s["lat4"], s["polls"]),
                         A.early_score(ev, donor_run["t0"])))
        out[name] = rows
    return (workload, seed), out


def vals(m):
    """Pool one experiment's per-seed rows into (delay ms, polls per run, censored share).

    Delay is pooled over records rather than averaged over seeds, so a seed with more records
    weighs more. That is the pooling every earlier step's scorer uses, and under an
    on/off load it matters more than it did: the number of records per run moves by a
    factor of two and more across seeds.
    """
    n4, c4, lat, polls = m.sum(0)
    return lat / n4, polls / len(m), c4 / (n4 + c4)


def interp(pts, x):
    """Log-log piecewise-linear reading of the fixed-interval frontier at polls rate `x`.

    Delay against polls is close to a straight line in logs over this range, and a
    linear reading would understate the frontier and so overstate every gain. Returns
    None when `x` falls outside the measured arms, because extrapolating a frontier that
    has no arm beyond the point would invent the comparison.
    """
    pts = sorted(pts)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + w * (math.log(y1) - math.log(y0)))
    return None


def gain(m, name, col):
    """How much delay the rule saves against the frontier at its own polls rate.

    Comparing delay alone would reward a rule for polling more often, so the rule is
    charged for its polls: the frontier is read at the rule's own polls rate and the gain
    is the fraction of that delay removed. None when the arms are not monotone (more
    polls, less delay) or the rule's polls fall outside them, because a frontier that
    does not fall cannot say what the rule's polls are worth.
    """
    delay, polls, _ = vals(m[name][col])
    arms = [(vals(m[a][col])[1], vals(m[a][col])[0]) for a in ARMS]
    ordered = sorted(arms)
    if not all(y0 > y1 for (_, y0), (_, y1) in zip(ordered, ordered[1:])):
        return None
    frontier = interp(arms, polls)
    if frontier is None:
        return None
    return 1 - delay / frontier


def experiment(res, workload, rep):
    """One experiment's arrays: {rule: {col: (20, 4) array}} for replicate stream `rep`.

    Column 0 is the all-record score, column 1 the headline score; both come from the
    same simulations, so the two scorings of one experiment are paired seed by seed.
    """
    return {k: {c: np.array([res[(workload, s)][k][rep][c] for s in SEEDS], float)
                for c in (0, 1)}
            for k in RULES}


def boot_index(rng):
    """The one (BOOT_N, 20) matrix of resampled seed positions, drawn once.

    The resampled unit is the seed number: a seed carries the value of all five stages of
    its workload, so a resample recomputes every stage from the same seeds and the
    comparison between stages stays paired. Drawing the matrix once and reusing it for
    every experiment and both workloads is what makes the printed bounds a function of
    BOOT_SEED alone, independent of the order the sections are printed in.
    """
    return rng.integers(0, len(list(SEEDS)), size=(BOOT_N, len(list(SEEDS))))


def boot_gains(m, idx, col):
    """Per-resample gains (T37, T37r, difference) for one experiment, and what was dropped.

    A resample whose arms are not monotone, or whose rule polls leave the arms, yields no
    gain. It is counted and left out rather than replaced, because substituting a value
    would hide how often the frontier is unreadable on 20 seeds.
    """
    a, b, d = [], [], []
    dropped = 0
    for row in idx:
        sub = {k: {col: m[k][col][row]} for k in RULES}
        g = {t: gain(sub, t, col) for t in TESTED}
        if any(v is None for v in g.values()):
            dropped += 1
            continue
        a.append(g["T37"])
        b.append(g["T37r"])
        d.append(g["T37r"] - g["T37"])
    return a, b, d, dropped


def boot_ratios(m, idx, col):
    """Per-resample T37r/T37 ratios minus one, in delay and in polls per run.

    The direct comparison needs no frontier, so no resample is ever dropped here. This is
    the quantity pass line 4 reads, and its half width is what the spec promises before
    the runs.
    """
    dd, pp = [], []
    for row in idx:
        x = vals(m["T37"][col][row])
        y = vals(m["T37r"][col][row])
        dd.append(y[0] / x[0] - 1)
        pp.append(y[1] / x[1] - 1)
    return dd, pp


def ci(v):
    """(lower, upper, half width) of the 95 % percentile interval of `v`, in percent."""
    lo, hi = np.percentile(v, [2.5, 97.5])
    return 100 * lo, 100 * hi, 100 * (hi - lo) / 2


def word(lo, hi):
    """The three words pass line 3 attaches to an interval, by where it sits around zero."""
    if lo > 0:
        return "seen"
    if hi < 0:
        return "reversed"
    return "not-seen"


def run_all(keys):
    """Simulate every (workload, seed) in `keys` in a worker pool."""
    with mp.Pool(WORKERS) as pool:
        return dict(pool.map(job, keys))


def control():
    """Reproduce ../16/predict.txt's T37 and T37r medians, or stop.

    The control runs the same `job` as the prediction, on the same donor, seeds, stream
    and scorer; only the creation times differ (the donor's logged stamps). If the two
    medians still print as the archived strings, then the engine, the rule table, the
    scorer and the replicate stream are the ones that prediction was made with, and the
    on/off numbers below sit on the same footing. Nothing is adjusted on a mismatch: an
    adjusted control would check nothing.
    """
    res = run_all([("w0", s) for s in SEEDS])
    bad = []
    for name in TESTED:
        v = np.array([vals(experiment(res, "w0", rep)[name][0]) for rep in range(NREP)])
        got = ("%.3f" % (np.median(v[:, 0]) / 1000.0), "%.2f" % np.median(v[:, 1]))
        want = W0_EXPECT[name]
        print("W0 control %-4s delay %s s (../16: %s) | polls %s (../16: %s) | %s"
              % (name, got[0], want[0], got[1], want[1],
                 "match" if got == want else "DIFFERENT"))
        if got != want:
            bad.append((name, got, want))
    if bad:
        print("W0 control failed; no prediction is printed. Differences:")
        for name, got, want in bad:
            print("  %s got delay %s s polls %s, ../16/predict.txt has %s s and %s"
                  % (name, got[0], got[1], want[0], want[1]))
        sys.exit(1)


def print_counts():
    """The record counts of this seed string, which the ticket's table predates.

    The ticket's per-run counts are evidence/15's draw. The generator draws from another
    seed string, so the counts are redrawn here; the headline count is the one that
    limits the precision of pass lines 3 to 5, because a record scheduled after t0+72 s
    is not scored in the headline at all.
    """
    print("record counts of the generated schedules (k 4, seeds 1-20, both origins):")
    for wl in WORKLOADS:
        allc = np.array([len(CREATED[(wl, s)]) for s in SEEDS], float)
        head = np.array([sum(1 for ts, _o, _q in CREATED[(wl, s)]
                             if ts - R.load_run(DONOR, P, s)["t0"] <= A.CUT_MS)
                         for s in SEEDS], float)
        thin = list(SEEDS)[int(head.argmin())]
        print("  %s all %.1f [%d .. %d] | headline (at_s <= %.0f s) %.1f [%d .. %d]"
              " | thinnest headline seed %d with %d records"
              % (wl, allc.mean(), allc.min(), allc.max(), CUT_S,
                 head.mean(), head.min(), head.max(), thin, head.min()))


def print_stages(res, workload):
    """Each stage's predicted headline and all-record values, with pass line 5's band.

    The median over the 40 experiments is the predicted value rather than the mean,
    because it is the middle of what a one-run-per-seed experiment produces and is not
    moved by one stream that drew a bad set of outages. The per-experiment minimum and
    maximum are printed beside it so that a measured value outside the band can be read
    against how much the thin condition moves on its own.
    """
    print("%s stages: headline = records scheduled at or before t0+%.0f s, 4 hops,"
          " uncensored mean; polls over [t0+10 s, t0+130 s]" % (workload, CUT_S))
    print("  stage          | head delay s median [band] min..max        | polls median"
          " [band] min..max           | head cens median min..max | all delay s"
          " median min..max  | all cens median min..max")
    for name in RULES:
        v = np.array([vals(experiment(res, workload, rep)[name][1]) for rep in range(NREP)])
        a = np.array([vals(experiment(res, workload, rep)[name][0]) for rep in range(NREP)])
        d, pl, c = (np.median(v[:, i]) for i in range(3))
        print("  %-14s | %7.3f [%7.3f..%7.3f] %6.3f..%6.3f | %6.2f [%6.2f..%6.2f]"
              " %6.2f..%6.2f | %5.3f %5.3f..%5.3f | %7.3f %6.3f..%6.3f | %5.3f %5.3f..%5.3f"
              % ("s9%s_%s" % (workload, STAGE_SUFFIX[name]),
                 d / 1000.0, d / 1000.0 * (1 - BAND), d / 1000.0 * (1 + BAND),
                 v[:, 0].min() / 1000.0, v[:, 0].max() / 1000.0,
                 pl, pl * (1 - BAND), pl * (1 + BAND), v[:, 1].min(), v[:, 1].max(),
                 c, v[:, 2].min(), v[:, 2].max(),
                 np.median(a[:, 0]) / 1000.0, a[:, 0].min() / 1000.0, a[:, 0].max() / 1000.0,
                 np.median(a[:, 2]), a[:, 2].min(), a[:, 2].max()))
    print("  the band is the median times 1 -+ %d %%: pass line 5 reads the head delay"
          " and the polls of each stage." % round(100 * BAND))


def print_gains(res, workload, idx):
    """The gain against the three-arm frontier, its interval and that interval's half width.

    Every number is the median over the 40 experiments, so what is printed is what a
    typical one-run-per-seed experiment produces rather than what pooling 40 of them would. The
    half width is the quantity spec.md's "what this step cannot show" argues with: an
    effect smaller than it cannot be separated from zero on 20 seeds, whatever its sign.
    The experiment counts below say how often the thin condition lands on each of the
    three words, which is the prediction for pass line 3 being written down as a number.
    """
    for col, label in ((1, "headline"), (0, "all records")):
        rows = {t: [] for t in TESTED}
        diff = []
        tally = {t: {"seen": 0, "not-seen": 0, "reversed": 0} for t in TESTED}
        tally["diff"] = {"seen": 0, "not-seen": 0, "reversed": 0}
        dropped_total = 0
        for rep in range(NREP):
            m = experiment(res, workload, rep)
            a, b, d, dropped = boot_gains(m, idx, col)
            dropped_total += dropped
            for t, v in (("T37", a), ("T37r", b)):
                point = gain(m, t, col)
                lo, hi, hw = ci(v)
                rows[t].append((100 * point, lo, hi, hw))
                tally[t][word(lo, hi)] += 1
            lo, hi, hw = ci(d)
            diff.append((100 * (gain(m, "T37r", col) - gain(m, "T37", col)), lo, hi, hw))
            tally["diff"][word(lo, hi)] += 1
        print("%s gain against the 3-arm log-log frontier (%s scoring), median of the"
              " %d experiments:" % (workload, label, NREP))
        for t in TESTED:
            v = np.array(rows[t])
            print("  %-4s gain %+6.2f %% [%+6.2f .. %+6.2f] half width %5.2f pp"
                  " | experiments seen %d / not seen %d / reversed %d"
                  % (t, np.median(v[:, 0]), np.median(v[:, 1]), np.median(v[:, 2]),
                     np.median(v[:, 3]), tally[t]["seen"], tally[t]["not-seen"],
                     tally[t]["reversed"]))
        v = np.array(diff)
        print("  T37r minus T37 %+6.2f pp [%+6.2f .. %+6.2f] half width %5.2f pp"
              " | experiments seen %d / not seen %d / reversed %d"
              % (np.median(v[:, 0]), np.median(v[:, 1]), np.median(v[:, 2]),
                 np.median(v[:, 3]), tally["diff"]["seen"], tally["diff"]["not-seen"],
                 tally["diff"]["reversed"]))
        print("  resamples with no readable frontier: %d of %d"
              % (dropped_total, NREP * BOOT_N))


def print_ratios(res, workload, idx):
    """The direct T37r/T37 comparison, which needs no frontier (pass line 4).

    Ratios rather than differences: the testbed will not reproduce the absolute delay
    exactly, and the ratio is the part of the prediction that should survive that.

    The polls ratio is printed once rather than once per scoring: polls are counted over the
    whole window under both scorings, so the two numbers would be the same number twice.
    """
    for col, label in ((1, "headline"), (0, "all records")):
        dd, pp = [], []
        for rep in range(NREP):
            m = experiment(res, workload, rep)
            a, b = boot_ratios(m, idx, col)
            dd.append((100 * (vals(m["T37r"][col])[0] / vals(m["T37"][col])[0] - 1),) + ci(a))
            pp.append((100 * (vals(m["T37r"][col])[1] / vals(m["T37"][col])[1] - 1),) + ci(b))
        rows = [("delay (%s scoring)" % label, np.array(dd))]
        if col == 1:
            rows.append(("polls (whole window)", np.array(pp)))
        for what, v in rows:
            print("%s T37r vs T37 %-22s %+6.2f %% [%+6.2f .. %+6.2f] half width %5.2f pp"
                  % (workload, what, np.median(v[:, 0]), np.median(v[:, 1]),
                     np.median(v[:, 2]), np.median(v[:, 3])))


def main():
    """The control, then the prediction, in a fixed order on stdout."""
    control()
    for wl in WORKLOADS:
        for seed in SEEDS:
            CREATED[(wl, seed)] = generate_created(wl, seed)
    print_counts()
    res = run_all([(wl, s) for wl in WORKLOADS for s in SEEDS])
    idx = boot_index(np.random.default_rng(BOOT_SEED))
    for wl in WORKLOADS:
        print_stages(res, wl)
        print_gains(res, wl, idx)
        print_ratios(res, wl, idx)


if __name__ == "__main__":
    main()
