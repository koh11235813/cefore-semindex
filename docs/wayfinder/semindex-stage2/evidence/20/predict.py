#!/usr/bin/env python3
"""The prediction for the s10 baseline testbed runs, archived before those runs.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B predict.py > predict.txt

  PYTHONPATH must point at this worktree's `src`: the interpreter's venv carries
  another checkout of the package, and the assert below stops the script if that is
  the one that got imported.

WHY THIS EXISTS
  Ticket 07 spends 200 testbed runs on seeds 121 to 160, which are the measuring stick
  of the whole three-way comparison and are never used for training, labels, model
  selection or tuning. Three of the five stages have no testbed number in this cell at
  all (the 6.0 s arm, T37r and the pinned F), so the step is also the first testbed
  check of the replay on them. A check is worth something only if the numbers it is
  held to were written down first, so every quantity the pass lines read is printed
  here, before the first run.

WHAT THE HELD-OUT SEEDS CANNOT SUPPLY, AND WHAT IS PUT IN THEIR PLACE
  evidence/13's run dict carries four exogenous inputs: the outage timeline, the record
  creation stamps, t0 and the per-node `agent_start`. The store's largest seed is 100,
  so no run of seeds 121 to 160 exists and none of the four can be read off a run.
    - the outage timeline and the creation stamps are drawn from this worktree's own
      `evalgen.build_experiment`, which depends on (p, seed, k, workload) alone, so the
      instants are the ones the testbed will schedule;
    - t0 and `agent_start` are borrowed from the logged run of donor seed
      1 + (seed - 121) % 20, which spreads the 40 new seeds over all 20 donors instead
      of putting one donor's clock under all of them.
  Three differences come with that substitution, all measured below rather than
  asserted: the testbed's realized outage instant against the scheduled one, the spread
  of `agent_start - t0` across the donors, and the lag between the instant a record was
  due and the instant the agent made it. None of them is corrected for. The replay
  creates a record at exactly the scheduled instant.

THE GATES
  Nothing is printed as a prediction before two reproductions pass.
    1. evidence/18/bound_cells.csv, cell s6b10p043 p 0.2 W0, seeds 1 to 20: the
       early-record delay and polls per run of the three arms this step runs and of
       T37r, to the digits that csv prints, and the pinned F member's delay and polls to
       the digits evidence/20/fpin.txt prints. This says that the engine, the frontier
       inputs and the replicate stream are the ones F was pinned with.
    2. `build_experiment`'s outage events and record schedule, against the stored
       `*.experiment.json` of the same 20 runs, compared as objects. evidence/19 read
       its schedule back out of a written file; here nothing is written, because
       `generate_run` would lay a ReCefore yaml and a per-node config beside every
       description and those configs carry key material.
  A mismatch in either prints the difference and no prediction.

WHAT IT CANNOT SHOW
  Every number here is simulated. The replay has been held against a measurement in
  this cell for the fixed 0.5, 1.359, 3.695 and 10.043 s arms and the short and wait
  policies only (evidence/13). A band printed here says what the replay expects the
  testbed to measure, and is itself no measurement.

The seed strings, the RNG constructors, the bootstrap size, the replicate count and the
donor map are fixed on purpose: changing any of them moves the printed numbers, and a
prediction that moves after it was archived is worth nothing.
"""
import glob
import json
import math
import multiprocessing as mp
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV18 = os.path.normpath(os.path.join(HERE, "..", "18"))
sys.path.insert(0, EV18)
import bound as B  # noqa: E402  evidence/18's engine, frontier, members and jobs

assert B.__file__.startswith(EV18), B.__file__

# The modules evidence/18 already resolved, taken through it rather than imported again:
# a second import of `after`/`replay` off a different sys.path entry would give a second
# module object and the reproduction gate would then compare two scorers.
A, R, T, W = B.A, B.R, B.T, B.W

from semindex import evalgen as G  # noqa: E402  the worktree's own generator

# The generator must be THIS worktree's file. The venv carries another checkout, whose
# `_disconnect_events` or `_record_schedule` could draw differently, which would make
# every number below a prediction for instants the testbed never schedules.
_SRC = os.path.realpath(os.path.join(HERE, *[os.pardir] * 5, "src"))
assert os.path.realpath(G.__file__).startswith(_SRC), (G.__file__, _SRC)

DONOR, P, WL = "s6b10p043", 0.2, "W0"
SEL_SEEDS = list(B.SEEDS)               # 1 to 20: the seeds the gates are read on
NEW_SEEDS = list(range(121, 161))       # the held-out seeds this step measures
K = 4                                   # records per generation time, as the sweep
DISCONNECT = (8.0, 12.0)                # the outage range, as the sweep
NREP = 40                               # replicate streams, i.e. 40 simulated experiments
WORKERS = min(B.WORKERS, os.cpu_count() or 1)
BOOT_N = 2000                           # seed bootstrap resamples, as the post-run analysis
BOOT_SEED = 20261122                    # the date this spec was written, as spec.md fixes
BAND = 0.06                             # pass line 4: within plus or minus 6 %
CUT_S = A.CUT_MS / 1000.0

# The pinned F, read off fpin.txt's winner line, and T37r, in evidence/18's member shape
# (i_min, factor, i_max, i_timeout, k_cap). `i_timeout == i_min` is the testbed's
# `on_timeout: reset`; fpin.py checked that empirically on T37r before pinning anything.
F_MEM = (3.695, 1.5, 10.043, 3.695, None)
T37R_MEM = (3.695, 2.0, 10.043, 3.695, None)

# One entry per testbed stage, in the order main10.sh runs them.
STAGES = [("s10b3p695", ("arm_logged", 3.695)),
          ("s10b6p0", ("arm_rule", 6.0)),
          ("s10b10p043", ("arm_logged", 10.043)),
          ("s10t37r", ("member", T37R_MEM)),
          ("s10f", ("member", F_MEM))]
STAGE_NAMES = [s for s, _ in STAGES]
ARM_STAGES = ["s10b3p695", "s10b6p0", "s10b10p043"]
TESTED = ["s10t37r", "s10f"]

# Gate 1: which bound_cells.csv row each stage has to reproduce. The 6.0 s arm has no
# logged run anywhere, which is why evidence/14 onward simulate it as a rule whose two
# bounds are equal; `fix6.0` is that rule's row.
GATE_RULE = {"s10b3p695": "fix3.695", "s10b6p0": "fix6.0",
             "s10b10p043": "fix10.043", "s10t37r": "T37r"}
# The two arms that DO have logged runs, whose poll settings are copied onto the
# held-out runs rather than invented here.
ARM_DONOR_STAGE = {3.695: "s6b3p695", 10.043: "s6b10p043"}
# fpin.txt's printed digits for the pinned F on seeds 1 to 20, the second half of gate 1.
FPIN_F = ("16665.0", "182.07")

ARM_SET: dict[float, dict] = {}         # filled before the pool is created
# The generated exogenous halves, keyed by (source, seed), likewise filled before the
# fork. Three sources: "gen" is the held-out seeds, "gen20" and "gen20r" are seeds 1 to
# 20 built twice, once with each seed's own clock and once with its neighbour's, which
# is what splits the cost of the substitution into its two halves.
SYN: dict[tuple[str, int], dict] = {}


# ---------------------------------------------------------------------------
# The held-out run dicts
# ---------------------------------------------------------------------------

def donor_seed(seed):
    """Which logged run lends t0 and `agent_start` to one held-out seed.

    The 40 held-out seeds are spread over all 20 donors, two each, so that no single
    donor's clock sits under the whole prediction. The map is arithmetic rather than
    drawn, so it does not move if this file is run again.
    """
    return 1 + (seed - NEW_SEEDS[0]) % len(SEL_SEEDS)


def experiment_of(seed):
    """The generated experiment description of one (p, seed), with no files written.

    `build_experiment` is used in place of `generate_run` because the random schedule it
    returns depends on (p, seed, k, workload) alone, while `generate_run` would also
    write a ReCefore yaml and a per-node config, and those configs carry key material
    that must not be copied anywhere. The stage name and the poll settings are passed
    because the generator wants a complete run; neither enters `events` or `schedule`,
    so one description per seed covers all five stages of that seed.
    """
    return G.build_experiment(
        P, seed, "s10b3p695", "/nonexistent", k=K, store_synchronous="NORMAL",
        disconnect_duration_s=DISCONNECT,
        state_poll_interval_choices_s=[3.695], state_poll_choice_jitter_frac=0.1)


def timeline_of(exp, t0):
    """The replay's outage timeline from a generated description's link events.

    A logged run's timeline is built from `results.json`'s `actual_at`, the instant the
    interface change actually happened. A description has only the scheduled `at`, so
    this is the scheduled timeline; the gap between the two is measured below and is not
    corrected for.
    """
    tl = defaultdict(list)
    for e in exp["events"]:
        if e["type"] in ("link_down", "link_up"):
            tl[frozenset(e["nodes"])].append(
                (t0 + float(e["at"]) * 1000.0, e["type"] == "link_up"))
    for v in tl.values():
        v.sort()
    return dict(tl)


def created_of(exp, t0):
    """The replay's `created` list from a generated description's record schedule.

    `at_s` becomes whole milliseconds the way evidence/15 and evidence/19 convert their
    own instants, `round(t0 + at_s*1000)`. `seq` runs 1..N per origin in schedule order,
    which is time order because the generator appends the k records of one tick together
    in increasing `at_s`. The replay therefore creates every record at exactly the
    instant the schedule names, while the testbed's agent creates it a little later; that
    lag is measured below.
    """
    out = []
    for host in G.RECORD_ORIGIN_HOSTS:
        at = [e["at_s"] for e in exp["schedule"][str(host)]]
        assert at and at == sorted(at), (host, at[:4])
        out += [(float(round(t0 + s * 1000.0)), f"h{host}", i)
                for i, s in enumerate(at, 1)]
    out.sort()
    return out


def neighbors_of(exp):
    """The adjacency the engine walks, from the description's own `links`.

    Taken from the description rather than from the donor run so that the topology comes
    from the same place as the outages: a link the timeline names must be a link the
    adjacency has, or an outage would apply to nothing.
    """
    nb = defaultdict(list)
    for a, b in (tuple(sorted(pair)) for pair in exp["links"]):
        nb[f"h{a}"].append(f"h{b}")
        nb[f"h{b}"].append(f"h{a}")
    return {n: sorted(v, key=lambda h: int(h[1:])) for n, v in nb.items()}


def synthetic_run(seed, donor_of):
    """The exogenous half of one run from generated schedules and a borrowed clock.

    `donor_of` says which logged run lends t0 and `agent_start`. It is `donor_seed` for
    the held-out seeds and the identity for seeds 1 to 20, where the point of the
    construction is to change the timeline and the creation stamps and nothing else.
    """
    donor = B.run_of(DONOR, P, donor_of(seed))
    t0 = donor["t0"]
    exp = experiment_of(seed)
    created = created_of(exp, t0)
    W.check_created(created, t0, ("s10", P, seed))
    return dict(stage="s10", p=P, seed=seed, run_dir=None, t0=t0,
                agent_start=donor["agent_start"], timeline=timeline_of(exp, t0),
                created=created, neighbors=neighbors_of(exp), group=exp["group"],
                policy=None, choices=None, choice_jitter=T.JF, rule=None,
                stop_at_ms=donor["stop_at_ms"])


def arm_settings():
    """The poll settings of the two logged arms, asserted constant over seeds 1 to 20.

    A fixed arm's `--interval-choices` setting is a property of the stage and not of the
    seed, so copying one seed's settings onto all 40 held-out runs is only safe if every
    logged seed carries the same three values. That is asserted here rather than
    assumed, because an arm whose choices differed by seed would make the frontier a
    comparison of two different arms.
    """
    out = {}
    for iv, stage in ARM_DONOR_STAGE.items():
        seen = {(B.run_of(stage, P, s)["policy"],
                 tuple(B.run_of(stage, P, s)["choices"]),
                 B.run_of(stage, P, s)["choice_jitter"]) for s in SEL_SEEDS}
        assert len(seen) == 1, (stage, seen)
        policy, choices, cj = seen.pop()
        out[iv] = dict(policy=policy, choices=list(choices), choice_jitter=cj)
    return out


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def base_run(seed, source):
    """The exogenous half of one simulated run, from the store or from the generator.

    "stored" is the control of the systematic check: the same five stages on the logged
    outages, logged creation stamps and logged clock of a seed 1 to 20 run, which is what
    every earlier replay number of this cell rests on. "gen20" replaces the outages and
    the stamps on those seeds and keeps the clock; "gen20r" replaces the clock as well,
    by the next seed's. "gen" is what the prediction is built from, and it carries both
    replacements at once.
    """
    if source == "stored":
        donor = B.run_of(DONOR, P, seed)
        run = dict(donor)
        run["created"] = B.created_of(WL, P, seed, donor)
        return run
    return SYN[(source, seed)]


def stage_run(base, spec):
    """One stage's run dict, built on an exogenous half.

    A logged arm keeps the settings the testbed stage will carry and no rule. The 6.0 s
    arm has no logged run, so it is a rule with equal bounds and factor 1, which is how
    evidence/14 onward have always simulated it. A member clears the choice settings so
    that the rule alone decides when a poll goes out.
    """
    kind, val = spec
    run = dict(base)
    if kind == "arm_logged":
        run.update(ARM_SET[val])
        run["rule"] = None
    elif kind == "arm_rule":
        run["policy"] = run["choices"] = None
        run["rule"] = dict(i_min=val, m=1.0, i_max=val, jf=T.JF)
    else:
        run["policy"] = run["choices"] = None
        run["rule"] = B.member_rule(val)
    return run


def job(arg):
    """Simulate all five stages of one (source, seed) NREP times, two scored rows each.

    The replicate stream is evidence/18's own, `B.rep_rng_of`, so that the gate above and
    the prediction below are drawn from one stream and the gate actually constrains the
    prediction. Each simulation is scored twice from the same event log and never
    simulated twice: the headline score over the records created at or before t0+72 s and
    the all-record score over the whole window.
    """
    source, seed = arg
    base = base_run(seed, source)
    t0 = base["t0"]
    out = {}
    for name, spec in STAGES:
        run = stage_run(base, spec)
        rows = []
        for rep in range(NREP):
            ev = B.sim(run, B.rep_rng_of(DONOR, P, seed, rep))
            s = R.score(ev, t0, P)
            rows.append(((s["n4"], s["c4"], s["lat4"], s["polls"]),
                         A.early_score(ev, t0)))
        out[name] = rows
    return (source, seed), out


def run_all(keys, pool):
    return dict(pool.map(job, keys, chunksize=1))


# ---------------------------------------------------------------------------
# Scoring, the frontier and the bootstrap: evidence/17's definitions
# ---------------------------------------------------------------------------

def vals(m):
    """Pool one experiment's per-seed rows into (delay ms, polls per run, censored share).

    evidence/17/spec.md "Quantities": delay is the pooled `sum(lat4) / sum(n4)` over the
    runs of a stage, the censored share is `sum(c4) / sum(n4 + c4)`, and polls per run is
    the mean of the windowed `polls_sent`. Pooling over records rather than averaging
    over seeds is what every earlier step's scorer does. Under `w0` every seed has the
    same 240 records, so the two poolings coincide here.
    """
    n4, c4, lat, polls = m.sum(0)
    return lat / n4, polls / len(m), c4 / (n4 + c4)


def interp(pts, x):
    """Log-log piecewise-linear reading of the three-arm frontier at polls rate `x`.

    evidence/17/spec.md "Quantities", three-arm frontier: piecewise linear in log-log
    through the 3.695, 6.0 and 10.043 s arms, read at the rule's own polls per run. The
    code is evidence/19/predict.py's `interp`. None when `x` falls outside the measured
    arms, because extrapolating a frontier that has no arm beyond the point would invent
    the comparison.
    """
    pts = sorted(pts)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + w * (math.log(y1) - math.log(y0)))
    return None


def gain(m, name, col):
    """How much delay the rule saves against the frontier at its own polls rate.

    evidence/17/spec.md: gain = 1 - delay(rule) / line(polls(rule)). Comparing delay
    alone would reward a rule for polling more often, so the rule is charged for its
    polls. None when the three arms are not monotone (more polls, less delay) or the
    rule's polls fall outside them, because a frontier that does not fall cannot say
    what the rule's polls are worth.
    """
    delay, polls, _ = vals(m[name][col])
    arms = [(vals(m[a][col])[1], vals(m[a][col])[0]) for a in ARM_STAGES]
    ordered = sorted(arms)
    if not all(y0 > y1 for (_, y0), (_, y1) in zip(ordered, ordered[1:])):
        return None
    frontier = interp(arms, polls)
    if frontier is None:
        return None
    return 1 - delay / frontier


def experiment(res, source, seeds, rep):
    """One experiment's arrays: {stage: {col: (len(seeds), 4) array}} for stream `rep`.

    Column 0 is the all-record score and column 1 the headline score; both come from the
    same simulations, so the two scorings of one experiment are paired seed by seed.
    """
    return {k: {c: np.array([res[(source, s)][k][rep][c] for s in seeds], float)
                for c in (0, 1)}
            for k in STAGE_NAMES}


def boot_index(n):
    """The one (BOOT_N, n) matrix of resampled seed positions, drawn once.

    spec.md pass line 2: 2000 resamples with `default_rng(20261122)`, the unit the seed
    number, one unit carrying that seed's value in all five stages. Drawing the matrix
    once and reusing it for every experiment is what makes the printed bounds a function
    of BOOT_SEED alone, independent of the order the sections are printed in.
    """
    return np.random.default_rng(BOOT_SEED).integers(0, n, size=(BOOT_N, n))


def boot_gains(m, idx, col):
    """Per-resample gains of the two rules and their difference, and what was dropped.

    A resample whose arms are not monotone, or whose rule polls leave the arms, yields no
    gain. It is counted and left out rather than replaced, because substituting a value
    would hide how often the frontier is unreadable on 40 seeds.
    """
    a, b, d = [], [], []
    dropped = 0
    for row in idx:
        sub = {k: {col: m[k][col][row]} for k in STAGE_NAMES}
        g = {t: gain(sub, t, col) for t in TESTED}
        if any(v is None for v in g.values()):
            dropped += 1
            continue
        a.append(g["s10t37r"])
        b.append(g["s10f"])
        d.append(g["s10f"] - g["s10t37r"])
    return a, b, d, dropped


def boot_ratios(m, idx, col):
    """Per-resample F/T37r ratios minus one, in delay and in polls per run.

    The direct comparison needs no frontier, so no resample is ever dropped here. This is
    the quantity pass line 3 reads.
    """
    dd, pp = [], []
    for row in idx:
        x = vals(m["s10t37r"][col][row])
        y = vals(m["s10f"][col][row])
        dd.append(y[0] / x[0] - 1)
        pp.append(y[1] / x[1] - 1)
    return dd, pp


def ci(v):
    """(lower, upper, half width) of the 95 % percentile interval of `v`, in percent."""
    lo, hi = np.percentile(v, [2.5, 97.5])
    return 100 * lo, 100 * hi, 100 * (hi - lo) / 2


def word(lo, hi):
    """The three words pass line 2 attaches to an interval, by where it sits around zero."""
    if lo > 0:
        return "seen"
    if hi < 0:
        return "reversed"
    return "not-seen"


# ---------------------------------------------------------------------------
# Gate 1: evidence/18's cell table and fpin.txt's pinned member
# ---------------------------------------------------------------------------

def gate_cells(pool):
    """Reproduce bound_cells.csv for this cell's three arms and T37r, and fpin.txt's F.

    If this fails, the engine, the frontier inputs or the replicate stream has moved
    since F was pinned, and every number below would be incomparable with the member the
    testbed is about to run. Nothing is adjusted on a mismatch: an adjusted gate would
    check nothing.
    """
    res = pool.map(B.job_base, [(DONOR, P, s, (WL,)) for s in SEL_SEEDS], chunksize=1)
    per_seed = {(k[1], s): v for _d, _p, s, out in res for k, v in out.items()
                if k[0] == WL}
    records = [out[("records", WL)] for _d, _p, _s, out in res]
    pub, _cols = B.read_cells(os.path.join(EV18, "bound_cells.csv"))
    bad = []
    print(f"Gate 1, evidence/18/bound_cells.csv, cell {DONOR} p {P} {WL}, seeds "
          f"{SEL_SEEDS[0]} to {SEL_SEEDS[-1]}, {B.REPS} replicates "
          "(early-record delay and polls per run):")
    kind_of = {n: k for n, k, _s in B.RULE_SPECS}
    for stage in STAGE_NAMES:
        if stage not in GATE_RULE:
            continue
        name = GATE_RULE[stage]
        row = B.cell_row(WL, DONOR, P, name, kind_of[name],
                         [per_seed[(name, s)] for s in SEL_SEEDS], records)
        want = pub[(WL, DONOR, P, name)]
        ok = (f"{row['early_delay']:.6f}" == want["early_delay"]
              and f"{row['early_polls_sent']:.6f}" == want["early_polls_sent"])
        bad += [] if ok else [name]
        print(f"  {stage:<11} {name:<10} delay {row['early_delay']:.6f} ms "
              f"(csv {want['early_delay']})  polls {row['early_polls_sent']:.6f} "
              f"(csv {want['early_polls_sent']})  {'same' if ok else 'DIFFERENT'}")

    res_f = pool.map(B.job_member, [(DONOR, P, s, WL, F_MEM) for s in SEL_SEEDS],
                     chunksize=1)
    by = {r[0][3]: r[1] for r in res_f}
    v = B.vals_of(B.seed_mat([by[s] for s in SEL_SEEDS], 1, 4))
    got = (f"{v['delay']:.1f}", f"{v['polls_sent']:.2f}")
    ok = got == FPIN_F
    bad += [] if ok else ["the pinned F"]
    print(f"  s10f        pinned F   delay {got[0]} ms (fpin.txt {FPIN_F[0]})  "
          f"polls {got[1]} (fpin.txt {FPIN_F[1]})  {'same' if ok else 'DIFFERENT'}")
    if bad:
        print("\nSTOP: gate 1 failed for " + ", ".join(bad) + ".")
        print("The engine or the replicate stream has moved since F was pinned, so no "
              "prediction is printed.")
        return False
    print("Reproduced to the printed digits: the three arms' and T37r's early-record "
          "delay and polls per run, and the pinned F member's delay and polls to the "
          "digits fpin.txt prints.")
    return True


# ---------------------------------------------------------------------------
# Gate 2 and the three measured costs of the substitution
# ---------------------------------------------------------------------------

def gate_generator():
    """`build_experiment` against the stored `*.experiment.json` of seeds 1 to 20.

    What this states is that the outage events and the record schedule the prediction is
    built from are the ones the testbed laid down for the seeds where both exist. It
    does not state that the instants have the shape `w0` and the outage shape are defined
    to have; `tests/test_evalgen.py` answers that against hand-written expectations.
    """
    bad = []
    for s in SEL_SEEDS:
        exp = experiment_of(s)
        rd = B.run_of(DONOR, P, s)["run_dir"]
        with open(glob.glob(os.path.join(rd, "*.experiment.json"))[0],
                  encoding="utf-8") as fh:
            stored = json.load(fh)
        mine = sorted((e["at"], e["type"], tuple(e["nodes"])) for e in exp["events"]
                      if e["type"] in ("link_down", "link_up"))
        theirs = sorted((e["at"], e["type"], tuple(e["nodes"])) for e in stored["events"]
                        if e["type"] in ("link_down", "link_up"))
        if mine != theirs:
            bad.append((s, "outage events"))
        if exp["schedule"] != stored["schedule"]:
            bad.append((s, "record schedule"))
    print(f"\nGate 2, the generator against the stored descriptions of seeds "
          f"{SEL_SEEDS[0]} to {SEL_SEEDS[-1]} at p {P}: "
          f"{'every outage event and every record schedule equal' if not bad else 'DIFFERENT'}.")
    if bad:
        print("STOP: gate 2 failed for " + ", ".join(f"seed {s} {w}" for s, w in bad)
              + ". No prediction is printed.")
        return False
    print("  Compared as objects, not as files: `generate_run` would write a ReCefore "
          "yaml and a per-node config beside each description, and those configs carry "
          "key material.")
    return True


def substitution_costs():
    """The three gaps between a borrowed exogenous half and a logged one, measured.

    All three are printed and none is corrected for. They are the part of the
    "what this step cannot show" section that has a number behind it.
    """
    worst = 0.0
    for s in SEL_SEEDS:
        rd = B.run_of(DONOR, P, s)["run_dir"]
        with open(os.path.join(rd, "results.json"), encoding="utf-8") as fh:
            for row in json.load(fh):
                if row.get("op_type") == "event":
                    worst = max(worst, abs(float(row["actual_at"])
                                           - float(row["event"]["at"])))
    print("\nWhat the borrowed exogenous half costs, measured on seeds "
          f"{SEL_SEEDS[0]} to {SEL_SEEDS[-1]} at p {P}:")
    print("  realized against scheduled outage instant: at most %.0f ms over every "
          "link event of the 20 runs. The held-out timeline is the scheduled one."
          % (worst * 1000.0))
    off = []
    for s in SEL_SEEDS:
        r = B.run_of(DONOR, P, s)
        off += [r["agent_start"][n] - r["t0"] for n in sorted(r["agent_start"])]
    off = np.array(off, float)
    print("  agent_start minus t0 over the 20 donors and 5 nodes: %.0f .. %.0f ms, "
          "median %.0f. A held-out seed borrows donor seed 1 + (seed - %d) %% %d, so "
          "each donor carries two held-out seeds."
          % (off.min(), off.max(), np.median(off), NEW_SEEDS[0], len(SEL_SEEDS)))
    lag = []
    for s in SEL_SEEDS:
        r = B.run_of(DONOR, P, s)
        exp = experiment_of(s)
        for host in G.RECORD_ORIGIN_HOSTS:
            at = [e["at_s"] for e in exp["schedule"][str(host)]]
            got = sorted(ts for ts, o, _q in r["created"] if o == f"h{host}")
            assert len(at) == len(got), (s, host, len(at), len(got))
            lag += [g - (r["t0"] + a * 1000.0) for g, a in zip(got, at)]
    lag = np.array(lag, float)
    print("  the agent's creation lag, logged stamp minus scheduled instant, over "
          "%d records: median %.0f ms, 5th to 95th percentile %.0f .. %.0f. The replay "
          "creates a record at exactly the scheduled instant, so every delay below is "
          "measured from a start that the testbed reaches a little later."
          % (len(lag), np.median(lag), np.percentile(lag, 5), np.percentile(lag, 95)))


def print_systematic(res):
    """Per stage, what replacing the logged exogenous half by the generated one moves.

    This is the one quantity that says how much of the prediction below rests on the
    substitution rather than on the replay. It is not a gate: it cannot be exact, since
    the generated creation stamps carry no agent lag and the generated timeline is the
    scheduled one. A stage whose two numbers sit close together is a stage where the
    substitution is cheap; one that moves a lot is a warning to carry into pass line 4.
    """
    print("\nThe substitution on the SAME seeds %d to %d: five stages scored three "
          "times, median over the %d streams. The held-out runs carry both replacements "
          "at once, so each column bounds one half of what they cost."
          % (SEL_SEEDS[0], SEL_SEEDS[-1], NREP))
    med = {}
    for src in ("stored", "gen20", "gen20r"):
        for name in STAGE_NAMES:
            v = np.array([vals(experiment(res, src, SEL_SEEDS, rep)[name][1])
                          for rep in range(NREP)])
            med[(src, name)] = (np.median(v[:, 0]) / 1000.0, np.median(v[:, 1]))
    print("  stage       | logged: delay s / polls | generated schedules, own clock | "
          "and a borrowed clock | schedules move | clock moves")
    for name in STAGE_NAMES:
        a, b, c = med[("stored", name)], med[("gen20", name)], med[("gen20r", name)]
        print("  %-11s | %7.3f / %7.2f       | %7.3f / %7.2f                | "
              "%7.3f / %7.2f      | delay %+5.1f %% polls %+5.1f %% | "
              "delay %+5.1f %% polls %+5.1f %%"
              % (name, a[0], a[1], b[0], b[1], c[0], c[1],
                 100 * (b[0] / a[0] - 1), 100 * (b[1] / a[1] - 1),
                 100 * (c[0] / b[0] - 1), 100 * (c[1] / b[1] - 1)))
    print("  \"schedules move\" is the generated timeline and creation stamps against "
          "the logged ones with the clock held fixed; \"clock moves\" is t0 and "
          "agent_start taken from seed 1 + seed mod 20 instead of the run's own, with the "
          "generated schedules held fixed. A held-out seed carries both.")


# ---------------------------------------------------------------------------
# The prediction
# ---------------------------------------------------------------------------

def print_counts(res):
    """The record counts and the headline set of the held-out schedules.

    Under `w0` every generation time falls at or before t0+71 s, so the headline set and
    the all-record set are the same set and evidence/17's pooled delay and evidence/19's
    headline delay are the same number. That is stated as a count rather than assumed,
    because the two scorings are reported side by side below.
    """
    allc = np.array([len(SYN[("gen", s)]["created"]) for s in NEW_SEEDS], float)
    head = np.array([sum(1 for ts, _o, _q in SYN[("gen", s)]["created"]
                         if ts - SYN[("gen", s)]["t0"] <= A.CUT_MS)
                     for s in NEW_SEEDS], float)
    print("\nThe held-out schedules, seeds %d to %d, k %d, both origins:"
          % (NEW_SEEDS[0], NEW_SEEDS[-1], K))
    print("  records per run all %.1f [%d .. %d] | headline (at_s <= %.0f s) %.1f "
          "[%d .. %d]" % (allc.mean(), allc.min(), allc.max(), CUT_S,
                          head.mean(), head.min(), head.max()))
    same = all(np.allclose(experiment(res, "gen", NEW_SEEDS, rep)[n][0],
                           experiment(res, "gen", NEW_SEEDS, rep)[n][1])
               for n in STAGE_NAMES for rep in range(NREP))
    print("  headline score equals all-record score on every stage and every stream: "
          f"{same}. Under w0 the two scorings select the same records, so one delay per "
          "stage serves both.")


def print_stages(res):
    """Each stage's predicted delay and polls per run, with pass line 4's band.

    The median over the 40 streams is the predicted value rather than the mean, because
    it is the middle of what a one-run-per-seed experiment produces and is not moved by
    one stream that drew an unlucky set of jitter phases. The per-stream minimum and
    maximum stand beside it, so a measured value outside the band can be read against
    how much the condition moves on its own.
    """
    print("\nPrediction for seeds %d to %d at p %s, w0, the s6 outage shape: "
          "delay = pooled sum(lat4)/sum(n4) over the 40 runs of a stage, polls per run "
          "over [t0+10 s, t0+130 s]." % (NEW_SEEDS[0], NEW_SEEDS[-1], P))
    print("  stage       | delay s median [band] min..max        | polls median [band] "
          "min..max            | censored share median min..max")
    for name in STAGE_NAMES:
        v = np.array([vals(experiment(res, "gen", NEW_SEEDS, rep)[name][1])
                      for rep in range(NREP)])
        d, pl, c = (np.median(v[:, i]) for i in range(3))
        print("  %-11s | %7.3f [%7.3f..%7.3f] %6.3f..%6.3f | %6.2f [%6.2f..%6.2f] "
              "%6.2f..%6.2f | %5.3f %5.3f..%5.3f"
              % (name, d / 1000.0, d / 1000.0 * (1 - BAND), d / 1000.0 * (1 + BAND),
                 v[:, 0].min() / 1000.0, v[:, 0].max() / 1000.0,
                 pl, pl * (1 - BAND), pl * (1 + BAND), v[:, 1].min(), v[:, 1].max(),
                 c, v[:, 2].min(), v[:, 2].max()))
    print("  the band is the median times 1 -+ %d %%. Pass line 4 reads these ten "
          "values, the delay and the polls of each of the five stages." % round(100 * BAND))
    print("  the three arms' polls values are near-tautological: a fixed arm's poll "
          "count follows from its interval and the window, so only the two rules' polls "
          "and the five delays are numbers the replay could get wrong.")


def print_gains(res, idx):
    """The gain of T37r and of F against the three-arm frontier, and the predicted word.

    Every number is the median over the 40 streams, so what is printed is what a typical
    one-run-per-seed experiment produces rather than what pooling 40 of them would. The
    half width is the quantity spec.md's "what this step cannot show" argues with: an
    effect smaller than it cannot be separated from zero on 40 seeds, whatever its sign.

    Only the headline scoring is printed. `print_counts` has already checked that the
    two scorings select the same records on every stage and every stream, so an
    all-record block would repeat these digits and could be mistaken for a second
    reading of the same hypothesis.
    """
    for col, label in ((1, "headline"),):
        rows = {t: [] for t in TESTED}
        diff = []
        tally = {t: {"seen": 0, "not-seen": 0, "reversed": 0} for t in TESTED}
        tally["diff"] = {"seen": 0, "not-seen": 0, "reversed": 0}
        dropped_total = 0
        for rep in range(NREP):
            m = experiment(res, "gen", NEW_SEEDS, rep)
            a, b, d, dropped = boot_gains(m, idx, col)
            dropped_total += dropped
            for t, v in (("s10t37r", a), ("s10f", b)):
                lo, hi, hw = ci(v)
                rows[t].append((100 * gain(m, t, col), lo, hi, hw))
                tally[t][word(lo, hi)] += 1
            lo, hi, hw = ci(d)
            diff.append((100 * (gain(m, "s10f", col) - gain(m, "s10t37r", col)),
                         lo, hi, hw))
            tally["diff"][word(lo, hi)] += 1
        print("\nGain against the three-arm log-log frontier (%s scoring), median of "
              "the %d streams, %d resamples of the 40 seeds with default_rng(%d):"
              % (label, NREP, BOOT_N, BOOT_SEED))
        for t in TESTED:
            v = np.array(rows[t])
            print("  %-8s gain %+6.2f %% [%+6.2f .. %+6.2f] half width %5.2f pp | "
                  "streams seen %d / not seen %d / reversed %d | predicted word %s"
                  % (t, np.median(v[:, 0]), np.median(v[:, 1]), np.median(v[:, 2]),
                     np.median(v[:, 3]), tally[t]["seen"], tally[t]["not-seen"],
                     tally[t]["reversed"],
                     word(np.median(v[:, 1]), np.median(v[:, 2]))))
        v = np.array(diff)
        print("  F minus T37r %+6.2f pp [%+6.2f .. %+6.2f] half width %5.2f pp | "
              "streams seen %d / not seen %d / reversed %d"
              % (np.median(v[:, 0]), np.median(v[:, 1]), np.median(v[:, 2]),
                 np.median(v[:, 3]), tally["diff"]["seen"], tally["diff"]["not-seen"],
                 tally["diff"]["reversed"]))
        print("  resamples with no readable frontier: %d of %d"
              % (dropped_total, NREP * BOOT_N))
    print("  the all-record scoring gives these same digits, because under w0 it selects "
          "the same records (see the equality checked above). It is not printed again.")


def print_ratios(res, idx):
    """F against T37r directly, which needs no frontier (pass line 3).

    Ratios rather than differences: the testbed will not reproduce the absolute delay
    exactly, and the ratio is the part of the prediction that should survive that. The
    polls ratio is printed once rather than once per scoring, because polls are counted
    over the whole window under both scorings. Only the headline scoring is printed, for
    the reason `print_gains` gives.
    """
    for col, label in ((1, "headline"),):
        dd, pp = [], []
        for rep in range(NREP):
            m = experiment(res, "gen", NEW_SEEDS, rep)
            a, b = boot_ratios(m, idx, col)
            dd.append((100 * (vals(m["s10f"][col])[0]
                              / vals(m["s10t37r"][col])[0] - 1),) + ci(a))
            pp.append((100 * (vals(m["s10f"][col])[1]
                              / vals(m["s10t37r"][col])[1] - 1),) + ci(b))
        rows = [("delay (%s scoring)" % label, np.array(dd))]
        if col == 1:
            rows.append(("polls (whole window)", np.array(pp)))
        for what, v in rows:
            print("  F vs T37r %-26s %+6.2f %% [%+6.2f .. %+6.2f] half width %5.2f pp"
                  % (what, np.median(v[:, 0]), np.median(v[:, 1]), np.median(v[:, 2]),
                     np.median(v[:, 3])))


FOOTER = """
Three sentences that belong beside every number above.
  The pinned F's gain here is a three-arm frontier gain on seeds 121 to 160, which F has
  never seen. It is not fpin.txt's +11.1, which is a seven-arm gain on the 20 seeds that
  chose the member and is in sample there.
  Each of the five stages is simulated on the held-out seeds with the outage timeline and
  the record schedule the generator draws, and with t0 and agent_start borrowed from a
  logged run. The three measured costs of that borrowing are printed above and none of
  them is corrected for.
  The replay has been held against a testbed measurement in this cell for the fixed
  0.5, 1.359, 3.695 and 10.043 s arms and the short and wait policies only
  (evidence/13/validate_cells.csv). The 6.0 s arm, T37r and F have no measurement in
  this cell, which is what pass line 4 of this step is for."""


def main():
    """The two gates, the measured costs, the systematic, then the prediction."""
    global ARM_SET
    ARM_SET = arm_settings()
    # ARM_SET and SYN are filled before the pool exists: the workers inherit them
    # through the fork, and a dict filled after the fork would reach no worker.
    for s in NEW_SEEDS:
        SYN[("gen", s)] = synthetic_run(s, donor_seed)
    for s in SEL_SEEDS:
        SYN[("gen20", s)] = synthetic_run(s, lambda x: x)
        # The next seed's clock, 20 wrapping to 1: a borrowed clock that is as unrelated
        # to the run's schedules as a held-out seed's borrowed clock is to its own.
        SYN[("gen20r", s)] = synthetic_run(s, lambda x: 1 + x % len(SEL_SEEDS))
    with mp.Pool(WORKERS) as pool:
        if not gate_cells(pool):
            return 1
        if not gate_generator():
            return 1
        substitution_costs()
        res = run_all([("stored", s) for s in SEL_SEEDS]
                      + [("gen20", s) for s in SEL_SEEDS]
                      + [("gen20r", s) for s in SEL_SEEDS]
                      + [("gen", s) for s in NEW_SEEDS], pool)
    print_systematic(res)
    print_counts(res)
    print_stages(res)
    idx = boot_index(len(NEW_SEEDS))
    print_gains(res, idx)
    print("\nF against T37r directly, no frontier, same resamples:")
    print_ratios(res, idx)
    print(FOOTER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
