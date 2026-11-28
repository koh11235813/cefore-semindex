#!/usr/bin/env python3
"""The prediction for the seven s12 testbed stages, written before those runs.

    cd <this directory> && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B predict.py MODE [options]

    --dry                 the gates only, on seeds 1 to 100; then the table code run on
                          the scoring's own seeds 81 to 100 as a self-test (not a
                          prediction).  --write-l-input PATH also writes the GPU side's
                          input file for a dry run there (check runs only);
                          --check-l-rows CSV --l-input PATH reads that dry run's rows
                          back through the report's checks.
    --held-out-once       the gates, then seeds 121 to 160: writes predict_runs.json (the
                          GPU side's input) and predict_rows.csv (the six CPU stages, one
                          row per seed and stream) and prints the prediction with L
                          pending.  Refuses if either file exists, so it runs once.
    --report --l-rows F   the gates again, then the full prediction from the two files
                          above and the GPU side's rows F.  Simulates no held-out seed.
    --workers N           processes for the simulations (default 3)

  PYTHONPATH is this worktree's `src`, not /home/lab_shared/wt/semindex-decider/src:
  evidence/20's predict.py (imported through ../22/label.py, which every scored run dict
  comes from) refuses a `semindex` imported from anywhere but five levels above itself.
  Gate 0 below holds the two `src/semindex` trees byte-equal file by file instead.

WHAT IS PREDICTED
  For each of the seven stages (the 3.695, 6.0 and 10.043 s arms, T37r, the pinned F,
  M and L) on seeds 121 to 160, p 0.2, w0, the s6 outage shape: the delay and the
  whole-window polls per run, each with its +-6 % band, and for T37r, F, M and L the gain
  against the three-arm log-log frontier with its 95 % seed-bootstrap interval, half
  width and the seen / not seen / reversed counts over the 40 replicate streams, plus the
  M minus L gain difference.  evidence/20's definitions throughout (`../20/spec.md`
  "Quantities"): delay is the pooled sum(lat4)/sum(n4) over a stage's 40 runs with the
  censored share beside it, the median over 40 streams is the predicted value, 2000
  resamples of one index matrix with the seed number as the unit.  The generator seed is
  new, default_rng(20261128), the date this script was written.

WHICH ENGINE, AND WHY ALL SEVEN STAGES SHARE IT
  All seven stages run through ../22/port.py's engine with the scoring's stream options:
  `pair_lag_seed = pair_jitter_seed = R.rep_seed("pairlag:s6b10p043", 0.2, seed,
  stream)`, `sim_rng` None, no rekey, run dicts from `label.run_dict`'s construction with
  the union choice set, choice jitter 0.1 and `redecide` (../23/eval.py:80 and 573-579,
  ../24/eval24.py `one_run`, printed at ../23/eval.txt:21-23).  The five baselines are
  deciders on that engine (`port.const_decider`, `label.F`, and T37r as `MemberF`), so
  the frontier, F, T37r, M and L are scored on the same draws.  That is NOT the engine
  evidence/20 predicted with: evidence/20 used evidence/18's `bound.sim` with one shared
  tick-lag stream `trickle:s6b10p043` and the arms and members on the rule path.  The
  two are bridged by gate 4 and compared, as a printed check and not a gate, by the
  "against evidence/20" table of the held-out output.

THE GATES, before any held-out number is printed
  0. code: the two `src/semindex` trees byte-equal; the five files the engine copy in
     heldout.py rests on at the scoring's digests.
  1. M's pins exactly as ./mdecider.py checks them at a node's start (`_check_pins` on
     ../24/coef_a.npy), run before anything else from evidence/ is imported, because that
     check refuses any loaded module it does not know; and coef_a.npy's arrays as
     m24.load_coef reads them equal M.from_file's.
  2. the generator: evidence/20's own `gate_generator` (outage events and record schedule
     against the stored *.experiment.json of seeds 1 to 20), and evidence/21's
     `experiment_of`, which the run dicts use, equal to evidence/20's on those seeds.
  3. the run dicts: `heldout.build_run` equal to `label.run_dict` on seeds 1, 2 and 81 to
     100; on 81 to 100 its exogenous half equal to evidence/20's `synthetic_run` with
     evidence/20's donor map (the held-out substitution); the JSON round trip exact,
     types included; and the guard-free engine leaving port.Engine's attributes.
  4. the bridge, seeds 1 to 20: the five baselines as deciders on this engine but on
     evidence/18's shared stream reproduce evidence/20's gate 1 to its printed digits
     (bound_cells.csv's early delay and polls for the arms and T37r, fpin.txt's for F).
     This is the only exact check T37r has: no scoring ever ran T37r on the pair streams.
  5. the scoring's rows, through the very path the held-out stages take (JSON-decoded run
     dict, guard-free engine, `heldout.one_run`): seeds 1, 2 x streams 0, 1 against
     ../23/eval_dry.csv (arms, F) and ../24/eval24_dry.csv (M); seeds 81 to 100 x 40
     streams against ../23/eval.csv and ../24/eval24.csv, every simulated column as the
     string the scoring wrote.  The second half reads seeds 81 to 100, which are above
     the "seeds 1 to 20" the gates were asked for; it is the only exact check of the
     generated-schedule, borrowed-clock run dict the held-out seeds use, because seeds
     1 to 20 are built from logged runs.  `MemberF(F)` is held to the F rows as well,
     which is what makes T37r's class F's arithmetic.
  A failed gate prints the difference and no prediction.

WHAT THIS CANNOT SHOW
  Every number is simulated.  The held-out run dicts carry the scheduled outage instants
  and record stamps and a borrowed clock, whose costs evidence/20 measured and did not
  correct for (../20/predict.txt).  L's rows come from the GPU machine (lpredict.py); this
  script holds them to the scoring's L rows on 24 runs and to the input file's digest, and
  cannot otherwise see how they were computed.
"""
import glob
import json
import math
import multiprocessing as mp
import os
import sys

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
EVID = os.path.dirname(HERE)
WT = os.path.normpath(os.path.join(HERE, *[os.pardir] * 5))
DECIDER_SRC = "/home/lab_shared/wt/semindex-decider/src"
COEF = os.path.join(EVID, "24", "coef_a.npy")
COEF_SHA = "bbf5a5c7abf1c6fa476c728b3855ecdcab601c580452a2101cf4203e560b8ebf"
L_MODEL_SHA = "6a39ca005019699bef789b6811abd006da7f9445708894ea444d9e8690441cc1"
RUNS_PATH = os.path.join(HERE, "predict_runs.json")
ROWS_PATH = os.path.join(HERE, "predict_rows.csv")

DRY_SEEDS, DRY_REPS = (1, 2), (0, 1)          # ../23/eval_dry.csv, ../24/eval24_dry.csv
SEL_SEEDS = tuple(range(81, 101))             # ../23/eval.csv, ../24/eval24.csv
BRIDGE_SEEDS = tuple(range(1, 21))            # evidence/20's gate 1
BOOT_N = 2000
BOOT_SEED = 20261128                          # the date this script was written
BAND = 0.06
DONOR, WL = "s6b10p043", "W0"

# One entry per testbed stage: (name, kind, what the decider is, row name in the scoring).
STAGES = (("s12b3p695", "arm", 3.695, "const3.695"),
          ("s12b6p0", "arm", 6.0, "const6.000"),
          ("s12b10p043", "arm", 10.043, "const10.043"),
          ("s12t37r", "member", "T37r", None),
          ("s12f", "F", None, "F"),
          ("s12m", "M", None, "a"))
L_STAGE = "s12l"
ARM_STAGES = ("s12b3p695", "s12b6p0", "s12b10p043")
RULE_STAGES = ("s12t37r", "s12f", "s12m", L_STAGE)
# The bridge's rows of bound_cells.csv, as evidence/20's GATE_RULE.
BRIDGE_ROW = {"s12b3p695": "fix3.695", "s12b6p0": "fix6.0", "s12b10p043": "fix10.043",
              "s12t37r": "T37r"}
CPU_CODE = ("25/predict.py", "25/heldout.py", "24/eval24.py", "24/m24.py",
            "24/features.py", "24/charges.py", "22/port.py", "22/label.py", "22/score.py",
            "13/replay.py")

WORKERS = 3
RUNS = {}          # seed -> run dict; filled before every pool so the fork carries it
COEFS = None       # (w_blocks, mu, sigma) of coef_a.npy, likewise
HO = E24 = P20 = None


def say(text=""):
    """One line of the report, flushed so a redirected run shows progress."""
    print(text, flush=True)


def stop(text):
    """A failed gate: name the difference, print no prediction, exit 1."""
    say("STOP: " + text)
    say("No prediction is printed.")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Gates 0 and 1, which must run before evidence/22 is imported
# ---------------------------------------------------------------------------
def gate_src():
    """The worktree's src/semindex and the merged src/semindex, byte for byte."""
    mine = os.path.join(WT, "src", "semindex")
    theirs = os.path.join(DECIDER_SRC, "semindex")
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(mine, "*.py")))
    other = sorted(os.path.basename(p) for p in glob.glob(os.path.join(theirs, "*.py")))
    if names != other:
        stop("gate 0: the two src/semindex trees list different files: %s"
             % sorted(set(names) ^ set(other)))
    import hashlib

    def sha(p):
        with open(p, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    bad = [n for n in names if sha(os.path.join(mine, n)) != sha(os.path.join(theirs, n))]
    if bad:
        stop("gate 0: these differ between %s and %s: %s" % (mine, theirs, bad))
    say("Gate 0, code: %d files of %s byte-equal to %s" % (len(names), mine, theirs))


def gate_m_pins():
    """mdecider's own pin check, in a process where nothing else of evidence/ is loaded."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import mdecider as MD
    try:
        MD._check_pins(COEF)
    except RuntimeError as exc:
        stop("gate 1: %s" % exc)
    say("Gate 1, M's pins: mdecider._check_pins(%s) passed, %d pinned files including "
        "coef_a.npy %s" % (os.path.relpath(COEF, EVID), len(MD.PINS), MD.PINS["24/coef_a.npy"]))
    if MD.PINS["24/coef_a.npy"] != COEF_SHA:
        stop("gate 1: mdecider pins coef_a.npy at %s, not %s"
             % (MD.PINS["24/coef_a.npy"], COEF_SHA))
    return MD


# ---------------------------------------------------------------------------
# Deciders and jobs (run in the pool; read RUNS and COEFS)
# ---------------------------------------------------------------------------
def t37r_choices():
    """The union choice set plus 7.39 s, which T37r reaches and F does not."""
    return sorted(set(HO.LBL.CHOICES) | set(HO.closure(HO.T37R_MEM)))


def make(kind, what):
    """The decider object of one stage, fresh per simulation."""
    if kind == "arm":
        return HO.PORT.const_decider(what)
    if kind == "F":
        return HO.LBL.F()
    if kind == "member":
        return HO.MemberF(HO.T37R_MEM if what == "T37r" else HO.PORT.F_MEM)
    if kind == "M":
        return E24.m24.M(*COEFS)
    raise ValueError(kind)


def stage_run(run, kind, what):
    """The run dict of one stage: the scoring's, with T37r's wider choice set."""
    if kind == "member" and what == "T37r":
        return dict(run, choices=t37r_choices())
    E24.check_choices(run, "stage run dict")
    return run


def job(arg):
    """(name, kind, what, seed, reps) -> the rows of one decider on one seed."""
    name, kind, what, seed, reps = arg
    run = stage_run(RUNS[seed], kind, what)
    rows = []
    for rep in reps:
        inner = make(kind, what)
        where = "%s seed %d stream %d" % (name, seed, rep)
        wrapper = E24.FirstCallEmpty(inner, where, E24.keeps_memory(inner))
        row = HO.one_run(name, run, rep, wrapper, inner, 1)
        row["model_sha"] = COEF_SHA if kind == "M" else "none"
        rows.append(row)
    return rows


def bridge_job(arg):
    """One baseline on one seed 1-20 run, evidence/18's shared stream, B.REPS replicates."""
    name, kind, what, seed = arg
    B = HO.B
    donor = B.run_of(DONOR, HO.MAIN_P, seed)
    run = dict(donor, created=B.created_of(WL, HO.MAIN_P, seed, donor), policy=None,
               rule=None, choices=list(HO.LBL.CHOICES), choice_jitter=HO.LBL.JF,
               redecide=True, decider=None)
    run = stage_run(run, kind, what)
    reps = []
    for rep in range(B.REPS):
        engine = HO.HeldOutEngine(dict(run, decider=make(kind, what)), "B", None,
                                  B.rep_rng_of(DONOR, HO.MAIN_P, seed, rep))
        ev = engine.go()
        reps.append((HO.SCORE.early_four(HO.SCORE.terms(ev, run["t0"])),))
    return name, seed, reps


def pool_map(fn, args):
    """Fork after RUNS and COEFS are filled, so every worker inherits them."""
    with mp.get_context("fork").Pool(WORKERS) as pool:
        return pool.map(fn, args, chunksize=1)


def simulate(stages, seeds, reps):
    """{name: [rows]} for the given stages, seeds and streams, through the pool."""
    args = [(name, kind, what, s, tuple(reps)) for name, kind, what in stages for s in seeds]
    out = {}
    for rows in pool_map(job, args):
        for row in rows:
            out.setdefault(row["decider"], []).append(row)
    return out


# ---------------------------------------------------------------------------
# Gates 2 to 5
# ---------------------------------------------------------------------------
def gate_generator():
    if not P20.gate_generator():
        raise SystemExit(1)
    PR = HO.LBL.PR
    bad = []
    for s in BRIDGE_SEEDS:
        a, b = PR.experiment_of(HO.MAIN_P, s), P20.experiment_of(s)
        if a["events"] != b["events"] or a["schedule"] != b["schedule"]:
            bad.append(s)
    if bad:
        stop("gate 2: evidence/21's experiment_of differs from evidence/20's on seeds %s"
             % bad)
    say("  and evidence/21's experiment_of (stage s11a, which the run dicts use) gives the "
        "same events and schedule as evidence/20's on all %d seeds." % len(BRIDGE_SEEDS))


EXO = ("t0", "agent_start", "timeline", "created", "neighbors", "stop_at_ms")


def check_substitution(seeds):
    """build_run's exogenous half against evidence/20's synthetic_run with its donor map."""
    bad = []
    for s in seeds:
        mine = HO.build_run(s)
        ref = P20.synthetic_run(s, P20.donor_seed)
        for k in EXO:
            diff = HO.same(mine[k], ref[k], "seed %d %s" % (s, k))
            if diff:
                bad.append(diff)
    return bad


def gate_runs(seeds_label):
    """Gate 3 on the seeds the gates read; returns the JSON-round-tripped run dicts."""
    seeds = list(DRY_SEEDS) + list(SEL_SEEDS)
    out = {}
    for s in seeds:
        mine = HO.build_run(s)
        diff = HO.same(mine, HO.LBL.run_dict(HO.MAIN_P, s), "seed %d" % s)
        if diff:
            stop("gate 3: build_run differs from label.run_dict: %s" % diff)
        back = HO.decode_run(json.loads(json.dumps(HO.encode_run(mine), sort_keys=True)))
        diff = HO.same(back, mine, "seed %d round trip" % s)
        if diff:
            stop("gate 3: the JSON round trip moved %s" % diff)
        out[s] = back
    bad = check_substitution(SEL_SEEDS)
    if bad:
        stop("gate 3: the held-out substitution differs from evidence/20's: %s" % bad[:3])
    ok, diff = HO.same_shape(out[SEL_SEEDS[0]])
    if not ok:
        stop("gate 3: HeldOutEngine and port.Engine leave different attributes: %s" % diff)
    say("\nGate 3, the run dicts (%s):" % seeds_label)
    say("  heldout.build_run equals label.run_dict, types included, on seeds 1, 2 and "
        "81 to 100 (%d seeds)." % len(seeds))
    say("  on 81 to 100 its t0, agent_start, timeline, created, neighbors and stop_at_ms "
        "equal evidence/20's synthetic_run with donor 1 + (seed - 121) % 20.")
    say("  the JSON round trip returns every one of the %d dicts exactly, and "
        "HeldOutEngine leaves port.Engine's attribute set." % len(seeds))
    return out


def gate_bridge():
    """Gate 4: evidence/20's gate 1 numbers, reproduced by the port's decider path."""
    B = HO.B
    pub, _cols = B.read_cells(os.path.join(EVID, "18", "bound_cells.csv"))
    args = [(name, kind, what, s) for name, kind, what, _r in STAGES if kind != "M"
            for s in BRIDGE_SEEDS]
    per = {}
    for name, seed, reps in pool_map(bridge_job, args):
        per[(name, seed)] = reps
    say("\nGate 4, the bridge to evidence/20: the five baselines as deciders on the port, "
        "on evidence/18's shared stream trickle:%s, seeds 1 to 20 x %d replicates "
        "(early-record delay and polls per run):" % (DONOR, B.REPS))
    bad = []
    for name, kind, what, _r in STAGES:
        if kind == "M":
            continue
        v = B.vals_of(B.seed_mat([per[(name, s)] for s in BRIDGE_SEEDS], 0, 4))
        if name in BRIDGE_ROW:
            want = pub[(WL, DONOR, HO.MAIN_P, BRIDGE_ROW[name])]
            got = ("%.6f" % v["delay"], "%.6f" % v["polls_sent"])
            ref = (want["early_delay"], want["early_polls_sent"])
            src = "bound_cells.csv " + BRIDGE_ROW[name]
        else:
            got = ("%.1f" % v["delay"], "%.2f" % v["polls_sent"])
            ref = P20.FPIN_F
            src = "fpin.txt pinned F"
        ok = got == tuple(ref)
        bad += [] if ok else [name]
        say("  %-11s delay %s ms (%s %s)  polls %s (%s)  %s"
            % (name, got[0], src, ref[0], got[1], ref[1], "same" if ok else "DIFFERENT"))
    if bad:
        stop("gate 4 failed for %s" % ", ".join(bad))
    say("  All five equal evidence/20's gate 1 to the printed digits: the decider path of "
        "the port is the rule path evidence/20 predicted with, for these five, on that "
        "stream.  The prediction below uses the scoring's pair streams instead.")


def compare(mine, ref, ref_name, where):
    """Every SIM_COLUMNS string of `mine` against the reference row of the same key."""
    by = {(r["decider"], int(r["seed"]), int(r["stream"])): r for r in ref}
    for row in mine:
        key = (ref_name, row["seed"], row["stream"])
        if key not in by:
            return "%s: %r missing from the reference" % (where, key)
        got = HO.sim_strings(row)
        for c in HO.SIM_COLUMNS:
            if got[c] != by[key][c]:
                return ("%s: %r column %s: %r against the scoring's %r"
                        % (where, key, c, got[c], by[key][c]))
    return None


def gate_rows():
    """Gate 5; returns the rows of seeds 81 to 100 for the self-test."""
    E23 = os.path.join(EVID, "23")
    refs = {"dry": (HO.read_csv(os.path.join(E23, "eval_dry.csv")),
                    HO.read_csv(os.path.join(EVID, "24", "eval24_dry.csv"))),
            "sel": (HO.read_csv(os.path.join(E23, "eval.csv")),
                    HO.read_csv(os.path.join(EVID, "24", "eval24.csv")))}
    gate_stages = [(n, k, w) for n, k, w, _r in STAGES if k != "member"]
    gate_stages.append(("s12f-member", "member", "F"))
    ref_name = {n: r for n, _k, _w, r in STAGES}
    ref_name["s12f-member"] = "F"
    say("\nGate 5, the scoring's own rows through the held-out path (JSON-decoded run "
        "dict, HeldOutEngine, heldout.one_run), every simulated column as the string "
        "the scoring wrote:")
    out = {}
    for label, seeds, reps in (("dry", DRY_SEEDS, DRY_REPS),
                               ("sel", SEL_SEEDS, range(HO.NREP))):
        rows = simulate(gate_stages, seeds, reps)
        for name, _k, _w in gate_stages:
            src = refs[label][1] if name == "s12m" else refs[label][0]
            diff = compare(rows[name], src, ref_name[name], name)
            if diff:
                stop("gate 5: " + diff)
            say("  %-3s seeds %3d to %3d x %2d streams  %-11s %4d rows equal to %s %s"
                % (label, seeds[0], seeds[-1], len(reps), name, len(rows[name]),
                   ("../24/eval24" if name == "s12m" else "../23/eval")
                   + ("_dry.csv" if label == "dry" else ".csv"), ref_name[name]))
        out[label] = rows
    printed = {}
    with open(os.path.join(EVID, "24", "eval24.txt"), encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) > 2 and parts[1].startswith("runs=") and "mean_score=" in line:
                printed[parts[0]] = line.split("mean_score=")[1].split()[0]
    say("  mean score over seeds 81 to 100 x 40 streams, against eval24.txt's printed line:")
    for name, _k, _w in gate_stages:
        v = sum(r["score_ms"] for r in out["sel"][name]) / len(out["sel"][name])
        say("    %-11s %9.3f  (eval24.txt %s %s)" % (name, v, ref_name[name],
                                                   printed[ref_name[name]]))
        if "%.3f" % v != printed[ref_name[name]]:
            stop("gate 5: mean score of %s differs from eval24.txt" % name)
    return out["sel"]


def mutations(run81):
    """Each guard and comparison shown refusing a mutated input; returns the miss count."""
    misses = 0

    def shows(label, fn):
        nonlocal misses
        try:
            fn()
        except (SystemExit, FileExistsError, AssertionError) as exc:
            say("  %-40s FIRES: %s" % (label, str(exc).splitlines()[0][:110]))
            return
        misses += 1
        say("  %-40s DID NOT FIRE" % label)

    def moved_row():
        rows = job(("s12b3p695", "arm", 3.695, SEL_SEEDS[0], (0,)))
        rows[0]["polls"] += 1
        diff = compare(rows, HO.read_csv(os.path.join(EVID, "23", "eval.csv")),
                       "const3.695", "mutated")
        assert diff is None, diff

    def tuple_lost():
        enc = json.loads(json.dumps(HO.encode_run(run81)))
        back = HO.decode_run(enc)
        back["created"] = [list(c) for c in back["created"]]
        diff = HO.same(back, run81)
        assert diff is None, diff

    say("\nEvery guard and comparison above, shown refusing a mutated input:")
    shows("port.Engine on a seed-121 dict", lambda: HO.PORT.Engine(
        dict(run81, seed=121), "B", None, None))
    if not HO.ARMED["held_out"]:
        shows("HeldOutEngine on 121, not armed", lambda: HO.HeldOutEngine(
            dict(run81, seed=121), "B", None, None))
        shows("build_run(121), not armed", lambda: HO.build_run(121))
    shows("HeldOutEngine on 101", lambda: HO.HeldOutEngine(
        dict(run81, seed=101), "B", None, None))
    shows("gate 5 compare, one poll moved", moved_row)
    shows("gate 3 round trip, a tuple made a list", tuple_lost)
    shows("an output file that exists", lambda: open(os.path.abspath(__file__), "x"))
    return misses


# ---------------------------------------------------------------------------
# The quantities: evidence/20's definitions
# ---------------------------------------------------------------------------
def matrices(rows, seeds):
    """{stream: (len(seeds), 4) array of (n4, c4, lat_ms, polls)} in seed order."""
    import numpy as np
    by = {(r["seed"], r["stream"]): r for r in rows}
    reps = sorted({r["stream"] for r in rows})
    return {rep: np.array([[float(by[(s, rep)][c]) for c in ("n4", "c4", "lat_ms", "polls")]
                           for s in seeds]) for rep in reps}


def gain(mats, rule, rows_idx=None):
    """1 - delay(rule) / frontier(polls(rule)), or None (evidence/20's `gain`)."""
    def v(name):
        m = mats[name] if rows_idx is None else mats[name][rows_idx]
        return P20.vals(m)
    arms = [(v(a)[1], v(a)[0]) for a in ARM_STAGES]
    ordered = sorted(arms)
    if not all(y0 > y1 for (_, y0), (_, y1) in zip(ordered, ordered[1:])):
        return None
    delay, polls, _c = v(rule)
    frontier = P20.interp(arms, polls)
    return None if frontier is None else 1 - delay / frontier


def print_stages(res, stages, nrep):
    import numpy as np
    say("  stage       | delay s median [band] min..max          | polls median [band] "
        "min..max              | censored share median min..max")
    out = {}
    for name in stages:
        v = np.array([P20.vals(res[name][rep]) for rep in range(nrep)])
        d, pl, c = (np.median(v[:, i]) for i in range(3))
        out[name] = (d / 1000.0, pl)
        say("  %-11s | %7.3f [%7.3f..%7.3f] %6.3f..%6.3f | %6.2f [%6.2f..%6.2f] "
            "%6.2f..%6.2f | %5.3f %5.3f..%5.3f"
            % (name, d / 1000.0, d / 1000.0 * (1 - BAND), d / 1000.0 * (1 + BAND),
               v[:, 0].min() / 1000.0, v[:, 0].max() / 1000.0, pl, pl * (1 - BAND),
               pl * (1 + BAND), v[:, 1].min(), v[:, 1].max(), c, v[:, 2].min(),
               v[:, 2].max()))
    say("  the band is the median times 1 -+ %d %%." % round(100 * BAND))
    return out


def print_gains(res, rules, nseeds, nrep):
    """Per rule and for M minus L: medians over streams, tallies, dropped resamples.

    2026-11-28, a deviation from evidence/20: a resample is dropped per quantity (the
    arms not monotone, or that rule's polls outside them), not jointly over every rule,
    so M's printed interval is the same whether L's rows are present or pending.
    evidence/20 dropped 0 of 80000, where the two forms are one.
    """
    import numpy as np
    idx = np.random.default_rng(BOOT_SEED).integers(0, nseeds, size=(BOOT_N, nseeds))
    quantities = [(r, (r,)) for r in rules]
    if "s12m" in rules and L_STAGE in rules:
        quantities.append(("M minus L", ("s12m", L_STAGE)))
    rows = {q: [] for q, _ in quantities}
    tally = {q: {"seen": 0, "not-seen": 0, "reversed": 0, "no line": 0} for q, _ in quantities}
    dropped = {q: 0 for q, _ in quantities}
    for rep in range(nrep):
        mats = {k: res[k][rep] for k in res}
        for q, parts in quantities:
            point = [gain(mats, p) for p in parts]
            boot = []
            for row in idx:
                g = [gain(mats, p, row) for p in parts]
                if any(x is None for x in g):
                    dropped[q] += 1
                    continue
                boot.append(g[0] if len(g) == 1 else g[0] - g[1])
            if any(x is None for x in point) or not boot:
                tally[q]["no line"] += 1
                continue
            lo, hi, hw = P20.ci(boot)
            pv = point[0] if len(point) == 1 else point[0] - point[1]
            rows[q].append((100 * pv, lo, hi, hw))
            tally[q][P20.word(lo, hi)] += 1
    say("  %-10s gain %%   [95 %% interval]     half width | streams seen / not seen / "
        "reversed / no line | predicted word (of N streams) | dropped resamples" % "")
    for q, _parts in quantities:
        v = np.array(rows[q])
        t = tally[q]
        if not len(v):
            say("  %-10s no readable gain on any stream" % q)
            continue
        unit = "pp" if q == "M minus L" else "% "
        # 2026-11-28 fix: the predicted word used to print with no count of how many
        # streams it came from. len(v) is exactly that count: rows[q] gets one entry per
        # stream except the ones tallied "no line" above, so it is already "streams minus
        # no line" and needs no separate subtraction here.
        say("  %-10s %+6.2f %s [%+6.2f .. %+6.2f] %5.2f pp | %2d / %2d / %2d / %2d | %-9s "
            "(%2d streams) | %d of %d"
            % (q, np.median(v[:, 0]), unit, np.median(v[:, 1]),
               np.median(v[:, 2]), np.median(v[:, 3]), t["seen"], t["not-seen"],
               t["reversed"], t["no line"],
               P20.word(np.median(v[:, 1]), np.median(v[:, 2])), len(v),
               dropped[q], nrep * BOOT_N))
    say("  each column is the median over the %d streams taken separately, so a printed "
        "half width is not half the difference of the printed ends (evidence/20's own "
        "convention)." % nrep)


def tables(rows, seeds, title, l_note):
    """The stage table and the gain table of one set of rows."""
    nrep = len({r["stream"] for r in next(iter(rows.values()))})
    res = {name: matrices(rows[name], seeds) for name in rows}
    say("\n" + title)
    names = [n for n, *_ in STAGES] + ([L_STAGE] if L_STAGE in rows else [])
    med = print_stages(res, names, nrep)
    if L_STAGE not in rows:
        say("  %-11s | %s" % (L_STAGE, l_note))
    say("\nGain against the three-arm log-log frontier, %d resamples of the %d seeds with "
        "default_rng(%d), seed number as the unit:" % (BOOT_N, len(seeds), BOOT_SEED))
    print_gains(res, [r for r in RULE_STAGES if r in rows], len(seeds), nrep)
    if L_STAGE not in rows:
        say("  %-10s %s" % (L_STAGE, l_note))
    return med


def against_ev20(med):
    """The five baselines beside evidence/20's printed medians: a printed check only."""
    ref = {}
    with open(os.path.join(EVID, "20", "predict.txt"), encoding="utf-8") as fh:
        inside = False
        for line in fh:
            if line.startswith("Prediction for seeds"):
                inside = True
                continue
            if inside and line.startswith("  s10"):
                parts = line.split("|")
                ref[parts[0].strip()] = (float(parts[1].split()[0]), float(parts[2].split()[0]))
            elif inside and not line.startswith("  "):
                break
    say("\nThe five baselines against evidence/20's prediction for the same seeds (its "
        "engine: evidence/18's bound.sim on the shared stream; printed, not a gate):")
    for name, _k, _w, _r in STAGES[:5]:
        old = ref[name.replace("s12", "s10")]
        say("  %-11s delay %7.3f s (ev20 %7.3f, %+5.1f %%)  polls %6.2f (ev20 %6.2f, %+5.1f %%)"
            % (name, med[name][0], old[0], 100 * (med[name][0] / old[0] - 1), med[name][1],
               old[1], 100 * (med[name][1] / old[1] - 1)))


# ---------------------------------------------------------------------------
# The modes
# ---------------------------------------------------------------------------
def l_check_list():
    """The 24 L runs the GPU side must reproduce first, with the scoring's own strings."""
    out = []
    for path, seeds, reps in ((os.path.join(EVID, "23", "eval_dry.csv"), DRY_SEEDS, DRY_REPS),
                              (os.path.join(EVID, "23", "eval.csv"), SEL_SEEDS, (0,))):
        by = {(int(r["seed"]), int(r["stream"])): r for r in HO.read_csv(path)
              if r["decider"] == "b"}
        for s in seeds:
            for rep in reps:
                r = by[(s, rep)]
                if r["model_sha"] != L_MODEL_SHA:
                    stop("the scoring's b row %r carries model %s" % ((s, rep), r["model_sha"]))
                out.append({"seed": s, "stream": rep,
                            "expect": {c: r[c] for c in HO.SIM_COLUMNS}})
    return out


def payload(runs, held_out):
    """The GPU side's input: run dicts, the L check runs, the held-out (seed, stream)."""
    return {"format": "ev25-predict-runs-1", "made_by": HO.sha_of(os.path.abspath(__file__)),
            "runs": {str(s): HO.encode_run(r) for s, r in sorted(runs.items())},
            "check": l_check_list(), "l_stage": L_STAGE,
            "held_out": [[s, rep] for s in held_out for rep in range(HO.NREP)]}


def read_l_rows(path, runs_sha, held_seeds):
    """L's rows from the GPU side, held to everything this side can check."""
    rows = HO.read_csv(path)
    want_code = HO.combined(HO.L_CODE)
    for r in rows:
        if r["input_sha"] != runs_sha:
            stop("L rows: input_sha %s, the input file is %s" % (r["input_sha"], runs_sha))
        if r["model_sha"] != L_MODEL_SHA:
            stop("L rows: model_sha %s, not candidate b's %s" % (r["model_sha"], L_MODEL_SHA))
        if r["script_sha"] != want_code:
            stop("L rows: script_sha %s, the local L code digests to %s"
                 % (r["script_sha"], want_code))
    check = [r for r in rows if r["decider"] == "b"]
    held = [r for r in rows if r["decider"] == L_STAGE]
    expect = {(c["seed"], c["stream"]): c["expect"] for c in l_check_list()}
    got = {(int(r["seed"]), int(r["stream"])): r for r in check}
    if sorted(got) != sorted(expect):
        stop("L rows: the check runs are %s, not %s" % (sorted(got), sorted(expect)))
    for key, want in expect.items():
        for c in HO.SIM_COLUMNS:
            if got[key][c] != want[c]:
                stop("L rows: check run %r column %s is %r, the scoring wrote %r"
                     % (key, c, got[key][c], want[c]))
    keys = sorted((int(r["seed"]), int(r["stream"])) for r in held)
    want_keys = sorted((s, rep) for s in held_seeds for rep in range(HO.NREP))
    if keys != want_keys:
        stop("L rows: %d held-out rows, %d expected, or keys differ" % (len(keys), len(want_keys)))
    say("L rows %s: %d check runs equal to the scoring's b rows in every simulated column, "
        "%d held-out rows complete; input, model and code digests as expected"
        % (path, len(check), len(held)))
    out = []
    for r in held:
        out.append(dict(r, seed=int(r["seed"]), stream=int(r["stream"])))
    return out


def rows_by_stage(rows):
    """{stage name: [rows]} of a flat row list."""
    out = {}
    for r in rows:
        out.setdefault(r["decider"], []).append(r)
    return out


def run_gates():
    """Gates 2 to 4 in order; returns gate 3's round-tripped run dicts."""
    gate_generator()
    runs = gate_runs("seeds the gates read")
    gate_bridge()
    return runs


def main(argv):
    """Gates first in every mode, then the mode's own part."""
    global HO, E24, P20, RUNS, COEFS, WORKERS
    modes = [m for m in ("--dry", "--held-out-once", "--report") if m in argv]
    if len(modes) != 1:
        raise SystemExit("give exactly one of --dry, --held-out-once, --report")
    mode = modes[0]
    if "--workers" in argv:
        WORKERS = int(argv[argv.index("--workers") + 1])
    if mode == "--held-out-once":
        for p in (RUNS_PATH, ROWS_PATH):
            if os.path.exists(p):
                raise SystemExit("refused: %s exists; the held-out prediction runs once" % p)
    say("predict.py %s, workers %d" % (mode, WORKERS))
    gate_src()
    gate_m_pins()
    import heldout
    import importlib
    HO = heldout
    sys.path.insert(0, os.path.join(EVID, "24"))
    E24 = importlib.import_module("eval24")
    P20 = HO.LBL.PR.P
    if not os.path.realpath(P20.G.__file__).startswith(os.path.join(WT, "src") + os.sep):
        stop("gate 0: semindex imported from %s" % P20.G.__file__)
    say("  semindex.evalgen imported from %s; %d engine files at the scoring's digests"
        % (P20.G.__file__, HO.check_pins()))
    W24, MU, SG = E24.m24.load_coef(COEF)
    M2 = E24.m24.M.from_file(COEF)
    import numpy as np
    if not (np.array_equal(W24, M2.w_blocks) and np.array_equal(MU, M2.mu)
            and np.array_equal(SG, M2.sigma)):
        stop("gate 1: m24.load_coef and M.from_file read coef_a.npy differently")
    if HO.sha_of(COEF) != COEF_SHA:
        stop("gate 1: coef_a.npy sha256 %s" % HO.sha_of(COEF))
    COEFS = (W24, MU, SG)
    say("  m24.load_coef (the scoring's) and M.from_file (the node's) read the same arrays.")

    RUNS = run_gates()
    sel_rows = gate_rows()
    misses = mutations(RUNS[SEL_SEEDS[0]])
    if misses:
        stop("%d mutation(s) did not fire" % misses)
    say("All gates passed.")

    if mode == "--dry":
        if "--write-l-input" in argv:
            path = argv[argv.index("--write-l-input") + 1]
            sha = HO.write_runs(path, payload(RUNS, ()))
            say("\nwrote the GPU side's dry input %s, sha256 %s (check runs only)" % (path, sha))
        if "--check-l-rows" in argv:
            # 2026-11-28: the GPU side's dry rows read through the report's own reader, so
            # the parsing and the four digest checks meet real GPU output before the
            # once-only steps run.
            read_l_rows(argv[argv.index("--check-l-rows") + 1],
                        HO.sha_of(argv[argv.index("--l-input") + 1]), ())
        RUNS_T = simulate([("s12t37r", "member", "T37r")], SEL_SEEDS, range(HO.NREP))
        rows = rows_by_stage([r for v in sel_rows.values() for r in v] + RUNS_T["s12t37r"])
        rows.pop("s12f-member")
        lsel = [dict(r, seed=int(r["seed"]), stream=int(r["stream"]))
                for r in HO.read_csv(os.path.join(EVID, "23", "eval.csv")) if r["decider"] == "b"]
        rows[L_STAGE] = [dict(r, decider=L_STAGE) for r in lsel]
        tables(rows, SEL_SEEDS, "SELF-TEST of the table code on seeds 81 to 100 (the "
               "scoring's selection seeds, 20 seeds x 40 streams; L from ../23/eval.csv's "
               "b rows).  NOT A PREDICTION: these seeds chose the candidates.", "")
        return 0

    if mode == "--held-out-once":
        HO.arm_held_out()
        bad = check_substitution(HO.HELD_OUT)
        if bad:
            stop("the held-out substitution differs from evidence/20's: %s" % bad[:3])
        held = {s: HO.build_run(s) for s in HO.HELD_OUT}
        runs_sha = HO.write_runs(RUNS_PATH, payload({**RUNS, **held}, HO.HELD_OUT))
        _p, decoded = HO.read_runs(RUNS_PATH)
        for s, r in held.items():
            diff = HO.same(decoded[s], r, "seed %d round trip" % s)
            if diff:
                stop("the written input file moved %s" % diff)
        RUNS = decoded
        say("\nwrote %s, sha256 %s: %d run dicts (%d held-out), %d L check runs, %d L "
            "held-out runs; each held-out dict reads back exactly"
            % (RUNS_PATH, runs_sha, len(decoded), len(held), len(_p["check"]),
               len(_p["held_out"])))
        say("held-out exogenous half equal to evidence/20's synthetic_run on all %d seeds"
            % len(held))
        rows = simulate([(n, k, w) for n, k, w, _r in STAGES], HO.HELD_OUT, range(HO.NREP))
        flat = [r for n, *_ in STAGES for r in sorted(rows[n], key=lambda r: (r["seed"], r["stream"]))]
        HO.write_rows(ROWS_PATH, flat, HO.combined(CPU_CODE), runs_sha)
        say("wrote %s, %d rows" % (ROWS_PATH, len(flat)))
        med = tables(rows, HO.HELD_OUT, "PREDICTION for seeds 121 to 160, p 0.2, w0, the s6 "
                     "outage shape, 40 streams of 40 single runs:",
                     "pending: lpredict.py on the GPU machine, then predict.py --report")
        against_ev20(med)
        return 0

    path = argv[argv.index("--l-rows") + 1]
    runs_sha = HO.sha_of(RUNS_PATH)
    cpu = HO.read_csv(ROWS_PATH)
    for r in cpu:
        if r["input_sha"] != runs_sha or r["script_sha"] != HO.combined(CPU_CODE):
            stop("predict_rows.csv was not written by this code from this input file")
    rows = rows_by_stage([dict(r, seed=int(r["seed"]), stream=int(r["stream"])) for r in cpu])
    rows[L_STAGE] = read_l_rows(path, runs_sha, HO.HELD_OUT)
    med = tables(rows, HO.HELD_OUT, "PREDICTION for seeds 121 to 160, p 0.2, w0, the s6 "
                 "outage shape, 40 streams of 40 single runs (from %s and %s):"
                 % (os.path.basename(ROWS_PATH), os.path.basename(path)), "")
    against_ev20(med)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
