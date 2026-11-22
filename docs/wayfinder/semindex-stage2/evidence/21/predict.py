#!/usr/bin/env python3
"""The prediction for the s11a testbed runs, archived before those runs.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B predict.py > predict.txt

  PYTHONPATH must point at this worktree's `src`: the interpreter's venv carries another
  checkout of the package, and the assert inside evidence/20's predict.py stops the
  script if that is the one that got imported.

WHY THIS EXISTS
  Ticket 08 spends 200 testbed runs on a stage that draws its poll interval uniformly
  from three arms and re-draws it whenever a poll resolves. The stage exists to leave a
  log a learner can read, not to put a number beside a policy, so the only thing a
  prediction can be for is a check that the replay and the testbed ran the same thing. A
  check is worth something only if the numbers it is held to were written down first, so
  every quantity pass line 5 reads is printed here, before the first run.

WHAT IS PREDICTED, AND FROM WHAT
  The cells are p 0.2 and p 0.4, seeds 1 to 100, workload w0, the s6 outage shape, k 4,
  the three arms 3.695 / 6.0 / 10.043 s, choice jitter 0.1, the re-decision on. Each
  (p, seed) is simulated 40 times, once per replicate stream; one stream is one
  experiment, the condition the testbed has: one run per seed, 100 seeds.

  Every run needs four exogenous inputs (evidence/13's run dict): the outage timeline,
  the record creation stamps, t0 and the per-node `agent_start`. Seeds that already have
  a run with this outage shape in the store take all four from that run. The rest take
  the timeline and the stamps from this worktree's own `evalgen.build_experiment`, which
  depends on (p, seed, k, workload) alone, and borrow t0 and `agent_start` from one of
  the logged runs. The split is printed below, per p.

WHAT IS REUSED FROM evidence/20 AND WHAT IS NOT
  Reused by import, so that one scorer and one engine serve both steps: `timeline_of`,
  `created_of` and `neighbors_of` (a generated description to the replay's inputs),
  `vals` (the pooled delay and the polls per run), and through evidence/20 the modules
  it resolved -- evidence/18's `run_of`, `rep_rng_of` and `sim`, evidence/14's
  `early_score`, evidence/13's constants.
  Rewritten here, because evidence/20's version fixes p 0.2, the stage `s10b3p695`, the
  single 3.695 s choice or the seed range 121 to 160 inside its body: `experiment_of`
  (takes p), `donor_seed` and `synthetic_run` (two p, two donor sets, seeds 1 to 100; the
  latter folded into `run_dict`, which also has a logged run to take instead),
  `gate_generator` (both p, the seeds that exist), `job` and the printing, which in
  evidence/20 walk a five-stage table this step does not have.

THE GATE
  Nothing is printed as a prediction before `build_experiment`'s outage events and
  record schedule reproduce the stored `*.experiment.json` of every seed that has a run
  with this shape. That says the instants this prediction is built from are the ones the
  testbed will schedule. Nothing is written while it runs, because `generate_run` would
  lay a ReCefore yaml and a per-node config beside every description and those configs
  carry key material.
  The replay's own gate is `gate.py` in this directory: the re-decision branch added to
  evidence/13's replay changes no existing mode, and over a single choice it reproduces
  the fixed arm event for event. Neither gate is a measurement of this stage.

WHAT IT CANNOT SHOW
  Every number here is simulated. The replay has been held against a testbed measurement
  for the fixed arms and the two stage-1 policies only (evidence/13); evidence/20's runs
  add a rule and have not been read against the replay yet, and **no run anywhere has
  ever been made with the re-decision on**, so the branch this prediction leans on has no
  measurement behind it at all. That is exactly what pass line 5 is for, and why a miss
  there is read as the replay's copy of the branch before anything else.

The seed strings, the RNG constructors, the replicate count and the donor map are fixed
on purpose: changing any of them moves the printed numbers, and a prediction that moves
after it was archived is worth nothing.
"""
import glob
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV20 = os.path.normpath(os.path.join(HERE, "..", "20"))

# evidence/20's predict.py by path, not by name: this file is also `predict.py` and sits
# first on sys.path, so a plain import would hand back this module itself.
_spec = importlib.util.spec_from_file_location("predict20", os.path.join(EV20, "predict.py"))
P = importlib.util.module_from_spec(_spec)
sys.modules["predict20"] = P
_spec.loader.exec_module(P)

# The modules evidence/20 already resolved, taken through it rather than imported again:
# a second import off a different sys.path entry would give a second module object, and
# the two steps would then run two engines and two scorers.
A, B, G, R, W = P.A, P.B, P.G, P.R, P.W

STAGE = "s11a"
PS = (0.2, 0.4)
SEEDS = list(range(1, 101))
K = 4                                   # records per generation time, as the sweep
DISCONNECT = (8.0, 12.0)                # the outage range, as the sweep
WORKLOAD = "w0"                         # the main cell's load, as the sweep
CHOICES = [3.695, 6.0, 10.043]          # the three arms drawn from
JF = 0.1                                # the choice jitter
NREP = 40                               # replicate streams, i.e. 40 simulated experiments
BAND = 0.06                             # pass line 5: within plus or minus 6 %
WORKERS = 3                             # the ceiling the testbed sweep leaves free
CUT_S = A.CUT_MS / 1000.0

# The seeds that already have a run with this outage shape, k 4 and the default (w0)
# load, found by reading every meta.json in the store: p 0.2 has the s6 family's seeds 1
# to 20, p 0.4 has those plus the s8 family's 21 to 40. One stage per p lends the
# exogenous half of all of them; `s8b6p0` is the only p 0.4 stage that covers 1 to 40 in
# one piece. The gate below fails loudly if one of these runs is not there.
REAL_STAGE = {0.2: "s6b10p043", 0.4: "s8b6p0"}
REAL_SEEDS = {0.2: list(range(1, 21)), 0.4: list(range(1, 41))}
# The label inside the replicate seed string. It is a string, not a stage that is read:
# evidence/14 onward derive every tick-lag stream from `trickle:<label>`, and keeping
# the label they used is what makes this step's streams the streams those steps drew.
STREAM_LABEL = "s6b10p043"

# Filled before the pool is created; the workers inherit them through the fork.
RUNS: dict[tuple[float, int], dict] = {}


# ---------------------------------------------------------------------------
# The run dicts
# ---------------------------------------------------------------------------

def experiment_of(p, seed):
    """The generated experiment description of one (p, seed), with no files written.

    evidence/20's own, with p as an argument instead of its module constant. The stage
    name and the poll settings are passed because the generator wants a complete run;
    neither enters `events` or `schedule`, which is why one description per (p, seed)
    serves both the gate and the prediction.
    """
    return G.build_experiment(
        p, seed, STAGE, "/nonexistent", k=K, store_synchronous="NORMAL",
        disconnect_duration_s=DISCONNECT, workload=WORKLOAD,
        state_poll_interval_choices_s=list(CHOICES), state_poll_choice_jitter_frac=JF,
        state_poll_choice_redecide=True)


def donor_seed(p, seed):
    """Which logged run lends t0 and `agent_start` to a seed that has no run of its own.

    The generated seeds are spread over every donor of their own p in turn, so that no
    single donor's clock sits under a large part of the prediction: 80 seeds over 20
    donors at p 0.2, 60 over 40 at p 0.4. The map is arithmetic rather than drawn, so it
    does not move if this file is run again.
    """
    donors = REAL_SEEDS[p]
    return donors[(seed - 1) % len(donors)]


def run_dict(p, seed):
    """The replay's run dict for one (p, seed), from the store where there is a run.

    A seed with a run keeps that run's realized outage instants, its logged creation
    stamps and its own clock -- everything the testbed actually did on those inputs. A
    seed without one gets the scheduled outage instants and the scheduled creation
    stamps, and borrows a clock. The poll settings are this stage's own in both cases.
    """
    if seed in REAL_SEEDS[p]:
        run = dict(B.run_of(REAL_STAGE[p], p, seed))
    else:
        donor = B.run_of(REAL_STAGE[p], p, donor_seed(p, seed))
        exp = experiment_of(p, seed)
        t0 = donor["t0"]
        created = P.created_of(exp, t0)
        W.check_created(created, t0, (STAGE, p, seed))
        run = dict(donor, seed=seed, run_dir=None, t0=t0, created=created,
                   timeline=P.timeline_of(exp, t0), neighbors=P.neighbors_of(exp),
                   group=exp["group"])
    run.update(stage=STAGE, p=p, policy=None, rule=None,
               choices=list(CHOICES), choice_jitter=JF, redecide=True)
    return run


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def decisions(events, t0):
    """What one simulated run's decision log holds, for pass line 4's counters.

    (send rows, reschedule rows, reschedule rows triggered by a timeout, reschedule rows
    the clamp moved, reschedule rows written at or after the agent's stop instant, one
    count per choice). The send rows are counted over the whole run and not over the
    window: pass line 4 asks what one run decided, while the polls per run of pass line 5
    is a windowed count and comes from the scorer instead.

    The stop-instant count is what the testbed will NOT have. This replay sends no poll
    at or after t0 + stop_at_s, but it does resolve a poll that is still outstanding
    there, while the agent has already left its loop and writes nothing; so every row
    counted here is one the testbed cannot produce, and the count bounds how far the
    reschedule totals below sit above what the runs will show.
    """
    sends = resched = timed_out = clamped = late = 0
    per_choice = [0] * len(CHOICES)
    for rows in events.values():
        for e in rows:
            if e.get("event") != "decision":
                continue
            if e.get("kind") == "state_poll":
                sends += 1
            elif e.get("kind") == "state_poll_reschedule":
                action = e["action"]
                resched += 1
                timed_out += action["trigger"] == "timed_out"
                clamped += bool(action["clamped"])
                late += e["ts_ms"] - t0 >= R.STOP_AT_MS
                per_choice[CHOICES.index(action["interval_s"])] += 1
    return [sends, resched, timed_out, clamped, late] + per_choice


def job(key):
    """Simulate one (p, seed) once per replicate stream and score each simulation.

    The replicate stream is evidence/18's own, so the streams this step draws are the
    streams evidence/14 onward drew on this cell. Each simulation is scored once, by
    evidence/14's headline scorer, and its decision log is counted; the events are
    dropped immediately, since 40 streams of five nodes would otherwise be held at once.
    """
    p, seed = key
    run = RUNS[key]
    rows = []
    for rep in range(NREP):
        events = B.sim(run, B.rep_rng_of(STREAM_LABEL, p, seed, rep))
        rows.append(list(A.early_score(events, run["t0"])) + decisions(events, run["t0"]))
    return key, rows


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def gate_generator():
    """`build_experiment` against the stored `*.experiment.json` of every run that exists.

    What this states is that the outage events and the record schedule the prediction is
    built from are the ones the testbed laid down for the seeds where both exist. It does
    not state that the instants have the shape `w0` and the outage shape are defined to
    have; `tests/test_evalgen.py` answers that against hand-written expectations.
    """
    bad = []
    for p in PS:
        for seed in REAL_SEEDS[p]:
            exp = experiment_of(p, seed)
            run_dir = B.run_of(REAL_STAGE[p], p, seed)["run_dir"]
            with open(glob.glob(os.path.join(run_dir, "*.experiment.json"))[0],
                      encoding="utf-8") as fh:
                stored = json.load(fh)
            mine = sorted((e["at"], e["type"], tuple(e["nodes"])) for e in exp["events"]
                          if e["type"] in ("link_down", "link_up"))
            theirs = sorted((e["at"], e["type"], tuple(e["nodes"])) for e in stored["events"]
                            if e["type"] in ("link_down", "link_up"))
            if mine != theirs:
                bad.append((p, seed, "outage events"))
            if exp["schedule"] != stored["schedule"]:
                bad.append((p, seed, "record schedule"))
    print("The gate, the generator against the stored descriptions of every seed that "
          "has a run with this outage shape")
    for p in PS:
        print(f"  p {p}: {REAL_STAGE[p]} seeds {REAL_SEEDS[p][0]} to {REAL_SEEDS[p][-1]}")
    if bad:
        print("STOP: the generator differs for "
              + ", ".join(f"p {p} seed {s} {w}" for p, s, w in bad)
              + ". No prediction is printed.")
        return False
    print("  every outage event and every record schedule equal, compared as objects.")
    return True


# ---------------------------------------------------------------------------
# The prediction
# ---------------------------------------------------------------------------

def stream(res, p, rep):
    """One experiment's rows: one row per seed, for stream `rep`.

    Columns: the scorer's n4, c4, lat4 and windowed polls, then `decisions`'s counters.
    """
    return np.array([res[(p, s)][rep] for s in SEEDS], float)


def print_split():
    """Which seeds carry a logged exogenous half and which carry a generated one."""
    print("\nThe exogenous half of each run, per p:")
    for p in PS:
        real = REAL_SEEDS[p]
        gen = [s for s in SEEDS if s not in real]
        print(f"  p {p}: {len(real)} seeds from logged runs of {REAL_STAGE[p]} (seeds "
              f"{real[0]} to {real[-1]}: realized outage instants, logged creation stamps, "
              f"their own t0 and agent_start) and {len(gen)} generated (scheduled outage "
              f"instants and scheduled creation stamps, t0 and agent_start borrowed from "
              f"the logged seed at position (seed - 1) mod {len(real)} of that list, "
              f"which here is {real[0]} + (seed - 1) mod {len(real)}).")
    print(f"  The two halves are not separated in the numbers below: the sweep runs all "
          f"{len(SEEDS)} seeds of each p as one stage, and so does this prediction.")


def print_headline(res):
    """Pass line 5's four values: the delay and the window poll count of each p.

    The median over the 40 streams is the predicted value rather than the mean, because
    it is the middle of what a one-run-per-seed experiment produces and is not moved by
    one stream that drew an unlucky set of jitter phases. The per-stream minimum and
    maximum stand beside it, so a measured value outside the band can be read against how
    much the condition moves on its own.
    """
    print(f"\nPass line 5. Seeds {SEEDS[0]} to {SEEDS[-1]}, w0, the s6 outage shape, the "
          "three arms re-drawn at every resolution.")
    print(f"  headline delay = pooled sum(lat4)/sum(n4) over the {len(SEEDS)} runs of a "
          f"stream, over the records created at or before t0+{CUT_S:.0f} s;")
    print(f"  polls per run = mean over the {len(SEEDS)} runs of the state_poll rows in "
          "[t0+10 s, t0+130 s]; a reschedule row is not a poll and is not counted here.")
    print("  p   | delay s median [band]            min..max       | polls median [band]"
          "            min..max        | censored share")
    for p in PS:
        v = np.array([P.vals(stream(res, p, rep)[:, :4]) for rep in range(NREP)])
        delay, polls, cens = (np.median(v[:, i]) for i in range(3))
        print("  %-3s | %7.3f [%7.3f..%7.3f] %7.3f..%7.3f | %6.2f [%6.2f..%6.2f] "
              "%6.2f..%6.2f | %5.3f"
              % (p, delay / 1000.0, delay / 1000.0 * (1 - BAND), delay / 1000.0 * (1 + BAND),
                 v[:, 0].min() / 1000.0, v[:, 0].max() / 1000.0,
                 polls, polls * (1 - BAND), polls * (1 + BAND),
                 v[:, 1].min(), v[:, 1].max(), cens))
    print("  The band is the median times 1 -+ %d %%. These four values -- two delays "
          "and two poll counts -- are what pass line 5 reads." % round(100 * BAND))


def print_decisions(res):
    """Pass line 4's counters: how much a run decides, and how the draw landed.

    Send rows and reschedule rows are counted separately because they are two different
    decisions: the send-time draw is the provisional one the source has always made, the
    reschedule row is the one this stage was built to log. Median, minimum and maximum
    are over the 4000 simulated runs of a p (100 seeds by 40 streams), which is wider
    than the 100 runs the testbed will have.
    """
    print("\nPass line 4. Decisions per run, over the %d simulated runs of each p "
          "(%d seeds by %d streams):" % (len(SEEDS) * NREP, len(SEEDS), NREP))
    print("  p   | send rows median min..max | reschedule rows median min..max | "
          "reschedule rows per send | of those, after the stop instant")
    for p in PS:
        m = np.concatenate([stream(res, p, rep) for rep in range(NREP)])
        s, r, late = m[:, 4], m[:, 5], m[:, 8]
        print("  %-3s | %6.1f %6d..%-6d     | %6.1f %6d..%-6d          | %5.3f "
              "                  | %5.3f of the reschedule rows"
              % (p, np.median(s), s.min(), s.max(), np.median(r), r.min(), r.max(),
                 r.sum() / s.sum(), late.sum() / r.sum()))
    print("  A send row is one `state_poll` decision, a reschedule row one "
          "`state_poll_reschedule`. Ticket 08 read 174 to 280 decisions per run off three "
          "earlier stages and called that a guess to be re-measured; the send rows above "
          "are this replay's answer for the uniform draw.")
    print("  The last column is the share of reschedule rows this replay writes at or "
          "after t0+%.0f s, where the testbed's agent has already stopped and writes "
          "nothing. The testbed's reschedule counts should sit that much BELOW the "
          "medians above, and this replay cannot show the gap any more closely."
          % (R.STOP_AT_MS / 1000.0))

    print("\n  Resolutions, per p, over the same runs:")
    print("  p   | timeout share median min..max | clamped share | "
          "share per arm 3.695 / 6.0 / 10.043 | predicted N | tolerance at that N")
    for p in PS:
        # Per stream, the totals over its 100 runs: reschedule rows, of which triggered
        # by a timeout, clamped, after the stop instant, then one column per choice.
        per_stream = np.array([stream(res, p, rep)[:, 5:].sum(0) for rep in range(NREP)])
        resched = per_stream[:, 0]
        tshare = per_stream[:, 1] / resched
        cshare = per_stream[:, 2] / resched
        arms = per_stream[:, 4:] / resched[:, None]
        n = np.median(resched)
        tol = 3 * math.sqrt((2.0 / 9.0) / n)
        print("  %-3s | %5.3f %5.3f..%5.3f           | %5.3f         | "
              "%5.3f / %5.3f / %5.3f               | %9.0f | %.4f +- %.4f"
              % (p, np.median(tshare), tshare.min(), tshare.max(), np.median(cshare),
                 np.median(arms[:, 0]), np.median(arms[:, 1]), np.median(arms[:, 2]),
                 n, 1.0 / 3.0, tol))
    print("  The timeout share is the share of reschedule rows whose trigger was a "
          "timeout, the clamped share the share whose re-decided instant had already "
          "passed and was pulled forward to now.")
    print("  N is the median number of reschedule rows one stream's 100 runs produce, "
          "and the tolerance beside it is 1/3 +- 3*sqrt((2/9)/N) read at that N. It is "
          "printed for information only: the pass line evaluates the same formula at the "
          "N counted after the run, per p, and that formula does not move.")


FOOTER = """
Three sentences that belong beside every number above.
  This stage's delay is not a quantity anyone compares policies on. A uniform draw over
  three arms is not a policy, and ticket 08 says so: the prediction exists only to check
  that the replay and the testbed ran the same thing, and pass line 5 is that check and
  nothing more.
  The re-decision branch has no testbed measurement behind it, in this replay or in any
  other step. Every delay and every poll count above therefore rests on a branch that was
  read off the source and checked against the source's own single-choice behaviour
  (gate.py), never against a measurement.
  The two halves of the exogenous inputs are not corrected for: a seed with a logged run
  carries realized outage instants and a logged clock, a generated seed carries the
  scheduled instants and a borrowed clock. evidence/20 measured what that borrowing costs
  on seeds 1 to 20 at p 0.2 -- at most 0.3 % in delay for the schedules and 1.1 % for the
  clock -- and this step neither re-measures nor corrects it."""


def main():
    """One timing line on stderr, then the gate and the prediction on stdout."""
    for p in PS:
        for seed in SEEDS:
            RUNS[(p, seed)] = run_dict(p, seed)
    key = (PS[0], SEEDS[0])
    t = time.time()
    B.sim(RUNS[key], B.rep_rng_of(STREAM_LABEL, *key, 0))
    one = time.time() - t
    total = one * len(PS) * len(SEEDS) * NREP
    # On stderr, not in predict.txt: a wall-clock reading differs from one run to the
    # next, and with it in the file the frozen prediction could not be reproduced byte
    # for byte, which is the check a reader runs to see that nothing moved.
    print("One replay of this stage takes %.3f s, so the %d replays below are %.0f s of "
          "one core: about %.0f s on 3 workers and %.0f s on 5."
          % (one, len(PS) * len(SEEDS) * NREP, total, total / 3, total / 5),
          file=sys.stderr)
    if not gate_generator():
        return 1
    with mp.Pool(WORKERS) as pool:
        res = dict(pool.map(job, list(RUNS), chunksize=1))
    print_split()
    print_headline(res)
    print_decisions(res)
    print(FOOTER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
