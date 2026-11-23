#!/usr/bin/env python3
"""Acceptance 5: the hindsight label of every resolution-time decision of a run.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B label.py [options]

    --ps 0.2,0.4      the cells to label              (default: both)
    --seeds 1-100     the seed range                  (default: 1-100)
    --passes 8        how many Mode B passes per arm  (default: 8)
    --out DIR         where the label files go        (default: the learn directory)
    --quiet           no per-run progress line

WHAT A LABEL MEANS
  To label one resolution-time decision k of a run: every decision before k is a uniform
  draw over the three arms, decision k is forced to one arm, and every decision of every
  node after k is made by the formula rule F (`port.F_MEM`).  The run is simulated to its
  end once per arm, and the arm with the best score is the label.  So a label says
  **which arm is best now if F takes over afterwards**.  It is NOT the hindsight-optimal
  sequence of intervals: one decision is moved, not all of them, and a decision maker
  learned from these labels leans towards F improved by one step.

  Why not the alternatives (the user's own two rejections): a uniform draw after k as
  well buries the difference between the arms in the later randomness, and F before k as
  well leaves every state F itself does not reach unlabelled, while the other learner
  learns from the uniformly drawn stage's logs and needs labels over that same spread.

THE SCORE OF ONE SIMULATED RUN, in ms
  `score.score_run`, which is `score.py`'s one definition:
  (sum of the delays of the delivered records + sum over the censored records of the time
  from the record's creation to the end of the window) / (delivered + censored)
  + weight x polls.
  Records, window and polls are evidence/14's early-record scorer's (`../14/after.py:39`):
  records created at or before t0 + 72 s, delivery measured at the far node, polls counted
  over [t0 + 10 s, t0 + 130 s].  A censored record enters the numerator at its LOWER
  BOUND rather than being dropped, because dropping it would reward an arm for not
  delivering a record at all.  The run is also scored the other way, censored records
  dropped, and how many labels that moves is printed.

  The weight is read from `weight.json` and is not recomputed here.  2026-11-23: it is
  now measured under THIS score, by `weight.py`, on the labelling's own engine
  configuration -- per-pair lag streams and per-pair jitter streams both on.  Until that
  re-measurement the weight levelled a delay that left the censored records out, which is
  not this score, and `weight.json` carries the superseded value and that reason.

WHAT THIS CANNOT SHOW
  That a label is the testbed's answer.  It is a score inside the replay, and no run
  anywhere has been made with a decision maker that reads the state.
"""
import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import math
import multiprocessing as mp
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import port as PORT  # noqa: E402
import score as SCORE  # noqa: E402

R = PORT.R

# evidence/21's prediction script by path, the way weight.py loads it: its `run_dict` is
# what "built the way weight.py builds them" means, and its SEEDS, CHOICES and JF are the
# main cell's own.
_spec = importlib.util.spec_from_file_location(
    "predict21", os.path.join(os.path.normpath(os.path.join(HERE, "..", "21")),
                              "predict.py"))
PR = importlib.util.module_from_spec(_spec)
sys.modules["predict21"] = PR
_spec.loader.exec_module(PR)

A = PR.A                                # evidence/14's early-record scorer
STAGE = PR.STAGE                        # "s11a", the uniformly drawn stage
JF = PR.JF                              # the choice jitter, 0.1

# One module object for the scorer, or `labelcheck.py` check 6 would compare two of them.
assert SCORE.A is A, (SCORE.A.__file__, A.__file__)

# port.py carries F as a literal so that the four shape checks of this directory do not
# have to import the generator; this is where that copy is held to evidence/20's own.
assert PORT.F_MEM == PR.P.F_MEM, (PORT.F_MEM, PR.P.F_MEM)
assert tuple(PORT.ARMS) == tuple(PR.CHOICES), (PORT.ARMS, PR.CHOICES)

ARMS = tuple(PORT.ARMS)
CHOICES = PORT.union_choices()           # the arms plus everything F reaches from them
PS = (0.2, 0.4)
MAIN_P = 0.2
SEEDS = list(PR.SEEDS)
LEARN_DIR = "/home/lab_shared/semindex-learn/ev22"
WORKERS = 5                              # the task's ceiling on a six-core machine

OUT_CSV = os.path.join(HERE, "labels_summary.csv")
OUT_TXT = os.path.join(HERE, "labels.txt")
OUT_SHA = os.path.join(HERE, "labels.sha256")

with open(os.path.join(HERE, "weight.json"), encoding="utf-8") as _fh:
    WEIGHT = json.load(_fh)["weight_ms_per_poll"]

RUNS: dict[tuple[float, int], dict] = {}        # filled before the pool is created


# ---------------------------------------------------------------------------
# The run dict and the score
# ---------------------------------------------------------------------------

def run_dict(p, seed):
    """One labelling run dict: evidence/21's own, with the union choice set.

    The port refuses an interval outside `choices`, and F leaves the three arms as soon
    as it lengthens, so the set has to be the closure (`port.union_choices`).  A labelling
    run never takes the send-time uniform draw -- its decider decides every decision -- so
    the wider set costs nothing but a reader of these rows seeing six intervals.
    """
    run = PR.run_dict(p, seed)
    return dict(run, policy=None, rule=None, choices=list(CHOICES), choice_jitter=JF,
                redecide=True, decider=None)


# 2026-11-23: the scorer moved to `score.py`, because `weight.py` has to level the arms
# on the SAME delay a label is decided by and the two files had drifted into two
# definitions.  Re-exported under its old name so that every caller still names one
# function and not a module path.
score_run = SCORE.score_run


# ---------------------------------------------------------------------------
# The decider of one labelling simulation
# ---------------------------------------------------------------------------

class F:
    """F as a decider on the port: the rule path's arithmetic, kept per ordered pair.

    One object, used two ways, so that the F the labelling suffix runs and the F
    `labelcheck.py` check 1 compares against the rule path are the same code.  A pair this
    object has never been told about is on F's own `i_min`.

    The update happens BEFORE the value is returned at a resolution, and no update happens
    at a send: that is what the rule path does (`../13/replay.py`'s `resolve` writes
    `nd.ival[nb]` and then reports it on the row it is writing, while `decide_interval`
    reports the current value untouched).
    """

    def __init__(self, ival=None):
        self.ival = dict(ival or {})

    def get(self, pair):
        return self.ival.get(pair, PORT.F_MEM[0])

    def __call__(self, obs):
        w = obs["where"]
        pair = (w["node"], w["neighbor"])
        if w["kind"] == "state_poll_reschedule":
            got = obs["resolved"]
            self.ival[pair] = PORT.f_update(self.get(pair), got["trigger"], got["empty"])
        return self.get(pair)


class Labeller:
    """Roll-in, then one forced arm at decision k, then F for the rest of the run.

    One instance per simulation; it is not reusable, because it carries the run's own
    state.  `where` is read, which a LEARNER may never do: this is a tool naming one
    decision of a run again on a later pass, which is what `where` exists for.

    The streams, and why each one is keyed the way it is:
    - the roll-in's uniform draws come from two streams per ORDERED PAIR, one for the
      resolution-time draws and one for the send-time draws, seeded from the run and the
      pair alone.  Per pair, so that one pair's decision count cannot shift another
      pair's draws; two streams, so that the send-time draws (which only arm a
      provisional timer the resolution replaces) cannot shift the resolution-time ones.
      Nothing in the seed string moves with the arm, with k or with the pass, so every
      simulation of one run draws the same roll-in;
    - the engine's lag and jitter streams are keyed from the run and the pair until
      decision k and are re-keyed there with the pass (`port.Engine.rekey_streams`), so
      that the run before k is one run and the randomness after k is averaged over the
      passes.  The arm is in neither key, so the three arms of one pass share their
      draws position by position.
    """

    def __init__(self, stream_label, k, arm, pass_suffix):
        self.stream_label = stream_label
        self.k = k                       # None: the roll-in run, which never switches
        self.arm = arm
        self.pass_suffix = pass_suffix
        self.engine = None               # set by the caller, which builds it after this
        self._send_rng = {}
        self._res_rng = {}
        self.f = None                    # the F object, built at the switch
        self.last_res = {}               # pair -> its last roll-in resolution interval
        self.n_res = 0                   # resolution-time decisions so far
        self.capture = []                # the roll-in run's own record of every decision

    def _draw(self, cache, kind, pair):
        """One uniform draw over the THREE ARMS, never over the union choice set.

        The index arithmetic is the engine's own (`../13/replay.py:914`), so that a run
        whose decider is this roll-in draws what the stage it imitates would draw.
        """
        g = cache.get(pair)
        if g is None:
            g = cache[pair] = random.Random(
                f"{kind}:{self.stream_label}:{pair[0]}:{pair[1]}")
        return ARMS[int(g.random() * len(ARMS))]

    def __call__(self, obs):
        w = obs["where"]
        pair = (w["node"], w["neighbor"])
        if self.f is not None:
            # After the switch every decision of every pair, at a send and at a
            # resolution alike, is F's.
            return self.f(obs)
        if w["kind"] != "state_poll_reschedule":
            # A send-time decision only arms a provisional timer that the resolution
            # replaces, so it is never labelled; before the switch it is a roll-in draw.
            return self._draw(self._send_rng, "rollin_send", pair)
        ordinal = self.n_res
        self.n_res += 1
        if self.k is None:
            # The roll-in run: no switch, and every decision is recorded so that the
            # labelling runs do not each have to rebuild the probe's own inputs.  The
            # prefix of a labelling run is this run, so the record at ordinal k is the
            # state that run's decision k is made in (`labelcheck.py` check 2).
            value = self._draw(self._res_rng, "rollin_res", pair)
            self.last_res[pair] = value
            self.capture.append({
                "node": w["node"], "neighbor": w["neighbor"], "ts_ms": w["ts_ms"],
                "state": obs["state"], "resolved": obs["resolved"],
                "history": obs["history"], "rollin_arm": value})
            return value
        if ordinal == self.k:
            # The probed decision.  F takes over from the NEXT decision on, starting from
            # each pair's most recent resolution-time interval of the roll-in and from the
            # forced arm on this pair; a pair the roll-in never resolved starts at F's own
            # i_min, which `F.get` is what decides.
            self.f = F(self.last_res)
            self.f.ival[pair] = self.arm
            self.engine.rekey_streams(self.pass_suffix)
            return self.arm
        value = self._draw(self._res_rng, "rollin_res", pair)
        self.last_res[pair] = value
        return value


def simulate(run, stream_label, base_seed, k, arm, pass_suffix):
    """One simulation through the port: (events, engine, decider)."""
    decider = Labeller(stream_label, k, arm, pass_suffix)
    engine = PORT.Engine(dict(run, decider=decider), "B", None, None, None,
                         base_seed, base_seed)
    decider.engine = engine
    return engine.go(), engine, decider


def stream_label_of(p, seed):
    """The seed string every stream of one run derives from: it carries p and the seed."""
    return R.rep_seed(f"label:{STAGE}", p, seed, 0)


# ---------------------------------------------------------------------------
# Labelling one run
# ---------------------------------------------------------------------------

def pick(means):
    """(the label, the margin, the tie flag) from one arm -> mean score map.

    The score is a delay in ms, so the BEST arm is the smallest.  The margin is the second
    best minus the best; an exactly equal first and second is a tie and gets no label,
    because a tie is not a weak preference, it is two runs that scored the same.  The
    ordering tie-break is the arm order, which only decides WHICH of two tied arms is
    called first and is therefore never reported as a label.
    """
    order = sorted(ARMS, key=lambda a: (means[a], ARMS.index(a)))
    margin = means[order[1]] - means[order[0]]
    tie = margin == 0.0
    return (None if tie else order[0]), margin, tie


def paired_t(pass_scores, passes):
    """(t, sd, n) of the paired difference between the best two arms over `passes`.

    `pass_scores` is {repr(arm): [one score per pass]} as a label row carries it, and
    `passes` are the positions in that list to read.  One definition, here beside `pick`,
    because `stability.py` bins its agreement by this number and a second copy would be a
    second definition of the thing being binned.

    2026-11-23.  The winner and the runner-up are the two smallest MEAN scores, the same
    ordering `pick` uses, and `d_j = score_j(runner-up) - score_j(winner)`, so `d` is
    positive on a pass that agreed with the mean.  `t = mean(d) / (sd(d) / sqrt(n))` with
    the SAMPLE standard deviation, which is the paired t statistic over the passes and not
    a test: nothing here computes a p-value or compares it to a critical value.

    `sd` exactly 0 has NO t -- every pass separated the two arms by the same amount, so
    the divisor is 0 -- and the caller is handed None rather than an infinity, because a
    bin that swallowed 0/0 and a genuinely huge t into one number would hide which it was.
    """
    means = {arm: statistics.fmean(pass_scores[repr(arm)][j] for j in passes)
             for arm in ARMS}
    order = sorted(ARMS, key=lambda a: (means[a], ARMS.index(a)))
    diffs = [pass_scores[repr(order[1])][j] - pass_scores[repr(order[0])][j]
             for j in passes]
    n = len(diffs)
    if n < 2:
        return None, None, n
    sd = statistics.stdev(diffs)
    if sd == 0.0:
        return None, 0.0, n
    return statistics.fmean(diffs) / (sd / math.sqrt(n)), sd, n


def label_run(task):
    """One (p, seed): the roll-in run, then every (decision, arm, pass).

    Returns (p, seed, rows, provisional timers fired, wall seconds, passes).  Rows are in
    decision order, which is what makes the file's bytes a function of the inputs alone.
    """
    p, seed, passes = task
    started = time.time()
    # `RUNS` is filled before the pool is created and inherited through the fork, which is
    # what keeps 200 run dicts from being rebuilt in every worker; a caller that drives
    # one run directly (the checks and the stability script) has no pool and gets it built
    # here instead.
    run = RUNS[(p, seed)] if (p, seed) in RUNS else run_dict(p, seed)
    label = stream_label_of(p, seed)
    _ev, engine, decider = simulate(run, label, label, None, None, None)
    fired = engine.provisional_fired
    probes = decider.capture
    t0 = run["t0"]
    rows = []
    for k, probe in enumerate(probes):
        per = {arm: [] for arm in ARMS}
        for j in passes:
            for arm in ARMS:
                events, eng, _d = simulate(run, label, label, k, arm, f"k{k}j{j}")
                fired += eng.provisional_fired
                per[arm].append(score_run(events, t0, WEIGHT))
        means = {arm: statistics.fmean(s[0] for s in per[arm]) for arm in ARMS}
        drop = {arm: statistics.fmean(s[1] for s in per[arm]) for arm in ARMS}
        best, margin, tie = pick(means)
        drop_best, _m, _t = pick(drop)
        rows.append({
            "p": p, "seed": seed, "k": k,
            "node": probe["node"], "neighbor": probe["neighbor"],
            # Relative to the run's own t0, which is what makes two runs comparable; the
            # absolute stamp is kept beside it because the stamps inside `resolved` and
            # `history` are the port's own and are absolute, and a reader needs one clock
            # for all three.
            "ts_rel_ms": round(probe["ts_ms"] - t0, 3),
            "ts_ms": probe["ts_ms"],
            "state": probe["state"], "resolved": probe["resolved"],
            "history": probe["history"],
            "rollin_arm": probe["rollin_arm"],
            "passes": list(passes),
            "mean_score": {repr(arm): means[arm] for arm in ARMS},
            "pass_scores": {repr(arm): [s[0] for s in per[arm]] for arm in ARMS},
            "label": best, "margin": margin, "tie": tie,
            # The same label with the censored records dropped instead of entered at
            # their lower bound, so that the sensitivity of the labels to that one choice
            # is readable from the file and not only from the printed summary.
            "label_drop_censored": drop_best,
            "polls": {repr(arm): [s[6] for s in per[arm]] for arm in ARMS},
            "censored": {repr(arm): [s[3] for s in per[arm]] for arm in ARMS},
        })
    return p, seed, rows, fired, time.time() - started, list(passes)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def cell_bytes(rows):
    """One cell's rows as the gzip stream that goes on disk.

    `mtime=0`, because gzip puts the current time in its header and the file's bytes would
    otherwise move every run.  `sort_keys`, because a dict's insertion order is not part
    of the data.  The rows are sorted by (seed, k) by the caller, because a worker pool
    finishes in whatever order it finishes in.  Built in memory so that the identity check
    can compare two labellings without writing either of them.
    """
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0) as fh:
        for row in rows:
            fh.write((json.dumps(row, sort_keys=True) + "\n").encode())
    return buf.getvalue()


def write_cell(path, rows):
    """`cell_bytes` on disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(cell_bytes(rows))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summary(per_run, cells, fired, wall, passes, lines):
    """Everything Acceptance 5 asks for, per cell and per run."""
    def out(text=""):
        print(text, flush=True)
        lines.append(text)

    out("Hindsight labels: which arm is best now if F takes over afterwards")
    out("==================================================================")
    out(f"F: i_min {PORT.F_MEM[0]}, m {PORT.F_MEM[1]}, i_max {PORT.F_MEM[2]}, on a "
        f"timeout I = {PORT.F_MEM[3]}, no cap on m.")
    out(f"Arms: {list(ARMS)} s.  Choice set declared to the port: {CHOICES} "
        "(the arms plus everything F reaches from them).")
    out(f"Weight: {WEIGHT!r} ms per poll, read from weight.json and not recomputed.")
    out(f"Passes per arm: {len(passes)} ({passes[0]} to {passes[-1]}).")
    out("A label is NOT the hindsight-optimal sequence: one decision is moved, the rest "
        "of the run is F.")
    out()
    for p in sorted(cells):
        rows = cells[p]
        ties = sum(1 for r in rows if r["tie"])
        counts = {arm: sum(1 for r in rows if r["label"] == arm) for arm in ARMS}
        margins = sorted(r["margin"] for r in rows)
        differ = sum(1 for r in rows
                     if r["label_drop_censored"] != r["label"])
        runs = [v for key, v in per_run.items() if key[0] == p]
        out(f"p {p}: {len(runs)} runs, {len(rows)} labelled decisions "
            f"({statistics.fmean(d['decisions'] for d in runs):.1f} per run, "
            f"min {min(d['decisions'] for d in runs)}, "
            f"max {max(d['decisions'] for d in runs)})")
        for arm in ARMS:
            out(f"  label {arm:>6} s : {counts[arm]:6d}  "
                f"({counts[arm] / len(rows):.4f})")
        out(f"  tie (no label) : {ties:6d}  ({ties / len(rows):.4f})")
        out(f"  margin ms, median {statistics.median(margins):.4f}, "
            f"min {margins[0]:.4f}, max {margins[-1]:.4f}")
        out("  the roll-in's own arm at the labelled decision: "
            + ", ".join(f"{arm} s {sum(1 for r in rows if r['rollin_arm'] == arm)}"
                        for arm in ARMS))
        out(f"  labels that move when the censored records are DROPPED instead of "
            f"entered at their lower bound: {differ} of {len(rows)} "
            f"({differ / len(rows):.4f}); a decision that ties under one scoring and "
            "has a label under the other counts as moved")
        out()
    out("Provisional timers that fired before their own poll had resolved, over every "
        f"simulation of this script: {fired}")
    out("  A fired provisional timer is a poll whose send-time interval decided the next "
        "send on its own, so a resolution-time label on it would be a label on the wrong "
        "instant.  Any count above 0 is a failure of this step, not a footnote.")
    out()
    out(f"wall {wall:.1f} s on {WORKERS} workers")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", default=",".join("%g" % p for p in PS))
    ap.add_argument("--seeds", default=f"{SEEDS[0]}-{SEEDS[-1]}")
    ap.add_argument("--passes", type=int, default=8)
    ap.add_argument("--out", default=LEARN_DIR)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    ps = [float(x) for x in args.ps.split(",")]
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = PORT.check_seeds(range(lo, hi + 1), "label.py")
    passes = list(range(args.passes))
    started = time.time()
    for p in ps:
        for seed in seeds:
            RUNS[(p, seed)] = run_dict(p, seed)
    keys = [(p, seed, passes) for p in ps for seed in seeds]
    cells, per_run, fired = {}, {}, 0
    with mp.Pool(WORKERS) as pool:
        for p, seed, rows, got, wall, _ps in pool.imap_unordered(label_run, keys,
                                                                 chunksize=1):
            cells.setdefault(p, []).extend(rows)
            fired += got
            per_run[(p, seed)] = {
                "decisions": len(rows),
                "ties": sum(1 for r in rows if r["tie"]),
                "labels": {arm: sum(1 for r in rows if r["label"] == arm)
                           for arm in ARMS},
                "median_margin": statistics.median([r["margin"] for r in rows]),
                "wall_s": wall}
            if not args.quiet:
                print(f"  p {p} seed {seed}: {len(rows)} decisions, {wall:.1f} s",
                      flush=True)
    files = {}
    for p in cells:
        cells[p].sort(key=lambda r: (r["seed"], r["k"]))
        path = os.path.join(args.out, f"labels_{R.pdir(p)}.jsonl.gz")
        write_cell(path, cells[p])
        files[path] = sha256_of(path)
    lines = summary(per_run, cells, fired, time.time() - started, passes, [])
    lines.append("")
    lines.append("Label files (outside the repository, one per cell)")
    for path in sorted(files):
        lines.append(f"  {path}")
        lines.append(f"    sha256 {files[path]}")
        print(lines[-2])
        print(lines[-1])
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(OUT_SHA, "w", encoding="utf-8") as fh:
        for path in sorted(files):
            fh.write(f"{files[path]}  {path}\n")
    with open(OUT_CSV, "w", encoding="utf-8") as fh:
        # `repr(a)` and not "%g", so that the csv spells an arm the way the JSONL rows key
        # it (`mean_score`, `pass_scores`): 2026-11-23, "%g" wrote `label_6` for the arm
        # the rows call "6.0", and a reader joining the two files on the arm string had to
        # know about both spellings.
        fh.write("p,seed,decisions,ties," + ",".join("label_" + repr(a) for a in ARMS)
                 + ",median_margin_ms,wall_s\n")
        for (p, seed) in sorted(per_run):
            d = per_run[(p, seed)]
            fh.write("%g,%d,%d,%d,%s,%.6f,%.2f\n"
                     % (p, seed, d["decisions"], d["ties"],
                        ",".join(str(d["labels"][a]) for a in ARMS),
                        d["median_margin"], d["wall_s"]))
    print(f"\nwrote {OUT_TXT}, {OUT_CSV} and {OUT_SHA}")
    return 1 if fired else 0


if __name__ == "__main__":
    sys.exit(main())
