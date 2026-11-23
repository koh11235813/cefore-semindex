#!/usr/bin/env python3
"""The checks the label tool has to carry.  Exits non-zero on any failure.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B labelcheck.py

WHAT IT CHECKS
  1. **"F after k" is the real F.**  F run through the port from the FIRST decision on
     reproduces the rule path's F -- the same poll send instants for every ordered pair and
     the same resolution-time interval sequence -- on two runs of each cell, with the
     shared lag stream so that both paths draw from one place.  Without this line, "the
     rest of the run is F" would be a claim about a decider nothing compared to F.
     The rule-path reference goes through `port.Engine` with `rule=member_rule(F_MEM)`,
     because the timeout coordinate of F is written by evidence/18's `resolve` override
     and the base engine's rule leaves the interval alone on a timeout -- a different F.
  2. **The three arms share their prefix.**  For sampled k, the three arms' runs are
     event-for-event equal at every stamp strictly before decision k's, and equal to the
     roll-in run's own events there; and the state, the resolution and the history the
     probed decision is made in are the ones the roll-in recorded at that ordinal.  So
     the three scores differ because of the decision and not because of the prefix.
  3. **A run labelled twice gives identical output bytes**, gzip header included.
  4. **A seed above the selection range is refused.**  The measuring seeds are reserved,
     so the check asks for one above the bound and requires the refusal; no seed above the
     bound appears anywhere else in this directory.
  5. **The choice set is inside the bound under which the "pending" divergence cannot
     happen.**  `min(choices) * (1 - choice_jitter)` has to exceed the Interest lifetime
     plus one tick (`spec.md` deviation 1).  The union choice set is wider than the three
     arms, so this is checked on the set the labelling actually declares.
  6. **The scorer's record set is evidence/14's.**  `score.score_run` repeats evidence/14's
     traversal because `early_score` returns the censored COUNT and not the censored TIME;
     this line requires the four numbers the two have in common to agree.  2026-11-23: the
     scorer moved out of `label.py` into `score.py`, because the weight has to be levelled
     on the delay a label is decided by and `weight.py` had been levelling it on
     `early_score`'s instead; this line is unchanged and still reads the one definition.

WHAT IT CANNOT SHOW
  That a label is right.  Every line here is about the mechanism: that the suffix is F,
  that the prefix is shared, that the bytes are reproducible, that the seeds and the choice
  set are inside their bounds and that the record set is the published one.  Whether the
  score ranks the arms the way the testbed would is Acceptance 6's question at best and
  the testbed's at worst, and neither is answered in this directory.
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label as L  # noqa: E402
import port as PORT  # noqa: E402
import score as SCORE  # noqa: E402

R = PORT.R
B = PORT.B

CELLS = (0.2, 0.4)
F_SEEDS = PORT.check_seeds([1, 2], "labelcheck.py")      # check 1, two runs per cell
PREFIX_SEED = PORT.check_seeds([1], "labelcheck.py")[0]  # checks 2 and 6
IDENTITY_PASSES = [0]                                    # check 3 needs no averaging
SAMPLED_K = 5                                            # how many k check 2 samples


def say(ok, text):
    print(f"{'PASS' if ok else 'FAIL'} {text}", flush=True)
    return 0 if ok else 1


def sends(events):
    """Per ordered pair, the stamp of every `state_poll` row in order."""
    out = {}
    for node, rows in events.items():
        for e in rows:
            if e.get("event") == "decision" and e.get("kind") == "state_poll":
                out.setdefault((node, e["action"]["neighbor"]), []).append(e["ts_ms"])
    return out


def intervals(events):
    """Per ordered pair, the interval of every `state_poll_reschedule` row in order."""
    out = {}
    for node, rows in events.items():
        for e in rows:
            if (e.get("event") == "decision"
                    and e.get("kind") == "state_poll_reschedule"):
                out.setdefault((node, e["action"]["neighbor"]), []).append(
                    e["action"]["interval_s"])
    return out


def before(events, stamp):
    """Every event stamped strictly before `stamp`, flattened and comparable.

    Strictly before, because the probed decision's own row carries that stamp and is the
    row that differs; and flattened with `repr` so that two runs are compared by value
    rather than by object.
    """
    return [(node, tuple(sorted((k, repr(v)) for k, v in e.items())))
            for node in sorted(events) for e in events[node] if e["ts_ms"] < stamp]


# ---------------------------------------------------------------------------

def check_f_is_f():
    """Check 1: the F decider and the rule path place the same polls."""
    bad = 0
    for p in CELLS:
        for seed in F_SEEDS:
            run = L.run_dict(p, seed)
            seed_int = R.rep_seed(f"labelcheck:F:{L.STAGE}", p, seed, 0)
            # The rule path.  `choices` and `policy` are silenced by `rule` inside the
            # engine, but the decider is not, so it has to be cleared by hand.
            ref = PORT.Engine(dict(run, rule=B.member_rule(PORT.F_MEM), decider=None),
                              "B", None, random.Random(seed_int)).go()
            # F through the port, from the first decision on.  The same shared stream, so
            # that a difference is a difference in the intervals and not in the lags.
            got, engine = PORT.sim(dict(run, decider=L.F()),
                                   random.Random(seed_int))
            ok_sends = sends(ref) == sends(got)
            ok_ivals = intervals(ref) == intervals(got)
            bad += say(ok_sends and ok_ivals,
                       f"p {p} seed {seed}: F through the port places the rule path's "
                       f"polls -- {sum(len(v) for v in sends(got).values())} sends over "
                       f"{len(sends(got))} pairs, send instants equal {ok_sends}, "
                       f"resolution intervals equal {ok_ivals}")
            if not (ok_sends and ok_ivals):
                for key in sorted(set(sends(ref)) | set(sends(got))):
                    a, b = sends(ref).get(key, []), sends(got).get(key, [])
                    if a != b:
                        first = next((i for i in range(min(len(a), len(b)))
                                      if a[i] != b[i]), min(len(a), len(b)))
                        print(f"  {key[0]}->{key[1]} {len(a)} vs {len(b)} sends, first "
                              f"differing at {first}: {a[first:first + 3]} vs "
                              f"{b[first:first + 3]}")
            if engine.provisional_fired:
                bad += say(False, f"p {p} seed {seed}: {engine.provisional_fired} "
                                  "provisional timers fired in the F run")
    return bad


def check_prefix():
    """Check 2: the three arms, and the roll-in, agree at every stamp before k."""
    bad = 0
    for p in CELLS:
        run = L.run_dict(p, PREFIX_SEED)
        tag = L.stream_label_of(p, PREFIX_SEED)
        roll_ev, _eng, roll = L.simulate(run, tag, tag, None, None, None)
        n = len(roll.capture)
        ks = sorted({0, n - 1, n // 2, n // 3, 2 * n // 3})[:SAMPLED_K]
        for k in ks:
            probe = roll.capture[k]
            stamp = probe["ts_ms"]
            want = before(roll_ev, stamp)
            seen, obs_ok = [], True
            for arm in L.ARMS:
                events, _e, _d = L.simulate(run, tag, tag, k, arm, f"k{k}j0")
                seen.append(before(events, stamp))
                # The probe's own inputs, which the label row reports as the state the
                # decision was made in.  Compared here because the row takes them from the
                # roll-in run and not from the arm's run.
                hit = [e for node in events for e in events[node]
                       if e.get("kind") == "state_poll_reschedule"
                       and e["ts_ms"] == stamp
                       and e["action"]["neighbor"] == probe["neighbor"]
                       and node == probe["node"]]
                obs_ok = obs_ok and any(
                    e["state"] == probe["state"]
                    and e["action"]["poll_epoch_ms"] == probe["resolved"]["poll_epoch_ms"]
                    and e["action"]["trigger"] == probe["resolved"]["trigger"]
                    and e["action"]["empty"] == probe["resolved"]["empty"]
                    and e["action"]["interval_s"] == arm
                    for e in hit)
            equal = all(s == want for s in seen)
            bad += say(equal and obs_ok,
                       f"p {p} seed {PREFIX_SEED} k {k} of {n}: the three arms and the "
                       f"roll-in carry the same {len(want)} events before the decision's "
                       f"own stamp (prefix equal {equal}), and each arm's row at that "
                       f"instant carries the roll-in's state and resolution and the "
                       f"forced arm ({obs_ok})")
            if not equal:
                for i, s in enumerate(seen):
                    if s != want:
                        j = next((x for x in range(min(len(s), len(want)))
                                  if s[x] != want[x]), min(len(s), len(want)))
                        print(f"  arm {L.ARMS[i]}: {len(s)} vs {len(want)} events, first "
                              f"differing at {j}")
    return bad


def check_identity():
    """Check 3: labelling the same run twice gives the same bytes."""
    first = L.label_run((0.2, PREFIX_SEED, IDENTITY_PASSES))[2]
    second = L.label_run((0.2, PREFIX_SEED, IDENTITY_PASSES))[2]
    a, b = L.cell_bytes(first), L.cell_bytes(second)
    ok = a == b
    bad = say(ok, f"labelling p 0.2 seed {PREFIX_SEED} twice gives identical bytes: "
                  f"{len(a)} bytes over {len(first)} rows")
    if not ok:
        rows = [r for r in zip(first, second) if r[0] != r[1]]
        print(f"  {len(rows)} rows differ; the first: {rows[0] if rows else None}")
    return bad


def check_seed_guard():
    """Check 4: a seed above the selection range is refused, by the guard every entry
    point of this directory passes its seeds through."""
    try:
        PORT.check_seeds([PORT.MAX_SEED + 1], "labelcheck.py check 4")
    except SystemExit as exc:
        return say("reserved for measuring" in str(exc),
                   f"a seed above {PORT.MAX_SEED} is refused: {exc}")
    return say(False, f"a seed above {PORT.MAX_SEED} was NOT refused")


def check_bound():
    """Check 5: the union choice set is inside the bound of spec.md deviation 1."""
    smallest = min(L.CHOICES) * (1.0 - L.JF) * 1000.0
    limit = R.POLL_TIMEOUT_MS + R.TICK_PERIOD_MS
    return say(smallest > limit,
               f"the smallest applied interval of the choice set {L.CHOICES} is "
               f"{smallest:.1f} ms against the {limit:.1f} ms of the Interest lifetime "
               f"plus one tick, a margin of {smallest - limit:.1f} ms, so no poll can be "
               "superseded before it resolves and `last_poll_status` \"pending\" stays "
               "unreachable")


def check_scorer():
    """Check 6: the record set, the window and the poll count are evidence/14's."""
    run = L.run_dict(0.2, PREFIX_SEED)
    tag = L.stream_label_of(0.2, PREFIX_SEED)
    off = []
    for k, arm in ((None, None), (0, L.ARMS[0]), (3, L.ARMS[2])):
        events, _e, _d = L.simulate(run, tag, tag, k, arm,
                                    None if k is None else f"k{k}j0")
        mine = SCORE.early_four(SCORE.terms(events, run["t0"]))
        theirs = tuple(SCORE.A.early_score(events, run["t0"]))
        if mine != theirs:
            off.append((k, mine, theirs))
    return say(not off,
               "on three runs the scorer's (delivered, censored, delay sum, polls) are "
               f"../14/after.py's early_score exactly{'' if not off else ': ' + str(off)}")


def main():
    bad = 0
    print(f"F: {PORT.F_MEM}, arms {list(L.ARMS)}, choice set {L.CHOICES}, "
          f"weight {L.WEIGHT!r} ms per poll\n")
    bad += check_f_is_f()
    bad += check_prefix()
    bad += check_identity()
    bad += check_seed_guard()
    bad += check_bound()
    bad += check_scorer()
    print(f"\n{'every check passed' if not bad else str(bad) + ' CHECK(S) FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
