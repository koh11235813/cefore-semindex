#!/usr/bin/env python3
"""Acceptance 3: with the per-pair lag streams on, one decision moves one pair's lags.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B lagcheck.py

Prints one PASS/FAIL line per check and exits non-zero if any failed.

WHY THIS MATTERS
  A labelling pass runs the same run once per candidate interval of one decision and
  compares the scores.  With the base engine's single shared tick-lag stream, every lag
  of every pair after the changed decision comes out of a different position of that one
  stream, so the three runs differ everywhere and the comparison is not a comparison of
  three candidates for one decision.  The per-pair streams make each pair's lag VALUES a
  function of the pair and the seed alone.  They do NOT make the other pairs hold still in
  the stronger sense of serving the same events; see check 1b.

WHAT IT CHECKS
  1. With the streams on, changing ONE resolution-time decision of one pair leaves every
     other pair's lag sequence equal, value by value, over the common length of the two
     sequences; the lengths are reported beside it, and the per-node first-poll offsets
     are compared the same way.  This is Acceptance 3's own wording, and 2026-11-23 it is
     also a TAUTOLOGY under the streams: position i of pair (node, nb) is by construction
     the i-th draw of `random.Random(f"pair_lag:{seed}:{node}:{nb}")`, a seed string that
     holds nothing a decision can touch, so the value at a common index cannot differ for
     any decider and any change, and the only observable difference -- the length -- is
     what the truncation discards.  The line is therefore a REGRESSION GUARD on the seed
     strings and on the routing (`sim_rng` is None, so any Mode B draw this step failed to
     route through a pair's stream raises rather than passing quietly -- checked, not merely
     intended: with both the shared stream and the pair seed absent the engine raises
     AttributeError on `None.uniform`), and it is not evidence about this run.  Check 1b is
     the quantity a decision can move.
     The first-poll-offset half of this line is weaker still: `start_lag` is called in the
     engine's initialisation loop, before any event or decider has run, so no decision can
     move an offset in ANY configuration.  Check 3's negative control therefore does not
     control it -- with the streams off the offsets come out all equal too, which check 3's
     own report line prints -- and all that comparison carries is the routing statement.
  1b. The quantity a decision CAN move, printed and not gated: each other pair's lag draw
     COUNT and draw-KIND sequence, and each other pair's per-send applied intervals.  On
     the run below they moved, so the other pairs do not hold still in that sense.  This
     is reported as a negative result, not as a pass line: Acceptance 3 asks for check 1,
     and making 1b the gate would redefine it.
  1c. 2026-11-23: the same measurement with the per-pair JITTER stream on as well, which
     is the remedy 1b asks for.  The two verdicts printed side by side are what says
     whether the remedy holds on this run; also not gated, for 1b's reason.  Read 1c's
     own draw-KIND sequence before calling a surviving residue randomness: a shifted send
     moves WHEN a record is advertised, so another pair's store, fetch plan and
     fetch-timeout draws can move through the simulated world rather than through a
     generator.
  2. The change is not vacuous: the forced arm really is on the targeted decision's own
     emitted row, and the target pair's run really did move (its lag draw count or its
     event stream).  2026-11-23: the third disjunct, "its own lag VALUES moved", is gone;
     with the per-pair streams on that pair draws from one seed string in both runs, so a
     value at a common position cannot differ and the disjunct could never carry the line.
  3. The negative control: with the streams off, the same single change moves other
     pairs' lags.  Without this line check 1 would also pass on an engine that drew the
     same lag everywhere.

WHAT IT CANNOT SHOW
  That two candidates see the same lags.  A draw is still the n-th draw of its pair's
  stream, so a candidate that changes how many draws a pair makes shifts that pair's later
  lags by position -- and it is not only the target pair: the per-node jitter stream moves
  another pair's send instants, which is how `h2->h3` below draws 26 lags in one run and 27
  in the other.  Check 1b measures that.  Check 1c measures the same thing with the
  per-pair JITTER stream on as well, and on that run no other pair moves at all, so the
  coupling 1b found is damped HERE and not only by averaging.  What neither check damps
  is the probed pair itself: it really does send a different number of polls, so its own
  draws land on different events.  Averaging that away over Mode B passes is the
  labelling's own job and its size is measured by `stability.py`, not here.
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import port as PORT  # noqa: E402

R = PORT.R

STAGE, P, SEED = "s11a", 0.2, PORT.check_seeds([1], "lagcheck.py")[0]
BASE_ARM, OTHER_ARM = 6.0, 10.043       # two of the s11a arms
TARGET = ("h2", "h1")                   # the ordered pair whose one decision changes
TARGET_KIND = "state_poll_reschedule"   # the resolution-time decision, the one that acts
TARGET_NTH = 4                          # the fifth such decision of that pair, from 0


class Rec(PORT.Engine):
    """The engine, keeping every tick lag it drew in order, per ordered pair.

    The lags cannot be read back out of the event log: a logged stamp is a due instant
    plus its lag, and the due instant moves with the decision, so a comparison over the
    log could not tell a moved lag from a moved decision.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lags = {}
        self.starts = {}
        self.kinds = {}
        self.kindseq = {}

    def tick_lag(self, node, nb, kind):
        value = super().tick_lag(node, nb, kind)
        self.lags.setdefault((node, nb), []).append(value)
        # The draw's own kind too, because a pair's lag count can move without its send
        # count moving: a lag is drawn per send AND per detected poll timeout.  The
        # SEQUENCE and not only the count, because a count can agree while an inserted
        # draw of one kind shifts every later draw onto a different event -- which is what
        # check 1b is for and what a count-only comparison would miss.
        self.kinds[(node, nb, kind)] = self.kinds.get((node, nb, kind), 0) + 1
        self.kindseq.setdefault((node, nb), []).append(kind)
        return value

    def start_lag(self, node):
        value = super().start_lag(node)
        self.starts[node] = value
        return value


def one_change(changed):
    """A decider returning BASE_ARM everywhere, or OTHER_ARM at exactly one decision.

    It counts the decisions of each (pair, kind) itself, which is what a labelling tool
    does; `where` is read here for that counting, and this is a tool, not a learner.
    """
    seen = {}

    def decide(obs):
        w = obs["where"]
        key = (w["node"], w["neighbor"], w["kind"])
        nth = seen.get(key, 0)
        seen[key] = nth + 1
        if (changed and key == TARGET + (TARGET_KIND,) and nth == TARGET_NTH):
            return OTHER_ARM
        return BASE_ARM

    return decide


def run(changed, pair_streams, pair_jitter=False):
    """One simulation: (engine, events, how many decider calls it made)."""
    run_dict = dict(R.load_run(STAGE, P, SEED),
                    policy=None, rule=None, redecide=True, decider=one_change(changed))
    # With the streams on, `sim_rng` is deliberately left None: any Mode B draw site this
    # step failed to route through a pair's stream then raises instead of quietly mixing
    # the two sources.  With them off the shared stream is the one evidence/18 would use.
    seed_string = R.rep_seed(f"lagcheck:{STAGE}", P, SEED, 0)
    engine = Rec(run_dict, "B", None,
                 None if pair_streams else random.Random(seed_string),
                 None, seed_string if pair_streams else None,
                 seed_string if pair_jitter else None)
    events = engine.go()
    return engine, events, sum(engine.decider_ord.values())


def differences(left, right):
    """The pairs whose lag sequences differ over the common length, and the lengths."""
    out = []
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key, []), right.get(key, [])
        n = min(len(a), len(b))
        first = next((i for i in range(n) if a[i] != b[i]), None)
        out.append((key, len(a), len(b), first))
    return out


def flat(events):
    return [(node, tuple(sorted((k, repr(v)) for k, v in e.items())))
            for node in sorted(events) for e in events[node]]


def applied(events):
    """Per ordered pair, the applied interval of each `state_poll` row in order."""
    out = {}
    for node, evs in events.items():
        for e in evs:
            if e.get("kind") == "state_poll":
                out.setdefault((node, e["action"]["neighbor"]), []).append(
                    e["action"]["applied_interval_s"])
    return out


def carried(ab, ac):
    """Per pair whose applied intervals moved: the moved slice, and whose values it is.

    2026-11-23: printed because the claim about this run is otherwise a claim no output
    shows.  The two pairs of one node EXCHANGE a contiguous run of that node's jitter
    draws and then each returns to its own values, so only a printed slice says how long
    the exchange lasts.  `src` names the other pair of the same node whose BASE values the
    changed slice is, and `own_again` says whether the pair's values after the slice are
    its own again.

    `own_again` is `hi < n and len(a) == len(b)`, and the row carries `n`, `len(a)` and
    `len(b)` so a reader can see it.  The sequences are the per-send APPLIED INTERVALS, not
    the raw lag draws: the applied interval is where a re-assigned jitter draw becomes
    visible, and the lag draws themselves are compared by check 1 at equal indices.  2026-11-23, second correction: it used to be
    `a[hi:n] == b[hi:n]`, which CANNOT BE FALSE -- `hi` is one past the LAST differing
    index below `n`, so the slice from `hi` to `n` has no differing index by construction,
    and it was also True when `hi == n` (the exchange ran to the end of the shorter
    sequence and there is no tail at all) and when the two sequences have different
    lengths (there is a tail the comparison never reached).  A verdict that cannot be
    False is not a measurement, and `spec.md` was citing this one as output.
    """
    out = []
    for key in sorted(set(ab) | set(ac)):
        a, b = ab.get(key, []), ac.get(key, [])
        n = min(len(a), len(b))
        idx = [i for i in range(n) if a[i] != b[i]]
        if not idx:
            continue
        lo, hi = idx[0], idx[-1] + 1
        src = [k for k in sorted(ab)
               if k != key and k[0] == key[0] and ab[k][lo:hi] == b[lo:hi]]
        out.append((key, lo, hi, [round(x, 4) for x in a[lo:hi]],
                    [round(x, 4) for x in b[lo:hi]], src,
                    hi < n and len(a) == len(b), n, len(a), len(b)))
    return out


def seq_moved(left, right, skip):
    """Per pair other than `skip`: the draw counts and the first differing draw KIND."""
    out = []
    for key in sorted(set(left) | set(right)):
        if key == skip:
            continue
        a, b = left.get(key, []), right.get(key, [])
        n = min(len(a), len(b))
        first = next((i for i in range(n) if a[i] != b[i]), None)
        if len(a) != len(b) or first is not None:
            out.append((key, len(a), len(b), first))
    return out


def say(ok, text):
    print(f"{'PASS' if ok else 'FAIL'} {text}", flush=True)
    return 0 if ok else 1


def report(label, rows, starts_equal):
    """One line per pair, so a reader sees the lengths and not only the verdict."""
    print(f"  {label}: per-pair lag sequences (pair, base n, changed n, first differing "
          "position or '-')")
    for key, na, nb, first in rows:
        mark = "-" if first is None else str(first)
        star = "  <- the changed pair" if key == TARGET else ""
        print(f"    {key[0]}->{key[1]}  {na:4d} {nb:4d}  {mark}{star}")
    # 2026-11-23: REPORTED, never gated.  `start_lag` runs in the engine's initialisation
    # loop before any event or decider executes, so no decision this script makes can move
    # an offset; the comparison carries only the sim_rng=None routing, and a check that
    # cannot come out the other way is not a check.
    print(f"  {label}: per-node first-poll offsets, reported and not gated "
          + ("(all equal)" if starts_equal else "(NOT all equal -- which would mean the "
             "initialisation loop itself changed, not a decision)"))


def report_1b(tag, base, chg, ev_base, ev_chg, n_others):
    """What the per-pair streams do NOT align, printed and not gated.

    `Node.jitter_rng` is per NODE and is drawn once per send over BOTH of that node's
    neighbours, in event order.  Moving one pair's sends therefore re-assigns that node's
    later jitter draws between its two pairs, which moves the other pair's applied
    intervals although every lag it drew is the same value at the same position, and a
    send instant shifted that way can land on the other side of a `link_down` and add a
    poll-timeout lag draw.  Printed, not gated: Acceptance 3 asks for check 1.

    2026-11-23: called twice, once with the per-pair jitter stream off (`1b`, the residue
    as the first half measured it) and once with it on (`1c`, the remedy).  A residue that
    survives the jitter stream is not randomness coupling: a shifted send changes WHEN a
    record is advertised, so the other pair's store, its fetch plan and therefore its
    fetch-timeout draws move through the simulated world and not through the generators.
    """
    kseq = seq_moved(base.kindseq, chg.kindseq, TARGET)
    jit = [r for r in differences(applied(ev_base), applied(ev_chg))
           if r[0] != TARGET and r[3] is not None]
    print(f"  {tag}: of the other pairs, those whose lag DRAW COUNT or draw-KIND sequence "
          "moved (pair, base n, changed n, first differing kind index): "
          + (", ".join(f"{k[0]}->{k[1]} {na} {nb} "
                       f"{'-' if first is None else first}" for k, na, nb, first in kseq)
             if kseq else "none"))
    moved_kinds = {k: (base.kinds.get(k, 0), chg.kinds.get(k, 0))
                   for k in sorted(set(base.kinds) | set(chg.kinds))
                   if base.kinds.get(k, 0) != chg.kinds.get(k, 0)}
    print(f"  {tag}: draw counts that moved, by (node, neighbour, draw kind): "
          + (str(moved_kinds) if moved_kinds else "none"))
    print(f"  {tag}: of the other pairs, those whose per-send APPLIED INTERVALS moved "
          "(pair, first differing send): "
          + (", ".join(f"{k[0]}->{k[1]} {first}" for k, _a, _b, first in jit)
             if jit else "none"))
    print(f"  {tag}: the moved slice of every pair, and whose values it carries (pair, "
          "positions, base, changed, the pair of the same node whose base values the "
          "changed slice is, own values again afterwards with the lengths it is read from)")
    for key, lo, hi, a, b, src, back, n, na, nb in carried(applied(ev_base),
                                                           applied(ev_chg)):
        names = ", ".join(f"{s[0]}->{s[1]}" for s in src) or "none"
        print(f"    {key[0]}->{key[1]}  [{lo}:{hi}]  base {a}  changed {b}  "
              f"from {names}  own again: {back} (common length n {n}, sends {na} and "
              f"{nb}, so the tail read is positions {hi} to {n})")
    print(f"  {tag} VERDICT: " + ("NOT HELD -- "
          f"{len(kseq)} of the {n_others} other pairs changed their lag draw count or "
          f"kind sequence and {len(jit)} changed their applied intervals, so the other "
          "pairs hold their lag VALUES still and not the events those draws serve"
          if kseq or jit else
          "on this run no other pair's draw count, kind sequence or applied interval "
          "moved"))
    return kseq, jit


def main():
    bad = 0
    print(f"{STAGE} p {P} seed {SEED}, every decision {BASE_ARM} s except the "
          f"{TARGET_NTH + 1}th {TARGET_KIND} of {TARGET[0]}->{TARGET[1]}, which becomes "
          f"{OTHER_ARM} s\n")

    on_base, ev_on_base, calls_on_base = run(False, True)
    on_chg, ev_on_chg, calls_on_chg = run(True, True)
    rows = differences(on_base.lags, on_chg.lags)
    starts_equal = on_base.starts == on_chg.starts
    others = [r for r in rows if r[0] != TARGET]
    moved = [r for r in others if r[3] is not None]
    report("streams on", rows, starts_equal)
    # 2026-11-23: the first-poll offsets are REPORTED by `report` above and no longer
    # gated here.  They were a conjunct of the PASS line below, which made part of that
    # line unable to fail: `start_lag` runs in the engine's initialisation loop before any
    # event or decider executes, so no decision can move an offset in ANY configuration,
    # and check 3's streams-off control does not control it either.  Gating on it dressed
    # a constant up as a verdict; stating it keeps the fact and drops the pretence.
    bad += say(not moved,
               f"streams on, SEEDING REGRESSION GUARD (Acceptance 3's wording): of the "
               f"{len(others)} pairs other than {TARGET[0]}->{TARGET[1]}, none has a lag "
               f"that moved over the common length of its two sequences. Position i of a "
               f"pair IS the i-th draw of pair_lag:<seed>:<node>:<nb>, whose seed string "
               f"no decision can touch, so this line is near-certain to pass on this "
               f"construction; it DOES fail if a draw is re-assigned between pairs, which "
               f"is what the per-node jitter stream does to the applied intervals, and "
               f"with sim_rng None it also says that every Mode B draw went through a "
               f"pair's or a node's own stream, since an unrouted draw would raise "
               f"AttributeError on the absent shared stream. "
               f"What a decision can move is reported as 1b below")
    if moved:
        print(f"  moved: {moved}")

    report_1b("1b", on_base, on_chg, ev_on_base, ev_on_chg, len(others))

    # 2026-11-23: the same measurement with the per-pair JITTER stream on as well, which
    # is the remedy 1b called for.  Printed beside 1b and not gated, for the same reason
    # 1b is not: Acceptance 3 asks for check 1.  The two verdicts side by side are what
    # says whether the remedy works on this run.
    pj_base, ev_pj_base, calls_pj_base = run(False, True, True)
    pj_chg, ev_pj_chg, calls_pj_chg = run(True, True, True)
    pj_rows = differences(pj_base.lags, pj_chg.lags)
    pj_others = [r for r in pj_rows if r[0] != TARGET]
    report("streams on, per-pair jitter on", pj_rows,
           pj_base.starts == pj_chg.starts)
    report_1b("1c", pj_base, pj_chg, ev_pj_base, ev_pj_chg, len(pj_others))

    target = next(r for r in rows if r[0] == TARGET)
    events_differ = flat(ev_on_base) != flat(ev_on_chg)
    # 2026-11-23: the disjunct `target[3] is not None` is gone.  `target[3]` is the first
    # position at which the TARGET pair's own lag VALUES differ, and with the per-pair lag
    # streams on that pair draws from the same seed string in both runs, so position i is
    # the i-th draw of one stream in both and the value cannot differ -- the same reason
    # check 1 is a tautology, one pair over.  It could never make this line pass, and
    # leaving it in made the line look like it had three independent ways to hold.
    # In its place, the positive statement the check actually needs: the arm really was
    # forced at the targeted decision, read off the emitted row rather than off the
    # decider's own bookkeeping.
    forced = [e["action"]["interval_s"] for e in ev_on_chg[TARGET[0]]
              if e.get("kind") == TARGET_KIND and e["action"]["neighbor"] == TARGET[1]]
    arm_forced = len(forced) > TARGET_NTH and forced[TARGET_NTH] == OTHER_ARM
    changed_something = (target[1] != target[2] or events_differ) and arm_forced
    bad += say(changed_something,
               f"the change is not vacuous: {TARGET[0]}->{TARGET[1]} drew {target[1]} "
               f"lags before and {target[2]} after, the event streams differ "
               f"({events_differ}), and the {TARGET_NTH + 1}th {TARGET_KIND} row of that "
               f"pair carries {forced[TARGET_NTH] if len(forced) > TARGET_NTH else None} "
               f"s, which is {OTHER_ARM} s: {arm_forced}")

    off_base, _ev, calls_off_base = run(False, False)
    off_chg, _ev2, calls_off_chg = run(True, False)
    off_rows = differences(off_base.lags, off_chg.lags)
    off_others = [r for r in off_rows if r[0] != TARGET]
    off_moved = [r for r in off_others if r[3] is not None]
    report("streams off", off_rows, off_base.starts == off_chg.starts)
    bad += say(bool(off_moved),
               f"negative control, streams off: {len(off_moved)} of the "
               f"{len(off_others)} other pairs have a lag that moved, so the guard above "
               "is the per-pair seeding doing the work and not a constant draw")

    # 2026-11-23: printed so that the provisional-timer count covers this script's runs
    # too.  The engine counts them and nothing here read the counter, which left the one
    # place the number is reported (weight.py) speaking only for its own simulations.
    # 2026-11-23: all six counts, not one.  This line used to print `on_base`'s count
    # alone as "N decider calls per run", which reads as a property of every run of the
    # check while three of the six runs were never looked at -- and the changed runs
    # legitimately differ, because the forced 10.043 s arm sends one poll fewer on the
    # probed pair, so they are printed rather than asserted equal.
    per_run = (("streams on, base", on_base, calls_on_base),
               ("streams on, changed", on_chg, calls_on_chg),
               ("jitter on, base", pj_base, calls_pj_base),
               ("jitter on, changed", pj_chg, calls_pj_chg),
               ("streams off, base", off_base, calls_off_base),
               ("streams off, changed", off_chg, calls_off_chg))
    print("\ndecider calls per run: "
          + ", ".join(f"{label} {calls}" for label, _eng, calls in per_run))
    # Read as three facts, not one: the two bases that share a jitter source agree; the
    # per-pair jitter base does not have to, because different jitter VALUES move the send
    # instants and the run then makes a different number of decisions; and a changed run
    # differs from its own base because the probed pair really does send a different
    # number of polls.  None of the three is a failure, and none of them was visible while
    # one count was printed as every run's.
    same_source = ("agree at %d" % calls_on_base
                   if calls_on_base == calls_off_base
                   else "DISAGREE: %d against %d" % (calls_on_base, calls_off_base))
    print("  the streams-on and streams-off bases draw from the same per-node jitter "
          f"stream and {same_source}"
          f"; the per-pair jitter base makes {calls_pj_base}, which differs by "
          f"{calls_pj_base - calls_on_base} because its jitter VALUES move the send "
          "instants and the run reaches a different number of decisions.")
    print("  each changed run against its own base: "
          + ", ".join("%s %+d" % (label, c - b) for label, b, c in
                      (("streams on", calls_on_base, calls_on_chg),
                       ("jitter on", calls_pj_base, calls_pj_chg),
                       ("streams off", calls_off_base, calls_off_chg)))
          + " -- the probed pair sending a different number of polls.")
    print("provisional timers that fired before their own poll had resolved: "
          + ", ".join(f"{label} {eng.provisional_fired}"
                      for label, eng, _calls in per_run))
    print(f"{'every check passed' if not bad else str(bad) + ' CHECK(S) FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
