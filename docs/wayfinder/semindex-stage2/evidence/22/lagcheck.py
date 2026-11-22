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
  1b. The quantity a decision CAN move, printed and not gated: each other pair's lag draw
     COUNT and draw-KIND sequence, and each other pair's per-send applied intervals.  On
     the run below they moved, so the other pairs do not hold still in that sense.  This
     is reported as a negative result, not as a pass line: Acceptance 3 asks for check 1,
     and making 1b the gate would redefine it.
  2. The change is not vacuous: the target pair's own run really did move (its decision
     count, its lag count or its event stream).
  3. The negative control: with the streams off, the same single change moves other
     pairs' lags.  Without this line check 1 would also pass on an engine that drew the
     same lag everywhere.

WHAT IT CANNOT SHOW
  That two candidates see the same lags.  A draw is still the n-th draw of its pair's
  stream, so a candidate that changes how many draws a pair makes shifts that pair's later
  lags by position -- and it is not only the target pair: the per-node jitter stream moves
  another pair's send instants, which is how `h2->h3` below draws 26 lags in one run and 27
  in the other.  Check 1b measures that; nothing here damps it.  Damping it by averaging
  over Mode B passes is the second half of this ticket's work and is not measured here.
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


def run(changed, pair_streams):
    """One simulation: (engine, events, how many decider calls it made)."""
    run_dict = dict(R.load_run(STAGE, P, SEED),
                    policy=None, rule=None, redecide=True, decider=one_change(changed))
    # With the streams on, `sim_rng` is deliberately left None: any Mode B draw site this
    # step failed to route through a pair's stream then raises instead of quietly mixing
    # the two sources.  With them off the shared stream is the one evidence/18 would use.
    seed_string = R.rep_seed(f"lagcheck:{STAGE}", P, SEED, 0)
    engine = Rec(run_dict, "B", None,
                 None if pair_streams else random.Random(seed_string),
                 None, seed_string if pair_streams else None)
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
    changed slice is, and the last field says whether the pair's values after the slice are
    its own again.
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
                    [round(x, 4) for x in b[lo:hi]], src, a[hi:n] == b[hi:n]))
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
    print(f"  {label}: per-node first-poll offsets "
          + ("all equal" if starts_equal else "NOT all equal"))


def main():
    bad = 0
    print(f"{STAGE} p {P} seed {SEED}, every decision {BASE_ARM} s except the "
          f"{TARGET_NTH + 1}th {TARGET_KIND} of {TARGET[0]}->{TARGET[1]}, which becomes "
          f"{OTHER_ARM} s\n")

    on_base, ev_on_base, calls = run(False, True)
    on_chg, ev_on_chg, _ = run(True, True)
    rows = differences(on_base.lags, on_chg.lags)
    starts_equal = on_base.starts == on_chg.starts
    others = [r for r in rows if r[0] != TARGET]
    moved = [r for r in others if r[3] is not None]
    report("streams on", rows, starts_equal)
    bad += say(not moved and starts_equal,
               f"streams on, SEEDING REGRESSION GUARD (Acceptance 3's wording, and it "
               f"cannot fail): of the {len(others)} pairs other than "
               f"{TARGET[0]}->{TARGET[1]}, none has a lag that moved over the common "
               f"length of its two sequences, and no node's first-poll offset moved. "
               f"Position i of a pair IS the i-th draw of pair_lag:<seed>:<node>:<nb>, "
               f"whose seed string no decision can touch, so no decider and no change "
               f"could make this line fail; with sim_rng None it does say that every "
               f"Mode B draw went through a pair's or a node's own stream, since an "
               f"unrouted draw would raise AttributeError on the absent shared stream. "
               f"What a decision can move is reported as 1b below")
    if moved:
        print(f"  moved: {moved}")

    # 1b.  What the lag streams do NOT align, which is the residue the second half has to
    # damp: `Node.jitter_rng` is per NODE and is drawn once per send over BOTH of that
    # node's neighbours, in event order.  Moving one pair's sends therefore re-assigns
    # that node's later jitter draws between its two pairs, which moves the other pair's
    # applied intervals although every lag it drew is the same value at the same position,
    # and a send instant shifted that way can land on the other side of a `link_down` and
    # add a poll-timeout lag draw.  Printed, not gated: Acceptance 3 asks for check 1, and
    # a per-pair jitter stream would change what every existing stage does.
    kseq = seq_moved(on_base.kindseq, on_chg.kindseq, TARGET)
    jit = [r for r in differences(applied(ev_on_base), applied(ev_on_chg))
           if r[0] != TARGET and r[3] is not None]
    print("  1b: of the other pairs, those whose lag DRAW COUNT or draw-KIND sequence "
          "moved (pair, base n, changed n, first differing kind index): "
          + (", ".join(f"{k[0]}->{k[1]} {na} {nb} "
                       f"{'-' if first is None else first}" for k, na, nb, first in kseq)
             if kseq else "none"))
    moved_kinds = {k: (on_base.kinds.get(k, 0), on_chg.kinds.get(k, 0))
                   for k in sorted(set(on_base.kinds) | set(on_chg.kinds))
                   if on_base.kinds.get(k, 0) != on_chg.kinds.get(k, 0)}
    print("  1b: draw counts that moved, by (node, neighbour, draw kind): "
          + (str(moved_kinds) if moved_kinds else "none"))
    print("  1b: of the other pairs, those whose per-send APPLIED INTERVALS moved "
          "(pair, first differing send): "
          + (", ".join(f"{k[0]}->{k[1]} {first}" for k, _a, _b, first in jit)
             if jit else "none"))
    print("  1b: the moved slice of every pair, and whose values it carries (pair, "
          "positions, base, changed, the pair of the same node whose base values the "
          "changed slice is, own values again afterwards)")
    for key, lo, hi, a, b, src, back in carried(applied(ev_on_base), applied(ev_on_chg)):
        names = ", ".join(f"{s[0]}->{s[1]}" for s in src) or "none"
        print(f"    {key[0]}->{key[1]}  [{lo}:{hi}]  base {a}  changed {b}  "
              f"from {names}  own again: {back}")
    print("  1b VERDICT: " + ("NOT HELD -- "
          f"{len(kseq)} of the {len(others)} other pairs changed their lag draw count or "
          f"kind sequence and {len(jit)} changed their applied intervals, so the other "
          "pairs hold their lag VALUES still and not the events those draws serve"
          if kseq or jit else
          "on this run no other pair's draw count, kind sequence or applied interval "
          "moved"))

    target = next(r for r in rows if r[0] == TARGET)
    events_differ = flat(ev_on_base) != flat(ev_on_chg)
    changed_something = (target[1] != target[2] or target[3] is not None or events_differ)
    bad += say(changed_something,
               f"the change is not vacuous: {TARGET[0]}->{TARGET[1]} drew {target[1]} "
               f"lags before and {target[2]} after, its first moved lag is at "
               f"{target[3]}, and the event streams differ: {events_differ}")

    off_base, _ev, _c = run(False, False)
    off_chg, _ev2, _c2 = run(True, False)
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
    print(f"\n{calls} decider calls per run")
    print("provisional timers that fired before their own poll had resolved: "
          + ", ".join(f"{label} {eng.provisional_fired}" for label, eng in
                      (("streams on, base", on_base), ("streams on, changed", on_chg),
                       ("streams off, base", off_base), ("streams off, changed", off_chg))))
    print(f"{'every check passed' if not bad else str(bad) + ' CHECK(S) FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
