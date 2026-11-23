#!/usr/bin/env python3
"""The VALUES of the state the port hands a decider, against the source's own objects.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B valuecheck.py

Prints one PASS/FAIL line per check and exits non-zero if any failed.

WHY IT EXISTS
  `statecheck.py` compares KEY SETS and TYPE NAMES, so a wrong number in any of the six
  fields passes every line of it.  This file is the value-level companion: it rebuilds the
  source's own `Tracker` and `_PollReplies` at every decision and requires the port's
  numbers to be theirs.

THE RUN IT USES
  A roll-in run of the labelling construction: every decision a uniform draw over the
  three arms, on the labelling run dict of p 0.2 seed 1.  Not a constant decider, because
  a constant arm never mixes intervals and the three statuses then do not all occur; the
  roll-in reaches poll timeouts and a different interval per decision, which is the run
  the labels are actually made on.

WHAT IT CHECKS
  1. Every origin row (`hi`, `missing`, the per-advertiser `hi` and `notified_max`) and
     `truncated`, against a real `Tracker` rebuilt at the decision instant from the
     replay node's own raw store.  `missing` is the trap: it is `missing_total`, the
     UNCAPPED backlog, and a port that reported the fetch-planning list would differ here.
  2. `last_poll_status` and `consecutive_poll_timeouts`, against a real `_PollReplies`
     driven by the send and resolution instants of the port's own rows, with the Interest
     lifetime the replay uses.

WHAT IT CANNOT SHOW
  That `in_flight` and `last_timeout_s` are right.  Neither has a reference here: the
  rebuilt `Tracker` is loaded from the store and has planned no fetch, so its
  `in_flight_count` and `last_timeout_at` are its defaults, and comparing them would
  compare the port against a fixture rather than against the source.  Both are at their
  default in every call of this run, which the situation report below states as what it
  is -- an unreached situation, not a verified one.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label as L  # noqa: E402
import port as PORT  # noqa: E402

from semindex.engine import _PollReplies  # noqa: E402
from semindex.tracker import Tracker  # noqa: E402

R = PORT.R

P, SEED = 0.2, PORT.check_seeds([1], "valuecheck.py")[0]
LIFETIME_S = R.POLL_TIMEOUT_MS / 1000.0   # the Interest lifetime the replay simulates
SHOW = 3                                  # how many mismatches are printed


def say(ok, text):
    print(f"{'PASS' if ok else 'FAIL'} {text}", flush=True)
    return 0 if ok else 1


def tracker_of(node):
    """A real `Tracker` loaded with one replay node's raw store.

    Only the two observation entry points the source itself uses are called, so the
    tracker computes `hi`, `missing_total` and the advertised rows with its own code:
    `observe_held` for this node's own progress and `observe_state` for one advertiser's.
    """
    tracker = Tracker()
    for origin, state in node.origins.items():
        extra = sorted(state.extra)
        if state.hi or extra:
            tracker.observe_held(origin, state.hi, extra)
        for other, (hi, ex) in state.advertised.items():
            tracker.observe_state(origin, other, hi, list(ex))
    return tracker


def rollin(p, seed):
    """One roll-in run, capturing (where, state) at every decision: (rows, events).

    The capture wraps the labelling roll-in decider rather than replacing it, so that the
    run this file checks is the run `label.py` makes and not a second construction of it.
    """
    run = L.run_dict(p, seed)
    tag = L.stream_label_of(p, seed)
    inner = L.Labeller(tag, None, None, None)
    captured = []
    engine = {}

    def decide(obs):
        node = engine["e"].nodes[obs["where"]["node"]]
        captured.append((dict(obs["where"]), obs["state"],
                         tracker_of(node).decision_state(obs["where"]["neighbor"])))
        return inner(obs)

    eng = PORT.Engine(dict(run, decider=decide), "B", None, None, None, tag, tag)
    engine["e"] = eng
    inner.engine = eng
    return captured, eng.go(), eng


def check_origins(captured):
    """Check 1: the origin rows and `truncated` are a real Tracker's."""
    off = []
    cells = sum(len(got["origins"]) for _w, got, _ref in captured)
    for where, got, ref in captured:
        if (got["origins"] != ref["origins"]
                or got["truncated"] != ref["truncated"]):
            off.append((where, got["origins"], ref["origins"],
                        got["truncated"], ref["truncated"]))
    bad = say(not off,
              f"all {len(captured)} decisions, {cells} origin rows: `hi`, `missing`, the "
              "per-advertiser `hi` and `notified_max` and `truncated` are what a real "
              f"Tracker reports from the same store ({len(off)} differing)")
    for row in off[:SHOW]:
        print(f"  {row}")
    return bad


def check_polls(events):
    """Check 2: the two poll fields are a real `_PollReplies`'s.

    The reference is driven by the rows the port emitted, in each node's own row order:
    a send calls `sent`, an answered resolution calls `replied`, and a timed-out
    resolution marks that poll resolved, which is what the source's own timeout path
    does.  The snapshot is then taken at the row's own stamp, which is where the source
    takes it (after a resolution, before a send).
    """
    off, compared = [], 0
    reached = {}
    for node, rows in events.items():
        per_nb = {}
        for e in rows:
            if e.get("event") != "decision":
                continue
            kind = e.get("kind")
            if kind not in ("state_poll", "state_poll_reschedule"):
                continue
            nb = e["action"]["neighbor"]
            ref = per_nb.setdefault(nb, _PollReplies())
            now = e["ts_ms"] / 1000.0
            if kind == "state_poll_reschedule":
                epoch = e["action"]["poll_epoch_ms"]
                if e["action"]["trigger"] == "answered":
                    ref.replied(epoch, now, LIFETIME_S)
                else:
                    for poll in ref.polls:
                        if poll.epoch_ms == epoch:
                            poll.resolved = e["action"]["trigger"]
            status, consec = ref.snapshot(now, LIFETIME_S)
            reached[(kind, status)] = reached.get((kind, status), 0) + 1
            compared += 1
            if (status != e["state"]["last_poll_status"]
                    or consec != e["state"]["consecutive_poll_timeouts"]):
                off.append((node, nb, kind, e["ts_ms"], (status, consec),
                            (e["state"]["last_poll_status"],
                             e["state"]["consecutive_poll_timeouts"])))
            if kind == "state_poll":
                ref.sent(e["ts_ms"], now)
    bad = say(not off,
              f"all {compared} rows: `last_poll_status` and "
              "`consecutive_poll_timeouts` are what a real _PollReplies reports from the "
              f"same send and resolution instants ({len(off)} differing)")
    for row in off[:SHOW]:
        print(f"  {row}")
    print("  the reference's own (row kind, status) distribution: "
          + ", ".join(f"{k[0]} {k[1]} {v}" for k, v in sorted(reached.items(), key=str)))
    return bad


def situations(captured):
    """Which of the situations the state can express this run never entered.

    Printed and not gated.  Each line is a branch of the state builder that carries no
    value-level coverage from this file, so that nothing above reads as wider than it is.
    """
    calls = len(captured)
    in_flight = sum(1 for _w, got, _r in captured if got["in_flight"] > 0)
    truncated = sum(1 for _w, got, _r in captured if got["truncated"])
    timeout = sum(1 for _w, got, _r in captured
                  if got["last_timeout_s"] is not None)
    pending = sum(1 for _w, got, _r in captured
                  if got["last_poll_status"] == "pending")
    cells = above = 0
    for _w, got, _r in captured:
        for row in got["origins"].values():
            for adv in row["advertised"].values():
                cells += 1
                above += adv["notified_max"] > adv["hi"]
    print(f"\nSituations this run did not reach, over {calls} decisions:")
    print(f"  in_flight > 0            : {in_flight} calls"
          + ("" if in_flight else "   NOT REACHED"))
    print(f"  truncated True           : {truncated} calls"
          + ("" if truncated else "   NOT REACHED"))
    print(f"  last_timeout_s set       : {timeout} calls"
          + ("" if timeout else "   NOT REACHED"))
    print(f"  notified_max > hi        : {above} of {cells} advertised cells"
          + ("" if above else "   NOT REACHED"))
    print("  last_poll_status pending : "
          f"{pending} calls" + ("" if pending else "   NOT REACHED"))
    print("  A situation marked NOT REACHED has no value-level coverage from this file at "
          "all: the branch that builds it was never taken, so its arithmetic is covered "
          "for its SHAPE only, by statecheck.py's Tracker fixture.")


def main():
    captured, events, engine = rollin(P, SEED)
    print(f"a roll-in run of the labelling construction, p {P} seed {SEED}: "
          f"{len(captured)} decider calls "
          f"({engine.decider_ord['state_poll']} sends, "
          f"{engine.decider_ord['state_poll_reschedule']} resolutions)")
    print("  provisional timers that fired before their own poll had resolved: "
          f"{engine.provisional_fired}\n")
    bad = check_origins(captured)
    bad += check_polls(events)
    situations(captured)
    print(f"\n{'every check passed' if not bad else str(bad) + ' CHECK(S) FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
