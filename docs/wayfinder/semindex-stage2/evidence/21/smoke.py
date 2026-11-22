#!/usr/bin/env python3
"""The gate between `main11.sh smoke` and `main11.sh sweep` (2026-11-22).

    /home/lab_shared/cefore-semindex/.venv/bin/python3 -B smoke.py

Prints one PASS/FAIL line per check and exits non-zero on a failure. The sweep starts
only when every line says PASS.

WHAT IT CHECKS, on the two s11smoke runs (seed 101, p 0.2 and p 0.4)
  - the run is valid: `evalvalid`'s verdict in meta.json carries no flag;
  - meta.json records the knob and the three arms, so the run that is being read is a run
    the knob was actually on in;
  - every node wrote `state_poll_reschedule` rows, so the knob reached every agent and
    not only the one that happened to be looked at;
  - every such row records `action_prob` 1/3 and an interval out of the three arms;
  - every such row carries the field names the same node's `state_poll` rows carry,
    which is the ticket's demand that the re-decision log what the node could observe;
  - no (node, neighbour) pair wrote more reschedule rows than `state_poll` rows, which
    is what a second re-decision of one poll would look like in the log;
  - the next poll to a neighbour leaves when the re-decision said it would: the gap from
    a re-decided poll's send to the next send lies between the reschedule row's
    `applied_interval_s` and that plus SLACK_S. A row that is written and then not
    applied to the schedule would pass every check above, and would also leave the
    delay, the poll count and the share per arm looking right, because the ignored draw
    and the applied one come out of the same uniform stream.

WHAT IT DELIBERATELY DOES NOT PRINT
  No delay, no poll count, no row count. The sweep's own pass lines are the delay, the
  window poll count and the decision counts, and a number from a pilot run seen before
  the prediction is read is a number that can steer the reading of the prediction. What
  is printed here is a shape, never an outcome.

WHAT IT CANNOT SHOW
  Two runs at one seed say that the mechanism is wired up, not that the draw is uniform
  or that the delays are what the replay predicts. Those are pass lines 4 and 5, counted
  over the 200 runs after the sweep.

It reads meta.json and `*/semindex/events.jsonl` under `debug/node_dirs/` and opens no
other file: the node directories also hold Cefore key material.
"""
import json
import os
import sys

STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")
STAGE = "s11smoke"
SEED = 101
PS = (0.2, 0.4)
CHOICES = (3.695, 6.0, 10.043)
ACTION_PROB = 1.0 / len(CHOICES)
NODES = ("h0", "h1", "h2", "h3", "h4")
# How late after the re-decided instant the next poll may leave. The agent sends on its
# next loop turn, about 100 ms apart: on two runs of evidence/20's rule stages (s10f seed
# 121, s10t37r seed 140), which re-decide through the same code, the lateness ran from
# 0.0001 s to 0.1002 s over 360 re-decided polls. 0.5 s leaves room for a slow turn and is still far below the 1.3 s that separate
# the nearest two arms' applied intervals (3.695 * 1.1 and 6.0 * 0.9), so a schedule that
# followed the send-time draw instead cannot hide inside it.
SLACK_S = 0.5
# The stamps are whole milliseconds, so a poll that leaves exactly on time can read up to
# a millisecond or two early once both ends are rounded.
STAMP_S = 0.002


def pdir(p):
    """Directory spelling of a p value, as evalrun writes it: p0, p0.2, p0.4 ..."""
    return "p0" if p == 0.0 else "p" + ("%g" % p)


def read_run(p):
    """(meta.json, {node: [decision events]}) of one smoke run.

    Only the decision events are kept: every check below reads a `state_poll` or a
    `state_poll_reschedule` row, and the rest of the log is the bulk of the file.
    """
    run_dir = os.path.join(STORE, STAGE, pdir(p), str(SEED))
    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    rows = {}
    for node in NODES:
        path = os.path.join(run_dir, "debug", "node_dirs", node, "semindex", "events.jsonl")
        with open(path, encoding="utf-8") as fh:
            rows[node] = [e for e in (json.loads(line) for line in fh if line.strip())
                          if e.get("event") == "decision"
                          and e.get("kind") in ("state_poll", "state_poll_reschedule")]
    return meta, rows


def say(ok, text):
    """One check's line; returns 1 when it failed, so main can total the failures."""
    print(f"{'PASS' if ok else 'FAIL'} {text}")
    return 0 if ok else 1


def check_run(p):
    """The seven checks on one smoke run."""
    meta, rows = read_run(p)
    head = f"{STAGE} p {p} seed {SEED}"
    verdict = meta.get("validity") or {}
    flags = verdict.get("flags") or []
    bad = say(bool(verdict.get("valid")) and not flags,
              f"{head} validity: valid={verdict.get('valid')} flags={flags or 'none'}")
    bad += say(meta.get("state_poll_choice_redecide") is True
               and meta.get("state_poll_interval_choices_s") == list(CHOICES),
               f"{head} settings: meta records the knob and the three arms")

    resched = {n: [e for e in rows[n] if e["kind"] == "state_poll_reschedule"] for n in NODES}
    missing = [n for n in NODES if not resched[n]]
    bad += say(not missing,
               f"{head} reschedule rows: written by every node"
               + (f" -- MISSING on {missing}" if missing else ""))

    wrong = [(n, e["action"].get("interval_s"), e.get("action_prob"))
             for n in NODES for e in resched[n]
             if e.get("action_prob") != ACTION_PROB or e["action"].get("interval_s") not in CHOICES]
    bad += say(not wrong, f"{head} draw: every reschedule row has action_prob 1/3 and an "
                          f"interval out of {list(CHOICES)}" + (f" -- {wrong[:3]}" if wrong else ""))

    # The field NAMES, not the values: the point is that the learner reading these rows
    # finds the same columns it finds on the send-time row of the same node.
    field_bad = []
    for n in NODES:
        want = {frozenset(e["state"]) for e in rows[n] if e["kind"] == "state_poll"}
        got = {frozenset(e["state"]) for e in resched[n]}
        if len(want) != 1:
            # There is then no single send-time fieldset to subtract from, and what the
            # reader needs is that the node's own send rows disagree with each other.
            field_bad.append((n, f"{len(want)} distinct state_poll fieldsets"))
        elif got != want:
            w = next(iter(want))
            # One difference per distinct reschedule fieldset, never over their union: a
            # field carried by one row and missing from another cancels out of a union,
            # and the line would then name a failing node without naming the lost field.
            field_bad.append((n, [sorted(fs ^ w) for fs in got if fs != w] or "no rows"))
    bad += say(not field_bad, f"{head} state fields: every reschedule row carries the field "
                              "names that node's state_poll rows carry"
                              + (f" -- {field_bad}" if field_bad else ""))

    over = []
    for n in NODES:
        sent, again = {}, {}
        for e in rows[n]:
            side = sent if e["kind"] == "state_poll" else again
            nb = e["action"]["neighbor"]
            side[nb] = side.get(nb, 0) + 1
        # The pair only, without its two counts: this file promises to print no row
        # count at all, and a count out of a pilot run is a number that can steer the
        # reading of the prediction even when it arrives on a FAIL line. The pair is
        # what a reader needs to go and look at the log.
        over += [(n, nb) for nb in again if again[nb] > sent.get(nb, 0)]
    bad += say(not over, f"{head} per neighbour: no pair re-decided more polls than it sent"
                         + (f" -- {over}" if over else ""))

    # Timing, not counts: for every poll that was re-decided without a clamp, the next
    # poll to the same neighbour has to leave at the re-decided instant. A clamped row
    # moved the instant to "now" and says nothing about the applied interval, and the
    # last poll of a pair has no successor, so both are left out.
    late = []
    for n in NODES:
        sends, again = {}, {}
        for e in rows[n]:
            nb = e["action"]["neighbor"]
            if e["kind"] == "state_poll":
                sends.setdefault(nb, []).append(e["ts_ms"])
            elif not e["action"]["clamped"]:
                again[(nb, e["action"]["poll_epoch_ms"])] = e["action"]["applied_interval_s"]
        for nb, stamps in sends.items():
            for sent, nxt in zip(stamps, stamps[1:]):
                applied = again.get((nb, sent))
                if applied is not None and not -STAMP_S <= (nxt - sent) / 1000.0 - applied <= SLACK_S:
                    late.append((n, nb))
                    break
    bad += say(not late, f"{head} schedule: the poll after a re-decided one leaves at the "
                         f"re-decided instant, within {SLACK_S} s"
                         + (f" -- NOT on {late}" if late else ""))
    return bad


def main():
    bad = 0
    for p in PS:
        bad += check_run(p)
        print()
    print("every check passed; the sweep may start" if not bad
          else f"{bad} CHECK(S) FAILED; the sweep does not start")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
