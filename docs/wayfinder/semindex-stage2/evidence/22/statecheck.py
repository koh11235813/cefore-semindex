#!/usr/bin/env python3
"""Acceptance 2: the state the port hands a decider is a real decision row's state.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B statecheck.py

Prints one PASS/FAIL line per check and exits non-zero if any failed.

WHAT IT CHECKS
  1. Key sets, strictly, between three sources of one shape: the state the port built for
     every decider call of one simulated run; the state a real `Tracker.decision_state`
     plus the two poll fields of `_PollReplies.snapshot` returns, both imported from this
     worktree's `src`; and the `state` of the logged `state_poll` and
     `state_poll_reschedule` rows of one `s11a` run, which is a stage whose rows carry
     the full state.  Nested: the variable levels (which origin, which advertiser) are
     compared as one wildcard each, every other level by name.  One extra or one missing
     key anywhere fails, in either direction.
  2. Value types per key, each source against a type set declared here from the source's
     own definitions.  A type outside it fails.  The three sources do not all exhibit
     every allowed type -- `last_timeout_s` is null in every logged row of the file this
     check may open -- so the per-source sets are printed rather than compared to each
     other.
  3. The history element's keys against the logged `state_poll_reschedule` action plus
     `ts_ms`.  `neighbor` is the one action field the element drops, because `where`
     already carries it and a learner may not read `where`; the check names that drop
     explicitly and fails on any other difference.
  4. The emitted row carries the state the decider was handed, for both kinds, so that a
     tool reading these runs reads the decider's own input.
  5. The state BUILDER reads no neighbour history.  Every `_Origin.hist` of every node is
     replaced by an object that raises on any access, the state is built again, and it has
     to come out equal to what the decider was handed.  The builder's source is inspected
     for the same thing, so that a future edit that reaches for `hist` or for another node
     fails here even if no run of this check happens to touch it.

WHAT IT CANNOT SHOW
  That the VALUES agree.  Checks 1 and 2 compare path sets and type-name sets, and checks
  4 and 5 compare the port against itself, so no check here looks at a number.  A wrong
  value in any of the six fields would pass every line below.  The replay's state is built
  from the replay's own store and schedule; the one value-level statement anywhere is
  evidence/13's validation of the replay against the testbed, which reads no decision
  state.  It also cannot show that `history` mirrors a real field: the source keeps no
  bounded exchange history at all, so the element's keys are matched against a row's
  fields and nothing more.

  That nothing hist-derived reaches the state.  Check 5 is about the BUILDER, not about
  the content: `origins[o].advertised[nb].hi` IS computed from that neighbour's
  `_Origin.hist`, at REPLY time in `ev_rx` (../13/replay.py:897, :909, through
  `progress_map_asof`), and copied into the deciding node's own dict there.  Poisoning
  the histories at the decision instant therefore cannot see it, and the rebuild is
  trivially equal.  That content is legitimate: a real state reply carries (hi, extra) per
  origin and the source feeds exactly that to its tracker (`src/semindex/engine.py:842`),
  which is the argument for the field -- not this check.  What check 5 does buy is that a
  FUTURE edit cannot reach into a neighbour's store at decision time.

  That the situations this cell does not reach are built right.  In the `s11a` p 0.2
  seed 1 run, `in_flight` is 0 at every call, `truncated` is False at every call (the five
  nodes see at most two origins each, against a cap of eight), `notified_max` never
  exceeds `hi` (no reply of this stage carries a sparse extra), `last_timeout_s` is null
  throughout and `last_poll_status` is never "pending".  The cap, the truncation flag and
  the sparse-extra arithmetic are exercised here only against the `Tracker` fixture of
  `tracker_states`, and only for their SHAPE.
"""
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import port as PORT  # noqa: E402

from semindex.engine import _PollReplies  # noqa: E402
from semindex.tracker import Tracker  # noqa: E402

R = PORT.R

STAGE, P, SEED = "s11a", 0.2, PORT.check_seeds([1], "statecheck.py")[0]
DECIDED = 6.0                           # one of the s11a arms; the shape is what matters

# The levels where the key is data (an origin name, a neighbour name) and not a field.
# Everything else is compared by name.
WILDCARDS = {("origins",), ("origins", "*", "advertised")}

# Each field's allowed Python types, read off the source: tracker.py:303-308 and
# :291-302 for the four tracker fields, engine.py:300-307 (_status) and :284-298 for the
# two poll fields.  `float | None` for last_timeout_s is tracker.py:251; `str | None` for
# last_poll_status is the same signature.
STATE_TYPES = {
    ("in_flight",): {"int"},
    ("last_timeout_s",): {"float", "NoneType"},
    ("truncated",): {"bool"},
    ("origins",): {"dict"},
    ("origins", "*"): {"dict"},
    ("origins", "*", "hi"): {"int"},
    ("origins", "*", "missing"): {"int"},
    ("origins", "*", "advertised"): {"dict"},
    ("origins", "*", "advertised", "*"): {"dict"},
    ("origins", "*", "advertised", "*", "hi"): {"int"},
    ("origins", "*", "advertised", "*", "notified_max"): {"int"},
    ("last_poll_status",): {"str", "NoneType"},
    ("consecutive_poll_timeouts",): {"int"},
}

# The history element, from the fields of a logged `state_poll_reschedule` row:
# engine.py:692-707 for the action, engine.py:432 for the stamp.
HISTORY_TYPES = {
    ("ts_ms",): {"int"},
    ("poll_epoch_ms",): {"int"},
    ("trigger",): {"str"},
    ("empty",): {"bool", "NoneType"},
    ("interval_s",): {"float"},
    ("applied_interval_s",): {"float"},
    ("clamped",): {"bool"},
}

# The one action field the history element does not keep.
HISTORY_DROPS = {"neighbor"}


class Poison:
    """Anything a reader of an `_Origin.hist` would do with it raises.

    Assigned over every node's every origin history while the state is rebuilt, so that a
    builder that reached for a neighbour's store history could not finish.  `_Origin` has
    __slots__, so the attribute takes any object.
    """

    def _blow(self, *_a, **_k):
        raise AssertionError("the state builder read an _Origin.hist")

    __iter__ = __len__ = __getitem__ = __bool__ = __contains__ = _blow


def walk(obj, path=()):
    """{path: type name} over a nested dict, data-keyed levels collapsed to '*'."""
    out = {path: type(obj).__name__} if path else {}
    if isinstance(obj, dict):
        wild = path in WILDCARDS
        for key, value in obj.items():
            out.update(walk(value, path + ("*" if wild else key,)))
    return out


def merge(samples):
    """{path: set of type names} over many samples of one shape."""
    out = {}
    for sample in samples:
        for path, name in walk(sample).items():
            out.setdefault(path, set()).add(name)
    return out


def dotted(path):
    return ".".join(path)


def say(ok, text):
    print(f"{'PASS' if ok else 'FAIL'} {text}", flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# The three sources
# ---------------------------------------------------------------------------

def tracker_states():
    """What a real `Tracker` plus a real `_PollReplies` report, over a few situations.

    Built so that every allowed type appears at least once: an empty tracker (no origin,
    no timeout), and a loaded one with more origins than the cap (truncated), two
    advertisers per origin, sparse extras above the advertised `hi`, a fetch in flight and
    one resolved fetch timeout.  The four poll situations are the four statuses
    `_PollReplies._status` can report, including the None before the first poll.
    """
    out = []
    lifetime_s = 2.0
    empty = Tracker()
    loaded = Tracker()
    # Nine origins, one more than DECISION_STATE_MAX_ORIGINS, so `truncated` is True and
    # the row shows the cap doing its work.
    for i in range(9):
        origin = f"h{i}"
        loaded.observe_held(origin, 3, [7])
        loaded.observe_state(origin, "h1", 5, [9])
        loaded.observe_state(origin, "h2", 4, ())
    loaded.plan_fetches("h1", 1000.0)
    loaded.on_fetch_timeout("h0", 4, "h1", 1002.0)
    now = 1002.0
    situations = []
    fresh = _PollReplies()
    situations.append(fresh.snapshot(now, lifetime_s))             # (None, 0)
    pending = _PollReplies()
    pending.sent(1, now)
    situations.append(pending.snapshot(now, lifetime_s))           # ("pending", 0)
    timed = _PollReplies()
    timed.sent(2, now - 3.0)
    situations.append(timed.snapshot(now, lifetime_s))             # ("timed_out", 1)
    answered = _PollReplies()
    answered.sent(3, now - 0.5)
    answered.replied(3, now, lifetime_s)
    situations.append(answered.snapshot(now, lifetime_s))          # ("answered", 0)
    for tracker in (empty, loaded):
        for last_poll_status, consecutive in situations:
            out.append({**tracker.decision_state("h1"),
                        "last_poll_status": last_poll_status,
                        "consecutive_poll_timeouts": consecutive})
    return out


def logged_states(run):
    """The `state` of every logged `state_poll` and `state_poll_reschedule` row, per kind."""
    out = {"state_poll": [], "state_poll_reschedule": []}
    actions = {"state_poll_reschedule": []}
    for rows in run["events"].values():
        for e in rows:
            if e.get("event") != "decision":
                continue
            kind = e.get("kind")
            if kind in out:
                out[kind].append(e["state"])
            if kind in actions:
                actions[kind].append(e["action"])
    return out, actions["state_poll_reschedule"]


def port_run(run):
    """One simulated run with a capturing decider: (obs list, emitted rows, engine).

    The decider also rebuilds the state with every history poisoned and keeps both, so
    check 5 reads a rebuild made at the real decision instant rather than at the end of
    the run, where the store no longer looks the way it did.
    """
    captured = []

    def decide(obs):
        engine = holder[0]
        node, nb = obs["where"]["node"], obs["where"]["neighbor"]
        saved = [(o, o.hist) for nd in engine.nodes.values() for o in nd.origins.values()]
        for origin_state, _h in saved:
            origin_state.hist = Poison()
        try:
            rebuilt = engine.nodes[node].decision_state(nb)
        finally:
            for origin_state, h in saved:
                origin_state.hist = h
        captured.append((obs, rebuilt))
        return DECIDED

    run = dict(run, policy=None, rule=None, redecide=True, decider=decide)
    holder = [None]
    engine = PORT.Engine(run, "B", None, None, None,
                         PORT.R.rep_seed(f"statecheck:{STAGE}", P, SEED, 0))
    holder[0] = engine
    events = engine.go()
    return captured, events, engine


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def check_keys(port_shape, tracker_shape, logged_shapes):
    """Check 1: the path sets of the three sources, strictly."""
    bad = 0
    for label, other in [("a real Tracker plus _PollReplies", tracker_shape)] + [
            (f"the logged {kind} rows", shape) for kind, shape in logged_shapes.items()]:
        extra = sorted(dotted(p) for p in set(port_shape) - set(other))
        missing = sorted(dotted(p) for p in set(other) - set(port_shape))
        bad += say(not extra and not missing,
                   f"key sets equal, the port against {label}: {len(port_shape)} paths")
        if extra:
            print(f"  only the port has: {extra}")
        if missing:
            print(f"  only {label} has: {missing}")
    return bad


def check_types(shapes):
    """Check 2: every observed type of every source inside the declared set."""
    bad = 0
    for label, shape, declared in shapes:
        off = {dotted(p): (sorted(ts), sorted(declared.get(p, set())))
               for p, ts in shape.items() if not ts <= declared.get(p, set())}
        bad += say(not off, f"value types of {label} inside the declared set")
        if off:
            print(f"  outside it: {off}")
    for label, shape, _d in shapes:
        print("  " + label + ": "
              + ", ".join(f"{dotted(p)}={'|'.join(sorted(t))}"
                          for p, t in sorted(shape.items())))
    return bad


def check_history(captured, logged_actions):
    """Check 3: the history element's keys against a logged reschedule action plus ts_ms."""
    seen = {frozenset(row) for obs, _r in captured for row in obs["history"]}
    action_keys = {frozenset(a) for a in logged_actions}
    bad = say(len(action_keys) == 1,
              f"the logged {len(logged_actions)} reschedule rows carry one action fieldset")
    if len(action_keys) != 1:
        return bad
    want = (set(next(iter(action_keys))) - HISTORY_DROPS) | {"ts_ms"}
    bad += say(len(seen) == 1 and set(next(iter(seen))) == want,
               f"every history element carries exactly {sorted(want)}: the logged action "
               f"fields less {sorted(HISTORY_DROPS)}, plus the row's own stamp")
    if not (len(seen) == 1 and set(next(iter(seen))) == want):
        print(f"  the port's history elements: {[sorted(s) for s in seen]}")
        print(f"  wanted: {sorted(want)}")
    longest = max((len(obs["history"]) for obs, _r in captured), default=0)
    bad += say(longest <= R.DECIDER_HISTORY,
               f"no history longer than {R.DECIDER_HISTORY}: the longest was {longest}")
    return bad


def check_rows(captured, events):
    """Check 4: the emitted row of each decision carries the state the decider saw."""
    rows = {}
    for node, evs in events.items():
        for e in evs:
            if e.get("event") != "decision" or e.get("kind") not in (
                    "state_poll", "state_poll_reschedule"):
                continue
            key = (node, e["action"]["neighbor"], e["kind"], e["ts_ms"])
            rows.setdefault(key, []).append(e["state"])
    missing, differ = 0, 0
    for obs, _r in captured:
        w = obs["where"]
        key = (w["node"], w["neighbor"], w["kind"], w["ts_ms"])
        got = rows.get(key)
        if not got:
            missing += 1
        elif not any(g == obs["state"] for g in got):
            differ += 1
    return say(not missing and not differ,
               f"all {len(captured)} decisions: the emitted row's state equals the state "
               f"the decider was handed ({missing} with no row, {differ} differing)")


def check_no_hist(captured):
    """Check 5: the poisoned rebuild, and the builder's own source.

    What this proves and what it does not: the rebuild says the BUILDER does not read a
    history at the decision instant, which is what makes a future edit that reaches for
    one fail here.  It does NOT say that nothing hist-derived is in the state -- the
    advertised `hi` of every origin is computed from the advertiser's `_Origin.hist` in
    `ev_rx` and copied into this node's own dict at reply time, so poisoning at the
    decision instant cannot see it and the rebuild is equal by construction.  That
    content is a real reply's content (see the module docstring).
    """
    bad = say(all(rebuilt == obs["state"] for obs, rebuilt in captured),
              f"all {len(captured)} decisions: the state the BUILDER produces with every "
              "_Origin.hist poisoned is the state the decider was handed, so the builder "
              "reads no history at the decision instant (reply-time content is not "
              "covered by this; see the module docstring)")
    src = inspect.getsource(R.Node.decision_state)
    body = "\n".join(line for line in src.splitlines()
                     if not line.lstrip().startswith("#"))
    body = body.split('"""')[-1] if body.count('"""') >= 2 else body
    reaches = [name for name in ("hist", "self.nodes", "progress_map_asof", "extras_asof")
               if name in body]
    bad += say(not reaches,
               "the builder's source names no simulator-only reader (checked for hist, "
               "self.nodes, progress_map_asof, extras_asof)")
    if reaches:
        print(f"  it names: {reaches}")
    return bad


def main():
    run = R.load_run(STAGE, P, SEED)
    captured, events, engine = port_run(run)
    logged, logged_actions = logged_states(run)
    port_shape = merge([obs["state"] for obs, _r in captured])
    tracker_shape = merge(tracker_states())
    logged_shapes = {k: merge(v) for k, v in logged.items()}
    print(f"{STAGE} p {P} seed {SEED}: {len(captured)} decider calls simulated, "
          f"{len(logged['state_poll'])} logged state_poll rows and "
          f"{len(logged['state_poll_reschedule'])} logged reschedule rows read")
    # Why `last_timeout_s` shows one type below where the declared set has two: it is the
    # instant of a FETCH timeout, and a fetch is only planned after a reply arrived, so
    # the link has to drop inside the few milliseconds between the plan and the fetch's
    # arrival for one to happen at all.  Counted here rather than asserted, because the
    # field's shape is what this check is about and its emptiness is a property of the
    # cell, not of the port.  The float branch is reached only by the Tracker fixture.
    sim_timeouts = sum(1 for evs in events.values() for e in evs
                       if e.get("event") == "fetch_timeout")
    log_timeouts = sum(1 for evs in run["events"].values() for e in evs
                       if e.get("event") == "fetch_timeout")
    print(f"  fetch timeouts: {sim_timeouts} in the simulated run, {log_timeouts} in the "
          "logged one, so last_timeout_s stays null in both")
    # 2026-11-23: printed so that the provisional-timer count covers this script's run
    # too.  The engine counts them and nothing here read the counter, which left the one
    # place the number is reported (weight.py) speaking only for its own simulations.
    print("  provisional timers that fired before their own poll had resolved: "
          f"{engine.provisional_fired}\n")
    bad = check_keys(port_shape, tracker_shape, logged_shapes)
    bad += check_types(
        [("the port", port_shape, STATE_TYPES),
         ("a real Tracker plus _PollReplies", tracker_shape, STATE_TYPES)]
        + [(f"the logged {k} rows", s, STATE_TYPES) for k, s in logged_shapes.items()]
        + [("the port's history elements",
            merge([row for obs, _r in captured for row in obs["history"]]),
            HISTORY_TYPES)])
    bad += check_history(captured, logged_actions)
    bad += check_rows(captured, events)
    bad += check_no_hist(captured)
    print(f"\n{'every check passed' if not bad else str(bad) + ' CHECK(S) FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
