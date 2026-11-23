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
     key anywhere fails, in either direction.  **What it compares is the UNION of the
     paths of each source's samples**, which is why 1b exists.
  1b. 2026-11-23: every INDIVIDUAL sample, one decision kind at a time, against the key
     set declared for its level -- six keys on the state, exactly `hi`, `missing`,
     `advertised` on every origin row and exactly `hi`, `notified_max` on every advertiser
     row -- and every value type inside the declared set for its path.  Check 1 unions the
     paths of all samples and lumps the two decision kinds into one shape, so a key missing
     from every `state_poll` call, or from 347 of 348 calls, or from one `origins[o]` row
     while another kept it, left the union complete and every line of this file printed
     PASS.  1b is the line that fails on that.
  2. Value types per key, each source against a type set declared here from the source's
     own definitions.  A type outside it fails.  The three sources do not all exhibit
     every allowed type -- `last_timeout_s` is null in every logged row of the file this
     check may open -- so the per-source sets are printed rather than compared to each
     other.
  3. The history element's keys against the logged `state_poll_reschedule` action plus
     `ts_ms`.  `neighbor` is the one action field the element drops, because `where`
     carries it once per call and every element of one history is the same pair's, so
     repeating it eight times would carry nothing; the check names that drop explicitly
     and fails on any other difference.
  4. The emitted row carries the state the decider was handed, for both kinds, so that a
     tool reading these runs reads the decider's own input.
  4b. `obs["resolved"]`: its keys against the logged `state_poll_reschedule` action, with
     the dropped fields named, and -- unlike every other line here -- its VALUES against
     the action of the very row the decision wrote.  At a send it has to be None, because
     no poll is being resolved there.
  5. The state BUILDER reads no neighbour history.  Every `_Origin.hist` of every node is
     replaced by an object that raises on any access, the state is built again, and it has
     to come out equal to what the decider was handed.  The builder's source is inspected
     for the same thing, so that a future edit that reaches for `hist` or for another node
     fails here even if no run of this check happens to touch it.  2026-11-23: the
     inspection reads the `ast` of BOTH `Node.decision_state` AND `_Engine.ask_decider`,
     which is the function that assembles `obs` and the one that holds `self.nodes`,
     `self.timeline`, `self.run` and `self.heap`; it had never been looked at.  `ast` and
     not a substring search, because `nd.poll_hist` contains "hist" and a substring check
     cannot tell that attribute from the forbidden one.  And `obs` itself and
     `obs["where"]` are held to their exact key sets on every call: nothing did that
     before, so an edit that added a leak key to `obs` passed this file.

WHAT IT CANNOT SHOW
  That the VALUES of the STATE agree.  Checks 1 and 2 compare path sets and type-name
  sets, and checks 4 and 5 compare the port against itself, so no line about the six state
  fields looks at a number: a wrong `missing`, a wrong `notified_max` or a
  `last_timeout_s` in milliseconds would pass every one of them.  `valuecheck.py` is the
  file that compares those numbers against a real `Tracker` and a real `_PollReplies`.
  The one value-level line here is check 4b's, and it is about `resolved` only.
  It also cannot show that `history` mirrors a real field: the source keeps no bounded
  exchange history at all, so the element's keys are matched against a row's fields and
  nothing more.

  That nothing hist-derived reaches the state.  Check 5 is about the BUILDER, not about
  the content: `origins[o].advertised[nb].hi` IS computed from that neighbour's
  `_Origin.hist`, at REPLY time in `ev_rx` (../13/replay.py:927, :939, through
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
import ast
import inspect
import os
import sys
import textwrap

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

# 2026-11-23: the key set of each LEVEL, so that check 1b can hold one sample to it
# instead of holding a union of samples to a union of paths.  Read off the source:
# tracker.py:291-308 for the four tracker fields and the two nested levels, engine.py:
# 529-533 at a send and :680-684 at a resolution for the two poll fields.
STATE_KEYS = frozenset({"in_flight", "last_timeout_s", "truncated", "origins",
                        "last_poll_status", "consecutive_poll_timeouts"})
ORIGIN_KEYS = frozenset({"hi", "missing", "advertised"})
ADVERTISED_KEYS = frozenset({"hi", "notified_max"})

# What `ask_decider` hands over (../13/replay.py:809-880), and what `where` holds.  A key
# beyond these is a leak: `obs` is meant to be everything a real node could read plus the
# one dict a TOOL may read, and nothing else.
OBS_KEYS = frozenset({"state", "history", "resolved", "where"})
WHERE_KEYS = frozenset({"node", "neighbor", "kind", "ts_ms", "ordinal"})

# The engine-only attributes neither the state builder nor the function that assembles
# `obs` may name.  `nodes` is every other node's object, `timeline` the outage schedule,
# `run` the whole run dict, `heap` the future, `hist` a neighbour's store history.
FORBIDDEN_ATTRS = frozenset({"nodes", "timeline", "run", "heap", "hist"})

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

# 2026-11-23: the action fields `obs["resolved"]` does not keep.  `neighbor` for the same
# reason the history element drops it; the other three are the decision's own OUTPUT, and
# handing a decider the interval it is about to return would be handing it the answer.
RESOLVED_DROPS = {"neighbor", "interval_s", "applied_interval_s", "clamped"}

RESOLVED_TYPES = {
    ("poll_epoch_ms",): {"int"},
    ("trigger",): {"str"},
    ("empty",): {"bool", "NoneType"},
}


class Poison:
    """Anything a reader of an `_Origin.hist` would do with it raises.

    Assigned over every node's every origin history while the state is rebuilt, so that a
    builder that reached for a neighbour's store history could not finish.  `_Origin` has
    __slots__, so the attribute takes any object.
    """

    def _blow(self, *_a, **_k):
        raise AssertionError("the state builder read an _Origin.hist")

    # 2026-11-23: `__getattr__` is here beside the five dunders because the five cover only
    # the ways of reading a history that this file happened to think of.  A builder writing
    # `o.hist.append(...)` or `o.hist.copy()` used to get an AttributeError, which is not
    # the named AssertionError and carries nothing about what the builder did wrong.  Every
    # attribute of a history now raises `_blow`'s message, so the recorded fault names the
    # fault.  Operators the type does not define (`o.hist * 2`, a comparison) still miss
    # it, which is why the caller records any exception and not only this one.
    __iter__ = __len__ = __getitem__ = __bool__ = __contains__ = __getattr__ = _blow
    __add__ = __radd__ = _blow


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
        # 2026-11-23: every exception of the rebuild is caught here, not the poison's
        # AssertionError alone.  Letting one leave the decider killed the script with a
        # traceback from inside `engine.go()`, so check 5 printed no FAIL line, its own
        # comparison never ran and the AST half below it was never reached -- a builder
        # that read a neighbour's store history would have ended this file in a crash
        # instead of in a named failing check.  `Poison` cannot raise the named
        # AssertionError on every operator a builder might reach for, so catching that one
        # type would have left the crash in place for the operators it misses.  The
        # rebuild is recorded as the exception, which check 5 reports as the fault it is.
        except Exception as blown:
            rebuilt = f"{node}->{nb} at {obs['where']['ts_ms']}: {blown}"
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

def check_keys(port_shapes, tracker_shape, logged_shapes):
    """Check 1: the path sets of the three sources, strictly, one decision kind at a time.

    2026-11-23: the port's two decision kinds are compared separately.  They used to be
    merged into one shape, so a key present on every resolution and absent from every send
    still gave a complete union.
    """
    bad = 0
    for kind, port_shape in sorted(port_shapes.items()):
        others = [("a real Tracker plus _PollReplies", tracker_shape)]
        if kind in logged_shapes:
            others.append((f"the logged {kind} rows", logged_shapes[kind]))
        for label, other in others:
            extra = sorted(dotted(p) for p in set(port_shape) - set(other))
            missing = sorted(dotted(p) for p in set(other) - set(port_shape))
            bad += say(not extra and not missing,
                       f"key sets equal, the port's {kind} states against {label}: "
                       f"{len(port_shape)} paths")
            if extra:
                print(f"  only the port has: {extra}")
            if missing:
                print(f"  only {label} has: {missing}")
    return bad


def sample_faults(sample):
    """Every level of ONE state whose key set or whose value type is not the declared one.

    2026-11-23.  `merge` unions the paths of many samples and `main` lumped the two
    decision kinds into one shape, so a key that was missing from every call of one kind,
    or from all but one call, or from one `origins[o]` row while another kept it, left the
    union complete and check 1 could not see it.  This function looks at one sample and
    names the level, so a partial omission fails and says where.
    """
    bad = []
    if set(sample) != set(STATE_KEYS):
        bad.append(("the state's own keys",
                    sorted(set(sample) ^ set(STATE_KEYS))))
    for origin, row in sample.get("origins", {}).items():
        if not isinstance(row, dict) or set(row) != set(ORIGIN_KEYS):
            bad.append((f"origins[{origin}]", sorted(set(row) ^ set(ORIGIN_KEYS))
                        if isinstance(row, dict) else type(row).__name__))
            continue
        for nb, adv in row["advertised"].items():
            if not isinstance(adv, dict) or set(adv) != set(ADVERTISED_KEYS):
                bad.append((f"origins[{origin}].advertised[{nb}]",
                            sorted(set(adv) ^ set(ADVERTISED_KEYS))
                            if isinstance(adv, dict) else type(adv).__name__))
    for path, name in walk(sample).items():
        allowed = STATE_TYPES.get(path)
        if allowed is None or name not in allowed:
            bad.append((dotted(path), f"type {name}, declared {sorted(allowed or ())}"))
    return bad


def check_each_sample(sources):
    """Check 1b: every individual sample of every source, one decision kind at a time."""
    bad = 0
    for label, samples in sources:
        faults = [(i, f) for i, f in
                  ((i, sample_faults(s)) for i, s in enumerate(samples)) if f]
        bad += say(not faults,
                   f"each of the {len(samples)} {label} on its own: exactly the "
                   f"{len(STATE_KEYS)} state keys, every origin row exactly "
                   f"{sorted(ORIGIN_KEYS)}, every advertiser row exactly "
                   f"{sorted(ADVERTISED_KEYS)}, every value type the declared one "
                   f"({len(faults)} samples with a fault)")
        for i, faulty in faults[:3]:
            print(f"  sample {i}: {faulty}")
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


def check_resolved(captured, events, logged_actions):
    """Check 4b: `obs["resolved"]`, by keys against the log and by VALUE against its row.

    2026-11-23.  The keys first, because the point of the field is that a real node has
    these three at the decision instant and writes them into the `action` of the same
    `state_poll_reschedule` row.  Then the values, which is the one value-level comparison
    in this file: the dict the decider was handed has to be the three fields of the very
    row that decision wrote.  A send has no poll being resolved, so None there is part of
    the contract and not an absence to be tolerated.
    """
    action_keys = {frozenset(a) for a in logged_actions}
    if len(action_keys) != 1:
        return say(False, "the logged reschedule rows carry one action fieldset "
                          "(checked again here because 4b reads it)")
    want = set(next(iter(action_keys))) - RESOLVED_DROPS
    seen = {frozenset(obs["resolved"]) for obs, _r in captured
            if obs["where"]["kind"] == "state_poll_reschedule"}
    bad = say(len(seen) == 1 and set(next(iter(seen))) == want,
              f"every resolution's `resolved` carries exactly {sorted(want)}: the logged "
              f"action fields less {sorted(RESOLVED_DROPS)}")
    if not (len(seen) == 1 and set(next(iter(seen))) == want):
        print(f"  the port's resolved dicts: {[sorted(s) for s in seen]}")
        print(f"  wanted: {sorted(want)}")
    sends = [obs for obs, _r in captured if obs["where"]["kind"] == "state_poll"]
    bad += say(all(obs["resolved"] is None for obs in sends),
               f"all {len(sends)} send-time decisions were handed `resolved` None, since "
               "no poll is being resolved there")
    rows = {}
    for node, evs in events.items():
        for e in evs:
            if e.get("event") == "decision" and e.get("kind") == "state_poll_reschedule":
                key = (node, e["action"]["neighbor"], e["ts_ms"])
                rows.setdefault(key, []).append(e["action"])
    res = [obs for obs, _r in captured
           if obs["where"]["kind"] == "state_poll_reschedule"]
    off = 0
    for obs in res:
        w = obs["where"]
        got = rows.get((w["node"], w["neighbor"], w["ts_ms"]), [])
        if not any({k: a[k] for k in want} == obs["resolved"] for a in got):
            off += 1
    bad += say(not off,
               f"all {len(res)} resolutions: the VALUES of `resolved` are the three "
               f"fields of the action of the row that decision wrote ({off} differing)")
    return bad


def attribute_names(func):
    """Every attribute and bare name the SOURCE of `func` reads or writes.

    `ast` and not a substring search over the text.  Two reasons, 2026-11-23:
    - a substring search cannot tell `nd.poll_hist` -- the history the port is built to
      hand over -- from a forbidden `hist`, so the forbidden set could not name `hist` at
      all without the check failing on the port's own field;
    - the old version stripped the docstring by splitting the source on the triple-quote
      and keeping the LAST piece, so a second triple-quoted string anywhere in the body
      would have taken every line before it out of the search.
    An `Attribute.attr` is the name after the dot and a `Name.id` a bare one; both are
    collected, so `self.nodes`, `nodes[...]` and a `from x import nodes` are all seen, and
    a string or a comment that merely spells a forbidden name is not.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


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
    # A rebuild that raised comes back as the text of its own exception, which is how the
    # decider records a builder that reached for a poisoned history (2026-11-23); a
    # rebuild that differs from the state the decider was handed is the other way this
    # fails, and the two are counted apart because they are different faults.
    blown = [r for _obs, r in captured if isinstance(r, str)]
    differ = [obs["where"] for obs, r in captured
              if not isinstance(r, str) and r != obs["state"]]
    bad = say(not blown and not differ,
              f"all {len(captured)} decisions: the state the BUILDER produces with every "
              "_Origin.hist poisoned is the state the decider was handed, so the builder "
              "reads no history at the decision instant (reply-time content is not "
              "covered by this; see the module docstring)"
              + ("" if not blown and not differ else
                 f" -- {len(blown)} rebuild(s) raised {blown[:2]} and {len(differ)} "
                 f"differed {differ[:2]}"))
    # 2026-11-23: `obs` and `where` held to their exact key sets, per call.  Nothing read
    # them before, so a key added to `obs` inside `ask_decider` -- an engine attribute, a
    # second name, anything -- passed every line of this file.
    off = [sorted(set(obs) ^ set(OBS_KEYS)) for obs, _r in captured
           if set(obs) != set(OBS_KEYS)]
    bad += say(not off,
               f"all {len(captured)} calls: `obs` carries exactly {sorted(OBS_KEYS)} "
               f"({len(off)} differing{'' if not off else ': ' + str(off[:3])})")
    off = [sorted(set(obs["where"]) ^ set(WHERE_KEYS)) for obs, _r in captured
           if set(obs["where"]) != set(WHERE_KEYS)]
    bad += say(not off,
               f"all {len(captured)} calls: `obs[\"where\"]` carries exactly "
               f"{sorted(WHERE_KEYS)} ({len(off)} differing"
               f"{'' if not off else ': ' + str(off[:3])})")
    # The two functions that decide what a decider sees: the builder, and the assembler
    # that holds the engine.  `_Engine.ask_decider` had never been inspected, and it is
    # the one with `self.nodes`, `self.timeline`, `self.run` and `self.heap` in scope.
    for label, func in (("Node.decision_state", R.Node.decision_state),
                        ("_Engine.ask_decider", R._Engine.ask_decider)):
        reaches = sorted(attribute_names(func) & set(FORBIDDEN_ATTRS))
        bad += say(not reaches,
                   f"{label} names none of {sorted(FORBIDDEN_ATTRS)} anywhere in its "
                   f"code{'' if not reaches else ': it names ' + str(reaches)}")
    return bad


def main():
    run = R.load_run(STAGE, P, SEED)
    captured, events, engine = port_run(run)
    logged, logged_actions = logged_states(run)
    # Split by decision kind, because a shape merged over both kinds hides a key that one
    # kind never carries.
    port_samples = {}
    for obs, _r in captured:
        port_samples.setdefault(obs["where"]["kind"], []).append(obs["state"])
    port_shapes = {k: merge(v) for k, v in port_samples.items()}
    tracker_fixtures = tracker_states()
    tracker_shape = merge(tracker_fixtures)
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
          f"{engine.provisional_fired}")
    # 2026-11-23: the five situations this run does not enter, counted over the states
    # this script actually checks, and with the population printed beside each.  They used
    # to be described in prose with a denominator no script produced, so a reader could
    # not tell which run or which population the share was taken over.  Counted, not
    # asserted: whether a cell reaches them is a property of the workload.
    cells = above = in_flight = truncated = pending = tmo = 0
    for obs, _r in captured:
        state = obs["state"]
        in_flight += state["in_flight"] > 0
        truncated += bool(state["truncated"])
        pending += state["last_poll_status"] == "pending"
        tmo += state["last_timeout_s"] is not None
        for origin_row in state["origins"].values():
            for adv in origin_row["advertised"].values():
                cells += 1
                above += adv["notified_max"] > adv["hi"]
    print(f"  over the {len(captured)} port states of this run, and the {cells} advertised "
          f"cells they hold: in_flight > 0 in {in_flight}, truncated in {truncated}, "
          f"last_poll_status \"pending\" in {pending}, last_timeout_s set in {tmo}, "
          f"notified_max above the advertiser's own hi in {above} cells")
    print("  a situation with a count of 0 is covered for its SHAPE only, by the Tracker "
          "fixture below, and by no value this run produced\n")
    bad = check_keys(port_shapes, tracker_shape, logged_shapes)
    bad += check_each_sample(
        [(f"port {kind} states", samples)
         for kind, samples in sorted(port_samples.items())]
        + [("Tracker plus _PollReplies fixtures", tracker_fixtures)]
        + [(f"logged {kind} states", samples)
           for kind, samples in sorted(logged.items())])
    bad += check_types(
        [(f"the port's {kind} states", s, STATE_TYPES)
         for kind, s in sorted(port_shapes.items())]
        + [("a real Tracker plus _PollReplies", tracker_shape, STATE_TYPES)]
        + [(f"the logged {k} rows", s, STATE_TYPES) for k, s in logged_shapes.items()]
        + [("the port's history elements",
            merge([row for obs, _r in captured for row in obs["history"]]),
            HISTORY_TYPES),
           ("the port's resolved dicts",
            merge([obs["resolved"] for obs, _r in captured
                   if obs["resolved"] is not None]),
            RESOLVED_TYPES)])
    bad += check_history(captured, logged_actions)
    bad += check_rows(captured, events)
    bad += check_resolved(captured, events, logged_actions)
    bad += check_no_hist(captured)
    print(f"\n{'every check passed' if not bad else str(bad) + ' CHECK(S) FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
