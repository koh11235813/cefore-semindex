#!/usr/bin/env python3
"""What the two prediction scripts of this directory share: the engine that may run the
held-out seeds, the stream of one (seed, stream), the run dict and its JSON form, T37r as
a decider, and the one row of one simulation.

Not a script.  The CPU side (six stages, this machine) and the GPU side (L) both run
on it, so that the seven stages of the prediction are simulated by one definition of an
engine, a run dict and a row; a second copy on either side would be two definitions of
the thing the gates are meant to pin.

It imports `../22/port.py`, `../22/label.py` and `../22/score.py` and nothing from the
generator directly: the generator reaches the run dicts only through `label.PR`, i.e.
evidence/21's `run_dict`, which is how every candidate of the comparison was scored.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVID = os.path.dirname(HERE)
EV22 = os.path.join(EVID, "22")
if EV22 not in sys.path:
    sys.path.insert(0, EV22)

import label as LBL    # noqa: E402
import port as PORT    # noqa: E402
import score as SCORE  # noqa: E402

R = PORT.R
B = PORT.B

MAIN_P = 0.2
HELD_OUT = tuple(range(121, 161))       # the measuring seeds, never used for learning
NREP = LBL.PR.NREP                      # 40 replicate streams, evidence/21's and the scoring's
# The scoring's seed string (../23/eval.py:80, ../24/eval24.py LAG_PREFIX).  Used as BOTH
# pair_lag_seed and pair_jitter_seed with sim_rng None and no rekey, which is what every
# row of ../23/eval.csv and ../24/eval24.csv was simulated with.
LAG_PREFIX = "pairlag:%s" % LBL.PR.STREAM_LABEL
T37R_MEM = (3.695, 2.0, 10.043, 3.695, None)   # evidence/20's T37r, member shape

# ---------------------------------------------------------------------------
# The code this module copies from, pinned
# ---------------------------------------------------------------------------
# 2026-11-28: `HeldOutEngine.__init__` repeats `port.Engine.__init__`'s body without its
# seed guard, so a change to port.py would leave the copy describing an engine that no
# longer exists.  The digests are the ones ../23/eval.txt and ../24/eval24.txt printed for
# the scoring, and ../25/lserver.py and ../25/mdecider.py pin on the testbed side, so a
# file that moved since the scoring refuses here as well.
PINS = (
    ("22/port.py", "5f74617a04b43d71f2a29153e26251e357cd44436925ecb9fa6b596fc53cca07"),
    ("22/label.py", "3a7b12bbdd95b1ca2eb83bed835f39b08a53e6ae302f6a776b6db321563c54d0"),
    ("22/score.py", "24743e3547ed877cec0a35c8925f12f20c2ed6941d4d805268701fbd0e0f3e49"),
    ("13/replay.py", "f689d3a82fb3740f2aedf4a6b523329c971de05b4947f103f2c0bb882b58f49d"),
    ("18/bound.py", "564b2cd6cc7fe4fab5dc9e82cf6881d841bf5b784ef261c9cde7371532d739bf"),
)

# The files whose bytes decide one L row, in the order their digests are combined.  One
# list for both sides, so the digest the GPU side writes into a row can be recomputed over
# the local copies, and equality says the GPU machine ran these bytes.
L_CODE = ("25/lpredict.py", "25/heldout.py", "25/lserver.py", "23/eval.py", "23/decider.py",
          "23/prefill.py", "22/observe.py", "22/port.py", "22/label.py", "22/score.py",
          "13/replay.py")


def sha_of(path):
    """The sha256 of one file's bytes."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def check_pins():
    """Every pinned file has the bytes the scoring recorded; returns the count."""
    for rel, want in PINS:
        got = sha_of(os.path.join(EVID, rel))
        if got != want:
            raise SystemExit("heldout: %s has sha256 %s, the scoring recorded %s"
                             % (rel, got, want))
    return len(PINS)


def combined(rels):
    """One digest over the named files, as ../23/decider.py's `combined_sha` joins them."""
    return hashlib.sha256("".join(sha_of(os.path.join(EVID, r)) for r in rels)
                          .encode()).hexdigest()


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
# Off until a caller that was given its explicit held-out flag turns it on.  Seeds 1 to 100
# are always allowed, because they are the seeds the gates are read on.
ARMED = {"held_out": False}


def arm_held_out():
    """Allow seeds 121 to 160 in `HeldOutEngine` and `build_run` for this process.

    Meant to be reached only through an explicit once-only flag, so that no default path
    of either side can simulate a measuring seed.
    """
    ARMED["held_out"] = True


class HeldOutEngine(PORT.Engine):
    """port.Engine without its constructor's seed guard, and with a narrower one.

    2026-11-28.  port.Engine refuses any run dict whose seed is above 100, because the
    learning side must never touch the measuring seeds.  This is the one use
    evidence/20/spec.md "What this step cannot show" allows them: to score, in the replay,
    a candidate that has already been selected, once, as the prediction written before
    that candidate's testbed runs.  So this subclass is the ONLY door, port.Engine keeps
    its guard for every other caller, and the door is shut unless `arm_held_out` ran:
    seeds 1 to 100 pass (the gates), 121 to 160 pass only when armed, anything else is
    refused.  The body below is port.Engine.__init__'s after the guard, which is why
    port.py is pinned by digest (`check_pins`) and `same_shape` compares the attributes.
    """

    def __init__(self, run, mode, lags, sim_rng, rule=None, pair_lag_seed=None,
                 pair_jitter_seed=None):
        seed = int(run["seed"])
        if not (1 <= seed <= 100 or (seed in HELD_OUT and ARMED["held_out"])):
            raise SystemExit("HeldOutEngine: seed %d refused (held-out armed: %s)"
                             % (seed, ARMED["held_out"]))
        B.Engine.__init__(self, run, mode, lags, sim_rng, rule)
        self.pair_lag_seed = pair_lag_seed
        self.pair_jitter_seed = pair_jitter_seed
        self._pair_rng = {}
        self._node_rng = {}
        self._jitter_rng = {}
        self.stream_suffix = None
        self.provisional_fired = 0


def same_shape(run):
    """The two constructors leave the same attribute names on a seed both accept."""
    a = set(vars(HeldOutEngine(run, "B", None, None, None, 1, 1)))
    b = set(vars(PORT.Engine(run, "B", None, None, None, 1, 1)))
    return a == b, sorted(a ^ b)


def stream_of(seed, rep):
    """The scoring's seed string value for one (seed, stream)."""
    return R.rep_seed(LAG_PREFIX, MAIN_P, seed, rep)


def sim(run, rep):
    """One Mode B simulation on the scoring's stream options: (events, engine)."""
    base = stream_of(run["seed"], rep)
    engine = HeldOutEngine(run, "B", None, None, None, base, base)
    return engine.go(), engine


# ---------------------------------------------------------------------------
# T37r as a decider
# ---------------------------------------------------------------------------
def member_update(member, ival, trigger, empty):
    """`port.f_update` with the member as an argument instead of port.F_MEM.

    Needed because T37r was never a decider of the scoring and port.f_update hard-wires
    F.  `MemberF(port.F_MEM)` reproducing ../23/eval.csv's F rows is the check that this
    is f_update's arithmetic and nothing else.
    """
    i_min, m, i_max, i_timeout, _k_cap = member
    if trigger != "answered":
        return i_timeout
    return min(ival * m, i_max) if empty else i_min


class MemberF(LBL.F):
    """label.F with the member as an argument.  The member is kept as `member` and not
    `mem`, because the first-call memory wrappers read `mem` before `ival`."""

    def __init__(self, member, ival=None):
        super().__init__(ival)
        self.member = tuple(member)

    def get(self, pair):
        """The pair's current interval, the member's i_min before its first resolution."""
        return self.ival.get(pair, self.member[0])

    def __call__(self, obs):
        """label.F's update order: the update before the value, at a resolution only."""
        w = obs["where"]
        pair = (w["node"], w["neighbor"])
        if w["kind"] == "state_poll_reschedule":
            got = obs["resolved"]
            self.ival[pair] = member_update(self.member, self.get(pair), got["trigger"],
                                            got["empty"])
        return self.get(pair)


def closure(member):
    """The arms plus every interval `member` reaches from them, ascending.

    port.union_choices for any member; the port refuses an interval outside the run's
    choices, and T37r reaches 7.39 s, which is not in the union set F needs.
    """
    seen = set(PORT.ARMS) | {member[0], member[3]}
    frontier = list(seen)
    while frontier:
        nxt = member_update(member, frontier.pop(), "answered", True)
        if nxt not in seen:
            seen.add(nxt)
            frontier.append(nxt)
    return sorted(seen)


# ---------------------------------------------------------------------------
# The run dict and its JSON form
# ---------------------------------------------------------------------------
def build_run(seed):
    """label.run_dict's body without its guard, for one seed of the main cell.

    `label.run_dict` is `evidence/21`'s run dict plus the union choice set, the choice
    jitter and `redecide`, and its first line refuses seeds above 100.  The gate holds this
    equal to `label.run_dict` on every seed it may be called with, so the held-out seeds
    get the construction every candidate was scored on: generated outage timeline and
    record schedule, t0 and agent_start borrowed from donor seed 1 + (seed - 1) % 20, which
    for 121 to 160 is evidence/20's 1 + (seed - 121) % 20.
    """
    if not (1 <= seed <= 100 or (seed in HELD_OUT and ARMED["held_out"])):
        raise SystemExit("build_run: seed %d refused (held-out armed: %s)"
                         % (seed, ARMED["held_out"]))
    run = LBL.PR.run_dict(MAIN_P, seed)
    return dict(run, policy=None, rule=None, choices=list(LBL.CHOICES),
                choice_jitter=LBL.JF, redecide=True, decider=None)


def encode_run(run):
    """A JSON-safe form: frozenset timeline keys become sorted lists, tuples lists."""
    out = dict(run)
    out["timeline"] = [[sorted(k), [list(x) for x in v]]
                       for k, v in sorted(run["timeline"].items(), key=lambda kv: sorted(kv[0]))]
    out["created"] = [list(c) for c in run["created"]]
    return out


def decode_run(obj):
    """`encode_run`'s inverse; `same` is what says the two are one dict."""
    out = dict(obj)
    out["timeline"] = {frozenset(k): [tuple(x) for x in v] for k, v in obj["timeline"]}
    out["created"] = [tuple(c) for c in obj["created"]]
    return out


def same(a, b, where="run"):
    """None when `a` and `b` are equal with equal types all the way down, else the path.

    Plain `==` would pass a list for a tuple's place only if both were lists, but would
    pass 1 for 1.0; the engine does not care about the second, the comparison does
    anyway, because a round trip that changed a type is a round trip worth knowing about.
    """
    if type(a) is not type(b):
        return "%s: %s against %s" % (where, type(a).__name__, type(b).__name__)
    if isinstance(a, dict):
        if set(a) != set(b):
            return "%s: keys %s" % (where, sorted(map(str, set(a) ^ set(b))))
        for k in a:
            bad = same(a[k], b[k], "%s[%r]" % (where, k))
            if bad:
                return bad
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return "%s: length %d against %d" % (where, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            bad = same(x, y, "%s[%d]" % (where, i))
            if bad:
                return bad
        return None
    return None if a == b else "%s: %r against %r" % (where, a, b)


def write_runs(path, payload):
    """Write the input file once; an existing file is refused, never replaced."""
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True)
    return sha_of(path)


def read_runs(path):
    """(payload, {seed: run dict}) of an input file written by `write_runs`."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload, {int(s): decode_run(r) for s, r in payload["runs"].items()}


# ---------------------------------------------------------------------------
# One simulation, one row
# ---------------------------------------------------------------------------
# ../23/eval.py's and ../24/eval24.py's columns, plus the input file's digest.
COLUMNS = ("decider", "seed", "stream", "n4", "c4", "lat_ms", "censored_ms", "polls",
           "delay_ms", "score_ms", "score_dropped_ms", "weight_ms_per_poll",
           "reads", "sends", "ties", "n_short", "n_mid", "n_long",
           "ord_poll", "ord_resched", "k", "script_sha", "model_sha")
OUT_COLUMNS = COLUMNS + ("input_sha",)
# The columns a row of this directory is held to a row of the scoring by: everything the
# simulation and the decider decide, nothing about which script or which K wrote it.
SIM_COLUMNS = ("n4", "c4", "lat_ms", "censored_ms", "polls", "delay_ms", "score_ms",
               "score_dropped_ms", "reads", "sends", "ties", "n_short", "n_mid", "n_long",
               "ord_poll", "ord_resched")


def one_run(name, run, rep, wrapper, inner, k):
    """Simulate one (seed, stream) with an already wrapped decider; the row as a dict.

    The row is ../23/eval.py's and ../24/eval24.py's `one_run` row, so a row of this
    directory compares with a row of theirs column by column.  The counters are checked
    the way both of theirs check them: one decider call per engine decision of each kind.
    """
    events, engine = sim(dict(run, decider=wrapper), rep)
    if not wrapper.checked:
        raise SystemExit("%s seed %d stream %d: the decider was never called"
                         % (name, run["seed"], rep))
    t = SCORE.terms(events, run["t0"])
    n4, c4, lat, censored, polls = t
    ordp = engine.decider_ord["state_poll"]
    ordr = engine.decider_ord["state_poll_reschedule"]
    row = {"decider": name, "seed": run["seed"], "stream": rep, "n4": n4, "c4": c4,
           "lat_ms": lat, "censored_ms": censored, "polls": polls,
           "delay_ms": SCORE.delay(t), "score_ms": SCORE.delay(t) + LBL.WEIGHT * polls,
           "score_dropped_ms": SCORE.delay_dropped(t) + LBL.WEIGHT * polls,
           "weight_ms_per_poll": LBL.WEIGHT, "ord_poll": ordp, "ord_resched": ordr,
           "k": k, "reads": "", "sends": "", "ties": "", "n_short": "", "n_mid": "",
           "n_long": ""}
    if hasattr(inner, "reads"):
        if inner.reads != ordr or inner.sends != ordp:
            raise SystemExit("%s seed %d stream %d: reads/sends (%d, %d) against the "
                             "engine's (%d, %d)" % (name, run["seed"], rep, inner.reads,
                                                    inner.sends, ordr, ordp))
        row.update(reads=inner.reads, sends=inner.sends, ties=inner.ties,
                   n_short=inner.arms[PORT.ARMS[0]], n_mid=inner.arms[PORT.ARMS[1]],
                   n_long=inner.arms[PORT.ARMS[2]])
    return row


def text_of(value):
    """One csv cell as the scoring wrote it: floats through repr, the rest through str."""
    return repr(value) if isinstance(value, float) else str(value)


def sim_strings(row):
    """The SIM_COLUMNS of a row as csv strings, for an exact comparison."""
    return {c: text_of(row[c]) if not isinstance(row[c], str) else row[c]
            for c in SIM_COLUMNS}


def read_csv(path):
    """Every row of a csv of the scoring's shape, as {column: string}."""
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    cols = lines[0].split(",")
    return [dict(zip(cols, ln.split(","))) for ln in lines[1:]]


def write_rows(path, rows, script_sha, input_sha):
    """Write rows once, `OUT_COLUMNS` wide; an existing file is refused."""
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(",".join(OUT_COLUMNS) + "\n")
        for row in rows:
            out = dict(row, script_sha=script_sha, input_sha=input_sha)
            fh.write(",".join(text_of(out[c]) for c in OUT_COLUMNS) + "\n")
