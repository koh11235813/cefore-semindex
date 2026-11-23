#!/usr/bin/env python3
"""What the scripts of this directory share: the seed guard, the per-pair random
streams, the provisional-timer counter and the formula rule F with its choice set.

Not a script.  All of it lives here once, because a second copy of the streams or of F
would be two definitions of the thing a check in this directory is meant to check.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B <script>.py

PYTHONPATH must point at this worktree's `src`: the interpreter's venv carries another
checkout of the package, and evidence/20's predict.py stops on that.
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EV13 = os.path.normpath(os.path.join(HERE, "..", "13"))
EV18 = os.path.normpath(os.path.join(HERE, "..", "18"))
sys.path.insert(0, EV13)
sys.path.insert(0, EV18)
import bound as B  # noqa: E402  evidence/18's engine, the one every later step simulates with
import replay as R  # noqa: E402

# One module object per file, or a check would compare two engines or two scorers.
# os.path.realpath, because evidence/18 puts un-normalised ".." entries on sys.path and
# a module imported through one of those carries the un-normalised path in __file__.
assert os.path.realpath(R.__file__).startswith(EV13), R.__file__
assert os.path.realpath(B.__file__).startswith(EV18), B.__file__
assert B.R is R, (B.R.__file__, R.__file__)

# The measuring seeds of this effort are 121 to 160 and the learning side may not touch
# them, so no script here accepts a seed above the selection range at all.  The bound is
# on the seed, not on a stage name: the same seed on another stage would still be a
# measuring seed's outage schedule.
MAX_SEED = 100

# ---------------------------------------------------------------------------
# F, the formula rule the labelling suffix runs, and the choice set it needs
# ---------------------------------------------------------------------------
# 2026-11-23.  The label of one decision means "which arm is best now if F takes over
# afterwards", so F is part of the label's definition and belongs beside the port rather
# than inside one script.
ARMS = (3.695, 6.0, 10.043)             # the three arms a decision is labelled over
# evidence/20's pinned member, in evidence/18's member shape (i_min, m, i_max, i_timeout,
# k_cap).  It is copied here rather than imported because importing evidence/20's
# prediction script pulls in the generator and numpy, and this module is imported by every
# check in the directory, including ones that need neither.  The copy is not trusted:
# `label.py` asserts it against evidence/20's own constant at import, where that module is
# loaded anyway, so a change there fails the labelling rather than passing silently.
F_MEM = (3.695, 1.5, 10.043, 3.695, None)


def f_update(ival, trigger, empty):
    """F's next interval after one resolution, by the arithmetic the rule path uses.

    The rule path spreads F over two files: `../13/replay.py`'s `resolve` lengthens on an
    empty reply and resets on a reply with content, and `../18/bound.py`'s `resolve`
    override is what writes `i_timeout` on a timeout (the base method leaves the interval
    alone there, which would be a different F).  Both are reproduced here so that a
    decider can BE F, and `labelcheck.py` check 1 is what says the two agree -- it runs F
    through this function and through the rule path and compares the send instants.
    """
    i_min, m, i_max, i_timeout, _k_cap = F_MEM
    if trigger != "answered":
        return i_timeout
    return min(ival * m, i_max) if empty else i_min


def union_choices():
    """The three arms plus every interval F can reach from them, ascending.

    The port refuses an interval outside `choices`, and F leaves the three arms as soon
    as it lengthens, so a labelling run has to declare the closure.  Computed by
    iterating F's own arithmetic rather than written out, so that every value is the
    float the arithmetic produces and the port's `interval not in choices` holds by
    construction and not by a typed literal rounding the same way.
    """
    seen = set(ARMS)
    seen.add(F_MEM[0])                  # the reset target, reachable from anywhere
    seen.add(F_MEM[3])                  # the timeout target, likewise
    frontier = list(seen)
    while frontier:
        x = frontier.pop()
        nxt = f_update(x, "answered", True)
        if nxt not in seen:
            seen.add(nxt)
            frontier.append(nxt)
    return sorted(seen)


def check_seeds(seeds, where):
    """Refuse any seed above MAX_SEED, naming the caller and the offending seeds."""
    bad = sorted({int(s) for s in seeds if int(s) > MAX_SEED})
    if bad:
        raise SystemExit(
            f"{where}: seeds above {MAX_SEED} are reserved for measuring and may not be"
            f" used by anything on the learning side; refused {bad}")
    return list(seeds)


def const_decider(interval_s):
    """A decider that returns one fixed interval, whatever it is handed.

    The two arms of the weight are run through the port this way rather than through the
    old fixed-arm path, so that the arm being scored is the arm the port decides and the
    weight is a statement about the port.
    """

    def decide(_obs):
        return interval_s

    return decide


class Engine(B.Engine):
    """evidence/18's engine with per-pair random streams and a provisional-timer count.

    1. **The lag streams** (opt-in; `pair_lag_seed` None is evidence/18's own behaviour,
       draw for draw).  The base engine draws every Mode B tick lag from one shared
       stream in event order, so a candidate that decides one interval differently moves
       every later lag of every pair of the run, and two candidates cannot be compared on
       one set of lags.  With a seed here, every lag of the ordered pair (node,
       neighbour) comes from that pair's own stream and the first-poll offset of a node
       from that node's own.
       The precedent is evidence/18's own V path (bound.py:173-180), which is live there
       only when a clairvoyant plan is set.
       **State the guarantee as what it is**: pair X's n-th lag draw is the same VALUE
       whatever another pair decides, because the seed string holds nothing a decision
       can touch.  WHICH EVENT that draw serves can still move, on the target pair and
       on another pair alike: `Node.jitter_rng` is per NODE and is drawn once per send
       over both of that node's neighbours, so moving one pair's sends re-assigns the
       node's later jitter draws between its pairs, shifts the other pair's send
       instants, and can change how many draws that pair makes (a lag is drawn per send
       AND per detected poll timeout).  2026-11-23, measured: changing one resolution of
       `h2->h1` on s11a p 0.2 seed 1 leaves every pair's lag values at their positions
       and still moves `h2->h3` from 26 lag draws to 27.  What the lag streams alone buy
       is that a pair's lag values are a function of the pair and the seed alone.
    1b. **The jitter stream** (opt-in; `pair_jitter_seed` None is the node's own shared
       stream, which is what every stage measured so far drew from).  This is the remedy
       for the residue 1 names: with a seed here the jitter of a poll comes from the
       ORDERED PAIR's stream instead of `Node.jitter_rng`, so moving one pair's sends can
       no longer re-assign the node's later jitter draws between its two pairs.  It is
       off by default and guarded on Mode A as well, because Mode A replays a logged run's
       own lags and every published cell was measured with the per-node stream; turning it
       on there or by default would move numbers that are already filed.
       The draw site is `decide_interval`: `Node.jitter_rng` is called there and nowhere
       else in the engine (a resolution re-applies `pend[1]`, the send's own draw), so
       rebinding the attribute just before the base method's single draw is the whole
       change.
    2. **The count** of send-time provisional timers that fired before their own poll had
       resolved.  The resolution-time decision is the one that takes effect, so this
       counter is how far the runs of a step are from that being true of every poll: a
       fired provisional timer is a poll whose send-time interval decided the next send
       on its own.  Counted in the engine because such a poll leaves nothing in the event
       log that tells it apart from a poll that resolved first.
    """

    def __init__(self, run, mode, lags, sim_rng, rule=None, pair_lag_seed=None,
                 pair_jitter_seed=None):
        super().__init__(run, mode, lags, sim_rng, rule)
        self.pair_lag_seed = pair_lag_seed
        self.pair_jitter_seed = pair_jitter_seed
        self._pair_rng = {}
        self._node_rng = {}
        self._jitter_rng = {}
        # 2026-11-23: what `rekey_streams` puts into the seed strings of the per-pair
        # streams.  None until a caller switches, which is what makes a run before the
        # switch the same run for every caller that has not switched yet.
        self.stream_suffix = None
        self.provisional_fired = 0

    def rekey_streams(self, suffix):
        """Start fresh per-pair lag and jitter streams, keyed by `suffix` as well.

        2026-11-23, for the labelling: one label compares three arms of one decision over
        several passes, and the passes exist to average out the randomness AFTER that
        decision while the run BEFORE it stays one run.  So a pass is a suffix switched on
        at the probed decision: everything drawn before comes from the pair's plain stream
        and is identical for every arm and every pass, and everything drawn after comes
        from a stream whose seed string carries the pass.  The suffix must not carry the
        ARM, or the three arms of one pass would not share their draws at all.

        Called from inside a decider, i.e. from inside `resolve`, which is the right
        instant: the first draw after the decider returns is the re-arm's tick lag.
        **What this aligns and what it does not**: within one pass the three arms draw the
        same values at the same POSITION of each pair's stream; which event a draw serves
        can still differ, because one arm may re-arm its timer where another takes the
        identity early return and draws nothing.

        `_node_rng` is deliberately not re-keyed and not cleared: the first-poll offset is
        drawn in `go`, before any decision exists, so no rekey can be reached before it.
        """
        self.stream_suffix = suffix
        self._pair_rng.clear()
        self._jitter_rng.clear()

    def _seed(self, kind, *parts):
        """The seed string of one per-pair stream, with the pass suffix when there is one."""
        tail = ":".join(str(x) for x in parts)
        if self.stream_suffix is None:
            return f"{kind}:{tail}"
        return f"{kind}:{self.stream_suffix}:{tail}"

    def tick_lag(self, node, nb, kind):
        if self.pair_lag_seed is None or self.mode == "A":
            return super().tick_lag(node, nb, kind)
        g = self._pair_rng.get((node, nb))
        if g is None:
            g = self._pair_rng[(node, nb)] = random.Random(
                self._seed("pair_lag", self.pair_lag_seed, node, nb))
        return g.uniform(0.0, R.TICK_PERIOD_MS)

    def start_lag(self, node):
        if self.pair_lag_seed is None:
            return super().start_lag(node)
        g = self._node_rng.get(node)
        if g is None:
            g = self._node_rng[node] = random.Random(
                f"node_start_lag:{self.pair_lag_seed}:{node}")
        return g.uniform(0.0, R.TICK_PERIOD_MS)

    def decide_interval(self, nd, nb, t=None, state=None):
        """The base method's single jitter draw, routed to the ordered pair's stream.

        The rebinding is on `Node.jitter_rng` rather than inside a copy of the base
        method, because the base method has four branches that each draw it once and a
        copy here would be a second definition of the interval sources.  The attribute is
        the node's own, so it is rebound per call: the node has two neighbours and the
        decision names which one.
        """
        if self.pair_jitter_seed is not None and self.mode != "A":
            g = self._jitter_rng.get((nd.name, nb))
            if g is None:
                g = self._jitter_rng[(nd.name, nb)] = random.Random(
                    self._seed("pair_jitter", self.pair_jitter_seed, nd.name, nb))
            nd.jitter_rng = g.random
        return super().decide_interval(nd, nb, t=t, state=state)

    def ev_poll(self, t, node, nb, gen):
        nd = self.nodes[node]
        pend = nd.pending.get(nb)
        # `pend[4]`, the provisional due instant, is written only where the engine arms a
        # timer it expects the resolution to replace, so it is what tells a provisional
        # timer from the plain fixed-arm timer that IS final (that path never resolves a
        # poll at all, and counting it would call every poll of a fixed arm a miss).
        # `pend[2]` false says this poll had not resolved when its timer fired.  The
        # engine's own liveness and stop tests are read first: a superseded timer and a
        # timer past the agent's stop instant both send nothing.
        if (gen == nd.gen[nb] and t - self.t0 < R.STOP_AT_MS
                and pend is not None and pend[4] is not None and not pend[2]):
            self.provisional_fired += 1
        return super().ev_poll(t, node, nb, gen)


def sim(run, sim_rng=None, pair_lag_seed=None, pair_jitter_seed=None):
    """One Mode B simulation through this directory's engine: (events, engine)."""
    engine = Engine(run, "B", None, sim_rng, None, pair_lag_seed, pair_jitter_seed)
    return engine.go(), engine
