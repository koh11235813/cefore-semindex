#!/usr/bin/env python3
"""What the four checks of this directory share: the seed guard, the per-pair lag
streams and the provisional-timer counter.

Not a script.  The seed guard, the lag streams and the provisional-timer counter live
here once, because a second copy of the lag streams would be two definitions of the
thing a check in this directory is meant to check.

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
    """evidence/18's engine with per-pair tick-lag streams and a provisional-timer count.

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
       and still moves `h2->h3` from 26 lag draws to 27.  What the streams buy is that a
       pair's lag values are a function of the pair and the seed alone; aligning events
       as well would need a per-node jitter stream, which would change every existing
       stage.
    2. **The count** of send-time provisional timers that fired before their own poll had
       resolved.  The resolution-time decision is the one that takes effect, so this
       counter is how far the runs of a step are from that being true of every poll: a
       fired provisional timer is a poll whose send-time interval decided the next send
       on its own.  Counted in the engine because such a poll leaves nothing in the event
       log that tells it apart from a poll that resolved first.
    """

    def __init__(self, run, mode, lags, sim_rng, rule=None, pair_lag_seed=None):
        super().__init__(run, mode, lags, sim_rng, rule)
        self.pair_lag_seed = pair_lag_seed
        self._pair_rng = {}
        self._node_rng = {}
        self.provisional_fired = 0

    def tick_lag(self, node, nb, kind):
        if self.pair_lag_seed is None or self.mode == "A":
            return super().tick_lag(node, nb, kind)
        g = self._pair_rng.get((node, nb))
        if g is None:
            g = self._pair_rng[(node, nb)] = random.Random(
                f"pair_lag:{self.pair_lag_seed}:{node}:{nb}")
        return g.uniform(0.0, R.TICK_PERIOD_MS)

    def start_lag(self, node):
        if self.pair_lag_seed is None:
            return super().start_lag(node)
        g = self._node_rng.get(node)
        if g is None:
            g = self._node_rng[node] = random.Random(
                f"node_start_lag:{self.pair_lag_seed}:{node}")
        return g.uniform(0.0, R.TICK_PERIOD_MS)

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


def sim(run, sim_rng=None, pair_lag_seed=None):
    """One Mode B simulation through this directory's engine: (events, engine)."""
    engine = Engine(run, "B", None, sim_rng, None, pair_lag_seed)
    return engine.go(), engine
