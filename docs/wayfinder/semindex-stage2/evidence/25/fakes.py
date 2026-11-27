"""Fake poll deciders for exercising the live engine's decider seam offline.

`Const` is the simplest decider the protocol admits. `Delaying` turns any
synchronous decider into one that defers every resolution, which is
otherwise only reachable by running a real network client against a model
server, so the deferred path, a late answer and an overlap can all be
driven with no model and no socket. `Recording` writes the four inputs a
decision was handed and what it returned, so a run's log can later be
checked field by field against the engine's own `events.jsonl`.
"""
import sys
# 2026-11-27: node processes run as root and are started with only PYTHONPATH in
# their environment (no PYTHONDONTWRITEBYTECODE), so without this flag root would
# leave __pycache__ directories in a checkout its own owner cannot delete.
sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402


class Const:
    """A decider that answers every call with the same interval, nothing else."""

    def __init__(self, interval):
        """Remember the one interval every call will answer with."""
        self._interval = interval

    def decide(self, obs):
        """Answer `self._interval` regardless of `obs`."""
        return self._interval

    def collect(self):
        """Nothing is ever deferred, so there is never anything to release."""
        return []

    def close(self):
        """Nothing to flush or close."""
        pass


class Delaying:
    """Wraps a synchronous decider so it looks like a deferring one.

    `inner` answers every resolution at once; this class holds that answer
    for `delay_s` before `collect()` lets it through, so the engine sees the
    same deferred-path behaviour a real network client produces, without a
    socket or a model.
    """

    def __init__(self, inner, delay_s):
        """Wrap `inner`; hold its resolutions for `delay_s` before release."""
        self._inner = inner
        self._delay_s = delay_s
        # The pair's last RELEASED answer, not its last computed one: see
        # decide()'s send branch below for why the two differ.
        self._mem = {}
        # FIFO of (token, interval, release_at); delay_s is constant, so
        # popping the front in order is both arrival and release order.
        self._held = []

    def decide(self, obs):
        """At a send, the last released value; at a resolution, defer and return None.

        A real client learns an answer only when it arrives, so a send sent
        while the answer is outstanding must carry the previous value,
        except at a pair's first send, which has nothing released yet.
        """
        where = obs["where"]
        if where["kind"] != "state_poll_reschedule":
            neighbor = where["neighbor"]
            if neighbor not in self._mem:
                self._mem[neighbor] = self._inner.decide(obs)
            return self._mem[neighbor]
        # inner's own state (its "mem", if it keeps one) has to advance in
        # ask order, so it is called here, immediately, and not deferred
        # itself; only the answer reaching the engine is held back.
        interval = self._inner.decide(obs)
        token = (where["neighbor"], obs["resolved"]["poll_epoch_ms"])
        self._held.append((token, interval, time.monotonic() + self._delay_s))
        return None

    def collect(self):
        """Release, in ask order, every held answer whose delay has elapsed."""
        released = []
        now = time.monotonic()
        while self._held and self._held[0][2] <= now:
            token, interval, _release_at = self._held.pop(0)
            self._mem[token[0]] = interval
            released.append((token, interval, {"delay_s": self._delay_s}))
        return released

    def close(self):
        """Close `inner`; answers still held here are simply never released."""
        self._inner.close()


class Recording:
    """Writes one JSON line per decision and per released answer to a file.

    This is the side that records what a decision was handed, so a later
    check can compare it against the engine's own log without trusting
    the engine to describe its own input correctly.
    """

    def __init__(self, inner, record_dir, node, meta):
        """Open `<record_dir>/<node>.jsonl` for append and write the start line.

        Append, not truncate, so a crash or a second run never loses lines
        a first run already wrote.
        """
        self._inner = inner
        self._seq = 0
        self._collected = 0
        os.makedirs(record_dir, exist_ok=True)
        self._fh = open(os.path.join(record_dir, "%s.jsonl" % node), "a", encoding="utf-8")
        with open(__file__, "rb") as fh:
            fakes_sha256 = hashlib.sha256(fh.read()).hexdigest()
        self._write({"rec": "start", "node": node, "pid": os.getpid(),
                     "t_wall": time.time(), "t_mono": time.monotonic(),
                     "fakes_sha256": fakes_sha256, **(meta or {})})

    def _write(self, obj):
        """Append one JSON line and flush, so a crash loses nothing already decided."""
        self._fh.write(json.dumps(obj) + "\n")
        self._fh.flush()

    def decide(self, obs):
        """Call `inner.decide`, recording the call whether it returns or raises.

        Snapshotted before the call, by round-trip through JSON and not a
        shallow copy: `obs["state"]` is handed over by reference, and a
        mutating inner decider would corrupt the record otherwise. The
        token is computed before the call too, so a raise still logs it.
        """
        seq = self._seq
        self._seq += 1
        snapshot = json.loads(json.dumps(obs))
        token = None
        if obs["resolved"] is not None:
            token = [obs["where"]["neighbor"], obs["resolved"]["poll_epoch_ms"]]
        t_wall = time.time()
        t_mono = time.monotonic()
        t0 = time.perf_counter()
        try:
            returned = self._inner.decide(obs)
        except Exception as exc:
            self._write({"rec": "decide", "seq": seq, "t_wall": t_wall, "t_mono": t_mono,
                         "elapsed_s": time.perf_counter() - t0, "obs": snapshot,
                         "returned": None, "token": token, "raised": repr(exc)})
            raise
        self._write({"rec": "decide", "seq": seq, "t_wall": t_wall, "t_mono": t_mono,
                     "elapsed_s": time.perf_counter() - t0, "obs": snapshot,
                     "returned": returned, "token": token})
        return returned

    def collect(self):
        """Call `inner.collect`, recording each released answer, and return it unchanged."""
        answers = self._inner.collect()
        for token, interval, info in answers:
            self._collected += 1
            self._write({"rec": "collect", "t_wall": time.time(), "t_mono": time.monotonic(),
                         "token": list(token), "interval": interval, "info": info})
        return answers

    def close(self):
        """Close `inner`, write the closing counts, and close the file."""
        self._inner.close()
        self._write({"rec": "close", "t_wall": time.time(), "t_mono": time.monotonic(),
                     "decides": self._seq, "collected": self._collected})
        self._fh.close()


def wrap(inner, *, node, record_dir=None, delay_s=None, meta=None):
    """Wrap `inner` as `Recording(Delaying(inner))`, skipping either layer not asked for.

    Delaying inside, Recording outside, so the recorder sees exactly what
    the engine sees: a deferral's `None` and the answers collect() released.
    """
    if delay_s is not None:
        inner = Delaying(inner, delay_s)
    if record_dir is not None:
        inner = Recording(inner, record_dir, node, {**(meta or {}), "delay_s": delay_s})
    return inner


def build_const(node, interval, record_dir=None, delay_s=None):
    """Factory for a constant decider, for the loader's `factory` key."""
    return wrap(Const(interval), node=node, record_dir=record_dir, delay_s=delay_s,
                meta={"inner": "const", "interval": interval})
