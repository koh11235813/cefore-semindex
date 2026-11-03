"""Dump the event stream of fixed, deterministic SyncEngine runs for stages A, B and C.

Run once against main and once against a change, then diff the two outputs: an
identical dump shows the change leaves those stages' behaviour as it was (see
README.md). Uses only APIs that exist on main 3d89cfe, and carries its own small
network (tests/fakes.py is not in an archived `src`), so both sides run the same
harness code.

Usage: PYTHONPATH=<src> python3 identity.py > out.jsonl
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from semindex import names
from semindex.agent import _poll_choice_jitter_rng, _poll_choice_rng, _poll_jitter_rng
from semindex.engine import SyncConfig, SyncEngine
from semindex.store import Store

SEED = 5
NODES = {"h0": ("h1",), "h1": ("h0", "h2"), "h2": ("h1",)}
STAGES = {
    "A": {"state_poll_interval_choices_s": (0.5, 1.359, 3.695, 10.043), "state_poll_choice_jitter_frac": 0.1},
    "B": {"state_poll_interval_choices_s": (3.695,), "state_poll_choice_jitter_frac": 0.1},
    "C": {},  # stage 1: 2 s +/- 20%
}
TICK_S = 0.1
END_S = 120.0
# Link h0-h1 per time window: replies take 10 ms normally, 2.5 s (past the 2 s
# lifetime, so a late reply) over [30, 40), and never come over [50, 62) (down:
# timeouts) and [80, 100) (a silent stretch past lifetime + grace).
SLOW = (30.0, 40.0)
DOWN = ((50.0, 62.0), (80.0, 100.0))


class Net:
    """Delivers each Interest's reply (or a record fetch's timeout) at a scheduled time."""

    def __init__(self) -> None:
        """No nodes and nothing in flight yet."""
        self.engines: dict[str, SyncEngine] = {}
        self.now = 0.0
        self.queue: list[tuple[float, int, str, str, bool]] = []  # (at, order, requester, name, reply?)
        self.order = 0
        self.lines: list[dict] = []

    def send(self, requester: str, name: str, lifetime_ms: int) -> None:
        """Schedule `name`'s outcome and log the send."""
        self.lines.append({"t": round(self.now, 6), "node": requester, "sent": name})
        holder = names.parse(name).holder
        on_link = {requester, holder} == {"h0", "h1"}
        down = on_link and any(a <= self.now < b for a, b in DOWN)
        is_state = isinstance(names.parse(name), names.StateName)
        delay = 2.5 if on_link and is_state and SLOW[0] <= self.now < SLOW[1] else 0.01
        if down:
            if not is_state:
                self._push(self.now + lifetime_ms / 1000.0, requester, name, False)
        else:
            self._push(self.now + delay, requester, name, True)

    def _push(self, at: float, requester: str, name: str, reply: bool) -> None:
        """Queue one outcome, ordered by time then by send order."""
        self.order += 1
        self.queue.append((at, self.order, requester, name, reply))

    def pump(self, now: float) -> None:
        """Deliver every outcome due by `now`, oldest first (one hop per call)."""
        self.now = now
        due = sorted(item for item in self.queue if item[0] <= now)
        self.queue = [item for item in self.queue if item[0] > now]
        for _, _, requester, name, reply in due:
            parsed = names.parse(name)
            engine = self.engines[requester]
            response = self.engines[parsed.holder].handle_interest(name) if reply else None
            if response is None:
                if isinstance(parsed, names.RecordName):
                    engine.on_fetch_timeout(name, now)
            elif isinstance(parsed, names.StateName):
                engine.on_state(parsed.holder, response.payload, now, poll_epoch_ms=parsed.epoch_ms)
            else:
                engine.on_record(name, response.payload, now)


class Tap:
    """One node's Transport into Net."""

    def __init__(self, net: Net, node: str) -> None:
        """Bind to `net` as `node`."""
        self.net, self.node = net, node

    def register(self, prefix: str) -> None:
        """Routing is by holder name; nothing to do."""

    def send_interest(self, name: str, lifetime_ms: int) -> None:
        """Hand the Interest to the network."""
        self.net.send(self.node, name, lifetime_ms)


def run(stage: str, workdir: Path) -> list[dict]:
    """Run one stage's three-node line for END_S and return its log lines."""
    net = Net()
    for node, neighbors in NODES.items():
        overrides = STAGES[stage]
        rngs = {"poll_jitter_rng": _poll_jitter_rng(node)}
        if overrides:
            # As build_agent wires them: the seeded streams whenever a seed is set.
            rngs["poll_choice_rng"] = _poll_choice_rng(SEED, node)
            rngs["poll_choice_jitter_rng"] = _poll_choice_jitter_rng(SEED, node)
        config = SyncConfig(group="g", node=node, neighbors=neighbors, **overrides)
        net.engines[node] = SyncEngine(config, Store(workdir / f"{stage}-{node}.db"), Tap(net, node), **rngs)
    for step in range(round(END_S / TICK_S) + 1):
        now = step * TICK_S
        net.pump(now)
        for node, engine in net.engines.items():
            net.now = now
            if step % 70 == 0 and node != "h1":
                ms = round(now * 1000)
                engine.create_record(
                    content_name=f"c{ms}", zone="zone1", disaster="flood", media="image", observed_at=ms
                )
            engine.on_tick(now)
            for event in engine.drain_events():
                net.lines.append({"t": round(now, 6), "node": node, "event": event})
    return net.lines


def main() -> None:
    """Print every stage's log as JSON lines, keys sorted."""
    with tempfile.TemporaryDirectory() as tmp:
        for stage in STAGES:
            for line in run(stage, Path(tmp)):
                sys.stdout.write(json.dumps({"stage": stage, **line}, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
