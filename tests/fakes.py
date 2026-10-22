"""In-memory multi-node transport for exercising SyncEngine without Cefore.

Interests are queued by `send_interest` and resolved only when the test
calls `pump`, never inside the call that queued them. This models one hop
of latency: a fetch Interest an engine sends while handling a reply (e.g.
the immediate "fetch plan" on_state triggers) is not resolved until the
*next* pump, the same way a real Interest sent mid-handling would not be
answered within that same call. It also means `send_interest` never
re-enters the engine that called it.
"""

from __future__ import annotations

from dataclasses import dataclass

from semindex import names
from semindex.engine import SyncEngine


@dataclass(frozen=True)
class _QueuedInterest:
    """One Interest waiting to be resolved: who sent it, and its full wire name."""

    requester: str
    name: str


class FakeNetwork:
    """A controllable multi-node network for unit tests: holder-scoped routing and partitions."""

    def __init__(self) -> None:
        self._nodes: dict[str, SyncEngine] = {}
        self._partitions: set[frozenset[str]] = set()
        self._queue: list[_QueuedInterest] = []

    def add_node(self, node: str, engine: SyncEngine) -> None:
        """Register `engine` as the node that answers Interests addressed to `node`."""
        self._nodes[node] = engine

    def partition(self, a: str, b: str) -> None:
        """Block delivery between `a` and `b` in both directions until `heal`."""
        self._partitions.add(frozenset((a, b)))

    def heal(self, a: str, b: str) -> None:
        """Restore delivery between `a` and `b` after a `partition`."""
        self._partitions.discard(frozenset((a, b)))

    def connected(self, a: str, b: str) -> bool:
        """Whether `a` and `b` can currently reach each other."""
        return frozenset((a, b)) not in self._partitions

    def pending_count(self) -> int:
        """Interests queued since the last pump; tests drain until this is 0."""
        return len(self._queue)

    def transport_for(self, node: str) -> "_NodeTransport":
        """A Transport bound to `node`'s identity, for its SyncEngine to use."""
        return _NodeTransport(self, node)

    def _enqueue(self, requester: str, name: str) -> None:
        """Queue an Interest `requester` sent, to be resolved on the next `pump`."""
        self._queue.append(_QueuedInterest(requester=requester, name=name))

    def pump(self, now: float) -> None:
        """Resolve every Interest queued since the last pump, oldest first.

        Interests queued *during* this resolution (e.g. a fetch sent from
        inside on_state) land in a fresh queue for the next pump, so one
        pump call models one hop of latency, matching ADR-0002's
        "propagation beyond one hop costs about one sync period per hop".
        """
        queue, self._queue = self._queue, []
        for item in queue:
            self._resolve(item, now)

    def _resolve(self, item: _QueuedInterest, now: float) -> None:
        """Deliver one queued Interest to its holder and the reply (or timeout) back to its requester."""
        try:
            parsed = names.parse(item.name)
        except names.MalformedNameError:
            return  # engines never build a malformed name; nothing to resolve
        requester_engine = self._nodes.get(item.requester)
        if requester_engine is None:
            return
        holder = parsed.holder
        if holder not in self._nodes or not self.connected(item.requester, holder):
            self._timeout(requester_engine, parsed, item.name, now)
            return
        response = self._nodes[holder].handle_interest(item.name)
        if response is None:
            self._timeout(requester_engine, parsed, item.name, now)
            return
        if isinstance(parsed, names.StateName):
            requester_engine.on_state(holder, response.payload, now)
        else:
            requester_engine.on_record(item.name, response.payload, now)

    def _timeout(self, requester_engine: SyncEngine, parsed, name: str, now: float) -> None:
        """Tell the requester its Interest for `name` got no reply, if it tracks timeouts at all.

        2026-10-22 fix: dropped the `holder` parameter this took
        alongside `name` -- on_fetch_timeout derives the holder from
        `name` itself now, so passing it separately here was already
        dead as soon as the engine stopped taking it as an argument.
        """
        if isinstance(parsed, names.RecordName):
            requester_engine.on_fetch_timeout(name, now)
        # A timed-out state request needs no callback: the requester's
        # next due on_tick simply asks that neighbor again.


@dataclass
class _NodeTransport:
    """Transport implementation bound to one node, routing through FakeNetwork."""

    _network: FakeNetwork
    _node: str

    def register(self, prefix: str) -> None:
        """No-op: routing is by the holder segment in each name, against FakeNetwork.add_node."""
        pass

    def send_interest(self, name: str, lifetime_ms: int) -> None:
        """Queue `name` on the network, addressed as sent by this node."""
        self._network._enqueue(self._node, name)
