"""The Transport protocol SyncEngine depends on, and the holder-side Response shape.

Receiving -- a state or record Data reply arriving, an Interest arriving
at this node's registered prefix, a fetch timing out -- is not part of
this protocol. Those are driven from outside by calling the engine's
on_state / on_record / on_fetch_timeout / handle_interest methods
directly, from whatever drives the real network (an in-memory one or a
Cefore-backed one) rather than from this interface. Keeping replies out
of this protocol is what keeps SyncEngine ignorant of how they get back
to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Transport(Protocol):
    """Outbound-only interface SyncEngine uses to register itself and fire Interests."""

    def register(self, prefix: str) -> None:
        """Register `prefix` so this node receives Interests addressed to it."""
        ...

    def send_interest(self, name: str, lifetime_ms: int) -> None:
        """Fire an Interest for `name` with the given lifetime; do not block for a reply."""
        ...


@dataclass(frozen=True)
class Response:
    """A holder's answer to one incoming Interest, ready for Transport to send as Data.

    `expiry_s` and `cache_time_s` are carried through from SyncConfig
    (map.md "時間・サイズの初期値") rather than acted on here: there is
    no real Data-sending transport in this module, so passing them along
    unopened is what lets a Cefore-backed adapter call send_data with the
    right values without SyncEngine needing to know Cefore exists.
    """

    payload: bytes
    expiry_s: float
    cache_time_s: float
