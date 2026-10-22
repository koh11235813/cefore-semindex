"""Wire names for the Sync protocol: holder-scoped state and record names.

This module is the single place that builds and parses them, so the
format changes in one place if it ever has to. See the map's "Wire names
(fixed)" decision for the authoritative grammar.

Grammar (ccnx:/semidx/<group>/<holder>/...):
    state name:  ccnx:/semidx/<group>/<holder>/sv/<requester>/<epoch_ms>
    record name: ccnx:/semidx/<group>/<holder>/rec/<origin>/<seq>

`<holder>` is whoever is being asked to answer: a direct neighbor for a
state request, and either a direct neighbor or, when relaying, whichever
holder is known to have the record -- never necessarily the record's
Index origin (ADR-0002). A node registers exactly one prefix for itself
as holder:
    ccnx:/semidx/<group>/<self>
"""

from __future__ import annotations

from dataclasses import dataclass

PREFIX = "ccnx:/semidx"


class MalformedNameError(ValueError):
    """A wire name did not match the fixed grammar above."""


@dataclass(frozen=True)
class StateName:
    """A parsed state-request name: holder's progress, as seen by `requester` at `epoch_ms`."""

    group: str
    holder: str
    requester: str
    epoch_ms: int


@dataclass(frozen=True)
class RecordName:
    """A parsed record name: `holder`'s copy of (origin, seq)."""

    group: str
    holder: str
    origin: str
    seq: int


def _validate_component(value: str, label: str) -> None:
    """Raise MalformedNameError unless `value` is a single, non-empty path segment.

    A '/' inside it would silently change how many segments split() sees,
    which is exactly the kind of ambiguity a wire grammar cannot allow.
    """
    if not value or "/" in value:
        raise MalformedNameError(f"{label} must be a non-empty name component without '/'")


def _parse_int_component(value: str, label: str) -> int:
    """Parse `value` as a plain decimal integer, or raise MalformedNameError.

    Requires plain decimal digits: int() would also accept a leading
    '+', internal '_' separators, or surrounding whitespace, none of
    which a Cefore name component should ever carry.
    """
    if not value.isdigit():
        raise MalformedNameError(f"{label} must be a non-negative decimal integer")
    return int(value)


def registration_prefix(group: str, node: str) -> str:
    """The single prefix a node registers for itself as holder."""
    _validate_component(group, "group")
    _validate_component(node, "node")
    return f"{PREFIX}/{group}/{node}"


def build_state_name(group: str, holder: str, requester: str, epoch_ms: int) -> str:
    """Build a state-request name: `requester` asking `holder` for its progress."""
    _validate_component(group, "group")
    _validate_component(holder, "holder")
    _validate_component(requester, "requester")
    if epoch_ms < 0:
        raise MalformedNameError("epoch_ms must be non-negative")
    return f"{PREFIX}/{group}/{holder}/sv/{requester}/{epoch_ms}"


def build_record_name(group: str, holder: str, origin: str, seq: int) -> str:
    """Build a record name: `holder`'s copy of (origin, seq)."""
    _validate_component(group, "group")
    _validate_component(holder, "holder")
    _validate_component(origin, "origin")
    if seq < 1:
        raise MalformedNameError("seq must be >= 1")
    return f"{PREFIX}/{group}/{holder}/rec/{origin}/{seq}"


def parse(name: str) -> StateName | RecordName:
    """Parse any Sync protocol name, dispatching on its kind segment (sv/rec)."""
    if not name.startswith(PREFIX + "/"):
        raise MalformedNameError(f"name must start with '{PREFIX}/'")
    rest = name[len(PREFIX) + 1 :]
    segments = rest.split("/")
    if len(segments) != 5:
        raise MalformedNameError("name must have exactly 5 segments after the fixed prefix")
    group, holder, kind, third, fourth = segments
    _validate_component(group, "group")
    _validate_component(holder, "holder")
    if kind == "sv":
        _validate_component(third, "requester")
        epoch_ms = _parse_int_component(fourth, "epoch_ms")
        return StateName(group=group, holder=holder, requester=third, epoch_ms=epoch_ms)
    if kind == "rec":
        _validate_component(third, "origin")
        seq = _parse_int_component(fourth, "seq")
        if seq < 1:
            raise MalformedNameError("seq must be >= 1")
        return RecordName(group=group, holder=holder, origin=third, seq=seq)
    raise MalformedNameError("name kind segment must be 'sv' or 'rec'")
