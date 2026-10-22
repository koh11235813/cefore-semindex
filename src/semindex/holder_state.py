"""Codec for Holder state: the per-origin summary a holder sends a requester.

CONTEXT.md "Holder state": per origin, the highest seq held contiguously
from 1 (`hi`) and the sparse seqs held above it (`extra`), nothing about
records merely heard of. This module only encodes and decodes the wire
representation; `store.OriginProgress` is the value it encodes.
"""

from __future__ import annotations

import json
from typing import Mapping

from semindex.store import OriginProgress

# map.md "Holder state": extra は最大 64、応答は正規形 JSON で 8 KiB 以下。
MAX_EXTRA_PER_ORIGIN = 64
MAX_ENCODED_BYTES = 8192

_ENTRY_KEYS = frozenset({"hi", "extra", "truncated"})


class HolderStateError(ValueError):
    """Encoded or received holder state violated the wire limits or shape."""


def _dump(body: dict) -> bytes:
    """Serialize `body` as canonical wire JSON: sorted keys, no incidental whitespace."""
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def encode(progress: Mapping[str, OriginProgress]) -> bytes:
    """Encode a mapping of origin -> OriginProgress as canonical wire JSON.

    Each origin's `extra` is capped at MAX_EXTRA_PER_ORIGIN, keeping the
    lowest (closest-to-hi) seqs -- those are the ones that extend the
    requester's contiguous range soonest -- and setting `truncated: true`
    on that origin when any were dropped. `hi` is always sent, even when
    `extra` is truncated away entirely.

    Raises HolderStateError if the encoding still exceeds
    MAX_ENCODED_BYTES after every origin's extra has been dropped to
    empty (too many origins, or origin names too long): this baseline
    has no paging to fall back to (map.md "ページングは無し").
    """
    body: dict[str, dict] = {}
    for origin in sorted(progress):
        p = progress[origin]
        extra_sorted = tuple(sorted(p.extra))
        truncated = len(extra_sorted) > MAX_EXTRA_PER_ORIGIN
        kept = extra_sorted[:MAX_EXTRA_PER_ORIGIN]
        body[origin] = {"hi": p.hi, "extra": list(kept), "truncated": truncated}

    data = _dump(body)
    if len(data) <= MAX_ENCODED_BYTES:
        return data

    # Still too large even with each origin's extra capped at 64: drop
    # extras origin by origin, in a fixed (sorted) order, until it fits.
    for origin in sorted(body):
        if body[origin]["extra"]:
            body[origin]["extra"] = []
            body[origin]["truncated"] = True
            data = _dump(body)
            if len(data) <= MAX_ENCODED_BYTES:
                return data

    raise HolderStateError(
        f"holder state for {len(body)} origins cannot fit in {MAX_ENCODED_BYTES} bytes even with all extras dropped"
    )


def decode(data: bytes) -> dict[str, dict]:
    """Parse wire bytes into {origin: {"hi": int, "extra": [int, ...], "truncated": bool}}.

    Rejects oversize input outright, then malformed JSON or shape, then
    any origin whose extra exceeds MAX_EXTRA_PER_ORIGIN or is not sorted
    and unique: a compliant sender never produces one, so a peer that
    does is treated the same as any other malformed input, never
    silently re-truncated on our side.
    """
    if len(data) > MAX_ENCODED_BYTES:
        raise HolderStateError(f"holder state payload is {len(data)} bytes, over the {MAX_ENCODED_BYTES} byte limit")
    try:
        obj = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HolderStateError("holder state payload is not valid JSON") from exc
    if not isinstance(obj, dict):
        raise HolderStateError("holder state payload must be a JSON object")

    result: dict[str, dict] = {}
    for origin, entry in obj.items():
        if not isinstance(origin, str) or not origin:
            raise HolderStateError("holder state origin keys must be non-empty strings")
        if not isinstance(entry, dict) or set(entry.keys()) != _ENTRY_KEYS:
            raise HolderStateError("each holder state entry must have exactly hi, extra and truncated")
        hi, extra, truncated = entry["hi"], entry["extra"], entry["truncated"]
        if not isinstance(hi, int) or isinstance(hi, bool) or hi < 0:
            raise HolderStateError("hi must be a non-negative integer")
        if not isinstance(extra, list) or not all(
            isinstance(s, int) and not isinstance(s, bool) and s > hi for s in extra
        ):
            raise HolderStateError("extra must be a list of integers greater than hi")
        if len(extra) > MAX_EXTRA_PER_ORIGIN:
            raise HolderStateError(f"extra for origin {origin!r} has more than {MAX_EXTRA_PER_ORIGIN} entries")
        if extra != sorted(extra) or len(set(extra)) != len(extra):
            raise HolderStateError(f"extra for origin {origin!r} must be sorted and unique")
        if not isinstance(truncated, bool):
            raise HolderStateError("truncated must be a boolean")
        result[origin] = {"hi": hi, "extra": extra, "truncated": truncated}
    return result
