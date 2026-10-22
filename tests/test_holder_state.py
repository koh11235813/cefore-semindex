"""Tests for the Holder state wire codec."""

from __future__ import annotations

import json

import pytest

from semindex import holder_state
from semindex.store import OriginProgress


def test_encode_decode_round_trip():
    """Encoding then decoding a progress mapping returns it unchanged."""
    progress = {
        "h0": OriginProgress(hi=3, extra=(5, 6)),
        "h1": OriginProgress(hi=0, extra=()),
    }
    payload = holder_state.encode(progress)
    decoded = holder_state.decode(payload)
    assert decoded == {
        "h0": {"hi": 3, "extra": [5, 6], "truncated": False},
        "h1": {"hi": 0, "extra": [], "truncated": False},
    }


def test_encode_is_canonical_json_sorted_keys_no_whitespace():
    """The wire encoding is exactly sorted-key, whitespace-free JSON, byte for byte."""
    progress = {"h0": OriginProgress(hi=1, extra=())}
    payload = holder_state.encode(progress)
    assert payload == b'{"h0":{"extra":[],"hi":1,"truncated":false}}'


def test_encode_truncates_extra_over_64_keeping_the_lowest():
    """An origin with more than 64 sparse seqs is truncated to the lowest 64, with truncated set."""
    extra = tuple(range(2, 2 + 70))  # 70 sparse seqs above hi
    progress = {"h0": OriginProgress(hi=1, extra=extra)}
    payload = holder_state.encode(progress)
    decoded = holder_state.decode(payload)
    assert decoded["h0"]["truncated"] is True
    assert decoded["h0"]["extra"] == list(range(2, 2 + 64))
    assert len(decoded["h0"]["extra"]) == 64


def test_encode_sends_hi_even_when_extra_is_fully_truncated():
    """hi is still sent even when every one of that origin's extra entries is dropped."""
    extra = tuple(range(100, 200))
    progress = {"h0": OriginProgress(hi=42, extra=extra)}
    decoded = holder_state.decode(holder_state.encode(progress))
    assert decoded["h0"]["hi"] == 42


def test_decode_rejects_oversize_payload_without_parsing_it():
    """A payload over the 8 KiB limit is rejected by size before any JSON parsing happens."""
    oversize = b"{" + b"x" * 9000
    with pytest.raises(holder_state.HolderStateError) as exc_info:
        holder_state.decode(oversize)
    assert str(exc_info.value) == "holder state payload is 9001 bytes, over the 8192 byte limit"


def test_decode_rejects_malformed_json():
    """Non-JSON bytes are rejected as malformed, not propagated as a JSON decode error."""
    with pytest.raises(holder_state.HolderStateError) as exc_info:
        holder_state.decode(b"not json")
    assert str(exc_info.value) == "holder state payload is not valid JSON"


def test_decode_rejects_extra_over_64_entries_from_a_peer():
    """A peer's extra list longer than 64 is rejected outright, never silently re-truncated on receipt."""
    body = {"h0": {"hi": 0, "extra": list(range(1, 66)), "truncated": False}}
    payload = json.dumps(body).encode("ascii")
    with pytest.raises(holder_state.HolderStateError) as exc_info:
        holder_state.decode(payload)
    assert str(exc_info.value) == "extra for origin 'h0' has more than 64 entries"


def test_decode_rejects_unsorted_extra():
    """A peer's extra list that is not sorted is rejected as malformed."""
    body = {"h0": {"hi": 0, "extra": [3, 1, 2], "truncated": False}}
    payload = json.dumps(body).encode("ascii")
    with pytest.raises(holder_state.HolderStateError) as exc_info:
        holder_state.decode(payload)
    assert str(exc_info.value) == "extra for origin 'h0' must be sorted and unique"


def test_decode_rejects_missing_entry_key():
    """An entry missing one of hi/extra/truncated is rejected as malformed."""
    body = {"h0": {"hi": 0, "extra": []}}
    payload = json.dumps(body).encode("ascii")
    with pytest.raises(holder_state.HolderStateError) as exc_info:
        holder_state.decode(payload)
    assert str(exc_info.value) == "each holder state entry must have exactly hi, extra and truncated"


def test_decode_rejects_origin_key_containing_slash():
    """An origin key containing '/' is rejected, since it is a wire name component wherever it is used downstream."""
    # 2026-10-22 fix regression: an origin was only checked for
    # non-emptiness, so one containing '/' decoded successfully here
    # and later crashed names.build_record_name uncaught while planning
    # a fetch for it (see test_engine.py's on_state regression test for
    # the full receive-path failure this caused).
    body = {"bad/origin": {"hi": 0, "extra": [], "truncated": False}}
    payload = json.dumps(body).encode("ascii")
    with pytest.raises(holder_state.HolderStateError) as exc_info:
        holder_state.decode(payload)
    assert str(exc_info.value) == "holder state origin keys must be non-empty name components without '/'"


def test_encode_truncates_extras_origin_by_origin_once_overall_size_exceeds_8kib():
    """When every origin's extra individually fits but the whole payload does not, extras are dropped origin by origin in sorted order until it fits, never dropping hi."""
    # Every one of these origins individually fits the 64-entry cap, but
    # encoding all 40 together exceeds MAX_ENCODED_BYTES, so encode()
    # must fall back to emptying extras origin by origin, in sorted
    # origin-name order, until the whole payload fits. hi must never be
    # dropped, and an origin that was not touched must keep its full
    # extra list intact.
    origins = [f"o{i:02d}" for i in range(40)]
    progress = {name: OriginProgress(hi=0, extra=tuple(range(1, 65))) for name in origins}
    payload = holder_state.encode(progress)
    assert len(payload) <= holder_state.MAX_ENCODED_BYTES
    decoded = holder_state.decode(payload)

    assert all(decoded[name]["hi"] == 0 for name in origins)
    truncated_names = [name for name in origins if decoded[name]["truncated"]]
    kept_names = [name for name in origins if not decoded[name]["truncated"]]
    # The fallback loop walks sorted(body), so the dropped origins are
    # an exact sorted-order prefix of the full set -- pinned exactly
    # rather than just checked non-empty, so this fails if the loop is
    # ever removed (the full, untruncated encoding of all 40 origins is
    # over the 8192-byte limit; the truncated result above is 8126
    # bytes, so without the loop encode() would raise HolderStateError
    # instead of returning a smaller payload).
    assert truncated_names == ["o00", "o01", "o02", "o03", "o04"]
    assert kept_names == origins[5:]
    for name in truncated_names:
        assert decoded[name]["extra"] == []
    for name in kept_names:
        assert decoded[name]["extra"] == list(range(1, 65))


def test_encode_raises_when_state_cannot_fit_even_with_every_extra_dropped():
    """When even every extra emptied cannot bring the payload under the limit, encode raises rather than silently paging."""
    # Enough origins, each with a long-enough name, that even emptying
    # every extra (map.md "ページングは無し": there is no further
    # fallback) cannot bring the payload under MAX_ENCODED_BYTES.
    origins = [f"origin-{i:04d}" for i in range(400)]
    progress = {name: OriginProgress(hi=0, extra=()) for name in origins}
    with pytest.raises(holder_state.HolderStateError) as exc_info:
        holder_state.encode(progress)
    assert str(exc_info.value) == (
        f"holder state for {len(origins)} origins cannot fit in "
        f"{holder_state.MAX_ENCODED_BYTES} bytes even with all extras dropped"
    )
