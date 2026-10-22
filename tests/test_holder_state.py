"""Tests for the Holder state wire codec."""

from __future__ import annotations

import json

import pytest

from semindex import holder_state
from semindex.store import OriginProgress


def test_encode_decode_round_trip():
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
    progress = {"h0": OriginProgress(hi=1, extra=())}
    payload = holder_state.encode(progress)
    assert payload == b'{"h0":{"extra":[],"hi":1,"truncated":false}}'


def test_encode_truncates_extra_over_64_keeping_the_lowest():
    extra = tuple(range(2, 2 + 70))  # 70 sparse seqs above hi
    progress = {"h0": OriginProgress(hi=1, extra=extra)}
    payload = holder_state.encode(progress)
    decoded = holder_state.decode(payload)
    assert decoded["h0"]["truncated"] is True
    assert decoded["h0"]["extra"] == list(range(2, 2 + 64))
    assert len(decoded["h0"]["extra"]) == 64


def test_encode_sends_hi_even_when_extra_is_fully_truncated():
    extra = tuple(range(100, 200))
    progress = {"h0": OriginProgress(hi=42, extra=extra)}
    decoded = holder_state.decode(holder_state.encode(progress))
    assert decoded["h0"]["hi"] == 42


def test_decode_rejects_oversize_payload_without_parsing_it():
    oversize = b"{" + b"x" * 9000
    with pytest.raises(holder_state.HolderStateError) as exc_info:
        holder_state.decode(oversize)
    assert str(exc_info.value) == "holder state payload is 9001 bytes, over the 8192 byte limit"


def test_decode_rejects_malformed_json():
    with pytest.raises(holder_state.HolderStateError) as exc_info:
        holder_state.decode(b"not json")
    assert str(exc_info.value) == "holder state payload is not valid JSON"


def test_decode_rejects_extra_over_64_entries_from_a_peer():
    body = {"h0": {"hi": 0, "extra": list(range(1, 66)), "truncated": False}}
    payload = json.dumps(body).encode("ascii")
    with pytest.raises(holder_state.HolderStateError) as exc_info:
        holder_state.decode(payload)
    assert str(exc_info.value) == "extra for origin 'h0' has more than 64 entries"


def test_decode_rejects_unsorted_extra():
    body = {"h0": {"hi": 0, "extra": [3, 1, 2], "truncated": False}}
    payload = json.dumps(body).encode("ascii")
    with pytest.raises(holder_state.HolderStateError) as exc_info:
        holder_state.decode(payload)
    assert str(exc_info.value) == "extra for origin 'h0' must be sorted and unique"


def test_decode_rejects_missing_entry_key():
    body = {"h0": {"hi": 0, "extra": []}}
    payload = json.dumps(body).encode("ascii")
    with pytest.raises(holder_state.HolderStateError) as exc_info:
        holder_state.decode(payload)
    assert str(exc_info.value) == "each holder state entry must have exactly hi, extra and truncated"
