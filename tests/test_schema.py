"""Tests for the Index record schema and canonical JSON wire codec."""

from __future__ import annotations

import pytest

from semindex import schema


def _make_record(**overrides) -> schema.IndexRecord:
    """Build an IndexRecord with sensible defaults, overridable per field."""
    fields = dict(
        origin="h0",
        seq=1,
        content_name="ccnx:/content/h0/1",
        zone="zone1",
        disaster="flood",
        media="image",
        observed_at=1000,
    )
    fields.update(overrides)
    return schema.IndexRecord(**fields)


def test_encode_decode_round_trip():
    """Encoding then decoding a record returns it unchanged."""
    record = _make_record()
    payload = schema.encode(record)
    assert schema.decode(payload) == record


def test_canonical_bytes_are_deterministic_regardless_of_field_order():
    """canonical_bytes depends only on the record's value, never on the order its fields were constructed in."""
    record = _make_record()
    # Build an equal record via keyword args in a different order; the
    # dataclass is the same value either way, and canonical_bytes must
    # not depend on anything but the value.
    reordered = schema.IndexRecord(
        media="image",
        disaster="flood",
        zone="zone1",
        seq=1,
        origin="h0",
        content_name="ccnx:/content/h0/1",
        observed_at=1000,
    )
    assert schema.canonical_bytes(record) == schema.canonical_bytes(reordered)


def test_content_hash_changes_when_payload_changes():
    """Two different records hash to two different content hashes."""
    record = _make_record()
    other = _make_record(seq=2)
    assert schema.content_hash(schema.canonical_bytes(record)) != schema.content_hash(schema.canonical_bytes(other))


def test_validate_rejects_unknown_zone():
    """A zone outside the fixed vocabulary is rejected with an exact diagnostic."""
    record = _make_record(zone="not-a-zone")
    with pytest.raises(schema.RecordError) as exc_info:
        schema.validate(record)
    assert str(exc_info.value) == "zone must be one of ['zone1', 'zone2', 'zone3', 'zone4', 'zone5']"


def test_validate_rejects_unknown_disaster():
    """A disaster outside the fixed vocabulary is rejected with an exact diagnostic."""
    record = _make_record(disaster="earthquake")
    with pytest.raises(schema.RecordError) as exc_info:
        schema.validate(record)
    assert str(exc_info.value) == "disaster must be one of ['collapse', 'fire', 'flood', 'landslide', 'unknown']"


def test_validate_rejects_unknown_media():
    """A media value outside the fixed vocabulary is rejected with an exact diagnostic."""
    record = _make_record(media="audio")
    with pytest.raises(schema.RecordError) as exc_info:
        schema.validate(record)
    assert str(exc_info.value) == "media must be one of ['image', 'sensor', 'video']"


def test_validate_rejects_seq_below_one():
    """seq below 1 is rejected with an exact diagnostic."""
    record = _make_record(seq=0)
    with pytest.raises(schema.RecordError) as exc_info:
        schema.validate(record)
    assert str(exc_info.value) == "seq must be an integer >= 1"


def test_validate_rejects_supersedes_pointing_at_self():
    """A record whose supersedes names its own (origin, seq) is rejected."""
    record = _make_record(seq=5, supersedes=schema.RecordKey(origin="h0", seq=5))
    with pytest.raises(schema.RecordError) as exc_info:
        schema.validate(record)
    assert str(exc_info.value) == "supersedes must not reference the record's own key"


@pytest.mark.parametrize(
    "field,value,expected_message",
    [
        # 2026-10-22 fix regression: `value not in <frozenset>` hashes
        # its operand first, so an unhashable value here used to raise
        # an uncaught TypeError instead of RecordError.
        ("zone", [], "zone must be one of ['zone1', 'zone2', 'zone3', 'zone4', 'zone5']"),
        ("zone", {}, "zone must be one of ['zone1', 'zone2', 'zone3', 'zone4', 'zone5']"),
        ("disaster", [], "disaster must be one of ['collapse', 'fire', 'flood', 'landslide', 'unknown']"),
        ("disaster", {}, "disaster must be one of ['collapse', 'fire', 'flood', 'landslide', 'unknown']"),
        ("media", [], "media must be one of ['image', 'sensor', 'video']"),
        ("media", {}, "media must be one of ['image', 'sensor', 'video']"),
    ],
)
def test_validate_rejects_non_string_attribute_as_record_error_not_type_error(field, value, expected_message):
    """An unhashable attrs value (list or dict) is rejected as RecordError, not an uncaught TypeError from the frozenset membership test."""
    record = _make_record(**{field: value})
    with pytest.raises(schema.RecordError) as exc_info:
        schema.validate(record)
    assert str(exc_info.value) == expected_message


def test_encode_rejects_oversize_record():
    """A record whose canonical encoding exceeds the 2 KiB wire limit is rejected at encode time."""
    # A content_name long enough to push the canonical encoding past the
    # 2 KiB wire limit.
    record = _make_record(content_name="ccnx:/content/" + "x" * 3000)
    encoded_size = len(schema.canonical_bytes(record))
    with pytest.raises(schema.RecordError) as exc_info:
        schema.encode(record)
    assert str(exc_info.value) == f"encoded record is {encoded_size} bytes, over the 2048 byte limit"


def test_decode_rejects_oversize_payload_without_parsing_it():
    """A payload over the 2 KiB limit is rejected by size before any JSON parsing happens."""
    oversize = b"{" + b"x" * 3000
    with pytest.raises(schema.RecordError) as exc_info:
        schema.decode(oversize)
    assert str(exc_info.value) == "record payload is 3001 bytes, over the 2048 byte limit"


def test_decode_rejects_malformed_json():
    """Non-JSON bytes are rejected as malformed, not propagated as a JSON decode error."""
    with pytest.raises(schema.RecordError) as exc_info:
        schema.decode(b"not json")
    assert str(exc_info.value) == "record payload is not valid JSON"


def test_decode_rejects_missing_key():
    """A payload missing one of the fixed top-level keys is rejected as malformed."""
    record = _make_record()
    payload = schema.encode(record)
    import json

    obj = json.loads(payload)
    del obj["model_id"]
    truncated = json.dumps(obj).encode("ascii")
    with pytest.raises(schema.RecordError) as exc_info:
        schema.decode(truncated)
    assert str(exc_info.value) == (
        "record payload must have exactly the keys "
        "['attrs', 'content_name', 'model_id', 'observed_at', 'origin', 'schema_version', 'seq', 'supersedes']"
    )


def test_decode_rejects_name_payload_type_mismatch():
    """A field of the wrong JSON type (a numeric string for seq) is rejected as malformed, not coerced."""
    record = _make_record()
    payload = schema.encode(record)
    import json

    obj = json.loads(payload)
    obj["seq"] = "1"  # seq must be an int, not a numeric string
    bad = json.dumps(obj).encode("ascii")
    with pytest.raises(schema.RecordError) as exc_info:
        schema.decode(bad)
    assert str(exc_info.value) == "seq must be an integer"


def test_decode_accepts_a_supersedes_record_key():
    """A non-null supersedes decodes back to the exact RecordKey it was encoded from."""
    record = _make_record(seq=2, supersedes=schema.RecordKey(origin="h0", seq=1))
    payload = schema.encode(record)
    assert schema.decode(payload).supersedes == schema.RecordKey(origin="h0", seq=1)


def test_decode_rejects_array_valued_attribute_as_record_error_not_type_error():
    """A crafted payload with a non-string attrs value is rejected as RecordError, the same path engine.on_record relies on, not an uncaught TypeError."""
    # The attack path the validate()-level tests above exercise directly:
    # a crafted wire payload reaching decode() with a non-string attrs
    # value must come back as RecordError, which engine.on_record
    # catches, not an uncaught TypeError that would crash the agent.
    record = _make_record()
    payload = schema.encode(record)
    import json

    obj = json.loads(payload)
    obj["attrs"]["zone"] = []
    bad = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(schema.RecordError) as exc_info:
        schema.decode(bad)
    assert str(exc_info.value) == "zone must be one of ['zone1', 'zone2', 'zone3', 'zone4', 'zone5']"


def test_decode_rejects_payload_with_incidental_whitespace():
    """A byte-for-byte different (non-canonical) encoding of a valid record is rejected outright, never re-canonicalized and accepted."""
    # 2026-10-22 fix regression: decode() used to accept any JSON
    # encoding of a valid record, then content_hash/the store hashed
    # and kept the *received* bytes -- so the same record re-sent with
    # different incidental whitespace would decode equal but hash
    # differently, and look like a conflict against itself.
    record = _make_record()
    canonical = schema.canonical_bytes(record)
    import json

    noncanonical = json.dumps(json.loads(canonical), sort_keys=True).encode("ascii")  # default separators add spaces
    assert noncanonical != canonical
    with pytest.raises(schema.RecordError) as exc_info:
        schema.decode(noncanonical)
    assert str(exc_info.value) == "record payload is not canonical JSON"


def test_decode_rejects_payload_with_reordered_keys():
    """A payload with the same keys and values in a different order is rejected outright, never re-canonicalized and accepted."""
    record = _make_record()
    canonical = schema.canonical_bytes(record)
    import json

    obj = json.loads(canonical)
    reordered = {key: obj[key] for key in reversed(list(obj.keys()))}
    noncanonical = json.dumps(reordered, separators=(",", ":")).encode("ascii")
    assert noncanonical != canonical
    with pytest.raises(schema.RecordError) as exc_info:
        schema.decode(noncanonical)
    assert str(exc_info.value) == "record payload is not canonical JSON"
