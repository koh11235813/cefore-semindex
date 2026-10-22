"""Tests for the Index record schema and canonical JSON wire codec."""

from __future__ import annotations

import pytest

from semindex import schema


def _make_record(**overrides) -> schema.IndexRecord:
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
    record = _make_record()
    payload = schema.encode(record)
    assert schema.decode(payload) == record


def test_canonical_bytes_are_deterministic_regardless_of_field_order():
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
    record = _make_record()
    other = _make_record(seq=2)
    assert schema.content_hash(schema.canonical_bytes(record)) != schema.content_hash(schema.canonical_bytes(other))


def test_validate_rejects_unknown_zone():
    record = _make_record(zone="not-a-zone")
    with pytest.raises(schema.RecordError) as exc_info:
        schema.validate(record)
    assert str(exc_info.value) == "zone must be one of ['zone1', 'zone2', 'zone3', 'zone4', 'zone5']"


def test_validate_rejects_unknown_disaster():
    record = _make_record(disaster="earthquake")
    with pytest.raises(schema.RecordError) as exc_info:
        schema.validate(record)
    assert str(exc_info.value) == "disaster must be one of ['collapse', 'fire', 'flood', 'landslide', 'unknown']"


def test_validate_rejects_unknown_media():
    record = _make_record(media="audio")
    with pytest.raises(schema.RecordError) as exc_info:
        schema.validate(record)
    assert str(exc_info.value) == "media must be one of ['image', 'sensor', 'video']"


def test_validate_rejects_seq_below_one():
    record = _make_record(seq=0)
    with pytest.raises(schema.RecordError) as exc_info:
        schema.validate(record)
    assert str(exc_info.value) == "seq must be an integer >= 1"


def test_validate_rejects_supersedes_pointing_at_self():
    record = _make_record(seq=5, supersedes=schema.RecordKey(origin="h0", seq=5))
    with pytest.raises(schema.RecordError) as exc_info:
        schema.validate(record)
    assert str(exc_info.value) == "supersedes must not reference the record's own key"


def test_encode_rejects_oversize_record():
    # A content_name long enough to push the canonical encoding past the
    # 2 KiB wire limit.
    record = _make_record(content_name="ccnx:/content/" + "x" * 3000)
    encoded_size = len(schema.canonical_bytes(record))
    with pytest.raises(schema.RecordError) as exc_info:
        schema.encode(record)
    assert str(exc_info.value) == f"encoded record is {encoded_size} bytes, over the 2048 byte limit"


def test_decode_rejects_oversize_payload_without_parsing_it():
    oversize = b"{" + b"x" * 3000
    with pytest.raises(schema.RecordError) as exc_info:
        schema.decode(oversize)
    assert str(exc_info.value) == "record payload is 3001 bytes, over the 2048 byte limit"


def test_decode_rejects_malformed_json():
    with pytest.raises(schema.RecordError) as exc_info:
        schema.decode(b"not json")
    assert str(exc_info.value) == "record payload is not valid JSON"


def test_decode_rejects_missing_key():
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
    record = _make_record(seq=2, supersedes=schema.RecordKey(origin="h0", seq=1))
    payload = schema.encode(record)
    assert schema.decode(payload).supersedes == schema.RecordKey(origin="h0", seq=1)
