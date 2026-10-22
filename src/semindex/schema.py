"""Index record schema, vocabulary and canonical JSON wire codec.

An Index record is the Semantic Index's unit of synchronization
(CONTEXT.md "Index record"): produced once by its Index origin and never
mutated afterward. A correction is a new record whose `supersedes` field
names the record it replaces; it is never an overwrite. This module owns
the record's shape, the fixed semantic-attribute vocabulary, and the
canonical byte encoding used both as the Cefore Data payload and as the
input to `content_hash`.

Validation here is schema validation only (map.md "検証はスキーマ検証と
衝突検出まで"): there is no signature or key distribution in this
baseline, so a record that passes `validate` is schema-valid, not
authenticated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

# Fixed vocabulary for the single-valued semantic attributes (map.md
# "Index record と永続化"). There is no mechanism in this baseline for
# extending the vocabulary at runtime; any other value is rejected.
ZONES = frozenset({"zone1", "zone2", "zone3", "zone4", "zone5"})
DISASTERS = frozenset({"flood", "fire", "landslide", "collapse", "unknown"})
MEDIA = frozenset({"image", "video", "sensor"})

SCHEMA_VERSION = 1
DEFAULT_MODEL_ID = "sample-v0"

# Wire limit for one encoded record (map.md "1 件 2 KiB 以下").
MAX_RECORD_BYTES = 2048

# The record's wire JSON must have exactly these top-level keys, and
# `attrs` exactly these keys; any other shape is malformed, not a record
# with unknown-but-ignorable extensions (there is no versioning story in
# this baseline beyond `schema_version`).
_TOP_LEVEL_KEYS = frozenset(
    {"origin", "seq", "content_name", "attrs", "observed_at", "schema_version", "model_id", "supersedes"}
)
_ATTRS_KEYS = frozenset({"zone", "disaster", "media"})
_SUPERSEDES_KEYS = frozenset({"origin", "seq"})


class RecordError(ValueError):
    """A record failed validation, decoding, or a wire size limit."""


@dataclass(frozen=True)
class RecordKey:
    """The (origin, seq) pair that identifies one Index record.

    This is also the shape `supersedes` points at: a correction names the
    record it replaces by this key, not by content_name. The field is
    part of the wire format now even though no effort in this baseline
    produces a non-null value, so a later effort does not have to migrate
    the wire format to add it.
    """

    origin: str
    seq: int


@dataclass(frozen=True)
class IndexRecord:
    """One immutable Index record.

    `content_hash` is deliberately not a field here: map.md requires it
    be computed by the receiver from the canonical bytes ("content_hash
    は受信側がこのバイト列の SHA-256 として計算する（レコードには入れな
    い）"), so it can never be forged by the sender nor drift from the
    payload that produced it.
    """

    origin: str
    seq: int
    content_name: str
    zone: str
    disaster: str
    media: str
    observed_at: int
    schema_version: int = SCHEMA_VERSION
    model_id: str = DEFAULT_MODEL_ID
    supersedes: RecordKey | None = None

    @property
    def key(self) -> RecordKey:
        """This record's own (origin, seq) identity."""
        return RecordKey(self.origin, self.seq)


def validate(record: IndexRecord) -> None:
    """Raise RecordError if `record` violates the fixed vocabulary or shape.

    Called on every record this node creates or receives before it can
    reach the store, so the store never holds a record that would fail
    re-validation after a restart.
    """
    if not isinstance(record.origin, str) or not record.origin:
        raise RecordError("origin must be a non-empty string")
    if not isinstance(record.seq, int) or isinstance(record.seq, bool) or record.seq < 1:
        raise RecordError("seq must be an integer >= 1")
    if not isinstance(record.content_name, str) or not record.content_name:
        raise RecordError("content_name must be a non-empty string")
    if record.zone not in ZONES:
        raise RecordError(f"zone must be one of {sorted(ZONES)}")
    if record.disaster not in DISASTERS:
        raise RecordError(f"disaster must be one of {sorted(DISASTERS)}")
    if record.media not in MEDIA:
        raise RecordError(f"media must be one of {sorted(MEDIA)}")
    if (
        not isinstance(record.observed_at, int)
        or isinstance(record.observed_at, bool)
        or record.observed_at < 0
    ):
        raise RecordError("observed_at must be a non-negative integer epoch millisecond")
    if record.schema_version != SCHEMA_VERSION:
        raise RecordError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(record.model_id, str) or not record.model_id:
        raise RecordError("model_id must be a non-empty string")
    if record.supersedes is not None:
        if record.supersedes.origin == record.origin and record.supersedes.seq == record.seq:
            raise RecordError("supersedes must not reference the record's own key")


def _to_wire_dict(record: IndexRecord) -> dict:
    """Build the plain-dict shape `canonical_bytes` serializes, nesting attrs and supersedes."""
    if record.supersedes is None:
        supersedes = None
    else:
        supersedes = {"origin": record.supersedes.origin, "seq": record.supersedes.seq}
    return {
        "origin": record.origin,
        "seq": record.seq,
        "content_name": record.content_name,
        "attrs": {"zone": record.zone, "disaster": record.disaster, "media": record.media},
        "observed_at": record.observed_at,
        "schema_version": record.schema_version,
        "model_id": record.model_id,
        "supersedes": supersedes,
    }


def canonical_bytes(record: IndexRecord) -> bytes:
    """Encode `record` as canonical wire JSON: sorted keys, no incidental whitespace.

    This is the exact byte sequence `content_hash` is computed over, so
    any change here (key order, separators, ascii-ness) changes every
    hash downstream; it must never depend on dict iteration order.
    """
    payload = _to_wire_dict(record)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def content_hash(data: bytes) -> str:
    """SHA-256 hex digest of canonical record bytes, computed by the receiver."""
    return hashlib.sha256(data).hexdigest()


def encode(record: IndexRecord) -> bytes:
    """Validate `record` and return its canonical wire bytes.

    Raises RecordError if the record is invalid, or if its canonical
    encoding would exceed MAX_RECORD_BYTES (map.md "1 件 2 KiB 以下").
    """
    validate(record)
    data = canonical_bytes(record)
    if len(data) > MAX_RECORD_BYTES:
        raise RecordError(f"encoded record is {len(data)} bytes, over the {MAX_RECORD_BYTES} byte limit")
    return data


def _require(condition: bool, message: str) -> None:
    """Raise RecordError(message) unless `condition` holds."""
    if not condition:
        raise RecordError(message)


def decode(data: bytes) -> IndexRecord:
    """Parse and validate wire bytes into an IndexRecord.

    Rejects oversize input before attempting to parse it -- a peer must
    not be able to force JSON parsing of an arbitrarily large payload --
    then rejects malformed JSON, a wrong key set, a field of the wrong
    type, and finally anything `validate` rejects.
    """
    if len(data) > MAX_RECORD_BYTES:
        raise RecordError(f"record payload is {len(data)} bytes, over the {MAX_RECORD_BYTES} byte limit")
    try:
        obj = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordError("record payload is not valid JSON") from exc
    _require(isinstance(obj, dict), "record payload must be a JSON object")
    _require(
        set(obj.keys()) == _TOP_LEVEL_KEYS,
        f"record payload must have exactly the keys {sorted(_TOP_LEVEL_KEYS)}",
    )

    attrs = obj["attrs"]
    _require(
        isinstance(attrs, dict) and set(attrs.keys()) == _ATTRS_KEYS,
        "attrs must be an object with exactly zone, disaster and media",
    )

    supersedes_obj = obj["supersedes"]
    supersedes: RecordKey | None
    if supersedes_obj is None:
        supersedes = None
    else:
        _require(
            isinstance(supersedes_obj, dict) and set(supersedes_obj.keys()) == _SUPERSEDES_KEYS,
            "supersedes must be null or an object with exactly origin and seq",
        )
        _require(isinstance(supersedes_obj["origin"], str), "supersedes.origin must be a string")
        _require(
            isinstance(supersedes_obj["seq"], int) and not isinstance(supersedes_obj["seq"], bool),
            "supersedes.seq must be an integer",
        )
        supersedes = RecordKey(supersedes_obj["origin"], supersedes_obj["seq"])

    _require(isinstance(obj["origin"], str), "origin must be a string")
    _require(isinstance(obj["seq"], int) and not isinstance(obj["seq"], bool), "seq must be an integer")
    _require(isinstance(obj["content_name"], str), "content_name must be a string")
    _require(
        isinstance(obj["observed_at"], int) and not isinstance(obj["observed_at"], bool),
        "observed_at must be an integer",
    )
    _require(
        isinstance(obj["schema_version"], int) and not isinstance(obj["schema_version"], bool),
        "schema_version must be an integer",
    )
    _require(isinstance(obj["model_id"], str), "model_id must be a string")

    record = IndexRecord(
        origin=obj["origin"],
        seq=obj["seq"],
        content_name=obj["content_name"],
        zone=attrs["zone"],
        disaster=attrs["disaster"],
        media=attrs["media"],
        observed_at=obj["observed_at"],
        schema_version=obj["schema_version"],
        model_id=obj["model_id"],
        supersedes=supersedes,
    )
    validate(record)
    return record
