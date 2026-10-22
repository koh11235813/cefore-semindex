"""Tests for the SQLite-backed Index record store."""

from __future__ import annotations

from pathlib import Path

from semindex import schema
from semindex.store import OriginProgress, Store


def _payload_and_hash(seq: int, origin: str = "h0") -> tuple[bytes, str]:
    record = schema.IndexRecord(
        origin=origin,
        seq=seq,
        content_name=f"ccnx:/content/{origin}/{seq}",
        zone="zone1",
        disaster="flood",
        media="image",
        observed_at=1000 + seq,
    )
    payload = schema.encode(record)
    return payload, schema.content_hash(payload)


def test_gaps_1_2_4_leave_contiguous_2_and_4_sparse(tmp_path: Path):
    store = Store(tmp_path / "store.db")
    for seq in (1, 2, 4):
        payload, digest = _payload_and_hash(seq)
        result = store.append_received("h0", seq, payload, digest)
        assert result.stored is True
    assert store.get_progress("h0") == OriginProgress(hi=2, extra=(4,))


def test_reordering_absorbs_extra_once_gap_fills(tmp_path: Path):
    store = Store(tmp_path / "store.db")
    payload2, hash2 = _payload_and_hash(2)
    payload1, hash1 = _payload_and_hash(1)
    store.append_received("h0", 2, payload2, hash2)
    assert store.get_progress("h0") == OriginProgress(hi=0, extra=(2,))
    store.append_received("h0", 1, payload1, hash1)
    assert store.get_progress("h0") == OriginProgress(hi=2, extra=())


def test_duplicate_append_is_idempotent(tmp_path: Path):
    store = Store(tmp_path / "store.db")
    payload, digest = _payload_and_hash(1)
    first = store.append_received("h0", 1, payload, digest)
    second = store.append_received("h0", 1, payload, digest)
    assert first == type(first)(stored=True, duplicate=False, conflict=False)
    assert second == type(second)(stored=False, duplicate=True, conflict=False)
    assert store.get_progress("h0") == OriginProgress(hi=1, extra=())


def test_conflict_keeps_first_and_reports_it(tmp_path: Path):
    store = Store(tmp_path / "store.db")
    payload_a, hash_a = _payload_and_hash(1)
    first = store.append_received("h0", 1, payload_a, hash_a)
    assert first.stored is True

    record_b = schema.IndexRecord(
        origin="h0",
        seq=1,
        content_name="ccnx:/content/h0/1-different",
        zone="zone2",
        disaster="fire",
        media="video",
        observed_at=9999,
    )
    payload_b = schema.encode(record_b)
    hash_b = schema.content_hash(payload_b)
    assert hash_b != hash_a

    second = store.append_received("h0", 1, payload_b, hash_b)
    assert second.stored is False
    assert second.conflict is True
    assert second.existing_hash == hash_a
    # The first write is kept, verbatim.
    assert store.get_record("h0", 1) == payload_a


def test_restart_does_not_collide_seqs_or_advance_progress_past_records(tmp_path: Path):
    db_path = tmp_path / "store.db"
    store = Store(db_path)
    payload1, hash1 = _payload_and_hash(1, origin="h0")
    payload2, hash2 = _payload_and_hash(2, origin="h0")
    store.append_received("h0", 1, payload1, hash1)
    store.append_received("h0", 2, payload2, hash2)
    store.close()

    reopened = Store(db_path)
    # Progress survived the restart exactly as it was written.
    assert reopened.get_progress("h0") == OriginProgress(hi=2, extra=())

    # append_own for this same origin must allocate seq 3, not collide
    # with 1 or 2, by reading the actual max from the records table.
    record = reopened.append_own(
        origin="h0",
        content_name="ccnx:/content/h0/3",
        zone="zone1",
        disaster="flood",
        media="image",
        observed_at=1003,
    )
    assert record.seq == 3
    # Progress is updated in the same transaction as the insert, so it
    # cannot be ahead of what records actually hold.
    assert reopened.get_progress("h0") == OriginProgress(hi=3, extra=())
    assert reopened.get_record("h0", 3) is not None


def test_append_own_allocates_sequentially_from_a_fresh_store(tmp_path: Path):
    store = Store(tmp_path / "store.db")
    first = store.append_own(
        origin="h1", content_name="a", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    second = store.append_own(
        origin="h1", content_name="b", zone="zone1", disaster="flood", media="image", observed_at=2
    )
    assert (first.seq, second.seq) == (1, 2)
    assert store.get_progress("h1") == OriginProgress(hi=2, extra=())


def test_all_progress_covers_every_origin_held(tmp_path: Path):
    store = Store(tmp_path / "store.db")
    payload, digest = _payload_and_hash(1, origin="h0")
    store.append_received("h0", 1, payload, digest)
    store.append_own(
        origin="h1", content_name="x", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    assert store.all_progress() == {
        "h0": OriginProgress(hi=1, extra=()),
        "h1": OriginProgress(hi=1, extra=()),
    }
