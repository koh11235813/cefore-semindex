"""Tests for the SQLite-backed Index record store."""

from __future__ import annotations

from pathlib import Path

import pytest

from semindex import schema
from semindex.store import OriginProgress, Store


def _payload_and_hash(seq: int, origin: str = "h0") -> tuple[bytes, str]:
    """Build one encoded record and its content hash for `origin`/`seq`, for tests that only need a well-formed payload."""
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
    """Receiving 1, 2 and 4 out of order leaves hi=2 contiguous and 4 held sparsely, not contiguous."""
    store = Store(tmp_path / "store.db")
    for seq in (1, 2, 4):
        payload, digest = _payload_and_hash(seq)
        result = store.append_received("h0", seq, payload, digest)
        assert result.stored is True
    assert store.get_progress("h0") == OriginProgress(hi=2, extra=(4,))


def test_reordering_absorbs_extra_once_gap_fills(tmp_path: Path):
    """Once the gap-filling seq arrives, a sparse extra entry right above it is absorbed into hi."""
    store = Store(tmp_path / "store.db")
    payload2, hash2 = _payload_and_hash(2)
    payload1, hash1 = _payload_and_hash(1)
    store.append_received("h0", 2, payload2, hash2)
    assert store.get_progress("h0") == OriginProgress(hi=0, extra=(2,))
    store.append_received("h0", 1, payload1, hash1)
    assert store.get_progress("h0") == OriginProgress(hi=2, extra=())


def test_duplicate_append_is_idempotent(tmp_path: Path):
    """Receiving the exact same (origin, seq, payload) twice reports the second as a duplicate and leaves progress unchanged."""
    store = Store(tmp_path / "store.db")
    payload, digest = _payload_and_hash(1)
    first = store.append_received("h0", 1, payload, digest)
    second = store.append_received("h0", 1, payload, digest)
    assert first == type(first)(stored=True, duplicate=False, conflict=False)
    assert second == type(second)(stored=False, duplicate=True, conflict=False)
    assert store.get_progress("h0") == OriginProgress(hi=1, extra=())


def test_conflict_keeps_first_and_reports_it(tmp_path: Path):
    """A second record at the same (origin, seq) with different content is kept out; the first write is the one still held."""
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


def test_append_own_rolls_back_the_insert_if_the_progress_fold_raises(tmp_path: Path, monkeypatch):
    """Discriminates that the record insert and the progress fold inside
    append_own run in one transaction: if the fold raises, the insert
    it followed must not have survived either -- a store.py docstring
    claim this exercises directly rather than trusting.
    """
    store = Store(tmp_path / "store.db")

    def _boom(self, origin, seq):
        """Stand-in for a progress fold that fails mid-transaction."""
        raise RuntimeError("simulated failure between the record insert and the progress fold")

    monkeypatch.setattr(Store, "_fold_progress", _boom)

    with pytest.raises(RuntimeError):
        store.append_own(
            origin="h0", content_name="a", zone="zone1", disaster="flood", media="image", observed_at=1
        )

    assert store.get_record("h0", 1) is None
    assert store.get_progress("h0") == OriginProgress(hi=0, extra=())


def test_append_received_rolls_back_the_insert_if_the_progress_fold_raises(tmp_path: Path, monkeypatch):
    """Same discrimination for the received-record path. Seeds a first
    record normally before patching, so a leaked insert of the second
    (without its progress fold) would show up as a held record at a
    seq higher than progress.hi admits -- not just an absence either
    way, which an unseeded store couldn't tell apart.
    """
    store = Store(tmp_path / "store.db")
    payload1, hash1 = _payload_and_hash(1)
    store.append_received("h0", 1, payload1, hash1)
    assert store.get_progress("h0") == OriginProgress(hi=1, extra=())

    def _boom(self, origin, seq):
        """Stand-in for a progress fold that fails mid-transaction."""
        raise RuntimeError("simulated failure between the record insert and the progress fold")

    monkeypatch.setattr(Store, "_fold_progress", _boom)

    payload2, hash2 = _payload_and_hash(2)
    with pytest.raises(RuntimeError):
        store.append_received("h0", 2, payload2, hash2)

    assert store.get_record("h0", 2) is None
    assert store.get_progress("h0") == OriginProgress(hi=1, extra=())


def test_restart_does_not_collide_seqs_or_advance_progress_past_records(tmp_path: Path):
    """Reopening the store after a restart keeps the same progress and allocates the next seq from the actual max in records, never colliding or running ahead."""
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
    """append_own allocates 1, then 2, for a fresh origin."""
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
    """all_progress reports every origin this node holds any records for, own and received alike."""
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


def _own_record(store: Store, *, origin: str, seq_hint: str, zone: str, disaster: str, media: str) -> schema.IndexRecord:
    """Insert one own record with the given attrs, for query_by_attrs tests that only care about attrs/origin."""
    return store.append_own(
        origin=origin,
        content_name=f"ccnx:/content/{origin}/{seq_hint}",
        zone=zone,
        disaster=disaster,
        media=media,
        observed_at=1,
    )


def test_query_by_attrs_with_no_filters_returns_every_record_ordered_by_origin_then_seq(tmp_path: Path):
    """No filters means every held record matches, ordered by (origin, seq) regardless of insertion order."""
    store = Store(tmp_path / "store.db")
    _own_record(store, origin="h1", seq_hint="a", zone="zone1", disaster="flood", media="image")
    _own_record(store, origin="h0", seq_hint="b", zone="zone2", disaster="fire", media="video")
    _own_record(store, origin="h0", seq_hint="c", zone="zone3", disaster="collapse", media="sensor")
    matches = store.query_by_attrs()
    assert [(m.origin, m.seq) for m in matches] == [("h0", 1), ("h0", 2), ("h1", 1)]


def test_query_by_attrs_filters_on_every_given_field(tmp_path: Path):
    """A record must match every given filter, not just one of them, to be returned."""
    store = Store(tmp_path / "store.db")
    _own_record(store, origin="h0", seq_hint="a", zone="zone1", disaster="flood", media="image")
    _own_record(store, origin="h0", seq_hint="b", zone="zone1", disaster="fire", media="image")
    matches = store.query_by_attrs(zone="zone1", disaster="flood")
    assert [(m.origin, m.seq) for m in matches] == [("h0", 1)]


def test_query_by_attrs_with_no_match_returns_empty_list(tmp_path: Path):
    """A filter combination no held record satisfies returns an empty list, not None or an error."""
    store = Store(tmp_path / "store.db")
    _own_record(store, origin="h0", seq_hint="a", zone="zone1", disaster="flood", media="image")
    assert store.query_by_attrs(zone="zone5") == []


def test_query_by_attrs_on_an_empty_store_returns_empty_list(tmp_path: Path):
    """Querying a store with no records at all returns an empty list."""
    store = Store(tmp_path / "store.db")
    assert store.query_by_attrs() == []
