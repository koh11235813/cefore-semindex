"""SQLite-backed, append-only store for Index records and per-origin progress.

Uses WAL mode (map.md "永続化は SQLite（WAL）") so a reader never blocks
the single-threaded agent's writes. Every record insert and its progress
update happen inside one transaction ("レコード挿入と進捗更新は同じトラ
ンザクション"): a crash between the two is impossible, so after any
restart `progress.hi` for an origin is exactly the highest seq that is
actually present contiguously from 1 in the records table -- it can
never be ahead of what was durably written.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from semindex import schema


@dataclass(frozen=True)
class OriginProgress:
    """A holder's summary of one origin's held records (CONTEXT.md "Holder state").

    `hi` is the highest seq held contiguously from 1 (0 if none held).
    `extra` is the sorted tuple of seqs held above `hi` that are not yet
    contiguous with it. This says only what is held, never what has
    merely been heard of.
    """

    hi: int
    extra: tuple[int, ...] = ()


@dataclass(frozen=True)
class StoreResult:
    """Outcome of inserting one received record."""

    stored: bool
    duplicate: bool
    conflict: bool
    existing_hash: str | None = None


class Store:
    """Append-only SQLite store for Index records and per-origin progress."""

    def __init__(self, path: str | Path) -> None:
        # The agent is single-threaded (CLAUDE.md), so the default
        # same-thread connection is correct; nothing here spans threads.
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create the records and progress tables if this is a fresh database file."""
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    origin TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    payload BLOB NOT NULL,
                    content_hash TEXT NOT NULL,
                    PRIMARY KEY (origin, seq)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS progress (
                    origin TEXT PRIMARY KEY,
                    hi INTEGER NOT NULL,
                    extra TEXT NOT NULL
                )
                """
            )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # -- reads ---------------------------------------------------------------

    def get_progress(self, origin: str) -> OriginProgress:
        """This node's current (hi, extra) for `origin`, or hi=0/extra=() if none held."""
        row = self._conn.execute("SELECT hi, extra FROM progress WHERE origin = ?", (origin,)).fetchone()
        if row is None:
            return OriginProgress(hi=0, extra=())
        hi, extra_json = row
        return OriginProgress(hi=hi, extra=tuple(json.loads(extra_json)))

    def all_progress(self) -> dict[str, OriginProgress]:
        """Every origin this node currently holds any records for."""
        rows = self._conn.execute("SELECT origin, hi, extra FROM progress").fetchall()
        return {
            origin: OriginProgress(hi=hi, extra=tuple(json.loads(extra_json)))
            for origin, hi, extra_json in rows
        }

    def get_record(self, origin: str, seq: int) -> bytes | None:
        """The canonical wire bytes stored for (origin, seq), or None if not held."""
        row = self._conn.execute(
            "SELECT payload FROM records WHERE origin = ? AND seq = ?", (origin, seq)
        ).fetchone()
        return row[0] if row is not None else None

    # -- writes ----------------------------------------------------------------

    def append_received(self, origin: str, seq: int, payload: bytes, content_hash: str) -> StoreResult:
        """Insert a record received from a neighbor, folding it into progress.

        Idempotent on an exact duplicate (same origin, seq and hash). A
        second record at the same (origin, seq) with a different hash is
        a conflict: the first write is kept (map.md "先着を残し conflict
        として記録する") and this call reports it without touching the
        store.
        """
        with self._conn:
            existing = self._conn.execute(
                "SELECT content_hash FROM records WHERE origin = ? AND seq = ?", (origin, seq)
            ).fetchone()
            if existing is not None:
                existing_hash = existing[0]
                if existing_hash == content_hash:
                    return StoreResult(stored=False, duplicate=True, conflict=False)
                return StoreResult(stored=False, duplicate=False, conflict=True, existing_hash=existing_hash)

            self._conn.execute(
                "INSERT INTO records (origin, seq, payload, content_hash) VALUES (?, ?, ?, ?)",
                (origin, seq, payload, content_hash),
            )
            self._fold_progress(origin, seq)
            return StoreResult(stored=True, duplicate=False, conflict=False)

    def append_own(
        self,
        *,
        origin: str,
        content_name: str,
        zone: str,
        disaster: str,
        media: str,
        observed_at: int,
        model_id: str = schema.DEFAULT_MODEL_ID,
        supersedes: schema.RecordKey | None = None,
    ) -> schema.IndexRecord:
        """Allocate the next seq for `origin` and store the record atomically.

        The seq allocated is one more than the highest seq already
        present for this origin in the records table, read inside the
        same transaction as the insert -- so two calls in a row, or a
        call right after reopening the database, can never allocate the
        same seq twice (map.md "seq は SQLite で採番し、再起動後は保存
        済み最大 + 1 から").
        """
        with self._conn:
            row = self._conn.execute("SELECT MAX(seq) FROM records WHERE origin = ?", (origin,)).fetchone()
            next_seq = 1 if row[0] is None else row[0] + 1
            record = schema.IndexRecord(
                origin=origin,
                seq=next_seq,
                content_name=content_name,
                zone=zone,
                disaster=disaster,
                media=media,
                observed_at=observed_at,
                model_id=model_id,
                supersedes=supersedes,
            )
            payload = schema.encode(record)  # validates and enforces the 2 KiB limit
            digest = schema.content_hash(payload)
            self._conn.execute(
                "INSERT INTO records (origin, seq, payload, content_hash) VALUES (?, ?, ?, ?)",
                (origin, next_seq, payload, digest),
            )
            self._fold_progress(origin, next_seq)
            return record

    def _fold_progress(self, origin: str, seq: int) -> None:
        """Fold a newly-inserted seq into (hi, extra); caller holds the transaction.

        A seq <= the current hi cannot reach here: append_received would
        have taken the duplicate/conflict branch, and append_own always
        allocates one past the current max. So this only ever extends hi
        (absorbing any extras that become contiguous) or adds to extra.
        """
        current = self.get_progress(origin)
        if seq == current.hi + 1:
            extra_set = set(current.extra)
            hi = seq
            while hi + 1 in extra_set:
                extra_set.discard(hi + 1)
                hi += 1
            extra = tuple(sorted(extra_set))
        else:
            extra = tuple(sorted(set(current.extra) | {seq}))
            hi = current.hi
        self._conn.execute(
            "INSERT INTO progress (origin, hi, extra) VALUES (?, ?, ?) "
            "ON CONFLICT(origin) DO UPDATE SET hi = excluded.hi, extra = excluded.extra",
            (origin, hi, json.dumps(extra)),
        )
