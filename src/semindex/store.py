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


def attrs_match(record, *, zone: str | None, disaster: str | None, media: str | None) -> bool:
    """Whether `record`'s zone/disaster/media satisfy a probe query: each filter is exact-equal, None is a wildcard, all ANDed.

    `record` is anything carrying `.zone`/`.disaster`/`.media` (a stored
    schema.IndexRecord, or a config's not-yet-created agent.ScheduledRecord).
    This is the one definition of "a probe matches a record", kept as a
    pure function so that rebuilding ground truth from configs applies
    exactly the rule the live index query applies, and the two cannot drift.
    """
    if zone is not None and record.zone != zone:
        return False
    if disaster is not None and record.disaster != disaster:
        return False
    if media is not None and record.media != media:
        return False
    return True


class Store:
    """Append-only SQLite store for Index records and per-origin progress."""

    def __init__(self, path: str | Path) -> None:
        """Open (or create) the SQLite database file at `path` in WAL mode."""
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
            # 2026-10-22 fix: a restarted agent used to re-run every
            # schedule/probe entry whose at_s had already come due in a
            # prior life. Unlike own records, a probe has no per-origin
            # progress row to reuse as a restart cursor (a probe is a
            # local query, not a record with a seq) -- so this single-row
            # table is a dedicated cursor instead: how many of this
            # node's scheduled probes (in at_s order) have already run,
            # across every life of this state_dir. The `id = 0` check
            # keeps it a true singleton (one counter per node, since each
            # node has its own state_dir/index.db).
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS probe_progress (
                    id INTEGER PRIMARY KEY CHECK (id = 0),
                    count INTEGER NOT NULL
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

    def get_probe_count(self) -> int:
        """How many of this node's scheduled probes have already executed, across every
        life of this state_dir (0 for a fresh store, or one that has never run a probe).
        """
        row = self._conn.execute("SELECT count FROM probe_progress WHERE id = 0").fetchone()
        return row[0] if row is not None else 0

    def get_record(self, origin: str, seq: int) -> bytes | None:
        """The canonical wire bytes stored for (origin, seq), or None if not held."""
        row = self._conn.execute(
            "SELECT payload FROM records WHERE origin = ? AND seq = ?", (origin, seq)
        ).fetchone()
        return row[0] if row is not None else None

    def query_by_attrs(
        self,
        *,
        zone: str | None = None,
        disaster: str | None = None,
        media: str | None = None,
    ) -> list[schema.IndexRecord]:
        """Every held record whose attrs match the given filters, ordered by (origin, seq).

        A filter left None matches any value (CONTEXT.md "Sync agent": this
        is a probe's local query against this node's own copy of the
        Semantic Index, never a remote lookup -- map.md explicitly excludes
        remote lookup from this baseline). `attrs` is not a separate column:
        this baseline's record counts do not justify duplicating it out of
        the stored payload just to index it, so this decodes every row and
        filters in Python. The fixed (origin, seq) ordering gives a probe a
        deterministic "first match" without needing its own tie-break rule.
        """
        rows = self._conn.execute("SELECT payload FROM records ORDER BY origin, seq").fetchall()
        matches: list[schema.IndexRecord] = []
        for (payload,) in rows:
            record = schema.decode(payload)
            if attrs_match(record, zone=zone, disaster=disaster, media=media):
                matches.append(record)
        return matches

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

    def record_probe_executed(self) -> None:
        """Durably record that one more of this node's scheduled probes has run.

        Wrapped in its own transaction, the same discipline append_received/
        append_own use, so a crash either lands this increment or it
        didn't -- never a partial write the restart cursor could read back
        as some in-between count.
        """
        with self._conn:
            self._conn.execute(
                "INSERT INTO probe_progress (id, count) VALUES (0, 1) "
                "ON CONFLICT(id) DO UPDATE SET count = count + 1"
            )

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
