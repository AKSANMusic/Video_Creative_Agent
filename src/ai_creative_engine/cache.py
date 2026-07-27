"""Lightweight SQLite (WAL) cache for image metadata.

The cache is the single source of truth for "have we already paid for this
image's inference?". Keyed by ``image_id`` (sha1 of raw bytes) so renames do
not invalidate entries.

Design notes
------------
- WAL journal + NORMAL synchronous: good concurrency, low risk, fast writes.
- Schema is versioned via the ``user_version`` pragma so future migrations are
  possible without an ad-hoc schema check.
- All errors are wrapped in :class:`CacheError`.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from .errors import CacheError
from .models import ImageMetadata

SCHEMA_VERSION = 1

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS image_metadata (
    image_id     TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    embedding_blob BLOB NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MetadataCache:
    """A small, dependency-free SQLite cache for :class:`ImageMetadata`."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)

    # --- connection lifecycle ------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn: Optional[sqlite3.Connection] = None
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            # Robust, fast defaults.
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION};")
            yield conn
        except sqlite3.Error as exc:
            raise CacheError(f"SQLite error on {self.db_path}: {exc}") from exc
        finally:
            if conn is not None:
                conn.close()

    # --- public API ----------------------------------------------------------

    def exists(self, image_id: str) -> bool:
        """Return True if ``image_id`` is already cached."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT 1 FROM image_metadata WHERE image_id = ? LIMIT 1;",
                    (image_id,),
                ).fetchone()
                return row is not None
        except CacheError:
            raise

    def get(self, image_id: str) -> Optional[ImageMetadata]:
        """Return cached metadata, or ``None`` if absent."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT payload_json FROM image_metadata WHERE image_id = ? LIMIT 1;",
                    (image_id,),
                ).fetchone()
            if row is None:
                return None
            data = json.loads(row["payload_json"])
            return ImageMetadata.model_validate(data)
        except json.JSONDecodeError as exc:
            raise CacheError(f"Corrupt JSON in cache for {image_id}: {exc}") from exc
        except CacheError:
            raise

    def upsert(self, meta: ImageMetadata) -> None:
        """Insert or replace metadata. Sets/updates timestamps atomically."""
        now = _utcnow_iso()
        payload = meta.model_dump_json()
        blob = meta.embedding_bytes
        try:
            with self._connect() as conn:
                # Preserve original created_at on re-upsert.
                existing = conn.execute(
                    "SELECT created_at FROM image_metadata WHERE image_id = ?;",
                    (meta.image_id,),
                ).fetchone()
                created_at = existing["created_at"] if existing is not None else now
                conn.execute(
                    """
                    INSERT INTO image_metadata
                        (image_id, payload_json, embedding_blob, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(image_id) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        embedding_blob = excluded.embedding_blob,
                        updated_at = excluded.updated_at;
                    """,
                    (meta.image_id, payload, blob, created_at, now),
                )
        except CacheError:
            raise

    def count(self) -> int:
        """Return the number of cached images."""
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS n FROM image_metadata;").fetchone()
                return int(row["n"])
        except CacheError:
            raise
