"""SQLite cache for narrative timelines.

Keyed by a composite key derived from the audio_id + the sorted image_ids +
sequencer params, so a timeline is reused only when both inputs and the
algorithm config are identical. Mirrors the WAL + NORMAL defaults of the other
caches.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from ..errors import CacheError
from .timeline import Timeline

SCHEMA_VERSION = 1

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS timeline (
    timeline_key TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compute_timeline_key(
    audio_id: str,
    image_ids: list[str],
    cut_on: str,
    dp_threshold: int,
    continuity_weight: float,
    section_overrides: Optional[list[Optional[float]]] = None,
) -> str:
    """Deterministic composite key (sha1) for a timeline request."""
    h = hashlib.sha1()
    h.update(audio_id.encode("utf-8"))
    for img_id in sorted(image_ids):
        h.update(img_id.encode("utf-8"))
    h.update(f"|{cut_on}|{dp_threshold}|{continuity_weight:.6f}".encode("utf-8"))
    if section_overrides and any(ov is not None for ov in section_overrides):
        # None serializes distinctly from any float; join with a sentinel.
        h.update("|ovr:".encode("utf-8"))
        for ov in section_overrides:
            token = "none" if ov is None else f"{ov:.6f}"
            h.update(token.encode("utf-8"))
            h.update(b",")
    return h.hexdigest()


class TimelineCache:
    """SQLite cache for :class:`Timeline` records."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn: Optional[sqlite3.Connection] = None
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
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

    def exists(self, key: str) -> bool:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT 1 FROM timeline WHERE timeline_key = ? LIMIT 1;",
                    (key,),
                ).fetchone()
                return row is not None
        except CacheError:
            raise

    def get(self, key: str) -> Optional[Timeline]:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT payload_json FROM timeline WHERE timeline_key = ? LIMIT 1;",
                    (key,),
                ).fetchone()
            if row is None:
                return None
            data = json.loads(row["payload_json"])
            return Timeline.model_validate(data)
        except json.JSONDecodeError as exc:
            raise CacheError(f"Corrupt JSON in timeline cache for {key}: {exc}") from exc
        except CacheError:
            raise

    def upsert(self, key: str, timeline: Timeline) -> None:
        now = _utcnow_iso()
        payload = timeline.model_dump_json()
        try:
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT created_at FROM timeline WHERE timeline_key = ?;",
                    (key,),
                ).fetchone()
                created_at = existing["created_at"] if existing is not None else now
                conn.execute(
                    """
                    INSERT INTO timeline
                        (timeline_key, payload_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(timeline_key) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at;
                    """,
                    (key, payload, created_at, now),
                )
        except CacheError:
            raise

    def count(self) -> int:
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS n FROM timeline;").fetchone()
                return int(row["n"])
        except CacheError:
            raise
