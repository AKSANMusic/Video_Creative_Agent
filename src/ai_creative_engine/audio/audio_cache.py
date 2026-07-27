"""SQLite cache for audio maps.

Mirrors the image-metadata cache pattern but stores :class:`AudioMap` payloads
keyed by ``audio_id`` (sha1 of raw audio bytes). Same WAL + NORMAL synchronous
defaults for robustness.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from ..errors import CacheError
from .audio_map import AudioMap

SCHEMA_VERSION = 1

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audio_map (
    audio_id     TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AudioCache:
    """A small SQLite cache for :class:`AudioMap` records."""

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

    def exists(self, audio_id: str) -> bool:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT 1 FROM audio_map WHERE audio_id = ? LIMIT 1;",
                    (audio_id,),
                ).fetchone()
                return row is not None
        except CacheError:
            raise

    def get(self, audio_id: str) -> Optional[AudioMap]:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT payload_json FROM audio_map WHERE audio_id = ? LIMIT 1;",
                    (audio_id,),
                ).fetchone()
            if row is None:
                return None
            data = json.loads(row["payload_json"])
            return AudioMap.model_validate(data)
        except json.JSONDecodeError as exc:
            raise CacheError(f"Corrupt JSON in audio cache for {audio_id}: {exc}") from exc
        except CacheError:
            raise

    def upsert(self, audio_map: AudioMap) -> None:
        now = _utcnow_iso()
        payload = audio_map.model_dump_json()
        try:
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT created_at FROM audio_map WHERE audio_id = ?;",
                    (audio_map.audio_id,),
                ).fetchone()
                created_at = existing["created_at"] if existing is not None else now
                conn.execute(
                    """
                    INSERT INTO audio_map
                        (audio_id, payload_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(audio_id) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at;
                    """,
                    (audio_map.audio_id, payload, created_at, now),
                )
        except CacheError:
            raise

    def count(self) -> int:
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS n FROM audio_map;").fetchone()
                return int(row["n"])
        except CacheError:
            raise
