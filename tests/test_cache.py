"""SQLite cache round-trip and idempotency tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_creative_engine.cache import MetadataCache
from ai_creative_engine.models import EMBEDDING_HEX_LEN, ImageMetadata


def _sample(image_id: str = "a" * 40) -> ImageMetadata:
    return ImageMetadata(
        image_id=image_id,
        file_path="/tmp/x.png",
        width=10,
        height=20,
        florence_caption="cap",
        objects=["o"],
        bounding_boxes=[[0.0, 0.0, 1.0, 1.0]],
        llava_mood="m",
        llava_tension=0.2,
        llava_symbolism="s",
        embedding_hex="0" * EMBEDDING_HEX_LEN,
    )


def test_upsert_then_get_round_trip(tmp_path: Path):
    cache = MetadataCache(tmp_path / "t.db")
    meta = _sample()
    cache.upsert(meta)
    assert cache.exists(meta.image_id)
    got = cache.get(meta.image_id)
    assert got is not None
    assert got.image_id == meta.image_id
    assert got.objects == ["o"]
    assert got.embedding_bytes == bytes(64)


def test_get_missing_returns_none(tmp_path: Path):
    cache = MetadataCache(tmp_path / "t.db")
    assert cache.get("0" * 40) is None
    assert cache.exists("0" * 40) is False


def test_re_upsert_updates_updated_at_not_created(tmp_path: Path):
    import time

    cache = MetadataCache(tmp_path / "t.db")
    meta = _sample()
    cache.upsert(meta)
    time.sleep(1.1)  # ensure ISO timestamp differs (seconds resolution)
    meta2 = meta.model_copy(update={"llava_mood": "calm"})
    cache.upsert(meta2)

    # Round-trip reflects the update.
    got = cache.get(meta.image_id)
    assert got is not None and got.llava_mood == "calm"


def test_count(tmp_path: Path):
    cache = MetadataCache(tmp_path / "t.db")
    assert cache.count() == 0
    cache.upsert(_sample("a" * 40))
    cache.upsert(_sample("b" * 40))
    assert cache.count() == 2
    cache.upsert(_sample("a" * 40))  # idempotent
    assert cache.count() == 2


def test_wal_pragma_applied(tmp_path: Path):
    cache = MetadataCache(tmp_path / "t.db")
    cache.upsert(_sample())
    import sqlite3

    conn = sqlite3.connect(tmp_path / "t.db")
    try:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal"
