"""SQLite audio cache round-trip + idempotency tests."""

from __future__ import annotations

from pathlib import Path

from ai_creative_engine.audio.audio_cache import AudioCache
from ai_creative_engine.audio.audio_map import AudioMap, Section


def _sample(audio_id: str = "a" * 40) -> AudioMap:
    return AudioMap(
        audio_id=audio_id,
        file_path="/tmp/track.mp3",
        duration=10.0,
        sample_rate=22050,
        bpm=120.0,
        beats=[0.5, 1.0],
        downbeats=[0.5],
        onsets=[0.5],
        rms_curve=[0.1] * 64,
        spectral_contrast_curve=[0.2] * 64,
        sections=[Section(index=0, start=0.0, end=10.0, mean_energy=0.3)],
        cross_modal=None,
    )


def test_upsert_then_get(tmp_path: Path):
    cache = AudioCache(tmp_path / "a.db")
    am = _sample()
    cache.upsert(am)
    assert cache.exists(am.audio_id)
    got = cache.get(am.audio_id)
    assert got is not None
    assert got.bpm == 120.0
    assert len(got.sections) == 1


def test_get_missing_returns_none(tmp_path: Path):
    cache = AudioCache(tmp_path / "a.db")
    assert cache.get("0" * 40) is None
    assert cache.exists("0" * 40) is False


def test_count_and_idempotent_upsert(tmp_path: Path):
    cache = AudioCache(tmp_path / "a.db")
    cache.upsert(_sample("a" * 40))
    cache.upsert(_sample("b" * 40))
    assert cache.count() == 2
    cache.upsert(_sample("a" * 40))  # idempotent
    assert cache.count() == 2


def test_wal_pragma_applied(tmp_path: Path):
    import sqlite3

    cache = AudioCache(tmp_path / "a.db")
    cache.upsert(_sample())
    conn = sqlite3.connect(tmp_path / "a.db")
    mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    conn.close()
    assert mode.lower() == "wal"
