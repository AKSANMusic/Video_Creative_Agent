"""CLI smoke test for the `sequence` command (offline)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ai_creative_engine.audio.exporter import export_audio_map
from ai_creative_engine.cache import MetadataCache
from ai_creative_engine.cli import app
from tests.narrative_fixtures import make_audio_map, make_images


def _seed(tmp_path: Path) -> tuple[Path, Path]:
    images = make_images(8)
    img_cache = MetadataCache(tmp_path / "img.db")
    for im in images:
        img_cache.upsert(im)
    am = make_audio_map(duration=8.0, n_sections=2)
    am_path = tmp_path / "audio_map.json"
    export_audio_map(am, am_path)
    return tmp_path / "img.db", am_path


def test_sequence_writes_timeline(tmp_path: Path):
    db, am_path = _seed(tmp_path)
    out = tmp_path / "timeline.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["sequence", "--audio-map", str(am_path), "--db", str(db), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Timeline written" in result.output
    assert "entries=" in result.output


def test_sequence_missing_audio_map(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["sequence", "--audio-map", str(tmp_path / "nope.json"), "--db", str(tmp_path / "x.db")],
    )
    assert result.exit_code == 2
    assert "not found" in result.output.lower()
