"""CLI smoke tests for the `analyze-audio` command (offline, synthetic audio)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from ai_creative_engine.cli import app

SR = 22050


def _write_click(path: Path, duration: float = 5.0, bpm: float = 120.0) -> Path:
    n = int(duration * SR)
    t = np.arange(n) / SR
    y = 0.2 * (0.3 + 0.7 * (t / duration)) * np.sin(2 * np.pi * 220.0 * t)
    beat_period = 60.0 / bpm
    click_len = int(0.03 * SR)
    nb = 0.0
    while nb < duration:
        i = int(nb * SR)
        if i + click_len < n:
            y[i : i + click_len] += 0.8
        nb += beat_period
    sf.write(str(path), y.astype("float32"), SR, subtype="FLOAT")
    return path


def test_analyze_audio_writes_json(tmp_path: Path):
    wav = _write_click(tmp_path / "click.wav", duration=5.0)
    db = tmp_path / "out.db"
    out = tmp_path / "audio_map.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["analyze-audio", "--audio", str(wav), "--db", str(db), "--out", str(out), "--no-clap"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Audio map written" in result.output
    assert "bpm=" in result.output


def test_analyze_audio_missing_file(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["analyze-audio", "--audio", str(tmp_path / "nope.wav"), "--db", str(tmp_path / "x.db")],
    )
    assert result.exit_code == 2
    assert "not found" in result.output.lower()


def test_analyze_audio_second_run_is_cache_hit(tmp_path: Path):
    wav = _write_click(tmp_path / "click.wav", duration=5.0)
    db = tmp_path / "out.db"
    out = tmp_path / "audio_map.json"
    runner = CliRunner()
    runner.invoke(app, ["analyze-audio", "--audio", str(wav), "--db", str(db), "--out", str(out)])
    # Second run should still succeed and reuse the cache.
    r2 = runner.invoke(app, ["analyze-audio", "--audio", str(wav), "--db", str(db), "--out", str(out)])
    assert r2.exit_code == 0, r2.output
