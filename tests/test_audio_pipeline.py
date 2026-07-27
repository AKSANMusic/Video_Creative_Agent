"""Audio pipeline orchestrator + exporter tests (offline, synthetic audio)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ai_creative_engine.audio.audio_cache import AudioCache
from ai_creative_engine.audio.audio_map import Section
from ai_creative_engine.audio.analyzer import AudioAnalyzer
from ai_creative_engine.audio.clap_adapter import CLAPAdapter
from ai_creative_engine.audio.exporter import export_audio_map, load_audio_map
from ai_creative_engine.audio.pipeline import AudioPipeline


SR = 22050


def _write_click(path: Path, duration: float = 6.0, bpm: float = 120.0) -> Path:
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


def _pipeline(tmp_path: Path, clap=None) -> AudioPipeline:
    analyzer = AudioAnalyzer(sample_rate=SR, hop_length=512, n_segments=4)
    cache = AudioCache(tmp_path / "a.db")
    return AudioPipeline(analyzer=analyzer, cache=cache, clap=clap)


def test_run_produces_audio_map(tmp_path: Path):
    wav = _write_click(tmp_path / "click.wav")
    pipe = _pipeline(tmp_path)
    am = pipe.run(wav)
    assert am.duration == pytest.approx(6.0, rel=1e-2)
    assert am.bpm > 0.0
    assert len(am.beats) >= 4
    assert len(am.sections) >= 2
    assert am.cross_modal is None  # no CLAP
    assert pipe.cache.count() == 1


def test_second_run_is_cache_hit(tmp_path: Path):
    wav = _write_click(tmp_path / "click.wav")
    pipe = _pipeline(tmp_path)
    am1 = pipe.run(wav)
    am2 = pipe.run(wav)
    assert am1.audio_id == am2.audio_id
    assert am2.bpm == am1.bpm  # served from cache, identical
    assert pipe.cache.count() == 1


def test_missing_file_raises(tmp_path: Path):
    pipe = _pipeline(tmp_path)
    with pytest.raises(FileNotFoundError):
        pipe.run(tmp_path / "nope.wav")


def test_sections_span_full_duration(tmp_path: Path):
    wav = _write_click(tmp_path / "click.wav", duration=5.0)
    pipe = _pipeline(tmp_path)
    am = pipe.run(wav)
    assert am.sections[0].start == pytest.approx(0.0, abs=1e-6)
    assert am.sections[-1].end == pytest.approx(am.duration, abs=0.2)


def test_exporter_round_trip(tmp_path: Path):
    wav = _write_click(tmp_path / "click.wav")
    pipe = _pipeline(tmp_path)
    am = pipe.run(wav)
    out = tmp_path / "audio_map.json"
    export_audio_map(am, out)
    assert out.exists()
    loaded = load_audio_map(out)
    assert loaded.audio_id == am.audio_id
    assert loaded.bpm == am.bpm
    assert len(loaded.beats) == len(am.beats)


def test_exported_payload_is_compact(tmp_path: Path):
    """audio_map.json for a short track should stay well under ~50 KB."""
    wav = _write_click(tmp_path / "click.wav", duration=10.0)
    pipe = _pipeline(tmp_path)
    am = pipe.run(wav)
    out = tmp_path / "audio_map.json"
    export_audio_map(am, out)
    size = out.stat().st_size
    assert size < 50_000, f"audio_map.json too large: {size} bytes"


def test_with_disabled_clap_cross_modal_none(tmp_path: Path):
    wav = _write_click(tmp_path / "click.wav")
    pipe = _pipeline(tmp_path, clap=CLAPAdapter(enabled=False))
    am = pipe.run(wav)
    assert am.cross_modal is None


def test_with_unavailable_clap_cross_modal_none(tmp_path: Path):
    """CLAP enabled but not installed -> pipeline must not crash; cross_modal None."""
    wav = _write_click(tmp_path / "click.wav")
    pipe = _pipeline(tmp_path, clap=CLAPAdapter(enabled=True))
    am = pipe.run(wav)
    assert am.cross_modal is None  # gracefully omitted
