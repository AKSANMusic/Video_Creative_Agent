"""Analyzer tests using a synthetic audio file (no network, deterministic).

We synthesize a short click track at a known BPM so beat detection has a
stable target, and a modulated tone so RMS / spectral-contrast curves are
non-flat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ai_creative_engine.audio.analyzer import AudioAnalyzer


SR = 22050


def _synth_click_track(
    out: Path, duration: float = 6.0, bpm: float = 120.0, sr: int = SR
) -> Path:
    """Write a click track: short impulses on each beat + background tone."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    # Background tone whose amplitude rises over time (non-flat RMS).
    env = 0.2 * (0.3 + 0.7 * (t / duration))
    y = env * np.sin(2 * np.pi * 220.0 * t)

    # Add impulses on each beat.
    beat_period = 60.0 / bpm
    click_len = int(0.03 * sr)
    next_beat = 0.0
    while next_beat < duration:
        i = int(next_beat * sr)
        if i + click_len < n:
            y[i : i + click_len] += 0.8
        next_beat += beat_period

    sf.write(str(out), y.astype("float32"), sr, subtype="FLOAT")
    return out


def test_analyzer_produces_expected_shape(tmp_path: Path):
    wav = _synth_click_track(tmp_path / "click.wav", duration=6.0, bpm=120.0)
    analyzer = AudioAnalyzer(sample_rate=SR, hop_length=512, n_segments=4)
    raw = analyzer.analyze(wav)

    assert raw.duration == pytest.approx(6.0, rel=1e-2)
    assert raw.sample_rate == SR
    # BPM should be in a plausible range around 120.
    assert 80.0 <= raw.bpm <= 180.0
    # At 120 bpm over 6s we expect several beats.
    assert len(raw.beats) >= 4
    # Downbeats are a subset of beats (every 4th).
    assert len(raw.downbeats) <= len(raw.beats)
    # Curves are normalized to [0,1] and downsampled to the bucket count.
    assert len(raw.rms_curve) == analyzer.curve_buckets
    assert len(raw.spectral_contrast_curve) == analyzer.curve_buckets
    assert all(0.0 <= v <= 1.0 for v in raw.rms_curve)
    # Boundaries span [0, duration].
    assert raw.section_boundaries[0] == pytest.approx(0.0, abs=1e-6)
    assert raw.section_boundaries[-1] == pytest.approx(raw.duration, abs=1e-1)


def test_beats_are_monotonic(tmp_path: Path):
    wav = _synth_click_track(tmp_path / "click.wav", duration=4.0, bpm=120.0)
    raw = AudioAnalyzer().analyze(wav)
    assert all(b2 >= b1 for b1, b2 in zip(raw.beats, raw.beats[1:]))


def test_empty_audio_raises(tmp_path: Path):
    # Write a zero-length (all-zero) file: librosa still loads it; duration>0
    # but there are no events. We instead force a degenerate case via a 1-sample
    # file to exercise the defensive path.
    sf.write(str(tmp_path / "tiny.wav"), np.zeros(1, dtype="float32"), SR, subtype="FLOAT")
    analyzer = AudioAnalyzer()
    # A 1-sample file has duration ~0 -> should raise AudioAnalysisError.
    from ai_creative_engine.audio.analyzer import AudioAnalysisError

    with pytest.raises(AudioAnalysisError):
        analyzer.analyze(tmp_path / "tiny.wav")


def test_downsample_normalize_handles_flat():
    analyzer = AudioAnalyzer(curve_buckets=8)
    flat = analyzer._downsample_normalize(np.zeros(1000))
    assert flat == [0.0] * 8


def test_downsample_normalize_range():
    analyzer = AudioAnalyzer(curve_buckets=8)
    out = analyzer._downsample_normalize(np.linspace(5.0, 10.0, 1000))
    assert len(out) == 8
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[-1] == pytest.approx(1.0, abs=1e-6)
