"""CLAP adapter tests: disabled path, unavailable-model path, fake-model path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ai_creative_engine.audio.audio_map import Section
from ai_creative_engine.audio.clap_adapter import CLAPAdapter


def _write_wav(path: Path, duration: float = 1.0, sr: int = 22050) -> Path:
    t = np.arange(int(duration * sr)) / sr
    y = 0.3 * np.sin(2 * np.pi * 440.0 * t).astype("float32")
    sf.write(str(path), y, sr, subtype="FLOAT")
    return path


def test_disabled_returns_none(tmp_path: Path):
    wav = _write_wav(tmp_path / "x.wav")
    adapter = CLAPAdapter(enabled=False)
    sections = [Section(index=0, start=0.0, end=1.0)]
    assert adapter.tag_sections(wav, sections) is None


def test_enabled_but_unavailable_returns_none(tmp_path: Path):
    """No laion_clap installed -> graceful None, never raises."""
    wav = _write_wav(tmp_path / "x.wav")
    adapter = CLAPAdapter(enabled=True)
    sections = [Section(index=0, start=0.0, end=1.0)]
    result = adapter.tag_sections(wav, sections)
    assert result is None


def test_enabled_but_no_sections_returns_none(tmp_path: Path):
    wav = _write_wav(tmp_path / "x.wav")
    adapter = CLAPAdapter(enabled=True)
    assert adapter.tag_sections(wav, []) is None


def test_fake_clap_model_returns_tags(monkeypatch, tmp_path: Path):
    """Inject a fake laion_clap module so the adapter produces deterministic tags."""
    import sys

    wav = _write_wav(tmp_path / "x.wav", duration=2.0)
    sections = [
        Section(index=0, start=0.0, end=1.0, mean_energy=0.5),
        Section(index=1, start=1.0, end=2.0, mean_energy=0.5),
    ]

    class FakeModel:
        def load_ckpt(self, name):
            self.name = name

        def to(self, device):
            self.device = device

        def get_audio_embedding_from_data(self, x, use_tensor=False):
            # Audio vector points along dim 0; matches 'tense'/'high'.
            v = np.zeros((1, 8), dtype="float32")
            v[0, 0] = 1.0
            return v

        def get_text_embedding(self, vocab):
            emb = np.zeros((len(vocab), 8), dtype="float32")
            for i, w in enumerate(vocab):
                # 'tense' and 'high' point along dim 0 (same as the audio vec);
                # everything else points along a different dimension so cosine
                # similarity is ~0 and they lose the argmax.
                if w in ("tense", "high"):
                    emb[i, 0] = 1.0
                else:
                    emb[i, 1] = 1.0
            return emb

    fake_module = type("M", (), {"CLAP_Module": lambda enable_fusion=False: FakeModel()})
    # The adapter reloads audio via librosa internally; ensure that works.
    monkeypatch.setitem(sys.modules, "laion_clap", fake_module)

    adapter = CLAPAdapter(enabled=True)
    tags = adapter.tag_sections(wav, sections)
    assert tags is not None
    assert len(tags) == 2
    assert tags[0].mood == "tense"
    assert tags[0].energy_word == "high"
    assert 0.0 <= tags[0].similarity <= 1.0
