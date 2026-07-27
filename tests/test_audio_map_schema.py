"""Schema validation tests for the audio map (Stage 2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_creative_engine.audio.audio_map import AudioMap, CrossModalTag, Section


def _valid_audio_map(**overrides):
    base = {
        "audio_id": "a" * 40,
        "file_path": "/tmp/track.mp3",
        "duration": 120.0,
        "sample_rate": 22050,
        "bpm": 120.0,
        "beats": [0.5, 1.0, 1.5],
        "downbeats": [0.5, 1.5],
        "onsets": [0.5, 0.8, 1.0],
        "rms_curve": [0.1] * 64,
        "spectral_contrast_curve": [0.2] * 64,
        "sections": [Section(index=0, start=0.0, end=60.0, label="intro", mean_energy=0.3)],
        "cross_modal": None,
    }
    base.update(overrides)
    return AudioMap(**base)


def test_valid_audio_map_round_trips():
    am = _valid_audio_map()
    assert am.audio_id == "a" * 40
    assert am.bpm == 120.0


def test_audio_id_must_be_hex():
    with pytest.raises(ValidationError):
        _valid_audio_map(audio_id="z" * 40)


def test_bpm_must_be_positive():
    with pytest.raises(ValidationError):
        _valid_audio_map(bpm=0.0)


def test_duration_must_be_positive():
    with pytest.raises(ValidationError):
        _valid_audio_map(duration=0.0)


def test_beats_must_be_monotonic():
    with pytest.raises(ValidationError):
        _valid_audio_map(beats=[1.0, 0.5])


def test_section_end_after_start():
    with pytest.raises(ValidationError):
        _valid_audio_map(sections=[Section(index=0, start=10.0, end=5.0)])


def test_section_mean_energy_clamped():
    with pytest.raises(ValidationError):
        _valid_audio_map(sections=[Section(index=0, start=0.0, end=10.0, mean_energy=1.5)])


def test_cross_modal_tag_similarity_range():
    CrossModalTag(section_index=0, mood="tense", energy_word="high", similarity=1.0)
    with pytest.raises(ValidationError):
        CrossModalTag(section_index=0, similarity=2.0)


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        _valid_audio_map(unexpected_field="boom")


def test_cross_modal_none_allowed():
    am = _valid_audio_map(cross_modal=None)
    assert am.cross_modal is None


def test_section_continuity_weight_override_optional():
    """Field defaults to None (backwards compatible)."""
    s = Section(index=0, start=0.0, end=10.0)
    assert s.continuity_weight_override is None


def test_section_continuity_weight_override_accepted():
    s = Section(index=0, start=0.0, end=10.0, continuity_weight_override=0.2)
    assert s.continuity_weight_override == 0.2


def test_section_continuity_weight_override_range():
    with pytest.raises(ValidationError):
        Section(index=0, start=0.0, end=10.0, continuity_weight_override=1.5)
    with pytest.raises(ValidationError):
        Section(index=0, start=0.0, end=10.0, continuity_weight_override=-0.1)


def test_audio_map_round_trips_override_via_json():
    """An audio map with the override serializes/deserializes losslessly."""
    am = _valid_audio_map(
        sections=[
            Section(index=0, start=0.0, end=60.0, label="chorus", mean_energy=0.8, continuity_weight_override=0.2),
            Section(index=1, start=60.0, end=120.0, label="verse", mean_energy=0.4, continuity_weight_override=0.9),
        ]
    )
    raw = am.model_dump_json()
    revived = AudioMap.model_validate_json(raw)
    assert revived.sections[0].continuity_weight_override == 0.2
    assert revived.sections[1].continuity_weight_override == 0.9


def test_old_audio_map_without_override_still_loads():
    """Legacy JSON produced before the field existed must still validate."""
    legacy = {
        "audio_id": "a" * 40,
        "file_path": "/tmp/track.mp3",
        "duration": 120.0,
        "sample_rate": 22050,
        "bpm": 120.0,
        "beats": [0.5, 1.0, 1.5],
        "downbeats": [0.5, 1.5],
        "onsets": [0.5, 0.8, 1.0],
        "rms_curve": [0.1] * 64,
        "spectral_contrast_curve": [0.2] * 64,
        "sections": [
            {"index": 0, "start": 0.0, "end": 60.0, "label": "intro", "mean_energy": 0.3},
        ],
        "cross_modal": None,
    }
    am = AudioMap.model_validate(legacy)
    assert am.sections[0].continuity_weight_override is None


def test_cross_modal_list_allowed():
    am = _valid_audio_map(
        cross_modal=[CrossModalTag(section_index=0, mood="bright", energy_word="rising", similarity=0.7)]
    )
    assert am.cross_modal is not None
    assert am.cross_modal[0].mood == "bright"
