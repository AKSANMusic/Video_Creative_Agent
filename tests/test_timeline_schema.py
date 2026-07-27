"""Schema validation tests for the Stage-3 timeline."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_creative_engine.narrative.timeline import Timeline, TimelineEntry, Transition
from tests.narrative_fixtures import make_audio_map, make_image


def _entry(index=0, start=0.0, end=1.0, image_id_seed=1, section_index=0):
    img = make_image(image_id_seed)
    return TimelineEntry(
        index=index,
        image_id=img.image_id,
        file_path=img.file_path,
        section_index=section_index,
        start=start,
        end=end,
        transition=Transition(type="cut", duration_s=0.0),
    )


def _audio_id():
    return make_audio_map().audio_id


def test_valid_timeline_round_trips():
    am = make_audio_map()
    e = _entry(0, 0.0, am.duration)
    tl = Timeline(
        audio_id=am.audio_id,
        audio_file=am.file_path,
        duration=am.duration,
        bpm=am.bpm,
        downbeats_used=[0.0],
        entries=[e],
    )
    assert len(tl.entries) == 1


def test_entry_end_after_start_required():
    with pytest.raises(ValidationError):
        _entry(0, 1.0, 0.5)


def test_transition_duration_cannot_exceed_entry_length():
    img = make_image(1)
    with pytest.raises(ValidationError):
        TimelineEntry(
            index=0,
            image_id=img.image_id,
            file_path=img.file_path,
            section_index=0,
            start=0.0,
            end=0.5,
            transition=Transition(type="dissolve", duration_s=1.0),
        )


def test_timeline_must_be_contiguous():
    am = make_audio_map()
    e1 = _entry(0, 0.0, 1.0)
    e2 = _entry(1, 2.0, 3.0)  # gap between 1.0 and 2.0
    with pytest.raises(ValidationError):
        Timeline(
            audio_id=am.audio_id,
            audio_file=am.file_path,
            duration=3.0,
            bpm=am.bpm,
            downbeats_used=[],
            entries=[e1, e2],
        )


def test_timeline_last_entry_must_cover_duration():
    am = make_audio_map(duration=8.0)
    e = _entry(0, 0.0, 5.0)  # ends before 8.0
    with pytest.raises(ValidationError):
        Timeline(
            audio_id=am.audio_id,
            audio_file=am.file_path,
            duration=8.0,
            bpm=am.bpm,
            downbeats_used=[],
            entries=[e],
        )


def test_audio_id_must_be_hex():
    with pytest.raises(ValidationError):
        Timeline(
            audio_id="z" * 40,
            audio_file="x",
            duration=1.0,
            bpm=120.0,
            downbeats_used=[],
            entries=[_entry(0, 0.0, 1.0)],
        )


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        Transition(type="cut", duration_s=0.0, unexpected=True)
