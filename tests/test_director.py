"""Unit tests for Phase 2 CinematicDirector module."""

from __future__ import annotations

import pytest

from ai_creative_engine.config import Settings
from ai_creative_engine.audio.audio_map import AudioMap, Section
from ai_creative_engine.models import ImageMetadata
from ai_creative_engine.narrative.timeline import Timeline, TimelineEntry, Transition
from ai_creative_engine.narrative.director import CinematicDirector


@pytest.fixture
def test_data():
    # Setup simple Settings
    settings = Settings(
        director_enabled=True,
        jlcut_enabled=True,
        jlcut_max_ms=300.0,
        saliency_camera_enabled=True
    )
    
    # Setup simple AudioMap
    audio_map = AudioMap(
        audio_id="a" * 40,
        file_path="track.mp3",
        duration=10.0,
        sample_rate=22050,
        bpm=120.0,
        beats=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        downbeats=[0.0, 4.0, 8.0],
        rms_curve=[0.2] * 64,
        sections=[
            Section(index=0, start=0.0, end=2.0, mean_energy=0.2),
            Section(index=1, start=2.0, end=8.0, mean_energy=0.8),
            Section(index=2, start=8.0, end=10.0, mean_energy=0.3)
        ]
    )

    # Setup simple ImageMetadata list
    images = [
        ImageMetadata(
            image_id="1" * 40,
            file_path="img1.png",
            width=100,
            height=100,
            florence_caption="a quiet room",
            llava_mood="calm",
            llava_tension=0.2,
            image_role="b_roll",
            saliency_center_x=0.3,
            saliency_center_y=0.4
        ),
        ImageMetadata(
            image_id="2" * 40,
            file_path="img2.png",
            width=100,
            height=100,
            florence_caption="a person running in terror",
            llava_mood="tense",
            llava_tension=0.9,
            image_role="a_roll",
            saliency_center_x=0.7,
            saliency_center_y=0.6
        )
    ]

    # Setup simple Timeline
    timeline = Timeline(
        audio_id="a" * 40,
        audio_file="track.mp3",
        duration=10.0,
        bpm=120.0,
        downbeats_used=[0.0, 4.0, 8.0],
        entries=[
            TimelineEntry(
                index=0,
                image_id="1" * 40,
                file_path="img1.png",
                section_index=0,
                start=0.0,
                end=2.0,
                transition=Transition(type="cut", duration_s=0.0)
            ),
            TimelineEntry(
                index=1,
                image_id="2" * 40,
                file_path="img2.png",
                section_index=1,
                start=2.0,
                end=8.0,
                transition=Transition(type="cut", duration_s=0.0)
            ),
            TimelineEntry(
                index=2,
                image_id="1" * 40,
                file_path="img1.png",
                section_index=2,
                start=8.0,
                end=10.0,
                transition=Transition(type="cut", duration_s=0.0)
            )
        ]
    )

    return settings, audio_map, images, timeline


def test_director_bypass_when_disabled(test_data):
    settings, audio_map, images, timeline = test_data
    settings.director_enabled = False
    
    director = CinematicDirector(settings)
    out = director.direct(timeline, audio_map, images)
    
    # Assert unmodified
    assert out.entries[0].audio_offset_ms == 0.0
    assert out.entries[1].narrative_act == 1
    assert out.entries[1].zoom_target_x == 0.5


def test_director_assigns_acts_and_offsets(test_data):
    settings, audio_map, images, timeline = test_data
    
    director = CinematicDirector(settings)
    out = director.direct(timeline, audio_map, images)
    
    # 1. Acts assignment
    # Entry 0 (start=0.0 -> progress=0%) -> Act 1
    # Entry 1 (start=2.0 -> progress=20%) -> Act 1
    # Entry 2 (start=8.0 -> progress=80%) -> Act 3
    assert out.entries[0].narrative_act == 1
    assert out.entries[1].narrative_act == 1
    assert out.entries[2].narrative_act == 3
    
    # 2. Focus coordinates (saliency matching)
    assert out.entries[0].zoom_target_x == 0.3
    assert out.entries[0].zoom_target_y == 0.4
    assert out.entries[1].zoom_target_x == 0.7
    assert out.entries[1].zoom_target_y == 0.6

    # 3. J-cut offset (low energy mood "calm" -> high energy "tense")
    assert out.entries[1].audio_offset_ms == -300.0
    # L-cut offset (high energy "tense" -> low energy "calm")
    assert out.entries[2].audio_offset_ms == 300.0
    
    # 4. Transitions
    # Transitioning to new section with A-roll: should trigger zoom transition
    assert out.entries[1].transition.type == "zoom"
    assert out.entries[1].transition.duration_s == 0.5
    
    # Transitioning acts: Act 2 to Act 3 should trigger act-boundary dissolve
    assert out.entries[2].transition.type == "dissolve"
    assert out.entries[2].transition.duration_s == 0.8
