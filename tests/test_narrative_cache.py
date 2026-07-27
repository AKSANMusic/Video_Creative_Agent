"""Timeline cache round-trip + determinism-key tests."""

from __future__ import annotations

from pathlib import Path

from ai_creative_engine.narrative.cache import TimelineCache, compute_timeline_key
from ai_creative_engine.narrative.sequencer import NarrativeSequencer
from tests.narrative_fixtures import make_audio_map, make_images


def test_key_changes_with_image_set():
    am = make_audio_map()
    k1 = compute_timeline_key(am.audio_id, ["a" * 40], "downbeats", 12, 0.5)
    k2 = compute_timeline_key(am.audio_id, ["b" * 40], "downbeats", 12, 0.5)
    assert k1 != k2


def test_key_changes_with_params():
    am = make_audio_map()
    imgs = ["a" * 40]
    k1 = compute_timeline_key(am.audio_id, imgs, "downbeats", 12, 0.5)
    k2 = compute_timeline_key(am.audio_id, imgs, "beats", 12, 0.5)
    k3 = compute_timeline_key(am.audio_id, imgs, "downbeats", 5, 0.5)
    assert k1 != k2 != k3


def test_cache_round_trip(tmp_path: Path):
    cache = TimelineCache(tmp_path / "t.db")
    images = make_images(6)
    am = make_audio_map(duration=8.0, n_sections=2)
    tl = NarrativeSequencer().sequence(images, am)
    key = compute_timeline_key(am.audio_id, [i.image_id for i in images], "downbeats", 12, 0.5)
    cache.upsert(key, tl)
    assert cache.exists(key)
    got = cache.get(key)
    assert got is not None
    assert [e.image_id for e in got.entries] == [e.image_id for e in tl.entries]
    assert cache.count() == 1


def test_key_changes_with_section_overrides():
    """Per-section continuity_weight_override values MUST fold into the cache key.

    Regression: two audio maps differing only in overrides produce different
    timelines, so they must not share a cache entry.
    """
    am = make_audio_map()
    imgs = ["a" * 40, "b" * 40]
    base = dict(
        audio_id=am.audio_id, image_ids=imgs, cut_on="downbeats",
        dp_threshold=12, continuity_weight=0.5,
    )
    k_none = compute_timeline_key(**base)
    k_all_none = compute_timeline_key(**base, section_overrides=[None, None])
    k_directed = compute_timeline_key(**base, section_overrides=[0.9, 0.2])
    k_directed_repeat = compute_timeline_key(**base, section_overrides=[0.9, 0.2])

    # Backward compat: omitting the arg == all-None list (no overrides in effect).
    assert k_none == k_all_none
    # Overrides change the key (this was the live-run bug).
    assert k_none != k_directed
    # Determinism: same overrides -> same key.
    assert k_directed == k_directed_repeat
    # Different override values -> different key.
    k_other = compute_timeline_key(**base, section_overrides=[0.2, 0.9])
    assert k_directed != k_other
