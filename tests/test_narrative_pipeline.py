"""End-to-end narrative pipeline + exporter tests (offline, synthetic data)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_creative_engine.audio.audio_map import AudioMap
from ai_creative_engine.cache import MetadataCache
from ai_creative_engine.errors import CreativeEngineError
from ai_creative_engine.narrative.cache import TimelineCache
from ai_creative_engine.narrative.exporter import export_timeline, load_timeline
from ai_creative_engine.narrative.pipeline import NarrativePipeline
from ai_creative_engine.narrative.sequencer import NarrativeSequencer
from tests.narrative_fixtures import make_audio_map, make_images


def _seed_image_cache(tmp_path: Path, images) -> MetadataCache:
    cache = MetadataCache(tmp_path / "img.db")
    for im in images:
        cache.upsert(im)
    return cache


def test_pipeline_builds_and_caches_timeline(tmp_path: Path):
    images = make_images(8)
    image_cache = _seed_image_cache(tmp_path, images)
    tl_cache = TimelineCache(tmp_path / "tl.db")
    pipe = NarrativePipeline(image_cache, tl_cache, NarrativeSequencer())
    am = make_audio_map(duration=8.0, n_sections=2)

    tl1 = pipe.run(am)
    assert tl1.duration == pytest.approx(am.duration, abs=1e-3)
    assert len(tl1.entries) >= 2
    assert tl_cache.count() == 1

    # Second run: cache hit, identical output.
    tl2 = pipe.run(am)
    assert [e.image_id for e in tl2.entries] == [e.image_id for e in tl1.entries]


def test_pipeline_export_round_trip(tmp_path: Path):
    images = make_images(6)
    image_cache = _seed_image_cache(tmp_path, images)
    tl_cache = TimelineCache(tmp_path / "tl.db")
    pipe = NarrativePipeline(image_cache, tl_cache, NarrativeSequencer())
    am = make_audio_map(duration=8.0, n_sections=2)

    out = tmp_path / "timeline.json"
    pipe.run_and_export(am, out)
    assert out.exists()
    loaded = load_timeline(out)
    assert loaded.audio_id == am.audio_id
    assert len(loaded.entries) >= 2


def test_pipeline_empty_image_cache_raises(tmp_path: Path):
    image_cache = MetadataCache(tmp_path / "empty.db")
    tl_cache = TimelineCache(tmp_path / "tl.db")
    pipe = NarrativePipeline(image_cache, tl_cache, NarrativeSequencer())
    am = make_audio_map()
    with pytest.raises(CreativeEngineError):
        pipe.run(am)


def test_pipeline_filters_unknown_image_ids(tmp_path: Path):
    images = make_images(4)
    image_cache = _seed_image_cache(tmp_path, images)
    tl_cache = TimelineCache(tmp_path / "tl.db")
    pipe = NarrativePipeline(image_cache, tl_cache, NarrativeSequencer())
    am = make_audio_map(duration=8.0, n_sections=2)

    # Pass one valid + one bogus id; bogus is skipped, timeline still builds.
    good_id = images[0].image_id
    bogus = "0" * 40
    tl = pipe.run(am, image_ids=[good_id, bogus])
    used = {e.image_id for e in tl.entries}
    assert good_id in used
    assert bogus not in used


def test_exported_payload_is_compact(tmp_path: Path):
    images = make_images(10)
    image_cache = _seed_image_cache(tmp_path, images)
    tl_cache = TimelineCache(tmp_path / "tl.db")
    pipe = NarrativePipeline(image_cache, tl_cache, NarrativeSequencer())
    am = make_audio_map(duration=8.0, n_sections=2)
    out = tmp_path / "timeline.json"
    pipe.run_and_export(am, out)
    size = out.stat().st_size
    # 8s track, ~16 entries -> well under ~20 KB.
    assert size < 20_000, f"timeline.json too large: {size}"
