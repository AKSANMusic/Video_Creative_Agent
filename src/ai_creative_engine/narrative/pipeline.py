"""Stage-3 orchestrator: images + audio map -> beat-locked timeline.json.

Flow
----
1. Load :class:`AudioMap` from a JSON file (Stage-2 output) or pass directly.
2. Load :class:`ImageMetadata` records from the Stage-1 SQLite cache.
3. Compute the deterministic timeline key; return cached timeline if present.
4. Run :class:`NarrativeSequencer`.
5. Cache + return.

The sequencer is deterministic, so caching by the composite key is safe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..audio.audio_map import AudioMap
from ..cache import MetadataCache
from ..config import get_settings, Settings
from ..errors import CreativeEngineError
from ..models import ImageMetadata
from .cache import TimelineCache, compute_timeline_key
from .exporter import export_timeline
from .sequencer import NarrativeSequencer
from .timeline import Timeline

log = logging.getLogger(__name__)


class NarrativePipeline:
    """Coordinates cache lookups + sequencing."""

    def __init__(
        self,
        image_cache: MetadataCache,
        timeline_cache: TimelineCache,
        sequencer: Optional[NarrativeSequencer] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.image_cache = image_cache
        self.timeline_cache = timeline_cache
        self.sequencer = sequencer or NarrativeSequencer()
        self.settings = settings or get_settings()

    # --- public API ----------------------------------------------------------

    def run(
        self,
        audio_map: AudioMap,
        image_ids: Optional[list[str]] = None,
    ) -> Timeline:
        """Build (or fetch from cache) a :class:`Timeline`."""
        images = self._load_images(image_ids)
        if not images:
            raise CreativeEngineError(
                "No images available for sequencing (image cache is empty)."
            )

        key = compute_timeline_key(
            audio_id=audio_map.audio_id,
            image_ids=[im.image_id for im in images],
            cut_on=self.sequencer.cut_on,
            dp_threshold=self.sequencer.dp_threshold,
            continuity_weight=self.sequencer.continuity_weight,
            section_overrides=[
                s.continuity_weight_override for s in sorted(audio_map.sections, key=lambda x: x.index)
            ],
            director_enabled=self.settings.director_enabled,
            pacing_reactivity=self.settings.pacing_reactivity,
            narrative_arc_enabled=self.settings.narrative_arc_enabled,
            saliency_camera_enabled=self.settings.saliency_camera_enabled,
            color_grading_enabled=self.settings.color_grading_enabled,
            jlcut_enabled=self.settings.jlcut_enabled,
            jlcut_max_ms=self.settings.jlcut_max_ms,
        )

        if self.timeline_cache.exists(key):
            cached = self.timeline_cache.get(key)
            if cached is not None:
                log.info("Timeline cache hit (key=%s)", key[:12])
                return cached
            log.warning("Timeline cache miss after reported hit; rebuilding.")

        timeline = self.sequencer.sequence(images, audio_map)

        # Apply Cinematic Director post-sequencing pass (Phase 2)
        if self.settings.director_enabled:
            from .director import CinematicDirector
            director = CinematicDirector(self.settings)
            timeline = director.direct(timeline, audio_map, images)

        self.timeline_cache.upsert(key, timeline)
        log.info(
            "Sequenced timeline: %d entries over %.2fs (cache key=%s)",
            len(timeline.entries),
            timeline.duration,
            key[:12],
        )
        return timeline

    def run_and_export(
        self,
        audio_map: AudioMap,
        out_path: Path | str,
        image_ids: Optional[list[str]] = None,
    ) -> Path:
        """Convenience: build the timeline and write it to ``out_path``."""
        timeline = self.run(audio_map, image_ids=image_ids)
        return export_timeline(timeline, out_path)

    # --- helpers -------------------------------------------------------------

    def _load_images(self, image_ids: Optional[list[str]]) -> list[ImageMetadata]:
        """Load images from the Stage-1 cache.

        If ``image_ids`` is None, load every cached image. Unknown ids are
        skipped with a warning (defensive: never crash on a stale id list).
        """
        if image_ids is None:
            # Walk the cache via the connection to enumerate ids.
            import sqlite3

            ids: list[str] = []
            try:
                with self.image_cache._connect() as conn:  # type: ignore[attr-defined]
                    rows = conn.execute("SELECT image_id FROM image_metadata ORDER BY image_id;").fetchall()
                    ids = [r["image_id"] for r in rows]
            except Exception as exc:  # noqa: BLE001
                raise CreativeEngineError(f"Failed to enumerate image cache: {exc}") from exc
        else:
            ids = list(image_ids)

        images: list[ImageMetadata] = []
        for iid in ids:
            meta = self.image_cache.get(iid)
            if meta is None:
                log.warning("Image id %s not found in cache; skipping.", iid)
                continue
            images.append(meta)
        return images
