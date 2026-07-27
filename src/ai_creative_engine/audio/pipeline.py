"""Stage-2 audio analysis orchestrator.

Flow
----
1. audio_id = sha1(raw bytes)
2. if audio_cache.exists(audio_id): return cached AudioMap  (never re-analyze)
3. raw = analyzer.analyze(path)
4. build sections from boundaries + per-section mean energy
5. (optional) clap.tag_sections(path, sections)
6. assemble AudioMap, cache.upsert immediately, return

One audio file's failure is reported via exceptions (there is only one file
per run, unlike the image batch).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from .analyzer import AudioAnalyzer, RawAnalysis
from .audio_cache import AudioCache
from .audio_map import AudioMap, CrossModalTag, Section
from .clap_adapter import CLAPAdapter

log = logging.getLogger(__name__)


class AudioPipeline:
    """Coordinates analyzer + CLAP adapter + cache."""

    def __init__(
        self,
        analyzer: AudioAnalyzer,
        cache: AudioCache,
        clap: Optional[CLAPAdapter] = None,
    ) -> None:
        self.analyzer = analyzer
        self.cache = cache
        self.clap = clap

    # --- public API ----------------------------------------------------------

    def run(self, audio_path: Path | str) -> AudioMap:
        """Analyze one audio file and return its :class:`AudioMap` (cached)."""
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        raw = audio_path.read_bytes()
        audio_id = hashlib.sha1(raw).hexdigest()

        if self.cache.exists(audio_id):
            log.info("Audio cache hit: %s (%s)", audio_path.name, audio_id)
            cached = self.cache.get(audio_id)
            if cached is not None:
                return cached
            log.warning("Cache reported hit but get() returned None; re-analyzing.")

        raw_analysis = self.analyzer.analyze(audio_path)
        sections = self._build_sections(raw_analysis)
        cross_modal = self._tag(audio_path, sections)

        audio_map = AudioMap(
            audio_id=audio_id,
            file_path=str(audio_path),
            duration=raw_analysis.duration,
            sample_rate=raw_analysis.sample_rate,
            bpm=raw_analysis.bpm,
            beats=raw_analysis.beats,
            downbeats=raw_analysis.downbeats,
            onsets=raw_analysis.onsets,
            rms_curve=raw_analysis.rms_curve,
            spectral_contrast_curve=raw_analysis.spectral_contrast_curve,
            sections=sections,
            cross_modal=cross_modal,
        )

        self.cache.upsert(audio_map)
        log.info(
            "Analyzed %s: bpm=%.1f beats=%d sections=%d cross_modal=%s",
            audio_path.name,
            audio_map.bpm,
            len(audio_map.beats),
            len(audio_map.sections),
            "on" if cross_modal else "off",
        )
        return audio_map

    # --- helpers -------------------------------------------------------------

    def _build_sections(self, raw: RawAnalysis) -> list[Section]:
        """Convert raw boundaries into :class:`Section` records with mean energy."""
        bounds = raw.section_boundaries
        if len(bounds) < 2:
            return []

        rms = raw.rms_curve
        n_buckets = len(rms) if rms else 0

        sections: list[Section] = []
        for idx in range(len(bounds) - 1):
            start = float(bounds[idx])
            end = float(bounds[idx + 1])
            if raw.duration <= 0:
                mean_energy = 0.0
            else:
                lo = max(0, int(start / raw.duration * n_buckets))
                hi = max(lo + 1, int(end / raw.duration * n_buckets))
                window = rms[lo:hi] if n_buckets else []
                mean_energy = (sum(window) / len(window)) if window else 0.0
                mean_energy = float(min(max(mean_energy, 0.0), 1.0))
            sections.append(
                Section(
                    index=idx,
                    start=start,
                    end=end,
                    label="",
                    mean_energy=mean_energy,
                )
            )
        return sections

    def _tag(self, audio_path: Path, sections: list[Section]) -> Optional[list[CrossModalTag]]:
        if self.clap is None:
            return None
        try:
            return self.clap.tag_sections(audio_path, sections)
        except Exception as exc:  # CLAP must never crash the pipeline.
            log.warning("CLAP tagging failed (%s); continuing without cross-modal tags", exc)
            return None
