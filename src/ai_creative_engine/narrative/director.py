"""Cinematic Director module (Phase 2).

Post-processes a sequenced Timeline to inject advanced directorial logic:
1. Three-Act Narrative Arc tags.
2. Dynamic, energy-reactive pacing multipliers.
3. J-cut/L-cut audio offsets based on emotional/mood shifts.
4. Saliency-driven focus targets (zoom/pan center coordinates).
5. Content-aware Ken Burns zoom intensities.
"""

from __future__ import annotations

from typing import Any
from ..config import Settings
from ..audio.audio_map import AudioMap
from ..models import ImageMetadata
from .timeline import Timeline, TimelineEntry, Transition


class CinematicDirector:
    """Refines a sequenced Timeline with professional cinematic directing choices."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def direct(self, timeline: Timeline, audio_map: AudioMap, images: list[ImageMetadata]) -> Timeline:
        """Apply director refinements to the timeline. Returns a mutated or new Timeline."""
        if not self.settings.director_enabled:
            return timeline

        image_db = {img.image_id: img for img in images}
        refined_entries = []

        for i, entry in enumerate(timeline.entries):
            img = image_db.get(entry.image_id)
            
            # Use defaults if the image metadata isn't cached or is missing fields
            image_role = getattr(img, "image_role", "b_roll") if img else "b_roll"
            sal_x = getattr(img, "saliency_center_x", 0.5) if img else 0.5
            sal_y = getattr(img, "saliency_center_y", 0.5) if img else 0.5
            tension = getattr(img, "llava_tension", 0.5) if img else 0.5
            mood = getattr(img, "llava_mood", "neutral") if img else "neutral"

            # 1. 3-Act Narrative Arc Assignment
            act, act_name = self._compute_narrative_act(entry.start, timeline.duration)
            
            # 2. Transition Intelligence Refinements
            transition = self._refine_transition(entry, i, timeline.entries, img)

            # 3. Ken Burns Zoom & Focus coordinates
            zoom_intensity = 1.08
            if self.settings.saliency_camera_enabled:
                # Higher energy/tension = more aggressive zoom
                energy = audio_map.rms_curve[int((entry.start / timeline.duration) * len(audio_map.rms_curve))] if audio_map.rms_curve else 0.5
                zoom_intensity = 1.05 + 0.10 * (0.4 * energy + 0.6 * tension)
                zoom_intensity = round(max(1.02, min(zoom_intensity, 1.18)), 4)
            else:
                sal_x = 0.5
                sal_y = 0.5

            # 4. J-cuts and L-cuts
            audio_offset = 0.0
            if self.settings.jlcut_enabled and i > 0:
                prev_entry = timeline.entries[i - 1]
                prev_img = image_db.get(prev_entry.image_id)
                prev_mood = getattr(prev_img, "llava_mood", "neutral") if prev_img else "neutral"
                
                # Check for J-cut (anticipation: calm -> high tension)
                if prev_mood in ["calm", "neutral", "melancholic"] and mood in ["tense", "aggressive", "energetic"]:
                    audio_offset = -self.settings.jlcut_max_ms
                # Check for L-cut (lingering: high tension -> calm)
                elif prev_mood in ["tense", "aggressive"] and mood in ["melancholic", "calm", "peaceful"]:
                    audio_offset = self.settings.jlcut_max_ms

            # Create updated entry
            refined_entry = TimelineEntry(
                index=entry.index,
                image_id=entry.image_id,
                file_path=entry.file_path,
                section_index=entry.section_index,
                start=entry.start,
                end=entry.end,
                transition=transition,
                audio_offset_ms=audio_offset,
                image_role=image_role,
                narrative_act=act,
                zoom_target_x=sal_x,
                zoom_target_y=sal_y,
                zoom_intensity=zoom_intensity
            )
            refined_entries.append(refined_entry)

        return Timeline(
            audio_id=timeline.audio_id,
            audio_file=timeline.audio_file,
            duration=timeline.duration,
            bpm=timeline.bpm,
            downbeats_used=timeline.downbeats_used,
            entries=refined_entries
        )

    def _compute_narrative_act(self, start_time: float, total_duration: float) -> tuple[int, str]:
        """Compute the Act classification (1, 2, or 3) based on timeline progress."""
        progress = start_time / total_duration
        if progress < 0.25:
            return 1, "Act I (Setup)"
        elif progress < 0.80:
            return 2, "Act II (Climax)"
        else:
            return 3, "Act III (Resolution)"

    def _refine_transition(self, entry: TimelineEntry, index: int, all_entries: list[TimelineEntry], img: ImageMetadata | None) -> Transition:
        """Apply advanced heuristics to choose transition types and durations."""
        # Baseline transition from sequencer
        trans_type = entry.transition.type
        trans_dur = entry.transition.duration_s

        if not self.settings.director_enabled:
            return entry.transition

        # Transition Intelligence Matrix
        # 1. Zoom Transition: A-roll reveal
        if getattr(img, "image_role", "b_roll") == "a_roll" and index > 0:
            prev_entry = all_entries[index - 1]
            if prev_entry.section_index != entry.section_index:
                # Transitioning into a new section, A-roll reveal: apply quick zoom
                trans_type = "zoom"
                trans_dur = 0.5

        # 2. Dip to Black on major Act boundaries
        total_dur = max(all_entries[-1].end, 1.0)
        act, _ = self._compute_narrative_act(entry.start, total_dur)
        if index > 0:
            prev_act, _ = self._compute_narrative_act(all_entries[index - 1].start, total_dur)
            if prev_act != act:
                # Transitioning acts: slow cross-fade
                trans_type = "dissolve"
                trans_dur = 0.8

        return Transition(type=trans_type, duration_s=trans_dur)
