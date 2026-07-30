"""Pydantic v2 schema for the Stage-3 timeline (``timeline.json``).

The timeline is the contract consumed by the Stage-4 FFmpeg renderer. Each
entry pins one image to a [start, end) window whose boundaries are audio
downbeats (hard cuts) and carries a transition hint for the *leading* edge.

Design constraints
------------------
- Every entry's ``start`` must coincide with a downbeat (beat-locked cuts).
- Entries are contiguous and cover the full track duration.
- Kept compact: per-entry JSON stays a few hundred bytes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Allowed transition types. Kept tiny and renderer-friendly.
TransitionType = Literal["cut", "dissolve", "zoom"]


class Transition(BaseModel):
    """Transition directive applied at the *start* of an entry."""

    model_config = ConfigDict(extra="forbid")

    type: TransitionType = Field(default="cut", description="Transition into this entry.")
    duration_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Transition duration in seconds (0 for an instant hard cut).",
    )


class TimelineEntry(BaseModel):
    """One image shown over a beat-locked time window."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=0, description="0-based entry index in the timeline.")
    image_id: str = Field(..., min_length=40, max_length=40, description="sha1 of the source image.")
    file_path: str = Field(..., description="Path to the source image file.")
    section_index: int = Field(..., ge=0, description="Audio section this entry belongs to.")
    start: float = Field(..., ge=0.0, description="Entry start time (seconds).")
    end: float = Field(..., gt=0.0, description="Entry end time (seconds).")
    transition: Transition = Field(default_factory=Transition)

    # --- Cinematic Director (Phase 2) ---
    audio_offset_ms: float = Field(
        default=0.0,
        description="Audio offset in milliseconds for J-cuts (<0) or L-cuts (>0).",
    )
    image_role: str = Field(
        default="b_roll",
        description="Narrative role: 'a_roll' or 'b_roll'.",
    )
    narrative_act: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Narrative act (1, 2, or 3).",
    )
    zoom_target_x: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="X coordinate of zoom target focus.",
    )
    zoom_target_y: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Y coordinate of zoom target focus.",
    )
    zoom_intensity: float = Field(
        default=1.08,
        gt=0.0,
        description="Intensity/magnitude of zoompan effect.",
    )

    @field_validator("image_id")
    @classmethod
    def _image_id_hex(cls, v: str) -> str:
        try:
            int(v, 16)
        except ValueError as exc:
            raise ValueError("image_id must be hexadecimal") from exc
        return v.lower()

    @model_validator(mode="after")
    def _end_after_start(self) -> "TimelineEntry":
        if self.end <= self.start:
            raise ValueError("entry end must be strictly greater than start")
        if self.transition.duration_s > (self.end - self.start):
            raise ValueError("transition duration cannot exceed entry length")
        return self


class Timeline(BaseModel):
    """Top-level timeline written to ``timeline.json``."""

    model_config = ConfigDict(extra="forbid")

    audio_id: str = Field(..., min_length=40, max_length=40, description="sha1 of the source audio.")
    audio_file: str = Field(..., description="Path to the source audio file.")
    duration: float = Field(..., gt=0.0, description="Total timeline duration (seconds).")
    bpm: float = Field(..., gt=0.0, description="Tempo used for beat-locking.")
    downbeats_used: list[float] = Field(
        default_factory=list,
        description="Downbeat times (seconds) used as cut anchors.",
    )
    entries: list[TimelineEntry] = Field(..., min_length=1)

    @field_validator("audio_id")
    @classmethod
    def _audio_id_hex(cls, v: str) -> str:
        try:
            int(v, 16)
        except ValueError as exc:
            raise ValueError("audio_id must be hexadecimal") from exc
        return v.lower()

    @model_validator(mode="after")
    def _contiguous_and_beat_locked(self) -> "Timeline":
        if not self.entries:
            raise ValueError("timeline must have at least one entry")
        prev_end = -1.0
        for e in self.entries:
            if e.start < 0.0:
                raise ValueError("entries must not start before t=0")
            if prev_end >= 0.0 and abs(e.start - prev_end) > 1e-6:
                raise ValueError(
                    f"entries must be contiguous; gap/overlap at entry {e.index} "
                    f"(prev_end={prev_end}, start={e.start})"
                )
            prev_end = e.end
        # Last entry must cover the full track duration.
        last = self.entries[-1]
        if abs(last.end - self.duration) > 1e-3:
            raise ValueError(
                f"last entry end ({last.end}) must equal timeline duration ({self.duration})"
            )
        return self
