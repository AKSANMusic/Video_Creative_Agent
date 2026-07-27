"""Pydantic v2 schema for the Stage-2 audio map (`audio_map.json`).

The audio map is the contract consumed by the Stage-3 narrative sequencer.
It is intentionally compact and frame-sparse: continuous signals (RMS,
spectral contrast) are *downsampled* to a fixed number of buckets so the JSON
stays tiny regardless of track length.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Downsampled resolution for continuous curves. ~64 buckets is enough for the
# sequencer to read energy shape without storing per-frame arrays.
CURVE_BUCKETS = 64


class Section(BaseModel):
    """A structural segment of the track (intro / verse / chorus / ...)."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=0, description="0-based section index.")
    start: float = Field(..., ge=0.0, description="Section start time (seconds).")
    end: float = Field(..., ge=0.0, description="Section end time (seconds).")
    label: str = Field(default="", description="Human label (e.g. 'intro'). May be empty.")
    mean_energy: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Normalized mean RMS in [0,1]."
    )
    continuity_weight_override: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Per-section override of the global continuity weight "
            "(ACE_SEQUENCER_CONTINUITY_WEIGHT), set by an external director LLM. "
            "None = use the global value. Lower = harder/more dynamic cuts "
            "(embedding similarity dominates); higher = smoother/more color-stable cuts."
        ),
    )

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v, info):
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError("section end must be >= start")
        return v


class CrossModalTag(BaseModel):
    """Optional CLAP-derived cross-modal tag for a section (mood/energy words)."""

    model_config = ConfigDict(extra="forbid")

    section_index: int = Field(..., ge=0)
    mood: str = Field(default="")
    energy_word: str = Field(default="")
    similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class AudioMap(BaseModel):
    """Top-level audio map written to `audio_map.json`."""

    model_config = ConfigDict(extra="forbid")

    audio_id: str = Field(..., min_length=40, max_length=40, description="sha1 of raw audio bytes (hex).")
    file_path: str = Field(..., description="Path to the source audio file.")
    duration: float = Field(..., gt=0.0, description="Track duration (seconds).")
    sample_rate: int = Field(..., gt=0, description="Analysis sample rate (Hz).")

    bpm: float = Field(..., gt=0.0, description="Estimated tempo (beats per minute).")
    beats: list[float] = Field(
        default_factory=list, description="Beat onset times (seconds), monotonically increasing."
    )
    downbeats: list[float] = Field(
        default_factory=list, description="Estimated downbeat times (seconds)."
    )
    onsets: list[float] = Field(
        default_factory=list, description="General onset times (seconds)."
    )

    # Downsampled continuous curves, normalized to [0,1] where applicable.
    rms_curve: list[float] = Field(
        default_factory=list,
        description=f"RMS energy downsampled to ~{CURVE_BUCKETS} buckets, normalized to [0,1].",
    )
    spectral_contrast_curve: list[float] = Field(
        default_factory=list,
        description=f"Spectral contrast (timbral) downsampled to ~{CURVE_BUCKETS} buckets, normalized to [0,1].",
    )

    sections: list[Section] = Field(default_factory=list, description="Structural sections.")
    cross_modal: Optional[list[CrossModalTag]] = Field(
        default=None, description="Optional CLAP cross-modal tags per section (None if CLAP disabled)."
    )

    @field_validator("audio_id")
    @classmethod
    def _audio_id_hex(cls, v: str) -> str:
        try:
            int(v, 16)
        except ValueError as exc:
            raise ValueError("audio_id must be hexadecimal") from exc
        return v.lower()

    @field_validator("beats")
    @classmethod
    def _beats_monotonic(cls, v: list[float]) -> list[float]:
        for a, b in zip(v, v[1:]):
            if b < a:
                raise ValueError("beats must be monotonically non-decreasing")
        return v
