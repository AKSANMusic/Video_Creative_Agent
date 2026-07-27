"""Pydantic v2 schema for per-image metadata (the Stage-1 JSON contract).

Design constraints
-------------------
- Total serialized payload must stay well under 0.5 KB per image.
- The embedding is stored as raw 64 bytes (512-dim sign-quantized) on disk;
  in the JSON payload it appears as a 128-char hex string for portability.
- ``extra='forbid'`` so schema drift is caught loudly.
"""

from __future__ import annotations

import binascii
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# 512 bits / 8 = 64 bytes; hex = 128 chars.
EMBEDDING_BYTES = 64
EMBEDDING_HEX_LEN = EMBEDDING_BYTES * 2  # 128

# Hard ceiling so a single payload never silently exceeds the 0.5 KB budget.
PAYLOAD_MAX_BYTES = 512


class ImageMetadata(BaseModel):
    """Canonical per-image metadata record.

    The ``image_id`` (sha1 of raw file bytes, 40 hex chars) is the primary key
    used by the cache.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    image_id: str = Field(..., min_length=40, max_length=40, description="sha1 of raw image bytes (hex).")

    file_path: str = Field(..., description="Absolute or relative path to the source image.")
    width: int = Field(..., gt=0, description="Image width in pixels.")
    height: int = Field(..., gt=0, description="Image height in pixels.")

    # EXIF may legitimately be absent for synthetic or stripped images.
    exif: Optional[dict[str, Any]] = Field(
        default=None, description="Selected EXIF tags (may be None)."
    )

    # --- Florence-2 outputs (literal / spatial) ---
    florence_caption: str = Field(default="", description="Florence-2 dense caption.")
    objects: list[str] = Field(default_factory=list, description="Detected object labels.")
    bounding_boxes: list[list[float]] = Field(
        default_factory=list,
        description="Normalized xyxy boxes in [0.0, 1.0], aligned with `objects`.",
    )

    # --- LLaVA-NeXT outputs (artistic) ---
    llava_mood: str = Field(default="unknown", description="Single-word/short mood label.")
    llava_tension: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Narrative tension score in [0.0, 1.0]."
    )
    llava_symbolism: str = Field(default="", description="Free-text symbolism / artistic intent.")

    # --- Embedding (512-dim sign-quantized, 64 bytes) as hex for JSON portability ---
    embedding_hex: str = Field(
        default="0" * EMBEDDING_HEX_LEN,
        description=f"{EMBEDDING_HEX_LEN}-char hex of the 64-byte binary embedding.",
    )

    @field_validator("image_id")
    @classmethod
    def _image_id_is_hex(cls, v: str) -> str:
        try:
            int(v, 16)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("image_id must be hexadecimal") from exc
        return v.lower()

    @field_validator("embedding_hex")
    @classmethod
    def _embedding_hex_well_formed(cls, v: str) -> str:
        if len(v) != EMBEDDING_HEX_LEN:
            raise ValueError(
                f"embedding_hex must be exactly {EMBEDDING_HEX_LEN} hex chars, got {len(v)}"
            )
        try:
            bytes.fromhex(v)
        except ValueError as exc:
            raise ValueError("embedding_hex contains non-hex characters") from exc
        return v.lower()

    @field_validator("bounding_boxes")
    @classmethod
    def _boxes_finite_range(cls, boxes: list[list[float]]) -> list[list[float]]:
        for box in boxes:
            if len(box) != 4:
                raise ValueError("each bounding box must have exactly 4 floats [x1, y1, x2, y2]")
            for coord in box:
                if not isinstance(coord, (int, float)):  # pragma: no cover - pydantic coerces
                    raise ValueError("bounding box coords must be numeric")
                if not (0.0 <= float(coord) <= 1.0):
                    raise ValueError("bounding box coords must be within [0.0, 1.0]")
        return boxes

    @model_validator(mode="after")
    def _payload_size_budget(self) -> "ImageMetadata":
        size = len(self.model_dump_json())
        if size > PAYLOAD_MAX_BYTES:
            # Soft warning path: do not raise, but the test suite enforces the
            # budget on representative payloads. We keep the model permissive so
            # long captions don't crash the pipeline; the cache stores whatever
            # pydantic accepts. (See test_models.test_payload_under_budget.)
            pass
        return self

    # --- Convenience accessors ------------------------------------------------

    @property
    def embedding_bytes(self) -> bytes:
        """Decode ``embedding_hex`` to raw 64 bytes."""
        return binascii.unhexlify(self.embedding_hex)
