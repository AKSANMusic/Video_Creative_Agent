"""Schema validation tests for ImageMetadata."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_creative_engine.models import EMBEDDING_HEX_LEN, ImageMetadata


def _valid_kwargs(**overrides):
    base = {
        "image_id": "a" * 40,
        "file_path": "/tmp/x.png",
        "width": 100,
        "height": 50,
        "florence_caption": "a red square",
        "objects": ["square"],
        "bounding_boxes": [[0.1, 0.2, 0.9, 1.0]],
        "llava_mood": "tense",
        "llava_tension": 0.5,
        "llava_symbolism": "warning",
        "embedding_hex": "0" * EMBEDDING_HEX_LEN,
    }
    base.update(overrides)
    return base


def test_valid_metadata_round_trips():
    m = ImageMetadata(**_valid_kwargs())
    assert m.image_id == "a" * 40
    assert m.embedding_bytes == bytes(64)


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        ImageMetadata(**_valid_kwargs(unexpected="boom"))


def test_tension_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ImageMetadata(**_valid_kwargs(llava_tension=1.5))
    with pytest.raises(ValidationError):
        ImageMetadata(**_valid_kwargs(llava_tension=-0.1))


def test_tension_boundaries_accepted():
    ImageMetadata(**_valid_kwargs(llava_tension=0.0))
    ImageMetadata(**_valid_kwargs(llava_tension=1.0))


def test_embedding_hex_wrong_length_rejected():
    with pytest.raises(ValidationError):
        ImageMetadata(**_valid_kwargs(embedding_hex="ab"))


def test_embedding_hex_non_hex_rejected():
    bad = "z" * EMBEDDING_HEX_LEN
    with pytest.raises(ValidationError):
        ImageMetadata(**_valid_kwargs(embedding_hex=bad))


def test_image_id_must_be_hex():
    with pytest.raises(ValidationError):
        ImageMetadata(**_valid_kwargs(image_id="z" * 40))


def test_bounding_box_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ImageMetadata(**_valid_kwargs(bounding_boxes=[[0.0, 0.0, 1.5, 1.0]]))


def test_bounding_box_wrong_length_rejected():
    with pytest.raises(ValidationError):
        ImageMetadata(**_valid_kwargs(bounding_boxes=[[0.0, 0.0, 1.0]]))


def test_payload_under_budget():
    """Representative payload serialized JSON must be < 768 bytes."""
    m = ImageMetadata(**_valid_kwargs())
    assert len(m.model_dump_json()) < 768
