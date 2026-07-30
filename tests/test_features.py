"""Unit tests for Phase 2 visual feature extraction module."""

from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image
import pytest

from ai_creative_engine.vision.features import VisualFeatureExtractor


@pytest.fixture
def dummy_image(tmp_path) -> Path:
    img_path = tmp_path / "test.png"
    # Create a 100x100 RGB image with a blue square on red background
    img = Image.new("RGB", (100, 100), color="red")
    for y in range(30, 70):
        for x in range(30, 70):
            img.putpixel((x, y), (0, 0, 255))  # blue square
    img.save(img_path)
    return img_path


def test_feature_extractor_pil_fallback(dummy_image, monkeypatch):
    """Test feature extraction when cv2 is not present (fallback path)."""
    # Force ImportError on cv2 to trigger PIL fallback path
    monkeypatch.setitem(sys.modules, "cv2", None)

    extractor = VisualFeatureExtractor()
    features = extractor.extract(dummy_image, metadata_so_far={"florence_caption": "a blue square"})

    assert "image_role" in features
    assert len(features["dominant_colors"]) == 3
    assert 0.0 <= features["mean_luminance"] <= 1.0
    assert 0.0 <= features["mean_saturation"] <= 1.0
    assert 0.0 <= features["color_temperature"] <= 1.0
    assert 0.0 <= features["saliency_center_x"] <= 1.0
    assert 0.0 <= features["saliency_center_y"] <= 1.0
    assert 0.0 <= features["edge_complexity"] <= 1.0
    assert features["subject_quadrant"] == "center"


def test_feature_extractor_role_classification():
    extractor = VisualFeatureExtractor()

    # Case 1: Minimal text/no boxes/no tension -> B-roll
    meta1 = {"florence_caption": "a blank wall", "llava_tension": 0.2}
    role1 = extractor._classify_role(meta1, 0.5, 0.5, ["#ffffff"])
    assert role1 == "b_roll"

    # Case 2: Subject keywords (person) -> A-roll
    meta2 = {"florence_caption": "a person walking on the beach", "llava_tension": 0.4}
    role2 = extractor._classify_role(meta2, 0.5, 0.5, ["#7f7f7f"])
    assert role2 == "a_roll"

    # Case 3: High tension + subject boxes -> A-roll
    meta3 = {
        "florence_caption": "a red car",
        "bounding_boxes": [[0.1, 0.1, 0.9, 0.9]],  # Large box
        "llava_tension": 0.8,
    }
    role3 = extractor._classify_role(meta3, 0.5, 0.5, ["#ff0000"])
    assert role3 == "a_roll"


def test_get_quadrant():
    extractor = VisualFeatureExtractor()
    assert extractor._get_quadrant(0.5, 0.5) == "center"
    assert extractor._get_quadrant(0.1, 0.1) == "top_left"
    assert extractor._get_quadrant(0.9, 0.9) == "bottom_right"
    assert extractor._get_quadrant(0.5, 0.1) == "top"
