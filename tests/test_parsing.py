"""Unit tests for the defensive Florence/LLaVA parsing logic (no network)."""

from __future__ import annotations

import pytest

from ai_creative_engine.vision.florence import FlorenceClient
from ai_creative_engine.vision.llava import LLaVAClient


def test_florence_parses_od_dict():
    # Florence-2 emits coords in [0, 1000]; full-frame box -> [0,0,1,1].
    res = FlorenceClient._parse(
        "a small red square",
        {"<OD>": {"bboxes": [[0, 0, 1000, 1000]], "labels": ["square"]}},
        "x.png",
    )
    assert res.caption == "a small red square"
    assert res.objects == ["square"]
    assert res.bounding_boxes == [[0.0, 0.0, 1.0, 1.0]]


def test_florence_scales_1000_range():
    res = FlorenceClient._parse(
        "c",
        {"<OD>": {"bboxes": [[100, 200, 500, 800]], "labels": ["x"]}},
        "x.png",
    )
    assert res.bounding_boxes == [[0.1, 0.2, 0.5, 0.8]]


def test_florence_normalizes_already_normalized():
    # Coords already in [0, 1] are left untouched.
    res = FlorenceClient._parse(
        "c",
        {"<OD>": {"bboxes": [[0.1, 0.2, 0.9, 1.0]], "labels": ["x"]}},
        "x.png",
    )
    assert res.bounding_boxes == [[0.1, 0.2, 0.9, 1.0]]


def test_florence_handles_missing_od():
    res = FlorenceClient._parse("caption only", None, "x.png")
    assert res.caption == "caption only"
    assert res.objects == []


def test_florence_handles_label_box_mismatch():
    res = FlorenceClient._parse(
        "c",
        {"<OD>": {"bboxes": [[0, 0, 500, 500], [0, 0, 500, 500]], "labels": ["a"]}},
        "x.png",
    )
    # Mismatched counts -> we still zip and get 1 valid pair.
    assert res.objects == ["a"]
    assert res.bounding_boxes == [[0.0, 0.0, 0.5, 0.5]]


def test_llava_parses_json_string():
    res = LLaVAClient._parse('{"mood":"calm","tension":0.25,"symbolism":"peace"}', "x.png")
    assert res.mood == "calm"
    assert res.tension == pytest.approx(0.25)
    assert res.symbolism == "peace"


def test_llava_parses_embedded_json_from_text():
    text = "Here is my analysis: {\"mood\":\"tense\",\"tension\":0.9,\"symbolism\":\"danger\"} thanks"
    res = LLaVAClient._parse(text, "x.png")
    assert res.mood == "tense"
    assert res.tension == pytest.approx(0.9)


def test_llava_clamps_tension():
    res = LLaVAClient._parse('{"mood":"x","tension":5.0}', "x.png")
    assert res.tension == 1.0
    res2 = LLaVAClient._parse('{"mood":"x","tension":-3.0}', "x.png")
    assert res2.tension == 0.0


def test_llava_falls_back_on_unparseable():
    res = LLaVAClient._parse("totally not json at all", "x.png")
    assert res.mood == "unknown"
    assert res.tension == 0.5
    assert res.symbolism == ""


def test_llava_empty_text_defaults():
    res = LLaVAClient._parse("", "x.png")
    assert res.mood == "unknown"
