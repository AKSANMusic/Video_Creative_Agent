"""Unit tests for scoring functions (pure math, no I/O)."""

from __future__ import annotations

import pytest

from ai_creative_engine.narrative import scoring


def test_tension_energy_score_perfect():
    assert scoring.tension_energy_score(0.5, 0.5) == pytest.approx(1.0)


def test_tension_energy_score_opposite():
    assert scoring.tension_energy_score(0.0, 1.0) == pytest.approx(0.0)


def test_tension_energy_score_clamps():
    # Out-of-range inputs are clamped, never raise.
    assert scoring.tension_energy_score(2.0, -1.0) == pytest.approx(0.0)


def test_hamming_distance_bytes_basic():
    a = bytes([0b00000000])
    b = bytes([0b11111111])
    assert scoring.hamming_distance_bytes(a, b) == 8
    assert scoring.hamming_distance_bytes(a, a) == 0


def test_hamming_distance_length_mismatch():
    with pytest.raises(ValueError):
        scoring.hamming_distance_bytes(b"\x00", b"\x00\x00")


def test_embedding_similarity_range():
    a = bytes([0] * 64)
    b = bytes([255] * 64)
    assert scoring.embedding_similarity(a, a) == pytest.approx(1.0)
    assert scoring.embedding_similarity(a, b) == pytest.approx(0.0)


def test_jsd_identical_zero():
    p = [0.25, 0.25, 0.25, 0.25]
    assert scoring.jsd(p, p) == pytest.approx(0.0, abs=1e-9)


def test_jsd_disjoint_high():
    p = [1.0, 0.0, 0.0, 0.0]
    q = [0.0, 0.0, 0.0, 1.0]
    # JSD base-2 over disjoint normalized dists -> <= 1, and noticeably > 0.
    val = scoring.jsd(p, q)
    assert 0.0 < val <= 1.0


def test_jsd_length_mismatch():
    with pytest.raises(ValueError):
        scoring.jsd([1.0], [1.0, 0.0])


def test_color_continuity_identical_one():
    a = bytes([10, 20, 30])
    assert scoring.color_continuity_from_embeddings(a, a) == pytest.approx(1.0, abs=1e-9)


def test_transition_cost_lower_for_similar():
    base = bytes([0] * 64)
    near = bytes([0] * 63 + [1])  # 1 bit different
    far = bytes([255] * 64)
    cost_near = scoring.transition_cost(base, near)
    cost_far = scoring.transition_cost(base, far)
    assert cost_near < cost_far
    assert 0.0 <= cost_near <= 1.0
    assert 0.0 <= cost_far <= 1.0


def test_transition_cost_override_wins_over_global():
    """When continuity_weight_override is given it replaces the global weight."""
    a = bytes([0] * 64)
    b = bytes([255] * 64)
    # Global weight 0.5 vs override 0.5 -> identical cost.
    base = scoring.transition_cost(a, b, continuity_weight=0.5)
    same = scoring.transition_cost(a, b, continuity_weight=0.9, continuity_weight_override=0.5)
    assert base == pytest.approx(same, abs=1e-12)


def test_transition_cost_override_changes_cost_monotonically():
    """Cost must respond to the override value across the blend range."""
    a = bytes(list(range(64)))  # diverse bytes -> non-trivial color histogram
    b = bytes(list(range(63, -1, -1)))
    # The blended cost is (1-cw)*sim + cw*cont; cost = 1 - blended. Different
    # override values must therefore yield different costs (sim != cont here).
    c_low = scoring.transition_cost(a, b, continuity_weight=0.0, continuity_weight_override=0.0)
    c_high = scoring.transition_cost(a, b, continuity_weight=0.0, continuity_weight_override=1.0)
    assert c_low != pytest.approx(c_high, abs=1e-9)


def test_transition_cost_override_none_uses_global():
    """None override is the explicit backwards-compatible path."""
    a = bytes([1] * 64)
    b = bytes([2] * 64)
    via_global = scoring.transition_cost(a, b, continuity_weight=0.3)
    via_none = scoring.transition_cost(a, b, continuity_weight=0.3, continuity_weight_override=None)
    assert via_global == pytest.approx(via_none, abs=1e-12)
