"""Scoring functions for the narrative sequencer.

All functions are pure, deterministic, and unit-testable. No I/O, no randomness.

Signals used
------------
- **Tension-energy match:** how well an image's narrative tension
  (``ImageMetadata.llava_tension`` in [0,1]) fits a section's mean energy
  (``Section.mean_energy`` in [0,1]). Score = 1 - |tension - energy|.
- **Embedding continuity:** Hamming distance between consecutive images'
  512-d binary embeddings (decoded from ``embedding_hex``). We convert to a
  similarity in [0,1] = 1 - (hamming_bits / 512). Low Hamming => visually/
  semantically similar => smoother cut.
- **JSD proxy for color continuity:** the architecture spec references
  Jensen-Shannon divergence over color distributions. We do not have a true
  color histogram in Stage-1 metadata, so we expose a generic ``jsd`` over
  two discrete distributions and a ``color_continuity_from_embeddings`` that
  derives a coarse pseudo-histogram from the binary embedding bytes. This is
  a documented, stable proxy that keeps Stage 3 self-contained; a future
  Stage-1 enhancement can swap in real color palettes without changing the
  sequencer's interface.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

EMBEDDING_BITS = 512


def tension_energy_score(tension: float, energy: float) -> float:
    """Return similarity in [0,1] between an image tension and a section energy."""
    t = float(min(max(tension, 0.0), 1.0))
    e = float(min(max(energy, 0.0), 1.0))
    return 1.0 - abs(t - e)


def hamming_distance_bytes(a: bytes, b: bytes) -> int:
    """Bit-wise Hamming distance between two equal-length byte strings."""
    if len(a) != len(b):
        raise ValueError(f"embedding length mismatch: {len(a)} vs {len(b)}")
    dist = 0
    for x, y in zip(a, b):
        # Brian Kernighan's bit count over XOR.
        v = x ^ y
        while v:
            dist += v & 1
            v >>= 1
    return dist


def embedding_similarity(a: bytes, b: bytes) -> float:
    """Return [0,1] similarity from Hamming distance (assumes 512-bit embeddings)."""
    if not a or not b:
        return 0.0
    dist = hamming_distance_bytes(a, b)
    bits = max(len(a), len(b)) * 8
    return 1.0 - (dist / bits)


def _bytes_to_histogram(data: bytes, bins: int = 16) -> list[float]:
    """Derive a normalized histogram from byte values (stable pseudo color hist)."""
    if not data:
        return [0.0] * bins
    counts = [0] * bins
    scale = 256 / bins
    for byte in data:
        idx = min(bins - 1, int(byte / scale))
        counts[idx] += 1
    total = sum(counts)
    if total == 0:
        return [0.0] * bins
    return [c / total for c in counts]


def jsd(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence between two discrete distributions.

    Returns a value in [0, 1] (natural-log base 2). 0 = identical.
    """
    if len(p) != len(q):
        raise ValueError(f"distribution length mismatch: {len(p)} vs {len(q)}")
    eps = 1e-12
    p = [max(float(x), eps) for x in p]
    q = [max(float(x), eps) for x in q]
    sp = sum(p)
    sq = sum(q)
    p = [x / sp for x in p]
    q = [x / sq for x in q]
    m = [(pv + qv) / 2.0 for pv, qv in zip(p, q)]

    def _kl(a: list[float], b: list[float]) -> float:
        return sum(av * math.log2(av / bv) for av, bv in zip(a, b) if av > 0)

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    # Clamp; JSD over normalized dists with base-2 is in [0, 1].
    return float(min(max(js, 0.0), 1.0))


def color_continuity_from_embeddings(a: bytes, b: bytes, bins: int = 16) -> float:
    """Return [0,1] color-continuity proxy from two embeddings (1 = identical)."""
    if not a or not b:
        return 0.0
    return 1.0 - jsd(_bytes_to_histogram(a, bins), _bytes_to_histogram(b, bins))


def color_continuity(
    a_hist: Optional[Sequence[float]] = None,
    b_hist: Optional[Sequence[float]] = None,
    a_bytes: Optional[bytes] = None,
    b_bytes: Optional[bytes] = None,
    bins: int = 16,
) -> float:
    """Return [0,1] color continuity (1 = identical).

    If true color histograms are provided, computes 1 - JSD directly.
    Otherwise falls back to embedding pseudo-histograms.
    """
    if a_hist and b_hist and len(a_hist) == len(b_hist) and len(a_hist) > 0 and sum(a_hist) > 0 and sum(b_hist) > 0:
        return 1.0 - jsd(a_hist, b_hist)
    if a_bytes and b_bytes:
        return color_continuity_from_embeddings(a_bytes, b_bytes, bins=bins)
    return 0.0


def transition_cost(
    prev_bytes: bytes,
    cur_bytes: bytes,
    continuity_weight: float = 0.5,
    continuity_weight_override: Optional[float] = None,
    bins: int = 16,
    prev_hist: Optional[Sequence[float]] = None,
    cur_hist: Optional[Sequence[float]] = None,
) -> float:
    """Combined cost of placing ``cur`` right after ``prev``.

    Lower is better. Blends embedding similarity (semantic smoothness) with a
    color-continuity proxy. ``continuity_weight`` in [0,1] controls the blend.
    If ``continuity_weight_override`` is not None it wins over the global
    ``continuity_weight`` (per-section director LLM cut-steering).
    """
    cw = continuity_weight_override if continuity_weight_override is not None else continuity_weight
    cw = float(min(max(cw, 0.0), 1.0))
    sim = embedding_similarity(prev_bytes, cur_bytes)
    cont = color_continuity(prev_hist, cur_hist, a_bytes=prev_bytes, b_bytes=cur_bytes, bins=bins)
    blended = (1.0 - cw) * sim + cw * cont
    # Cost = 1 - blended_similarity, in [0,1].
    return 1.0 - blended


__all__ = [
    "EMBEDDING_BITS",
    "tension_energy_score",
    "hamming_distance_bytes",
    "embedding_similarity",
    "jsd",
    "color_continuity",
    "color_continuity_from_embeddings",
    "transition_cost",
]
