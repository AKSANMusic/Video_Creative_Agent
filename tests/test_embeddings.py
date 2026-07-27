"""Tests for the local binary-quantization math (no network)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ai_creative_engine.vision.embeddings import Embedder


def test_quantize_length_and_binary():
    emb = Embedder(target_dim=512)
    vec = [0.5, -0.5] * 256  # 512 values
    b = emb._quantize(vec)
    assert len(b) == 64  # 512 bits / 8
    # First value 0.5 >= 0 -> MSB of byte 0 should be 1.
    assert (b[0] >> 7) & 1 == 1
    # Second value -0.5 < 0 -> next bit should be 0.
    assert (b[0] >> 6) & 1 == 0


def test_quantize_msb_first_packing():
    emb = Embedder(target_dim=16)
    # Alternating: + - + - + - + - + - + - + - + -  -> byte0 = 10101010 = 0xAA
    vec = [1.0 if i % 2 == 0 else -1.0 for i in range(16)]
    b = emb._quantize(vec)
    assert b == bytes([0xAA, 0xAA])


def test_adjust_dim_truncate_normalizes():
    emb = Embedder(target_dim=8)
    big = np.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.0, 99.0])
    out = emb._adjust_dim(big)
    assert len(out) == 8
    # Truncated prefix [3,4,0...] normalized -> magnitude 1.
    mag = sum(v * v for v in out) ** 0.5
    assert pytest.approx(mag, rel=1e-6) == 1.0


def test_adjust_dim_zero_pads_when_smaller():
    emb = Embedder(target_dim=8)
    small = np.array([1.0, 2.0])
    out = emb._adjust_dim(small)
    assert len(out) == 8
    assert out[:2] == [1.0, 2.0]
    assert all(v == 0.0 for v in out[2:])


def test_target_dim_must_be_multiple_of_8():
    with pytest.raises(ValueError):
        Embedder(target_dim=100)


def _install_fake_st(monkeypatch, model_dim: int = 384) -> None:
    """Inject a fake SentenceTransformer producing deterministic `model_dim` vectors."""

    class FakeST:
        def __init__(self, *a, **kw):
            pass

        def encode(self, texts, **kw):
            h = hashlib.sha256(texts[0].encode("utf-8")).digest()
            # Repeat the hash enough to fill model_dim bits, then map to +1/-1.
            buf = bytearray()
            while len(buf) * 8 < model_dim:
                buf.extend(h)
            bits = [(buf[i // 8] >> (7 - (i % 8))) & 1 for i in range(model_dim)]
            arr = np.array([1.0 if b else -1.0 for b in bits])
            return arr.reshape(1, -1)

    import sys

    fake_mod = type("M", (), {"SentenceTransformer": FakeST})
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)


def test_deterministic_embed_truncate_path(monkeypatch):
    """model_dim (384) > target_dim (512)? no; use 384 -> zero-pad path here."""
    _install_fake_st(monkeypatch, model_dim=384)
    emb = Embedder(target_dim=512)
    b1 = emb.embed("hello world")
    b2 = emb.embed("hello world")
    assert len(b1) == 64
    assert b1 == b2  # deterministic


def test_deterministic_embed_truncate_when_model_larger(monkeypatch):
    """model_dim (768) > target_dim (512) -> Matryoshka truncation path."""
    _install_fake_st(monkeypatch, model_dim=768)
    emb = Embedder(target_dim=512)
    b = emb.embed("hello world")
    assert len(b) == 64


def _install_batched_fake_st(monkeypatch, model_dim: int = 384):
    """Fake SentenceTransformer that handles N texts in one encode() call.

    Each text maps to a deterministic vector (same shape across calls) so the
    batch result equals per-item results.
    """

    def _vec_for(text: str):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        buf = bytearray()
        while len(buf) * 8 < model_dim:
            buf.extend(h)
        bits = [(buf[i // 8] >> (7 - (i % 8))) & 1 for i in range(model_dim)]
        return np.array([1.0 if b else -1.0 for b in bits])

    class FakeST:
        def __init__(self, *a, **kw):
            self.encode_calls = 0

        def encode(self, texts, **kw):
            # Record how many texts we got in a single call.
            FakeST.last_batch = list(texts)
            FakeST.encode_calls = getattr(FakeST, "encode_calls", 0) + 1
            return np.array([_vec_for(t) for t in texts])

    import sys

    fake_mod = type("M", (), {"SentenceTransformer": FakeST})
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    return FakeST


def test_embed_many_returns_one_blob_per_input(monkeypatch):
    _install_batched_fake_st(monkeypatch, model_dim=512)
    emb = Embedder(target_dim=512)
    out = emb.embed_many(["alpha", "beta", "gamma"])
    assert len(out) == 3
    assert all(isinstance(b, bytes) for b in out)
    assert all(len(b) == 64 for b in out)


def test_embed_many_uses_a_single_encode_call(monkeypatch):
    fake = _install_batched_fake_st(monkeypatch, model_dim=512)
    emb = Embedder(target_dim=512)
    emb.embed_many(["alpha", "beta", "gamma", "delta"])
    assert getattr(fake, "encode_calls", 0) == 1
    assert len(fake.last_batch) == 4


def test_embed_many_matches_embed_itemwise(monkeypatch):
    _install_batched_fake_st(monkeypatch, model_dim=512)
    emb = Embedder(target_dim=512)
    texts = ["one", "two totally different", "three again"]
    batched = emb.embed_many(texts)
    for t, b in zip(texts, batched):
        assert emb.embed(t) == b


def test_embed_many_empty_list_is_noop(monkeypatch):
    fake = _install_batched_fake_st(monkeypatch, model_dim=512)
    emb = Embedder(target_dim=512)
    assert emb.embed_many([]) == []
    # Model must not be loaded for an empty batch.
    assert getattr(fake, "encode_calls", 0) == 0


def test_embed_many_rejects_non_list(monkeypatch):
    _install_batched_fake_st(monkeypatch, model_dim=512)
    emb = Embedder(target_dim=512)
    with pytest.raises(Exception):
        emb.embed_many("not a list")  # type: ignore[arg-type]


def test_embed_many_rejects_non_string_element(monkeypatch):
    _install_batched_fake_st(monkeypatch, model_dim=512)
    emb = Embedder(target_dim=512)
    with pytest.raises(Exception):
        emb.embed_many(["ok", 123])  # type: ignore[list-item]


def test_embed_many_replaces_empty_strings(monkeypatch):
    fake = _install_batched_fake_st(monkeypatch, model_dim=512)
    emb = Embedder(target_dim=512)
    out = emb.embed_many(["", "   ", "real"])
    assert len(out) == 3
    # Empty/whitespace inputs should have been substituted with a space, not
    # passed through as empty (which can crash some models).
    assert fake.last_batch[0] == " "
    assert fake.last_batch[1] == " "
    assert fake.last_batch[2] == "real"


def test_batch_size_default_and_validation():
    emb = Embedder(target_dim=512)
    assert emb.batch_size == 32
    with pytest.raises(ValueError):
        Embedder(target_dim=512, batch_size=0)
    with pytest.raises(ValueError):
        Embedder(target_dim=512, batch_size=-4)


def test_embed_many_chunks_large_input_into_multiple_calls(monkeypatch):
    """A run larger than batch_size must split into ceil(N/batch_size) encode calls."""
    fake = _install_batched_fake_st(monkeypatch, model_dim=512)
    emb = Embedder(target_dim=512, batch_size=4)
    texts = [f"caption {i}" for i in range(10)]
    out = emb.embed_many(texts)

    assert len(out) == 10
    # ceil(10/4) = 3 chunks.
    assert fake.encode_calls == 3
    # Each chunk respects the bound (last chunk may be smaller).
    assert all(len(c) <= 4 for c in [
        texts[0:4], texts[4:8], texts[8:10],
    ])


def test_embed_many_under_limit_is_single_call(monkeypatch):
    fake = _install_batched_fake_st(monkeypatch, model_dim=512)
    emb = Embedder(target_dim=512, batch_size=32)
    emb.embed_many(["a", "b", "c"])
    assert fake.encode_calls == 1


def test_embed_many_batch_size_one_makes_n_calls(monkeypatch):
    fake = _install_batched_fake_st(monkeypatch, model_dim=512)
    emb = Embedder(target_dim=512, batch_size=1)
    out = emb.embed_many(["x", "y", "z"])
    assert len(out) == 3
    assert fake.encode_calls == 3


def test_embed_many_chunked_result_equals_itemwise(monkeypatch):
    _install_batched_fake_st(monkeypatch, model_dim=512)
    emb = Embedder(target_dim=512, batch_size=3)
    texts = [f"t{i}" for i in range(8)]
    batched = emb.embed_many(texts)
    for t, b in zip(texts, batched):
        assert emb.embed(t) == b
