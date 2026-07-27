"""Shared pytest fixtures and helpers for offline tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from ai_creative_engine.vision.embeddings import Embedder, embedding_to_hex
from ai_creative_engine.vision.florence import FlorenceResult
from ai_creative_engine.vision.llava import LLaVAResult


def make_image(path: Path, color: str = "red") -> Path:
    """Write a tiny deterministic image whose bytes are unique per (color).

    Different colors => different bytes => distinct sha1 image_ids, so the
    pipeline doesn't treat multiple fixtures as the same image.
    """
    img = Image.new("RGB", (64, 48), color)
    # Add a distinct drawn marker so even same-color images differ.
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), color, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


class FakeFlorence:
    """Deterministic stand-in for FlorenceClient. Never calls the network."""

    def __init__(self, result: FlorenceResult | None = None, fail_on: set[str] | None = None) -> None:
        self.result = result or FlorenceResult(
            caption="a colored square",
            objects=["square"],
            bounding_boxes=[[0.1, 0.1, 0.9, 0.9]],
        )
        self.fail_on = fail_on or set()
        self.calls: list[str] = []

    def extract(self, image_path: Path | str) -> FlorenceResult:
        name = Path(image_path).name
        self.calls.append(name)
        if name in self.fail_on:
            raise RuntimeError(f"forced failure for {name}")
        return self.result


class FakeLLaVA:
    """Deterministic stand-in for LLaVAClient. Never calls the network."""

    def __init__(self, result: LLaVAResult | None = None, fail_on: set[str] | None = None) -> None:
        self.result = result or LLaVAResult(mood="tense", tension=0.8, symbolism="a warning sign")
        self.fail_on = fail_on or set()
        self.calls: list[str] = []

    def extract(self, image_path: Path | str) -> LLaVAResult:
        name = Path(image_path).name
        self.calls.append(name)
        if name in self.fail_on:
            raise RuntimeError(f"forced failure for {name}")
        return self.result


class DeterministicEmbedder:
    """Bypasses sentence-transformers: deterministic bits derived from text sha1.

    Exposes the same public surface as :class:`Embedder` so the pipeline and CLI
    can use it interchangeably.
    """

    def __init__(self, target_dim: int = 512) -> None:
        self.target_dim = target_dim

    def embed(self, text: str) -> bytes:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        need = self.target_dim // 8
        buf = bytearray()
        while len(buf) < need:
            buf.extend(digest)
        return bytes(buf[:need])
 
    def embed_many(self, texts: list[str]) -> list[bytes]:
        """Batch equivalent of :meth:`embed`; mirrors Embedder.embed_many."""
        return [self.embed(t) for t in texts]

    def to_hex(self, text: str) -> str:
        return embedding_to_hex(self.embed(text))


@pytest.fixture
def tmp_images_dir(tmp_path: Path) -> Path:
    """A directory with 3 byte-distinct images (red/green/blue)."""
    d = tmp_path / "imgs"
    d.mkdir()
    make_image(d / "a.png", color="red")
    make_image(d / "b.png", color="green")
    make_image(d / "c.png", color="blue")
    return d


@pytest.fixture
def fakes() -> dict[str, Any]:
    return {
        "florence": FakeFlorence(),
        "llava": FakeLLaVA(),
        "embedder": DeterministicEmbedder(),
    }
