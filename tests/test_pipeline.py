"""Pipeline tests with fake vision clients (no network)."""

from __future__ import annotations

from pathlib import Path

from ai_creative_engine.cache import MetadataCache
from ai_creative_engine.pipeline import ExtractionPipeline
from tests.conftest import FakeFlorence, FakeLLaVA, DeterministicEmbedder, make_image


def _pipeline(tmp_path: Path, **overrides) -> ExtractionPipeline:
    florence = overrides.get("florence", FakeFlorence())
    llava = overrides.get("llava", FakeLLaVA())
    embedder = overrides.get("embedder", DeterministicEmbedder())
    cache = MetadataCache(tmp_path / "t.db")
    return ExtractionPipeline(florence=florence, llava=llava, embedder=embedder, cache=cache)


def _make_three(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    make_image(d / "a.png", color="red")
    make_image(d / "b.png", color="green")
    make_image(d / "c.png", color="blue")


def test_run_processes_all_new(tmp_path: Path):
    d = tmp_path / "imgs"
    _make_three(d)
    pipe = _pipeline(tmp_path)
    stats = pipe.run(d)
    assert (stats.new, stats.cached, stats.failed, stats.total) == (3, 0, 0, 3)
    assert pipe.cache.count() == 3


def test_second_run_is_all_cache_hits(tmp_path: Path):
    d = tmp_path / "imgs"
    _make_three(d)
    pipe = _pipeline(tmp_path)
    pipe.run(d)
    stats2 = pipe.run(d)
    assert (stats2.new, stats2.cached) == (0, 3)


def test_fail_soft_continues_after_one_image_error(tmp_path: Path):
    d = tmp_path / "imgs"
    _make_three(d)
    florence = FakeFlorence(fail_on={"b.png"})
    pipe = _pipeline(tmp_path, florence=florence)
    stats = pipe.run(d)
    assert stats.failed == 1
    assert stats.new == 2
    assert "b.png" in stats.failed_files


def test_only_supported_files_processed(tmp_path: Path):
    d = tmp_path / "imgs"
    d.mkdir()
    make_image(d / "a.png", color="red")
    (d / "notes.txt").write_text("ignore me")
    (d / "ignore.gif").write_bytes(b"not an image really")
    pipe = _pipeline(tmp_path)
    stats = pipe.run(d)
    assert stats.total == 1  # only a.png counted


def test_max_images_caps_count(tmp_path: Path):
    d = tmp_path / "imgs"
    _make_three(d)
    cache = MetadataCache(tmp_path / "t.db")
    pipe = ExtractionPipeline(
        florence=FakeFlorence(),
        llava=FakeLLaVA(),
        embedder=DeterministicEmbedder(),
        cache=cache,
        max_images=2,
    )
    stats = pipe.run(d)
    assert stats.total == 2
    assert stats.new == 2


class _CountingEmbedder:
    """Wraps DeterministicEmbedder to count embed_many calls and capture inputs."""

    def __init__(self) -> None:
        self._inner = DeterministicEmbedder()
        self.embed_calls = 0
        self.embed_many_calls = 0
        self.batches: list[list[str]] = []

    def embed(self, text: str) -> bytes:
        self.embed_calls += 1
        return self._inner.embed(text)

    def embed_many(self, texts: list[str]) -> list[bytes]:
        self.embed_many_calls += 1
        self.batches.append(list(texts))
        return self._inner.embed_many(texts)


def test_run_calls_embed_many_exactly_once(tmp_path: Path):
    """The batched pipeline must embed all new captions in a single encode call."""
    d = tmp_path / "imgs"
    _make_three(d)
    embedder = _CountingEmbedder()
    pipe = _pipeline(tmp_path, embedder=embedder)
    stats = pipe.run(d)

    assert (stats.new, stats.failed) == (3, 0)
    assert embedder.embed_many_calls == 1
    assert embedder.embed_calls == 0
    assert len(embedder.batches) == 1
    assert len(embedder.batches[0]) == 3


def test_run_skips_cached_before_embedding(tmp_path: Path):
    """Second run should hit cache and not invoke the embedder at all."""
    d = tmp_path / "imgs"
    _make_three(d)
    embedder = _CountingEmbedder()
    pipe = _pipeline(tmp_path, embedder=embedder)
    pipe.run(d)

    # Fresh counter for the second run on the same pipeline (cache populated).
    embedder2 = _CountingEmbedder()
    pipe2 = _pipeline(tmp_path, embedder=embedder2)
    stats2 = pipe2.run(d)

    assert (stats2.new, stats2.cached) == (0, 3)
    assert embedder2.embed_many_calls == 0
    assert embedder2.embed_calls == 0


def test_run_batches_only_successful_extractions(tmp_path: Path):
    """A failed vision extraction must be excluded from the embed batch."""
    d = tmp_path / "imgs"
    _make_three(d)
    embedder = _CountingEmbedder()
    pipe = _pipeline(tmp_path, florence=FakeFlorence(fail_on={"b.png"}), embedder=embedder)
    stats = pipe.run(d)

    assert (stats.new, stats.failed) == (2, 1)
    assert embedder.embed_many_calls == 1
    # Only the two successful captions reach the batch.
    assert len(embedder.batches[0]) == 2


def test_run_embedding_failure_fails_soft(tmp_path: Path, monkeypatch):
    """If the batched embed call raises, all pending images are failed, not crashed."""
    d = tmp_path / "imgs"
    _make_three(d)

    class _BoomEmbedder(_CountingEmbedder):
        def embed_many(self, texts: list[str]) -> list[bytes]:
            self.embed_many_calls += 1
            raise RuntimeError("forced embed failure")

    embedder = _BoomEmbedder()
    pipe = _pipeline(tmp_path, embedder=embedder)
    stats = pipe.run(d)

    assert stats.new == 0
    assert stats.failed == 3
    assert set(stats.failed_files) == {"a.png", "b.png", "c.png"}
