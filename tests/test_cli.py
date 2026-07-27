"""CLI smoke tests using typer's CliRunner (offline)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_creative_engine import cli as cli_module
from ai_creative_engine.cli import app
from tests.conftest import FakeFlorence, FakeLLaVA, DeterministicEmbedder, make_image


def test_missing_token_exits_non_zero(tmp_path, monkeypatch):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    # Ensure no .env is picked up from the CWD-ish location.
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["extract", "--images", str(tmp_path)])
    assert result.exit_code == 2
    assert "REPLICATE_API_TOKEN" in result.output


def _patch_pipeline(monkeypatch, tmp_path):
    """Replace _build_pipeline with one using fake clients."""
    from ai_creative_engine.cache import MetadataCache
    from ai_creative_engine.pipeline import ExtractionPipeline

    def fake_build(settings, max_images):
        return ExtractionPipeline(
            florence=FakeFlorence(),
            llava=FakeLLaVA(),
            embedder=DeterministicEmbedder(),
            cache=MetadataCache(settings.db_path),
            max_images=max_images,
        )

    monkeypatch.setattr(cli_module, "_build_pipeline", fake_build)


def test_extract_happy_path(tmp_path, monkeypatch):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "fake-token-for-tests")
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    make_image(imgs / "a.png")
    _patch_pipeline(monkeypatch, tmp_path)

    db = tmp_path / "out.db"
    runner = CliRunner()
    result = runner.invoke(app, ["extract", "--images", str(imgs), "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "new=1" in result.output


def test_info_command(tmp_path, monkeypatch):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "fake-token-for-tests")
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    make_image(imgs / "a.png")
    _patch_pipeline(monkeypatch, tmp_path)

    db = tmp_path / "out.db"
    runner = CliRunner()
    runner.invoke(app, ["extract", "--images", str(imgs), "--db", str(db)])
    info = runner.invoke(app, ["info", "--db", str(db)])
    assert info.exit_code == 0, info.output
    assert "Cached images" in info.output
    assert "1" in info.output
