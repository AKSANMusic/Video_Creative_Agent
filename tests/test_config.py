"""Tests for AppConfig (Settings): defaults, env overrides, validators."""

from __future__ import annotations

import pytest

from ai_creative_engine.config import Settings


def test_defaults():
    s = Settings()
    assert s.embedding_batch_size == 32
    assert s.embedding_dim == 384


def test_embedding_batch_size_env_override(monkeypatch):
    monkeypatch.setenv("ACE_EMBEDDING_BATCH_SIZE", "64")
    s = Settings()
    assert s.embedding_batch_size == 64


def test_embedding_batch_size_rejects_zero(monkeypatch):
    monkeypatch.setenv("ACE_EMBEDDING_BATCH_SIZE", "0")
    with pytest.raises(Exception):
        Settings()


def test_embedding_batch_size_rejects_negative(monkeypatch):
    monkeypatch.setenv("ACE_EMBEDDING_BATCH_SIZE", "-1")
    with pytest.raises(Exception):
        Settings()
