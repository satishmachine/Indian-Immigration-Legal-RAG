"""Unit tests for core settings module."""

from __future__ import annotations

import os

import pytest

from core.config.settings import AppEnvironment, get_settings, reset_settings


@pytest.mark.unit
class TestSettings:
    """Tests for the Settings singleton and configuration loading."""

    def test_default_environment_is_development(self) -> None:
        """Settings should default to 'development' environment."""
        s = get_settings()
        assert s.app.env == AppEnvironment.DEVELOPMENT

    def test_singleton_returns_same_instance(self) -> None:
        """get_settings() must return the same object on repeated calls."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reset_settings_forces_reload(self) -> None:
        """reset_settings() must invalidate the singleton."""
        s1 = get_settings()
        reset_settings()
        s2 = get_settings()
        assert s1 is not s2

    def test_is_development_flag(self) -> None:
        """is_development property must be True for development env."""
        s = get_settings()
        assert s.is_development is True
        assert s.is_production is False

    def test_qdrant_url_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """QdrantSettings.url should build a correct HTTP URL."""
        monkeypatch.setenv("QDRANT_HOST", "qdrant.example.com")
        monkeypatch.setenv("QDRANT_PORT", "6333")
        monkeypatch.setenv("QDRANT_HTTPS", "false")
        reset_settings()
        s = get_settings()
        assert s.qdrant.url == "http://qdrant.example.com:6333"

    def test_qdrant_https_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """QdrantSettings.url should use https when configured."""
        monkeypatch.setenv("QDRANT_HTTPS", "true")
        reset_settings()
        s = get_settings()
        assert s.qdrant.url.startswith("https://")

    def test_retrieval_top_k_validation_passes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Valid top_k_rerank must not raise."""
        monkeypatch.setenv("RETRIEVAL_TOP_K_DENSE", "20")
        monkeypatch.setenv("RETRIEVAL_TOP_K_SPARSE", "20")
        monkeypatch.setenv("RETRIEVAL_TOP_K_RERANK", "5")
        reset_settings()
        s = get_settings()
        assert s.retrieval.top_k_rerank == 5

    def test_chunk_overlap_less_than_chunk_size(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Chunk overlap must be less than chunk size or validation fails."""
        monkeypatch.setenv("INGESTION_CHUNK_SIZE", "500")
        monkeypatch.setenv("INGESTION_CHUNK_OVERLAP", "600")  # invalid
        reset_settings()
        with pytest.raises(Exception):  # noqa: BLE001  (pydantic ValidationError)
            get_settings()
