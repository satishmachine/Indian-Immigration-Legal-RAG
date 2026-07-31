"""
services.embedding.factory
===========================
Clean Architecture Factory for OpenAI / Euri Embedding Provider.

Primary Provider: OpenAI (text-embedding-ada-002, 1536 dim) via Euri API Proxy.
"""

from __future__ import annotations

import logging
from typing import Any

from core.config import get_settings
from core.config.settings import EmbeddingProvider
from services.embedding.base import BaseEmbeddingService
from services.embedding.openai_provider import OpenAIEmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """Factory class to instantiate OpenAI / Euri embedding service."""

    @classmethod
    def create(
        cls,
        provider: str | EmbeddingProvider | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> BaseEmbeddingService:
        """
        Instantiate OpenAI / Euri embedding provider (text-embedding-ada-002).

        Args:
            provider: Provider identifier string or Enum (defaults to 'openai').
            model_name: Model identifier override (defaults to text-embedding-ada-002).
            kwargs: Constructor options.

        Returns:
            OpenAIEmbeddingService instance.
        """
        cfg = get_settings().embedding
        model = model_name or cfg.model_name or "text-embedding-ada-002"
        logger.info("Instantiating OpenAIEmbeddingService with model %r", model)
        return OpenAIEmbeddingService(model_name=model, **kwargs)


def get_embedding_service(
    provider: str | EmbeddingProvider | None = None,
    model_name: str | None = None,
    **kwargs: Any,
) -> BaseEmbeddingService:
    """Helper function to get configured embedding service instance."""
    return EmbeddingFactory.create(provider=provider, model_name=model_name, **kwargs)
