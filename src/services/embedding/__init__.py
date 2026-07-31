"""
services.embedding
==================
Clean Architecture Embedding Pipeline.

Public API:
- BaseEmbeddingService (ABC)
- OpenAIEmbeddingService (text-embedding-ada-002, 1536 dim)
- EmbeddingFactory
- get_embedding_service()
"""

from __future__ import annotations

from core.interfaces.interfaces import EmbeddingService
from services.embedding.base import BaseEmbeddingService
from services.embedding.factory import EmbeddingFactory
from services.embedding.openai_provider import OpenAIEmbeddingService

__all__: list[str] = [
    "BaseEmbeddingService",
    "EmbeddingFactory",
    "EmbeddingService",
    "OpenAIEmbeddingService",
    "get_embedding_service",
]

_embedding_service_instance: BaseEmbeddingService | None = None


def get_embedding_service(
    provider: str | None = None,
    model_name: str | None = None,
    force_new: bool = False,
) -> BaseEmbeddingService:
    """
    Factory function returning OpenAIEmbeddingService instance.

    Args:
        provider: Provider identifier string (defaults to 'openai').
        model_name: Model name override (defaults to 'text-embedding-ada-002').
        force_new: Force creating a new instance instead of returning cached singleton.

    Returns:
        Configured BaseEmbeddingService instance.
    """
    global _embedding_service_instance  # noqa: PLW0603
    if _embedding_service_instance is None or force_new or provider or model_name:
        instance = EmbeddingFactory.create(provider=provider, model_name=model_name)
        if not (provider or model_name) and not force_new:
            _embedding_service_instance = instance
        return instance
    return _embedding_service_instance
