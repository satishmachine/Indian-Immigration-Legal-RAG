"""
retrieval.reranker
==================
Reranking engine package.

Public API:
- RerankerService (ABC)
- BGEReranker (BAAI/bge-reranker-v2-m3)
- CrossEncoderReranker
- CohereReranker
- RerankerFactory
- get_reranker()
"""

from __future__ import annotations

from core.interfaces.reranker_interface import RerankerService
from retrieval.reranker.api_reranker import ApiReranker
from retrieval.reranker.cohere_reranker import CohereReranker
from retrieval.reranker.factory import RerankerFactory

__all__: list[str] = [
    "ApiReranker",
    "CohereReranker",
    "RerankerFactory",
    "RerankerService",
    "get_reranker",
]

_reranker_instance: RerankerService | None = None


def get_reranker(
    provider: str | None = None,
    model_name: str | None = None,
) -> RerankerService:
    """Factory function returning the configured RerankerService instance."""
    global _reranker_instance  # noqa: PLW0603
    if _reranker_instance is None or provider or model_name:
        instance = RerankerFactory.create(provider=provider, model_name=model_name)
        if not (provider or model_name):
            _reranker_instance = instance
        return instance
    return _reranker_instance
