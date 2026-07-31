"""
retrieval.reranker.cohere_reranker
===================================
Cohere Reranker Provider implementation.

Uses Cohere's state-of-the-art API models:
- rerank-v3.5
- rerank-english-v3.0
- rerank-multilingual-v3.0
"""

from __future__ import annotations

import logging
import os
from core.interfaces.reranker_interface import RerankerService
from core.models.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class CohereReranker(RerankerService):
    """
    Cohere API Reranker Service.

    Args:
        api_key: Cohere API key (defaults to COHERE_API_KEY env var).
        model_name: Cohere model identifier (default: rerank-v3.5).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "rerank-v3.5",
    ) -> None:
        self._api_key = api_key or os.getenv("COHERE_API_KEY")
        self._model_name = model_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise ValueError("Cohere API key missing. Set COHERE_API_KEY in .env.")
            import cohere
            self._client = cohere.ClientV2(api_key=self._api_key)
        return self._client

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Re-rank candidates via Cohere API."""
        if not results or not query.strip():
            return []

        client = self._get_client()
        docs = [r.content for r in results]
        limit = top_n or len(results)

        response = client.rerank(
            model=self._model_name,
            query=query,
            documents=docs,
            top_n=limit,
        )

        reranked: list[RetrievalResult] = []
        for rank, hit in enumerate(response.results):
            original = results[hit.index]
            res_copy = original.model_copy(
                update={
                    "score": round(hit.relevance_score, 4),
                    "rank": rank,
                    "retrieval_method": "cohere_rerank",
                }
            )
            reranked.append(res_copy)

        return reranked
