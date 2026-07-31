"""
retrieval.reranker.api_reranker
===============================
API-based Reranker Provider.

Performs remote API re-ranking using API services (e.g. Cohere, Euri API) or
hybrid RRF score pass-through, eliminating local neural model downloads and
Hugging Face dependencies.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from core.interfaces.reranker_interface import RerankerService
from core.models.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class ApiReranker(RerankerService):
    """
    API Reranker Service using external API calls or score pass-through.

    Args:
        model_name: Reranker model identifier (default: rerank-v3.5).
        api_key: Optional API key.
    """

    def __init__(
        self,
        model_name: str = "rerank-v3.5",
        api_key: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key or os.getenv("COHERE_API_KEY")

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Re-rank candidate passages via API service or hybrid rank pass-through.
        """
        if not results or not query.strip():
            return []

        limit = top_n or len(results)

        # Attempt API reranking if Cohere API key is configured
        if self._api_key:
            try:
                import cohere
                client = cohere.ClientV2(api_key=self._api_key)
                docs = [r.content for r in results]
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
                            "score": round(float(hit.relevance_score), 4),
                            "rank": rank,
                            "retrieval_method": "cohere_api_rerank",
                        }
                    )
                    reranked.append(res_copy)
                return reranked

            except Exception as exc:
                logger.warning("API reranker invocation failed (%s). Falling back to RRF score ranking.", exc)

        # Fallback / Default API pass-through: Sort descending by hybrid search score
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        final_results = sorted_results[:limit]

        for rank, res in enumerate(final_results):
            res.rank = rank
            res.retrieval_method = "api_hybrid_pass_through"

        logger.info("ApiReranker returned top %d passages using hybrid score ranking", len(final_results))
        return final_results
