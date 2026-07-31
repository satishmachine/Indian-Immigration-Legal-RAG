"""
retrieval.dense.dense_retriever
===============================
Dense Vector Similarity Retriever.

Interacts with EmbeddingService and VectorStoreRepository to execute
high-dimensional vector search with metadata filtering.
"""

from __future__ import annotations

import logging
from typing import Any

from core.exceptions import DenseRetrievalError, EmbeddingQuotaExceededError
from core.interfaces.interfaces import EmbeddingService
from core.interfaces.vector_store_repository import VectorStoreRepository
from core.models.retrieval import RetrievalResult
from services.embedding import get_embedding_service
from services.vector_store import get_vector_repository

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense Vector Similarity Retriever.

    Args:
        embedding_service: EmbeddingService instance.
        vector_repository: VectorStoreRepository instance.
        collection_name: Qdrant collection name override.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_repository: VectorStoreRepository | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_repository = vector_repository or get_vector_repository()
        self.collection_name = collection_name

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
    ) -> list[RetrievalResult]:
        """
        Embed query string and execute dense similarity vector search.

        Args:
            query: User search query text.
            top_k: Number of top results to retrieve.
            filters: Metadata filter dictionary.
            score_threshold: Minimum similarity score threshold.

        Returns:
            List of RetrievalResult objects.
        """
        if not query.strip():
            return []

        # 1. Generate query embedding
        try:
            query_vector = self.embedding_service.embed_query(query)
            if not query_vector:
                raise DenseRetrievalError("Embedding service returned empty query vector.")
        except Exception as exc:
            if isinstance(exc, (DenseRetrievalError, EmbeddingQuotaExceededError)):
                raise exc
            err_str = str(exc).lower()
            if "403" in err_str or "quota" in err_str or "token limit" in err_str or "permission_denied" in err_str:
                logger.error("Dense retrieval embedding quota exceeded: %s", exc, exc_info=True)
                raise EmbeddingQuotaExceededError(
                    message=f"Embedding API quota exceeded: {exc}",
                    details={"query": query, "raw_error": str(exc)},
                ) from exc
            logger.error("Dense retrieval embedding failed: %s", exc, exc_info=True)
            raise DenseRetrievalError(
                message=f"Dense embedding generation failed: {exc}",
                details={"query": query, "raw_error": str(exc)},
            ) from exc

        # 2. Execute dense similarity search in vector database
        try:
            results = self.vector_repository.search_dense(
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
                score_threshold=score_threshold,
                collection_name=self.collection_name,
            )
        except Exception as exc:
            logger.error("Dense vector database search failed: %s", exc, exc_info=True)
            raise DenseRetrievalError(
                message=f"Vector database search failed: {exc}",
                details={"query": query, "raw_error": str(exc)},
            ) from exc

        logger.debug(
            "DenseRetriever retrieved %d results for query %r",
            len(results),
            query[:50],
        )
        return results
