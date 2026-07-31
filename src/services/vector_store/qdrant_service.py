"""
services.vector_store.qdrant_service
====================================
Qdrant VectorStore implementation wrapping QdrantVectorRepository.
"""

from __future__ import annotations

from typing import Any

from core.interfaces.interfaces import VectorStore
from core.models.document import Chunk
from core.models.retrieval import RetrievalResult
from services.vector_store.qdrant_repository import QdrantVectorRepository


class QdrantVectorStore(VectorStore):
    """Qdrant VectorStore adapter delegating to QdrantVectorRepository."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
    ) -> None:
        self._repo = QdrantVectorRepository(
            host=host,
            port=port,
            api_key=api_key,
            default_collection=collection_name,
            default_vector_size=vector_size,
        )

    def collection_exists(self) -> bool:
        return self._repo.collection_exists()

    def delete_collection(self) -> None:
        self._repo.delete_collection()

    def upsert(self, chunks: list[Chunk]) -> int:
        return self._repo.upsert(chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        return self._repo.search_dense(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )
