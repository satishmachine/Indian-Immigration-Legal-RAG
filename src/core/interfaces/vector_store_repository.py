"""
core.interfaces.vector_store_repository
========================================
Abstract Vector Store Repository Domain Interface.

Defines Clean Architecture repository pattern methods for vector database operations:
- Collections Management
- Upsert & Batch Upsert
- Deletion (by IDs or metadata filter)
- Metadata Filtering
- Dense & Hybrid Search
- Index Management
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.models.document import Chunk
from core.models.retrieval import RetrievalResult, SearchQuery


class VectorStoreRepository(ABC):
    """Abstract Repository Interface for Vector Databases."""

    # ── Collection Management ───────────────────────────────────────────────

    @abstractmethod
    def create_collection(
        self,
        collection_name: str | None = None,
        vector_size: int | None = None,
        distance: str = "Cosine",
        recreate: bool = False,
    ) -> bool:
        """Create or recreate a vector database collection."""

    @abstractmethod
    def delete_collection(self, collection_name: str | None = None) -> bool:
        """Delete a collection."""

    @abstractmethod
    def collection_exists(self, collection_name: str | None = None) -> bool:
        """Check if collection exists."""

    @abstractmethod
    def get_collection_info(self, collection_name: str | None = None) -> dict[str, Any]:
        """Return collection configuration and stats."""

    # ── Index Management ───────────────────────────────────────────────────

    @abstractmethod
    def create_payload_index(
        self,
        field_name: str,
        field_schema: str = "keyword",
        collection_name: str | None = None,
    ) -> bool:
        """Create a payload index on a metadata field for fast filtering."""

    # ── Data Operations (Upsert & Delete) ──────────────────────────────────

    @abstractmethod
    def upsert(self, chunks: list[Chunk], collection_name: str | None = None) -> int:
        """Upsert Chunk domain entities with vectors and metadata payloads."""

    @abstractmethod
    def delete_points(self, point_ids: list[str], collection_name: str | None = None) -> int:
        """Delete points by IDs."""

    @abstractmethod
    def delete_by_filter(self, filters: dict[str, Any], collection_name: str | None = None) -> int:
        """Delete points matching metadata filter."""

    # ── Search & Retrieval ──────────────────────────────────────────────────

    @abstractmethod
    def search_dense(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
        collection_name: str | None = None,
    ) -> list[RetrievalResult]:
        """Execute dense similarity vector search."""

    @abstractmethod
    def search_hybrid(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 5,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> list[RetrievalResult]:
        """Execute hybrid search combining dense similarity and text/sparse matching."""
