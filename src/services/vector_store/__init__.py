"""
services.vector_store
=====================
Vector store repository factory and implementations.
"""

from __future__ import annotations

from core.config import get_settings
from core.interfaces.interfaces import VectorStore
from core.interfaces.vector_store_repository import VectorStoreRepository
from services.vector_store.qdrant_repository import QdrantVectorRepository
from services.vector_store.qdrant_service import QdrantVectorStore

__all__: list[str] = [
    "QdrantVectorRepository",
    "QdrantVectorStore",
    "VectorStore",
    "VectorStoreRepository",
    "get_vector_repository",
    "get_vector_store",
]

_vector_repository_instance: VectorStoreRepository | None = None
_vector_store_instance: VectorStore | None = None


def get_vector_repository() -> VectorStoreRepository:
    """
    Factory function returning the configured VectorStoreRepository instance.
    """
    global _vector_repository_instance  # noqa: PLW0603
    if _vector_repository_instance is None:
        cfg = get_settings().qdrant
        _vector_repository_instance = QdrantVectorRepository(
            host=cfg.host,
            port=cfg.port,
            default_collection=cfg.collection_name,
            default_vector_size=cfg.vector_size,
        )
    return _vector_repository_instance


def get_vector_store() -> VectorStore:
    """
    Factory function returning the configured VectorStore instance.
    """
    global _vector_store_instance  # noqa: PLW0603
    if _vector_store_instance is None:
        cfg = get_settings().qdrant
        _vector_store_instance = QdrantVectorStore(
            host=cfg.host,
            port=cfg.port,
            collection_name=cfg.collection_name,
            vector_size=cfg.vector_size,
        )
    return _vector_store_instance
