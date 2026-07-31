"""
services.embedding.base
=======================
Abstract Base Class for all Embedding Providers.

Defines the Clean Architecture contract for dense vector embedding generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from core.interfaces.interfaces import EmbeddingService


class BaseEmbeddingService(EmbeddingService, ABC):
    """
    Abstract base class for all embedding providers.
    Every provider subclass must implement embed_documents, embed_query, and dimension.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique provider name identifier (e.g. 'bge', 'openai', 'voyage', 'jina', 'ollama')."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active model name identifier (e.g. 'BAAI/bge-m3')."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider_name!r}, model={self.model_name!r}, dim={self.dimension})"
