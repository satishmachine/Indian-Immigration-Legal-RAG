"""
ingestion.chunkers.base
=======================
Abstract base class and Protocol for text chunkers.

Every chunker takes a ``Document`` domain entity and returns a list of
``Chunk`` entities with properly populated character offsets, chunk indices,
and inherited document metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.models.document import Chunk, Document


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split *document* into a list of Chunk entities.

        Args:
            document: Source document containing raw content and metadata.

        Returns:
            List of Chunk objects.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
