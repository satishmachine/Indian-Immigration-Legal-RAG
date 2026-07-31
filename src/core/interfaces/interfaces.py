"""
Abstract interfaces (Protocols) for the Legal Assistant.

Using structural subtyping (typing.Protocol) keeps implementations
decoupled from the interface, enabling easy swapping of backends
(e.g., Qdrant → pgvector) without changing callers.

All protocols are runtime-checkable so isinstance() works in tests.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

from core.models.document import Chunk, Document
from core.models.retrieval import RetrievalResult, SearchQuery


# ---------------------------------------------------------------------------
# Document Parser
# ---------------------------------------------------------------------------


@runtime_checkable
class DocumentParser(Protocol):
    """Parse a raw file into a list of Document objects."""

    @abstractmethod
    def parse(self, file_path: str) -> list[Document]:
        """
        Parse the file at *file_path* and return extracted Documents.

        Args:
            file_path: Absolute path to the source file.

        Returns:
            A list of Document objects (one per logical document section).
        """
        ...

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Return True if this parser can handle the given file type."""
        ...


# ---------------------------------------------------------------------------
# Text Chunker
# ---------------------------------------------------------------------------


@runtime_checkable
class TextChunker(Protocol):
    """Split a Document into smaller Chunks suitable for embedding."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split *document* into overlapping text chunks.

        Args:
            document: The source Document to chunk.

        Returns:
            List of Chunk objects with inherited metadata.
        """
        ...


# ---------------------------------------------------------------------------
# Embedding Service
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingService(Protocol):
    """Generate dense vector embeddings for text."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts (for indexing)."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text (optimised for similarity search)."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStore(Protocol):
    """Persist and search dense vector embeddings."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> int:
        """
        Upsert chunks into the vector store.

        Args:
            chunks: Pre-embedded Chunk objects.

        Returns:
            Number of vectors successfully upserted.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Perform dense similarity search.

        Args:
            query_vector: Query embedding vector.
            top_k:        Number of results to return.
            filters:      Optional metadata filters.

        Returns:
            Ranked list of RetrievalResult objects.
        """
        ...

    @abstractmethod
    def delete_collection(self) -> None:
        """Drop the entire collection (use with caution)."""
        ...

    @abstractmethod
    def collection_exists(self) -> bool:
        """Return True if the collection already exists."""
        ...


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


@runtime_checkable
class Retriever(Protocol):
    """High-level retrieval interface used by the RAG chain."""

    @abstractmethod
    def retrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: Structured search query with metadata filters.

        Returns:
            Ranked list of RetrievalResult objects.
        """
        ...


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


@runtime_checkable
class Reranker(Protocol):
    """Re-score and re-rank a list of retrieval results."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Rerank *results* using a cross-encoder model.

        Args:
            query:   Original query string.
            results: Candidate results from the first-stage retriever.
            top_k:   Number of results to return after reranking.

        Returns:
            Top-k results sorted by cross-encoder score (descending).
        """
        ...


# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMService(Protocol):
    """Interface for interacting with an LLM provider."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        context: list[str],
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
        """
        Generate a legal answer given a prompt and retrieved context.

        Args:
            prompt:  The user's question / instruction.
            context: Retrieved document passages as context.
            kwargs:  Provider-specific overrides (temperature, max_tokens).

        Returns:
            Generated text response.
        """
        ...


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@runtime_checkable
class CacheService(Protocol):
    """Key-value cache interface."""

    @abstractmethod
    def get(self, key: str) -> Any | None:  # noqa: ANN401
        """Return cached value or None if not found / expired."""
        ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:  # noqa: ANN401
        """Store a value with an optional TTL in seconds."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a cached value."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        ...
