"""
Domain models for retrieval queries and results.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.models.document import DocumentMetadata, LegalDomain


# ---------------------------------------------------------------------------
# Search Query
# ---------------------------------------------------------------------------


class MetadataFilter(BaseModel):
    """A single metadata filter condition."""

    field: str = Field(description="Metadata field name to filter on.")
    value: Any = Field(description="Expected value.")
    operator: str = Field(
        default="eq",
        description="Comparison operator: eq | in | gte | lte | ne",
    )


class SearchQuery(BaseModel):
    """
    A structured retrieval query with optional metadata filters.

    Passed to Retriever.retrieve() by the RAG chain.
    """

    query_text: str = Field(description="The user's natural-language query.")
    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    filters: list[MetadataFilter] = Field(
        default_factory=list,
        description="Optional metadata filters (e.g. domain=citizenship).",
    )
    legal_domains: list[LegalDomain] = Field(
        default_factory=list,
        description="Restrict retrieval to specific legal domains.",
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include document metadata in results.",
    )


# ---------------------------------------------------------------------------
# Retrieval Result
# ---------------------------------------------------------------------------


class RetrievalResult(BaseModel):
    """A single retrieved chunk with its relevance score."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_id: str = Field(description="ID of the retrieved Chunk.")
    document_id: str = Field(description="ID of the parent Document.")
    content: str = Field(description="Chunk text content.")
    score: float = Field(description="Relevance score (0–1, higher is better).")
    metadata: DocumentMetadata | dict[str, Any] | Any
    rank: int = Field(default=0, ge=0, description="Rank position (0-indexed).")
    retrieval_method: str = Field(
        default="dense",
        description="How this result was retrieved: dense | sparse | hybrid",
    )

    @property
    def citation(self) -> str:
        """Return a formatted citation string for UI display."""
        if hasattr(self.metadata, "title"):
            year = f" ({self.metadata.year})" if getattr(self.metadata, "year", None) else ""
            title = getattr(self.metadata, "title", "Untitled")
            source = getattr(self.metadata, "source_file", "Unknown")
        elif isinstance(self.metadata, dict):
            year = f" ({self.metadata.get('year')})" if self.metadata.get('year') else ""
            title = self.metadata.get('title', 'Untitled')
            source = self.metadata.get('source_file', 'Unknown')
        else:
            year, title, source = "", "Untitled", "Unknown"
        return f"{title}{year} — {source}"

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(rank={self.rank}, "
            f"score={self.score:.4f}, "
            f"doc={self.metadata.title!r})"
        )


# ---------------------------------------------------------------------------
# RAG Response
# ---------------------------------------------------------------------------


class RAGResponse(BaseModel):
    """Complete response from the RAG pipeline."""

    answer: str = Field(description="Generated legal answer.")
    sources: list[RetrievalResult] = Field(
        description="Source chunks used to generate the answer."
    )
    query: str = Field(description="Original user query.")
    model_used: str = Field(description="LLM model that generated the answer.")
    tokens_used: int | None = Field(
        default=None,
        description="Total tokens consumed (prompt + completion).",
    )
    latency_ms: float | None = Field(
        default=None,
        description="End-to-end pipeline latency in milliseconds.",
    )
    cached: bool = Field(
        default=False,
        description="Whether the response was served from cache.",
    )
