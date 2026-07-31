"""
core.interfaces.reranker_interface
===================================
Abstract Reranker Service Domain Interface.

Defines the Clean Architecture contract for passage re-ranking models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from core.models.retrieval import RetrievalResult


class RerankerService(ABC):
    """Abstract Base Class for Reranking Engines."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Re-rank candidate RetrievalResult items using cross-attention scoring.

        Args:
            query: User query string.
            results: List of candidate RetrievalResult objects from hybrid retrieval.
            top_n: Optional cut-off for top N re-ranked results.

        Returns:
            Re-ordered list of RetrievalResult objects with updated cross-encoder scores.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active reranker model identifier."""
