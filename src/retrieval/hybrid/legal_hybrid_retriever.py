"""
retrieval.hybrid.legal_hybrid_retriever
=======================================
LangChain LCEL Legal Hybrid Retriever.

Inherits from `langchain_core.retrievers.BaseRetriever` to seamlessly integrate
into LangChain LCEL pipelines (e.g. `retriever | prompt | llm`).

Features:
- Dense Vector Retrieval (Qdrant) + Sparse Keyword Retrieval (BM25)
- RRF (Reciprocal Rank Fusion) and Weighted Score Fusion
- Metadata Filtering (by act_name, year, section_number, legal_domain, etc.)
- Maximal Marginal Relevance (MMR) for result diversity selection
- Returns standard LangChain `Document` objects with statutory metadata
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from core.models.document import Chunk
from core.models.retrieval import RetrievalResult
from retrieval.dense.dense_retriever import DenseRetriever
from retrieval.hybrid.fusion import reciprocal_rank_fusion, weighted_score_fusion
from retrieval.hybrid.mmr import maximal_marginal_relevance
from retrieval.sparse.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


class LegalHybridRetriever(BaseRetriever):
    """
    Production Legal Hybrid Retriever implementing LangChain `BaseRetriever`.

    Merges Dense (Qdrant) and Sparse (BM25) retrieval results, applies
    metadata filtering and MMR diversity selection, returning standard LangChain Documents.

    Fields:
        dense_retriever: DenseRetriever instance.
        bm25_retriever: BM25Retriever instance.
        top_k: Number of final top results to return.
        alpha: Dense vs Sparse weight ratio for weighted fusion [0, 1].
        fusion_mode: Fusion algorithm ('rrf' or 'weighted').
        use_mmr: Whether to apply MMR for result diversity.
        mmr_lambda: MMR diversity trade-off multiplier [0, 1].
        filters: Global metadata filter dictionary.
    """

    dense_retriever: Any = Field(default_factory=DenseRetriever)
    bm25_retriever: Any = Field(default_factory=BM25Retriever)
    top_k: int = Field(default=5, ge=1)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    fusion_mode: Literal["rrf", "weighted"] = Field(default="rrf")
    use_mmr: bool = Field(default=True)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    filters: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
        **kwargs: Any,
    ) -> list[LCDocument]:
        """
        Execute Hybrid Retrieval pipeline and return LangChain Document entities.

        Args:
            query: User search query string.
            run_manager: LangChain callback manager.
            kwargs: Optional dynamic parameters (e.g. override filters or top_k).

        Returns:
            List of LangChain Document objects.
        """
        active_filters = kwargs.get("filters") or self.filters
        active_top_k = kwargs.get("top_k") or self.top_k
        fetch_k = active_top_k * 3  # Over-fetch for fusion and MMR diversity

        logger.info("Executing LegalHybridRetriever query=%r (top_k=%d, fusion=%s, mmr=%s)", query, active_top_k, self.fusion_mode, self.use_mmr)

        # 1. Execute Dense Vector Retrieval
        dense_results: list[RetrievalResult] = []
        dense_error: Exception | None = None
        try:
            dense_results = self.dense_retriever.retrieve(
                query=query,
                top_k=fetch_k,
                filters=active_filters,
            )
        except Exception as exc:
            dense_error = exc
            logger.warning("Dense vector retrieval failed (%s). Attempting BM25 fallback.", exc)

        # 2. Execute BM25 Sparse Keyword Retrieval
        sparse_results: list[RetrievalResult] = []
        sparse_error: Exception | None = None
        try:
            sparse_results = self.bm25_retriever.retrieve(
                query=query,
                top_k=fetch_k,
                filters=active_filters,
            )
        except Exception as exc:
            sparse_error = exc
            logger.warning("BM25 sparse keyword retrieval failed (%s).", exc, exc_info=True)

        # Scenario 3 Trigger: If BOTH Dense AND Sparse retrieval failed
        if dense_error is not None and sparse_error is not None:
            logger.error("All retrieval channels failed for query %r", query[:50], exc_info=True)
            from core.exceptions import AllRetrieversFailedError
            raise AllRetrieversFailedError(
                message=f"All document retrieval channels failed. Dense error: {dense_error}. Sparse error: {sparse_error}",
                details={
                    "dense_error": str(dense_error),
                    "sparse_error": str(sparse_error),
                    "query": query,
                },
            ) from dense_error

        # 3. Score Fusion & Result Merging
        fused_candidates: list[RetrievalResult] = []
        if self.fusion_mode == "rrf":
            result_lists = [r for r in [dense_results, sparse_results] if r]
            if result_lists:
                fused_candidates = reciprocal_rank_fusion(
                    result_lists=result_lists,
                    rrf_k=60,
                    top_k=fetch_k,
                )
        else:
            fused_candidates = weighted_score_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                alpha=self.alpha,
                top_k=fetch_k,
            )

        # Fallback if fusion yields no results
        if not fused_candidates:
            fused_candidates = dense_results or sparse_results

        # 4. Maximal Marginal Relevance (MMR) for Diversity Selection
        final_results: list[RetrievalResult] = []
        if self.use_mmr and len(fused_candidates) > active_top_k:
            final_results = maximal_marginal_relevance(
                candidates=fused_candidates,
                top_k=active_top_k,
                lambda_mult=self.mmr_lambda,
            )
        else:
            final_results = fused_candidates[:active_top_k]

        # 5. Convert RetrievalResults to standard LangChain Document entities
        lc_documents: list[LCDocument] = []
        for res in final_results:
            # Build metadata payload dict for LangChain Document
            lc_metadata = {
                "chunk_id": res.chunk_id,
                "document_id": res.document_id,
                "score": res.score,
                "rank": res.rank,
                "retrieval_method": res.retrieval_method,
                "source": res.metadata.source_file,
                "title": res.metadata.title,
                "legal_domain": res.metadata.legal_domain.value if hasattr(res.metadata.legal_domain, "value") else str(res.metadata.legal_domain),
                "year": res.metadata.year,
                "jurisdiction": res.metadata.jurisdiction,
            }
            # Flatten extra section metadata
            lc_metadata.update(res.metadata.extra)

            lc_documents.append(
                LCDocument(
                    page_content=res.content,
                    metadata=lc_metadata,
                )
            )

        logger.info("LegalHybridRetriever returning %d LangChain Documents", len(lc_documents))
        return lc_documents

    def index_chunks_for_bm25(self, chunks: list[Chunk]) -> None:
        """Index a list of Chunk entities into the BM25 sparse retriever."""
        if hasattr(self.bm25_retriever, "index"):
            self.bm25_retriever.index(chunks)
            logger.info("Indexed %d chunks in BM25Retriever", len(chunks))
