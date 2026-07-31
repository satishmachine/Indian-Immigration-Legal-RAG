"""
retrieval.compression_retriever
===============================
LangChain LCEL Legal Compression Retriever.

Inherits from `langchain_core.retrievers.BaseRetriever`.
Combines LegalHybridRetriever (Dense + BM25 RRF) with BGEReranker (BAAI/bge-reranker-v2-m3)
to perform two-stage retrieval and passage compression.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from core.models.document import DocumentMetadata, LegalDomain
from core.models.retrieval import RetrievalResult
from retrieval.hybrid.legal_hybrid_retriever import LegalHybridRetriever
from retrieval.reranker.api_reranker import ApiReranker
from retrieval.reranker.factory import RerankerFactory

logger = logging.getLogger(__name__)


class LegalCompressionRetriever(BaseRetriever):
    """
    Two-Stage Legal Compression Retriever implementing LangChain `BaseRetriever`.

    Stage 1: Hybrid Search (Dense Qdrant + Sparse BM25 RRF Fusion) -> Fetches candidate pool.
    Stage 2: API Reranking -> Re-ranks and compresses to top N passages.
    """

    base_retriever: Any = Field(default_factory=LegalHybridRetriever)
    reranker: Any = Field(default_factory=RerankerFactory.create)
    top_n: int = Field(default=5, ge=1)

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
        Execute 2-stage retrieval: Hybrid search -> BGE Reranking compression.

        Args:
            query: User query string.
            run_manager: LangChain callback manager.
            kwargs: Optional dynamic parameters.

        Returns:
            List of re-ranked, compressed LangChain Document objects.
        """
        active_top_n = kwargs.get("top_n") or self.top_n
        filters = kwargs.get("filters")

        logger.info("Executing LegalCompressionRetriever query=%r (top_n=%d)", query[:50], active_top_n)

        # Stage 1: Fetch candidate pool via Hybrid Retriever
        candidates_lc: list[LCDocument] = self.base_retriever.invoke(
            query,
            filters=filters,
            top_k=active_top_n * 3,
        )

        if not candidates_lc:
            logger.warning("No candidate documents returned from base hybrid retriever")
            return []

        # Convert LCDocument to RetrievalResult domain models for reranker
        candidate_results: list[RetrievalResult] = []
        for rank, lc_doc in enumerate(candidates_lc):
            meta_dict = lc_doc.metadata or {}
            domain_val = meta_dict.get("legal_domain", "unknown")
            doc_meta = DocumentMetadata(
                source_file=meta_dict.get("source", "unknown"),
                title=meta_dict.get("title", "Untitled"),
                legal_domain=LegalDomain(domain_val) if domain_val in LegalDomain._value2member_map_ else LegalDomain.UNKNOWN,
                year=meta_dict.get("year"),
                jurisdiction=meta_dict.get("jurisdiction", "India"),
                extra={k: v for k, v in meta_dict.items() if k not in ("source", "title", "legal_domain", "year", "jurisdiction", "chunk_id", "document_id", "score", "rank", "retrieval_method")},
            )

            candidate_results.append(
                RetrievalResult(
                    chunk_id=meta_dict.get("chunk_id", f"chk_{rank}"),
                    document_id=meta_dict.get("document_id", f"doc_{rank}"),
                    content=lc_doc.page_content,
                    score=float(meta_dict.get("score", 0.0)),
                    metadata=doc_meta,
                    rank=rank,
                    retrieval_method=meta_dict.get("retrieval_method", "hybrid"),
                )
            )

        # Stage 2: Execute BGE Cross-Encoder Reranking
        reranked_results = self.reranker.rerank(
            query=query,
            results=candidate_results,
            top_n=active_top_n,
        )

        # Convert back to standard LangChain Documents
        compressed_docs: list[LCDocument] = []
        for res in reranked_results:
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
            lc_metadata.update(res.metadata.extra)

            compressed_docs.append(
                LCDocument(
                    page_content=res.content,
                    metadata=lc_metadata,
                )
            )

        logger.info("LegalCompressionRetriever returning top %d compressed passages", len(compressed_docs))
        return compressed_docs
