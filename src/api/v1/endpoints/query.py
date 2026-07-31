"""
api.v1.endpoints.query
======================
RAG Legal Search and QA Endpoint.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status
from api.v1.schemas import QueryRequest, QueryResponse
from chains.legal_rag_chain import LegalRAGChain
from core.models.response import LegalRAGResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Query"])
_rag_chain_instance: LegalRAGChain | None = None


def get_rag_chain() -> LegalRAGChain:
    """Singleton getter for LegalRAGChain."""
    global _rag_chain_instance  # noqa: PLW0603
    if _rag_chain_instance is None:
        _rag_chain_instance = LegalRAGChain()
    return _rag_chain_instance


@router.post("/query", response_model=QueryResponse)
def execute_query(payload: QueryRequest) -> QueryResponse:
    """Execute legal query through the LangChain LCEL RAG pipeline."""
    try:
        chain = get_rag_chain()
        res: LegalRAGResponse = chain.query(
            question=payload.question,
            session_id=payload.session_id,
            filters=payload.filters,
        )

        return QueryResponse(
            answer=res.answer,
            citations=res.citations,
            referenced_sections=res.referenced_sections,
            has_penalty_clause=res.has_penalty_clause,
            confidence_score=res.confidence_score,
            session_id=payload.session_id,
        )
    except Exception as exc:
        logger.error("RAG query endpoint error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process legal query: {exc}",
        ) from exc
