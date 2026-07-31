"""
api.v1.endpoints.stats
======================
Vector Database Statistics Endpoint.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status
from api.v1.schemas import IndexStatsResponse
from services.vector_store import get_vector_repository

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Statistics"])


@router.get("/stats", response_model=IndexStatsResponse)
def get_index_statistics() -> IndexStatsResponse:
    """Get Qdrant collection statistics and indexed vectors count."""
    try:
        repo = get_vector_repository()
        info = repo.get_collection_info()
        return IndexStatsResponse(
            collection_name=info.get("collection_name", "legal_documents"),
            points_count=info.get("points_count", 0),
            vectors_count=info.get("vectors_count", 0),
            status=info.get("status", "unknown"),
        )
    except Exception as exc:
        logger.error("Failed to retrieve index stats: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch vector database statistics: {exc}",
        ) from exc
