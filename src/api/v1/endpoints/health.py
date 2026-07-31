"""
api.v1.endpoints.health
=======================
Health Check Endpoint for Docker and K8s Liveness/Readiness Probes.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter
from api.v1.schemas import HealthCheckResponse
from services.vector_store import get_vector_repository

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthCheckResponse)
@router.get("/", response_model=HealthCheckResponse)
def health_check() -> HealthCheckResponse:
    """Execute healthcheck probe on system dependencies."""
    qdrant_connected = False
    try:
        repo = get_vector_repository()
        stats = repo.get_collection_info()
        qdrant_connected = stats.get("status") != "not_found"
    except Exception as exc:
        logger.warning("Healthcheck Qdrant probe failed: %s", exc)

    return HealthCheckResponse(
        status="ok" if qdrant_connected else "degraded",
        version="1.0.0",
        qdrant_connected=qdrant_connected,
    )
