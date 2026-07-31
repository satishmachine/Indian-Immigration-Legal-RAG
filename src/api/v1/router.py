"""
api.v1.router
=============
Main v1 API Router mounting endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter
from api.v1.endpoints import health, ingestion, query, stats

api_v1_router = APIRouter(prefix="/v1")
api_v1_router.include_router(health.router)
api_v1_router.include_router(query.router)
api_v1_router.include_router(ingestion.router)
api_v1_router.include_router(stats.router)
