"""
api.v1.schemas
==============
Pydantic API Request/Response Schemas for FastAPI Endpoints.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from core.models.response import LegalCitation


class QueryRequest(BaseModel):
    """RAG Search and QA query payload."""

    question: str = Field(min_length=3, description="User statutory legal query")
    session_id: str = Field(default="default_session", description="Conversation session identifier")
    filters: dict[str, Any] | None = Field(default=None, description="Metadata filtering dictionary")


class QueryResponse(BaseModel):
    """Structured RAG response payload."""

    answer: str = Field(description="Generated statutory answer")
    citations: list[LegalCitation] = Field(default_factory=list, description="Statutory citations")
    referenced_sections: list[str] = Field(default_factory=list, description="Referenced statutory section numbers")
    has_penalty_clause: bool = Field(default=False, description="Penalty clause detection")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Response confidence score")
    session_id: str = Field(description="Conversation session identifier")


class IndexStatsResponse(BaseModel):
    """Vector database collection statistics response."""

    collection_name: str
    points_count: int
    vectors_count: int
    status: str


class HealthCheckResponse(BaseModel):
    """Service healthcheck status response."""

    status: str = "ok"
    version: str = "1.0.0"
    qdrant_connected: bool = True
