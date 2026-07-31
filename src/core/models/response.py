"""
core.models.response
====================
Pydantic Data Models for Grounded Legal RAG Responses and Citations.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LegalCitation(BaseModel):
    """Statutory Legal Citation."""

    act_name: str = Field(description="Name of the Act or Rule")
    year: int | None = Field(default=None, description="Year of enactment")
    section_number: str = Field(description="Section or Rule number")
    section_title: str | None = Field(default=None, description="Section title or heading")
    page_number: int | None = Field(default=None, description="Source PDF page number")
    pdf_name: str | None = Field(default=None, description="Source PDF file name")
    citation_text: str = Field(description="Standard Indian legal citation string")
    snippet: str = Field(description="Direct verbatim quote from statutory text")
    score: float = Field(default=0.0, description="Reranker / retrieval relevance score")


class LegalRAGResponse(BaseModel):
    """Complete Grounded Legal RAG Answer."""

    question: str = Field(description="Original user question")
    answer: str = Field(description="Detailed legal analysis and answer text")
    citations: list[LegalCitation] = Field(default_factory=list, description="Extracted statutory citations")
    referenced_sections: list[str] = Field(default_factory=list, description="List of referenced section numbers (e.g. ['Section 5', 'Section 21'])")
    has_penalty_clause: bool = Field(default=False, description="Whether answer involves penal or criminal consequences")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score based on grounding")
    retrieval_status: str = Field(default="success", description="Retrieval status: 'success' | 'no_documents_found' | 'retrieval_failed'")
    error_title: str | None = Field(default=None, description="User-facing error title for Scenario 3")
    error_message: str | None = Field(default=None, description="User-facing system message for Scenario 3")
    developer_details: str | None = Field(default=None, description="Raw exception details/traceback for developers")
