"""
Domain models for the Legal Assistant.

All models use Pydantic v2 for validation and serialisation.
These are pure domain objects – no ORM, no DB-specific fields.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class LegalDomain(StrEnum):
    """Top-level legal domain classifier."""

    CITIZENSHIP = "citizenship"
    IMMIGRATION = "immigration"
    EMIGRATION = "emigration"
    PASSPORT = "passport"
    VISA = "visa"
    FOREIGNERS = "foreigners"
    UNKNOWN = "unknown"


class DocumentType(StrEnum):
    """Type of source legal document."""

    ACT = "act"
    RULE = "rule"
    REGULATION = "regulation"
    AMENDMENT = "amendment"
    NOTIFICATION = "notification"
    CIRCULAR = "circular"
    UNKNOWN = "unknown"


class DocumentStatus(StrEnum):
    """Ingestion status of a document."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Document Metadata
# ---------------------------------------------------------------------------


class DocumentMetadata(BaseModel):
    """
    Rich metadata attached to every ingested legal document.

    Used for metadata-filtered retrieval in Qdrant.
    """

    model_config = {"frozen": True}

    source_file: str = Field(description="Original filename or path of the source PDF.")
    title: str = Field(description="Full official title of the legal instrument.")
    legal_domain: LegalDomain = Field(
        default=LegalDomain.UNKNOWN,
        description="Primary legal domain (citizenship, immigration, etc.).",
    )
    document_type: DocumentType = Field(
        default=DocumentType.UNKNOWN,
        description="Type of legal instrument.",
    )
    year: int | None = Field(
        default=None,
        ge=1947,
        le=2100,
        description="Year of enactment or last amendment.",
    )
    jurisdiction: str = Field(
        default="India",
        description="Applicable jurisdiction.",
    )
    language: str = Field(default="en", description="ISO 639-1 language code.")
    tags: list[str] = Field(
        default_factory=list,
        description="Additional searchable tags.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra metadata (page numbers, section IDs, etc.).",
    )

    @field_validator("source_file")
    @classmethod
    def source_file_not_empty(cls, v: str) -> str:
        """Validate source file is not blank."""
        if not v.strip():
            msg = "source_file must not be empty."
            raise ValueError(msg)
        return v.strip()


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class Document(BaseModel):
    """
    A full legal document as loaded from a source file.

    One Document corresponds to one source PDF / DOCX.
    It is split into Chunks before embedding.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique document ID (UUID4).",
    )
    content: str = Field(description="Raw extracted text content.")
    metadata: DocumentMetadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)

    @property
    def word_count(self) -> int:
        """Return approximate word count."""
        return len(self.content.split())

    def __repr__(self) -> str:
        return (
            f"Document(id={self.id!r}, "
            f"title={self.metadata.title!r}, "
            f"words={self.word_count})"
        )


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """
    A text chunk derived from a Document, ready for embedding and indexing.

    Chunks carry a copy of the parent document's metadata so retrieval
    results can be filtered and displayed without joining back to the
    original document.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique chunk ID (UUID4).",
    )
    document_id: str = Field(description="ID of the parent Document.")
    content: str = Field(description="Chunk text content.")
    metadata: DocumentMetadata
    chunk_index: int = Field(ge=0, description="Position of this chunk in its document.")
    start_char: int = Field(ge=0, description="Character offset of chunk start in document.")
    end_char: int = Field(ge=0, description="Character offset of chunk end in document.")
    embedding: list[float] | None = Field(
        default=None,
        exclude=True,
        description="Dense embedding vector (excluded from serialisation by default).",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("end_char")
    @classmethod
    def end_after_start(cls, v: int, info: Any) -> int:
        """Ensure end_char >= start_char."""
        start = info.data.get("start_char", 0)
        if v < start:
            msg = f"end_char ({v}) must be >= start_char ({start})."
            raise ValueError(msg)
        return v

    @property
    def char_length(self) -> int:
        """Return length in characters."""
        return self.end_char - self.start_char

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.id!r}, "
            f"doc_id={self.document_id!r}, "
            f"index={self.chunk_index}, "
            f"chars={self.char_length})"
        )
