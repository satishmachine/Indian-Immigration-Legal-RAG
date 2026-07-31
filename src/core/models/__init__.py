"""
src.core.models
===============
Domain model package — all Pydantic v2 domain entities.

Public API
----------
    from src.core.models import Document, Chunk, DocumentMetadata
    from src.core.models import LegalSectionMetadata, LegalDocumentMetadata
    from src.core.models import PenaltyInfo, AuthorityReference
    from src.core.models import DefinitionEntry, CrossReference
    from src.core.models import SearchQuery, RetrievalResult, RAGResponse
"""

from core.models.document import (
    Chunk,
    Document,
    DocumentMetadata,
    DocumentStatus,
    DocumentType,
    LegalDomain,
)
from core.models.metadata import (
    AuthorityReference,
    AuthorityType,
    CrossReference,
    DefinitionEntry,
    LegalDocumentMetadata,
    LegalSectionMetadata,
    PenaltyInfo,
    PenaltyType,
)
from core.models.retrieval import (
    MetadataFilter,
    RAGResponse,
    RetrievalResult,
    SearchQuery,
)

__all__: list[str] = [
    # document.py
    "Chunk",
    "Document",
    "DocumentMetadata",
    "DocumentStatus",
    "DocumentType",
    "LegalDomain",
    # metadata.py
    "AuthorityReference",
    "AuthorityType",
    "CrossReference",
    "DefinitionEntry",
    "LegalDocumentMetadata",
    "LegalSectionMetadata",
    "PenaltyInfo",
    "PenaltyType",
    # retrieval.py
    "MetadataFilter",
    "RAGResponse",
    "RetrievalResult",
    "SearchQuery",
]
