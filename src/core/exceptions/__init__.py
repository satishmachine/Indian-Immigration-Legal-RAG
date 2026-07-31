"""
Exception hierarchy entrypoint.

Imports:
    from core.exceptions import DocumentParseError, NoResultsFoundError, ...
"""

from core.exceptions.exceptions import (
    AllRetrieversFailedError,
    AuthenticationError,
    AuthorizationError,
    BaseAppError,
    ChunkingError,
    CollectionNotFoundError,
    ConfigurationError,
    DenseRetrievalError,
    DocumentParseError,
    DuplicateDocumentError,
    EmbeddingError,
    EmbeddingQuotaExceededError,
    IngestionError,
    LLMContextLengthError,
    LLMError,
    LLMRateLimitError,
    NoResultsFoundError,
    RerankerError,
    RetrievalError,
    SparseRetrievalError,
    ValidationError,
    VectorStoreError,
    VectorStoreUpsertError,
)

__all__ = [
    "BaseAppError",
    "ConfigurationError",
    "IngestionError",
    "DocumentParseError",
    "ChunkingError",
    "DuplicateDocumentError",
    "VectorStoreError",
    "CollectionNotFoundError",
    "VectorStoreUpsertError",
    "RetrievalError",
    "NoResultsFoundError",
    "RerankerError",
    "DenseRetrievalError",
    "EmbeddingQuotaExceededError",
    "SparseRetrievalError",
    "AllRetrieversFailedError",
    "LLMError",
    "LLMRateLimitError",
    "LLMContextLengthError",
    "EmbeddingError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
]
