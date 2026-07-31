"""
Custom exception hierarchy for the Indian Immigration Legal Assistant.

All application-specific exceptions inherit from BaseAppError, which
carries a machine-readable error_code and an HTTP status hint so API
layers can map them to responses without catch-all blocks.

Design Principles:
- Never raise generic Exception or RuntimeError in application code.
- Catch narrowly; handle at the boundary layer (API / CLI).
- All exceptions are serialisable (only primitive fields).
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseAppError(Exception):
    """
    Root exception for the Legal Assistant application.

    Attributes:
        message:     Human-readable error description.
        error_code:  Machine-readable slug (e.g. 'document_not_found').
        status_code: HTTP status hint for API error mapping.
        details:     Arbitrary extra context (logged but not sent to client).
    """

    def __init__(
        self,
        message: str,
        error_code: str = "internal_error",
        status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details: dict[str, Any] = details or {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"error_code={self.error_code!r}, "
            f"message={self.message!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dict suitable for JSON error responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
        }


# ---------------------------------------------------------------------------
# Configuration Errors
# ---------------------------------------------------------------------------


class ConfigurationError(BaseAppError):
    """Raised when required configuration is missing or invalid."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="configuration_error",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            details=details,
        )


# ---------------------------------------------------------------------------
# Document Ingestion Errors
# ---------------------------------------------------------------------------


class IngestionError(BaseAppError):
    """Base class for all document ingestion errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "ingestion_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            details=details,
        )


class DocumentParseError(IngestionError):
    """Raised when a document cannot be parsed (corrupt PDF, unsupported format)."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            message=f"Failed to parse document '{path}': {reason}",
            error_code="document_parse_error",
            details={"path": path, "reason": reason},
        )


class ChunkingError(IngestionError):
    """Raised when text chunking fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, error_code="chunking_error")


class DuplicateDocumentError(IngestionError):
    """Raised when attempting to ingest a document that already exists."""

    def __init__(self, doc_id: str) -> None:
        super().__init__(
            message=f"Document '{doc_id}' has already been ingested.",
            error_code="duplicate_document",
            details={"doc_id": doc_id},
        )


# ---------------------------------------------------------------------------
# Vector Store Errors
# ---------------------------------------------------------------------------


class VectorStoreError(BaseAppError):
    """Base class for Qdrant / vector store errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "vector_store_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            details=details,
        )


class CollectionNotFoundError(VectorStoreError):
    """Raised when a Qdrant collection does not exist."""

    def __init__(self, collection: str) -> None:
        super().__init__(
            message=f"Qdrant collection '{collection}' does not exist.",
            error_code="collection_not_found",
            details={"collection": collection},
        )


class VectorStoreUpsertError(VectorStoreError):
    """Raised when upserting vectors to Qdrant fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, error_code="vector_upsert_error")


# ---------------------------------------------------------------------------
# Retrieval Errors
# ---------------------------------------------------------------------------


class RetrievalError(BaseAppError):
    """Base class for retrieval pipeline errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "retrieval_error",
        status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class NoResultsFoundError(RetrievalError):
    """Raised when a query returns zero results above the score threshold."""

    def __init__(self, query: str, threshold: float) -> None:
        super().__init__(
            message=(
                f"No relevant legal documents found for query "
                f"(threshold={threshold}): {query!r}"
            ),
            error_code="no_results_found",
            details={"query": query, "threshold": threshold},
        )


class RerankerError(RetrievalError):
    """Raised when the cross-encoder reranker fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, error_code="reranker_error")


class DenseRetrievalError(RetrievalError):
    """Raised when dense vector retrieval fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="dense_retrieval_error",
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            details=details,
        )


class EmbeddingQuotaExceededError(DenseRetrievalError):
    """Raised when the embedding API returns HTTP 403 or quota exceeded."""

    def __init__(self, message: str = "Embedding API daily quota/token limit exceeded.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            details=details or {},
        )
        self.error_code = "embedding_quota_exceeded"
        self.status_code = HTTPStatus.TOO_MANY_REQUESTS


class SparseRetrievalError(RetrievalError):
    """Raised when sparse BM25 keyword retrieval fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="sparse_retrieval_error",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            details=details,
        )


class AllRetrieversFailedError(RetrievalError):
    """Raised when all retrieval channels (Dense and Sparse) fail."""

    def __init__(self, message: str = "All document retrieval methods failed.", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_code="all_retrievers_failed",
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            details=details,
        )


# ---------------------------------------------------------------------------
# LLM / Generation Errors
# ---------------------------------------------------------------------------


class LLMError(BaseAppError):
    """Base class for LLM provider errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "llm_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=HTTPStatus.BAD_GATEWAY,
            details=details,
        )


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider rate-limits the request."""

    def __init__(self, provider: str, retry_after: int | None = None) -> None:
        super().__init__(
            message=f"Rate limit exceeded for provider '{provider}'.",
            error_code="llm_rate_limit",
            details={"provider": provider, "retry_after": retry_after},
        )


class LLMContextLengthError(LLMError):
    """Raised when the prompt exceeds the model's context window."""

    def __init__(self, token_count: int, max_tokens: int) -> None:
        super().__init__(
            message=(
                f"Prompt is too long ({token_count} tokens); "
                f"model maximum is {max_tokens} tokens."
            ),
            error_code="context_length_exceeded",
            details={"token_count": token_count, "max_tokens": max_tokens},
        )


# ---------------------------------------------------------------------------
# Embedding Errors
# ---------------------------------------------------------------------------


class EmbeddingError(BaseAppError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            error_code="embedding_error",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


# ---------------------------------------------------------------------------
# Validation Errors
# ---------------------------------------------------------------------------


class ValidationError(BaseAppError):
    """Raised for invalid input that passes Pydantic but fails domain rules."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(
            message=f"Validation error on '{field}': {message}",
            error_code="validation_error",
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            details={"field": field},
        )


# ---------------------------------------------------------------------------
# Auth Errors
# ---------------------------------------------------------------------------


class AuthenticationError(BaseAppError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Invalid credentials.") -> None:
        super().__init__(
            message=message,
            error_code="authentication_error",
            status_code=HTTPStatus.UNAUTHORIZED,
        )


class AuthorizationError(BaseAppError):
    """Raised when a user does not have permission for an action."""

    def __init__(self, action: str) -> None:
        super().__init__(
            message=f"You are not authorised to perform '{action}'.",
            error_code="authorization_error",
            status_code=HTTPStatus.FORBIDDEN,
            details={"action": action},
        )
