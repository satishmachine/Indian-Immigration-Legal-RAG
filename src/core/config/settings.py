"""
Indian Immigration & Emigration Legal Assistant
================================================
Settings — Single Source of Truth for All Configuration.

Design decisions
----------------
* **Pydantic-Settings v2** — every value is validated, typed, and documented.
* **Nested BaseSettings groups** — each domain owns its own prefix / env vars.
* **No hardcoded secrets** — every sensitive field is ``SecretStr`` and must be
  provided via ``.env`` or the environment.
* **Computed properties** — URL builders, feature flags, and path helpers live
  here so callers never concatenate strings themselves.
* **Singleton with reset** — ``get_settings()`` returns a cached instance;
  ``reset_settings()`` is provided exclusively for test isolation.

Usage
-----
    from src.core.config import settings          # convenience singleton
    from src.core.config import get_settings      # factory (testable)

    api_key = settings.openai.api_key.get_secret_value()
    qdrant_url = settings.qdrant.url
"""

from __future__ import annotations

import sys
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    AnyHttpUrl,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Project root — used to build absolute directory defaults
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AppEnvironment(StrEnum):
    """Deployment target environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    """Standard Python logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(StrEnum):
    """Log output format."""

    CONSOLE = "console"  # Human-readable rich output (development)
    JSON = "json"        # Structured JSON for log aggregators (production)


class EmbeddingProvider(StrEnum):
    """Supported embedding model providers."""

    BGE = "bge"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    VOYAGE = "voyage"
    JINA = "jina"
    OLLAMA = "ollama"
    GOOGLE = "google"
    COHERE = "cohere"


class LLMProvider(StrEnum):
    """Active LLM provider for generation."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    OLLAMA = "ollama"


class CacheBackend(StrEnum):
    """Caching backend strategy."""

    DISK = "disk"
    REDIS = "redis"
    NONE = "none"


class StreamlitTheme(StrEnum):
    """Streamlit UI theme."""

    DARK = "dark"
    LIGHT = "light"


# ---------------------------------------------------------------------------
# Base class for nested settings groups — ensures .env is always loaded
# ---------------------------------------------------------------------------


class _BaseGroupSettings(BaseSettings):
    """Base class for nested settings groups with automatic .env loading."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class AppSettings(_BaseGroupSettings):
    """Core application metadata and feature toggles."""

    name: str = Field(
        default="Indian Immigration Legal Assistant",
        alias="APP_NAME",
        description="Human-readable application name.",
    )
    env: AppEnvironment = Field(
        default=AppEnvironment.DEVELOPMENT,
        alias="APP_ENV",
        description="Deployment environment (development | staging | production).",
    )
    version: str = Field(
        default="0.1.0",
        alias="APP_VERSION",
        description="Semantic version string.",
    )
    debug: bool = Field(
        default=False,
        alias="DEBUG",
        description="Enable verbose debug mode. Must be False in production.",
    )


# ---------------------------------------------------------------------------


class LoggingSettings(BaseSettings):
    """Structured logging configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    level: LogLevel = Field(
        default=LogLevel.INFO,
        alias="LOG_LEVEL",
        description="Minimum log level to emit.",
    )
    format: LogFormat = Field(
        default=LogFormat.CONSOLE,
        alias="LOG_FORMAT",
        description="'console' for dev; 'json' for production log aggregators.",
    )
    log_dir: Path = Field(
        default=_PROJECT_ROOT / "logs",
        alias="LOG_DIR",
        description="Directory where rotating log files are written.",
    )
    max_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        alias="LOG_MAX_BYTES",
        description="Maximum size in bytes before a log file is rotated.",
    )
    backup_count: int = Field(
        default=5,
        alias="LOG_BACKUP_COUNT",
        description="Number of rotated log files to retain.",
    )
    enable_sql_echo: bool = Field(
        default=False,
        alias="LOG_SQL_ECHO",
        description="Echo SQLAlchemy queries to the log (development only).",
    )

    @property
    def app_log_path(self) -> Path:
        """Absolute path to the main application log file."""
        return self.log_dir / "app.log"

    @property
    def error_log_path(self) -> Path:
        """Absolute path to the error-only log file."""
        return self.log_dir / "errors.log"


# ---------------------------------------------------------------------------


class APISettings(BaseSettings):
    """FastAPI server settings."""

    model_config = SettingsConfigDict(extra="ignore")

    host: str = Field(
        default="0.0.0.0",
        alias="API_HOST",
        description="Host address for the FastAPI server.",
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        alias="API_PORT",
        description="TCP port for the FastAPI server.",
    )
    prefix: str = Field(
        default="/api/v1",
        alias="API_PREFIX",
        description="Global URL prefix for all API routes.",
    )
    workers: int = Field(
        default=1,
        ge=1,
        alias="API_WORKERS",
        description="Uvicorn worker count. Set to CPU count in production.",
    )
    reload: bool = Field(
        default=True,
        alias="API_RELOAD",
        description="Hot-reload on source changes. Must be False in production.",
    )
    cors_origins: list[AnyHttpUrl] = Field(
        default=["http://localhost:8501"],
        alias="API_CORS_ORIGINS",
        description="Allowed CORS origins. JSON list in env.",
    )
    secret_key: SecretStr = Field(
        alias="API_SECRET_KEY",
        description="HMAC signing key. Must be >=32 random characters.",
    )
    request_timeout_seconds: int = Field(
        default=120,
        alias="API_REQUEST_TIMEOUT_SECONDS",
        description="Maximum request duration before 504 is returned.",
    )


# ---------------------------------------------------------------------------


class AuthSettings(BaseSettings):
    """JWT authentication settings."""

    model_config = SettingsConfigDict(extra="ignore")

    jwt_secret_key: SecretStr = Field(
        alias="JWT_SECRET_KEY",
        description="Secret used to sign JWT tokens. Must be >=32 random characters.",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        alias="JWT_ALGORITHM",
        description="HMAC algorithm for JWT signing.",
    )
    access_token_expire_minutes: int = Field(
        default=30,
        ge=1,
        alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
        description="Access token TTL in minutes.",
    )
    refresh_token_expire_days: int = Field(
        default=7,
        ge=1,
        alias="JWT_REFRESH_TOKEN_EXPIRE_DAYS",
        description="Refresh token TTL in days.",
    )


# ---------------------------------------------------------------------------


class RateLimitSettings(BaseSettings):
    """API rate-limiting settings."""

    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(
        default=True,
        alias="RATE_LIMIT_ENABLED",
        description="Enable or disable rate limiting globally.",
    )
    requests: int = Field(
        default=60,
        ge=1,
        alias="RATE_LIMIT_REQUESTS",
        description="Maximum requests allowed per time window.",
    )
    window_seconds: int = Field(
        default=60,
        ge=1,
        alias="RATE_LIMIT_WINDOW_SECONDS",
        description="Length of the sliding time window in seconds.",
    )


# ---------------------------------------------------------------------------
# LLM Providers
# ---------------------------------------------------------------------------


class OpenAISettings(_BaseGroupSettings):
    """OpenAI API provider settings (supports Euri Proxy API)."""

    api_key: SecretStr = Field(
        default=SecretStr("placeholder"),
        alias="OPENAI_API_KEY",
        description="OpenAI / Euri API key.",
    )
    org_id: str | None = Field(
        default=None,
        alias="OPENAI_ORG_ID",
        description="Optional OpenAI organisation ID.",
    )
    base_url: AnyHttpUrl | None = Field(
        default="https://api.euron.one/api/v1/euri",  # type: ignore[assignment]
        alias="OPENAI_BASE_URL",
        description="Override base URL (e.g. https://api.euron.one/api/v1/euri).",
    )
    chat_model: str = Field(
        default="gpt-4.1-mini",
        alias="OPENAI_CHAT_MODEL",
        description="Default chat completion model identifier.",
    )
    embedding_model: str = Field(
        default="text-embedding-ada-002",
        alias="OPENAI_EMBEDDING_MODEL",
        description="Default text embedding model identifier.",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        alias="OPENAI_MAX_TOKENS",
        description="Maximum tokens in the completion response.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        alias="OPENAI_TEMPERATURE",
        description="Sampling temperature (default 0.7).",
    )
    request_timeout: int = Field(
        default=60,
        ge=1,
        alias="OPENAI_REQUEST_TIMEOUT",
        description="HTTP request timeout in seconds.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        alias="OPENAI_MAX_RETRIES",
        description="Number of automatic retries on transient failures.",
    )


# ---------------------------------------------------------------------------


class AnthropicSettings(_BaseGroupSettings):
    """Anthropic (Claude) API provider settings."""

    api_key: SecretStr | None = Field(
        default=None,
        alias="ANTHROPIC_API_KEY",
        description="Anthropic API key (sk-ant-...).",
    )
    base_url: AnyHttpUrl | None = Field(
        default=None,
        alias="ANTHROPIC_BASE_URL",
        description="Override base URL (e.g. for proxies).",
    )
    chat_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        alias="ANTHROPIC_CHAT_MODEL",
        description="Default Claude model identifier.",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        alias="ANTHROPIC_MAX_TOKENS",
        description="Maximum tokens in the completion response.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        alias="ANTHROPIC_TEMPERATURE",
        description="Sampling temperature.",
    )
    request_timeout: int = Field(
        default=60,
        ge=1,
        alias="ANTHROPIC_REQUEST_TIMEOUT",
        description="HTTP request timeout in seconds.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        alias="ANTHROPIC_MAX_RETRIES",
        description="Number of automatic retries on transient failures.",
    )

    @property
    def is_configured(self) -> bool:
        """Return True if an API key has been supplied."""
        return self.api_key is not None


# ---------------------------------------------------------------------------


class GeminiSettings(_BaseGroupSettings):
    """Google Gemini API provider settings."""

    api_key: SecretStr | None = Field(
        default=None,
        alias="GOOGLE_API_KEY",
        description="Google AI Studio / Vertex AI API key.",
    )
    chat_model: str = Field(
        default="gemini-1.5-flash",
        alias="GOOGLE_CHAT_MODEL",
        description="Default Gemini model identifier.",
    )
    embedding_model: str = Field(
        default="models/text-embedding-004",
        alias="GOOGLE_EMBEDDING_MODEL",
        description="Default Gemini embedding model identifier.",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        alias="GOOGLE_MAX_TOKENS",
        description="Maximum tokens in the completion response.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        alias="GOOGLE_TEMPERATURE",
        description="Sampling temperature.",
    )
    request_timeout: int = Field(
        default=60,
        ge=1,
        alias="GOOGLE_REQUEST_TIMEOUT",
        description="HTTP request timeout in seconds.",
    )

    @property
    def is_configured(self) -> bool:
        """Return True if an API key has been supplied."""
        return self.api_key is not None


# ---------------------------------------------------------------------------


class GroqSettings(_BaseGroupSettings):
    """Groq Cloud API provider settings (ultra-fast inference)."""

    api_key: SecretStr | None = Field(
        default=None,
        alias="GROQ_API_KEY",
        description="Groq Cloud API key (gsk_...).",
    )
    base_url: AnyHttpUrl = Field(
        default="https://api.groq.com/openai/v1",  # type: ignore[assignment]
        alias="GROQ_BASE_URL",
        description="Groq API base URL (OpenAI-compatible endpoint).",
    )
    chat_model: str = Field(
        default="llama-3.3-70b-versatile",
        alias="GROQ_CHAT_MODEL",
        description="Default Groq model identifier.",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        alias="GROQ_MAX_TOKENS",
        description="Maximum tokens in the completion response.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        alias="GROQ_TEMPERATURE",
        description="Sampling temperature.",
    )
    request_timeout: int = Field(
        default=30,
        ge=1,
        alias="GROQ_REQUEST_TIMEOUT",
        description="HTTP request timeout in seconds.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        alias="GROQ_MAX_RETRIES",
        description="Number of automatic retries on transient failures.",
    )

    @property
    def is_configured(self) -> bool:
        """Return True if an API key has been supplied."""
        return self.api_key is not None


# ---------------------------------------------------------------------------


class OllamaSettings(BaseSettings):
    """Ollama local inference server settings."""

    model_config = SettingsConfigDict(extra="ignore")

    base_url: AnyHttpUrl = Field(
        default="http://localhost:11434",  # type: ignore[assignment]
        alias="OLLAMA_BASE_URL",
        description="Ollama server base URL.",
    )
    chat_model: str = Field(
        default="llama3.2",
        alias="OLLAMA_CHAT_MODEL",
        description="Default Ollama model to use for chat.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        alias="OLLAMA_EMBEDDING_MODEL",
        description="Default Ollama model to use for embeddings.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        alias="OLLAMA_TEMPERATURE",
        description="Sampling temperature.",
    )
    request_timeout: int = Field(
        default=120,
        ge=1,
        alias="OLLAMA_REQUEST_TIMEOUT",
        description="HTTP request timeout in seconds (Ollama can be slow on CPU).",
    )
    keep_alive: str = Field(
        default="5m",
        alias="OLLAMA_KEEP_ALIVE",
        description="How long to keep the model loaded (e.g. '5m', '1h', '-1').",
    )

    @property
    def api_url(self) -> str:
        """Return the Ollama OpenAI-compatible API URL."""
        return f"{self.base_url}/v1"


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------


class QdrantSettings(_BaseGroupSettings):
    """Qdrant vector database settings."""

    host: str = Field(
        default="localhost",
        alias="QDRANT_HOST",
        description="Qdrant server hostname or IP.",
    )
    port: int = Field(
        default=6333,
        alias="QDRANT_PORT",
        description="Qdrant HTTP/REST port.",
    )
    grpc_port: int = Field(
        default=6334,
        alias="QDRANT_GRPC_PORT",
        description="Qdrant gRPC port.",
    )
    qdrant_url: str | None = Field(
        default=None,
        alias="QDRANT_URL",
        description="Full Qdrant Cloud URL (e.g. https://<cluster-id>.cloud.qdrant.io).",
    )
    api_key: SecretStr | None = Field(
        default=None,
        alias="QDRANT_API_KEY",
        description="API key for Qdrant Cloud. Leave empty for local.",
    )
    https: bool = Field(
        default=False,
        alias="QDRANT_HTTPS",
        description="Use HTTPS. Set True in production / Qdrant Cloud.",
    )
    collection_name: str = Field(
        default="legal_documents",
        alias="QDRANT_COLLECTION_NAME",
        description="Target Qdrant collection name.",
    )
    vector_size: int = Field(
        default=1536,
        ge=1,
        alias="QDRANT_VECTOR_SIZE",
        description="Embedding dimension. Must match the embedding model output (1536 for text-embedding-ada-002).",
    )
    prefer_grpc: bool = Field(
        default=False,
        alias="QDRANT_PREFER_GRPC",
        description="Use gRPC transport instead of HTTP for higher throughput.",
    )
    timeout: int = Field(
        default=30,
        ge=1,
        alias="QDRANT_TIMEOUT",
        description="Client operation timeout in seconds.",
    )

    @property
    def url(self) -> str:
        """Full Qdrant REST connection URL."""
        if self.qdrant_url:
            return self.qdrant_url
        scheme = "https" if self.https else "http"
        return f"{scheme}://{self.host}:{self.port}"


# ---------------------------------------------------------------------------
# Embedding & Retrieval
# ---------------------------------------------------------------------------


class EmbeddingSettings(_BaseGroupSettings):
    """Embedding model configuration."""

    provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OPENAI,
        alias="EMBEDDING_PROVIDER",
        description="Embedding backend to use (bge, openai, voyage, jina, ollama).",
    )
    model_name: str = Field(
        default="text-embedding-ada-002",
        alias="EMBEDDING_MODEL_NAME",
        description="Model identifier (text-embedding-ada-002, BAAI/bge-m3, etc.).",
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        alias="EMBEDDING_BATCH_SIZE",
        description="Number of texts to encode per forward pass.",
    )
    device: str = Field(
        default="cpu",
        alias="EMBEDDING_DEVICE",
        description="Compute device: 'cpu', 'cuda', or 'mps'.",
    )
    normalize: bool = Field(
        default=True,
        alias="EMBEDDING_NORMALIZE",
        description="L2-normalize embeddings (recommended for cosine similarity).",
    )


# ---------------------------------------------------------------------------


class RetrievalSettings(BaseSettings):
    """Hybrid retrieval configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    top_k_dense: int = Field(
        default=20,
        ge=1,
        alias="RETRIEVAL_TOP_K_DENSE",
        description="Candidate count from dense (semantic) retrieval.",
    )
    top_k_sparse: int = Field(
        default=20,
        ge=1,
        alias="RETRIEVAL_TOP_K_SPARSE",
        description="Candidate count from sparse (BM25) retrieval.",
    )
    top_k_rerank: int = Field(
        default=5,
        ge=1,
        alias="RETRIEVAL_TOP_K_RERANK",
        description="Final results returned after reranking.",
    )
    score_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        alias="RETRIEVAL_SCORE_THRESHOLD",
        description="Minimum similarity score; results below this are dropped.",
    )
    hybrid_alpha: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.5,
        alias="HYBRID_ALPHA",
        description="Fusion weight: 0.0 = sparse only, 1.0 = dense only.",
    )

    @model_validator(mode="after")
    def _validate_top_k_order(self) -> "RetrievalSettings":
        """top_k_rerank must not exceed the total candidate pool."""
        total = self.top_k_dense + self.top_k_sparse
        if self.top_k_rerank > total:
            msg = (
                f"RETRIEVAL_TOP_K_RERANK ({self.top_k_rerank}) must be <= "
                f"RETRIEVAL_TOP_K_DENSE + RETRIEVAL_TOP_K_SPARSE ({total})"
            )
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------


class RerankerSettings(BaseSettings):
    """Cross-encoder / FlashRank reranker settings."""

    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(
        default=True,
        alias="RERANKER_ENABLED",
        description="Toggle reranking on/off.",
    )
    model: str = Field(
        default="rerank-v3.5",
        alias="RERANKER_MODEL",
        description="API reranker model identifier (default: rerank-v3.5).",
    )
    max_length: int = Field(
        default=512,
        ge=64,
        alias="RERANKER_MAX_LENGTH",
        description="Maximum token length passed to the reranker.",
    )
    device: str = Field(
        default="cpu",
        alias="RERANKER_DEVICE",
        description="Compute device for reranker: 'cpu', 'cuda', or 'mps'.",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        alias="RERANKER_BATCH_SIZE",
        description="Batch size for cross-encoder inference.",
    )


# ---------------------------------------------------------------------------
# Ingestion & File Storage
# ---------------------------------------------------------------------------


class IngestionSettings(BaseSettings):
    """Document ingestion and chunking settings."""

    model_config = SettingsConfigDict(extra="ignore")

    chunk_size: int = Field(
        default=1000,
        ge=100,
        alias="INGESTION_CHUNK_SIZE",
        description="Target chunk size in characters.",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        alias="INGESTION_CHUNK_OVERLAP",
        description="Character overlap between consecutive chunks.",
    )
    batch_size: int = Field(
        default=50,
        ge=1,
        alias="INGESTION_BATCH_SIZE",
        description="Number of chunks to embed and upsert per batch.",
    )

    @field_validator("chunk_overlap", mode="after")
    @classmethod
    def _overlap_less_than_chunk(cls, v: int, info: Any) -> int:  # noqa: ANN401
        """chunk_overlap must be strictly less than chunk_size."""
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            msg = (
                f"INGESTION_CHUNK_OVERLAP ({v}) must be < "
                f"INGESTION_CHUNK_SIZE ({chunk_size})"
            )
            raise ValueError(msg)
        return v


# ---------------------------------------------------------------------------


class DirectorySettings(BaseSettings):
    """
    File-system path configuration.

    All directories are resolved as absolute paths relative to the project
    root so the application works regardless of the current working directory.
    """

    model_config = SettingsConfigDict(extra="ignore")

    # Source legal documents (read-only corpus)
    document_dir: Path = Field(
        default=_PROJECT_ROOT / "Data_Set",
        alias="DOCUMENT_DIR",
        description="Directory containing the raw legal document corpus.",
    )
    # Temporary uploads from users / API
    upload_dir: Path = Field(
        default=_PROJECT_ROOT / "uploads",
        alias="UPLOAD_DIR",
        description="Temporary staging directory for uploaded files.",
    )
    # RAGAS / eval datasets and results
    evaluation_dir: Path = Field(
        default=_PROJECT_ROOT / "evaluation",
        alias="EVALUATION_DIR",
        description="Directory for evaluation datasets, runners, and reports.",
    )
    # Processed / derived artefacts
    processed_dir: Path = Field(
        default=_PROJECT_ROOT / "processed",
        alias="PROCESSED_DIR",
        description="Directory for post-ingestion processed documents.",
    )

    @property
    def data_set(self) -> Path:
        """Alias for document_dir pointing to Data_Set directory."""
        return self.document_dir

    def create_all(self) -> None:
        """Create all configured directories if they do not already exist."""
        for path in (
            self.document_dir,
            self.upload_dir,
            self.evaluation_dir,
            self.processed_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


class StreamlitSettings(BaseSettings):
    """Streamlit front-end settings."""

    model_config = SettingsConfigDict(extra="ignore")

    host: str = Field(
        default="localhost",
        alias="STREAMLIT_HOST",
        description="Host on which Streamlit listens.",
    )
    port: int = Field(
        default=8501,
        ge=1024,
        le=65535,
        alias="STREAMLIT_PORT",
        description="TCP port for the Streamlit server.",
    )
    theme: StreamlitTheme = Field(
        default=StreamlitTheme.DARK,
        alias="STREAMLIT_THEME",
        description="UI colour theme ('dark' | 'light').",
    )
    page_title: str = Field(
        default="Indian Immigration Legal Assistant",
        alias="STREAMLIT_PAGE_TITLE",
        description="Browser tab / page title.",
    )
    page_icon: str = Field(
        default="⚖️",
        alias="STREAMLIT_PAGE_ICON",
        description="Emoji or path to an image used as the browser favicon.",
    )
    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        alias="STREAMLIT_MAX_UPLOAD_SIZE_MB",
        description="Maximum file upload size in megabytes.",
    )
    api_base_url: AnyHttpUrl = Field(
        default="http://localhost:8000",  # type: ignore[assignment]
        alias="STREAMLIT_API_BASE_URL",
        description="FastAPI backend URL used by the Streamlit client.",
    )

    @property
    def server_url(self) -> str:
        """Full URL where the Streamlit app is accessible."""
        return f"http://{self.host}:{self.port}"


# ---------------------------------------------------------------------------
# Supporting Services
# ---------------------------------------------------------------------------


class CacheSettings(BaseSettings):
    """Response and embedding cache settings."""

    model_config = SettingsConfigDict(extra="ignore")

    backend: CacheBackend = Field(
        default=CacheBackend.DISK,
        alias="CACHE_BACKEND",
        description="Cache implementation: 'disk', 'redis', or 'none'.",
    )
    ttl_seconds: int = Field(
        default=3600,
        ge=0,
        alias="CACHE_TTL_SECONDS",
        description="Default cache entry TTL in seconds (0 = no expiry).",
    )
    max_size_mb: int = Field(
        default=512,
        ge=1,
        alias="CACHE_MAX_SIZE_MB",
        description="Maximum disk cache size in megabytes.",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        alias="REDIS_URL",
        description="Redis connection URL. Only used when backend='redis'.",
    )


# ---------------------------------------------------------------------------


class LangSmithSettings(BaseSettings):
    """LangSmith tracing and observability settings."""

    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(
        default=False,
        alias="LANGCHAIN_TRACING_V2",
        description="Enable LangSmith tracing.",
    )
    endpoint: str = Field(
        default="https://api.smith.langchain.com",
        alias="LANGCHAIN_ENDPOINT",
        description="LangSmith ingestion endpoint.",
    )
    api_key: SecretStr | None = Field(
        default=None,
        alias="LANGCHAIN_API_KEY",
        description="LangSmith API key.",
    )
    project: str = Field(
        default="indian-immigration-legal-assistant",
        alias="LANGCHAIN_PROJECT",
        description="LangSmith project name.",
    )


# ---------------------------------------------------------------------------
# Root Settings — aggregates all domain-scoped settings groups
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Root settings object — aggregates all domain-scoped settings groups.

    Loaded once at application startup via ``get_settings()``.
    Pass via dependency injection; never import in business-logic modules.

    Environment loading order (highest priority first):
    1. OS environment variables
    2. ``.env`` file at the project root
    3. Field defaults
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter=None,
    )

    # Active LLM provider used when provider-agnostic code asks for a model
    active_llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        alias="ACTIVE_LLM_PROVIDER",
        description="Which LLM backend to use for generation.",
    )

    # Nested groups
    app: AppSettings = Field(default_factory=AppSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    api: APISettings = Field(
        default_factory=lambda: APISettings(API_SECRET_KEY="CHANGE_ME")  # type: ignore[call-arg]
    )
    auth: AuthSettings = Field(
        default_factory=lambda: AuthSettings(JWT_SECRET_KEY="CHANGE_ME")  # type: ignore[call-arg]
    )
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)

    # LLM providers
    openai: OpenAISettings = Field(
        default_factory=lambda: OpenAISettings(OPENAI_API_KEY="placeholder")  # type: ignore[call-arg]
    )
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)

    # Infrastructure
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)

    # Application domains
    directories: DirectorySettings = Field(default_factory=DirectorySettings)
    streamlit: StreamlitSettings = Field(default_factory=StreamlitSettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)

    @property
    def dirs(self) -> DirectorySettings:
        """Convenience alias for self.directories."""
        return self.directories

    @property
    def is_production(self) -> bool:
        """Return True when deployed to production."""
        return self.app.env == AppEnvironment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Return True during local development."""
        return self.app.env == AppEnvironment.DEVELOPMENT

    @property
    def is_staging(self) -> bool:
        """Return True when deployed to staging."""
        return self.app.env == AppEnvironment.STAGING

    @property
    def active_llm_settings(
        self,
    ) -> OpenAISettings | AnthropicSettings | GeminiSettings | GroqSettings | OllamaSettings:
        """Return the settings object for the currently active LLM provider."""
        mapping: dict[
            LLMProvider,
            OpenAISettings | AnthropicSettings | GeminiSettings | GroqSettings | OllamaSettings,
        ] = {
            LLMProvider.OPENAI: self.openai,
            LLMProvider.ANTHROPIC: self.anthropic,
            LLMProvider.GOOGLE: self.gemini,
            LLMProvider.GROQ: self.groq,
            LLMProvider.OLLAMA: self.ollama,
        }
        return mapping[self.active_llm_provider]

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def bootstrap(self) -> None:
        """
        Perform all one-time environment setup tasks on first load.

        * Create required file-system directories.
        * Wire LangSmith environment variables so LangChain picks them up.
        """
        import os  # noqa: PLC0415

        self.directories.create_all()
        self.logging.log_dir.mkdir(parents=True, exist_ok=True)

        if self.langsmith.enabled and self.langsmith.api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = self.langsmith.endpoint
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith.api_key.get_secret_value()
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith.project

    def log_startup_banner(self) -> None:
        """Emit a structured log with non-sensitive configuration on startup."""
        import structlog  # noqa: PLC0415

        log = structlog.get_logger(__name__)
        log.info(
            "configuration_loaded",
            app=self.app.name,
            env=self.app.env.value,
            version=self.app.version,
            log_level=self.logging.level.value,
            active_llm=self.active_llm_provider.value,
            embedding_model=self.embedding.model_name,
            qdrant_url=self.qdrant.url,
            collection=self.qdrant.collection_name,
            reranker_enabled=self.reranker.enabled,
            langsmith_enabled=self.langsmith.enabled,
        )

    def as_safe_dict(self) -> dict[str, object]:
        """
        Return a sanitised dictionary safe to log or display in UIs.

        All ``SecretStr`` values are replaced with ``"***"``.
        """

        def _sanitise(obj: object) -> object:
            if isinstance(obj, SecretStr):
                return "***"
            if isinstance(obj, dict):
                return {k: _sanitise(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitise(i) for i in obj]
            return obj

        return _sanitise(self.model_dump())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Module-level singleton (cached)
# ---------------------------------------------------------------------------

_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Return the cached ``Settings`` singleton.

    Reads ``.env`` and environment variables on the first call; subsequent
    calls return the in-process cached instance.

    Prefer dependency injection over importing ``settings`` directly in
    business-logic or infrastructure code.
    """
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()
        _settings.bootstrap()
    return _settings


@lru_cache(maxsize=1)
def get_settings_cached() -> Settings:
    """
    LRU-cached variant suitable for use as a FastAPI dependency.

    Example::

        from fastapi import Depends
        from src.core.config import get_settings_cached, Settings

        def my_route(cfg: Settings = Depends(get_settings_cached)):
            ...
    """
    instance = Settings()
    instance.bootstrap()
    return instance


def reset_settings() -> None:
    """
    Reset the settings singleton.

    **Use only in tests** to force re-reading of environment variables between
    test cases.  Never call this in production code.
    """
    global _settings  # noqa: PLW0603
    _settings = None
    get_settings_cached.cache_clear()


# Convenience alias — import this in application code
settings: Settings = get_settings()


