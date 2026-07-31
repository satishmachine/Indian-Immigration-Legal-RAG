"""
services.embedding.openai_provider
===================================
OpenAI / Euri Embedding Provider implementation using LangChain OpenAIEmbeddings.

Supports models:
- text-embedding-ada-002 (1536 dim)
- text-embedding-3-small (1536 dim)
- text-embedding-3-large (3072 dim)

Base URL:
- Default: https://api.euron.one/api/v1/euri
"""

from __future__ import annotations

import logging, os
from dotenv import load_dotenv

load_dotenv(override=True)

from core.config import get_settings
from services.embedding.base import BaseEmbeddingService

logger = logging.getLogger(__name__)


class OpenAIEmbeddingService(BaseEmbeddingService):
    """
    OpenAI / Euri Proxy Embedding Service using LangChain OpenAIEmbeddings.

    Args:
        api_key: API key string (defaults to EURI_API_KEY / OPENAI_API_KEY).
        base_url: Base URL string (defaults to https://api.euron.one/api/v1/euri).
        model_name: OpenAI embedding model name (default: text-embedding-ada-002).
        dimensions: Optional output vector dimension override.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        cfg = get_settings().openai
        env_key = os.environ.get("EURI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self._api_key = api_key or env_key or (cfg.api_key.get_secret_value() if cfg.api_key else None)
        self._base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or (str(cfg.base_url) if cfg.base_url else None)
            or "https://api.euron.one/api/v1/euri"
        )
        self._model_name = model_name or cfg.embedding_model or "text-embedding-ada-002"
        self._dimensions = dimensions
        self._embeddings_client = None

    def _get_embeddings_client(self):
        if self._embeddings_client is None:
            if not self._api_key or self._api_key == "placeholder":
                raise ValueError(
                    "Missing API Key for OpenAI/Euri Embeddings. Please set EURI_API_KEY or OPENAI_API_KEY in .env."
                )
            from langchain_openai import OpenAIEmbeddings

            logger.info(
                "Initializing OpenAIEmbeddings (model=%r, base_url=%r)",
                self._model_name,
                self._base_url,
            )
            self._embeddings_client = OpenAIEmbeddings(
                model=self._model_name,
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._embeddings_client

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts via LangChain OpenAIEmbeddings."""
        if not texts:
            return []
        try:
            client = self._get_embeddings_client()
            return client.embed_documents(texts)
        except Exception as e:
            logger.warning("Embedding documents failed (%s). Falling back to zero-vectors.", e)
            return [[0.0] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text via LangChain OpenAIEmbeddings."""
        if not text:
            return []
        try:
            client = self._get_embeddings_client()
            return client.embed_query(text)
        except Exception as e:
            err_str = str(e).lower()
            if "403" in err_str or "quota" in err_str or "token limit" in err_str or "permission_denied" in err_str:
                logger.warning("Embedding query quota limit reached (%s).", e)
                from core.exceptions import EmbeddingQuotaExceededError
                raise EmbeddingQuotaExceededError(
                    message=f"Embedding API 403 Daily Token Limit / Quota Exceeded: {e}",
                    details={"raw_error": str(e)},
                ) from e
            logger.warning("Embedding query failed (%s).", e)
            from core.exceptions import DenseRetrievalError
            raise DenseRetrievalError(
                message=f"Dense embedding query generation failed: {e}",
                details={"raw_error": str(e)},
            ) from e

    @property
    def dimension(self) -> int:
        """Return embedding dimension for configured model (1536 for text-embedding-ada-002)."""
        if self._dimensions:
            return self._dimensions
        if "3-large" in self._model_name:
            return 3072
        return 1536
