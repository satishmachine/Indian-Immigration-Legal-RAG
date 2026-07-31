"""
retrieval.reranker.factory
==========================
Clean Architecture Factory for Reranker Engines.
"""

from __future__ import annotations

import logging
from typing import Any

from core.config import get_settings
from core.interfaces.reranker_interface import RerankerService
from retrieval.reranker.api_reranker import ApiReranker
from retrieval.reranker.cohere_reranker import CohereReranker

logger = logging.getLogger(__name__)


class RerankerFactory:
    """Factory to instantiate and configure API-based RerankerService providers."""

    _registry: dict[str, type[RerankerService]] = {
        "api": ApiReranker,
        "cohere": CohereReranker,
    }

    @classmethod
    def create(
        cls,
        provider: str | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> RerankerService:
        """
        Create a RerankerService instance.

        Args:
            provider: Reranker provider ('api', 'cohere').
            model_name: Model identifier override.
            kwargs: Constructor options.

        Returns:
            RerankerService instance.
        """
        cfg = get_settings().reranker
        model = model_name or cfg.model or "rerank-v3.5"

        if provider:
            prov_key = provider.lower()
        elif "cohere" in model.lower():
            prov_key = "cohere"
        else:
            prov_key = "api"

        if prov_key not in cls._registry:
            prov_key = "api"

        logger.info("Instantiating API reranker provider %r with model %r", prov_key, model)

        if prov_key == "cohere":
            return CohereReranker(model_name=model, **kwargs)

        return ApiReranker(model_name=model, **kwargs)

