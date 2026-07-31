"""
services.chat.llm_factory
==========================
Factory for instantiating LangChain LLM instances.
Supports Groq (llama-3.3-70b-versatile), OpenAI / Euri Proxy, and Mock fallback.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from core.config import get_settings

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for LangChain Chat Models supporting Groq, OpenAI/Euri, and Mock modes."""

    @classmethod
    def create(
        cls,
        provider: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.7,
        streaming: bool = False,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """
        Instantiate Chat LLM based on provider (groq / openai / euri).
        Default model for Groq: llama-3.3-70b-versatile.
        """
        settings = get_settings()

        groq_key = (
            api_key
            or os.environ.get("GROQ_API_KEY")
            or (settings.groq.api_key.get_secret_value() if settings.groq.api_key else None)
        )
        if groq_key:
            groq_key = groq_key.strip("'\"")

        # 1. Try Groq Provider if GROQ_API_KEY is present or provider == 'groq'
        if groq_key and groq_key not in ("placeholder", "your_groq_api_key_here", ""):
            resolved_model = (
                model_name
                or os.environ.get("GROQ_CHAT_MODEL")
                or "llama-3.3-70b-versatile"
            )

            logger.info("Instantiating ChatGroq (model=%r, temperature=%.2f, streaming=%s)", resolved_model, temperature, streaming)
            if GROQ_AVAILABLE:
                return ChatGroq(
                    model=resolved_model,
                    groq_api_key=groq_key,
                    temperature=temperature,
                    streaming=streaming,
                    **kwargs,
                )
            else:
                return ChatOpenAI(
                    model=resolved_model,
                    api_key=groq_key,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=temperature,
                    streaming=streaming,
                    **kwargs,
                )

        # 2. Fallback to OpenAI / Euri Proxy
        cfg = settings.openai
        euri_key = os.environ.get("EURI_API_KEY") or os.environ.get("OPENAI_API_KEY") or (cfg.api_key.get_secret_value() if cfg.api_key else None)
        resolved_euri_key = euri_key.strip("'\"") if euri_key else None

        if resolved_euri_key and resolved_euri_key not in ("placeholder", "your_openai_api_key_here"):
            resolved_model = model_name or os.environ.get("OPENAI_CHAT_MODEL") or getattr(cfg, "chat_model", "gpt-4.1-mini")
            resolved_base_url = (
                base_url
                or os.environ.get("OPENAI_BASE_URL")
                or (str(cfg.base_url) if cfg.base_url else None)
                or "https://api.euron.one/api/v1/euri"
            )

            logger.info(
                "Instantiating ChatOpenAI (model=%r, base_url=%r, temperature=%.2f)",
                resolved_model,
                resolved_base_url,
                temperature,
            )
            return ChatOpenAI(
                model=resolved_model,
                api_key=resolved_euri_key,
                base_url=resolved_base_url,
                temperature=temperature,
                streaming=streaming,
                **kwargs,
            )

        # 3. Fallback to Mock LLM if no keys are configured
        logger.warning("No GROQ_API_KEY or EURI_API_KEY found. Falling back to MockLegalLLM.")
        from services.chat.mock_llm import MockLegalLLM
        return MockLegalLLM()


def get_llm(
    provider: str | None = None,
    model_name: str | None = None,
    temperature: float = 0.7,
    streaming: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    """Helper function to get configured LLM instance."""
    return LLMFactory.create(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        streaming=streaming,
        **kwargs,
    )
