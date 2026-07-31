"""
services.chat.mock_llm
======================
Deterministic Mock Legal LLM for Offline Testing & Demo Mode.

Used as a fallback when LLM API keys are unconfigured or set to placeholders.
"""

from __future__ import annotations

import logging
from typing import Any
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class MockLegalLLM(SimpleChatModel):
    """Deterministic Mock LLM for Statutory Legal QA."""

    @property
    def _llm_type(self) -> str:
        return "mock_legal_llm"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> str:
        last_text = messages[-1].content if messages else ""
        system_text = str(messages[0].content) if messages else ""

        # Rephrase prompt mode
        if "rephrase" in system_text.lower() or "standalone" in system_text.lower():
            return f"Standalone Question: {last_text}"

        return (
            "### Direct Legal Answer\n"
            "Under Section 21 of The Immigration and Foreigners Act, 2025, any foreigner entering India "
            "without valid travel documents or visa commits an offence punishable with imprisonment up to 5 years.\n\n"
            "### Statutory Provisions & Analysis\n"
            "As prescribed in Section 21 of the Act, entering India without valid authority is strictly prohibited.\n\n"
            "### Penalties & Offences\n"
            "Punishable with imprisonment for a term up to 5 years and fine under Section 21."
        )
