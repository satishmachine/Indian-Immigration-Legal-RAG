"""
chains.history_aware_retriever
==============================
LangChain LCEL History-Aware Retriever Component.

Reformulates conversational user queries into standalone search queries when chat history is present,
then retrieves relevant statutory context.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document as LCDocument
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableBranch, RunnablePassthrough

from chains.prompts import REPHRASE_QUESTION_PROMPT

logger = logging.getLogger(__name__)


def create_history_aware_legal_retriever(
    llm: BaseChatModel,
    retriever: BaseRetriever,
) -> Runnable:
    """
    Construct an LCEL History-Aware Retriever chain.

    If `chat_history` exists: rephrases user query into a standalone query using `llm`, then retrieves docs.
    If `chat_history` is empty: passes input directly to `retriever`.

    Args:
        llm: LCEL BaseChatModel.
        retriever: LCEL BaseRetriever (LegalCompressionRetriever / LegalHybridRetriever).

    Returns:
        LCEL Runnable returning list[LCDocument].
    """
    rephrase_chain = REPHRASE_QUESTION_PROMPT | llm | StrOutputParser()

    def _get_input(input_dict: dict[str, Any]) -> str:
        return input_dict.get("input") or input_dict.get("question") or ""

    def _has_history(input_dict: dict[str, Any]) -> bool:
        history = input_dict.get("chat_history")
        return bool(history and len(history) > 0)

    # Branch 1: Has history -> Rephrase question -> Retrieve docs
    history_branch = (
        rephrase_chain
        | retriever
    )

    # Branch 2: No history -> Direct retrieval
    no_history_branch = (
        _get_input
        | retriever
    )

    # Combine into RunnableBranch
    history_aware_retriever = RunnableBranch(
        (_has_history, history_branch),
        no_history_branch,
    )

    return history_aware_retriever
