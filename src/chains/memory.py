"""
chains.memory
=============
Conversation Memory Manager.

Provides thread/session-based chat history management using `InMemoryChatMessageHistory`.
"""

from __future__ import annotations

import logging
from typing import Dict

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

logger = logging.getLogger(__name__)

# Session history store mapping session_id -> BaseChatMessageHistory
_session_store: Dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieve or create an in-memory chat message history for the given session ID.

    Args:
        session_id: Unique conversation identifier.

    Returns:
        BaseChatMessageHistory instance.
    """
    if session_id not in _session_store:
        logger.info("Creating new InMemoryChatMessageHistory for session %r", session_id)
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


def clear_session_history(session_id: str) -> bool:
    """
    Clear chat history for a session ID.

    Args:
        session_id: Unique conversation identifier.

    Returns:
        True if history existed and was cleared, False otherwise.
    """
    if session_id in _session_store:
        _session_store[session_id].clear()
        del _session_store[session_id]
        logger.info("Cleared session history for %r", session_id)
        return True
    return False
