"""
app.state
=========
Centralized Session State Manager for Legal AI Assistant.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any
import streamlit as st


class AppState:
    """State manager for Streamlit application."""

    @staticmethod
    def initialize() -> None:
        """Initialize all session state defaults."""
        defaults: dict[str, Any] = {
            "session_id": str(uuid.uuid4()),
            "messages": [],
            "conversations": [],
            "thumbs": {},
            "selected_pdf": None,
            "uploaded_pdfs": [],
            "language": "English",
            "pending_question": None,
            "domain_filter": None,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value

    @staticmethod
    def get_session_id() -> str:
        if "session_id" not in st.session_state:
            st.session_state["session_id"] = str(uuid.uuid4())
        return st.session_state["session_id"]

    @staticmethod
    def clear_chat() -> None:
        """Reset active conversation."""
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []
        st.session_state["thumbs"] = {}
        st.session_state["selected_pdf"] = None

    @staticmethod
    def add_message(role: str, content: str, response: Any | None = None) -> None:
        """Append message to active conversation history."""
        messages: list[dict[str, Any]] = st.session_state.get("messages", [])
        messages.append({
            "role": role,
            "content": content,
            "response": response,
            "timestamp": datetime.now().strftime("%I:%M %p"),
        })
        st.session_state["messages"] = messages

        # Update conversation list title
        if role == "user" and len([m for m in messages if m["role"] == "user"]) == 1:
            AppState._add_to_conversations(content[:45])

    @staticmethod
    def _add_to_conversations(title: str) -> None:
        convs: list[dict[str, Any]] = st.session_state.get("conversations", [])
        convs.insert(0, {
            "session_id": AppState.get_session_id(),
            "title": title,
            "time": datetime.now().strftime("%d %b, %H:%M"),
        })
        st.session_state["conversations"] = convs

    @staticmethod
    def get_messages() -> list[dict[str, Any]]:
        return st.session_state.get("messages", [])

    @staticmethod
    def get_conversations() -> list[dict[str, Any]]:
        return st.session_state.get("conversations", [])

    @staticmethod
    def set_thumbs(index: int, value: str | None) -> None:
        thumbs: dict[int, str | None] = st.session_state.get("thumbs", {})
        thumbs[index] = value
        st.session_state["thumbs"] = thumbs

    @staticmethod
    def get_thumbs(index: int) -> str | None:
        return st.session_state.get("thumbs", {}).get(index)

    @staticmethod
    def set_pending_question(question: str) -> None:
        st.session_state["pending_question"] = question

    @staticmethod
    def pop_pending_question() -> str | None:
        q = st.session_state.get("pending_question")
        if q:
            st.session_state["pending_question"] = None
        return q

    @staticmethod
    def set_language(lang: str) -> None:
        st.session_state["language"] = lang

    @staticmethod
    def get_language() -> str:
        return st.session_state.get("language", "English")

    @staticmethod
    def set_selected_pdf(pdf_name: str | None) -> None:
        st.session_state["selected_pdf"] = pdf_name

    @staticmethod
    def get_selected_pdf() -> str | None:
        return st.session_state.get("selected_pdf")

    @staticmethod
    def add_uploaded_pdf(pdf_name: str) -> None:
        pdfs: list[str] = st.session_state.get("uploaded_pdfs", [])
        if pdf_name not in pdfs:
            pdfs.append(pdf_name)
            st.session_state["uploaded_pdfs"] = pdfs

    @staticmethod
    def get_uploaded_pdfs() -> list[str]:
        return st.session_state.get("uploaded_pdfs", [])

    @staticmethod
    def export_chat_json() -> str:
        messages = AppState.get_messages()
        export_data = []
        for msg in messages:
            item = {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg.get("timestamp"),
            }
            resp = msg.get("response")
            if resp:
                item["confidence_score"] = getattr(resp, "confidence_score", 1.0)
                item["has_penalty"] = getattr(resp, "has_penalty_clause", False)
                item["citations"] = [c.model_dump() for c in getattr(resp, "citations", [])]
            export_data.append(item)
        return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
