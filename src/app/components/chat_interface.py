"""
app.components.chat_interface
==============================
Main Chat Window: Streaming responses, copy, export, feedback, citations, and similar questions.
"""

from __future__ import annotations

import html as html_lib
import logging
import time
from typing import Callable, Generator

import streamlit as st

from app.components.header import render_header
from app.state import AppState
from core.models.response import LegalCitation, LegalRAGResponse

logger = logging.getLogger(__name__)

# ── Dynamic Similar Questions ──────────────────────────────────────────────────
_SIMILAR_QUESTIONS = [
    "What are the penalties for visa overstay in India?",
    "How to apply for OCI card registration?",
    "Grounds for foreigner deportation under Foreigners Act 1946?",
    "What are the requirements for citizenship by naturalization?",
]


def render_chat_interface(
    query_fn: Callable[[str, str, dict | None], LegalRAGResponse],
    domain_filter: str | None = None,
) -> None:
    """Render main chat interface window."""

    # 1. Header Bar with Language Tabs
    render_header()

    # 2. Render Existing Chat Messages
    messages = AppState.get_messages()

    for idx, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        response: LegalRAGResponse | None = msg.get("response")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant" and response:
            with st.chat_message("assistant"):
                _render_assistant_output(response, message_index=idx)

    # 3. Handle Pending Question (from Similar Questions pills)
    pending_q = AppState.pop_pending_question()
    if pending_q:
        _execute_chat_query(
            question=pending_q,
            query_fn=query_fn,
            domain_filter=domain_filter,
        )
        st.rerun()

    # 4. Bottom Clearance Spacer
    st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)

    # 5. Fixed Chat Input Strip
    question = st.chat_input("Ask Indian Legal AI...")
    if question and question.strip():
        _execute_chat_query(
            question=question.strip(),
            query_fn=query_fn,
            domain_filter=domain_filter,
        )
        st.rerun()


def _execute_chat_query(
    question: str,
    query_fn: Callable[[str, str, dict | None], LegalRAGResponse],
    domain_filter: str | None,
) -> None:
    """Execute query with streaming effect, save state, and render output."""

    # Render User Message
    with st.chat_message("user"):
        st.markdown(question)

    # Render Assistant Message with Streaming Effect
    with st.chat_message("assistant"):
        session_id = AppState.get_session_id()
        filters = {"legal_domain": domain_filter} if domain_filter else None

        with st.spinner("Searching statutory legal documents…"):
            try:
                response = query_fn(question=question, session_id=session_id, filters=filters)
            except Exception as exc:
                logger.error("RAG Query Exception: %s", exc, exc_info=True)
                response = LegalRAGResponse(
                    question=question,
                    answer="",
                    retrieval_status="retrieval_failed",
                    error_title="System Retrieval Exception",
                    error_message=f"An error occurred while retrieving legal documents: {exc}",
                    developer_details=str(exc),
                    confidence_score=0.0,
                )

        if response.retrieval_status == "success":
            # Real-time Streaming Output Emulation
            def stream_generator() -> Generator[str, None, None]:
                words = response.answer.split(" ")
                for i, word in enumerate(words):
                    yield word + (" " if i < len(words) - 1 else "")
                    time.sleep(0.003)  # Fast, smooth typing delay

            st.write_stream(stream_generator())
        else:
            st.markdown(response.answer)

        # Render complete metadata, citations, and actions
        _render_assistant_output(response, message_index=len(AppState.get_messages()), is_live_stream=True)

    # Save to state
    AppState.add_message("user", question)
    AppState.add_message("assistant", response.answer, response=response)


def _render_assistant_output(response: LegalRAGResponse, message_index: int, is_live_stream: bool = False) -> None:
    """Render assistant answer text, metadata, penal warning, citations, and action toolbar."""

    # ── BRANCH 1: Retrieval Failed Error Card ─────────────────────────────────
    if response.retrieval_status == "retrieval_failed":
        err_title = response.error_title or "System Retrieval Error"
        err_msg = response.error_message or "Our statutory knowledge base is temporarily unavailable."
        st.markdown(
            f"""
            <div style="background:var(--danger-dim); border:1px solid var(--danger-border); border-radius:8px; padding:0.85rem 1rem; margin:0.4rem 0;">
                <div style="color:#FCA5A5; font-size:0.88rem; font-weight:600; margin-bottom:0.25rem;">
                    🔴 {html_lib.escape(err_title)}
                </div>
                <div style="color:var(--text-primary); font-size:0.82rem; line-height:1.5;">
                    {html_lib.escape(err_msg)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── BRANCH 2: No Relevant Documents Warning Card ──────────────────────────
    if response.retrieval_status == "no_documents_found":
        st.markdown(
            """
            <div style="background:var(--warning-dim); border:1px solid rgba(245,158,11,0.25); border-radius:8px; padding:0.85rem 1rem; margin:0.4rem 0;">
                <div style="color:#FBBF24; font-size:0.88rem; font-weight:600; margin-bottom:0.25rem;">
                    📂 No Relevant Statutory Documents Found
                </div>
                <div style="color:var(--text-primary); font-size:0.82rem; line-height:1.5;">
                    I couldn't find any relevant statutory provisions in the indexed legal documents for this query.
                    Try broadening your query or selecting 'All Domains'.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Render Answer Text (for non-live history reruns) ──────────────────────
    if not is_live_stream and response.answer:
        st.markdown(response.answer)

    # ── Penal Warning Bar ─────────────────────────────────────────────────────
    if response.has_penalty_clause:
        st.markdown(
            '<div class="penalty-bar">'
            '⚠️ This response involves penal or criminal consequences under Indian statutory law.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Grounded Citations Panel ──────────────────────────────────────────────
    if response.citations:
        _render_expandable_sources(response.citations, message_index=message_index)

    # ── Action Toolbar: Copy, Thumbs Up/Down, Export ────────────────────────
    _render_action_toolbar(response, message_index)

    # ── Similar Questions Pills ───────────────────────────────────────────────
    _render_similar_questions(message_index)


def _render_expandable_sources(citations: list[LegalCitation], message_index: int = 0) -> None:
    """Render collapsible statutory sources panel."""
    from app.components.citation_card import render_citation_card

    with st.expander(f"📚 {len(citations)} Statutory Citation{'s' if len(citations) != 1 else ''}", expanded=False):
        for i, citation in enumerate(citations):
            render_citation_card(citation, index=i + 1, message_index=message_index)


def _render_action_toolbar(response: LegalRAGResponse, index: int) -> None:
    """Render action buttons: Copy, Thumbs Up/Down."""
    col_copy, col_up, col_down, col_space = st.columns([1.2, 1.2, 1.2, 8])

    thumbs_val = AppState.get_thumbs(index)

    with col_copy:
        if st.button("📋 Copy", key=f"btn_copy_{index}", help="Copy answer text"):
            escaped_text = response.answer.replace("`", "'").replace("\n", "\\n")[:4000]
            st.components.v1.html(
                f"<script>navigator.clipboard.writeText(`{escaped_text}`);</script>",
                height=0,
            )
            st.toast("Copied to clipboard!", icon="✅")

    with col_up:
        up_style = "primary" if thumbs_val == "up" else "secondary"
        if st.button("👍", key=f"btn_up_{index}", type=up_style, help="Helpful"):
            AppState.set_thumbs(index, "up" if thumbs_val != "up" else None)
            st.rerun()

    with col_down:
        down_style = "primary" if thumbs_val == "down" else "secondary"
        if st.button("👎", key=f"btn_down_{index}", type=down_style, help="Not helpful"):
            AppState.set_thumbs(index, "down" if thumbs_val != "down" else None)
            st.rerun()


def _render_similar_questions(index: int) -> None:
    """Render 2 similar question pill buttons."""
    st.markdown('<div style="font-size:0.72rem; color:var(--text-dim); margin-top:0.6rem; margin-bottom:0.25rem;">Suggested Follow-up Questions:</div>', unsafe_allow_html=True)
    q1, q2 = _SIMILAR_QUESTIONS[index % len(_SIMILAR_QUESTIONS)], _SIMILAR_QUESTIONS[(index + 1) % len(_SIMILAR_QUESTIONS)]

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(f"🔍 {q1[:42]}…", key=f"sim_q1_{index}", use_container_width=True):
            AppState.set_pending_question(q1)
            st.rerun()
    with col2:
        if st.button(f"🔍 {q2[:42]}…", key=f"sim_q2_{index}", use_container_width=True):
            AppState.set_pending_question(q2)
            st.rerun()
