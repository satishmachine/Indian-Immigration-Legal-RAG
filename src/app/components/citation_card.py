"""
app.components.citation_card
============================
LexisNexis & Westlaw style statutory citation card renderer.
"""

from __future__ import annotations

import html as html_lib

import streamlit as st

from app.state import AppState
from core.models.response import LegalCitation


def render_citation_card(citation: LegalCitation, index: int = 1, message_index: int = 0) -> None:
    """
    Render LexisNexis / Westlaw style statutory citation card.

    Args:
        citation: LegalCitation domain model object.
        index: Citation display index within message.
        message_index: Message index in chat history.
    """
    score_pct = int(citation.score * 100) if citation.score else 0

    score_badge = (
        f'<span class="legal-citation-badge">{score_pct}% Match</span>'
        if score_pct > 0
        else ""
    )

    sec_title = f" — {html_lib.escape(citation.section_title)}" if citation.section_title else ""
    page_info = f" • Page {citation.page_number}" if citation.page_number else ""
    year_info = f" ({citation.year})" if citation.year else ""

    snippet_html = ""
    if citation.snippet:
        escaped_snippet = html_lib.escape(citation.snippet[:280])
        snippet_html = f'<div class="legal-citation-quote">"{escaped_snippet}…"</div>'

    st.markdown(
        f"""
        <div class="legal-citation-card">
            <div class="legal-citation-header">
                <span class="legal-citation-act">🏛️ {html_lib.escape(citation.act_name)}{year_info}</span>
                {score_badge}
            </div>
            <div class="legal-citation-section">
                Section {html_lib.escape(citation.section_number)}{sec_title}{page_info}
            </div>
            {snippet_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # "Open PDF in New Web Tab" Button
    from services.pdf_service import build_pdf_viewer_url, resolve_pdf_filename

    pdf_filename = getattr(citation, "pdf_name", None)
    target_url = build_pdf_viewer_url(
        act_name=citation.act_name,
        section_number=citation.section_number,
        section_title=citation.section_title,
        page_number=citation.page_number,
        pdf_name=pdf_filename,
    )
    resolved_pdf = resolve_pdf_filename(citation.act_name, pdf_filename)
    sec_label = f" (Sec. {html_lib.escape(citation.section_number)})" if citation.section_number and citation.section_number.upper() != "N/A" else ""

    st.markdown(
        f"""
        <div style="margin-top: 0.6rem;">
            <a href="{target_url}" target="_blank" class="pdf-open-newtab-btn">
                <span>📄 Open {html_lib.escape(resolved_pdf)}{sec_label} in New Web Tab</span>
                <span style="font-size: 0.9rem; margin-left: 0.3rem;">↗</span>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
