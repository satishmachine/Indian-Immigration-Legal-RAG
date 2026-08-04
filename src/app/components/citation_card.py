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

    # "Open PDF in New Web Tab" Button via Blob URL (bypasses Chrome data: blocking & Streamlit router)
    from services.pdf_service import build_pdf_viewer_url, resolve_pdf_filename
    import streamlit.components.v1 as components

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

    button_html = f"""
    <style>
    body {{
        margin: 0;
        padding: 0;
        background: transparent;
    }}
    .pdf-btn {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        padding: 0.45rem 0.9rem;
        background: #1E293B;
        border: 1px solid #3B82F6;
        color: #93C5FD;
        border-radius: 6px;
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        cursor: pointer;
        box-sizing: border-box;
        transition: all 0.2s ease;
    }}
    .pdf-btn:hover {{
        background: #2563EB;
        color: #FFFFFF;
        border-color: #60A5FA;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }}
    </style>
    <button onclick="openViewer()" class="pdf-btn">
        <span>📄 Open {html_lib.escape(resolved_pdf)}{sec_label} in New Web Tab</span>
        <span style="font-size: 0.9rem; margin-left: 0.35rem;">↗</span>
    </button>
    <script>
    function openViewer() {{
        const rawUrl = "{target_url}";
        if (rawUrl.startsWith("data:text/html;base64,")) {{
            const b64 = rawUrl.substring("data:text/html;base64,".length);
            const binary = atob(b64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {{
                bytes[i] = binary.charCodeAt(i);
            }}
            const blob = new Blob([bytes], {{type: 'text/html'}});
            const url = URL.createObjectURL(blob);
            window.open(url, '_blank');
        }} else {{
            window.open(rawUrl, '_blank');
        }}
    }}
    </script>
    """
    components.html(button_html, height=44)

