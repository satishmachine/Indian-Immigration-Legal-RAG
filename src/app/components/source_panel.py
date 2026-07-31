"""
app.components.source_panel
============================
Expandable Statutory Sources Panel.
"""

from __future__ import annotations

import streamlit as st

from app.components.citation_card import render_citation_card
from core.models.response import LegalCitation


def render_source_panel(
    citations: list[LegalCitation],
    referenced_sections: list[str] | None = None,
    message_index: int = 0,
) -> None:
    """Render collapsible statutory sources panel."""
    if not citations:
        return

    label = f"📚 Grounded in {len(citations)} Statutory Citation{'s' if len(citations) != 1 else ''}"

    with st.expander(label, expanded=False):
        if referenced_sections:
            sec_badges = " ".join(
                f'<span style="background:rgba(255,255,255,0.06); border:1px solid var(--border); color:var(--accent-text); padding:0.12rem 0.45rem; border-radius:4px; font-size:0.72rem; font-weight:600;">{s}</span>'
                for s in referenced_sections[:8]
            )
            st.markdown(
                f'<div style="margin-bottom:0.65rem; display:flex; flex-wrap:wrap; gap:0.35rem;">{sec_badges}</div>',
                unsafe_allow_html=True,
            )

        for i, citation in enumerate(citations):
            render_citation_card(citation, index=i + 1 + (message_index * 10))
