"""
app.components.sidebar
======================
Interactive Sidebar Navigation with PDF Upload, Live Metrics, Domain Filters, and Conversation History.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from app.state import AppState
from services.vector_store import get_vector_repository

logger = logging.getLogger(__name__)

_LEGAL_DOMAINS = [
    ("All Legal Domains", None),
    ("Immigration Law", "immigration"),
    ("Citizenship Act", "citizenship"),
    ("Passport Act & Rules", "passport"),
    ("Visa Guidelines", "visa"),
    ("Emigration Act", "emigration"),
    ("Foreigners Registration", "foreigners"),
]


def render_sidebar() -> dict[str, str | None]:
    """Render interactive sidebar navigation."""
    with st.sidebar:

        # Brand Header Emblem
        from app.components.header import _get_logo_base64
        logo_src = _get_logo_base64()
        logo_img_html = f'<img src="{logo_src}" style="width:32px; height:32px; object-fit:contain;" alt="BOI Seal">' if logo_src else '<span style="font-size:1.5rem;">⚖️</span>'

        st.markdown(
            f"""
            <div style="padding: 1.1rem 1rem 0.85rem; border-bottom: 1px solid var(--border-subtle); display:flex; align-items:center; gap:0.7rem;">
                {logo_img_html}
                <div>
                    <div style="font-size:0.95rem; font-weight:700; color:var(--text-primary);">Bureau of Immigration</div>
                    <div style="font-size:0.68rem; color:var(--text-dim);">Statutory Legal AI Assistant</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div style="padding: 0.75rem 0.85rem 0.25rem;">', unsafe_allow_html=True)

        # Action Buttons
        col_new, col_clear = st.columns([1, 1])
        with col_new:
            if st.button("✏️ New Chat", key="sidebar_new_chat_btn", use_container_width=True):
                AppState.clear_chat()
                st.rerun()
        with col_clear:
            if st.button("🗑️ Clear", key="sidebar_clear_chat_btn", use_container_width=True):
                AppState.clear_chat()
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # PDF Uploader Section
        st.markdown('<div style="padding: 0.5rem 0.85rem 0.2rem; font-size:0.70rem; font-weight:700; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.08em;">Upload Statutory PDF</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding: 0 0.85rem 0.4rem;">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload Statutory PDF",
                type=["pdf"],
                label_visibility="collapsed",
                key="sidebar_pdf_uploader",
            )
            if uploaded_file:
                _handle_pdf_upload(uploaded_file)
            st.markdown("</div>", unsafe_allow_html=True)

        # Available PDF Documents List
        uploaded_pdfs = AppState.get_uploaded_pdfs()
        dataset_pdfs = ["Citizenship_Act_1955.pdf", "Passport_Act_1967.pdf", "Foreigners_Act_1946.pdf", "Emigration_Act_1983.pdf"]
        all_pdfs = list(dict.fromkeys(uploaded_pdfs + dataset_pdfs))

        if all_pdfs:
            st.markdown('<div style="padding: 0.3rem 0.85rem 0.2rem; font-size:0.70rem; font-weight:700; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.08em;">Statutory PDF Library</div>', unsafe_allow_html=True)
            for pdf_item in all_pdfs[:6]:
                is_selected = AppState.get_selected_pdf() == pdf_item
                btn_icon = "🟢" if is_selected else "📄"
                if st.button(f"{btn_icon}  {pdf_item[:28]}", key=f"view_pdf_btn_{pdf_item}", use_container_width=True):
                    AppState.set_selected_pdf(pdf_item)
                    st.rerun()

        st.markdown('<hr style="border-color:var(--border-subtle); margin:0.4rem 0.85rem;">', unsafe_allow_html=True)

        # Legal Domain Filter
        st.markdown('<div style="padding: 0.3rem 0.85rem 0.2rem; font-size:0.70rem; font-weight:700; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.08em;">Retrieval Domain Scope</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding: 0 0.85rem 0.4rem;">', unsafe_allow_html=True)
            domain_labels = [d[0] for d in _LEGAL_DOMAINS]
            domain_idx = st.selectbox(
                "Legal Domain Scope",
                options=range(len(domain_labels)),
                format_func=lambda i: domain_labels[i],
                index=0,
                key="sidebar_domain_select",
                label_visibility="collapsed",
            )
            selected_domain = _LEGAL_DOMAINS[domain_idx][1]
            st.markdown("</div>", unsafe_allow_html=True)

        # System Metrics Card
        _render_system_metrics_card()

        st.markdown('<hr style="border-color:var(--border-subtle); margin:0.4rem 0.85rem;">', unsafe_allow_html=True)

        # Conversation History
        st.markdown('<div style="padding: 0.3rem 0.85rem 0.2rem; font-size:0.70rem; font-weight:700; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.08em;">Conversation History</div>', unsafe_allow_html=True)
        _render_history_list()

    return {"domain_filter": selected_domain}


def _handle_pdf_upload(uploaded_file: Any) -> None:
    """Save uploaded file, sync to static PDFs, run ingestion pipeline, and store embeddings in Qdrant Cloud."""
    root_dir = Path(__file__).resolve().parents[3]
    uploads_dir = root_dir / "uploads"
    static_pdf_dir = root_dir / "static" / "pdfs"

    uploads_dir.mkdir(exist_ok=True, parents=True)
    static_pdf_dir.mkdir(exist_ok=True, parents=True)

    dest_path = uploads_dir / uploaded_file.name
    static_dest = static_pdf_dir / uploaded_file.name

    file_bytes = uploaded_file.getbuffer()

    with open(dest_path, "wb") as fh:
        fh.write(file_bytes)

    with open(static_dest, "wb") as fh:
        fh.write(file_bytes)

    AppState.add_uploaded_pdf(uploaded_file.name)
    AppState.set_selected_pdf(uploaded_file.name)

    # Ingest document and store embeddings directly in Qdrant Cloud
    try:
        from ingestion.pipeline.ingestion_pipeline import IngestionMetrics, IngestionPipeline
        pipeline = IngestionPipeline(document_dir=uploads_dir)
        metrics = IngestionMetrics()
        pipeline._process_single_file(dest_path, metrics)
        st.toast(f"Saved & stored {metrics.total_vectors_indexed} embeddings in Qdrant Cloud: {uploaded_file.name}", icon="☁️")
    except Exception as exc:
        st.toast(f"Saved PDF {uploaded_file.name} (Ingestion notice: {exc})", icon="ℹ️")


def _render_system_metrics_card() -> None:
    """Render live vector store health status."""
    try:
        repo = get_vector_repository()
        info = repo.get_collection_info()
        points = info.get("points_count", 0)
        is_active = points > 0
    except Exception:
        is_active = False

    status_color = "#10B981" if is_active else "#EF4444"
    status_txt = "Active" if is_active else "Inactive"

    st.markdown(
        f"""
        <div class="sidebar-metrics-card">
            <div class="sidebar-metric-row">
                <span class="sidebar-metric-label">Vector DB</span>
                <span class="sidebar-metric-val" style="color:{status_color}; font-weight:600;">● {status_txt}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_history_list() -> None:
    """Render conversation history list."""
    conversations = AppState.get_conversations()
    current_session = AppState.get_session_id()

    if not conversations:
        st.markdown(
            '<div style="padding:0.4rem 0.85rem; font-size:0.76rem; color:var(--text-dim);">'
            'No history yet.</div>',
            unsafe_allow_html=True,
        )
        return

    import html as html_lib
    for conv in conversations[:12]:
        is_active = conv["session_id"] == current_session
        active_border = "border-left: 2px solid var(--accent); padding-left: 0.5rem; background: var(--bg-hover);" if is_active else ""
        title = html_lib.escape(conv.get("title", "New Chat")[:30])
        st.markdown(
            f'<div style="padding: 0.35rem 0.85rem; font-size: 0.80rem; color: var(--text-muted); cursor: pointer; border-radius: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; {active_border}">'
            f'💬  {title}</div>',
            unsafe_allow_html=True,
        )
