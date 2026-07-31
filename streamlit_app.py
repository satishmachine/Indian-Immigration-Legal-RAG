"""
streamlit_app.py
================
Modern Streamlit Application for Indian Immigration & Emigration Legal AI.

Features:
- Responsive Wide Layout & Dark Mode Theme
- PDF Uploading, Ingestion & Split PDF Viewer Panel
- Real-time Streaming Legal Responses
- Highlighted Citations & Expandable Sources
- Action Toolbar: Copy Button, Export Answer (JSON), Thumbs Up/Down Feedback
- Similar Question Pills
- Decoupled UI and Clean Architecture
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src/ directory to Python path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streamlit as st

# Configure Responsive Wide Layout
st.set_page_config(
    page_title="Bureau of Immigration – Statutory AI Assistant",
    page_icon="assets/boi_logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.components import (
    render_chat_interface,
    render_pdf_viewer,
    render_sidebar,
)
from app.state import AppState
from app.theme import apply_custom_theme
from chains.legal_rag_chain import LegalRAGChain
from core.models.response import LegalRAGResponse


@st.cache_resource(show_spinner="Checking statutory vector database…")
def ensure_data_set_indexed() -> int:
    """Ensure PDFs in Data_Set are embedded into Qdrant vector store."""
    from services.embedding import get_embedding_service
    from services.vector_store import get_vector_repository

    try:
        repo = get_vector_repository()
        emb_service = get_embedding_service()
        info = repo.get_collection_info()
        points_count = info.get("points_count", 0)

        if points_count == 0:
            try:
                from ingestion.pipeline.ingestion_pipeline import IngestionPipeline
                pipeline = IngestionPipeline(embedding_service=emb_service)
                metrics = pipeline.run(recreate_collection=True)
                return metrics.total_vectors_indexed
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Initial indexing warning: %s", exc)
                return 0
        return points_count
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Vector DB check warning: %s", exc)
        return 0


@st.cache_resource(show_spinner="Initializing Legal RAG Pipeline…")
def get_rag_chain() -> LegalRAGChain:
    """Cached singleton instance of LegalRAGChain pipeline."""
    return LegalRAGChain()


def main() -> None:
    """Main Application Orchestrator."""
    # 1. Initialize State & Dark Theme
    AppState.initialize()
    apply_custom_theme()

    # Sync static PDF files for new web tab viewer
    try:
        from services.pdf_service import ensure_static_pdfs_synced
        ensure_static_pdfs_synced()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Static PDF sync warning: %s", exc)

    # 2. Check Vector DB Status
    ensure_data_set_indexed()

    # 3. Render Sidebar Navigation
    sidebar_state = render_sidebar()
    domain_filter = sidebar_state.get("domain_filter")

    # 4. RAG Query Execution Delegate
    def execute_rag_query(
        question: str,
        session_id: str,
        filters: dict | None,
    ) -> LegalRAGResponse:
        chain = get_rag_chain()
        return chain.query(question=question, session_id=session_id, filters=filters)

    # 5. Check if PDF Viewer is Active
    selected_pdf = AppState.get_selected_pdf()

    if selected_pdf:
        # Split Layout: Chat on Left (7), PDF Viewer on Right (5)
        col_chat, col_pdf = st.columns([7, 5])
        with col_chat:
            render_chat_interface(query_fn=execute_rag_query, domain_filter=domain_filter)
        with col_pdf:
            render_pdf_viewer(selected_pdf)
    else:
        # Standard Centered Layout
        render_chat_interface(query_fn=execute_rag_query, domain_filter=domain_filter)


if __name__ == "__main__":
    main()
