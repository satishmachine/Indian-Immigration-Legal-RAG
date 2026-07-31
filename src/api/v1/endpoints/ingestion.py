"""
api.v1.endpoints.ingestion
==========================
Statutory PDF Upload and Ingestion Endpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from core.config import get_settings
from ingestion.pipeline.ingestion_pipeline import IngestionMetrics, IngestionPipeline

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Ingestion"])


@router.post("/ingest")
def upload_and_ingest_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload and immediately parse, chunk, embed, and index a statutory PDF document."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported for statutory ingestion.",
        )

    try:
        data_set_dir = Path(get_settings().dirs.data_set)
        data_set_dir.mkdir(parents=True, exist_ok=True)
        dest_path = data_set_dir / file.filename

        contents = file.file.read()
        dest_path.write_bytes(contents)

        pipeline = IngestionPipeline()
        metrics = IngestionMetrics()
        pipeline._process_single_file(dest_path, metrics)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": metrics.total_chunks,
            "words_indexed": metrics.total_words,
            "vectors_indexed": metrics.total_vectors_indexed,
        }
    except Exception as exc:
        logger.error("Ingestion upload failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest PDF document: {exc}",
        ) from exc
