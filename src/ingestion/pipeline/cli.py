"""
ingestion.pipeline.cli
======================
CLI runner for the legal document ingestion pipeline.

Usage
-----
    uv run python -m src.ingestion.pipeline.cli
    # Or with arguments:
    uv run python -m src.ingestion.pipeline.cli --dir Data_Set --recreate
"""

from __future__ import annotations

import argparse
import sys
import time

from core.config import get_settings
from ingestion.pipeline.ingestion_pipeline import IngestionPipeline


def main() -> None:
    """CLI entrypoint for ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest legal documents into vector store (Qdrant)."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Path to directory containing PDF/DOCX/TXT legal documents (default: Data_Set).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate vector store collection before ingestion.",
    )
    args = parser.parse_args()

    cfg = get_settings()
    doc_dir = args.dir or cfg.directories.document_dir

    print("=" * 60)
    print("  Indian Immigration Legal Assistant — Ingestion Pipeline")
    print("=" * 60)
    print(f"  Target Corpus Dir : {doc_dir}")
    print(f"  Qdrant Collection : {cfg.qdrant.collection_name}")
    print(f"  Vector Dimension  : {cfg.qdrant.vector_size}")
    print(f"  Embedding Model   : {cfg.embedding.model_name}")
    print(f"  Recreate Col.     : {args.recreate}")
    print("=" * 60)

    start_time = time.perf_counter()
    pipeline = IngestionPipeline(document_dir=doc_dir)
    metrics = pipeline.run(recreate_collection=args.recreate)
    elapsed = time.perf_counter() - start_time

    print()
    print("=" * 60)
    print("  Ingestion Execution Metrics")
    print("=" * 60)
    print(f"  Files Found       : {metrics.total_files_found}")
    print(f"  Files Processed   : {metrics.files_processed}")
    print(f"  Files Failed      : {metrics.files_failed}")
    print(f"  Total Chunks      : {metrics.total_chunks}")
    print(f"  Total Words       : {metrics.total_words}")
    print(f"  Vectors Indexed   : {metrics.total_vectors_indexed}")
    print(f"  Execution Time    : {elapsed:.2f} seconds")
    if metrics.processed_files:
        print("  Processed Files   :")
        for fname in metrics.processed_files:
            print(f"    - {fname}")
    if metrics.errors:
        print("  Errors            :")
        for err in metrics.errors:
            print(f"    ! {err}")
    print("=" * 60)

    if metrics.files_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
