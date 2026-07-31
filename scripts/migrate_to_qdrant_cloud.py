"""
scripts/migrate_to_qdrant_cloud.py
==================================
Migrates statutory vector embeddings to Qdrant Cloud.

Usage:
    python scripts/migrate_to_qdrant_cloud.py [--url QDRANT_URL] [--api-key API_KEY] [--recreate]

Or set in .env:
    QDRANT_URL=https://<your-cluster-id>.<region>.cloud.qdrant.io:6333
    QDRANT_API_KEY=your_api_key_here
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add src/ directory to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv

load_dotenv(override=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate statutory vector embeddings to Qdrant Cloud."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("QDRANT_URL"),
        help="Qdrant Cloud URL (e.g. https://xyz.cloud.qdrant.io:6333)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("QDRANT_API_KEY"),
        help="Qdrant Cloud API Key",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=os.environ.get("QDRANT_COLLECTION_NAME", "legal_documents"),
        help="Collection name in Qdrant Cloud",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the Qdrant Cloud collection before indexing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    url = args.url or os.environ.get("QDRANT_URL")
    api_key = args.api_key or os.environ.get("QDRANT_API_KEY")

    print("=" * 72)
    print("  Bureau of Immigration AI -- Qdrant Cloud Migration Runner")
    print("=" * 72)

    if not url and not api_key:
        print("[ERROR] Qdrant Cloud URL or API Key not provided.")
        print("  Please pass --url and --api-key or set QDRANT_URL and QDRANT_API_KEY in .env.")
        print("  Example:")
        print("    python scripts/migrate_to_qdrant_cloud.py --url https://xyz.cloud.qdrant.io:6333 --api-key sk_...")
        sys.exit(1)

    if url:
        os.environ["QDRANT_URL"] = url
    if api_key:
        os.environ["QDRANT_API_KEY"] = api_key

    # Reset cached settings so new env vars take effect
    from core.config import reset_settings, get_settings
    reset_settings()

    cfg = get_settings()
    from services.embedding import get_embedding_service
    from services.vector_store import get_vector_repository
    from ingestion.pipeline.ingestion_pipeline import IngestionPipeline

    emb_service = get_embedding_service()
    repo = get_vector_repository()

    print(f"  Target Qdrant Cloud URL : {cfg.qdrant.url}")
    print(f"  Collection Name         : {cfg.qdrant.collection_name}")
    print(f"  Embedding Provider      : {emb_service.provider_name} ({emb_service.model_name})")
    print(f"  Vector Dimension        : {emb_service.dimension}")
    print("-" * 72)

    print("Connecting to Qdrant Cloud...")
    try:
        info = repo.get_collection_info()
        print(f"[OK] Connection successful! Current Points Count: {info.get('points_count', 0)}")
    except Exception as exc:
        print(f"[INFO] Initial collection info check note: {exc}")

    print("\nStarting fresh document ingestion & embedding to Qdrant Cloud...")
    start_time = time.perf_counter()

    pipeline = IngestionPipeline(embedding_service=emb_service)
    metrics = pipeline.run(recreate_collection=True)

    elapsed = time.perf_counter() - start_time

    print("=" * 72)
    print("  Qdrant Cloud Migration Completed Successfully!")
    print("=" * 72)
    print(f"  Files Processed     : {metrics.files_processed} / {metrics.total_files_found}")
    print(f"  Total Chunks        : {metrics.total_chunks}")
    print(f"  Total Words         : {metrics.total_words}")
    print(f"  Vectors Indexed     : {metrics.total_vectors_indexed}")
    print(f"  Elapsed Time        : {elapsed:.2f} seconds")
    print("=" * 72)
    print("Your legal assistant is now fully connected to Qdrant Cloud!")


if __name__ == "__main__":
    main()
