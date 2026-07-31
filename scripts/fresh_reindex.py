"""
scripts/fresh_reindex.py
========================
Wipes local Qdrant database and re-indexes all Data_Set PDFs fresh with 1024-dim BAAI/bge-m3 embeddings.
"""

import os, shutil, sys, time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

QDRANT_DB_DIR = ROOT_DIR / "qdrant_db"

print("=" * 70)
print("  Indian Legal AI Platform — Fresh Embedding & Indexing Runner")
print("=" * 70)

# Step 1: Wipe local ./qdrant_db folder if not locked
if QDRANT_DB_DIR.exists():
    print(f"Removing existing local vector database directory: {QDRANT_DB_DIR}")
    try:
        shutil.rmtree(QDRANT_DB_DIR)
        print("Successfully deleted ./qdrant_db directory.")
    except Exception as err:
        print(f"Note: Could not delete ./qdrant_db directory ({err}). Will recreate collection via Qdrant API.")

# Step 2: Initialize IngestionPipeline and run fresh re-indexing
from core.config import get_settings
from services.embedding import get_embedding_service
from ingestion.pipeline.ingestion_pipeline import IngestionPipeline

cfg = get_settings()
emb = get_embedding_service()

print(f"Embedding Provider : {emb.provider_name} ({emb.model_name})")
print(f"Vector Dimension   : {emb.dimension}")
print(f"Qdrant Collection  : {cfg.qdrant.collection_name}")
print(f"Target Document Dir: {cfg.directories.document_dir}")
print("-" * 70)

start_time = time.perf_counter()
pipeline = IngestionPipeline(embedding_service=emb)
metrics = pipeline.run(recreate_collection=True)
elapsed = time.perf_counter() - start_time

print("=" * 70)
print("  Fresh Re-indexing Execution Completed!")
print("=" * 70)
print(f"  Files Processed   : {metrics.files_processed} / {metrics.total_files_found}")
print(f"  Total Chunks      : {metrics.total_chunks}")
print(f"  Total Words       : {metrics.total_words}")
print(f"  Vectors Indexed   : {metrics.total_vectors_indexed}")
print(f"  Elapsed Time      : {elapsed:.2f} seconds")
print("=" * 70)
