"""
scripts/delete_embeddings.py
=============================
Deletes the 'legal_documents' collection and wipes all stored vector embeddings from Qdrant.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

print("=" * 70)
print("  Deleting Vector Embeddings from Qdrant Database")
print("=" * 70)

from core.config import get_settings
from services.vector_store import get_vector_repository

cfg = get_settings()
repo = get_vector_repository()
client = repo.get_client()

col_name = cfg.qdrant.collection_name

try:
    if repo.collection_exists(col_name):
        print(f"Collection '{col_name}' found.")
        deleted = repo.delete_collection(col_name)
        if deleted:
            print(f"SUCCESS: Collection '{col_name}' and all its vector embeddings have been deleted!")
        else:
            print(f"Warning: Failed to delete collection '{col_name}'.")
    else:
        print(f"Collection '{col_name}' does not exist or is already empty.")
except Exception as exc:
    print(f"Error while deleting collection '{col_name}': {exc}")

# Verify collection info
info = repo.get_collection_info(col_name)
print(f"Current Collection Info: {info}")
print("=" * 70)
