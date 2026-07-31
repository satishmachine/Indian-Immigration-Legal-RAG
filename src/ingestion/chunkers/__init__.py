"""
ingestion.chunkers
==================
Chunking module for legal documents.
"""

from ingestion.chunkers.base import BaseChunker
from ingestion.chunkers.custom_legal_chunker import CustomLegalChunker
from ingestion.chunkers.legal_chunker import LegalChunker

__all__: list[str] = [
    "BaseChunker",
    "CustomLegalChunker",
    "LegalChunker",
]
