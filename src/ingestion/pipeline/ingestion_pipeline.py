"""
ingestion.pipeline.ingestion_pipeline
======================================
Production RAG Document Ingestion Pipeline for Indian Immigration Legal Assistant.

Coordinates:
1. Document scanning in target corpus directory (`Data_Set`).
2. Document Parsing & Text Extraction (`PDFParser`, `DOCXParser`, `TXTParser`).
3. Metadata extraction & statutory legal structure analysis (`RegexMetadataExtractor`).
4. Structure-aware Legal Chunking (`LegalChunker`).
5. Dense Vector Embedding (`EmbeddingService`).
6. Vector Database Indexing & Payload Persistence (`QdrantVectorStore`).

Usage
-----
    from ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    metrics = pipeline.run()
    print(metrics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from core.config import get_settings
from core.interfaces.interfaces import EmbeddingService, VectorStore
from core.models.document import Chunk, Document
from ingestion.chunkers import BaseChunker, CustomLegalChunker
from ingestion.parsers import ParserRegistry, get_parser
from services.embedding import get_embedding_service
from services.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class IngestionMetrics:
    """Ingestion execution statistics."""

    total_files_found: int = 0
    files_processed: int = 0
    files_failed: int = 0
    total_documents: int = 0
    total_chunks: int = 0
    total_words: int = 0
    total_vectors_indexed: int = 0
    errors: list[str] = field(default_factory=list)
    processed_files: list[str] = field(default_factory=list)


class IngestionPipeline:
    """
    End-to-end Legal Document Ingestion Pipeline.

    Args:
        document_dir: Directory containing PDFs/documents to ingest.
        chunker: Custom chunker (defaults to CustomLegalChunker).
        embedding_service: Custom embedding service (defaults to get_embedding_service()).
        vector_store: Custom vector store (defaults to get_vector_store()).
        batch_size: Chunks per embed/upsert batch.
    """

    def __init__(
        self,
        document_dir: str | Path | None = None,
        chunker: BaseChunker | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        batch_size: int | None = None,
    ) -> None:
        cfg = get_settings()
        self.document_dir = Path(document_dir or cfg.directories.document_dir).resolve()
        self.chunker = chunker or CustomLegalChunker()
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()
        self.batch_size = batch_size or cfg.ingestion.batch_size

    def run(self, recreate_collection: bool = False) -> IngestionMetrics:
        """
        Run end-to-end ingestion pipeline on target document directory.

        Args:
            recreate_collection: If True, drops and recreates vector collection before indexing.

        Returns:
            IngestionMetrics summary object.
        """
        metrics = IngestionMetrics()

        if not self.document_dir.exists():
            msg = f"Document directory '{self.document_dir}' does not exist."
            logger.error(msg)
            metrics.errors.append(msg)
            return metrics

        # 1. Discover files
        supported_extensions = set(ParserRegistry.supported_extensions())
        files = [
            f for f in self.document_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        metrics.total_files_found = len(files)
        logger.info("Found %d supported document files in %s", len(files), self.document_dir)

        if not files:
            logger.warning("No supported files found in %s", self.document_dir)
            return metrics

        # 2. Recreate collection if requested
        if recreate_collection:
            logger.info("Recreating vector store collection as requested...")
            self.vector_store.delete_collection()

        # 3. Process each document file
        for file_path in sorted(files):
            try:
                self._process_single_file(file_path, metrics)
                metrics.files_processed += 1
                metrics.processed_files.append(file_path.name)
            except Exception as exc:  # noqa: BLE001
                metrics.files_failed += 1
                err_msg = f"Failed to ingest '{file_path.name}': {exc}"
                logger.error(err_msg, exc_info=True)
                metrics.errors.append(err_msg)

        logger.info(
            "Ingestion completed: %d/%d files processed successfully. Total chunks: %d, Total vectors: %d.",
            metrics.files_processed,
            metrics.total_files_found,
            metrics.total_chunks,
            metrics.total_vectors_indexed,
        )
        return metrics

    def _process_single_file(self, file_path: Path, metrics: IngestionMetrics) -> None:
        """Parse, chunk, embed, and index a single document file."""
        logger.info("--> Processing file: %s", file_path.name)

        # Parse document
        parser = get_parser(file_path)
        parse_result = parser.parse(file_path)
        doc: Document = parse_result.document
        metrics.total_documents += 1
        metrics.total_words += doc.word_count

        # Split into chunks
        chunks: list[Chunk] = self.chunker.chunk(doc)
        if not chunks:
            logger.warning("No chunks generated for %s", file_path.name)
            return

        metrics.total_chunks += len(chunks)

        # Embed and index in batches
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            texts = [c.content for c in batch_chunks]

            # Generate dense embeddings
            logger.debug("Generating embeddings for batch of %d chunks...", len(batch_chunks))
            embeddings = self.embedding_service.embed_documents(texts)

            # Assign embeddings to chunks
            for chunk, emb in zip(batch_chunks, embeddings, strict=True):
                chunk.embedding = emb

            # Upsert into Qdrant
            upserted_count = self.vector_store.upsert(batch_chunks)
            metrics.total_vectors_indexed += upserted_count

        logger.info("Successfully ingested '%s' (%d chunks, %d words)", file_path.name, len(chunks), doc.word_count)

    @staticmethod
    def load_dataset_chunks() -> list[Chunk]:
        """Parse and chunk all documents in Data_Set/ for BM25 and fallback indexing."""
        data_set_dir = Path(get_settings().directories.document_dir)
        if not data_set_dir.exists():
            return []

        chunker = CustomLegalChunker()
        all_chunks: list[Chunk] = []

        for path in sorted(data_set_dir.glob("*.pdf")):
            try:
                parser = get_parser(path)
                parse_res = parser.parse(path)
                chunks = chunker.chunk(parse_res.document)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning("Failed loading chunks for BM25 from %s: %s", path.name, e)

        return all_chunks
