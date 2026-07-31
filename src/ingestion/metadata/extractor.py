"""
src.ingestion.metadata.extractor
=================================
Abstract base class for metadata extractors.

Defines the contract that all concrete extractor implementations must
satisfy.  New extractors (e.g. LLM-based, spaCy NER) implement this
interface and can be swapped without changing callers.

Usage
-----
    from src.ingestion.metadata.extractor import MetadataExtractor
    from src.ingestion.metadata.regex_extractor import RegexMetadataExtractor

    extractor: MetadataExtractor = RegexMetadataExtractor()
    section_meta = extractor.extract_section(text, page_number=5)
    doc_meta = extractor.extract_document(text, source_file="act.pdf")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.models.metadata import LegalDocumentMetadata, LegalSectionMetadata


class MetadataExtractor(ABC):
    """
    Abstract base class for metadata extractors.

    Implementations must be stateless so they can be reused safely across
    threads and document batches.
    """

    # ── Version — override in concrete implementations ────────────────────────
    VERSION: str = "1.0.0"

    # ── Section-level extraction ──────────────────────────────────────────────

    @abstractmethod
    def extract_section(
        self,
        text: str,
        *,
        page_number: int | None = None,
        context_hint: str | None = None,
    ) -> LegalSectionMetadata:
        """
        Extract structured metadata from a single section / chunk of text.

        Args:
            text:         Raw text of the section or chunk.
            page_number:  Page number where this text appears (1-indexed).
            context_hint: Optional surrounding text that may aid extraction
                          (e.g. header text from the previous page).

        Returns:
            A ``LegalSectionMetadata`` instance with all extractable fields
            populated.  Fields that could not be extracted are set to their
            defaults (``None`` or empty list).
        """

    # ── Document-level extraction ─────────────────────────────────────────────

    @abstractmethod
    def extract_document(
        self,
        full_text: str,
        source_file: str | Path,
    ) -> LegalDocumentMetadata:
        """
        Extract document-wide metadata from the complete text of an act.

        Args:
            full_text:   Complete raw text of the document.
            source_file: Path or filename of the source PDF/DOCX.

        Returns:
            A ``LegalDocumentMetadata`` instance aggregated over the entire
            document.
        """

    # ── Batch helper (default implementation) ────────────────────────────────

    def extract_sections_batch(
        self,
        texts: list[str],
        *,
        page_numbers: list[int] | None = None,
    ) -> list[LegalSectionMetadata]:
        """
        Extract metadata from a list of section texts.

        Provides a default sequential implementation.  Override in subclasses
        to add parallelism or caching.

        Args:
            texts:        List of raw section texts.
            page_numbers: Optional corresponding page numbers (same length as
                          ``texts``).  Uses ``None`` for missing entries.

        Returns:
            List of ``LegalSectionMetadata`` in the same order as ``texts``.
        """
        pages = page_numbers or ([None] * len(texts))  # type: ignore[list-item]
        return [
            self.extract_section(text, page_number=page)
            for text, page in zip(texts, pages, strict=False)
        ]

    # ── Utility ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(version={self.VERSION!r})"
