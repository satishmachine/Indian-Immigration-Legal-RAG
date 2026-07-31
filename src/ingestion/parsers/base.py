"""
ingestion.parsers.base
======================
Abstract base class for all document parsers.

Every concrete parser (PDF, DOCX, TXT, …) must inherit from
``BaseDocumentParser`` and implement ``parse()`` and ``supports()``.

Design rules
------------
* **Stateless** — parsers hold no mutable state; safe for concurrent use.
* **Single responsibility** — each parser handles exactly one file format.
* **Metadata integration** — parsers run ``MetadataExtractor`` after text
  extraction and attach the result to every ``Document``.
* **Graceful degradation** — extraction errors are logged and re-raised as
  ``DocumentParseError``; callers decide whether to skip or abort.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from core.models.document import Document, DocumentMetadata, DocumentStatus, DocumentType, LegalDomain
from core.models.metadata import LegalDocumentMetadata, LegalSectionMetadata
from ingestion.metadata import MetadataExtractor, get_default_extractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class DocumentParseError(RuntimeError):
    """Raised when a parser cannot extract text from a file."""

    def __init__(self, path: str | Path, reason: str) -> None:
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"Failed to parse '{self.path.name}': {reason}")


class UnsupportedFileTypeError(DocumentParseError):
    """Raised when no parser supports the given file type."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path, f"unsupported file extension '{Path(path).suffix}'")


# ---------------------------------------------------------------------------
# Parse result container
# ---------------------------------------------------------------------------


class ParseResult:
    """
    Container returned by every parser's ``parse()`` method.

    Holds the extracted ``Document`` plus optional enriched metadata
    produced during the parse phase.
    """

    __slots__ = ("document", "section_metadata", "doc_metadata", "warnings")

    def __init__(
        self,
        document: Document,
        section_metadata: list[LegalSectionMetadata] | None = None,
        doc_metadata: LegalDocumentMetadata | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.document = document
        self.section_metadata: list[LegalSectionMetadata] = section_metadata or []
        self.doc_metadata: LegalDocumentMetadata | None = doc_metadata
        self.warnings: list[str] = warnings or []

    def __repr__(self) -> str:
        return (
            f"ParseResult(doc_id={self.document.id!r}, "
            f"sections={len(self.section_metadata)}, "
            f"warnings={len(self.warnings)})"
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseDocumentParser(ABC):
    """
    Abstract base class for document parsers.

    Subclasses implement ``_extract_text()`` for their format.
    The common pipeline (hashing, metadata extraction, ``Document``
    construction) lives here so it is not duplicated across parsers.

    Args:
        extractor: Metadata extractor instance.  Defaults to the shared
                   ``RegexMetadataExtractor`` singleton.
        legal_domain: Override the detected legal domain.
        document_type: Override the detected document type.
    """

    #: File extensions this parser handles (lowercase, with dot).
    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset()

    def __init__(
        self,
        extractor: MetadataExtractor | None = None,
        legal_domain: LegalDomain = LegalDomain.UNKNOWN,
        document_type: DocumentType = DocumentType.UNKNOWN,
    ) -> None:
        self._extractor: MetadataExtractor = extractor or get_default_extractor()
        self._legal_domain = legal_domain
        self._document_type = document_type

    # ── Public API ─────────────────────────────────────────────────────────

    def parse(self, file_path: str | Path) -> ParseResult:
        """
        Parse a file and return a ``ParseResult`` containing a ``Document``
        and extracted metadata.

        Args:
            file_path: Absolute or relative path to the source file.

        Returns:
            ``ParseResult`` with the loaded document and metadata.

        Raises:
            DocumentParseError: If text extraction fails.
            UnsupportedFileTypeError: If the file extension is not supported.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not self.supports(path):
            raise UnsupportedFileTypeError(path)

        logger.info("Parsing document %s with %s", path, self.__class__.__name__)

        try:
            raw_text, page_map = self._extract_text(path)
        except DocumentParseError:
            raise
        except Exception as exc:
            raise DocumentParseError(path, str(exc)) from exc

        if not raw_text.strip():
            raise DocumentParseError(path, "extracted text is empty")

        # ── Compute content hash ──────────────────────────────────────────
        content_hash = hashlib.sha256(raw_text.encode()).hexdigest()

        # ── Extract document-level metadata ───────────────────────────────
        warnings: list[str] = []
        doc_meta: LegalDocumentMetadata | None = None
        section_metas: list[LegalSectionMetadata] = []

        try:
            doc_meta = self._extractor.extract_document(raw_text, path)
            # Also collect per-page section metadata for chunk attachment
            if page_map:
                for page_no, page_text in page_map.items():
                    sec_meta = self._extractor.extract_section(
                        page_text, page_number=page_no
                    )
                    section_metas.append(sec_meta)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Metadata extraction failed: {exc}")
            logger.warning("Metadata extraction failed for %s: %s", path, exc)

        # ── Build DocumentMetadata ────────────────────────────────────────
        title = (
            doc_meta.display_title
            if doc_meta and doc_meta.act_name
            else path.stem.replace("_", " ").title()
        )

        metadata = DocumentMetadata(
            source_file=str(path),
            title=title,
            legal_domain=self._detect_domain(raw_text, doc_meta),
            document_type=self._detect_document_type(raw_text, doc_meta),
            year=doc_meta.act_year if doc_meta else None,
            jurisdiction="India",
            language="en",
            tags=list(doc_meta.keywords[:20]) if doc_meta else [],
            extra={
                "content_hash": content_hash,
                "total_pages": doc_meta.total_pages if doc_meta else 0,
                "total_sections": doc_meta.total_sections if doc_meta else 0,
                "act_number": doc_meta.act_number if doc_meta else None,
            },
        )

        document = Document(
            content=raw_text,
            metadata=metadata,
            status=DocumentStatus.COMPLETED,
        )

        logger.info(
            "Parsed document %s (id=%s, title=%r, words=%d, sections=%d)",
            path.name,
            document.id,
            title,
            document.word_count,
            len(section_metas),
        )

        return ParseResult(
            document=document,
            section_metadata=section_metas,
            doc_metadata=doc_meta,
            warnings=warnings,
        )

    def supports(self, file_path: str | Path) -> bool:
        """Return True if this parser handles the given file's extension."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    # ── Abstract methods ───────────────────────────────────────────────────

    @abstractmethod
    def _extract_text(self, path: Path) -> tuple[str, dict[int, str]]:
        """
        Extract raw text from the file.

        Args:
            path: Resolved absolute path to the file.

        Returns:
            Tuple of:
            - ``full_text``: Complete extracted text (all pages joined).
            - ``page_map``:  Dict mapping 1-indexed page numbers to their
                             extracted text.  Empty dict if page-level
                             extraction is not supported.

        Raises:
            DocumentParseError: On extraction failure.
        """

    # ── Private helpers ────────────────────────────────────────────────────

    def _detect_domain(
        self,
        text: str,
        doc_meta: LegalDocumentMetadata | None,
    ) -> LegalDomain:
        """Infer the primary legal domain from keywords and act name."""
        if self._legal_domain != LegalDomain.UNKNOWN:
            return self._legal_domain

        text_lower = text.lower()
        act_name = (doc_meta.act_name or "").lower() if doc_meta else ""
        combined = f"{act_name} {text_lower[:2000]}"

        domain_signals: dict[LegalDomain, list[str]] = {
            LegalDomain.CITIZENSHIP: ["citizenship", "citizen", "naturalisation"],
            LegalDomain.IMMIGRATION: ["immigration", "immigrant", "port of entry"],
            LegalDomain.EMIGRATION: ["emigration", "emigrant", "ecr", "emigration clearance"],
            LegalDomain.PASSPORT: ["passport", "travel document", "passport authority"],
            LegalDomain.VISA: ["visa", "tourist visa", "work visa", "e-visa"],
            LegalDomain.FOREIGNERS: ["foreigner", "foreign national", "frro", "alien"],
        }

        best_domain = LegalDomain.UNKNOWN
        best_score = 0
        for domain, signals in domain_signals.items():
            score = sum(1 for s in signals if s in combined)
            if score > best_score:
                best_score = score
                best_domain = domain

        return best_domain

    def _detect_document_type(
        self,
        text: str,
        doc_meta: LegalDocumentMetadata | None,
    ) -> DocumentType:
        """Infer document type from the document structure."""
        if self._document_type != DocumentType.UNKNOWN:
            return self._document_type

        act_name = (doc_meta.act_name or "").lower() if doc_meta else ""
        text_lower = text[:1000].lower()
        combined = f"{act_name} {text_lower}"

        if "amendment" in combined:
            return DocumentType.AMENDMENT
        if any(w in combined for w in ("rules", "regulation")):
            return DocumentType.RULE
        if "notification" in combined or "circular" in combined:
            return DocumentType.NOTIFICATION
        if "act" in combined:
            return DocumentType.ACT
        return DocumentType.UNKNOWN

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"extensions={sorted(self.SUPPORTED_EXTENSIONS)})"
        )
