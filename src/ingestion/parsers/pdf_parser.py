"""
ingestion.parsers.pdf_parser
=============================
Production-grade PDF parser for Indian legal acts.

Strategy (in priority order)
-----------------------------
1. **pdfplumber** — best for structured PDFs with selectable text (most
   official gazette PDFs fall here).  Extracts text per-page with layout
   awareness.
2. **PyMuPDF (fitz)** — faster fallback; used when pdfplumber yields empty
   pages.  Also handles encrypted PDFs and some scanned layouts.
3. **OCR via pytesseract** — last resort for scanned / image-only PDFs.
   Activated automatically when text layers are absent.

Dependencies (already installed)
---------------------------------
    pdfplumber, PyMuPDF (fitz), pytesseract, pdf2image, Pillow

Usage
-----
    from ingestion.parsers.pdf_parser import PDFParser

    parser = PDFParser()
    result = parser.parse("Data_Set/citizenship_act.pdf")
    print(result.document.metadata.title)
    print(result.document.word_count)
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.models.document import DocumentType, LegalDomain
from ingestion.parsers.base import BaseDocumentParser, DocumentParseError

logger = logging.getLogger(__name__)

# Lazy imports — avoids startup cost when only other parsers are used
_pdfplumber: object | None = None
_fitz: object | None = None


def _get_pdfplumber():  # type: ignore[return]
    global _pdfplumber  # noqa: PLW0603
    if _pdfplumber is None:
        try:
            import pdfplumber as _pp
            _pdfplumber = _pp
        except ImportError:
            _pdfplumber = False
    return _pdfplumber if _pdfplumber is not False else None


def _get_fitz():  # type: ignore[return]
    global _fitz  # noqa: PLW0603
    if _fitz is None:
        try:
            import fitz as _f
            _fitz = _f
        except ImportError:
            _fitz = False
    return _fitz if _fitz is not False else None


# Minimum characters per page to consider the text layer usable
_MIN_TEXT_CHARS_PER_PAGE = 30
# OCR DPI for scanned documents
_OCR_DPI = 300


class PDFParser(BaseDocumentParser):
    """
    Multi-strategy PDF parser.

    Args:
        ocr_enabled:    Allow OCR fallback for scanned pages (default True).
        ocr_language:   Tesseract language code (default ``"eng"``).
        min_text_chars: Minimum extracted chars per page before declaring
                        it a scanned page (default 30).
        extractor:      Metadata extractor (defaults to regex singleton).
        legal_domain:   Override domain detection.
        document_type:  Override document type detection.
    """

    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf"})

    def __init__(
        self,
        ocr_enabled: bool = True,
        ocr_language: str = "eng",
        min_text_chars: int = _MIN_TEXT_CHARS_PER_PAGE,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._ocr_enabled = ocr_enabled
        self._ocr_language = ocr_language
        self._min_text_chars = min_text_chars

    # ── Core extraction ────────────────────────────────────────────────────

    def _extract_text(self, path: Path) -> tuple[str, dict[int, str]]:
        """
        Extract text from a PDF using the best available strategy.

        Returns:
            (full_text, page_map) where page_map is {page_no: text}.
        """
        # --- Strategy 1: pdfplumber ---
        page_map = self._try_pdfplumber(path)

        # --- Strategy 2: PyMuPDF for empty/failed pages ---
        # --- Strategy 1: pdfplumber ---
        page_map = self._try_pdfplumber(path)

        # --- Strategy 2: PyMuPDF for empty/failed pages ---
        if self._has_empty_pages(page_map):
            logger.debug("pdfplumber incomplete for %s, trying PyMuPDF", path)
            fitz_map = self._try_fitz(path)
            # Merge: prefer pdfplumber where it succeeded
            for page_no, text in fitz_map.items():
                if page_no not in page_map or len(page_map[page_no].strip()) < self._min_text_chars:
                    page_map[page_no] = text

        # --- Strategy 3: OCR for remaining empty pages ---
        if self._ocr_enabled and self._has_empty_pages(page_map):
            logger.info("Text layer absent in %s, starting Tesseract OCR...", path)
            ocr_map = self._try_ocr(path, skip_pages=set(page_map.keys()))
            page_map.update(ocr_map)

        if not page_map:
            raise DocumentParseError(path, "all extraction strategies failed")

        full_text = self._assemble_full_text(page_map)
        return full_text, page_map

    # ── Strategy implementations ───────────────────────────────────────────

    def _try_pdfplumber(self, path: Path) -> dict[int, str]:
        """Extract text using pdfplumber (best for native PDFs)."""
        pp = _get_pdfplumber()
        if pp is None:
            logger.warning("pdfplumber is not available")
            return {}

        page_map: dict[int, str] = {}
        try:
            with pp.open(str(path)) as pdf:  # type: ignore[attr-defined]
                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                        # Also extract tables as text rows
                        tables = page.extract_tables() or []
                        for table in tables:
                            for row in table:
                                if row:
                                    text += "\n" + " | ".join(
                                        cell or "" for cell in row
                                    )
                        page_map[i] = text.strip()
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("pdfplumber page %d error in %s: %s", i, path.name, exc)
                        page_map[i] = ""
        except Exception as exc:
            logger.warning("pdfplumber failed for %s: %s", path, exc)

        return page_map

    def _try_fitz(self, path: Path) -> dict[int, str]:
        """Extract text using PyMuPDF (faster, handles more PDF variants)."""
        fitz = _get_fitz()
        if fitz is None:
            logger.warning("pymupdf is not available")
            return {}

        page_map: dict[int, str] = {}
        try:
            doc = fitz.open(str(path))  # type: ignore[attr-defined]
            for i, page in enumerate(doc, start=1):
                try:
                    # "blocks" mode preserves reading order better than raw
                    blocks = page.get_text("blocks", sort=True)  # type: ignore[attr-defined]
                    text = "\n".join(
                        b[4].strip() for b in blocks if b[4].strip()
                    )
                    page_map[i] = text
                except Exception as exc:  # noqa: BLE001
                    logger.debug("fitz page %d error in %s: %s", i, path.name, exc)
                    page_map[i] = ""
            doc.close()
        except Exception as exc:
            logger.warning("fitz failed for %s: %s", path, exc)

        return page_map

    def _try_ocr(self, path: Path, skip_pages: set[int]) -> dict[int, str]:
        """
        OCR scanned pages using pytesseract + pdf2image (or pypdfium2 fallback).

        Only processes pages NOT already in ``skip_pages`` with enough text.
        """
        try:
            import pytesseract  # noqa: PLC0415
        except ImportError:
            logger.warning("ocr_deps_missing: pytesseract not installed")
            return {}

        images = []
        try:
            from pdf2image import convert_from_path  # noqa: PLC0415
            images = convert_from_path(
                str(path),
                dpi=_OCR_DPI,
                fmt="jpeg",
            )
        except Exception as exc:
            logger.debug("pdf2image failed for %s (%s), trying pypdfium2 fallback", path.name, exc)
            try:
                import pypdfium2 as pdfium  # noqa: PLC0415
                pdf = pdfium.PdfDocument(str(path))
                scale = _OCR_DPI / 72.0
                images = [page.render(scale=scale).to_pil() for page in pdf]
            except Exception as exc2:
                logger.warning("Both pdf2image and pypdfium2 failed for %s: %s", path.name, exc2)
                return {}

        page_map: dict[int, str] = {}
        for i, img in enumerate(images, start=1):
            if i in skip_pages:
                # Skip pages that already have adequate text
                continue
            try:
                text = pytesseract.image_to_string(
                    img,
                    lang=self._ocr_language,
                    config="--psm 6",  # Assume uniform block of text
                )
                page_map[i] = text.strip()
                logger.debug("OCR page %d completed (%d chars)", i, len(text))
            except Exception as exc:  # noqa: BLE001
                logger.warning("OCR page %d failed in %s: %s", i, path.name, exc)
                page_map[i] = ""

        return page_map

    # ── Helpers ────────────────────────────────────────────────────────────

    def _has_empty_pages(self, page_map: dict[int, str]) -> bool:
        """Return True if any page has fewer chars than the minimum threshold."""
        if not page_map:
            return True
        return any(
            len(text.strip()) < self._min_text_chars
            for text in page_map.values()
        )

    @staticmethod
    def _assemble_full_text(page_map: dict[int, str]) -> str:
        """Join page texts in order, separated by form-feed characters."""
        pages_sorted = sorted(page_map.items())
        return "\f".join(text for _, text in pages_sorted if text.strip())
