"""
ingestion.parsers.txt_parser
=============================
Plain-text document parser.

Handles UTF-8 / Latin-1 encoded ``.txt`` files.  Attempts UTF-8 first,
falls back to Latin-1 for legacy gazette text files.

Builds a page map by splitting on form-feed characters (``\\f``) — the
standard page separator in plain-text legal documents exported from PDFs.
Falls back to line-count-based splitting when no form-feeds are present.

Usage
-----
    from ingestion.parsers.txt_parser import TXTParser

    parser = TXTParser()
    result = parser.parse("Data_Set/citizenship_act.txt")
"""

from __future__ import annotations

import logging
from pathlib import Path

from ingestion.parsers.base import BaseDocumentParser, DocumentParseError

logger = logging.getLogger(__name__)

# Fallback lines per page when no form-feeds are present
_LINES_PER_PAGE = 60
# Encodings to try in order
_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


class TXTParser(BaseDocumentParser):
    """
    Parser for plain-text (``.txt``) files.

    Args:
        lines_per_page: Lines per synthetic page when form-feeds are absent.
    """

    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".text"})

    def __init__(self, lines_per_page: int = _LINES_PER_PAGE, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._lines_per_page = lines_per_page

    def _extract_text(self, path: Path) -> tuple[str, dict[int, str]]:
        """Read the text file and build a page map."""
        raw = self._read_with_fallback(path)

        if not raw.strip():
            raise DocumentParseError(path, "file is empty")

        # ── Page map ───────────────────────────────────────────────────────
        if "\f" in raw:
            # Form-feed page breaks — standard for PDFs exported as text
            pages = raw.split("\f")
            page_map = {
                i: page.strip()
                for i, page in enumerate(pages, start=1)
                if page.strip()
            }
        else:
            # No form-feeds — split by line count
            lines = raw.splitlines()
            page_map: dict[int, str] = {}
            for page_no, start in enumerate(
                range(0, len(lines), self._lines_per_page), start=1
            ):
                chunk = "\n".join(lines[start : start + self._lines_per_page]).strip()
                if chunk:
                    page_map[page_no] = chunk

        full_text = raw
        return full_text, page_map

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _read_with_fallback(path: Path) -> str:
        """Try multiple encodings; raise ``DocumentParseError`` if all fail."""
        for encoding in _ENCODINGS:
            try:
                return path.read_text(encoding=encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        raise DocumentParseError(
            path,
            f"could not decode file with any of: {_ENCODINGS}",
        )
