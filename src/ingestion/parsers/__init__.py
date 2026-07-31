"""
ingestion.parsers
=================
Document parser package — public API.

Usage
-----
    from ingestion.parsers import ParserRegistry, get_parser

    # Auto-select parser by file extension
    parser = get_parser("citizenship_act.pdf")
    result = parser.parse("Data_Set/citizenship_act.pdf")

    # Or use a specific parser directly
    from ingestion.parsers import PDFParser, DOCXParser, TXTParser
    pdf_parser = PDFParser(ocr_enabled=True)

Registry
--------
The ``ParserRegistry`` singleton maps file extensions to parser instances.
Register custom parsers via ``ParserRegistry.register()``.
"""

from __future__ import annotations

from pathlib import Path

from ingestion.parsers.base import (
    BaseDocumentParser,
    DocumentParseError,
    ParseResult,
    UnsupportedFileTypeError,
)
from ingestion.parsers.docx_parser import DOCXParser
from ingestion.parsers.pdf_parser import PDFParser
from ingestion.parsers.txt_parser import TXTParser

__all__: list[str] = [
    # Parsers
    "BaseDocumentParser",
    "PDFParser",
    "DOCXParser",
    "TXTParser",
    # Registry
    "ParserRegistry",
    "get_parser",
    # Exceptions
    "DocumentParseError",
    "ParseResult",
    "UnsupportedFileTypeError",
]


class ParserRegistry:
    """
    Registry that maps file extensions to parser instances.

    Maintains a dict of ``{extension: parser_instance}`` and exposes
    ``get()`` / ``register()`` for extensibility.

    This is a class with class-level state (not a singleton instance) so it
    can be used without instantiation: ``ParserRegistry.get(".pdf")``.
    """

    _registry: dict[str, BaseDocumentParser] = {}

    @classmethod
    def register(cls, parser: BaseDocumentParser) -> None:
        """
        Register a parser for all file extensions it declares.

        Args:
            parser: An instance of a ``BaseDocumentParser`` subclass.
        """
        for ext in parser.SUPPORTED_EXTENSIONS:
            cls._registry[ext.lower()] = parser

    @classmethod
    def get(cls, file_path: str | Path) -> BaseDocumentParser:
        """
        Return the registered parser for the given file's extension.

        Args:
            file_path: Path to the document (only the extension is used).

        Returns:
            The matching ``BaseDocumentParser`` instance.

        Raises:
            UnsupportedFileTypeError: If no parser is registered for the ext.
        """
        ext = Path(file_path).suffix.lower()
        if ext not in cls._registry:
            raise UnsupportedFileTypeError(file_path)
        return cls._registry[ext]

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Return all registered file extensions."""
        return sorted(cls._registry.keys())

    @classmethod
    def reset(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._registry.clear()


# ---------------------------------------------------------------------------
# Register defaults
# ---------------------------------------------------------------------------

ParserRegistry.register(PDFParser())
ParserRegistry.register(DOCXParser())
ParserRegistry.register(TXTParser())


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def get_parser(file_path: str | Path) -> BaseDocumentParser:
    """
    Return the appropriate parser for *file_path* from the global registry.

    Args:
        file_path: Path to the document file.

    Returns:
        The matching parser instance.

    Raises:
        UnsupportedFileTypeError: If the file type is not supported.
    """
    return ParserRegistry.get(file_path)
