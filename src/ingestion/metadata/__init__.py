"""
src.ingestion.metadata
======================
Metadata extraction package for the Indian Immigration Legal Assistant.

Public API
----------
    from src.ingestion.metadata import MetadataExtractor          # ABC
    from src.ingestion.metadata import RegexMetadataExtractor     # default impl
    from src.ingestion.metadata import get_default_extractor      # factory

Design
------
* ``MetadataExtractor`` — abstract base class defining the contract.
* ``RegexMetadataExtractor`` — production extractor (regex + heuristics,
  offline, zero external API calls).
* ``get_default_extractor()`` — returns the recommended extractor singleton.
  Use this in pipelines rather than instantiating directly.
"""

from ingestion.metadata.extractor import MetadataExtractor
from ingestion.metadata.regex_extractor import RegexMetadataExtractor

__all__: list[str] = [
    "MetadataExtractor",
    "RegexMetadataExtractor",
    "get_default_extractor",
]

# Module-level singleton — initialised on first call
_default_extractor: MetadataExtractor | None = None


def get_default_extractor() -> MetadataExtractor:
    """
    Return the default (regex-based) metadata extractor singleton.

    The instance is created once and reused.  Thread-safe for read operations
    since ``RegexMetadataExtractor`` is stateless.

    Returns:
        A ``RegexMetadataExtractor`` instance.
    """
    global _default_extractor  # noqa: PLW0603
    if _default_extractor is None:
        _default_extractor = RegexMetadataExtractor()
    return _default_extractor
