"""
core.citation
=============
Statutory Citation Engine Package.

Exports:
- CitationDetails
- CitationEngine
"""

from __future__ import annotations

from core.citation.citation_engine import CitationEngine
from core.citation.models import CitationDetails

__all__: list[str] = [
    "CitationDetails",
    "CitationEngine",
]
