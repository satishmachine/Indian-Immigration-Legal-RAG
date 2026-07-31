"""
ingestion.chunkers.custom_legal_chunker
========================================
Production Custom Statutory Legal Chunking Engine.

DOES NOT USE RecursiveCharacterTextSplitter.

Follows strict Indian statutory hierarchy:
    Act → Chapter → Section → Subsection → Explanation → Illustration → Exception → Proviso

Guarantees:
- **No legal section is ever split**: Each Section (with all its sub-elements intact) forms an atomic, complete chunk.
- **Hierarchical Statutory Parsing**: Recursively parses section body into sub-elements (Subsections, Explanations, Illustrations, Exceptions, Provisos).
- **Metadata Persistence**: Rich LegalSectionMetadata is compiled and attached to every Chunk entity.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from core.models.document import Chunk, Document, DocumentMetadata
from core.models.metadata import LegalSectionMetadata
from ingestion.chunkers.base import BaseChunker
from ingestion.metadata import MetadataExtractor, get_default_extractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AST Structural Nodes for Statutory Elements
# ---------------------------------------------------------------------------


@dataclass
class StatutorySectionNode:
    """AST Node representing a single unbroken Legal Section."""

    act_name: str | None = None
    act_year: int | None = None
    chapter_number: str | None = None
    chapter_name: str | None = None
    section_number: str | None = None
    section_title: str | None = None
    full_text: str = ""
    subsections: list[str] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)
    illustrations: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    provisos: list[str] = field(default_factory=list)
    start_char: int = 0
    end_char: int = 0


class CustomLegalChunker(BaseChunker):
    """
    Statutory Legal Chunking Engine.

    Parses legal text along statutory hierarchy (Act -> Chapter -> Section -> Sub-elements)
    without splitting individual legal sections across chunk boundaries.

    Args:
        extractor: MetadataExtractor instance (defaults to RegexMetadataExtractor singleton).
    """

    def __init__(self, extractor: MetadataExtractor | None = None) -> None:
        self.extractor = extractor or get_default_extractor()

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Convert a Document into atomic Section Chunks with attached metadata.

        Args:
            document: Document entity.

        Returns:
            List of Chunk entities.
        """
        raw_text = document.content
        if not raw_text.strip():
            return []

        # 1. Parse Document AST into StatutorySectionNodes
        nodes = self._parse_statutory_ast(raw_text, document)

        # 2. Convert StatutorySectionNodes to Chunk entities with metadata
        chunks: list[Chunk] = []
        for idx, node in enumerate(nodes):
            sec_meta: LegalSectionMetadata = self.extractor.extract_section(node.full_text)

            # Merge node attributes with extracted metadata
            act_name = node.act_name or sec_meta.act_name or document.metadata.title
            act_year = node.act_year or sec_meta.act_year or document.metadata.year
            chapter_number = node.chapter_number or sec_meta.chapter_number
            chapter_name = node.chapter_name or sec_meta.chapter_name
            section_number = node.section_number or sec_meta.section_number
            section_title = node.section_title or sec_meta.section_title

            # Extra payload metadata
            meta_extra = dict(document.metadata.extra)
            meta_extra.update({
                "act_name": act_name,
                "act_year": act_year,
                "chapter_number": chapter_number,
                "chapter_name": chapter_name,
                "section_number": section_number,
                "section_title": section_title,
                "sub_sections": node.subsections,
                "explanations": node.explanations or [e for e in sec_meta.explanations],
                "illustrations": node.illustrations or [i for i in sec_meta.illustrations],
                "exceptions": node.exceptions or [ex for ex in sec_meta.exceptions],
                "provisos": node.provisos or [p for p in sec_meta.provisos],
                "has_explanation": len(node.explanations) > 0,
                "has_illustration": len(node.illustrations) > 0,
                "has_exception": len(node.exceptions) > 0,
                "has_proviso": len(node.provisos) > 0,
                "has_penalty": sec_meta.has_penalty,
                "penalty_type": sec_meta.penalty.penalty_type.value if sec_meta.penalty else None,
                "fine_amount": sec_meta.penalty.fine_amount if sec_meta.penalty else None,
                "imprisonment_years": sec_meta.penalty.imprisonment_years if sec_meta.penalty else None,
                "keywords": sec_meta.keywords,
                "authorities": [a.name for a in sec_meta.authorities],
            })

            title = (
                f"{act_name} — Section {section_number}"
                if section_number
                else document.metadata.title
            )

            chunk_metadata = DocumentMetadata(
                source_file=document.metadata.source_file,
                title=title,
                legal_domain=document.metadata.legal_domain,
                document_type=document.metadata.document_type,
                year=act_year,
                jurisdiction=document.metadata.jurisdiction,
                language=document.metadata.language,
                tags=list(set(document.metadata.tags + sec_meta.keywords)),
                extra=meta_extra,
            )

            chunk = Chunk(
                id=str(uuid.uuid4()),
                document_id=document.id,
                content=node.full_text,
                metadata=chunk_metadata,
                chunk_index=idx,
                start_char=node.start_char,
                end_char=node.end_char,
            )
            chunks.append(chunk)

        logger.info(
            "CustomLegalChunker generated %d atomic unbroken Section chunks for %r",
            len(chunks),
            document.metadata.title,
        )
        return chunks

    def _parse_statutory_ast(self, text: str, document: Document) -> list[StatutorySectionNode]:
        """
        Parse raw text into hierarchical statutory nodes: Act -> Chapter -> Section -> Sub-elements.
        Guarantees that no section is broken.
        """
        # First extract overall Act details
        act_name, act_year = self.extractor._extract_act_name_year(text[:2000])

        # Find all Chapter boundaries
        chapter_regex = re.compile(
            r"(?=(?:^|\n)\s*CHAPTER\s+(?P<chap_num>[IVXLCDM\d]+)(?:\s*[:\-–\s]\s*(?P<chap_title>[^\n]+))?)",
            re.IGNORECASE | re.MULTILINE,
        )

        # Regex for Section headers
        # Matches e.g. "Section 5. Citizenship by birth.-" or "5. Citizenship by birth.-" or "Section 12."
        section_regex = re.compile(
            r"(?:^|\n)\s*(?:Section\s+)?(?P<sec_num>\d+[A-Z]?)\.\s*(?P<sec_title>[A-Z][^\n—\.-]{2,100})?(?:\.|\s*[\.—\-])",
            re.MULTILINE,
        )

        section_matches = list(section_regex.finditer(text))

        if not section_matches:
            # Fallback for preamble or single section document
            node = StatutorySectionNode(
                act_name=act_name or document.metadata.title,
                act_year=act_year or document.metadata.year,
                full_text=text.strip(),
                start_char=0,
                end_char=len(text),
            )
            self._parse_sub_elements(node)
            return [node]

        nodes: list[StatutorySectionNode] = []
        current_chapter_num: str | None = None
        current_chapter_title: str | None = None

        # Build Section nodes (unbroken boundaries)
        for i, match in enumerate(section_matches):
            start = match.start()
            end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(text)
            section_raw_text = text[start:end].strip()

            # Check if a Chapter heading precedes this section
            chapter_search = chapter_regex.search(text[:start])
            if chapter_search:
                chap_match = list(chapter_regex.finditer(text[:start]))[-1]
                current_chapter_num = chap_match.group("chap_num")
                current_chapter_title = chap_match.group("chap_title")

            sec_num = match.group("sec_num")
            sec_title = match.group("sec_title")

            node = StatutorySectionNode(
                act_name=act_name,
                act_year=act_year,
                chapter_number=current_chapter_num,
                chapter_name=current_chapter_title,
                section_number=sec_num,
                section_title=sec_title.strip() if sec_title else None,
                full_text=section_raw_text,
                start_char=start,
                end_char=end,
            )

            # Extract internal sub-elements (Subsections, Explanations, Illustrations, Exceptions, Provisos)
            self._parse_sub_elements(node)
            nodes.append(node)

        return nodes

    def _parse_sub_elements(self, node: StatutorySectionNode) -> None:
        """
        Deconstruct a Section's body text into its statutory sub-elements:
        Subsection -> Explanation -> Illustration -> Exception -> Proviso
        """
        text = node.full_text

        # 1. Provisos ("Provided that...", "Provided further that...")
        proviso_regex = re.compile(
            r"(?:Provided\s+(?:further|also)?\s*that[^\.\n;]+[\.\n;])",
            re.IGNORECASE,
        )
        node.provisos = [p.strip() for p in proviso_regex.findall(text)]

        # 2. Explanations ("Explanation.—...", "Explanation 1.—...")
        explanation_regex = re.compile(
            r"(?:Explanation(?:\s+\d+)?[\.\s—\-]+[^\n]+(?:\n[^\n]+){0,5})",
            re.IGNORECASE,
        )
        node.explanations = [e.strip() for e in explanation_regex.findall(text)]

        # 3. Illustrations ("Illustration.—...", "Illustration (a)...")
        illustration_regex = re.compile(
            r"(?:Illustration[s]?(?:\s*\([a-z]\)|\s+\d+)?[\.\s—\-]+[^\n]+(?:\n[^\n]+){0,5})",
            re.IGNORECASE,
        )
        node.illustrations = [i.strip() for i in illustration_regex.findall(text)]

        # 4. Exceptions ("Exception.—...", "Exception 1.—...")
        exception_regex = re.compile(
            r"(?:Exception(?:\s+\d+)?[\.\s—\-]+[^\n]+(?:\n[^\n]+){0,5})",
            re.IGNORECASE,
        )
        node.exceptions = [ex.strip() for ex in exception_regex.findall(text)]

        # 5. Subsections ("(1)...", "(2)...", "(a)...")
        subsection_regex = re.compile(
            r"^\s*\((?P<sub>\d+|[a-z]|[ivx]+)\)\s+(?P<text>[^\n]+)",
            re.MULTILINE,
        )
        node.subsections = [
            f"({m.group('sub')}) {m.group('text').strip()}"
            for m in subsection_regex.finditer(text)
        ]
