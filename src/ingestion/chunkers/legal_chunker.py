"""
ingestion.chunkers.legal_chunker
================================
Structure-aware legal text chunker.

Combines legal domain awareness (splitting on Section, Chapter, and Act boundaries)
with recursive character splitting to ensure chunks stay within specified token/character bounds
while preserving semantic legal context.

Features:
- Splits along statutory boundaries (Section headers, Chapter titles).
- Respects chunk size and chunk overlap configured in application settings.
- Enriches each chunk's metadata with section-level legal metadata (Act, Section, Chapter, Penalty, Keywords, Authorities).
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import TYPE_CHECKING

from core.config import get_settings
from core.models.document import Chunk, Document, DocumentMetadata
from ingestion.chunkers.base import BaseChunker
from ingestion.metadata import MetadataExtractor, get_default_extractor

if TYPE_CHECKING:
    from core.models.metadata import LegalSectionMetadata

logger = logging.getLogger(__name__)

# Default legal structural split separators in priority order
LEGAL_SEPARATORS = [
    r"\n(?=CHAPTER\s+[IVXLCDM\d]+)",            # Chapter boundaries
    r"\n(?=Section\s+\d+[A-Z]?)",               # Section boundaries e.g. Section 5
    r"\n(?=\d+[A-Z]?\.\s+[A-Z])",               # Numbered section e.g. 5. Citizenship by birth.-
    r"\n(?=\([a-z0-9]+\)\s+[A-Z])",             # Sub-sections e.g. (1) A person born in...
    "\n\n",                                     # Paragraph breaks
    "\n",                                       # Line breaks
    " ",                                        # Words
    "",                                         # Fallback characters
]


class LegalChunker(BaseChunker):
    """
    Structure-aware chunker designed for Indian legal statutes, acts, and rules.

    Args:
        chunk_size: Maximum character length per chunk (default from settings).
        chunk_overlap: Overlap character length between chunks (default from settings).
        extractor: MetadataExtractor instance to extract section metadata per chunk.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        extractor: MetadataExtractor | None = None,
    ) -> None:
        cfg = get_settings()
        self.chunk_size = chunk_size or cfg.ingestion.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.ingestion.chunk_overlap
        self.extractor = extractor or get_default_extractor()

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a Document into Chunks with legal structural preservation.

        Args:
            document: Document entity to split.

        Returns:
            List of Chunk entities with updated metadata and character offsets.
        """
        raw_text = document.content
        if not raw_text.strip():
            logger.warning("Empty document content for doc_id=%s", document.id)
            return []

        # 1. Structural section splitting
        raw_segments = self._split_by_legal_sections(raw_text)

        # 2. Refine large segments recursively if they exceed chunk_size
        final_segments: list[tuple[str, int, int]] = []  # (text, start_char, end_char)
        for seg_text, start_char, end_char in raw_segments:
            if len(seg_text) <= self.chunk_size:
                final_segments.append((seg_text, start_char, end_char))
            else:
                sub_segs = self._recursive_split(seg_text, start_char)
                final_segments.extend(sub_segs)

        # 3. Create Chunk objects and enrich metadata
        chunks: list[Chunk] = []
        for idx, (chunk_text, start_char, end_char) in enumerate(final_segments):
            # Extract section metadata for this specific chunk text
            sec_meta: LegalSectionMetadata = self.extractor.extract_section(chunk_text)

            # Augment parent document metadata with section-level details
            chunk_metadata_extra = dict(document.metadata.extra)
            chunk_metadata_extra.update({
                "act_name": sec_meta.act_name or document.metadata.title,
                "act_year": sec_meta.act_year or document.metadata.year,
                "chapter_number": sec_meta.chapter_number,
                "chapter_name": sec_meta.chapter_name,
                "section_number": sec_meta.section_number,
                "section_title": sec_meta.section_title,
                "sub_section": sec_meta.sub_section,
                "page_number": sec_meta.page_number,
                "keywords": sec_meta.keywords,
                "has_penalty": sec_meta.has_penalty,
                "penalty_type": sec_meta.penalty.penalty_type.value if sec_meta.penalty else None,
                "fine_amount": sec_meta.penalty.fine_amount if sec_meta.penalty else None,
                "imprisonment_years": sec_meta.penalty.imprisonment_years if sec_meta.penalty else None,
                "related_sections": sec_meta.related_sections,
                "authorities": [a.name for a in sec_meta.authorities],
            })

            enriched_metadata = DocumentMetadata(
                source_file=document.metadata.source_file,
                title=sec_meta.full_section_id if sec_meta.section_number else document.metadata.title,
                legal_domain=document.metadata.legal_domain,
                document_type=document.metadata.document_type,
                year=sec_meta.act_year or document.metadata.year,
                jurisdiction=document.metadata.jurisdiction,
                language=document.metadata.language,
                tags=list(set(document.metadata.tags + sec_meta.keywords)),
                extra=chunk_metadata_extra,
            )

            chunk = Chunk(
                id=str(uuid.uuid4()),
                document_id=document.id,
                content=chunk_text,
                metadata=enriched_metadata,
                chunk_index=idx,
                start_char=start_char,
                end_char=end_char,
            )
            chunks.append(chunk)

        logger.info(
            "Created %d chunks for doc_id=%s (title=%r)",
            len(chunks),
            document.id,
            document.metadata.title,
        )
        return chunks

    def _split_by_legal_sections(self, text: str) -> list[tuple[str, int, int]]:
        """Split text along section or chapter regex boundaries."""
        # Combined section boundary pattern
        section_pattern = re.compile(
            r"(?=(?:^|\n)(?:CHAPTER\s+[IVXLCDM\d]+|\d+[A-Z]?\.\s+[A-Z]|Section\s+\d+))",
            re.IGNORECASE | re.MULTILINE,
        )

        matches = list(section_pattern.finditer(text))
        if not matches:
            return [(text, 0, len(text))]

        segments: list[tuple[str, int, int]] = []
        indices = [m.start() for m in matches]
        if indices[0] > 0:
            segments.append((text[: indices[0]], 0, indices[0]))

        for i in range(len(indices)):
            start = indices[i]
            end = indices[i + 1] if i + 1 < len(indices) else len(text)
            segment_text = text[start:end]
            if segment_text.strip():
                segments.append((segment_text, start, end))

        return segments

    def _recursive_split(self, text: str, base_offset: int) -> list[tuple[str, int, int]]:
        """Recursively split text to fit within chunk_size with chunk_overlap."""
        sub_chunks: list[tuple[str, int, int]] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            if end < text_len:
                # Try to break at a natural separator (newline or space)
                break_point = text.rfind("\n", start, end)
                if break_point == -1 or break_point <= start:
                    break_point = text.rfind(" ", start, end)
                if break_point > start:
                    end = break_point

            chunk_str = text[start:end]
            if chunk_str.strip():
                sub_chunks.append((chunk_str, base_offset + start, base_offset + end))

            if end >= text_len:
                break
            start = max(end - self.chunk_overlap, start + 1)

        return sub_chunks
