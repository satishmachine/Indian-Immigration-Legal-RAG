"""
src.ingestion.metadata.regex_extractor
=======================================
Production-grade regex + heuristic metadata extractor for Indian legal acts.

Implements ``MetadataExtractor`` using the pattern library in
``src.ingestion.metadata.patterns``.

Architecture
------------
* **Stateless** — all state lives in the compiled patterns (module-level).
* **Layered extraction** — each field has a dedicated private method so it
  can be tested in isolation and overridden in subclasses.
* **Confidence scoring** — a lightweight confidence score is computed from
  the ratio of non-None fields to total expected fields.
* **Graceful degradation** — every field extraction is wrapped in a try/except
  so a failure in one field never blocks the rest.

Thread safety
-------------
All instance methods are pure functions over their inputs; no shared mutable
state is modified during extraction.  Safe to call from multiple threads.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from core.models.metadata import (
    AuthorityReference,
    AuthorityType,
    CrossReference,
    DefinitionEntry,
    LegalDocumentMetadata,
    LegalSectionMetadata,
    PenaltyInfo,
    PenaltyType,
)
from ingestion.metadata.extractor import MetadataExtractor
from ingestion.metadata.patterns import (
    AUTHORITY_PATTERNS,
    PAT_ACT_NAME,
    PAT_CHAPTER,
    PAT_CROSS_REF,
    PAT_DEFINITION,
    PAT_PAGE,
    PAT_PENALTY,
    PAT_RELATED_SECTION,
    PAT_SECTION,
    RE_ACT_NUMBER,
    RE_CROSS_REF_EXTERNAL,
    RE_PENALTY_FINE_AMOUNT,
    RE_PENALTY_YEARS,
    RE_SECTION_REF_SAME_ACT,
    RE_SECTION_WITH_TITLE,
    RE_SUB_SECTION,
    ALL_KEYWORD_SEEDS,
    get_keyword_pattern,
)

logger = logging.getLogger(__name__)

# Maximum character length for text passed to extraction sub-methods.
# Very long texts (full acts) are truncated at the header-region heuristic.
_HEADER_CHARS = 2000  # First N chars used for act/chapter detection

# Confidence weight per field (must sum to 1.0)
_CONFIDENCE_WEIGHTS: dict[str, float] = {
    "act_name": 0.20,
    "chapter_number": 0.10,
    "section_number": 0.15,
    "section_title": 0.10,
    "keywords": 0.10,
    "definitions": 0.10,
    "authorities": 0.10,
    "penalty": 0.05,
    "related_sections": 0.05,
    "cross_references": 0.05,
}


class RegexMetadataExtractor(MetadataExtractor):
    """
    Rule-based metadata extractor using compiled regular expressions.

    This is the primary production extractor.  It requires no LLM calls and
    operates fully offline at high throughput.

    Example::

        extractor = RegexMetadataExtractor()

        # Section-level
        meta = extractor.extract_section(chunk_text, page_number=12)
        print(meta.full_section_id)

        # Document-level
        doc_meta = extractor.extract_document(full_text, "citizenship_act.pdf")
        print(doc_meta.display_title)
    """

    VERSION = "1.0.0"

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract_section(
        self,
        text: str,
        *,
        page_number: int | None = None,
        context_hint: str | None = None,
    ) -> LegalSectionMetadata:
        """Extract all metadata fields from a single section text."""
        combined = f"{context_hint}\n{text}" if context_hint else text

        act_name, act_year = self._extract_act_name_year(combined[:_HEADER_CHARS])
        chapter_number, chapter_name = self._extract_chapter(combined[:_HEADER_CHARS])
        section_number, section_title = self._extract_section(text)
        sub_section = self._extract_sub_section(text)
        keywords = self._extract_keywords(text)
        definitions = self._extract_definitions(text)
        authorities = self._extract_authorities(text)
        penalty = self._extract_penalty(text)
        related_sections = self._extract_related_sections(text)
        cross_references = self._extract_cross_references(text)

        # Compute confidence score
        confidence = self._compute_confidence(
            act_name=act_name,
            chapter_number=chapter_number,
            section_number=section_number,
            section_title=section_title,
            keywords=keywords,
            definitions=definitions,
            authorities=authorities,
            penalty=penalty,
            related_sections=related_sections,
            cross_references=cross_references,
        )

        return LegalSectionMetadata(
            act_name=act_name,
            act_year=act_year,
            chapter_number=chapter_number,
            chapter_name=chapter_name,
            section_number=section_number,
            section_title=section_title,
            sub_section=sub_section,
            page_number=page_number,
            keywords=keywords,
            definitions=definitions,
            authorities=authorities,
            penalty=penalty,
            related_sections=related_sections,
            cross_references=cross_references,
            extracted_at=datetime.utcnow(),
            extractor_version=self.VERSION,
            extraction_confidence=confidence,
        )

    def extract_document(
        self,
        full_text: str,
        source_file: str | Path,
    ) -> LegalDocumentMetadata:
        """
        Extract document-wide metadata from the complete text of an act.

        Splits the document into logical sections and aggregates results.
        """
        source = str(source_file)

        # Extract act-level identity fields from the document header
        header = full_text[:_HEADER_CHARS]
        act_name, act_year = self._extract_act_name_year(header)
        act_number = self._extract_act_number(header)

        # Split into logical sections and extract each
        sections_text = self._split_into_sections(full_text)
        logger.debug("Split document %s into %d sections", source, len(sections_text))

        section_metas: list[LegalSectionMetadata] = []
        for i, (sec_text, page_no) in enumerate(sections_text):
            try:
                meta = self.extract_section(sec_text, page_number=page_no)
                # Propagate act-level info if not detected in section
                if meta.act_name is None and act_name:
                    meta = meta.model_copy(
                        update={"act_name": act_name, "act_year": act_year}
                    )
                section_metas.append(meta)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Section extraction failed for %s at index %d: %s",
                    source,
                    i,
                    exc,
                )

        # Aggregate into document-level metadata
        doc_meta = LegalDocumentMetadata.aggregate_from_sections(source, section_metas)

        # Override with act-number if found
        if act_number:
            doc_meta = doc_meta.model_copy(update={"act_number": act_number})

        return doc_meta

    # ── Private: Act Name & Year ───────────────────────────────────────────────

    def _extract_act_name_year(self, text: str) -> tuple[str | None, int | None]:
        """Extract the act name and year from the document header region."""
        for pattern in PAT_ACT_NAME:
            match = pattern.search(text)
            if match:
                try:
                    name = match.group("act_name").strip()
                    year_str = match.group("year")
                    year = int(year_str) if year_str else None
                    if 1947 <= (year or 0) <= 2100:
                        return name, year
                    return name, None
                except (IndexError, ValueError):
                    continue
        return None, None

    def _extract_act_number(self, text: str) -> str | None:
        """Extract the official act number (e.g. 'Act No. 57 of 1955')."""
        match = RE_ACT_NUMBER.search(text)
        if match:
            try:
                return f"Act No. {match.group('act_number')} of {match.group('year')}"
            except IndexError:
                return None
        return None

    # ── Private: Chapter ──────────────────────────────────────────────────────

    def _extract_chapter(self, text: str) -> tuple[str | None, str | None]:
        """Extract chapter number and chapter title."""
        for pattern in PAT_CHAPTER:
            match = pattern.search(text)
            if match:
                try:
                    number = match.group("chapter_number").strip()
                    name: str | None = None
                    try:
                        raw_name = match.group("chapter_name")
                        name = raw_name.strip() if raw_name else None
                    except (IndexError, AttributeError):
                        pass
                    return number, name
                except (IndexError, AttributeError):
                    continue
        return None, None

    # ── Private: Section ──────────────────────────────────────────────────────

    def _extract_section(self, text: str) -> tuple[str | None, str | None]:
        """Extract section number and section heading."""
        # Try pattern with title first (most specific)
        match = RE_SECTION_WITH_TITLE.search(text)
        if match:
            try:
                number = match.group("section_number").strip()
                title = match.group("section_title").strip() or None
                return number, title
            except IndexError:
                pass

        # Fallback: any section pattern
        for pattern in PAT_SECTION[1:]:
            m = pattern.search(text)
            if m:
                try:
                    return m.group("section_number").strip(), None
                except IndexError:
                    continue

        return None, None

    def _extract_sub_section(self, text: str) -> str | None:
        """Extract the first sub-section identifier found in the text."""
        match = RE_SUB_SECTION.search(text)
        if match:
            try:
                return f"({match.group('sub_section')})"
            except IndexError:
                pass
        return None

    # ── Private: Keywords ─────────────────────────────────────────────────────

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract domain keywords from text using the seed vocabulary."""
        found: list[str] = []
        text_lower = text.lower()
        for keyword in ALL_KEYWORD_SEEDS:
            pattern = get_keyword_pattern(keyword)
            if pattern.search(text_lower):
                found.append(keyword)
        return found

    # ── Private: Definitions ──────────────────────────────────────────────────

    def _extract_definitions(self, text: str) -> list[DefinitionEntry]:
        """Extract term-definition pairs from the text."""
        definitions: list[DefinitionEntry] = []
        seen_terms: set[str] = set()

        for pattern in PAT_DEFINITION:
            for match in pattern.finditer(text):
                try:
                    term = match.group("term").strip()
                    definition = match.group("definition").strip()

                    # Skip very short or duplicate terms
                    if len(term) < 2 or term.lower() in seen_terms:
                        continue
                    if len(definition) < 10:
                        continue

                    # Extract optional section ref
                    section_ref: str | None = None
                    try:
                        section_ref = match.group("ref")
                    except IndexError:
                        pass

                    seen_terms.add(term.lower())
                    definitions.append(
                        DefinitionEntry(
                            term=term,
                            definition=definition,
                            section_ref=section_ref,
                        )
                    )
                except (IndexError, ValueError):
                    continue

        return definitions

    # ── Private: Authorities ──────────────────────────────────────────────────

    def _extract_authorities(self, text: str) -> list[AuthorityReference]:
        """Extract government authority references from the text."""
        authorities: list[AuthorityReference] = []
        seen_names: set[str] = set()

        for authority_type_str, pattern in AUTHORITY_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(0)
                name = raw.strip()
                if name.lower() in seen_names:
                    continue
                seen_names.add(name.lower())

                # Map string key to AuthorityType enum
                auth_type = _AUTHORITY_TYPE_MAP.get(
                    authority_type_str, AuthorityType.OTHER
                )

                # Capture a slightly wider context snippet
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                raw_context = text[start:end].strip()

                authorities.append(
                    AuthorityReference(
                        name=name,
                        authority_type=auth_type,
                        raw_text=raw_context,
                    )
                )

        return authorities

    # ── Private: Penalty ─────────────────────────────────────────────────────

    def _extract_penalty(self, text: str) -> PenaltyInfo | None:
        """Extract structured penalty information from the text."""
        for pattern in PAT_PENALTY:
            match = pattern.search(text)
            if not match:
                continue

            try:
                raw_text = match.group("raw").strip()
            except IndexError:
                raw_text = match.group(0).strip()

            # Expand search window: capture up to 300 chars after the penalty clause
            # to pick up fine amounts that appear after the imprisonment term.
            window_start = match.start()
            window_end = min(len(text), match.end() + 300)
            search_window = text[window_start:window_end]

            # Determine penalty type
            window_lower = search_window.lower()
            has_imprisonment = "imprisonment" in window_lower
            has_fine = any(
                w in window_lower
                for w in ("fine", "rs.", "rs ", "inr", "\u20b9", "rupees")
            )

            if has_imprisonment and has_fine:
                penalty_type = PenaltyType.BOTH
            elif has_imprisonment:
                penalty_type = PenaltyType.IMPRISONMENT
            elif has_fine:
                penalty_type = PenaltyType.FINE
            elif "deportation" in window_lower or "deported" in window_lower:
                penalty_type = PenaltyType.DEPORTATION
            elif "cancel" in window_lower or "revoke" in window_lower:
                penalty_type = PenaltyType.CANCELLATION
            else:
                penalty_type = PenaltyType.UNKNOWN

            # Extract imprisonment duration (search within window)
            imprisonment_years: float | None = None
            years_match = RE_PENALTY_YEARS.search(search_window)
            if years_match:
                try:
                    if years_match.group("years"):
                        imprisonment_years = float(years_match.group("years"))
                    elif years_match.group("months"):
                        imprisonment_years = float(years_match.group("months")) / 12
                except (IndexError, ValueError):
                    pass

            # Extract fine amount — search the expanded window first, then fall
            # back to the narrower raw_text so we don't miss amounts like
            # "Rs. 10,000" that appear after the imprisonment clause.
            fine_amount: float | None = None
            for search_text in (search_window, raw_text):
                fine_match = RE_PENALTY_FINE_AMOUNT.search(search_text)
                if fine_match:
                    try:
                        amount_str = fine_match.group("amount")
                        if amount_str:
                            fine_amount = float(amount_str.replace(",", ""))
                            break
                    except (IndexError, ValueError, AttributeError):
                        pass

            return PenaltyInfo(
                penalty_type=penalty_type,
                imprisonment_years=imprisonment_years,
                fine_amount=fine_amount,
                raw_text=raw_text,
            )

        return None

    # ── Private: Related Sections ─────────────────────────────────────────────

    def _extract_related_sections(self, text: str) -> list[str]:
        """Extract section numbers referenced within the same act."""
        sections: list[str] = []
        seen: set[str] = set()

        for pattern in PAT_RELATED_SECTION:
            for match in pattern.finditer(text):
                try:
                    sec = match.group("section").strip()
                    if sec not in seen:
                        seen.add(sec)
                        sections.append(sec)
                except IndexError:
                    continue

        return sections

    # ── Private: Cross References ────────────────────────────────────────────

    def _extract_cross_references(self, text: str) -> list[CrossReference]:
        """Extract references to other acts, rules, or schedules."""
        refs: list[CrossReference] = []
        seen_texts: set[str] = set()

        # External references (section X of the Y Act)
        for match in RE_CROSS_REF_EXTERNAL.finditer(text):
            try:
                ref_text = match.group(0).strip()
                if ref_text in seen_texts:
                    continue
                seen_texts.add(ref_text)

                act_name: str | None = None
                section_number: str | None = None
                try:
                    act_name = match.group("act_name").strip()
                except IndexError:
                    pass
                try:
                    section_number = match.group("number").strip()
                except IndexError:
                    pass

                refs.append(
                    CrossReference(
                        ref_type="section",
                        ref_text=ref_text,
                        act_name=act_name,
                        section_number=section_number,
                    )
                )
            except (IndexError, ValueError):
                continue

        # Schedule references
        schedule_pattern = re.compile(
            r"\b(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|"
            r"Ninth|Tenth|Eleventh|Twelfth)\s+Schedule\b",
            re.IGNORECASE,
        )
        for match in schedule_pattern.finditer(text):
            ref_text = match.group(0).strip()
            if ref_text not in seen_texts:
                seen_texts.add(ref_text)
                refs.append(
                    CrossReference(
                        ref_type="schedule",
                        ref_text=ref_text,
                    )
                )

        return refs

    # ── Private: Document Splitting ───────────────────────────────────────────

    def _split_into_sections(self, text: str) -> list[tuple[str, int | None]]:
        """
        Split a full document into (section_text, page_number) tuples.

        Uses form-feed characters (``\\f``) as page boundaries and section
        headers as logical split points.  Returns a list of (text, page_no)
        tuples where page_no is 1-indexed.
        """
        # Split on form-feed (page breaks) first
        pages = text.split("\f")
        sections: list[tuple[str, int | None]] = []

        for page_idx, page_text in enumerate(pages):
            page_no = page_idx + 1  # 1-indexed

            # Within each page, split on section headers
            section_split = re.split(
                r"(?=^\s*\d+[A-Z]?\.\s+[A-Z])",
                page_text,
                flags=re.MULTILINE,
            )

            if len(section_split) <= 1:
                # No section splits — treat whole page as one section
                stripped = page_text.strip()
                if stripped:
                    sections.append((stripped, page_no))
            else:
                for sec in section_split:
                    stripped = sec.strip()
                    if stripped:
                        sections.append((stripped, page_no))

        # If no page breaks found, return the whole document as one chunk
        if not sections and text.strip():
            sections.append((text.strip(), None))

        return sections

    # ── Private: Confidence ───────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(
        *,
        act_name: str | None,
        chapter_number: str | None,
        section_number: str | None,
        section_title: str | None,
        keywords: list[str],
        definitions: list[DefinitionEntry],
        authorities: list[AuthorityReference],
        penalty: PenaltyInfo | None,
        related_sections: list[str],
        cross_references: list[CrossReference],
    ) -> float:
        """
        Compute an extraction confidence score in [0.0, 1.0].

        Weights mirror ``_CONFIDENCE_WEIGHTS``.
        """
        score = 0.0
        if act_name:
            score += _CONFIDENCE_WEIGHTS["act_name"]
        if chapter_number:
            score += _CONFIDENCE_WEIGHTS["chapter_number"]
        if section_number:
            score += _CONFIDENCE_WEIGHTS["section_number"]
        if section_title:
            score += _CONFIDENCE_WEIGHTS["section_title"]
        if keywords:
            score += _CONFIDENCE_WEIGHTS["keywords"]
        if definitions:
            score += _CONFIDENCE_WEIGHTS["definitions"]
        if authorities:
            score += _CONFIDENCE_WEIGHTS["authorities"]
        if penalty:
            score += _CONFIDENCE_WEIGHTS["penalty"]
        if related_sections:
            score += _CONFIDENCE_WEIGHTS["related_sections"]
        if cross_references:
            score += _CONFIDENCE_WEIGHTS["cross_references"]
        return round(min(score, 1.0), 3)


# ---------------------------------------------------------------------------
# Authority type string → enum mapping
# ---------------------------------------------------------------------------

_AUTHORITY_TYPE_MAP: dict[str, AuthorityType] = {
    "central_government": AuthorityType.CENTRAL_GOVERNMENT,
    "state_government": AuthorityType.STATE_GOVERNMENT,
    "registrar_general": AuthorityType.REGISTRAR_GENERAL,
    "passport_authority": AuthorityType.PASSPORT_AUTHORITY,
    "immigration_officer": AuthorityType.IMMIGRATION_OFFICER,
    "frro": AuthorityType.FOREIGNERS_REGIONAL_REGISTRATION_OFFICER,
    "tribunal": AuthorityType.TRIBUNAL,
    "court": AuthorityType.COURT,
}
