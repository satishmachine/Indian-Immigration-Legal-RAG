"""
src.core.models.metadata
========================
Strongly-typed legal document metadata model.

This is the canonical metadata schema for the Indian Immigration Legal
Assistant.  Every field has:

* A clear type annotation
* Pydantic v2 validation with constraints where appropriate
* A description used in OpenAPI docs and schema exports

Design rules
------------
* ``frozen=True`` — metadata is immutable after extraction.
* All string fields are stripped of whitespace by default.
* Optional fields default to ``None`` (not empty string) to allow
  clean ``if field:`` checks in callers.
* Lists default to empty list (never None) so callers can always iterate.

Integration
-----------
This model replaces / extends the thin ``DocumentMetadata`` in
``src.core.models.document``.  The ``LegalSectionMetadata`` is attached
to individual chunks; ``LegalDocumentMetadata`` is attached to the
full document.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class PenaltyType(StrEnum):
    """Category of penalty prescribed in a legal provision."""

    IMPRISONMENT = "imprisonment"
    FINE = "fine"
    BOTH = "both"          # imprisonment AND fine
    DEPORTATION = "deportation"
    CANCELLATION = "cancellation"
    NONE = "none"
    UNKNOWN = "unknown"


class AuthorityType(StrEnum):
    """Category of government authority referenced in a legal provision."""

    CENTRAL_GOVERNMENT = "central_government"
    STATE_GOVERNMENT = "state_government"
    REGISTRAR_GENERAL = "registrar_general"
    PASSPORT_AUTHORITY = "passport_authority"
    IMMIGRATION_OFFICER = "immigration_officer"
    FOREIGNERS_REGIONAL_REGISTRATION_OFFICER = "frro"
    TRIBUNAL = "tribunal"
    COURT = "court"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class PenaltyInfo(BaseModel):
    """Structured representation of a legal penalty clause."""

    model_config = {"frozen": True}

    penalty_type: PenaltyType = Field(
        default=PenaltyType.UNKNOWN,
        description="Primary category of the prescribed penalty.",
    )
    imprisonment_years: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum imprisonment term in years (None if no imprisonment).",
    )
    fine_amount: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum fine in Indian Rupees (None if no fine).",
    )
    raw_text: str = Field(
        default="",
        description="Original penalty clause text as extracted from the document.",
    )

    @property
    def summary(self) -> str:
        """Return a human-readable penalty summary."""
        parts: list[str] = []
        if self.imprisonment_years is not None:
            parts.append(f"imprisonment up to {self.imprisonment_years} year(s)")
        if self.fine_amount is not None:
            parts.append(f"fine up to ₹{self.fine_amount:,.0f}")
        return " and/or ".join(parts) if parts else self.raw_text or "unspecified"


class AuthorityReference(BaseModel):
    """A government authority referenced in a legal provision."""

    model_config = {"frozen": True}

    name: str = Field(description="Full name of the authority as it appears in the text.")
    authority_type: AuthorityType = Field(
        default=AuthorityType.OTHER,
        description="Classified type of this authority.",
    )
    raw_text: str = Field(
        default="",
        description="Verbatim text fragment that mentions this authority.",
    )

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        return v.strip()


class DefinitionEntry(BaseModel):
    """A term-definition pair extracted from a 'Definitions' section."""

    model_config = {"frozen": True}

    term: str = Field(description="The legal term being defined.")
    definition: str = Field(description="The statutory definition text.")
    section_ref: str | None = Field(
        default=None,
        description="Section number where this definition appears (e.g. '2(a)').",
    )

    @field_validator("term", "definition", mode="before")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class CrossReference(BaseModel):
    """A cross-reference to another section, act, or rule."""

    model_config = {"frozen": True}

    ref_type: str = Field(
        description="Type of reference: 'section' | 'act' | 'rule' | 'schedule'.",
    )
    ref_text: str = Field(
        description="Full verbatim reference text (e.g. 'Section 12 of the Passports Act, 1967').",
    )
    act_name: str | None = Field(
        default=None,
        description="Name of the referenced act if different from the current document.",
    )
    section_number: str | None = Field(
        default=None,
        description="Specific section number referenced.",
    )

    @field_validator("ref_text", mode="before")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


# ---------------------------------------------------------------------------
# Section-level metadata (attached to Chunks)
# ---------------------------------------------------------------------------


PositiveInt = Annotated[int, Field(ge=1)]


class LegalSectionMetadata(BaseModel):
    """
    Fine-grained metadata extracted for a single section or chunk of an act.

    Attached to ``Chunk.metadata`` after the metadata extraction step.
    All fields are optional — extractors populate only what they can
    confidently identify.
    """

    model_config = {"frozen": True}

    # ── Structural location ──────────────────────────────────────────────────
    act_name: str | None = Field(
        default=None,
        description="Full official name of the parent act (e.g. 'The Citizenship Act, 1955').",
    )
    act_year: int | None = Field(
        default=None,
        ge=1947,
        le=2100,
        description="Year the act was enacted.",
    )
    chapter_number: str | None = Field(
        default=None,
        description="Roman or Arabic chapter number (e.g. 'I', 'II', '3').",
    )
    chapter_name: str | None = Field(
        default=None,
        description="Full title of the chapter.",
    )
    section_number: str | None = Field(
        default=None,
        description="Section identifier (e.g. '5', '12A', '3(1)(b)').",
    )
    section_title: str | None = Field(
        default=None,
        description="Official heading of the section.",
    )
    sub_section: str | None = Field(
        default=None,
        description="Sub-section identifier within the section (e.g. '(1)', '(a)').",
    )
    page_number: PositiveInt | None = Field(
        default=None,
        description="Source page number in the original PDF.",
    )
    page_range: tuple[int, int] | None = Field(
        default=None,
        description="(start_page, end_page) if the section spans multiple pages.",
    )

    # ── Content ──────────────────────────────────────────────────────────────
    keywords: list[str] = Field(
        default_factory=list,
        description="Domain-relevant keywords extracted from this section.",
    )
    definitions: list[DefinitionEntry] = Field(
        default_factory=list,
        description="Term-definition pairs extracted from definitions clauses.",
    )
    authorities: list[AuthorityReference] = Field(
        default_factory=list,
        description="Government authorities referenced in this section.",
    )
    penalty: PenaltyInfo | None = Field(
        default=None,
        description="Structured penalty information if a penalty clause is present.",
    )
    related_sections: list[str] = Field(
        default_factory=list,
        description=(
            "Section numbers within the same act that are explicitly referenced "
            "(e.g. ['3', '5', '12A'])."
        ),
    )
    cross_references: list[CrossReference] = Field(
        default_factory=list,
        description="References to other acts, rules, or schedules.",
    )

    # ── Statutory Sub-clause Elements ──────────────────────────────────────────
    explanations: list[str] = Field(
        default_factory=list,
        description="Explanation clauses extracted from this section.",
    )
    illustrations: list[str] = Field(
        default_factory=list,
        description="Illustration examples extracted from this section.",
    )
    exceptions: list[str] = Field(
        default_factory=list,
        description="Exception clauses extracted from this section.",
    )
    provisos: list[str] = Field(
        default_factory=list,
        description="Proviso clauses ('Provided that...') extracted from this section.",
    )

    # ── Extraction provenance ─────────────────────────────────────────────────
    extracted_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when metadata was extracted.",
    )
    extractor_version: str = Field(
        default="1.0.0",
        description="Version of the metadata extractor that produced this record.",
    )
    extraction_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall extraction confidence score (0 = low, 1 = high).",
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("act_name", "chapter_name", "section_title", "sub_section", mode="before")
    @classmethod
    def _strip_str(cls, v: str | None) -> str | None:
        return v.strip() if isinstance(v, str) else v

    @field_validator("section_number", mode="before")
    @classmethod
    def _normalise_section(cls, v: str | None) -> str | None:
        """Normalise section numbers: strip whitespace, uppercase letters."""
        if not isinstance(v, str):
            return v
        return v.strip().upper().lstrip("S").lstrip("EC").lstrip("TION").strip() or None

    @field_validator("keywords", mode="before")
    @classmethod
    def _deduplicate_keywords(cls, v: list[str]) -> list[str]:
        """Lowercase, strip, and deduplicate keywords while preserving order."""
        seen: set[str] = set()
        result: list[str] = []
        for kw in v:
            normalised = kw.lower().strip()
            if normalised and normalised not in seen:
                seen.add(normalised)
                result.append(normalised)
        return result

    @field_validator("related_sections", mode="before")
    @classmethod
    def _deduplicate_sections(cls, v: list[str]) -> list[str]:
        """Deduplicate related section references."""
        return list(dict.fromkeys(s.strip() for s in v if s.strip()))

    @model_validator(mode="after")
    def _validate_page_range(self) -> "LegalSectionMetadata":
        """Ensure page_range start <= end."""
        if self.page_range is not None:
            start, end = self.page_range
            if start > end:
                msg = f"page_range start ({start}) must be <= end ({end})"
                raise ValueError(msg)
        return self

    # ── Computed helpers ──────────────────────────────────────────────────────
    @property
    def full_section_id(self) -> str:
        """Return a canonical section identifier string for display."""
        parts: list[str] = []
        if self.act_name:
            parts.append(self.act_name)
        if self.act_year:
            parts.append(f"({self.act_year})")
        if self.section_number:
            parts.append(f"§ {self.section_number}")
        if self.sub_section:
            parts.append(self.sub_section)
        return " ".join(parts) if parts else "Unknown Section"

    @property
    def has_penalty(self) -> bool:
        """Return True if a penalty clause was found."""
        return (
            self.penalty is not None
            and self.penalty.penalty_type != PenaltyType.NONE
        )

    @property
    def has_definitions(self) -> bool:
        """Return True if at least one definition was extracted."""
        return len(self.definitions) > 0

    @property
    def has_explanation(self) -> bool:
        """Return True if an Explanation clause is present."""
        return len(self.explanations) > 0

    @property
    def has_illustration(self) -> bool:
        """Return True if an Illustration clause is present."""
        return len(self.illustrations) > 0

    @property
    def has_exception(self) -> bool:
        """Return True if an Exception clause is present."""
        return len(self.exceptions) > 0

    @property
    def has_proviso(self) -> bool:
        """Return True if a Proviso clause ('Provided that...') is present."""
        return len(self.provisos) > 0

    @property
    def citation(self) -> str:
        """Return a formatted legal citation string."""
        parts: list[str] = []
        if self.act_name:
            parts.append(self.act_name)
        if self.act_year:
            parts.append(f"{self.act_year}")
        if self.section_number:
            parts.append(f"Section {self.section_number}")
        if self.page_number:
            parts.append(f"p. {self.page_number}")
        return ", ".join(parts) if parts else "Unknown"

    def to_qdrant_payload(self) -> dict[str, object]:
        """
        Serialise to a flat dict suitable for Qdrant point payload.

        Nested objects are JSON-serialised; lists become flat JSON arrays.
        Qdrant payload values must be scalars, lists, or dicts.
        """
        return {
            "act_name": self.act_name,
            "act_year": self.act_year,
            "chapter_number": self.chapter_number,
            "chapter_name": self.chapter_name,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "sub_section": self.sub_section,
            "page_number": self.page_number,
            "page_range_start": self.page_range[0] if self.page_range else None,
            "page_range_end": self.page_range[1] if self.page_range else None,
            "keywords": self.keywords,
            "has_penalty": self.has_penalty,
            "penalty_type": self.penalty.penalty_type.value if self.penalty else None,
            "penalty_raw": self.penalty.raw_text if self.penalty else None,
            "imprisonment_years": self.penalty.imprisonment_years if self.penalty else None,
            "fine_amount": self.penalty.fine_amount if self.penalty else None,
            "related_sections": self.related_sections,
            "authority_names": [a.name for a in self.authorities],
            "authority_types": [a.authority_type.value for a in self.authorities],
            "has_definitions": self.has_definitions,
            "definition_terms": [d.term for d in self.definitions],
            "cross_ref_count": len(self.cross_references),
            "extraction_confidence": self.extraction_confidence,
            "extractor_version": self.extractor_version,
        }


# ---------------------------------------------------------------------------
# Document-level metadata (attached to full Documents)
# ---------------------------------------------------------------------------


class LegalDocumentMetadata(BaseModel):
    """
    Document-wide metadata for an entire act or legal instrument.

    Aggregated from all section-level extractions.  Stored alongside the
    document record in the relational database and used to populate
    document-level Qdrant payload fields.
    """

    model_config = {"frozen": True}

    # ── Identity ──────────────────────────────────────────────────────────────
    source_file: str = Field(description="Original filename or storage path.")
    act_name: str | None = Field(
        default=None,
        description="Full official name of the act.",
    )
    act_year: int | None = Field(
        default=None,
        ge=1947,
        le=2100,
        description="Year of enactment.",
    )
    act_number: str | None = Field(
        default=None,
        description="Official act number (e.g. 'Act No. 57 of 1955').",
    )
    jurisdiction: str = Field(
        default="India",
        description="Applicable jurisdiction.",
    )
    language: str = Field(
        default="en",
        description="ISO 639-1 language code.",
    )

    # ── Structure summary ─────────────────────────────────────────────────────
    total_chapters: int = Field(
        default=0,
        ge=0,
        description="Total number of chapters in the act.",
    )
    total_sections: int = Field(
        default=0,
        ge=0,
        description="Total number of sections extracted.",
    )
    total_pages: int = Field(
        default=0,
        ge=0,
        description="Total page count of the source document.",
    )

    # ── Aggregated content ────────────────────────────────────────────────────
    keywords: list[str] = Field(
        default_factory=list,
        description="Union of all section-level keywords (deduplicated).",
    )
    definitions: list[DefinitionEntry] = Field(
        default_factory=list,
        description="All term-definition pairs from the entire document.",
    )
    authorities: list[AuthorityReference] = Field(
        default_factory=list,
        description="All authority references across the document (deduplicated).",
    )
    sections_with_penalties: list[str] = Field(
        default_factory=list,
        description="Section numbers that contain penalty clauses.",
    )
    cross_references: list[CrossReference] = Field(
        default_factory=list,
        description="All external cross-references found in the document.",
    )

    # ── Provenance ────────────────────────────────────────────────────────────
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the document was ingested.",
    )
    extracted_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when metadata extraction completed.",
    )
    extractor_version: str = Field(
        default="1.0.0",
        description="Version of the metadata extractor used.",
    )

    @field_validator("source_file", mode="before")
    @classmethod
    def _strip_source(cls, v: str) -> str:
        if not v.strip():
            msg = "source_file must not be empty."
            raise ValueError(msg)
        return v.strip()

    @field_validator("keywords", mode="before")
    @classmethod
    def _deduplicate_keywords(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for kw in v:
            normalised = kw.lower().strip()
            if normalised and normalised not in seen:
                seen.add(normalised)
                result.append(normalised)
        return result

    @property
    def display_title(self) -> str:
        """Return a short human-readable title."""
        if self.act_name and self.act_year:
            return f"{self.act_name}, {self.act_year}"
        return self.act_name or self.source_file

    @property
    def has_penalties(self) -> bool:
        """Return True if any section in the document prescribes a penalty."""
        return len(self.sections_with_penalties) > 0

    @classmethod
    def aggregate_from_sections(
        cls,
        source_file: str,
        sections: list[LegalSectionMetadata],
    ) -> "LegalDocumentMetadata":
        """
        Build a document-level metadata record by aggregating section records.

        Args:
            source_file: Path to the source PDF / DOCX.
            sections:    List of section-level metadata records.

        Returns:
            A fully populated ``LegalDocumentMetadata`` instance.
        """
        # Extract act-level fields from the first section that has them
        act_name = next((s.act_name for s in sections if s.act_name), None)
        act_year = next((s.act_year for s in sections if s.act_year), None)

        # Aggregate keywords
        all_kw: list[str] = [kw for s in sections for kw in s.keywords]

        # Aggregate definitions
        all_defs: list[DefinitionEntry] = [d for s in sections for d in s.definitions]

        # Aggregate authorities (deduplicate by name)
        seen_authorities: set[str] = set()
        unique_authorities: list[AuthorityReference] = []
        for s in sections:
            for a in s.authorities:
                if a.name not in seen_authorities:
                    seen_authorities.add(a.name)
                    unique_authorities.append(a)

        # Sections with penalties
        penalty_sections = [
            s.section_number
            for s in sections
            if s.has_penalty and s.section_number is not None
        ]

        # All cross-references
        all_xrefs: list[CrossReference] = [xr for s in sections for xr in s.cross_references]

        # Chapter and section counts
        chapters = {s.chapter_number for s in sections if s.chapter_number}
        pages = [s.page_number for s in sections if s.page_number is not None]
        max_page = max(pages) if pages else 0

        return cls(
            source_file=source_file,
            act_name=act_name,
            act_year=act_year,
            total_chapters=len(chapters),
            total_sections=len(sections),
            total_pages=max_page,
            keywords=all_kw,
            definitions=all_defs,
            authorities=unique_authorities,
            sections_with_penalties=penalty_sections,
            cross_references=all_xrefs,
            extracted_at=datetime.utcnow(),
        )
