"""
core.citation.models
====================
Pydantic Data Models for Structured Statutory Citations.
"""

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field


class CitationDetails(BaseModel):
    """
    Detailed Statutory Citation entity containing Act, Chapter, Section, Page, PDF,
    highlighted snippet, and clickable link.
    """

    act_name: str = Field(description="Official name of the Act or Rule")
    act_year: int | None = Field(default=None, description="Year of enactment")
    chapter_number: str | None = Field(default=None, description="Chapter identifier (e.g., 'II')")
    chapter_name: str | None = Field(default=None, description="Chapter title (e.g., 'CITIZENSHIP BY BIRTH')")
    section_number: str = Field(description="Section or Rule number (e.g., '5')")
    section_title: str | None = Field(default=None, description="Title of the statutory section")
    page_number: int | None = Field(default=None, description="Source PDF page number")
    pdf_name: str = Field(description="Basename of the source PDF file")
    pdf_path: str = Field(description="Absolute file path to the source PDF file")
    highlighted_snippet: str = Field(description="Text passage snippet with key terms highlighted")
    clickable_link: str = Field(description="Clickable markdown link pointing to file URI or PDF viewer action")
    citation_string: str = Field(description="Formatted legal citation string (e.g. 'The Citizenship Act, 1955, Ch. II, Sec. 5, p. 3')")
    relevance_score: float = Field(default=0.0, description="Reranker / hybrid retrieval score")

    def to_markdown_badge(self) -> str:
        """Return HTML/Markdown badge representation."""
        year_str = f" ({self.act_year})" if self.act_year else ""
        ch_str = f", Ch. {self.chapter_number}" if self.chapter_number else ""
        sec_str = f", Sec. {self.section_number}" if self.section_number != "N/A" else ""
        pg_str = f", p. {self.page_number}" if self.page_number else ""

        title = f"{self.act_name}{year_str}{ch_str}{sec_str}{pg_str}"
        return f"[{title}]({self.clickable_link})"
