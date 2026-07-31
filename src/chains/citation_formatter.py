"""
chains.citation_formatter
=========================
Statutory Context and Citation Formatter.

Formats retrieved LangChain Document passages into grounded text blocks for LLM prompts,
and extracts structured LegalCitation Pydantic objects.
"""

from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.documents import Document as LCDocument

from core.models.response import LegalCitation

logger = logging.getLogger(__name__)


def format_docs_for_prompt(docs: Sequence[LCDocument]) -> str:
    """
    Format retrieved LangChain documents into structured statutory text blocks for the LLM prompt.

    Args:
        docs: Sequence of retrieved LangChain Document objects.

    Returns:
        Formatted string containing statutory titles, sections, citations, and content snippets.
    """
    if not docs:
        return "No relevant statutory passages found in the database."

    formatted_blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        title = meta.get("title") or "Untitled Document"
        act_name = meta.get("act_name") or title
        year = meta.get("year")
        section = meta.get("section_number") or meta.get("section_no") or "N/A"
        page = meta.get("page_number") or meta.get("page") or "N/A"

        year_str = f" ({year})" if year else ""
        page_str = f" | Page {page}" if page and str(page).upper() != "N/A" else ""
        header = f"--- [Passage {idx}] {act_name}{year_str} | Section {section}{page_str} ---"

        formatted_blocks.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n".join(formatted_blocks)


def extract_citations_from_docs(docs: Sequence[LCDocument]) -> list[LegalCitation]:
    """
    Extract structured LegalCitation Pydantic objects from retrieved LangChain documents.

    Args:
        docs: Sequence of retrieved LangChain Document objects.

    Returns:
        List of LegalCitation objects.
    """
    from pathlib import Path

    citations: list[LegalCitation] = []
    for doc in docs:
        meta = doc.metadata or {}
        act_name = meta.get("act_name") or meta.get("title") or "Act"
        year = meta.get("year")
        sec_num = str(meta.get("section_number") or meta.get("section_no") or "General")
        sec_title = meta.get("section_title") or meta.get("title")
        page_no = meta.get("page_number") or meta.get("page")
        if str(page_no).upper() == "N/A":
            page_no = None
        score = float(meta.get("score", 0.0))

        pdf_raw = meta.get("pdf_name") or meta.get("source_file") or meta.get("source") or meta.get("file_name")
        pdf_filename = Path(str(pdf_raw)).name if pdf_raw else None

        year_str = f", {year}" if year else ""
        page_str = f", p. {page_no}" if page_no else ""
        sec_str = f"Section {sec_num}" if sec_num != "General" else ""
        cite_str = f"{act_name}{year_str}, {sec_str}{page_str}".strip(", ")

        citations.append(
            LegalCitation(
                act_name=act_name,
                year=year if isinstance(year, int) else None,
                section_number=sec_num,
                section_title=sec_title,
                page_number=page_no if isinstance(page_no, int) else None,
                pdf_name=pdf_filename,
                citation_text=cite_str,
                snippet=doc.page_content[:250].strip() + ("..." if len(doc.page_content) > 250 else ""),
                score=score,
            )
        )

    return citations
