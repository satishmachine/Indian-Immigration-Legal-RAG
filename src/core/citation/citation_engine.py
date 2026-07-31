"""
core.citation.citation_engine
==============================
Reusable Statutory Citation Engine.

Extracts, highlights, and builds clickable statutory citations containing:
- Act Name & Year
- Chapter Number & Name
- Section Number & Title
- Page Number
- Source PDF File Name & Absolute Path
- Highlighted Source Snippets
- Clickable Markdown/HTML Deep Links
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document as LCDocument

from core.citation.models import CitationDetails
from core.models.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class CitationEngine:
    """
    Production Citation Engine for Statutory Legal RAG Systems.
    """

    @classmethod
    def create_citation(
        cls,
        doc_or_meta: LCDocument | RetrievalResult | dict[str, Any],
        query: str | None = None,
    ) -> CitationDetails:
        """
        Build a complete structured CitationDetails object.

        Args:
            doc_or_meta: LangChain Document, RetrievalResult, or metadata dictionary.
            query: User search query string to highlight matching terms.

        Returns:
            CitationDetails instance.
        """
        meta: dict[str, Any] = {}
        content: str = ""
        score: float = 0.0

        if isinstance(doc_or_meta, LCDocument):
            meta = doc_or_meta.metadata or {}
            content = doc_or_meta.page_content
            score = float(meta.get("score", 0.0))
        elif isinstance(doc_or_meta, RetrievalResult):
            meta = doc_or_meta.metadata.extra.copy()
            meta.update(
                {
                    "source": doc_or_meta.metadata.source_file,
                    "title": doc_or_meta.metadata.title,
                    "year": doc_or_meta.metadata.year,
                    "legal_domain": doc_or_meta.metadata.legal_domain,
                    "score": doc_or_meta.score,
                }
            )
            content = doc_or_meta.content
            score = doc_or_meta.score
        elif isinstance(doc_or_meta, dict):
            meta = doc_or_meta
            content = str(meta.get("content", ""))
            score = float(meta.get("score", 0.0))

        # 1. Extract Act Name & Year
        act_name = str(meta.get("act_name") or meta.get("title") or "Statutory Act")
        act_year = meta.get("act_year") or meta.get("year")
        act_year_int = int(act_year) if act_year and str(act_year).isdigit() else None

        # 2. Extract Chapter Number & Name
        chapter_no = meta.get("chapter_number") or meta.get("chapter_no")
        chapter_name = meta.get("chapter_name")

        # 3. Extract Section Number & Title
        sec_num = str(meta.get("section_number") or meta.get("section_no") or "N/A")
        sec_title = meta.get("section_title") or meta.get("title")

        # 4. Extract Page Number
        page_no = meta.get("page_number") or meta.get("page")
        page_int = int(page_no) if page_no and str(page_no).isdigit() else None

        # 5. Extract PDF File Path and Name
        raw_source = str(meta.get("source") or meta.get("source_file") or "document.pdf")
        pdf_path_obj = Path(raw_source).resolve()
        pdf_name = pdf_path_obj.name
        pdf_path_str = pdf_path_obj.as_posix()

        # 6. Build Highlighted Source Snippet
        snippet_raw = content[:300].strip() + ("..." if len(content) > 300 else "")
        highlighted_snippet = cls.highlight_terms(snippet_raw, query)

        # 7. Construct Clickable Link
        page_fragment = f"#page={page_int}" if page_int else ""
        clickable_link = f"file:///{pdf_path_str}{page_fragment}"

        # 8. Build Formatted Citation String
        year_part = f", {act_year_int}" if act_year_int else ""
        ch_part = f", Ch. {chapter_no}" if chapter_no else ""
        sec_part = f", Sec. {sec_num}" if sec_num != "N/A" else ""
        page_part = f", p. {page_int}" if page_int else ""

        citation_string = f"{act_name}{year_part}{ch_part}{sec_part}{page_part}"

        return CitationDetails(
            act_name=act_name,
            act_year=act_year_int,
            chapter_number=str(chapter_no) if chapter_no else None,
            chapter_name=str(chapter_name) if chapter_name else None,
            section_number=sec_num,
            section_title=str(sec_title) if sec_title else None,
            page_number=page_int,
            pdf_name=pdf_name,
            pdf_path=pdf_path_str,
            highlighted_snippet=highlighted_snippet,
            clickable_link=clickable_link,
            citation_string=citation_string,
            relevance_score=round(score, 4),
        )

    @classmethod
    def highlight_terms(cls, text: str, query: str | None) -> str:
        """
        Highlight query terms inside snippet using HTML `<mark>` tags.

        Args:
            text: Text snippet to highlight.
            query: Search query string.

        Returns:
            HTML string with highlighted terms.
        """
        if not query or not query.strip():
            return text

        # Extract words longer than 2 chars
        terms = [re.escape(w) for w in re.findall(r"\w+", query) if len(w) > 2]
        if not terms:
            return text

        pattern = re.compile(r"\b(" + "|".join(terms) + r")\b", flags=re.IGNORECASE)

        def replace_fn(match: re.Match) -> str:
            val = match.group(0)
            return f'<mark style="background-color: rgba(59, 130, 246, 0.3); color: #93C5FD; padding: 0 3px; border-radius: 3px; font-weight: 600;">{val}</mark>'

        return pattern.sub(replace_fn, text)

    @classmethod
    def generate_citations_from_docs(
        cls,
        docs: Sequence[LCDocument],
        query: str | None = None,
    ) -> list[CitationDetails]:
        """
        Process a list of LangChain documents into a list of CitationDetails.

        Args:
            docs: Sequence of retrieved LangChain Document objects.
            query: Optional query for text highlighting.

        Returns:
            List of CitationDetails objects.
        """
        return [cls.create_citation(doc, query=query) for doc in docs]
