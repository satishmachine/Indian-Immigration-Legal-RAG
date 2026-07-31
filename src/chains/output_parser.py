"""
chains.output_parser
====================
Output Parser converting LLM text and citations into LegalRAGResponse entities.
"""

from __future__ import annotations

import re
from typing import Sequence

from langchain_core.documents import Document as LCDocument

from chains.citation_formatter import extract_citations_from_docs
from core.models.response import LegalRAGResponse


class LegalRAGOutputParser:
    """Parser converting raw LLM completion output and retrieved docs into LegalRAGResponse."""

    @staticmethod
    def parse(
        question: str,
        answer_text: str,
        docs: Sequence[LCDocument],
    ) -> LegalRAGResponse:
        """
        Parse LLM response and retrieved documents into a strongly-typed LegalRAGResponse.

        Args:
            question: User question string.
            answer_text: Text response from the LLM.
            docs: Retrieved LangChain Document objects.

        Returns:
            LegalRAGResponse object.
        """
        citations = extract_citations_from_docs(docs)

        # Extract referenced sections (e.g. 'Section 5', 'section 21') using regex
        sec_matches = re.findall(r"\b(?:section|sec\.|rule)\s+\d+[a-z]?(?:\(\d+\))?", answer_text, flags=re.IGNORECASE)
        unique_sections = sorted(list({s.title() for s in sec_matches}))

        # Detect penalty / criminal consequence keywords
        penalty_keywords = ["offence", "offense", "penalty", "punishable", "imprisonment", "fine", "conviction", "prosecution"]
        has_penalty = any(k in answer_text.lower() for k in penalty_keywords)

        # Calculate grounding confidence score
        confidence = 1.0 if citations else 0.5
        if "provided statutory documents do not contain sufficient information" in answer_text.lower():
            confidence = 0.2

        return LegalRAGResponse(
            question=question,
            answer=answer_text.strip(),
            citations=citations,
            referenced_sections=unique_sections,
            has_penalty_clause=has_penalty,
            confidence_score=confidence,
        )
