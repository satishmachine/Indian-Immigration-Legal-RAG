"""
evaluation.metrics
==================
Metric Calculation Functions for Legal RAG Evaluation.

Computes:
- Retrieval Recall
- Retrieval Precision
- Faithfulness
- Groundedness
- Hallucination Rate
- Citation Accuracy
"""

from __future__ import annotations

import re
from typing import Sequence
from langchain_core.documents import Document as LCDocument
from core.models.response import LegalCitation


def compute_retrieval_recall(
    retrieved_sections: Sequence[str],
    expected_sections: Sequence[str],
) -> float:
    """Compute Retrieval Recall metric."""
    if not expected_sections:
        return 1.0
    if not retrieved_sections:
        return 0.0

    ret_set = {str(s).strip().lower() for s in retrieved_sections}
    exp_set = {str(s).strip().lower() for s in expected_sections}

    hits = ret_set.intersection(exp_set)
    return len(hits) / max(len(exp_set), 1)


def compute_retrieval_precision(
    retrieved_sections: Sequence[str],
    expected_sections: Sequence[str],
) -> float:
    """Compute Retrieval Precision metric."""
    if not retrieved_sections:
        return 0.0
    if not expected_sections:
        return 1.0

    ret_set = {str(s).strip().lower() for s in retrieved_sections}
    exp_set = {str(s).strip().lower() for s in expected_sections}

    hits = ret_set.intersection(exp_set)
    return len(hits) / max(len(ret_set), 1)


def compute_faithfulness(answer: str, context_docs: Sequence[LCDocument]) -> float:
    """Compute Faithfulness metric (alignment of answer text with context)."""
    if not answer.strip():
        return 0.0
    if not context_docs:
        return 0.0

    combined_context = " ".join(d.page_content for d in context_docs).lower()
    answer_sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 10]

    if not answer_sentences:
        return 1.0

    supported_count = 0
    for stmt in answer_sentences:
        words = [w.lower() for w in re.findall(r"\w+", stmt) if len(w) > 3]
        if not words:
            supported_count += 1
            continue

        match_count = sum(1 for w in words if w in combined_context)
        match_ratio = match_count / max(len(words), 1)
        if match_ratio >= 0.4:
            supported_count += 1

    return supported_count / max(len(answer_sentences), 1)


def compute_groundedness(
    answer: str,
    citations: Sequence[LegalCitation],
    context_docs: Sequence[LCDocument],
) -> float:
    """Compute Groundedness metric."""
    if "does not contain sufficient information" in answer.lower():
        return 1.0
    if not citations or not context_docs:
        return 0.0

    faithfulness = compute_faithfulness(answer, context_docs)
    citation_ratio = min(len(citations) / max(len(context_docs), 1), 1.0)
    return round((0.7 * faithfulness) + (0.3 * citation_ratio), 4)


def compute_hallucination_rate(groundedness: float) -> float:
    """Compute Hallucination Rate metric (1.0 - Groundedness)."""
    return round(max(1.0 - groundedness, 0.0), 4)


def compute_citation_accuracy(
    citations: Sequence[LegalCitation],
    context_docs: Sequence[LCDocument],
) -> float:
    """Compute Citation Accuracy metric."""
    if not citations:
        return 0.0
    if not context_docs:
        return 0.0

    context_text = " ".join(d.page_content for d in context_docs).lower()
    valid_citations = 0

    for cite in citations:
        sec = str(cite.section_number).lower()
        if sec in context_text or f"section {sec}" in context_text or sec == "general":
            valid_citations += 1

    return valid_citations / max(len(citations), 1)
