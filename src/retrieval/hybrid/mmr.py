"""
retrieval.hybrid.mmr
====================
Maximal Marginal Relevance (MMR) Diversity Selector.

Selects a subset of candidate passages that maximizes relevance to the user query
while minimizing redundancy among selected results.
"""

from __future__ import annotations

import logging
import re
from core.models.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


def text_similarity(text1: str, text2: str) -> float:
    """Calculate token-based Jaccard similarity between two text passages."""
    tokens1 = set(re.findall(r"\w+", text1.lower()))
    tokens2 = set(re.findall(r"\w+", text2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / max(len(union), 1)


def maximal_marginal_relevance(
    candidates: list[RetrievalResult],
    top_k: int = 5,
    lambda_mult: float = 0.5,
) -> list[RetrievalResult]:
    """
    Apply Maximal Marginal Relevance (MMR) selection on candidate results.

    Args:
        candidates: Pre-ranked list of RetrievalResult objects (sorted by initial score).
        top_k: Number of diverse results to select.
        lambda_mult: Diversity parameter between 0 (max diversity) and 1 (max relevance). Default 0.5.

    Returns:
        List of selected diverse RetrievalResult objects.
    """
    if not candidates or top_k <= 0:
        return []

    if len(candidates) <= top_k:
        return candidates

    selected: list[RetrievalResult] = []
    unselected = list(candidates)

    # 1. Select the top 1 candidate with the highest initial score
    first_pick = unselected.pop(0)
    selected.append(first_pick)

    # 2. Iteratively pick candidates maximizing MMR metric
    while len(selected) < top_k and unselected:
        best_score = -float("inf")
        best_idx = 0

        for i, cand in enumerate(unselected):
            relevance = cand.score

            # Calculate max similarity with any already selected document
            max_sim = max(
                text_similarity(cand.content, sel.content)
                for sel in selected
            )

            # MMR formula
            mmr_score = (lambda_mult * relevance) - ((1.0 - lambda_mult) * max_sim)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        chosen = unselected.pop(best_idx)
        selected.append(chosen)

    # Re-rank selected results
    for rank, res in enumerate(selected):
        res.rank = rank

    logger.debug("MMR selection reduced %d candidates to %d diverse results", len(candidates), len(selected))
    return selected
