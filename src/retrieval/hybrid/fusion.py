"""
retrieval.hybrid.fusion
======================
Score Fusion Algorithms for Hybrid Retrieval.

Implements:
- Reciprocal Rank Fusion (RRF)
- Weighted Score Fusion
"""

from __future__ import annotations

import logging
from core.models.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    rrf_k: int = 60,
    top_k: int = 10,
) -> list[RetrievalResult]:
    """
    Reciprocal Rank Fusion (RRF) across multiple retrieved result lists.

    Args:
        result_lists: List of ranked RetrievalResult lists (e.g. [dense_results, bm25_results]).
        rrf_k: Smoothing constant parameter (default: 60).
        top_k: Top K fused results to return.

    Returns:
        Fused and re-ranked list of RetrievalResult objects.
    """
    scores: dict[str, float] = {}
    items: dict[str, RetrievalResult] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list, start=1):
            chunk_id = item.chunk_id
            rrf_score = 1.0 / (rrf_k + rank)

            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score

            if chunk_id not in items:
                items[chunk_id] = item

    # Sort descending by RRF score
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused_results: list[RetrievalResult] = []
    for rank, (chunk_id, score) in enumerate(sorted_items[:top_k]):
        original = items[chunk_id]
        fused = original.model_copy(
            update={
                "score": round(score, 6),
                "rank": rank,
                "retrieval_method": "rrf_hybrid",
            }
        )
        fused_results.append(fused)

    return fused_results


def weighted_score_fusion(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    alpha: float = 0.5,
    top_k: int = 10,
) -> list[RetrievalResult]:
    """
    Weighted Convex Score Fusion combining dense vector scores and sparse BM25 scores.
    Score = alpha * dense_score + (1 - alpha) * sparse_score.

    Args:
        dense_results: Dense vector similarity retrieval results.
        sparse_results: BM25 sparse retrieval results.
        alpha: Weight parameter [0, 1]. alpha=1 means dense only, alpha=0 means sparse only.
        top_k: Number of top fused results to return.

    Returns:
        List of RetrievalResult entities sorted by fused score.
    """
    dense_map = {r.chunk_id: r for r in dense_results}
    sparse_map = {r.chunk_id: r for r in sparse_results}

    all_ids = set(dense_map.keys()).union(set(sparse_map.keys()))
    fused: list[RetrievalResult] = []

    for chunk_id in all_ids:
        dense_item = dense_map.get(chunk_id)
        sparse_item = sparse_map.get(chunk_id)

        d_score = dense_item.score if dense_item else 0.0
        s_score = sparse_item.score if sparse_item else 0.0

        fused_score = (alpha * d_score) + ((1.0 - alpha) * s_score)

        base_item = dense_item or sparse_item
        if base_item:
            res = base_item.model_copy(
                update={
                    "score": round(fused_score, 4),
                    "retrieval_method": "weighted_hybrid",
                }
            )
            fused.append(res)

    fused.sort(key=lambda x: x.score, reverse=True)
    final_results = fused[:top_k]
    for rank, res in enumerate(final_results):
        res.rank = rank

    return final_results
