"""
retrieval.hybrid
================
Hybrid retrieval package.
"""

from retrieval.hybrid.fusion import reciprocal_rank_fusion, weighted_score_fusion
from retrieval.hybrid.legal_hybrid_retriever import LegalHybridRetriever
from retrieval.hybrid.mmr import maximal_marginal_relevance, text_similarity

__all__: list[str] = [
    "LegalHybridRetriever",
    "maximal_marginal_relevance",
    "reciprocal_rank_fusion",
    "text_similarity",
    "weighted_score_fusion",
]
