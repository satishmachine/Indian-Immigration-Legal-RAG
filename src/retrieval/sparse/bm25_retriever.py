"""
retrieval.sparse.bm25_retriever
===============================
Production Pure-Python BM25Okapi Sparse Keyword Retriever.

Zero external dependencies. Implements BM25Okapi algorithm with legal-aware tokenization
and metadata filtering for statutory text passages.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from core.exceptions import SparseRetrievalError
from core.models.document import Chunk
from core.models.retrieval import RetrievalResult


def _stem_term(term: str) -> str:
    """Light stemming for legal terms to normalize plurals and verb forms."""
    t = term.lower()
    if len(t) <= 3:
        return t
    if t.endswith("ies"):
        return t[:-3] + "y"
    if t.endswith("es") and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        return t[:-1]
    if t.endswith("ing") and len(t) > 5:
        return t[:-3]
    if t.endswith("ed") and len(t) > 4:
        return t[:-2]
    return t


def legal_tokenize(text: str) -> list[str]:
    """
    Tokenizer for legal texts with light stemming.
    Preserves section references, numbers, and legal terms.
    """
    raw_tokens = re.findall(r"\w+", text.lower())
    result: list[str] = []
    for t in raw_tokens:
        if len(t) > 1:
            result.append(t)
            stemmed = _stem_term(t)
            if stemmed != t:
                result.append(stemmed)
    return result


class BM25Okapi:
    """
    BM25Okapi scoring engine.

    Args:
        corpus: List of tokenized documents.
        k1: Term frequency saturation parameter (default: 1.5).
        b: Document length normalization parameter (default: 0.75).
    """

    def __init__(
        self,
        corpus: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / max(self.corpus_size, 1)

        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}

        self._initialize(corpus)

    def _initialize(self, corpus: list[list[str]]) -> None:
        df: dict[str, int] = {}

        for doc in corpus:
            freqs: dict[str, int] = {}
            for token in doc:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_freqs.append(freqs)

            for token in freqs:
                df[token] = df.get(token, 0) + 1

        for token, freq in df.items():
            # Standard Lucene / Okapi BM25 IDF formula with smoothing
            idf_val = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[token] = max(idf_val, 1e-4)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        """Compute BM25 scores for all corpus documents against query_tokens."""
        scores = [0.0] * self.corpus_size

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf = self.idf[token]

            for i, freqs in enumerate(self.doc_freqs):
                if token not in freqs:
                    continue
                tf = freqs[token]
                doc_len = self.doc_len[i]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avgdl, 1e-4)))
                scores[i] += idf * (numerator / denominator)

        return scores


class BM25Retriever:
    """
    BM25 Sparse Keyword Retriever for legal chunks.

    Args:
        chunks: List of Chunk entities to index.
        k1: BM25 k1 parameter.
        b: BM25 b parameter.
    """

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        if chunks:
            self.index(chunks)

    def index(self, chunks: list[Chunk]) -> None:
        """Index a list of Chunk entities into the BM25 engine."""
        self.chunks = chunks
        tokenized_corpus = [legal_tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Perform BM25 keyword search with optional metadata filtering.

        Args:
            query: User search query string.
            top_k: Top K results to return.
            filters: Metadata filter dictionary.

        Returns:
            Ranked list of RetrievalResult objects.
        """
        if not self.chunks or not self.bm25:
            try:
                from ingestion.pipeline.ingestion_pipeline import IngestionPipeline
                chunks = IngestionPipeline.load_dataset_chunks()
                if chunks:
                    self.index(chunks)
            except Exception as e:
                logger.warning("BM25 auto-indexing failed: %s", e)

        if not self.chunks or not self.bm25:
            return []

        try:
            query_tokens = legal_tokenize(query)
            if not query_tokens:
                return []

            scores = self.bm25.get_scores(query_tokens)

            # Normalize BM25 scores to [0, 1] range for rank fusion compatibility
            max_score = max(scores) if scores else 1.0
            norm_factor = max_score if max_score > 0 else 1.0

            candidates: list[tuple[float, int, Chunk]] = []
            for idx, (score, chunk) in enumerate(zip(scores, self.chunks, strict=True)):
                if score <= 0:
                    continue

                # Apply metadata filtering if specified
                if filters and not self._matches_filter(chunk, filters):
                    continue

                normalized_score = min(score / norm_factor, 1.0)
                candidates.append((normalized_score, idx, chunk))

            # Sort descending by score
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_candidates = candidates[:top_k]

            results: list[RetrievalResult] = []
            for rank, (score, _, chunk) in enumerate(top_candidates):
                results.append(
                    RetrievalResult(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        score=score,
                        rank=rank + 1,
                        retrieval_method="bm25_sparse",
                        metadata=chunk.metadata,
                    )
                )

            return results
        except Exception as exc:
            if isinstance(exc, SparseRetrievalError):
                raise exc
            logger.error("BM25 sparse retrieval failed: %s", exc, exc_info=True)
            raise SparseRetrievalError(
                message=f"BM25 keyword retrieval failed: {exc}",
                details={"query": query, "raw_error": str(exc)},
            ) from exc

    def _matches_filter(self, chunk: Chunk, filters: dict[str, Any]) -> bool:
        """Check if chunk metadata matches the filter dictionary."""
        extra = chunk.metadata.extra
        for k, expected in filters.items():
            actual = extra.get(k) or getattr(chunk.metadata, k, None)
            if actual is None:
                return False

            actual_val = getattr(actual, "value", str(actual)).lower()

            if isinstance(expected, list):
                expected_list = [getattr(e, "value", str(e)).lower() for e in expected]
                if actual_val not in expected_list:
                    return False
            else:
                expected_val = getattr(expected, "value", str(expected)).lower()
                if actual_val != expected_val:
                    return False
        return True
