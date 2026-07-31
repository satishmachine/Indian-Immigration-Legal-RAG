"""
services.vector_store.qdrant_repository
========================================
Production Qdrant Repository Implementation following Repository Pattern.

Supports:
- Collection Lifecycle (Create, Delete, Exists, Info)
- Payload Index Management (Keyword, Integer, Float, Text indexes)
- Vector & Payload Upsert
- Point & Filtered Deletion
- Advanced Metadata Filtering (eq, ne, in, range, gte, lte)
- Dense Vector Search
- Hybrid RAG Search (Dense similarity + BM25 keyword score fusion)
"""

from __future__ import annotations

import logging, os, re, sys
from typing import Any

from core.config import get_settings
from core.interfaces.vector_store_repository import VectorStoreRepository
from core.models.document import Chunk, DocumentMetadata, LegalDomain
from core.models.retrieval import RetrievalResult

logger = logging.getLogger(__name__)

# Lazy import qdrant_client
_qdrant_lib: object | None = None
_global_qdrant_client: Any = None


def _get_qdrant_lib():
    global _qdrant_lib  # noqa: PLW0603
    if _qdrant_lib is None:
        import qdrant_client
        _qdrant_lib = qdrant_client
    return _qdrant_lib


class QdrantVectorRepository(VectorStoreRepository):
    """Qdrant Repository Pattern Implementation."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        default_collection: str | None = None,
        default_vector_size: int | None = None,
    ) -> None:
        cfg = get_settings().qdrant
        self._host = host or cfg.host
        self._port = port or cfg.port
        self._api_key = api_key or (cfg.api_key.get_secret_value() if cfg.api_key else None)
        self._default_collection = default_collection or cfg.collection_name
        self._default_vector_size = default_vector_size or cfg.vector_size
        self._https = cfg.https
        self._client: Any = None

    def get_client(self) -> Any:
        """Lazy client instantiation supporting Qdrant Cloud or local store."""
        global_client = getattr(sys, "_qdrant_global_client_singleton", None)
        if global_client is not None:
            self._client = global_client
            return self._client

        if self._client is None:
            qc = _get_qdrant_lib()
            cfg = get_settings().qdrant
            cloud_url = getattr(cfg, "qdrant_url", None) or os.environ.get("QDRANT_URL")
            cloud_key = self._api_key or os.environ.get("QDRANT_API_KEY")

            if cloud_url or cloud_key:
                target_url = cloud_url or cfg.url
                logger.info("Initializing Qdrant Cloud Client (url=%r)", target_url)
                self._client = qc.QdrantClient(
                    url=target_url,
                    api_key=cloud_key,
                    timeout=cfg.timeout,
                    prefer_grpc=cfg.prefer_grpc,
                )
            elif self._host and self._host not in ("localhost", "127.0.0.1", "0.0.0.0"):
                logger.info("Initializing Remote Qdrant Client (%s:%s)", self._host, self._port)
                self._client = qc.QdrantClient(
                    host=self._host,
                    port=self._port,
                    api_key=self._api_key,
                    https=self._https,
                    timeout=cfg.timeout,
                )
            else:
                logger.info("Initializing Local Qdrant Store ('./qdrant_db')")
                try:
                    self._client = qc.QdrantClient(path="./qdrant_db", check_compatibility=False)
                except Exception as lock_err:
                    logger.warning("Local storage './qdrant_db' locked (%s). Using in-memory store (':memory:').", lock_err)
                    self._client = qc.QdrantClient(":memory:", check_compatibility=False)

            sys._qdrant_global_client_singleton = self._client
            self.create_collection(recreate=False)
            self._ensure_default_indexes()

        return self._client

    # ── Collection Management ───────────────────────────────────────────────

    def create_collection(
        self,
        collection_name: str | None = None,
        vector_size: int | None = None,
        distance: str = "Cosine",
        recreate: bool = False,
    ) -> bool:
        """Create or recreate Qdrant collection."""
        col = collection_name or self._default_collection
        dim = vector_size or self._default_vector_size
        qc = _get_qdrant_lib()
        from qdrant_client.models import Distance, VectorParams

        client = self.get_client()

        dist_enum = Distance.COSINE
        if distance.lower() == "euclidean":
            dist_enum = Distance.EUCLID
        elif distance.lower() == "dot":
            dist_enum = Distance.DOT

        exists = self._collection_exists_on_client(client, col)

        if exists:
            try:
                info = client.get_collection(col)
                existing_params = getattr(getattr(info, "config", None), "params", None)
                vectors = getattr(existing_params, "vectors", None)
                existing_size = None
                if hasattr(vectors, "size"):
                    existing_size = vectors.size
                elif isinstance(vectors, dict):
                    for v in vectors.values():
                        if hasattr(v, "size"):
                            existing_size = v.size
                            break
                        elif isinstance(v, dict) and "size" in v:
                            existing_size = v["size"]
                            break
                if existing_size and existing_size != dim:
                    logger.warning("Vector size mismatch in collection %r: existing=%d, expected=%d. Recreating collection.", col, existing_size, dim)
                    client.delete_collection(col)
                    exists = False
            except Exception as check_err:
                logger.debug("Failed to inspect vector size for collection %r: %s", col, check_err)

        if exists and recreate:
            logger.info("Recreating collection %r", col)
            client.delete_collection(col)
            exists = False

        if not exists:
            logger.info("Creating Qdrant collection %r (dim=%d, distance=%s)", col, dim, distance)
            client.create_collection(
                collection_name=col,
                vectors_config=VectorParams(size=dim, distance=dist_enum),
            )
            return True
        return False

    def delete_collection(self, collection_name: str | None = None) -> bool:
        """Delete Qdrant collection."""
        col = collection_name or self._default_collection
        client = self.get_client()
        if self._collection_exists_on_client(client, col):
            client.delete_collection(collection_name=col)
            logger.info("Deleted collection %r", col)
            return True
        return False

    def collection_exists(self, collection_name: str | None = None) -> bool:
        """Check if collection exists."""
        col = collection_name or self._default_collection
        client = self.get_client()
        return self._collection_exists_on_client(client, col)

    def get_collection_info(self, collection_name: str | None = None) -> dict[str, Any]:
        """Return collection info and metrics."""
        col = collection_name or self._default_collection
        client = self.get_client()
        try:
            info = client.get_collection(col)
            return {
                "collection_name": col,
                "points_count": getattr(info, "points_count", 0),
                "vectors_count": getattr(info, "vectors_count", 0),
                "status": str(getattr(info, "status", "green")),
            }
        except Exception as exc:
            logger.error("Failed to get info for collection %r: %s", col, exc)
            return {"collection_name": col, "status": "not_found", "points_count": 0}

    # ── Index Management ───────────────────────────────────────────────────

    def create_payload_index(
        self,
        field_name: str,
        field_schema: str = "keyword",
        collection_name: str | None = None,
    ) -> bool:
        """Create a payload index on a metadata field."""
        col = collection_name or self._default_collection
        client = self.get_client()
        from qdrant_client.models import PayloadSchemaType

        schema_type = PayloadSchemaType.KEYWORD
        if field_schema == "integer":
            schema_type = PayloadSchemaType.INTEGER
        elif field_schema == "float":
            schema_type = PayloadSchemaType.FLOAT
        elif field_schema == "text":
            schema_type = PayloadSchemaType.TEXT
        elif field_schema == "bool":
            schema_type = PayloadSchemaType.BOOL

        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                client.create_payload_index(
                    collection_name=col,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            logger.info("Created payload index on %r (%s) in %r", field_name, field_schema, col)
            return True
        except Exception as exc:
            logger.debug("Payload index creation note for %r: %s", field_name, exc)
            return False

    def _ensure_default_indexes(self) -> None:
        """Create default statutory metadata indexes."""
        fields = [
            ("act_name", "keyword"),
            ("act_year", "integer"),
            ("section_number", "keyword"),
            ("legal_domain", "keyword"),
            ("document_type", "keyword"),
            ("meta_has_penalty", "bool"),
            ("meta_has_proviso", "bool"),
            ("meta_has_explanation", "bool"),
        ]
        for field_name, schema in fields:
            self.create_payload_index(field_name, schema)

    # ── Upsert & Delete ───────────────────────────────────────────────────

    def upsert(self, chunks: list[Chunk], collection_name: str | None = None) -> int:
        """Upsert Chunk entities into Qdrant."""
        if not chunks:
            return 0
        col = collection_name or self._default_collection
        client = self.get_client()
        from qdrant_client.models import PointStruct

        points: list[Any] = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue

            payload = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "source_file": chunk.metadata.source_file,
                "title": chunk.metadata.title,
                "legal_domain": chunk.metadata.legal_domain.value if hasattr(chunk.metadata.legal_domain, "value") else str(chunk.metadata.legal_domain),
                "document_type": chunk.metadata.document_type.value if hasattr(chunk.metadata.document_type, "value") else str(chunk.metadata.document_type),
                "year": chunk.metadata.year,
                "jurisdiction": chunk.metadata.jurisdiction,
                "tags": chunk.metadata.tags,
            }
            for k, v in chunk.metadata.extra.items():
                payload[f"meta_{k}"] = v

            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        if points:
            if not self._collection_exists_on_client(client, col):
                dim = len(points[0].vector) if points[0].vector else self._default_vector_size
                self.create_collection(collection_name=col, vector_size=dim, recreate=False)
            client.upsert(collection_name=col, points=points)
            logger.info("Upserted %d points to collection %r", len(points), col)
        return len(points)

    def delete_points(self, point_ids: list[str], collection_name: str | None = None) -> int:
        """Delete points by IDs."""
        if not point_ids:
            return 0
        col = collection_name or self._default_collection
        client = self.get_client()
        from qdrant_client.models import PointIdsList

        client.delete(collection_name=col, points_selector=PointIdsList(points=point_ids))
        logger.info("Deleted %d points from %r", len(point_ids), col)
        return len(point_ids)

    def delete_by_filter(self, filters: dict[str, Any], collection_name: str | None = None) -> int:
        """Delete points matching metadata filter."""
        col = collection_name or self._default_collection
        client = self.get_client()
        from qdrant_client.models import FilterSelector

        qdrant_filter = self._build_qdrant_filter(filters)
        if qdrant_filter is None:
            return 0

        client.delete(collection_name=col, points_selector=FilterSelector(filter=qdrant_filter))
        logger.info("Deleted points matching filter %s from %r", filters, col)
        return 1

    # ── Search Implementation ─────────────────────────────────────────────

    def search_dense(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
        collection_name: str | None = None,
    ) -> list[RetrievalResult]:
        """Execute dense vector similarity search using query_points or search."""
        col = collection_name or self._default_collection
        client = self.get_client()
        qdrant_filter = self._build_qdrant_filter(filters)

        try:
            if hasattr(client, "query_points"):
                res = client.query_points(
                    collection_name=col,
                    query=query_vector,
                    query_filter=qdrant_filter,
                    limit=top_k,
                    score_threshold=score_threshold if score_threshold > 0 else None,
                )
                hits = res.points
            elif hasattr(client, "search"):
                hits = client.search(
                    collection_name=col,
                    query_vector=query_vector,
                    query_filter=qdrant_filter,
                    limit=top_k,
                    score_threshold=score_threshold if score_threshold > 0 else None,
                )
        except Exception as exc:
            if "not aligned" in str(exc) or "shapes" in str(exc) or "dimension" in str(exc):
                logger.warning("Vector size mismatch during search in collection %r: %s. Recreating collection and indexing Data_Set.", col, exc)
                self.create_collection(collection_name=col, vector_size=len(query_vector), recreate=True)
                try:
                    from ingestion.pipeline.ingestion_pipeline import IngestionPipeline
                    IngestionPipeline().run(recreate_collection=False)
                    return self.search_dense(
                        query_vector=query_vector,
                        top_k=top_k,
                        filters=filters,
                        score_threshold=score_threshold,
                        collection_name=collection_name,
                    )
                except Exception as ing_exc:
                    logger.error("Auto-ingestion error after collection recreate: %s", ing_exc)
                    return []
            logger.error("Dense retrieval error in collection %r: %s", col, exc)
            return []

        results: list[RetrievalResult] = []
        for rank, hit in enumerate(hits):
            payload = getattr(hit, "payload", {}) or {}
            doc_meta = DocumentMetadata(
                source_file=payload.get("source_file", "unknown"),
                title=payload.get("title", "Untitled"),
                legal_domain=LegalDomain(payload.get("legal_domain", "unknown")),
                year=payload.get("year"),
                jurisdiction=payload.get("jurisdiction", "India"),
                tags=payload.get("tags", []),
                extra={k.replace("meta_", ""): v for k, v in payload.items() if k.startswith("meta_")},
            )

            results.append(
                RetrievalResult(
                    chunk_id=str(getattr(hit, "id", "")),
                    document_id=payload.get("document_id", ""),
                    content=payload.get("content", ""),
                    score=float(getattr(hit, "score", 0.0)),
                    metadata=doc_meta,
                    rank=rank,
                    retrieval_method="dense",
                )
            )

        return results

    def search_hybrid(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 5,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Execute Hybrid Search combining Dense vector search and BM25 text match fusion.
        Score = alpha * dense_score + (1 - alpha) * text_match_score.
        """
        dense_results = self.search_dense(
            query_vector=query_vector,
            top_k=top_k * 2,
            filters=filters,
            collection_name=collection_name,
        )

        if not dense_results:
            return []

        # Perform BM25 / keyword scoring on candidate passages
        query_terms = [w.lower() for w in re.findall(r"\w+", query_text) if len(w) > 2]
        scored_results: list[RetrievalResult] = []

        for res in dense_results:
            content_lower = res.content.lower()
            term_matches = sum(1 for term in query_terms if term in content_lower)
            text_score = min(term_matches / max(len(query_terms), 1), 1.0)

            # Combined hybrid score
            hybrid_score = (alpha * res.score) + ((1.0 - alpha) * text_score)

            res_copy = res.model_copy(
                update={"score": round(hybrid_score, 4), "retrieval_method": "hybrid"}
            )
            scored_results.append(res_copy)

        scored_results.sort(key=lambda x: x.score, reverse=True)
        final_results = scored_results[:top_k]
        for rank, res in enumerate(final_results):
            res.rank = rank

        return final_results

    # ── Filter Building Helper ─────────────────────────────────────────────

    def _build_qdrant_filter(self, filters: dict[str, Any] | None) -> Any:
        """Construct Qdrant Filter from key-value or operator dictionary."""
        if not filters:
            return None

        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

        conditions: list[Any] = []
        for key, val in filters.items():
            field_key = key if key.startswith("meta_") or key in ("source_file", "title", "legal_domain", "year", "document_type") else f"meta_{key}"

            if isinstance(val, list):
                conditions.append(FieldCondition(key=field_key, match=MatchAny(any=val)))
            elif isinstance(val, dict):
                # Operator dict e.g. {"gte": 1967, "lte": 2025}
                gte = val.get("gte")
                lte = val.get("lte")
                gt = val.get("gt")
                lt = val.get("lt")
                if any(x is not None for x in (gte, lte, gt, lt)):
                    conditions.append(
                        FieldCondition(
                            key=field_key,
                            range=Range(gte=gte, lte=lte, gt=gt, lt=lt),
                        )
                    )
            else:
                conditions.append(FieldCondition(key=field_key, match=MatchValue(value=val)))

        return Filter(must=conditions) if conditions else None

    @staticmethod
    def _collection_exists_on_client(client: Any, collection_name: str) -> bool:
        """Check if collection exists on client instance."""
        try:
            cols = client.get_collections().collections
            return any(c.name == collection_name for c in cols)
        except Exception:
            return False
