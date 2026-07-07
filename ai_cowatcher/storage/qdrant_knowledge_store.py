"""Qdrant persistence for the curated title knowledge base (RAG)."""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import KnowledgeChunkRecord, KnowledgeSearchHit
from ai_cowatcher.observability.prometheus_metrics import observe_storage_query


class QdrantKnowledgeStore:
    """Separate Qdrant collection for curated, non-spoiler knowledge chunks."""

    def __init__(self, settings: Settings, client: QdrantClient | None = None):
        self._settings = settings
        self._client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self._collection = settings.qdrant_knowledge_collection

    def ensure_collection(self, vector_size: int) -> None:
        if self._client.collection_exists(self._collection):
            info = self._client.get_collection(self._collection)
            existing_size = info.config.params.vectors.size
            if existing_size != vector_size:
                raise ValueError(
                    f"Qdrant collection {self._collection} expects dim {existing_size}, "
                    f"got {vector_size}"
                )
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )

    def upsert_chunks(
        self, chunks: list[KnowledgeChunkRecord], vectors: list[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")

        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.title_id}:{chunk.chunk_id}")),
                    vector=vector,
                    payload={
                        "title_id": chunk.title_id,
                        "chunk_id": chunk.chunk_id,
                        "category": chunk.category,
                        "text": chunk.text,
                        "source": chunk.source,
                    },
                )
            )

        if points:
            self._client.upsert(collection_name=self._collection, points=points)

    def delete_title(self, title_id: str) -> None:
        if not self._client.collection_exists(self._collection):
            return
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="title_id",
                            match=qmodels.MatchValue(value=title_id),
                        )
                    ]
                )
            ),
        )

    def count_title_chunks(self, title_id: str) -> int:
        if not self._client.collection_exists(self._collection):
            return 0
        result = self._client.count(
            collection_name=self._collection,
            count_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="title_id",
                        match=qmodels.MatchValue(value=title_id),
                    )
                ]
            ),
            exact=True,
        )
        return int(result.count)

    def search_knowledge(
        self,
        *,
        title_id: str,
        query_vector: list[float],
        top_k: int,
        category: str | None = None,
    ) -> list[KnowledgeSearchHit]:
        """Semantic search over curated knowledge — NO current_ts / spoiler filter.

        Unlike scene_lookup and character_lookup, this tool intentionally has no
        playback-position constraint. Chunks are vetted offline public facts
        (actor bios, crew, sports stats, production trivia) that do not depend on
        how far the viewer has watched.
        """
        if not self._client.collection_exists(self._collection):
            return []

        must_filters = [
            qmodels.FieldCondition(
                key="title_id",
                match=qmodels.MatchValue(value=title_id),
            ),
        ]
        if category:
            must_filters.append(
                qmodels.FieldCondition(
                    key="category",
                    match=qmodels.MatchValue(value=category),
                )
            )

        with observe_storage_query("qdrant", "search_knowledge"):
            results = self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                query_filter=qmodels.Filter(must=must_filters),
                limit=top_k,
                with_payload=True,
            ).points

        return [
            KnowledgeSearchHit(
                chunk_id=str(point.payload.get("chunk_id", "")),
                title_id=str(point.payload.get("title_id", title_id)),
                category=str(point.payload.get("category", "")),
                text=str(point.payload.get("text", "")),
                source=str(point.payload.get("source", "")),
                score=float(point.score or 0.0),
            )
            for point in results
            if point.payload is not None
        ]
