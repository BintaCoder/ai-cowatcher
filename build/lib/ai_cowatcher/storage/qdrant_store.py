"""Qdrant vector persistence for scene events."""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import SceneEventRecord, SceneLookupHit


class QdrantSceneStore:
    def __init__(self, settings: Settings, client: QdrantClient | None = None):
        self._settings = settings
        self._client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self._collection = settings.qdrant_collection

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

    def upsert_scene_events(
        self, events: list[SceneEventRecord], vectors: list[list[float]]
    ) -> None:
        if len(events) != len(vectors):
            raise ValueError("events and vectors length mismatch")

        points = []
        for event, vector in zip(events, vectors, strict=True):
            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{event.title_id}:{event.scene_id}")),
                    vector=vector,
                    payload={
                        "title_id": event.title_id,
                        "scene_id": event.scene_id,
                        "start_ts": event.start_ts,
                        "end_ts": event.end_ts,
                        "transcript": event.transcript,
                        "caption": event.caption,
                        "face_cluster_ids": event.face_cluster_ids,
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

    def count_title_scenes(self, title_id: str) -> int:
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

    def search_scenes(
        self,
        *,
        title_id: str,
        query_vector: list[float],
        current_ts: float,
        top_k: int,
    ) -> list[SceneLookupHit]:
        """Semantic search with spoiler guard: only scenes with end_ts <= current_ts."""
        if not self._client.collection_exists(self._collection):
            return []

        spoiler_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="title_id",
                    match=qmodels.MatchValue(value=title_id),
                ),
                qmodels.FieldCondition(
                    key="end_ts",
                    range=qmodels.Range(lte=current_ts),
                ),
            ]
        )

        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=spoiler_filter,
            limit=top_k,
            with_payload=True,
        ).points

        hits = [
            SceneLookupHit(
                scene_id=str(point.payload.get("scene_id", "")),
                title_id=str(point.payload.get("title_id", title_id)),
                start_ts=float(point.payload.get("start_ts", 0.0)),
                end_ts=float(point.payload.get("end_ts", 0.0)),
                transcript=str(point.payload.get("transcript", "")),
                caption=str(point.payload.get("caption", "")),
                face_cluster_ids=tuple(point.payload.get("face_cluster_ids") or ()),
                score=float(point.score or 0.0),
            )
            for point in results
            if point.payload is not None
        ]
        hits.sort(key=lambda hit: (hit.start_ts, hit.scene_id))
        return hits
