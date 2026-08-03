"""Qdrant vector persistence for scene events."""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import SceneEventRecord, SceneLookupHit
from ai_cowatcher.observability.prometheus_metrics import observe_storage_query


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
                        "speaker_cluster_ids": event.speaker_cluster_ids,
                        "audio_object_key": event.audio_object_key,
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
        spoiler_safe: bool = True,
    ) -> list[SceneLookupHit]:
        """Semantic search; optionally enforce spoiler guard (start_ts <= current_ts)."""
        if not self._client.collection_exists(self._collection):
            return []

        must_filters = [
            qmodels.FieldCondition(
                key="title_id",
                match=qmodels.MatchValue(value=title_id),
            ),
        ]
        if spoiler_safe:
            # Scene has *started* by now — includes the in-progress scene the viewer is
            # watching. (end_ts-only filtering excluded the current scene mid-clip and
            # caused empty retrieval → excessive "I don't know" answers.)
            must_filters.append(
                qmodels.FieldCondition(
                    key="start_ts",
                    range=qmodels.Range(lte=current_ts),
                )
            )

        spoiler_filter = qmodels.Filter(must=must_filters)

        with observe_storage_query("qdrant", "search_scenes"):
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
                speaker_cluster_ids=tuple(point.payload.get("speaker_cluster_ids") or ()),
                score=float(point.score or 0.0),
                audio_object_key=(
                    str(point.payload["audio_object_key"])
                    if point.payload.get("audio_object_key")
                    else None
                ),
            )
            for point in results
            if point.payload is not None
        ]
        hits.sort(key=lambda hit: (hit.start_ts, hit.scene_id))
        return hits

    def playhead_scenes(
        self,
        *,
        title_id: str,
        current_ts: float,
        limit: int = 3,
    ) -> list[SceneLookupHit]:
        """Scenes at/just before the playhead — no embedding, spoiler-safe.

        Prefer the overlapping (in-play) scene first, then recently finished ones.
        Ideal for “who/what is on screen now” without BGE latency.
        """
        if not self._client.collection_exists(self._collection) or limit <= 0:
            return []

        spoiler_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="title_id",
                    match=qmodels.MatchValue(value=title_id),
                ),
                qmodels.FieldCondition(
                    key="start_ts",
                    range=qmodels.Range(lte=current_ts),
                ),
            ]
        )
        # Pull a window and rank in process (scroll has no order_by on all versions).
        fetch_n = max(limit * 8, 24)
        with observe_storage_query("qdrant", "playhead_scenes"):
            points, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=spoiler_filter,
                limit=fetch_n,
                with_payload=True,
                with_vectors=False,
            )

        scored: list[SceneLookupHit] = []
        for point in points:
            if point.payload is None:
                continue
            start_ts = float(point.payload.get("start_ts", 0.0))
            end_ts = float(point.payload.get("end_ts", 0.0))
            # Prefer the currently playing scene (contains playhead).
            if start_ts <= current_ts <= end_ts:
                proximity = 0.0
            else:
                proximity = max(0.0, current_ts - end_ts)
            scored.append(
                SceneLookupHit(
                    scene_id=str(point.payload.get("scene_id", "")),
                    title_id=str(point.payload.get("title_id", title_id)),
                    start_ts=start_ts,
                    end_ts=end_ts,
                    transcript=str(point.payload.get("transcript", "")),
                    caption=str(point.payload.get("caption", "")),
                    face_cluster_ids=tuple(point.payload.get("face_cluster_ids") or ()),
                    speaker_cluster_ids=tuple(point.payload.get("speaker_cluster_ids") or ()),
                    score=1.0 / (1.0 + proximity),
                    audio_object_key=(
                        str(point.payload["audio_object_key"])
                        if point.payload.get("audio_object_key")
                        else None
                    ),
                )
            )
        scored.sort(key=lambda h: (-h.score, -h.start_ts, h.scene_id))
        selected = scored[:limit]
        selected.sort(key=lambda h: (h.start_ts, h.scene_id))
        return selected
