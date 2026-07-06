"""Tests for spoiler-safe scene_lookup retrieval."""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ai_cowatcher.config import Settings
from ai_cowatcher.providers.mock import MockTextEmbedder
from ai_cowatcher.retrieval.scene_lookup import SceneLookupTool
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore


@pytest.fixture
def test_settings() -> Settings:
    return Settings(MOCK_MODE=True, QDRANT_COLLECTION="test_scene_lookup")


@pytest.fixture
def qdrant_store(test_settings: Settings) -> QdrantSceneStore:
    client = QdrantClient(":memory:")
    store = QdrantSceneStore(test_settings, client=client)
    store.ensure_collection(MockTextEmbedder.vector_size)
    return store


def _upsert_scene(
    store: QdrantSceneStore,
    *,
    title_id: str,
    scene_id: str,
    start_ts: float,
    end_ts: float,
    transcript: str,
    vector: list[float],
) -> None:
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{title_id}:{scene_id}"))
    store._client.upsert(
        collection_name=store._collection,
        points=[
            qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "title_id": title_id,
                    "scene_id": scene_id,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "transcript": transcript,
                    "caption": "",
                    "face_cluster_ids": [],
                },
            )
        ],
    )


def test_scene_lookup_never_returns_future_scenes_even_when_best_match(
    test_settings: Settings,
    qdrant_store: QdrantSceneStore,
):
    """Spoiler guard: end_ts > current_ts scenes are excluded regardless of score."""
    embedder = MockTextEmbedder()
    query_vector = embedder.embed_texts(["killer reveal secret twist"])[0]
    title_id = "mystery-001"

    _upsert_scene(
        qdrant_store,
        title_id=title_id,
        scene_id="s0000",
        start_ts=0.0,
        end_ts=10.0,
        transcript="Opening credits.",
        vector=[0.1] * embedder.vector_size,
    )
    _upsert_scene(
        qdrant_store,
        title_id=title_id,
        scene_id="s0001",
        start_ts=50.0,
        end_ts=80.0,
        transcript="The killer is Marcus and nobody expected it.",
        vector=query_vector,
    )

    tool = SceneLookupTool(embedder, qdrant_store, test_settings)
    hits = tool.lookup(
        title_id=title_id,
        query_text="killer reveal secret twist",
        current_ts=30.0,
        top_k=5,
    )

    scene_ids = {hit.scene_id for hit in hits}
    assert "s0001" not in scene_ids
    assert all(hit.end_ts <= 30.0 for hit in hits)

    if hits:
        assert hits[0].scene_id == "s0000"


def test_scene_lookup_returns_hits_in_chronological_order(
    test_settings: Settings,
    qdrant_store: QdrantSceneStore,
):
    embedder = MockTextEmbedder()
    title_id = "chrono-001"
    shared_vector = embedder.embed_texts(["detective interview"])[0]

    _upsert_scene(
        qdrant_store,
        title_id=title_id,
        scene_id="s0002",
        start_ts=40.0,
        end_ts=55.0,
        transcript="Later interview.",
        vector=shared_vector,
    )
    _upsert_scene(
        qdrant_store,
        title_id=title_id,
        scene_id="s0000",
        start_ts=0.0,
        end_ts=15.0,
        transcript="Early interview.",
        vector=shared_vector,
    )
    _upsert_scene(
        qdrant_store,
        title_id=title_id,
        scene_id="s0001",
        start_ts=20.0,
        end_ts=35.0,
        transcript="Middle interview.",
        vector=shared_vector,
    )

    tool = SceneLookupTool(embedder, qdrant_store, test_settings)
    hits = tool.lookup(
        title_id=title_id,
        query_text="detective interview",
        current_ts=60.0,
        top_k=3,
    )

    assert [hit.scene_id for hit in hits] == ["s0000", "s0001", "s0002"]
    assert hits == sorted(hits, key=lambda hit: (hit.start_ts, hit.scene_id))
