"""Tests for Prometheus /metrics exposition."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ai_cowatcher.agent.completion import MockCompletionClient
from ai_cowatcher.config import Settings
from ai_cowatcher.main import create_app
from ai_cowatcher.providers.mock import MockTextEmbedder
from ai_cowatcher.realtime.viewing_session import build_viewing_session
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore

SPOILER_QUESTION = "Who is the killer?"


@pytest.fixture
def prom_settings() -> Settings:
    return Settings(
        MOCK_MODE=True,
        PROMETHEUS_ENABLED=True,
        QDRANT_COLLECTION=f"test_prom_{uuid.uuid4().hex[:8]}",
    )


@pytest.fixture
def prom_session(prom_settings: Settings):
    embedder = MockTextEmbedder()
    client = QdrantClient(":memory:")
    store = QdrantSceneStore(prom_settings, client=client)
    store.ensure_collection(embedder.vector_size)

    title_id = "thriller-prom"
    query_vector = embedder.embed_texts([SPOILER_QUESTION])[0]
    client.upsert(
        collection_name=store._collection,
        points=[
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=query_vector,
                payload={
                    "title_id": title_id,
                    "scene_id": "s0001",
                    "start_ts": 50.0,
                    "end_ts": 70.0,
                    "transcript": "The killer is Marcus.",
                    "caption": "",
                    "face_cluster_ids": [],
                },
            )
        ],
    )

    return build_viewing_session(
        prom_settings,
        qdrant_store=store,
        embedder=embedder,
        completion_client=MockCompletionClient(),
    )


def test_metrics_endpoint_exports_ask_histogram(prom_session, prom_settings: Settings):
    prom_session.ask(
        title_id="thriller-prom",
        current_ts=25.0,
        question=SPOILER_QUESTION,
        user_id="viewer-1",
    )

    app = create_app(prom_settings)
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "cowatcher_ask_request_duration_seconds" in body
    assert "cowatcher_ask_model_tier_total" in body
    assert "cowatcher_tool_call_duration_seconds" in body


def test_metrics_disabled_returns_404():
    settings = Settings(MOCK_MODE=True, PROMETHEUS_ENABLED=False)
    app = create_app(settings)
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 404
