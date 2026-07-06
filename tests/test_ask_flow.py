"""End-to-end tests for POST /ask and spoiler-safe answers."""

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
def test_settings() -> Settings:
    return Settings(MOCK_MODE=True, QDRANT_COLLECTION="test_ask_flow")


@pytest.fixture
def embedder() -> MockTextEmbedder:
    return MockTextEmbedder()


@pytest.fixture
def seeded_qdrant(test_settings: Settings, embedder: MockTextEmbedder) -> QdrantSceneStore:
    client = QdrantClient(":memory:")
    store = QdrantSceneStore(test_settings, client=client)
    store.ensure_collection(embedder.vector_size)

    title_id = "thriller-001"
    query_vector = embedder.embed_texts([SPOILER_QUESTION])[0]
    filler_vector = embedder.embed_texts(["crime scene investigation"])[0]

    def upsert(
        scene_id: str,
        start_ts: float,
        end_ts: float,
        transcript: str,
        vector: list[float],
    ) -> None:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{title_id}:{scene_id}"))
        client.upsert(
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

    upsert(
        "s0000",
        start_ts=0.0,
        end_ts=20.0,
        transcript="Detectives arrive at the crime scene.",
        vector=filler_vector,
    )
    upsert(
        "s0001",
        start_ts=50.0,
        end_ts=70.0,
        transcript="The killer is Marcus and the room goes silent.",
        vector=query_vector,
    )
    return store


@pytest.fixture
def viewing_session(test_settings: Settings, seeded_qdrant: QdrantSceneStore, embedder: MockTextEmbedder):
    return build_viewing_session(
        test_settings,
        qdrant_store=seeded_qdrant,
        embedder=embedder,
        completion_client=MockCompletionClient(),
    )


def test_ask_before_reveal_says_unknown(viewing_session):
    result = viewing_session.ask(
        title_id="thriller-001",
        current_ts=25.0,
        question=SPOILER_QUESTION,
        user_id="viewer-42",
    )

    assert "don't know yet" in result.answer.lower()
    assert "marcus" not in result.answer.lower()


def test_ask_after_reveal_returns_grounded_answer(viewing_session):
    result = viewing_session.ask(
        title_id="thriller-001",
        current_ts=75.0,
        question=SPOILER_QUESTION,
        user_id="viewer-42",
    )

    assert "marcus" in result.answer.lower()
    assert "don't know yet" not in result.answer.lower()


def test_post_ask_endpoint(viewing_session, test_settings: Settings, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "ai_cowatcher.api.ask_routes.build_viewing_session",
        lambda: viewing_session,
    )
    app = create_app(test_settings)
    client = TestClient(app)

    early = client.post(
        "/ask",
        json={
            "title_id": "thriller-001",
            "current_ts": 25.0,
            "question": SPOILER_QUESTION,
            "user_id": "viewer-42",
        },
    )
    assert early.status_code == 200
    assert "don't know yet" in early.json()["answer"].lower()

    late = client.post(
        "/ask",
        json={
            "title_id": "thriller-001",
            "current_ts": 75.0,
            "question": SPOILER_QUESTION,
            "user_id": "viewer-42",
        },
    )
    assert late.status_code == 200
    assert "marcus" in late.json()["answer"].lower()
