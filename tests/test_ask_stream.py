"""Tests for progressive /ask/stream SSE path."""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ai_cowatcher.agent.completion import MockCompletionClient
from ai_cowatcher.config import Settings
from ai_cowatcher.main import create_app
from ai_cowatcher.providers.mock import MockTextEmbedder
from ai_cowatcher.realtime.viewing_session import ViewingSession, build_viewing_session
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore

QUESTION = "What just happened?"


def _parse_sse(body: str) -> list[dict]:
    events: list[dict] = []
    for block in body.split("\n\n"):
        line = next((part for part in block.split("\n") if part.startswith("data: ")), None)
        if not line:
            continue
        events.append(json.loads(line[len("data: ") :]))
    return events


@pytest.fixture
def stream_settings() -> Settings:
    return Settings(MOCK_MODE=True, QDRANT_COLLECTION=f"test_ask_stream_{uuid.uuid4().hex[:8]}")


@pytest.fixture
def stream_session(stream_settings: Settings) -> ViewingSession:
    embedder = MockTextEmbedder()
    client = QdrantClient(":memory:")
    store = QdrantSceneStore(stream_settings, client=client)
    store.ensure_collection(embedder.vector_size)
    vector = embedder.embed_texts([QUESTION])[0]
    client.upsert(
        collection_name=store._collection,
        points=[
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "title_id": "stream-title",
                    "scene_id": "s0000",
                    "start_ts": 0.0,
                    "end_ts": 12.0,
                    "transcript": "A detective waves hello in the foyer.",
                    "caption": "",
                    "face_cluster_ids": [],
                },
            )
        ],
    )
    return build_viewing_session(
        stream_settings,
        qdrant_store=store,
        embedder=embedder,
        completion_client=MockCompletionClient(),
    )


@pytest.fixture
def stream_client(stream_settings: Settings, stream_session: ViewingSession, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "ai_cowatcher.main.build_viewing_session",
        lambda *_a, **_k: stream_session,
    )
    monkeypatch.setattr(
        "ai_cowatcher.main.build_navigation_session",
        lambda *_a, **_k: MagicMock(),
    )
    monkeypatch.setattr("ai_cowatcher.main.create_db_engine", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr("ai_cowatcher.main.init_database", lambda *_a, **_k: None)
    monkeypatch.setattr("ai_cowatcher.main._build_embedder", lambda *_a, **_k: MockTextEmbedder())
    monkeypatch.setattr("ai_cowatcher.main.QdrantSceneStore", lambda *_a, **_k: MagicMock())

    with TestClient(create_app(stream_settings)) as client:
        yield client


def test_ask_stream_emits_tool_and_token_events(stream_client: TestClient):
    response = stream_client.post(
        "/ask/stream",
        json={
            "title_id": "stream-title",
            "current_ts": 10.0,
            "question": QUESTION,
            "user_id": "viewer-stream",
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    events = _parse_sse(response.text)
    types = [event["type"] for event in events]
    assert "status" in types
    assert "tool_start" in types
    assert "token" in types
    assert types[-1] == "done"

    done = events[-1]
    assert done["answer"]
    assert done["title_id"] == "stream-title"
    assert done["model_tier"] in ("fast", "escalated")

    tokens = "".join(event.get("text", "") for event in events if event["type"] == "token")
    assert tokens.strip() == done["answer"].strip()


def test_watch_page_references_stream_endpoint(stream_client: TestClient):
    response = stream_client.get("/watch")
    assert response.status_code == 200
    assert "/ask/stream" in response.text
    assert "createSentenceSpeaker" in response.text
