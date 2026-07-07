"""Tests for /ask latency optimizations (warm session, deferred memory)."""

from __future__ import annotations

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


@pytest.fixture
def perf_settings() -> Settings:
    return Settings(MOCK_MODE=True, QDRANT_COLLECTION=f"test_perf_{uuid.uuid4().hex[:8]}")


@pytest.fixture
def perf_session(perf_settings: Settings) -> ViewingSession:
    embedder = MockTextEmbedder()
    client = QdrantClient(":memory:")
    store = QdrantSceneStore(perf_settings, client=client)
    store.ensure_collection(embedder.vector_size)
    title_id = "perf-title"
    vector = embedder.embed_texts(["hello"])[0]
    client.upsert(
        collection_name=store._collection,
        points=[
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "title_id": title_id,
                    "scene_id": "s0000",
                    "start_ts": 0.0,
                    "end_ts": 10.0,
                    "transcript": "A character waves hello.",
                    "caption": "",
                    "face_cluster_ids": [],
                },
            )
        ],
    )
    return build_viewing_session(
        perf_settings,
        qdrant_store=store,
        embedder=embedder,
        completion_client=MockCompletionClient(),
    )


def test_lifespan_builds_warm_viewing_session(
    perf_settings: Settings,
    perf_session: ViewingSession,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "ai_cowatcher.main.build_viewing_session",
        lambda _settings: perf_session,
    )
    with TestClient(create_app(perf_settings)) as client:
        assert client.app.state.viewing_session is perf_session


def test_ask_defers_memory_persist(perf_session: ViewingSession):
    memory = MagicMock()
    perf_session._user_memory_store = memory  # noqa: SLF001

    perf_session.ask(
        title_id="perf-title",
        current_ts=5.0,
        question="What just happened?",
        user_id="u1",
        persist_memory=False,
    )

    memory.append_turn.assert_not_called()

    perf_session.persist_memory(
        user_id="u1",
        title_id="perf-title",
        question="What just happened?",
        answer="Someone waved.",
        current_ts=5.0,
    )
    assert memory.append_turn.call_count == 2


def test_title_display_name_cached(perf_session: ViewingSession):
    perf_session._title_display_names["cached-title"] = "Cached Show"  # noqa: SLF001
    perf_session._session_factory = MagicMock()  # noqa: SLF001

    assert perf_session._lookup_title_display_name("cached-title") == "Cached Show"  # noqa: SLF001
    perf_session._session_factory.assert_not_called()  # noqa: SLF001
