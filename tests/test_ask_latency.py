"""Latency instrumentation + playhead BGE-skip + stream stages."""

from __future__ import annotations

import json
import logging
import uuid
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ai_cowatcher.agent.completion import MockCompletionClient
from ai_cowatcher.config import Settings
from ai_cowatcher.observability.ask_latency import (
    STAGE_CACHE_LOOKUP,
    STAGE_GATE,
    STAGE_LLM_TTFT,
    STAGE_LLM_TOTAL,
    STAGE_SCENE_RETRIEVE,
    STAGE_TOTAL,
    AskLatencyTracker,
)
from ai_cowatcher.providers.mock import MockTextEmbedder
from ai_cowatcher.realtime.viewing_session import ViewingSession, build_viewing_session
from ai_cowatcher.retrieval.scene_lookup import SceneLookupTool, is_playhead_local_question
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore


@pytest.fixture
def latency_settings() -> Settings:
    return Settings(
        MOCK_MODE=True,
        UTTERANCE_GATE_STRATEGY="merged",
        PILOT_LOW_LATENCY=True,
        QA_CACHE_ENABLED=False,
        QDRANT_COLLECTION=f"test_lat_{uuid.uuid4().hex[:8]}",
        QUERY_EMBEDDING_CACHE_TTL_SEC=60,
    )


@pytest.fixture
def latency_session(latency_settings: Settings) -> ViewingSession:
    embedder = MockTextEmbedder()
    client = QdrantClient(":memory:")
    store = QdrantSceneStore(latency_settings, client=client)
    store.ensure_collection(embedder.vector_size)
    vector = embedder.embed_texts(["hello"])[0]
    client.upsert(
        collection_name=store._collection,
        points=[
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "title_id": "lat-title",
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
        latency_settings,
        qdrant_store=store,
        embedder=embedder,
        completion_client=MockCompletionClient(),
        qa_cache=None,
    )


def test_playhead_local_detector():
    assert is_playhead_local_question("Who is that?")
    assert is_playhead_local_question("What just happened?")
    assert is_playhead_local_question("What's going on?")
    assert is_playhead_local_question("Who's on screen right now?")
    assert not is_playhead_local_question("Why did the director cast Marcus?")


def test_playhead_path_skips_embedder(
    latency_settings: Settings, caplog: pytest.LogCaptureFixture
):
    embedder = MagicMock(wraps=MockTextEmbedder())
    client = QdrantClient(":memory:")
    store = QdrantSceneStore(latency_settings, client=client)
    store.ensure_collection(MockTextEmbedder.vector_size)
    client.upsert(
        collection_name=store._collection,
        points=[
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.1] * MockTextEmbedder.vector_size,
                payload={
                    "title_id": "lat-title",
                    "scene_id": "s0000",
                    "start_ts": 0.0,
                    "end_ts": 10.0,
                    "transcript": "Hello there.",
                    "caption": "",
                    "face_cluster_ids": [],
                },
            )
        ],
    )
    tool = SceneLookupTool(embedder, store, latency_settings)
    with caplog.at_level(logging.INFO):
        hits = tool.lookup(
            title_id="lat-title",
            query_text="What just happened?",
            current_ts=5.0,
        )
    assert hits
    assert all(h.start_ts <= 5.0 for h in hits)
    embedder.embed_texts.assert_not_called()
    assert any(
        "scene_lookup_path" in r.message and '"skip_bge":true' in r.message
        for r in caplog.records
    )


def test_embed_cache_reuses_vector(latency_settings: Settings):
    embedder = MagicMock(wraps=MockTextEmbedder())
    client = QdrantClient(":memory:")
    store = QdrantSceneStore(latency_settings, client=client)
    store.ensure_collection(MockTextEmbedder.vector_size)
    tool = SceneLookupTool(embedder, store, latency_settings)
    tool.lookup(
        title_id="lat-title",
        query_text="Who directed this film?",
        current_ts=5.0,
    )
    tool.lookup(
        title_id="lat-title",
        query_text="Who directed this film?",
        current_ts=6.0,
    )
    # First call embeds; second should hit TTL cache
    assert embedder.embed_texts.call_count == 1


def test_ask_stream_records_stages(
    latency_session: ViewingSession, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.INFO):
        events = list(
            latency_session.ask_stream(
                title_id="lat-title",
                current_ts=5.0,
                question="What just happened?",
                user_id="u1",
                persist_memory=False,
            )
        )
    types = [e.type for e in events]
    assert "status" in types
    assert "token" in types
    assert types[-1] == "done"

    stage_logs = [
        json.loads(r.message)
        for r in caplog.records
        if "ask_latency_stages" in r.message
    ]
    assert stage_logs
    stages = stage_logs[-1]["stages_ms"]
    assert STAGE_TOTAL in stages
    assert STAGE_CACHE_LOOKUP in stages
    assert STAGE_GATE in stages
    assert STAGE_SCENE_RETRIEVE in stages
    assert STAGE_LLM_TTFT in stages
    assert STAGE_LLM_TOTAL in stages


def test_tracker_marks_ttft_and_total():
    t = AskLatencyTracker(path="ask_stream")
    with t.stage(STAGE_GATE):
        pass
    t.mark_llm_request_sent()
    t.mark_llm_first_token()
    t.mark_llm_stream_complete()
    payload = t.finish()
    assert payload["event"] == "ask_latency_stages"
    assert STAGE_LLM_TTFT in payload["stages_ms"]
    assert STAGE_LLM_TOTAL in payload["stages_ms"]
