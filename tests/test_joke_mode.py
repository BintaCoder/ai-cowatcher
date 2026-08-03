"""Tests for joke / one-liner co-watch mode."""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ai_cowatcher.agent.brevity import enforce_brief_answer
from ai_cowatcher.agent.completion import MockCompletionClient
from ai_cowatcher.agent.joke_intent import is_joke_request, joke_fallback_answer
from ai_cowatcher.config import Settings
from ai_cowatcher.providers.mock import MockTextEmbedder
from ai_cowatcher.realtime.viewing_session import build_viewing_session
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore


@pytest.mark.parametrize(
    "question,expected",
    [
        ("joke", True),
        ("Joke!", True),
        ("one liner", True),
        ("one-liner", True),
        ("tell me a joke", True),
        ("make me laugh", True),
        ("say something funny", True),
        ("What just happened?", False),
        ("who's the killer?", False),
        ("what's the joke about?", False),
        ("why is that funny?", False),
    ],
)
def test_is_joke_request(question: str, expected: bool):
    assert is_joke_request(question) is expected


def test_joke_brevity_cap():
    long = (
        "This is a very long jokey monologue that rambles on about towels "
        "and national security and then keeps going forever unfortunately."
    )
    out = enforce_brief_answer(long, "joke", joke_mode=True)
    assert len(out.split()) <= 20


def test_joke_fallback_uses_scene():
    payloads = [
        [
            {
                "transcript": "They reorganize the towels by category again.",
                "caption": "",
            }
        ]
    ]
    line = joke_fallback_answer(payloads)
    assert line
    assert "towels" in line.lower() or "plot twist" in line.lower()


def test_ask_joke_returns_one_liner():
    settings = Settings(MOCK_MODE=True, QDRANT_COLLECTION="test_joke_mode")
    embedder = MockTextEmbedder()
    client = QdrantClient(":memory:")
    store = QdrantSceneStore(settings, client=client)
    store.ensure_collection(embedder.vector_size)
    title_id = "sitcom-001"
    vector = embedder.embed_texts(["funny dialogue conversation banter what just happened"])[0]
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{title_id}:s1"))
    client.upsert(
        collection_name=store._collection,
        points=[
            qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "title_id": title_id,
                    "scene_id": "s1",
                    "start_ts": 0.0,
                    "end_ts": 30.0,
                    "transcript": "Chandler and Joey argue over the remote again.",
                    "caption": "",
                    "face_cluster_ids": [],
                },
            )
        ],
    )
    session = build_viewing_session(
        settings,
        completion_client=MockCompletionClient(),
        embedder=embedder,
        qdrant_store=store,
    )
    result = session.ask(
        title_id=title_id,
        current_ts=10.0,
        question="joke",
        user_id="tester",
    )
    assert result.escalation_reason in ("joke_request", "merged:JOKE")
    assert result.model_tier == "fast"
    assert result.answer
    assert "not sure yet" not in result.answer.lower()
    assert len(result.answer.split()) <= 28
