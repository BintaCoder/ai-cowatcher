"""Tests for two-tier Q&A cache (exact + semantic)."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient

from ai_cowatcher.config import Settings
from ai_cowatcher.providers.mock import MockTextEmbedder
from ai_cowatcher.realtime.viewing_session import ViewingSession
from ai_cowatcher.storage.qa_cache import (
    InMemoryExactKV,
    QACache,
    build_qa_cache,
    should_cache_answer,
)


@pytest.fixture
def cache_settings() -> Settings:
    return Settings(
        MOCK_MODE=True,
        QA_CACHE_ENABLED=True,
        QA_CACHE_COLLECTION=f"qa_test_{uuid.uuid4().hex[:10]}",
        QA_CACHE_SEMANTIC_THRESHOLD=0.5,
    )


@pytest.fixture
def qa_cache(cache_settings: Settings) -> QACache:
    embedder = MockTextEmbedder()
    client = QdrantClient(":memory:")
    return QACache(
        exact_kv=InMemoryExactKV(),
        qdrant=client,
        embedder=embedder,
        settings=cache_settings,
    )


def test_exact_hit_after_store(qa_cache: QACache):
    qa_cache.store(
        "t1",
        42.0,
        "What just happened?",
        "They're arguing in the foyer.",
    )
    hit = qa_cache.lookup("t1", 45.0, "what just happened")  # same 30s bucket, norm
    assert hit is not None
    assert hit.source == "exact"
    assert "foyer" in hit.answer


def test_bucket_mismatch_is_miss(qa_cache: QACache):
    qa_cache.store("t1", 10.0, "Who is that?", "The detective.")
    # Far later playhead → different bucket
    hit = qa_cache.lookup("t1", 400.0, "Who is that?")
    assert hit is None


def test_semantic_hit_when_exact_cleared(qa_cache: QACache, cache_settings: Settings):
    question = "What just happened on screen?"
    answer = "A detective waves from the foyer."
    qa_cache.store("t1", 12.0, question, answer)

    # Wipe exact tier only; vector remains in Qdrant
    qa_cache._exact = InMemoryExactKV()
    hit = qa_cache.lookup("t1", 15.0, question)
    assert hit is not None
    assert hit.source == "semantic"
    assert hit.answer == answer


def test_should_cache_filters():
    assert should_cache_answer(
        answer="Hello there.",
        speak=True,
        skip_memory=False,
        escalation_reason="merged:CONTENT",
    )
    assert not should_cache_answer(
        answer="",
        speak=False,
        skip_memory=True,
        escalation_reason="merged:FILLER",
    )
    assert not should_cache_answer(
        answer="x",
        speak=True,
        skip_memory=False,
        escalation_reason="merged:NAVIGATE",
        navigate=True,
    )
    assert not should_cache_answer(
        answer="Not sure yet — nothing's made that clear so far.",
        speak=True,
        skip_memory=False,
        escalation_reason="merged:CONTENT",
    )


def test_session_cache_round_trip(cache_settings: Settings):
    agent = MagicMock()
    agent.answer.return_value = MagicMock(
        text="They're fighting over the remote.",
        model_tier="fast",
        model_name="mock",
        escalation_reason="merged:CONTENT",
        used_context=True,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        skip_memory=False,
        speak=True,
    )
    embedder = MockTextEmbedder()
    cache = build_qa_cache(
        cache_settings,
        embedder=embedder,
        qdrant_client=QdrantClient(":memory:"),
        exact_kv=InMemoryExactKV(),
    )
    session = ViewingSession(
        agent,  # type: ignore[arg-type]
        cache_settings,
        qa_cache=cache,
    )
    first = session.ask(
        title_id="demo",
        current_ts=20.0,
        question="What's going on?",
        user_id="u1",
        persist_memory=False,
    )
    assert first.model_tier == "fast"
    assert agent.answer.call_count == 1

    second = session.ask(
        title_id="demo",
        current_ts=22.0,
        question="Whats going on?",
        user_id="u1",
        persist_memory=False,
    )
    assert second.model_tier == "cache"
    assert "cache:exact" in second.escalation_reason
    assert agent.answer.call_count == 1  # no second LLM call
    assert "remote" in second.answer
