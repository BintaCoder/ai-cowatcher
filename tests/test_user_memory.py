"""Tests for user_memory — Postgres source of truth, Redis-style cache, per-user isolation."""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from ai_cowatcher.agent.completion import MockCompletionClient
from ai_cowatcher.agent.conversation_agent import ConversationAgent
from ai_cowatcher.config import Settings
from ai_cowatcher.db.base import Base
from ai_cowatcher.db import models  # noqa: F401
from ai_cowatcher.providers.mock import MockTextEmbedder
from ai_cowatcher.retrieval.scene_lookup import SceneLookupTool
from ai_cowatcher.retrieval.user_memory import UserMemoryTool
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore
from ai_cowatcher.storage.user_memory_store import InMemoryMemoryCache, UserMemoryStore

TITLE = "memory-title"


@pytest.fixture
def sqlite_session_factory():
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@pytest.fixture
def settings() -> Settings:
    return Settings(MOCK_MODE=True, QDRANT_COLLECTION=f"test_mem_{uuid.uuid4().hex[:8]}")


@pytest.fixture
def memory_store(settings: Settings, sqlite_session_factory) -> UserMemoryStore:
    return UserMemoryStore(
        session_factory=sqlite_session_factory,
        cache=InMemoryMemoryCache(),
        settings=settings,
    )


def test_user_memory_isolated_by_user_id(memory_store: UserMemoryStore, settings: Settings):
    memory_store.append_turn(
        user_id="user-a",
        title_id=TITLE,
        role="user",
        content="Who is Marcus?",
        current_ts=10.0,
    )
    memory_store.append_turn(
        user_id="user-a",
        title_id=TITLE,
        role="assistant",
        content="Marcus is a detective.",
        current_ts=10.0,
    )

    tool = UserMemoryTool(memory_store, settings)
    a_result = tool.lookup(user_id="user-a", title_id=TITLE, mode="turns")
    b_result = tool.lookup(user_id="user-b", title_id=TITLE, mode="turns")

    assert a_result["found"] is True
    assert len(a_result["turns"]) == 2
    assert "Marcus" in a_result["turns"][0]["content"]

    assert b_result["found"] is False
    assert b_result["turns"] == []

    # Direct store access must also respect user_id scoping.
    b_turns = memory_store.get_recent_turns("user-b", TITLE, max_turns=10)
    assert b_turns == []


def test_user_memory_reads_from_redis_cache_before_postgres(
    memory_store: UserMemoryStore, settings: Settings
):
    memory_store.append_turn(
        user_id="user-a",
        title_id=TITLE,
        role="user",
        content="First question",
        current_ts=1.0,
    )
    cached = memory_store._cache.get_recent("user-a", TITLE)
    assert cached is not None
    assert len(cached) == 1

    turns = memory_store.get_recent_turns("user-a", TITLE, max_turns=5)
    assert len(turns) == 1
    assert turns[0].content == "First question"


def test_agent_uses_user_memory_for_continuity(
    memory_store: UserMemoryStore,
    settings: Settings,
    sqlite_session_factory,
):
    memory_store.append_turn(
        user_id="viewer-1",
        title_id=TITLE,
        role="user",
        content="Who is the detective?",
        current_ts=20.0,
    )
    memory_store.append_turn(
        user_id="viewer-1",
        title_id=TITLE,
        role="assistant",
        content="That's Marcus.",
        current_ts=20.0,
    )

    embedder = MockTextEmbedder()
    qdrant = QdrantSceneStore(settings, client=QdrantClient(":memory:"))
    qdrant.ensure_collection(embedder.vector_size)

    agent = ConversationAgent(
        completion_client=MockCompletionClient(),
        scene_lookup=SceneLookupTool(embedder, qdrant, settings),
        settings=settings,
        user_memory=UserMemoryTool(memory_store, settings),
    )
    answer = agent.answer(
        title_id=TITLE,
        current_ts=25.0,
        question="What did I ask earlier?",
        user_id="viewer-1",
    )
    assert "detective" in answer.text.lower() or "marcus" in answer.text.lower()
