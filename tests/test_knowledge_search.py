"""Tests for knowledge_search — curated RAG without playback-position filter."""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

from ai_cowatcher.agent.completion import MockCompletionClient
from ai_cowatcher.agent.conversation_agent import ConversationAgent
from ai_cowatcher.config import Settings
from ai_cowatcher.domain import KnowledgeChunkRecord
from ai_cowatcher.ingestion.knowledge_index import index_title_knowledge
from ai_cowatcher.providers.mock import MockTextEmbedder
from ai_cowatcher.retrieval.knowledge_search import KnowledgeSearchTool
from ai_cowatcher.retrieval.scene_lookup import SceneLookupTool
from ai_cowatcher.storage.qdrant_knowledge_store import QdrantKnowledgeStore
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore

TITLE = "kb-test"


@pytest.fixture
def settings() -> Settings:
    suffix = uuid.uuid4().hex[:8]
    return Settings(
        MOCK_MODE=True,
        QDRANT_COLLECTION=f"test_scenes_{suffix}",
        QDRANT_KNOWLEDGE_COLLECTION=f"test_knowledge_{suffix}",
    )


@pytest.fixture
def embedder() -> MockTextEmbedder:
    return MockTextEmbedder()


@pytest.fixture
def knowledge_store(settings: Settings, embedder: MockTextEmbedder) -> QdrantKnowledgeStore:
    client = QdrantClient(":memory:")
    store = QdrantKnowledgeStore(settings, client=client)
    chunks = [
        KnowledgeChunkRecord(
            chunk_id="director",
            title_id=TITLE,
            category="crew",
            text="The show was created by Marta Delgado, a veteran TV writer.",
            source="production_notes",
        ),
        KnowledgeChunkRecord(
            chunk_id="sports-record",
            title_id=TITLE,
            category="sports_statistics",
            text="The featured team finished the season 12-4 with a league-best defense.",
            source="production_notes",
        ),
    ]
    store.ensure_collection(embedder.vector_size)
    vectors = embedder.embed_texts([c.embedding_text for c in chunks])
    store.upsert_chunks(chunks, vectors)
    return store


def test_knowledge_search_has_no_timestamp_filter(
    settings: Settings, embedder: MockTextEmbedder, knowledge_store: QdrantKnowledgeStore
):
    tool = KnowledgeSearchTool(embedder, knowledge_store, settings)
    hits = tool.search(title_id=TITLE, query_text="who created the show")
    assert hits
    assert any("Marta Delgado" in hit.text for hit in hits)

    # Same result regardless of hypothetical playback position — no current_ts param exists.
    hits_late = tool.search(title_id=TITLE, query_text="who created the show")
    assert any(hit.chunk_id == "director" for hit in hits_late)


def test_knowledge_search_category_filter(
    settings: Settings, embedder: MockTextEmbedder, knowledge_store: QdrantKnowledgeStore
):
    tool = KnowledgeSearchTool(embedder, knowledge_store, settings)
    hits = tool.search(
        title_id=TITLE,
        query_text="season record",
        category="sports_statistics",
    )
    assert len(hits) == 1
    assert hits[0].category == "sports_statistics"


def test_agent_uses_knowledge_search_for_director_question(
    settings: Settings, embedder: MockTextEmbedder, knowledge_store: QdrantKnowledgeStore
):
    scene_store = QdrantSceneStore(settings, client=QdrantClient(":memory:"))
    scene_store.ensure_collection(embedder.vector_size)

    agent = ConversationAgent(
        completion_client=MockCompletionClient(),
        scene_lookup=SceneLookupTool(embedder, scene_store, settings),
        settings=settings,
        knowledge_search=KnowledgeSearchTool(embedder, knowledge_store, settings),
    )
    answer = agent.answer(
        title_id=TITLE,
        current_ts=5.0,
        question="Who created this show?",
        user_id="u1",
    )
    assert "Marta Delgado" in answer.text


def test_index_title_knowledge_from_json(tmp_path, settings: Settings, embedder: MockTextEmbedder):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    path = knowledge_dir / f"{TITLE}.json"
    path.write_text(
        '{"title_id":"'
        + TITLE
        + '","chunks":[{"chunk_id":"a","category":"crew","text":"Creator is Sam Lee."}]}',
        encoding="utf-8",
    )
    settings = settings.model_copy(update={"knowledge_dir": str(knowledge_dir)})
    store = QdrantKnowledgeStore(settings, client=QdrantClient(":memory:"))
    result = index_title_knowledge(
        TITLE,
        settings=settings,
        embedder=embedder,
        knowledge_store=store,
        knowledge_path=path,
    )
    assert result.chunk_count == 1
    assert store.count_title_chunks(TITLE) == 1
