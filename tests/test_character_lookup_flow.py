"""Spoiler-safety test for character_lookup wired into the conversation agent.

Asserts the required behaviour: "have I seen him before?" asked BEFORE two
characters' relationship is revealed does not leak that relationship, and DOES
surface it once the current timestamp passes the reveal. The single
conversation agent (one reasoning pass) chooses character_lookup itself.
"""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

from ai_cowatcher.agent.completion import MockCompletionClient
from ai_cowatcher.agent.conversation_agent import ConversationAgent
from ai_cowatcher.config import Settings
from ai_cowatcher.domain import SceneEventRecord
from ai_cowatcher.enrichment.identity import link_identities
from ai_cowatcher.enrichment.relationships import build_appearances, build_relationships
from ai_cowatcher.providers.mock import MockTextEmbedder
from ai_cowatcher.retrieval.character_lookup import CharacterLookupTool
from ai_cowatcher.retrieval.scene_lookup import SceneLookupTool
from ai_cowatcher.storage.character_store import InMemoryCharacterStore
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore

TITLE = "drama-flow"
QUESTION = "Have I seen him before?"


def _scenes() -> list[SceneEventRecord]:
    fc0, fc1 = f"{TITLE}-fc-0", f"{TITLE}-fc-1"
    sc0, sc1 = f"{TITLE}-sc-SPEAKER_00", f"{TITLE}-sc-SPEAKER_01"
    return [
        SceneEventRecord("s0000", TITLE, 0.0, 20.0, "A man arrives in town.", "", [fc0], [sc0]),
        SceneEventRecord("s0001", TITLE, 30.0, 50.0, "He waits at the diner.", "", [fc0], [sc0]),
        SceneEventRecord(
            "s0002",
            TITLE,
            120.0,
            140.0,
            "You are my sister, he finally admits.",
            "",
            [fc0, fc1],
            [sc0, sc1],
        ),
    ]


@pytest.fixture
def agent() -> ConversationAgent:
    settings = Settings(MOCK_MODE=True, QDRANT_COLLECTION=f"test_char_{uuid.uuid4().hex[:8]}")

    scenes = _scenes()
    characters = link_identities(TITLE, scenes, min_cooccur=1)
    appearances = build_appearances(characters, scenes)
    relationships = build_relationships(TITLE, characters, scenes)

    store = InMemoryCharacterStore()
    store.replace_title_characters(TITLE, characters, appearances, relationships)

    embedder = MockTextEmbedder()
    qdrant = QdrantSceneStore(settings, client=QdrantClient(":memory:"))
    qdrant.ensure_collection(embedder.vector_size)

    return ConversationAgent(
        completion_client=MockCompletionClient(),
        scene_lookup=SceneLookupTool(embedder, qdrant, settings),
        settings=settings,
        cast_lookup=None,
        character_lookup=CharacterLookupTool(store),
    )


def test_before_reveal_does_not_leak_relationship(agent: ConversationAgent):
    answer = agent.answer(
        title_id=TITLE,
        current_ts=40.0,
        question=QUESTION,
        user_id="viewer-1",
    )
    text = answer.text.lower()
    assert "seen them" in text  # acknowledges prior appearances
    assert "sister" not in text
    assert "sibling" not in text


def test_after_reveal_surfaces_relationship(agent: ConversationAgent):
    answer = agent.answer(
        title_id=TITLE,
        current_ts=130.0,
        question=QUESTION,
        user_id="viewer-1",
    )
    text = answer.text.lower()
    assert "sibling" in text or "sister" in text
