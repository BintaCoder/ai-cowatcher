"""Persona free-path + pilot merged path regression (prompt 5)."""

from __future__ import annotations

import time
import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ai_cowatcher.agent.completion import MockCompletionClient
from ai_cowatcher.config import Settings
from ai_cowatcher.observability import prometheus_metrics as pm
from ai_cowatcher.personas.loader import list_personas
from ai_cowatcher.providers.mock import MockTextEmbedder
from ai_cowatcher.realtime.viewing_session import build_viewing_session
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore


def _session(settings: Settings):
    embedder = MockTextEmbedder()
    client = QdrantClient(":memory:")
    store = QdrantSceneStore(settings, client=client)
    store.ensure_collection(embedder.vector_size)
    vector = embedder.embed_texts(["hello"])[0]
    client.upsert(
        collection_name=store._collection,
        points=[
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "title_id": "demo",
                    "scene_id": "s0000",
                    "start_ts": 0.0,
                    "end_ts": 30.0,
                    "transcript": "Ross walks into Central Perk looking upset.",
                    "caption": "",
                    "face_cluster_ids": [],
                },
            )
        ],
    )
    return build_viewing_session(
        settings,
        qdrant_store=store,
        embedder=embedder,
        completion_client=MockCompletionClient(),
        qa_cache=None,
    )


@pytest.fixture
def pilot_settings() -> Settings:
    return Settings(
        MOCK_MODE=True,
        PILOT_LOW_LATENCY=True,
        UTTERANCE_GATE_ENABLED=True,
        UTTERANCE_GATE_STRATEGY="merged",
        QA_CACHE_ENABLED=False,
        QDRANT_COLLECTION=f"test_persona_gate_{uuid.uuid4().hex[:8]}",
    )


@pytest.mark.parametrize("persona", list_personas(), ids=lambda p: p.persona_id)
@pytest.mark.parametrize(
    "question,kind",
    [
        ("You there?", "social"),
        ("Hey, how\'s it going?", "social"),
        ("Thanks, that helped", "social"),
        ("Hmm okay", "filler"),
        ("Wait—", "filler"),
    ],
)
def test_persona_social_filler_free_path(pilot_settings, persona, question, kind):
    session = _session(pilot_settings)
    t0 = time.perf_counter()
    result = session.ask(
        title_id="demo",
        current_ts=12.0,
        question=question,
        user_id="tester",
        persona_id=persona.persona_id,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 200.0, f"{persona.persona_id} {question!r} took {elapsed_ms:.1f}ms"
    assert result.model_tier == "gate"
    assert result.model_name == "gate:free"
    assert (result.escalation_reason or "").startswith("gate:")
    if kind == "filler":
        assert not result.speak
    else:
        assert result.speak
        assert result.answer == persona.canned_social_reply


def test_all_personas_content_keeps_legacy_counter_flat(pilot_settings):
    before = 0.0
    for metric in pm.LEGACY_TOOL_PATH_TOTAL.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                before += float(sample.value)

    session = _session(pilot_settings)
    for persona in list_personas():
        result = session.ask(
            title_id="demo",
            current_ts=12.0,
            question="What just happened?",
            user_id="tester",
            persona_id=persona.persona_id,
        )
        assert result.model_tier != "gate" or result.escalation_reason  # content path
        # Free path not taken for content; but not legacy either under pilot.
        assert result.model_name != ""

    after = 0.0
    for metric in pm.LEGACY_TOOL_PATH_TOTAL.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                after += float(sample.value)
    assert after == before


def test_unknown_persona_falls_back_to_default_not_legacy(pilot_settings):
    session = _session(pilot_settings)
    before = 0.0
    for metric in pm.LEGACY_TOOL_PATH_TOTAL.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                before += float(sample.value)
    result = session.ask(
        title_id="demo",
        current_ts=12.0,
        question="You there?",
        user_id="tester",
        persona_id="not_a_real_persona_xyz",
    )
    assert result.model_name == "gate:free"
    after = 0.0
    for metric in pm.LEGACY_TOOL_PATH_TOTAL.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                after += float(sample.value)
    assert after == before
