"""Tests for tiered conversation model routing."""

from __future__ import annotations

import json
import logging

import pytest
from qdrant_client import QdrantClient

from ai_cowatcher.agent.completion import MockCompletionClient
from ai_cowatcher.agent.metrics import conversation_tier_counts
from ai_cowatcher.agent.tier_routing import (
    HeuristicEscalationClassifier,
    TierRouter,
    build_tier_router,
)
from ai_cowatcher.config import Settings
from ai_cowatcher.providers.mock import MockTextEmbedder
from ai_cowatcher.realtime.viewing_session import ViewingSession, build_viewing_session
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore


@pytest.fixture
def tier_settings() -> Settings:
    return Settings(
        MOCK_MODE=True,
        QDRANT_COLLECTION="test_tier_routing",
        LLM_ESCALATION_STRATEGY="heuristic",
        LLM_ESCALATION_MIN_CHARS=80,
        LLM_ESCALATION_KEYWORDS="why,explain,compare,theme",
        LLM_MOCK_TIER_FAST_MODEL="mock-llm-fast",
        LLM_MOCK_TIER_ESCALATED_MODEL="mock-llm-escalated",
    )


def _build_viewing_session(settings: Settings) -> ViewingSession:
    client = QdrantClient(":memory:")
    qdrant = QdrantSceneStore(settings, client=client)
    embedder = MockTextEmbedder()
    qdrant.ensure_collection(embedder.vector_size)
    return build_viewing_session(
        settings,
        qdrant_store=qdrant,
        embedder=embedder,
        completion_client=MockCompletionClient(),
    )


def test_heuristic_classifier_stays_fast_for_short_factual_question(tier_settings: Settings):
    classifier = HeuristicEscalationClassifier(tier_settings)
    escalate, reason = classifier.should_escalate("Who is in this scene?")
    assert escalate is False
    assert reason == "default_fast_tier"


def test_heuristic_classifier_escalates_on_keyword(tier_settings: Settings):
    classifier = HeuristicEscalationClassifier(tier_settings)
    escalate, reason = classifier.should_escalate("Why did she leave the room?")
    assert escalate is True
    assert reason == "keyword:why"


def test_heuristic_classifier_escalates_on_length(tier_settings: Settings):
    classifier = HeuristicEscalationClassifier(tier_settings)
    question = "What happened in the opening scene before the detectives arrived at the building?"
    assert len(question.strip()) >= tier_settings.llm_escalation_min_chars
    escalate, reason = classifier.should_escalate(question)
    assert escalate is True
    assert reason.startswith("question_length>=")


def test_tier_router_selects_fast_or_escalated_model(tier_settings: Settings):
    router = TierRouter(tier_settings, HeuristicEscalationClassifier(tier_settings))

    fast = router.select_tier("Who is in this scene?")
    assert fast.decision.tier == "fast"
    assert fast.decision.model == "mock-llm-fast"

    escalated = router.select_tier("Explain the detective's motivation.")
    assert escalated.decision.tier == "escalated"
    assert escalated.decision.model == "mock-llm-escalated"
    assert escalated.decision.reason == "keyword:explain"


def test_conversation_agent_uses_escalated_model_and_records_telemetry(
    tier_settings: Settings,
    caplog: pytest.LogCaptureFixture,
):
    session = _build_viewing_session(tier_settings)

    with caplog.at_level(logging.INFO, logger="ai_cowatcher.ask"):
        result = session.ask(
            title_id="demo",
            current_ts=10.0,
            question="Explain why the detective lied to his partner.",
            user_id="viewer-1",
        )

    assert result.model_tier == "escalated"
    assert result.model_name == "mock-llm-escalated"
    assert conversation_tier_counts()["escalated"] == 1
    assert any('"event":"ask_request"' in record.message for record in caplog.records)
    payload = json.loads(caplog.records[-1].message)
    assert payload["model_tier"] == "escalated"
    assert payload["title_id"] == "demo"
    assert "latency_ms" in payload


def test_conversation_agent_uses_fast_model_for_simple_question(
    tier_settings: Settings,
    caplog: pytest.LogCaptureFixture,
):
    session = _build_viewing_session(tier_settings)

    with caplog.at_level(logging.INFO, logger="ai_cowatcher.ask"):
        result = session.ask(
            title_id="demo",
            current_ts=10.0,
            question="Who is in this scene?",
            user_id="viewer-2",
        )

    assert result.model_tier == "fast"
    assert result.model_name == "mock-llm-fast"
    assert conversation_tier_counts()["fast"] == 1
    payload = json.loads(caplog.records[-1].message)
    assert payload["model_tier"] == "fast"


def test_prompt_strategy_escalates_via_classifier_call(tier_settings: Settings):
    settings = tier_settings.model_copy(
        update={"llm_escalation_strategy": "prompt"},
    )
    completion = MockCompletionClient()
    router = build_tier_router(settings, completion)

    selection = router.select_tier("Explain the theme of betrayal in this episode.")
    assert selection.decision.tier == "escalated"
    assert selection.decision.reason == "prompt_classifier:yes"
    assert completion.models_used == [settings.conversation_fast_model]
    assert selection.usage is not None
    assert selection.usage.total_tokens == 25
