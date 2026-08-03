"""Tests for the utterance gate (noise ignore + intents + ambiguous classify)."""

from __future__ import annotations

import pytest

from ai_cowatcher.agent.completion import MockCompletionClient
from ai_cowatcher.agent.utterance_gate import classify_utterance
from ai_cowatcher.config import Settings


@pytest.fixture
def gate_settings() -> Settings:
    return Settings(
        MOCK_MODE=True,
        UTTERANCE_GATE_ENABLED=True,
        UTTERANCE_GATE_STRATEGY="heuristic",
    )


@pytest.mark.parametrize(
    "question,action",
    [
        ("um", "ignore"),
        ("uh huh", "ignore"),
        ("hmm", "ignore"),
        ("hi", "social"),
        ("thanks", "social"),
        ("what's the weather", "off_topic"),
        ("joke", "joke"),
        ("one liner", "joke"),
        ("make me laugh", "joke"),
        ("what just happened?", "content"),
        ("who is that?", "content"),
        ("go to 10:00", "navigate"),
        ("skip to the credits", "navigate"),
        ("where does he appear", "navigate"),
    ],
)
def test_heuristic_gate_actions(gate_settings: Settings, question: str, action: str):
    decision = classify_utterance(question, settings=gate_settings, completion=None)
    assert decision.action == action


def test_ignore_has_silent_reply(gate_settings: Settings):
    decision = classify_utterance("um", settings=gate_settings)
    assert decision.speak is False
    assert decision.reply == ""
    assert decision.short_circuit is True


def test_content_not_short_circuit(gate_settings: Settings):
    decision = classify_utterance("What just happened?", settings=gate_settings)
    assert decision.short_circuit is False


def test_merged_strategy_defers_ambiguous_to_agent():
    settings = Settings(
        MOCK_MODE=True,
        UTTERANCE_GATE_ENABLED=True,
        UTTERANCE_GATE_STRATEGY="merged",
    )
    decision = classify_utterance(
        "purple banana twelve",
        settings=settings,
        completion=MockCompletionClient(),
    )
    assert decision.action == "content"
    assert decision.reason == "gate:merged_pending"
    assert decision.short_circuit is False


def test_prompt_strategy_on_ambiguous():
    settings = Settings(
        MOCK_MODE=True,
        UTTERANCE_GATE_ENABLED=True,
        UTTERANCE_GATE_STRATEGY="prompt",
    )
    client = MockCompletionClient()
    decision = classify_utterance(
        "purple banana twelve",
        settings=settings,
        completion=client,
    )
    assert decision.action == "ignore"
    assert "prompt" in decision.reason

    yes_case = classify_utterance(
        "is he the same person?",
        settings=settings,
        completion=client,
    )
    assert yes_case.action == "content"
