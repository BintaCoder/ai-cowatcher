"""Tiered model routing for the conversation agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Protocol

from ai_cowatcher.agent.completion import CompletionClient
from ai_cowatcher.agent.token_usage import TokenUsage
from ai_cowatcher.config import Settings

logger = logging.getLogger(__name__)

ModelTier = Literal["fast", "escalated"]

_CLASSIFIER_SYSTEM_PROMPT = """You classify viewer questions about a TV show.
Reply with only YES or NO.

Answer YES when the question needs nuanced reasoning, such as:
- explaining motivations or themes
- comparing characters or scenes
- synthesizing information across multiple moments
- interpreting symbolism or foreshadowing

Answer NO for straightforward factual lookups about what has happened."""


@dataclass(frozen=True)
class ModelTierDecision:
    tier: ModelTier
    model: str
    reason: str


@dataclass(frozen=True)
class TierSelectionResult:
    decision: ModelTierDecision
    usage: TokenUsage | None = None


class EscalationClassifier(Protocol):
    def should_escalate(self, question: str) -> tuple[bool, str]:
        ...


class HeuristicEscalationClassifier:
    """Escalate on question length or configured keywords."""

    def __init__(self, settings: Settings):
        self._min_chars = settings.llm_escalation_min_chars
        self._keywords = settings.llm_escalation_keyword_list

    def should_escalate(self, question: str) -> tuple[bool, str]:
        stripped = question.strip()
        if len(stripped) >= self._min_chars:
            return True, f"question_length>={self._min_chars}"

        lower = stripped.lower()
        for keyword in self._keywords:
            if keyword in lower:
                return True, f"keyword:{keyword}"

        return False, "default_fast_tier"


class PromptEscalationClassifier:
    """Cheap classification call on the fast tier model."""

    def __init__(self, settings: Settings, completion_client: CompletionClient):
        self._settings = settings
        self._completion = completion_client

    def classify(self, question: str) -> tuple[bool, str, TokenUsage | None]:
        result = self._completion.complete(
            model=self._settings.conversation_fast_model,
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=8,
        )
        answer = (result.content or "").strip().upper()
        if answer.startswith("YES"):
            return True, "prompt_classifier:yes", result.usage
        return False, "prompt_classifier:no", result.usage

    def should_escalate(self, question: str) -> tuple[bool, str]:
        escalate, reason, _ = self.classify(question)
        return escalate, reason


class TierRouter:
    """Select fast vs escalated conversation model for a question."""

    def __init__(self, settings: Settings, classifier: EscalationClassifier):
        self._settings = settings
        self._classifier = classifier

    def select_tier(self, question: str) -> TierSelectionResult:
        if isinstance(self._classifier, PromptEscalationClassifier):
            escalate, reason, usage = self._classifier.classify(question)
        else:
            escalate, reason = self._classifier.should_escalate(question)
            usage = None

        if escalate:
            return TierSelectionResult(
                decision=ModelTierDecision(
                    tier="escalated",
                    model=self._settings.conversation_escalated_model,
                    reason=reason,
                ),
                usage=usage,
            )
        return TierSelectionResult(
            decision=ModelTierDecision(
                tier="fast",
                model=self._settings.conversation_fast_model,
                reason=reason,
            ),
            usage=usage,
        )


def build_escalation_classifier(
    settings: Settings,
    completion_client: CompletionClient | None = None,
) -> EscalationClassifier:
    if settings.llm_escalation_strategy == "prompt":
        if completion_client is None:
            raise ValueError("completion_client is required for prompt escalation strategy")
        return PromptEscalationClassifier(settings, completion_client)
    return HeuristicEscalationClassifier(settings)


def build_tier_router(
    settings: Settings,
    completion_client: CompletionClient | None = None,
) -> TierRouter:
    classifier = build_escalation_classifier(settings, completion_client)
    logger.debug(
        "Tier router ready strategy=%s fast=%s escalated=%s",
        settings.llm_escalation_strategy,
        settings.conversation_fast_model,
        settings.conversation_escalated_model,
    )
    return TierRouter(settings, classifier)
