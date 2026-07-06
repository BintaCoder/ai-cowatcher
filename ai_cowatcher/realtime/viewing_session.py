"""Viewing session orchestrator for real-time co-watcher questions."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ai_cowatcher.agent.completion import CompletionClient, build_completion_client
from ai_cowatcher.agent.conversation_agent import ConversationAgent, build_conversation_agent
from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.interfaces import TextEmbedder
from ai_cowatcher.observability.ask_telemetry import AskRecord, is_dont_know_answer, record_ask_request
from ai_cowatcher.providers import mock
from ai_cowatcher.providers.real import BgeM3Embedder
from ai_cowatcher.retrieval.scene_lookup import SceneLookupTool
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore


@dataclass
class AskResult:
    answer: str
    title_id: str
    user_id: str
    current_ts: float
    model_tier: str
    model_name: str
    escalation_reason: str
    used_context: bool
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class ViewingSession:
    """Coordinates the conversation agent for a single viewer question."""

    def __init__(self, agent: ConversationAgent):
        self._agent = agent

    def ask(
        self,
        *,
        title_id: str,
        current_ts: float,
        question: str,
        user_id: str,
    ) -> AskResult:
        started = time.perf_counter()
        answer = self._agent.answer(
            title_id=title_id,
            current_ts=current_ts,
            question=question,
            user_id=user_id,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        record_ask_request(
            AskRecord(
                title_id=title_id,
                user_id=user_id,
                current_ts=current_ts,
                latency_ms=round(latency_ms, 2),
                model_tier=answer.model_tier,
                model_name=answer.model_name,
                escalation_reason=answer.escalation_reason,
                used_context=answer.used_context,
                dont_know=is_dont_know_answer(answer.text),
                prompt_tokens=answer.prompt_tokens,
                completion_tokens=answer.completion_tokens,
                total_tokens=answer.total_tokens,
            )
        )

        return AskResult(
            answer=answer.text,
            title_id=title_id,
            user_id=user_id,
            current_ts=current_ts,
            model_tier=answer.model_tier,
            model_name=answer.model_name,
            escalation_reason=answer.escalation_reason,
            used_context=answer.used_context,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
            total_tokens=answer.total_tokens,
        )


def _build_embedder(settings: Settings) -> TextEmbedder:
    if settings.mock_mode:
        return mock.MockTextEmbedder()
    return BgeM3Embedder(settings)


def build_viewing_session(
    settings: Settings | None = None,
    *,
    qdrant_store: QdrantSceneStore | None = None,
    embedder: TextEmbedder | None = None,
    completion_client: CompletionClient | None = None,
) -> ViewingSession:
    settings = settings or get_settings()
    qdrant = qdrant_store or QdrantSceneStore(settings)
    embedder = embedder or _build_embedder(settings)
    scene_lookup = SceneLookupTool(embedder, qdrant, settings)
    agent = build_conversation_agent(
        settings,
        scene_lookup,
        completion_client=completion_client,
    )
    return ViewingSession(agent)
