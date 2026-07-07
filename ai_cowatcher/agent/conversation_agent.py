"""Single orchestrating conversation agent with tool-calling."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from ai_cowatcher.agent.completion import CompletionClient, build_completion_client
from ai_cowatcher.agent.prompts import CONVERSATION_SYSTEM_PROMPT
from ai_cowatcher.agent.tier_routing import TierRouter, build_tier_router
from ai_cowatcher.agent.token_usage import TokenUsage
from ai_cowatcher.agent.tools import CAST_LOOKUP_TOOL, SCENE_LOOKUP_TOOL
from ai_cowatcher.config import Settings
from ai_cowatcher.retrieval.cast_lookup import CastLookupTool
from ai_cowatcher.retrieval.scene_lookup import SceneLookupTool

logger = logging.getLogger(__name__)

_MAX_TOOL_ROUNDS = 3
_UNKNOWN_ANSWER = "I don't know yet based on what has aired so far."


@dataclass(frozen=True)
class AgentAnswer:
    text: str
    model_tier: str
    model_name: str
    escalation_reason: str
    used_context: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class ConversationAgent:
    """One agent that calls scene_lookup as needed and synthesizes an answer."""

    def __init__(
        self,
        completion_client: CompletionClient,
        scene_lookup: SceneLookupTool,
        settings: Settings,
        tier_router: TierRouter | None = None,
        cast_lookup: CastLookupTool | None = None,
    ):
        self._completion = completion_client
        self._scene_lookup = scene_lookup
        self._settings = settings
        self._tier_router = tier_router or build_tier_router(settings, completion_client)
        self._cast_lookup = cast_lookup

    def _available_tools(self) -> list[dict[str, Any]]:
        tools = [SCENE_LOOKUP_TOOL]
        if self._cast_lookup is not None:
            tools.append(CAST_LOOKUP_TOOL)
        return tools

    def answer(
        self,
        *,
        title_id: str,
        current_ts: float,
        question: str,
        user_id: str,
        title_display_name: str | None = None,
    ) -> AgentAnswer:
        del user_id  # reserved for future personalization

        tier_selection = self._tier_router.select_tier(question)
        tier_decision = tier_selection.decision
        usage = tier_selection.usage or TokenUsage.empty()

        confident_title = self._settings.resolve_title_display_name(
            title_id, title_display_name
        )
        search_title = confident_title or self._settings.derive_title_from_id(title_id)

        text, loop_usage, used_context = self._run_tool_loop(
            question=question,
            title_id=title_id,
            current_ts=current_ts,
            tier_decision=tier_decision,
            confident_title=confident_title,
            search_title=search_title,
        )
        usage = usage.merge(loop_usage)

        return AgentAnswer(
            text=text,
            model_tier=tier_decision.tier,
            model_name=tier_decision.model,
            escalation_reason=tier_decision.reason,
            used_context=used_context,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    def _run_tool_loop(
        self,
        *,
        question: str,
        title_id: str,
        current_ts: float,
        tier_decision,
        confident_title: str | None = None,
        search_title: str | None = None,
    ) -> tuple[str, TokenUsage | None, bool]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": CONVERSATION_SYSTEM_PROMPT}
        ]
        if self._cast_lookup is not None:
            hint = self._cast_title_hint(confident_title, search_title)
            if hint:
                messages.append({"role": "system", "content": hint})
        messages.append({"role": "user", "content": question})

        usage = TokenUsage.empty()
        used_context = False
        tools = self._available_tools()

        for _ in range(_MAX_TOOL_ROUNDS):
            result = self._completion.complete(
                model=tier_decision.model,
                messages=messages,
                tools=tools,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
            )
            usage = usage.merge(result.usage)

            if not result.tool_calls:
                if result.content:
                    return result.content.strip(), usage, used_context
                return _UNKNOWN_ANSWER, usage, used_context

            for tool_call in result.tool_calls:
                payload, hit_context = self._dispatch_tool(
                    tool_call,
                    question=question,
                    title_id=title_id,
                    current_ts=current_ts,
                    search_title=search_title,
                )
                if payload is None:
                    logger.warning("Ignoring unsupported tool call: %s", tool_call.name)
                    continue
                used_context = used_context or hit_context

                messages.append(
                    {
                        "role": "assistant",
                        "content": result.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": json.dumps(tool_call.arguments),
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": json.dumps(payload),
                    }
                )

        return _UNKNOWN_ANSWER, usage, used_context

    @staticmethod
    def _cast_title_hint(
        confident_title: str | None, search_title: str | None
    ) -> str | None:
        """Guide the model on what title to pass to cast_lookup."""
        if confident_title:
            return (
                f'The viewer is watching "{confident_title}". '
                "If they ask about actors or cast, call cast_lookup with this title name."
            )
        if search_title:
            return (
                f'The viewer\'s content id likely corresponds to the title "{search_title}". '
                "If they ask about actors or cast, call cast_lookup using your best guess of "
                "the real title (start with this one)."
            )
        return None

    def _dispatch_tool(
        self,
        tool_call,
        *,
        question: str,
        title_id: str,
        current_ts: float,
        search_title: str | None = None,
    ) -> tuple[Any, bool]:
        """Return (payload, used_context). payload is None for unsupported tools."""
        if tool_call.name == "scene_lookup":
            query_text = str(tool_call.arguments.get("query_text", question))
            hits = self._scene_lookup.lookup(
                title_id=title_id,
                query_text=query_text,
                current_ts=current_ts,
            )
            return [hit.to_tool_dict() for hit in hits], bool(hits)

        if tool_call.name == "cast_lookup" and self._cast_lookup is not None:
            title_name = (
                str(tool_call.arguments.get("title_name", ""))
                or search_title
                or self._settings.effective_search_title(title_id)
                or ""
            )
            year = tool_call.arguments.get("year")
            result = self._cast_lookup.lookup(
                title_name=title_name,
                year=int(year) if isinstance(year, (int, str)) and str(year).isdigit() else None,
            )
            return result, bool(result.get("cast"))

        return None, False


def build_conversation_agent(
    settings: Settings,
    scene_lookup: SceneLookupTool,
    completion_client: CompletionClient | None = None,
    tier_router: TierRouter | None = None,
    cast_lookup: CastLookupTool | None = None,
) -> ConversationAgent:
    completion = completion_client or build_completion_client(settings)
    return ConversationAgent(
        completion_client=completion,
        scene_lookup=scene_lookup,
        settings=settings,
        tier_router=tier_router or build_tier_router(settings, completion),
        cast_lookup=cast_lookup,
    )
