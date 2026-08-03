"""Single orchestrating conversation agent with tool-calling."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ai_cowatcher.agent.brevity import enforce_brief_answer
from ai_cowatcher.agent.completion import (
    CompletionClient,
    assistant_tool_calls_message,
    build_completion_client,
    _chunk_text_for_stream,
)
from ai_cowatcher.agent.grounding import grounded_fallback_answer, is_refusal_answer
from ai_cowatcher.agent.intent_tags import (
    IntentStreamParser,
    MERGED_SYSTEM_PROMPT,
    build_merged_user_turn,
    parse_tagged_answer,
    social_body_or_default,
)
from ai_cowatcher.agent.joke_intent import (
    is_joke_request,
    joke_fallback_answer,
    joke_scene_query,
    soft_no_scene_joke,
)
from ai_cowatcher.agent.multimodal import (
    build_multimodal_messages,
    collect_audio_keys,
    load_scene_clips,
    should_use_multimodal,
)
from ai_cowatcher.agent.prompts import CONVERSATION_SYSTEM_PROMPT, JOKE_MODE_SYSTEM_PROMPT
from ai_cowatcher.agent.stream_events import AskStreamEvent
from ai_cowatcher.agent.tier_routing import ModelTierDecision, TierRouter, build_tier_router
from ai_cowatcher.agent.token_usage import TokenUsage
from ai_cowatcher.agent.tools import (
    CAST_LOOKUP_TOOL,
    CHARACTER_LOOKUP_TOOL,
    KNOWLEDGE_SEARCH_TOOL,
    SCENE_LOOKUP_TOOL,
    USER_MEMORY_TOOL,
)
from ai_cowatcher.config import Settings
from ai_cowatcher.observability.prometheus_metrics import observe_tool_call
from ai_cowatcher.retrieval.cast_lookup import CastLookupTool
from ai_cowatcher.retrieval.character_lookup import CharacterLookupTool
from ai_cowatcher.retrieval.knowledge_search import KnowledgeSearchTool
from ai_cowatcher.retrieval.scene_lookup import SceneLookupTool
from ai_cowatcher.retrieval.user_memory import UserMemoryTool
from ai_cowatcher.storage.object_store import ObjectStore

logger = logging.getLogger(__name__)

_MAX_TOOL_ROUNDS = 3
_UNKNOWN_ANSWER = "Not sure yet — nothing's made that clear so far."

_TOOL_STATUS = {
    "scene_lookup": "Searching scenes…",
    "character_lookup": "Checking character history…",
    "cast_lookup": "Looking up cast…",
    "knowledge_search": "Searching production knowledge…",
    "user_memory": "Recalling earlier conversation…",
}


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
    skip_memory: bool = False
    speak: bool = True


class ConversationAgent:
    """One agent that calls scene_lookup as needed and synthesizes an answer."""

    def __init__(
        self,
        completion_client: CompletionClient,
        scene_lookup: SceneLookupTool,
        settings: Settings,
        tier_router: TierRouter | None = None,
        cast_lookup: CastLookupTool | None = None,
        character_lookup: CharacterLookupTool | None = None,
        knowledge_search: KnowledgeSearchTool | None = None,
        user_memory: UserMemoryTool | None = None,
        object_store: ObjectStore | None = None,
    ):
        self._completion = completion_client
        self._scene_lookup = scene_lookup
        self._settings = settings
        self._tier_router = tier_router or build_tier_router(settings, completion_client)
        self._cast_lookup = cast_lookup
        self._character_lookup = character_lookup
        self._knowledge_search = knowledge_search
        self._user_memory = user_memory
        self._object_store = object_store

    def _gated_answer(
        self,
        *,
        question: str,
    ) -> tuple[AgentAnswer | None, TokenUsage]:
        from ai_cowatcher.agent.utterance_gate import classify_utterance

        decision = classify_utterance(
            question,
            settings=self._settings,
            completion=self._completion,
        )
        usage = decision.usage or TokenUsage.empty()
        if not decision.short_circuit:
            return None, usage

        model = self._settings.conversation_fast_model
        return (
            AgentAnswer(
                text=decision.reply,
                model_tier="fast",
                model_name=model,
                escalation_reason=decision.reason,
                used_context=False,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                skip_memory=decision.skip_memory or decision.action == "ignore",
                speak=decision.speak and bool(decision.reply.strip()),
            ),
            usage,
        )

    def _available_tools(self) -> list[dict[str, Any]]:
        tools = [SCENE_LOOKUP_TOOL]
        if self._user_memory is not None:
            tools.append(USER_MEMORY_TOOL)
        if self._knowledge_search is not None:
            tools.append(KNOWLEDGE_SEARCH_TOOL)
        if self._character_lookup is not None:
            tools.append(CHARACTER_LOOKUP_TOOL)
        if self._cast_lookup is not None:
            tools.append(CAST_LOOKUP_TOOL)
        return tools

    def _use_merged_intent(self) -> bool:
        # Pilot low-latency always uses merged (1 retrieve + 1 LLM), never multi-tool hops.
        if getattr(self._settings, "pilot_low_latency", False):
            return True
        return (
            bool(getattr(self._settings, "utterance_gate_enabled", True))
            and getattr(self._settings, "utterance_gate_strategy", "merged") == "merged"
        )

    def answer(
        self,
        *,
        title_id: str,
        current_ts: float,
        question: str,
        user_id: str,
        title_display_name: str | None = None,
    ) -> AgentAnswer:
        gated, gate_usage = self._gated_answer(question=question)
        if gated is not None:
            return gated

        if self._use_merged_intent():
            return self._answer_merged(
                title_id=title_id,
                current_ts=current_ts,
                question=question,
                user_id=user_id,
                title_display_name=title_display_name,
                gate_usage=gate_usage,
            )

        tier_selection = self._tier_router.select_tier(question)
        tier_decision = tier_selection.decision
        usage = gate_usage.merge(tier_selection.usage or TokenUsage.empty())

        confident_title = self._settings.resolve_title_display_name(
            title_id, title_display_name
        )
        search_title = confident_title or self._settings.derive_title_from_id(title_id)

        if is_joke_request(question):
            tier_decision = self._force_fast_tier(tier_decision, reason="joke_request")
            text, loop_usage, used_context = self._run_joke_loop(
                question=question,
                title_id=title_id,
                current_ts=current_ts,
                tier_decision=tier_decision,
                confident_title=confident_title,
                search_title=search_title,
            )
            reason = "joke_request"
        else:
            text, loop_usage, used_context = self._run_tool_loop(
                question=question,
                title_id=title_id,
                current_ts=current_ts,
                user_id=user_id,
                tier_decision=tier_decision,
                confident_title=confident_title,
                search_title=search_title,
            )
            reason = tier_decision.reason
        usage = usage.merge(loop_usage)

        return AgentAnswer(
            text=text,
            model_tier=tier_decision.tier,
            model_name=tier_decision.model,
            escalation_reason=reason,
            used_context=used_context,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            skip_memory=False,
            speak=True,
        )

    def answer_stream(
        self,
        *,
        title_id: str,
        current_ts: float,
        question: str,
        user_id: str,
        title_display_name: str | None = None,
    ) -> Iterator[AskStreamEvent]:
        """Yield progressive status/token events and a final ``done`` event."""
        yield AskStreamEvent(type="status", message="Listening for intent…")

        gated, gate_usage = self._gated_answer(question=question)
        if gated is not None:
            if gated.text:
                yield AskStreamEvent(type="status", message="Quick reply…")
                for piece in _chunk_text_for_stream(gated.text):
                    yield AskStreamEvent(type="token", text=piece)
            else:
                yield AskStreamEvent(
                    type="status",
                    message="No clear question — try “what just happened?” or “who’s that?”",
                )
            yield AskStreamEvent(
                type="done",
                answer=gated.text,
                model_tier=gated.model_tier,
                model_name=gated.model_name,
                escalation_reason=gated.escalation_reason,
                used_context=False,
                speak=gated.speak,
                skip_memory=gated.skip_memory,
                intent=_intent_from_gate_reason(gated.escalation_reason),
            )
            del gate_usage
            return

        if self._use_merged_intent():
            yield from self._answer_merged_stream(
                title_id=title_id,
                current_ts=current_ts,
                question=question,
                user_id=user_id,
                title_display_name=title_display_name,
                gate_usage=gate_usage,
            )
            return

        yield AskStreamEvent(type="status", message="Routing question…")

        tier_selection = self._tier_router.select_tier(question)
        tier_decision = tier_selection.decision
        usage = gate_usage.merge(tier_selection.usage or TokenUsage.empty())

        confident_title = self._settings.resolve_title_display_name(
            title_id, title_display_name
        )
        search_title = confident_title or self._settings.derive_title_from_id(title_id)

        if is_joke_request(question):
            tier_decision = self._force_fast_tier(tier_decision, reason="joke_request")
            yield AskStreamEvent(type="status", message="Cooking a one-liner…")
            text, loop_usage, used_context = yield from self._run_joke_loop_stream(
                question=question,
                title_id=title_id,
                current_ts=current_ts,
                tier_decision=tier_decision,
                confident_title=confident_title,
                search_title=search_title,
            )
            reason = "joke_request"
        else:
            yield AskStreamEvent(type="status", message="Thinking…")
            text, loop_usage, used_context = yield from self._run_tool_loop_stream(
                question=question,
                title_id=title_id,
                current_ts=current_ts,
                user_id=user_id,
                tier_decision=tier_decision,
                confident_title=confident_title,
                search_title=search_title,
            )
            reason = tier_decision.reason
        usage = usage.merge(loop_usage)

        yield AskStreamEvent(
            type="done",
            answer=text,
            model_tier=tier_decision.tier,
            model_name=tier_decision.model,
            escalation_reason=reason,
            used_context=used_context,
            speak=True,
            skip_memory=False,
        )
        del usage

    def _prefetch_scene_evidence(
        self,
        *,
        question: str,
        title_id: str,
        current_ts: float,
    ) -> tuple[list[Any], str]:
        """Speculative scene_lookup so the tagged completion is grounded."""
        query = joke_scene_query(question) if is_joke_request(question) else question
        with observe_tool_call("scene_lookup"):
            hits = self._scene_lookup.lookup(
                title_id=title_id,
                query_text=query,
                current_ts=current_ts,
            )
        payload = [hit.to_tool_dict() for hit in hits]
        if not payload:
            return [], "(no scene matches)"
        return [payload], json.dumps(payload, ensure_ascii=False)

    def _merged_messages(
        self,
        *,
        title_id: str,
        current_ts: float,
        question: str,
        scene_evidence: str,
    ) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": MERGED_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_merged_user_turn(
                    title_id=title_id,
                    current_ts=current_ts,
                    question=question,
                    scene_evidence=scene_evidence,
                ),
            },
        ]

    def _resolve_merged_tag(
        self,
        *,
        tag: str,
        body: str,
        question: str,
        tool_payloads: list[Any],
        used_context: bool,
    ) -> tuple[str, bool, bool, bool, str]:
        """Return (text, speak, skip_memory, navigate, reason)."""
        joke_mode = tag == "JOKE" or is_joke_request(question)
        if tag == "FILLER":
            return "", False, True, False, "merged:FILLER"
        if tag == "NAVIGATE":
            return "", False, True, True, "merged:NAVIGATE"
        if tag == "SOCIAL":
            text = enforce_brief_answer(
                social_body_or_default(body), question, joke_mode=False
            )
            return text, True, True, False, "merged:SOCIAL"
        if tag == "JOKE":
            text, used = self._finalize_answer(
                raw=body or None,
                question=question,
                tool_payloads=tool_payloads,
                used_context=used_context,
                joke_mode=True,
                force_lookup_if_empty=False,
            )
            return text, bool(text.strip()), False, False, "merged:JOKE"
        # CONTENT (or unknown → already treated as CONTENT by parser)
        if used_context and not (body or "").strip():
            body = grounded_fallback_answer(question, tool_payloads) or _UNKNOWN_ANSWER
        text, _used = self._finalize_answer(
            raw=body or None,
            question=question,
            tool_payloads=tool_payloads,
            used_context=used_context,
            joke_mode=False,
            force_lookup_if_empty=False,
        )
        if joke_mode and tag == "CONTENT":
            # Rare: body tagged CONTENT but question was a joke request
            pass
        return text, bool(text.strip()), False, False, f"merged:{tag}"

    def _answer_merged(
        self,
        *,
        title_id: str,
        current_ts: float,
        question: str,
        user_id: str,
        title_display_name: str | None,
        gate_usage: TokenUsage,
    ) -> AgentAnswer:
        del user_id, title_display_name  # evidence path only for now
        model = self._settings.conversation_fast_model
        payloads, scene_evidence = self._prefetch_scene_evidence(
            question=question, title_id=title_id, current_ts=current_ts
        )
        used_context = bool(payloads)
        messages = self._merged_messages(
            title_id=title_id,
            current_ts=current_ts,
            question=question,
            scene_evidence=scene_evidence,
        )
        result = self._completion.complete(
            model=model,
            messages=messages,
            tools=None,
            temperature=self._settings.llm_temperature,
            max_tokens=self._answer_max_tokens(),
        )
        usage = gate_usage.merge(result.usage or TokenUsage.empty())
        parsed = parse_tagged_answer(result.content)

        tag = parsed.tag
        body = parsed.body
        # If multimodal preferred for CONTENT, only run it when text body is weak.
        if tag == "CONTENT" and payloads:
            body_ok = bool((body or "").strip()) and not is_refusal_answer(body or "")
            if body_ok:
                reason_suffix = ""
            else:
                mm = self._try_multimodal_answer(
                    question=question,
                    tool_payloads=payloads,
                    model=model,
                )
                if mm is not None:
                    mm_text, mm_usage = mm
                    if mm_usage is not None:
                        usage = usage.merge(mm_usage)
                    body = mm_text
                    used_context = True
                    reason_suffix = "+multimodal"
                else:
                    reason_suffix = ""
        else:
            reason_suffix = ""

        text, speak, skip_memory, navigate, reason = self._resolve_merged_tag(
            tag=tag,
            body=body,
            question=question,
            tool_payloads=payloads,
            used_context=used_context,
        )
        del navigate  # non-stream: client still does local navigate heuristics
        return AgentAnswer(
            text=text,
            model_tier="fast",
            model_name=model,
            escalation_reason=reason + reason_suffix,
            used_context=used_context,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            skip_memory=skip_memory,
            speak=speak,
        )

    def _answer_merged_stream(
        self,
        *,
        title_id: str,
        current_ts: float,
        question: str,
        user_id: str,
        title_display_name: str | None,
        gate_usage: TokenUsage,
    ) -> Iterator[AskStreamEvent]:
        del user_id, title_display_name
        model = self._settings.conversation_fast_model
        yield AskStreamEvent(type="status", message="Checking scenes…")
        yield AskStreamEvent(
            type="tool_start", tool="scene_lookup", message="Checking scenes…"
        )
        payloads, scene_evidence = self._prefetch_scene_evidence(
            question=question, title_id=title_id, current_ts=current_ts
        )
        yield AskStreamEvent(
            type="tool_end", tool="scene_lookup", message="scene_lookup done"
        )
        used_context = bool(payloads)
        messages = self._merged_messages(
            title_id=title_id,
            current_ts=current_ts,
            question=question,
            scene_evidence=scene_evidence,
        )

        yield AskStreamEvent(type="status", message="Answering…")
        parser = IntentStreamParser()
        tag: str | None = None
        body_parts: list[str] = []
        emit_tokens = True  # set False for FILLER / NAVIGATE after tag

        # Prefer multimodal only when it won't stack on a text-only path already good
        # enough; streaming wants tokens ASAP (merged path). Full multimodal still for empty clips path.
        prefer_mm = (
            payloads
            and should_use_multimodal(self._settings)
            and self._object_store
            and getattr(self._settings, "multimodal_prefer_over_text", False)
        )
        if prefer_mm:
            mm = self._try_multimodal_answer(
                question=question,
                tool_payloads=payloads,
                model=model,
            )
            if mm is not None:
                mm_text, mm_usage = mm
                del mm_usage, gate_usage
                text, speak, skip_memory, navigate, reason = self._resolve_merged_tag(
                    tag="CONTENT",
                    body=mm_text,
                    question=question,
                    tool_payloads=payloads,
                    used_context=True,
                )
                yield AskStreamEvent(type="intent", intent="CONTENT", message="CONTENT")
                for piece in _chunk_text_for_stream(text):
                    yield AskStreamEvent(type="token", text=piece)
                yield AskStreamEvent(
                    type="done",
                    answer=text,
                    model_tier="fast",
                    model_name=model,
                    escalation_reason=reason + "+multimodal",
                    used_context=True,
                    speak=speak and not navigate,
                    skip_memory=skip_memory,
                    intent="CONTENT",
                    navigate=navigate,
                )
                return

        for delta in self._completion.complete_stream(
            model=model,
            messages=messages,
            temperature=self._settings.llm_temperature,
            max_tokens=self._answer_max_tokens(),
        ):
            for kind, value in parser.feed(delta):
                if kind == "tag":
                    tag = value
                    yield AskStreamEvent(type="intent", intent=tag, message=tag)
                    if tag in ("FILLER", "NAVIGATE"):
                        emit_tokens = False
                elif kind == "text" and emit_tokens:
                    body_parts.append(value)
                    # Stream body only for SOCIAL / JOKE / CONTENT
                    if tag in ("SOCIAL", "JOKE", "CONTENT") or tag is None:
                        yield AskStreamEvent(type="token", text=value)

        for kind, value in parser.finish():
            if kind == "tag" and tag is None:
                tag = value
                yield AskStreamEvent(type="intent", intent=tag, message=tag)
                if tag in ("FILLER", "NAVIGATE"):
                    emit_tokens = False
            elif kind == "text" and emit_tokens:
                body_parts.append(value)
                if tag in ("SOCIAL", "JOKE", "CONTENT"):
                    yield AskStreamEvent(type="token", text=value)

        raw_body = "".join(body_parts)
        final_tag = tag or "CONTENT"
        text, speak, skip_memory, navigate, reason = self._resolve_merged_tag(
            tag=final_tag,
            body=raw_body,
            question=question,
            tool_payloads=payloads,
            used_context=used_context,
        )

        # If finalize rewrote answer (grounding), re-emit a clean last answer
        streamed = raw_body.strip()
        if (
            text
            and text.strip() != streamed
            and final_tag in ("CONTENT", "JOKE", "SOCIAL")
            and not navigate
        ):
            # Client already saw tokens; done.answer is source of truth
            pass

        del gate_usage
        yield AskStreamEvent(
            type="done",
            answer=text,
            model_tier="fast",
            model_name=model,
            escalation_reason=reason,
            used_context=used_context,
            speak=speak and not navigate,
            skip_memory=skip_memory,
            intent=final_tag,
            navigate=navigate,
        )


    def _force_fast_tier(self, _tier_decision: ModelTierDecision, *, reason: str) -> ModelTierDecision:
        return ModelTierDecision(
            tier="fast",
            model=self._settings.conversation_fast_model,
            reason=reason,
        )

    def _tool_max_tokens(self) -> int:
        return max(self._settings.llm_max_tokens, self._settings.llm_tool_max_tokens)

    def _answer_max_tokens(self) -> int:
        return self._settings.llm_max_tokens

    def _try_multimodal_answer(
        self,
        *,
        question: str,
        tool_payloads: list[Any],
        model: str,
    ) -> tuple[str, TokenUsage | None] | None:
        """Answer from retrieved scene audio + text when clips are available."""
        if not should_use_multimodal(self._settings):
            return None
        if self._object_store is None:
            return None
        keys = collect_audio_keys(
            tool_payloads, max_clips=self._settings.multimodal_max_clips
        )
        if not keys:
            return None
        clips = load_scene_clips(keys, self._object_store)
        if not clips:
            return None
        try:
            messages = build_multimodal_messages(
                question=question,
                tool_payloads=tool_payloads,
                clips=clips,
            )
            result = self._completion.complete(
                model=model,
                messages=messages,
                tools=None,
                temperature=self._settings.llm_temperature,
                max_tokens=self._answer_max_tokens(),
            )
            text = (result.content or "").strip()
            if not text:
                return None
            logger.info(
                "Multimodal answer used clips=%d bytes=%d",
                len(clips),
                sum(len(b) for _, b in clips),
            )
            return text, result.usage
        except Exception:  # noqa: BLE001 — fall back to text tool path
            logger.exception("Multimodal scene-audio completion failed; falling back")
            return None

    def _compose_final_answer(
        self,
        *,
        raw: str | None,
        question: str,
        tool_payloads: list[Any],
        used_context: bool,
        title_id: str | None,
        current_ts: float | None,
        model: str,
        joke_mode: bool = False,
    ) -> tuple[str, bool, TokenUsage | None]:
        raw_text = (raw or "").strip()
        # Multimodal is a second Gemini call (+ clip I/O). Only use it when text tools
        # did not already produce a usable answer — pilot latency otherwise multiplies.
        if not joke_mode and (not raw_text or is_refusal_answer(raw_text)):
            mm = self._try_multimodal_answer(
                question=question,
                tool_payloads=tool_payloads,
                model=model,
            )
            if mm is not None:
                text, usage = mm
                final, used = self._finalize_answer(
                    raw=text,
                    question=question,
                    tool_payloads=tool_payloads,
                    used_context=True,
                    title_id=title_id,
                    current_ts=current_ts,
                    force_lookup_if_empty=False,
                    joke_mode=False,
                )
                return final, used, usage
        final, used = self._finalize_answer(
            raw=raw,
            question=question,
            tool_payloads=tool_payloads,
            used_context=used_context,
            title_id=title_id,
            current_ts=current_ts,
            joke_mode=joke_mode,
        )
        return final, used, None

    def _force_scene_grounding(
        self,
        *,
        question: str,
        title_id: str,
        current_ts: float,
    ) -> list[Any]:
        """Always available backup retrieval when the model refuses without tools."""
        with observe_tool_call("scene_lookup"):
            hits = self._scene_lookup.lookup(
                title_id=title_id,
                query_text=question,
                current_ts=current_ts,
            )
        return [[hit.to_tool_dict() for hit in hits]] if hits else []

    def _finalize_answer(
        self,
        *,
        raw: str | None,
        question: str,
        tool_payloads: list[Any],
        used_context: bool,
        title_id: str | None = None,
        current_ts: float | None = None,
        force_lookup_if_empty: bool = True,
        joke_mode: bool = False,
    ) -> tuple[str, bool]:
        default = soft_no_scene_joke() if joke_mode else _UNKNOWN_ANSWER
        text = enforce_brief_answer(
            (raw or "").strip() or default,
            question,
            joke_mode=joke_mode,
        )
        payloads = list(tool_payloads)
        if is_refusal_answer(text) and not payloads and force_lookup_if_empty:
            if title_id is not None and current_ts is not None:
                query = joke_scene_query(question) if joke_mode else question
                payloads = self._force_scene_grounding(
                    question=query,
                    title_id=title_id,
                    current_ts=current_ts,
                )
        if is_refusal_answer(text) and payloads:
            if joke_mode:
                fallback = joke_fallback_answer(payloads)
            else:
                fallback = grounded_fallback_answer(question, payloads)
            if fallback:
                return enforce_brief_answer(
                    fallback, question, joke_mode=joke_mode
                ), True
        if joke_mode and not payloads:
            return soft_no_scene_joke(), False
        return text, used_context or bool(payloads)

    def _run_joke_loop(
        self,
        *,
        question: str,
        title_id: str,
        current_ts: float,
        tier_decision: ModelTierDecision,
        confident_title: str | None = None,
        search_title: str | None = None,
    ) -> tuple[str, TokenUsage | None, bool]:
        """Fast joke path: always ground on current scenes, then one short punchline."""
        messages, payloads, used_context = self._joke_grounded_messages(
            question=question,
            title_id=title_id,
            current_ts=current_ts,
            confident_title=confident_title,
            search_title=search_title,
        )
        if not payloads:
            return soft_no_scene_joke(), TokenUsage.empty(), False

        result = self._completion.complete(
            model=tier_decision.model,
            messages=messages,
            tools=None,
            temperature=min(0.85, max(0.55, self._settings.llm_temperature + 0.3)),
            max_tokens=self._answer_max_tokens(),
        )
        text, used = self._finalize_answer(
            raw=result.content,
            question=question,
            tool_payloads=payloads,
            used_context=used_context,
            joke_mode=True,
            force_lookup_if_empty=False,
        )
        return text, result.usage, used

    def _run_joke_loop_stream(
        self,
        *,
        question: str,
        title_id: str,
        current_ts: float,
        tier_decision: ModelTierDecision,
        confident_title: str | None = None,
        search_title: str | None = None,
    ) -> Iterator[AskStreamEvent]:
        yield AskStreamEvent(type="status", message="Checking the scene…")
        yield AskStreamEvent(type="tool_start", tool="scene_lookup", message="Checking the scene…")
        messages, payloads, used_context = self._joke_grounded_messages(
            question=question,
            title_id=title_id,
            current_ts=current_ts,
            confident_title=confident_title,
            search_title=search_title,
        )
        yield AskStreamEvent(type="tool_end", tool="scene_lookup", message="scene_lookup done")

        if not payloads:
            joke = soft_no_scene_joke()
            yield AskStreamEvent(type="status", message="Answering…")
            for piece in _chunk_text_for_stream(joke):
                yield AskStreamEvent(type="token", text=piece)
            return joke, TokenUsage.empty(), False

        yield AskStreamEvent(type="status", message="Answering…")
        parts: list[str] = []
        usage = TokenUsage.empty()
        for delta in self._completion.complete_stream(
            model=tier_decision.model,
            messages=messages,
            temperature=min(0.85, max(0.55, self._settings.llm_temperature + 0.3)),
            max_tokens=self._answer_max_tokens(),
        ):
            parts.append(delta)
        # complete_stream may not report usage — empty is fine for joke path
        answer_text, used = self._finalize_answer(
            raw="".join(parts),
            question=question,
            tool_payloads=payloads,
            used_context=used_context,
            joke_mode=True,
            force_lookup_if_empty=False,
        )
        for piece in _chunk_text_for_stream(answer_text):
            yield AskStreamEvent(type="token", text=piece)
        return answer_text, usage, used

    def _joke_grounded_messages(
        self,
        *,
        question: str,
        title_id: str,
        current_ts: float,
        confident_title: str | None,
        search_title: str | None,
    ) -> tuple[list[dict[str, Any]], list[Any], bool]:
        query = joke_scene_query(question)
        with observe_tool_call("scene_lookup"):
            hits = self._scene_lookup.lookup(
                title_id=title_id,
                query_text=query,
                current_ts=current_ts,
            )
        payload = [hit.to_tool_dict() for hit in hits]
        messages = self._initial_messages(
            question, confident_title, search_title, joke_mode=True
        )
        if not payload:
            return messages, [], False

        tool_call_id = "joke_scene_lookup"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "scene_lookup",
                            "arguments": json.dumps({"query_text": query}),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": "scene_lookup",
                "content": json.dumps(payload),
            }
        )
        return messages, [payload], True

    def _run_tool_loop_stream(
        self,
        *,
        question: str,
        title_id: str,
        current_ts: float,
        user_id: str,
        tier_decision,
        confident_title: str | None = None,
        search_title: str | None = None,
    ) -> Iterator[AskStreamEvent]:
        """Yield stream events; return value is (text, usage, used_context)."""
        messages = self._initial_messages(question, confident_title, search_title)
        usage = TokenUsage.empty()
        used_context = False
        tools = self._available_tools()
        tool_payloads: list[Any] = []

        for _ in range(_MAX_TOOL_ROUNDS):
            result = self._completion.complete(
                model=tier_decision.model,
                messages=messages,
                tools=tools,
                temperature=self._settings.llm_temperature,
                max_tokens=self._tool_max_tokens(),
            )
            usage = usage.merge(result.usage)

            if not result.tool_calls:
                answer_text, used_context, mm_usage = self._compose_final_answer(
                    raw=result.content,
                    question=question,
                    tool_payloads=tool_payloads,
                    used_context=used_context,
                    title_id=title_id,
                    current_ts=current_ts,
                    model=tier_decision.model,
                )
                if mm_usage is not None:
                    usage = usage.merge(mm_usage)
                yield AskStreamEvent(type="status", message="Answering…")
                for piece in _chunk_text_for_stream(answer_text):
                    yield AskStreamEvent(type="token", text=piece)
                return answer_text, usage, used_context

            messages.append(assistant_tool_calls_message(result))
            for tool_call in result.tool_calls:
                status = _TOOL_STATUS.get(tool_call.name, f"Running {tool_call.name}…")
                yield AskStreamEvent(type="status", message=status)
                yield AskStreamEvent(type="tool_start", tool=tool_call.name, message=status)

                payload, hit_context = self._dispatch_tool(
                    tool_call,
                    question=question,
                    title_id=title_id,
                    current_ts=current_ts,
                    user_id=user_id,
                    search_title=search_title,
                )
                if payload is None:
                    logger.warning("Ignoring unsupported tool call: %s", tool_call.name)
                    yield AskStreamEvent(
                        type="tool_end",
                        tool=tool_call.name,
                        message=f"{tool_call.name} skipped",
                    )
                    continue

                used_context = used_context or hit_context
                tool_payloads.append(payload)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": json.dumps(payload),
                    }
                )
                yield AskStreamEvent(
                    type="tool_end",
                    tool=tool_call.name,
                    message=f"{tool_call.name} done",
                )

            yield AskStreamEvent(type="status", message="Composing answer…")

        yield AskStreamEvent(type="status", message="Answering…")
        parts: list[str] = []
        for delta in self._completion.complete_stream(
            model=tier_decision.model,
            messages=messages,
            temperature=self._settings.llm_temperature,
            max_tokens=self._answer_max_tokens(),
        ):
            parts.append(delta)
        answer_text, used_context, mm_usage = self._compose_final_answer(
            raw="".join(parts),
            question=question,
            tool_payloads=tool_payloads,
            used_context=used_context,
            title_id=title_id,
            current_ts=current_ts,
            model=tier_decision.model,
        )
        if mm_usage is not None:
            usage = usage.merge(mm_usage)
        for piece in _chunk_text_for_stream(answer_text):
            yield AskStreamEvent(type="token", text=piece)
        return answer_text, usage, used_context

    def _run_tool_loop(
        self,
        *,
        question: str,
        title_id: str,
        current_ts: float,
        user_id: str,
        tier_decision,
        confident_title: str | None = None,
        search_title: str | None = None,
    ) -> tuple[str, TokenUsage | None, bool]:
        messages = self._initial_messages(question, confident_title, search_title)
        usage = TokenUsage.empty()
        used_context = False
        tools = self._available_tools()
        tool_payloads: list[Any] = []

        for _ in range(_MAX_TOOL_ROUNDS):
            result = self._completion.complete(
                model=tier_decision.model,
                messages=messages,
                tools=tools,
                temperature=self._settings.llm_temperature,
                max_tokens=self._tool_max_tokens(),
            )
            usage = usage.merge(result.usage)

            if not result.tool_calls:
                text, used_context, mm_usage = self._compose_final_answer(
                    raw=result.content,
                    question=question,
                    tool_payloads=tool_payloads,
                    used_context=used_context,
                    title_id=title_id,
                    current_ts=current_ts,
                    model=tier_decision.model,
                )
                if mm_usage is not None:
                    usage = usage.merge(mm_usage)
                return text, usage, used_context

            messages.append(assistant_tool_calls_message(result))
            for tool_call in result.tool_calls:
                payload, hit_context = self._dispatch_tool(
                    tool_call,
                    question=question,
                    title_id=title_id,
                    current_ts=current_ts,
                    user_id=user_id,
                    search_title=search_title,
                )
                if payload is None:
                    logger.warning("Ignoring unsupported tool call: %s", tool_call.name)
                    continue
                used_context = used_context or hit_context
                tool_payloads.append(payload)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": json.dumps(payload),
                    }
                )

        # Exhausted tool rounds — try multimodal with whatever tools returned.
        mm = self._try_multimodal_answer(
            question=question,
            tool_payloads=tool_payloads,
            model=tier_decision.model,
        )
        if mm is not None:
            text, mm_usage = mm
            if mm_usage is not None:
                usage = usage.merge(mm_usage)
            final, used = self._finalize_answer(
                raw=text,
                question=question,
                tool_payloads=tool_payloads,
                used_context=True,
                force_lookup_if_empty=False,
            )
            return final, usage, used

        fallback = grounded_fallback_answer(question, tool_payloads)
        return (fallback or _UNKNOWN_ANSWER), usage, bool(tool_payloads) or used_context

    def _initial_messages(
        self,
        question: str,
        confident_title: str | None,
        search_title: str | None,
        *,
        joke_mode: bool = False,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": CONVERSATION_SYSTEM_PROMPT}
        ]
        if joke_mode:
            messages.append({"role": "system", "content": JOKE_MODE_SYSTEM_PROMPT})
        if self._cast_lookup is not None and not joke_mode:
            hint = self._cast_title_hint(confident_title, search_title)
            if hint:
                messages.append({"role": "system", "content": hint})
        messages.append({"role": "user", "content": question})
        return messages

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
        user_id: str,
        search_title: str | None = None,
    ) -> tuple[Any, bool]:
        """Return (payload, used_context). payload is None for unsupported tools."""
        if tool_call.name == "user_memory" and self._user_memory is not None:
            with observe_tool_call("user_memory"):
                mode = str(tool_call.arguments.get("mode", "summary"))
                max_turns = tool_call.arguments.get("max_turns")
                result = self._user_memory.lookup(
                    user_id=user_id,
                    title_id=title_id,
                    mode=mode,
                    max_turns=int(max_turns) if isinstance(max_turns, (int, str)) and str(max_turns).isdigit() else None,
                )
            return result, bool(result.get("found"))

        if tool_call.name == "scene_lookup":
            with observe_tool_call("scene_lookup"):
                query_text = str(tool_call.arguments.get("query_text", question))
                hits = self._scene_lookup.lookup(
                    title_id=title_id,
                    query_text=query_text,
                    current_ts=current_ts,
                )
            return [hit.to_tool_dict() for hit in hits], bool(hits)

        if tool_call.name == "character_lookup" and self._character_lookup is not None:
            with observe_tool_call("character_lookup"):
                character = tool_call.arguments.get("character")
                result = self._character_lookup.lookup(
                    title_id=title_id,
                    character=str(character) if character else None,
                    current_ts=current_ts,
                )
            used = bool(result.get("found") and result.get("appearances"))
            return result, used

        if tool_call.name == "cast_lookup" and self._cast_lookup is not None:
            with observe_tool_call("cast_lookup"):
                title_name = (
                    str(tool_call.arguments.get("title_name", ""))
                    or search_title
                    or self._settings.effective_search_title(title_id)
                    or ""
                )
                year = tool_call.arguments.get("year")
                result = self._cast_lookup.lookup(
                    title_id=title_id,
                    title_name=title_name,
                    year=int(year) if isinstance(year, (int, str)) and str(year).isdigit() else None,
                )
            return result, bool(result.get("cast"))

        if tool_call.name == "knowledge_search" and self._knowledge_search is not None:
            with observe_tool_call("knowledge_search"):
                query_text = str(tool_call.arguments.get("query_text", question))
                category = tool_call.arguments.get("category")
                hits = self._knowledge_search.search(
                    title_id=title_id,
                    query_text=query_text,
                    category=str(category) if category else None,
                )
            return [hit.to_tool_dict() for hit in hits], bool(hits)

        return None, False


def _intent_from_gate_reason(reason: str) -> str | None:
    if "ignore" in reason:
        return "FILLER"
    if "social" in reason:
        return "SOCIAL"
    if "off_topic" in reason:
        return "SOCIAL"
    if "navigate" in reason:
        return "NAVIGATE"
    if "joke" in reason:
        return "JOKE"
    return None


def build_conversation_agent(
    settings: Settings,
    scene_lookup: SceneLookupTool,
    completion_client: CompletionClient | None = None,
    tier_router: TierRouter | None = None,
    cast_lookup: CastLookupTool | None = None,
    character_lookup: CharacterLookupTool | None = None,
    knowledge_search: KnowledgeSearchTool | None = None,
    user_memory: UserMemoryTool | None = None,
    object_store: ObjectStore | None = None,
) -> ConversationAgent:
    completion = completion_client or build_completion_client(settings)
    return ConversationAgent(
        completion_client=completion,
        scene_lookup=scene_lookup,
        settings=settings,
        tier_router=tier_router or build_tier_router(settings, completion),
        cast_lookup=cast_lookup,
        character_lookup=character_lookup,
        knowledge_search=knowledge_search,
        user_memory=user_memory,
        object_store=object_store,
    )
