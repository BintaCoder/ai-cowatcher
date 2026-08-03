"""LiteLLM completion clients — real router and deterministic mock for tests."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import litellm

from ai_cowatcher.agent.token_usage import TokenUsage, usage_from_litellm_response
from ai_cowatcher.config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class CompletionResult:
    content: str | None
    tool_calls: list[ToolCall]
    usage: TokenUsage | None = None


class CompletionClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        ...

    def complete_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Yield text deltas for the final answer (no tool calls)."""
        ...


def _parse_tool_calls(message: Any) -> list[ToolCall]:
    raw_calls = getattr(message, "tool_calls", None) or []
    parsed: list[ToolCall] = []
    for index, call in enumerate(raw_calls):
        function = call.function
        arguments_raw = function.arguments or "{}"
        try:
            arguments = json.loads(arguments_raw)
        except json.JSONDecodeError:
            arguments = {}
        parsed.append(
            ToolCall(
                id=call.id or f"call_{index}",
                name=function.name,
                arguments=arguments,
            )
        )
    return parsed


class LiteLLMCompletionClient:
    """Routes conversation completions through LiteLLM."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = litellm.completion(**kwargs)
        message = response.choices[0].message
        return CompletionResult(
            content=message.content,
            tool_calls=_parse_tool_calls(message),
            usage=usage_from_litellm_response(response),
        )

    def complete_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            text = getattr(delta, "content", None)
            if text:
                yield text


_UNKNOWN_PHRASE = "Not sure yet — nothing's made that clear so far."


def _chunk_text_for_stream(text: str) -> Iterator[str]:
    """Yield small pieces so the UI can paint progressively in mock/tests."""
    if not text:
        return
    # Prefer word-ish chunks so sentence TTS can kick in early.
    parts = re.findall(r"\S+\s*", text)
    if not parts:
        yield text
        return
    for part in parts:
        yield part


class MockCompletionClient:
    """Deterministic mock LiteLLM router that exercises the tool-calling loop."""

    def __init__(self) -> None:
        self.models_used: list[str] = []

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        self.models_used.append(model)
        del temperature, max_tokens, tools
        return self._decide(messages)

    def complete_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        self.models_used.append(model)
        del temperature, max_tokens
        result = self._decide(messages)
        content = (result.content or _UNKNOWN_PHRASE).strip()
        yield from _chunk_text_for_stream(content)

    def _decide(self, messages: list[dict[str, Any]]) -> CompletionResult:
        if messages and messages[-1].get("role") == "tool":
            question = _latest_user_message(messages)
            tool_content = messages[-1]["content"]
            if _is_knowledge_tool_result(tool_content):
                return CompletionResult(
                    content=self._answer_from_knowledge(tool_content),
                    tool_calls=[],
                    usage=_mock_usage(messages),
                )
            if _is_user_memory_tool_result(tool_content):
                return CompletionResult(
                    content=self._answer_from_user_memory(tool_content),
                    tool_calls=[],
                    usage=_mock_usage(messages),
                )
            if _is_character_tool_result(tool_content):
                return CompletionResult(
                    content=self._answer_from_character(tool_content, question),
                    tool_calls=[],
                    usage=_mock_usage(messages),
                )
            return CompletionResult(
                content=self._answer_from_tool_result(tool_content, question),
                tool_calls=[],
                usage=_mock_usage(messages),
            )

        if _is_utterance_gate_request(messages):
            question = _latest_user_message(messages)
            meaningful = _mock_utterance_gate_yes(question)
            return CompletionResult(
                content="YES" if meaningful else "NO",
                tool_calls=[],
                usage=TokenUsage(prompt_tokens=20, completion_tokens=1, total_tokens=21),
            )

        if _is_classifier_request(messages):
            question = _latest_user_message(messages)
            escalate = _mock_prompt_classifier_escalates(question)
            return CompletionResult(
                content="YES" if escalate else "NO",
                tool_calls=[],
                usage=TokenUsage(prompt_tokens=24, completion_tokens=1, total_tokens=25),
            )

        question = _latest_user_message(messages)
        if _is_joke_question(question):
            return CompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="mock_call_scene_lookup",
                        name="scene_lookup",
                        arguments={
                            "query_text": "funny dialogue conversation banter what just happened"
                        },
                    )
                ],
                usage=TokenUsage(prompt_tokens=48, completion_tokens=12, total_tokens=60),
            )
        if _is_continuity_question(question):
            return CompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="mock_call_user_memory",
                        name="user_memory",
                        arguments={"mode": "summary"},
                    )
                ],
                usage=TokenUsage(prompt_tokens=48, completion_tokens=12, total_tokens=60),
            )
        if _is_knowledge_question(question):
            return CompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="mock_call_knowledge_search",
                        name="knowledge_search",
                        arguments={"query_text": question},
                    )
                ],
                usage=TokenUsage(prompt_tokens=48, completion_tokens=12, total_tokens=60),
            )
        if _is_character_question(question):
            return CompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="mock_call_character_lookup",
                        name="character_lookup",
                        arguments={"character": _character_name_hint(question)},
                    )
                ],
                usage=TokenUsage(prompt_tokens=48, completion_tokens=12, total_tokens=60),
            )
        return CompletionResult(
            content=None,
            tool_calls=[
                ToolCall(
                    id="mock_call_scene_lookup",
                    name="scene_lookup",
                    arguments={"query_text": question},
                )
            ],
            usage=TokenUsage(prompt_tokens=48, completion_tokens=12, total_tokens=60),
        )

    def _answer_from_character(self, tool_content: str, question: str) -> str:
        try:
            result = json.loads(tool_content)
        except json.JSONDecodeError:
            return _UNKNOWN_PHRASE
        if not result.get("found") or not result.get("appearances"):
            return "No, this looks like the first time you're seeing them."
        count = result.get("appearance_count") or len(result.get("appearances", []))
        parts = [f"Yes, you've seen them in {count} earlier scene(s)."]
        for rel in result.get("relationships", []):
            summary = rel.get("summary")
            if summary:
                parts.append(str(summary))
        return " ".join(parts)

    def _answer_from_knowledge(self, tool_content: str) -> str:
        try:
            chunks = json.loads(tool_content)
        except json.JSONDecodeError:
            return _UNKNOWN_PHRASE
        if not isinstance(chunks, list) or not chunks:
            return "I don't have that in our production notes."
        text = str(chunks[0].get("text", "")).strip()
        return text or "I don't have that in our production notes."

    def _answer_from_user_memory(self, tool_content: str) -> str:
        try:
            result = json.loads(tool_content)
        except json.JSONDecodeError:
            return _UNKNOWN_PHRASE
        if not result.get("found"):
            return "We haven't chatted about this title yet."
        summary = str(result.get("summary", "")).strip()
        if summary:
            return summary
        return "I don't have earlier messages to refer to."

    def _answer_from_tool_result(self, tool_content: str, question: str) -> str:
        try:
            scenes = json.loads(tool_content)
        except json.JSONDecodeError:
            return _UNKNOWN_PHRASE

        if not scenes:
            return _UNKNOWN_PHRASE

        if _is_joke_question(question):
            return self._joke_from_scenes(scenes)

        combined = " ".join(
            f"{scene.get('transcript', '')} {scene.get('caption', '')}" for scene in scenes
        )
        killer_match = re.search(r"killer is ([A-Za-z]+)", combined, re.IGNORECASE)
        if killer_match:
            killer = killer_match.group(1)
            return f"Looks like the killer is {killer}."

        if re.search(r"\bkiller\b", question, re.IGNORECASE):
            return _UNKNOWN_PHRASE

        snippet = scenes[0].get("transcript") or scenes[0].get("caption") or ""
        if snippet:
            # One short friend-line, not a recap dump.
            short = " ".join(str(snippet).split())
            words = short.split()
            if len(words) > 18:
                short = " ".join(words[:18]).rstrip(",;:") + "…"
            return short
        return _UNKNOWN_PHRASE

    def _joke_from_scenes(self, scenes: list[Any]) -> str:
        snippet = scenes[0].get("transcript") or scenes[0].get("caption") or ""
        if not snippet:
            return "Nothing to riff on yet — hit me again in a sec."
        short = " ".join(str(snippet).split())
        words = short.split()
        if len(words) > 10:
            short = " ".join(words[:10]).rstrip(",;:")
        return f'Couch take: "{short}" — peak drama.'


def _is_joke_question(question: str) -> bool:
    from ai_cowatcher.agent.joke_intent import is_joke_request

    return is_joke_request(question)


def _latest_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


_CHARACTER_INTENT = re.compile(
    r"seen\s+(?:him|her|them)\s+before"
    r"|who(?:'s| is| are)\s+(?:he|she|they|this)"
    r"|how\s+do\s+(?:they|these two)\s+know"
    r"|know\s+each\s+other"
    r"|their\s+relationship"
    r"|(?:have|has)\s+(?:they|we)\s+met"
    r"|met\s+before"
    r"|related\s+to\s+each\s+other",
    re.IGNORECASE,
)


def _is_character_question(question: str) -> bool:
    return bool(_CHARACTER_INTENT.search(question))


def _character_name_hint(question: str) -> str:
    """Extract a capitalized name if the viewer named someone, else empty.

    Empty means "the person currently on screen" (e.g. 'have I seen him before?').
    """
    match = re.search(r"\b(?:is|are|does|did)\s+([A-Z][a-z]+)\b", question)
    if match:
        return match.group(1)
    return ""


def _is_character_tool_result(tool_content: str) -> bool:
    try:
        parsed = json.loads(tool_content)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and ("found" in parsed or "appearances" in parsed)


def _is_knowledge_tool_result(tool_content: str) -> bool:
    try:
        parsed = json.loads(tool_content)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(parsed, list)
        and bool(parsed)
        and isinstance(parsed[0], dict)
        and "chunk_id" in parsed[0]
    )


def _is_user_memory_tool_result(tool_content: str) -> bool:
    try:
        parsed = json.loads(tool_content)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and "turns" in parsed and "summary" in parsed


_CONTINUITY_INTENT = re.compile(
    r"\b(as i (?:said|mentioned)|what did i ask|earlier i asked|before i asked|"
    r"what was my (?:last )?question|i mentioned earlier)\b",
    re.IGNORECASE,
)


def _is_continuity_question(question: str) -> bool:
    return bool(_CONTINUITY_INTENT.search(question))


_KNOWLEDGE_INTENT = re.compile(
    r"\b(director|creator|created by|created|biograph|sports stat|production|"
    r"who made|who directed|who created|crew|showrunner)\b",
    re.IGNORECASE,
)


def _is_knowledge_question(question: str) -> bool:
    return bool(_KNOWLEDGE_INTENT.search(question))


def _is_utterance_gate_request(messages: list[dict[str, Any]]) -> bool:
    if not messages:
        return False
    content = str(messages[0].get("content", ""))
    return messages[0].get("role") == "system" and "meaningful co-watch request" in content


def _mock_utterance_gate_yes(question: str) -> bool:
    lower = question.lower()
    if any(
        token in lower
        for token in (
            "what",
            "who",
            "why",
            "how",
            "happen",
            "joke",
            "going on",
            "talking",
            "scene",
            "character",
            "killer",
            "said",
        )
    ):
        return True
    if len(lower.strip()) >= 12 and "?" in question:
        return True
    return False


def _is_classifier_request(messages: list[dict[str, Any]]) -> bool:
    return bool(messages) and messages[0].get("role") == "system" and "Reply with only YES or NO" in str(
        messages[0].get("content", "")
    ) and "meaningful co-watch" not in str(messages[0].get("content", ""))


def _mock_prompt_classifier_escalates(question: str) -> bool:
    lower = question.lower()
    return any(token in lower for token in ("why", "explain", "compare", "motivation", "theme"))


def _mock_usage(messages: list[dict[str, Any]]) -> TokenUsage:
    question = _latest_user_message(messages)
    prompt_tokens = max(32, len(question) // 2)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=18,
        total_tokens=prompt_tokens + 18,
    )


def build_completion_client(settings: Settings) -> CompletionClient:
    if settings.mock_mode:
        return MockCompletionClient()
    return LiteLLMCompletionClient(settings)
