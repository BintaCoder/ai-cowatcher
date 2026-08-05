"""LiteLLM completion clients — real router and deterministic mock for tests."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import litellm

from ai_cowatcher.agent.token_usage import TokenUsage, usage_from_litellm_response
from ai_cowatcher.config import Settings
from ai_cowatcher.observability.ask_latency import AskLatencyTracker

logger = logging.getLogger(__name__)

# Persistent HTTP client so Gemini calls reuse TCP/TLS (not a new handshake each ask).
_LITELLM_HTTP_CLIENT: httpx.Client | None = None


def ensure_litellm_http_pool() -> httpx.Client:
    """Install a process-wide httpx client for LiteLLM connection reuse."""
    global _LITELLM_HTTP_CLIENT
    if _LITELLM_HTTP_CLIENT is None:
        _LITELLM_HTTP_CLIENT = httpx.Client(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            http2=False,
        )
    # LiteLLM reads client_session when present for OpenAI-compatible HTTP paths.
    litellm.client_session = _LITELLM_HTTP_CLIENT  # type: ignore[attr-defined]
    return _LITELLM_HTTP_CLIENT


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Gemini 3+ requires thought_signature on functionCall parts in follow-up turns.
    thought_signature: str | None = None


@dataclass
class CompletionResult:
    content: str | None
    tool_calls: list[ToolCall]
    usage: TokenUsage | None = None


# Google-recommended dummy so multi-turn tools work when a signature was stripped/missing
# https://ai.google.dev/gemini-api/docs/thought-signatures
_GEMINI_SKIP_THOUGHT_SIGNATURE = "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I="


def _extract_thought_signature(call: Any) -> str | None:
    """Pull thought_signature from LiteLLM / OpenAI-shaped tool_call objects."""
    psf = getattr(call, "provider_specific_fields", None)
    if isinstance(psf, dict):
        sig = psf.get("thought_signature") or psf.get("thoughtSignature")
        if sig:
            return str(sig)
    # dict-shaped (already normalized)
    if isinstance(call, dict):
        psf = call.get("provider_specific_fields") or {}
        if isinstance(psf, dict):
            sig = psf.get("thought_signature") or psf.get("thoughtSignature")
            if sig:
                return str(sig)
        extra = call.get("extra_content") or call.get("extra_body")
        if isinstance(extra, dict):
            google = extra.get("google") or extra.get("vertex") or {}
            if isinstance(google, dict) and google.get("thought_signature"):
                return str(google["thought_signature"])
    # nested model_extra
    model_extra = getattr(call, "model_extra", None) or {}
    if isinstance(model_extra, dict):
        psf = model_extra.get("provider_specific_fields")
        if isinstance(psf, dict) and psf.get("thought_signature"):
            return str(psf["thought_signature"])
    return None


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
                thought_signature=_extract_thought_signature(call),
            )
        )
    return parsed


def assistant_tool_calls_message(result: CompletionResult) -> dict[str, Any]:
    """Build the assistant message for multi-turn tool loops (Gemini-safe)."""
    tool_calls: list[dict[str, Any]] = []
    for index, tool_call in enumerate(result.tool_calls):
        # First tool call in a parallel set must carry a signature for Gemini 3.
        sig = tool_call.thought_signature
        if index == 0 and not sig:
            sig = _GEMINI_SKIP_THOUGHT_SIGNATURE
        entry: dict[str, Any] = {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": json.dumps(tool_call.arguments),
            },
        }
        if sig:
            entry["provider_specific_fields"] = {"thought_signature": sig}
        tool_calls.append(entry)
    return {
        "role": "assistant",
        "content": result.content,
        "tool_calls": tool_calls,
    }


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
        latency: AskLatencyTracker | None = None,
    ) -> Iterator[str]:
        """Yield text deltas for the final answer (no tool calls)."""
        ...


class LiteLLMCompletionClient:
    """Routes conversation completions through LiteLLM."""

    def __init__(self, settings: Settings):
        self._settings = settings
        ensure_litellm_http_pool()

    def _common_kwargs(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Gemini 3 burns max_tokens on reasoning first; low effort + higher budget
        # leaves room for the spoken answer (short co-watch replies).
        effort = (getattr(self._settings, "llm_reasoning_effort", None) or "").strip()
        if effort and effort.lower() not in ("", "off", "default"):
            kwargs["reasoning_effort"] = effort
        return kwargs

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        kwargs = self._common_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
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
        latency: AskLatencyTracker | None = None,
    ) -> Iterator[str]:
        """Yield provider deltas immediately (no full-response buffer)."""
        kwargs = self._common_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        kwargs["stream"] = True
        if latency is not None:
            latency.mark_llm_request_sent()
        t_sent = time.perf_counter()
        logger.info(
            json.dumps(
                {
                    "event": "llm_stream_request",
                    "model": model,
                    "max_tokens": max_tokens,
                },
                separators=(",", ":"),
            )
        )
        response = litellm.completion(**kwargs)
        first_token = True
        for chunk in response:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            text = getattr(delta, "content", None)
            if text:
                if first_token:
                    first_token = False
                    ttft_ms = (time.perf_counter() - t_sent) * 1000.0
                    if latency is not None:
                        latency.mark_llm_first_token()
                    logger.info(
                        json.dumps(
                            {
                                "event": "llm_stream_first_token",
                                "model": model,
                                "ttft_ms": round(ttft_ms, 2),
                            },
                            separators=(",", ":"),
                        )
                    )
                # Forward as received — never accumulate the full completion first.
                yield text
        total_ms = (time.perf_counter() - t_sent) * 1000.0
        if latency is not None:
            latency.mark_llm_stream_complete()
        logger.info(
            json.dumps(
                {
                    "event": "llm_stream_complete",
                    "model": model,
                    "total_ms": round(total_ms, 2),
                    "had_token": not first_token,
                },
                separators=(",", ":"),
            )
        )


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
        latency: AskLatencyTracker | None = None,
    ) -> Iterator[str]:
        self.models_used.append(model)
        del temperature, max_tokens
        if latency is not None:
            latency.mark_llm_request_sent()
        result = self._decide(messages)
        content = (result.content or _UNKNOWN_PHRASE).strip()
        first = True
        for piece in _chunk_text_for_stream(content):
            if first and latency is not None:
                latency.mark_llm_first_token()
                first = False
            yield piece
        if latency is not None:
            latency.mark_llm_stream_complete()

    def _decide(self, messages: list[dict[str, Any]]) -> CompletionResult:
        if _is_merged_intent_request(messages):
            return CompletionResult(
                content=_mock_merged_tagged_reply(messages),
                tool_calls=[],
                usage=_mock_usage(messages),
            )

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
        # Keep one brief sentence so brevity isn't forced to drop the relationship.
        rel_bits: list[str] = []
        for rel in result.get("relationships", []):
            summary = rel.get("summary")
            if summary:
                rel_bits.append(str(summary).rstrip("."))
            elif rel.get("rel_type"):
                rel_bits.append(f"their {rel['rel_type']} link is already on the table")
        if rel_bits:
            return f"Yes — you've seen them earlier, and {rel_bits[0].lower()}."
        return f"Yes, you've seen them in {count} earlier scene(s)."

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


def _is_merged_intent_request(messages: list[dict[str, Any]]) -> bool:
    if not messages:
        return False
    content = str(messages[0].get("content", ""))
    return (
        messages[0].get("role") == "system"
        and "intent router and answer generator" in content
    )


def _mock_merged_viewer_question(messages: list[dict[str, Any]]) -> str:
    user = _latest_user_message(messages)
    match = re.search(r'viewer_question:\s*"([^"]*)"', user, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return user.strip()


def _mock_merged_scene_json(messages: list[dict[str, Any]]) -> str | None:
    user = _latest_user_message(messages)
    # First JSON array after Tool evidence
    match = re.search(r"Tool evidence.*?\n(\[.*)", user, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    blob = match.group(1).strip()
    if blob.startswith("(no"):
        return None
    # take first line that looks like JSON or whole array
    try:
        # find matching array end roughly
        json.loads(blob)
        return blob
    except json.JSONDecodeError:
        # truncated dumps may still start with [{
        start = blob.find("[")
        end = blob.rfind("]")
        if start >= 0 and end > start:
            try:
                json.loads(blob[start : end + 1])
                return blob[start : end + 1]
            except json.JSONDecodeError:
                return None
    return None


def _mock_merged_tagged_reply(messages: list[dict[str, Any]]) -> str:
    question = _mock_merged_viewer_question(messages)
    lower = question.lower().strip()
    persona_line = _mock_persona_social_from_messages(messages)

    if not lower or lower in ("um", "uh", "hmm", "ah", "oh"):
        return "[FILLER]"
    if re.fullmatch(r"(hi|hello|hey|thanks|thank you|bye|cool|lol)\.?!?", lower):
        social = persona_line or "Right here with you — ask about the show anytime."
        return f"[SOCIAL]\n\n{social}"
    from ai_cowatcher.predictions import looks_like_prediction, persona_prediction_ack

    if looks_like_prediction(question):
        ack = persona_prediction_ack(_mock_persona_id_from_messages(messages), question)
        return f"[PREDICTION]\n\n{ack}"
    if any(
        token in lower
        for token in ("go to", "jump to", "skip to", "take me to", "credits", "rewind")
    ) or re.search(r"\d{1,2}:\d{2}", lower):
        return "[NAVIGATE]"
    if _is_joke_question(question):
        tool_json = _mock_merged_scene_json(messages)
        if tool_json:
            try:
                scenes = json.loads(tool_json)
                mock = MockCompletionClient()
                line = mock._joke_from_scenes(scenes) if scenes else "Waiting on a beat to riff on."
            except json.JSONDecodeError:
                line = "Waiting on a beat to riff on."
        else:
            line = "Waiting on a beat to riff on."
        # Light persona flavor without breaking fact grounding.
        if "witty" in _mock_persona_id_from_messages(messages):
            line = f"Okay but — {line}"
        elif "calm" in _mock_persona_id_from_messages(messages):
            line = line.rstrip(".!") + "."
        return f"[JOKE]\n\n{line}"

    tool_json = _mock_merged_scene_json(messages)
    if tool_json:
        try:
            scenes = json.loads(tool_json)
            mock = MockCompletionClient()
            body = mock._answer_from_tool_result(tool_json, question)
        except json.JSONDecodeError:
            body = "Something's happening on screen — details still fuzzy."
    else:
        body = "Not sure yet — nothing's made that clear so far."
    pid = _mock_persona_id_from_messages(messages)
    if "witty" in pid and body and not body.lower().startswith("not sure"):
        body = body.rstrip(".!") + " — wild beat."
    elif "calm" in pid and body:
        body = body  # neutral delivery
    return f"[CONTENT]\n\n{body}"


def _mock_persona_social_from_messages(messages: list[dict[str, Any]]) -> str | None:
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = str(msg.get("content") or "")
        match = re.search(
            r'SOCIAL canned reply when tag is SOCIAL[^:]*:\s*"([^"]+)"',
            content,
        )
        if match:
            return match.group(1).strip()
    return None


def _mock_persona_id_from_messages(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = str(msg.get("content") or "")
        match = re.search(r"Companion persona:.*?\(([^)]+)\)", content)
        if match:
            return match.group(1).strip().lower()
    return ""


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
    ensure_litellm_http_pool()
    return LiteLLMCompletionClient(settings)
