"""Utterance gate: ignore noise, route intents, ambiguous-only mini classifier."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from ai_cowatcher.agent.completion import CompletionClient
from ai_cowatcher.agent.joke_intent import is_joke_request
from ai_cowatcher.agent.token_usage import TokenUsage
from ai_cowatcher.config import Settings

logger = logging.getLogger("ai_cowatcher.gate")

GateAction = Literal["ignore", "social", "joke", "navigate", "content", "off_topic"]

_IGNORE_REPLY = ""  # silence — UI shows status only
_SOCIAL_REPLY = "Right here — ask about what's on screen if you want."
_OFF_TOPIC_REPLY = "I only help with this show for now — try “who’s that?” or “what just happened?”"
_AMBIGUOUS_IGNORE_REPLY = ""

_FILLER_ONLY = re.compile(
    r"^(?:"
    r"um+|uh+|uhm+|erm+|hmm+|mm+|mhm+|ah+|oh+|huh+|ha+|eh+|ugh+|"
    r"like|well|so|yeah|yep|yup|ya|nah|nope|"
    r"okay|ok|right|sure|alright|whatever|"
    r"you know|i mean|kind of|kinda|sort of|"
    r"wait(?:\s+what)?|"
    r"(?:uh|um|hmm|ah)(?:\s+(?:uh|um|hmm|ah|like|so|well|okay|ok))*"
    r")\.?\!?$",
    re.IGNORECASE,
)

# Multi-token fillers (STT fragments) that fullmatch list never covers alone.
_FILLER_PHRASE = re.compile(
    r"^(?:"
    r"hmm+\s+okay|hmm+\s+ok|ok(?:ay)?\s+yeah|"
    r"uh\s+huh|mm+\s+hmm|"
    r"wait[\s.…-]*"
    r")$",
    re.IGNORECASE,
)

_SOCIAL = re.compile(
    r"^(?:"
    r"hi|hello|hey|hey there|hiya|howdy|"
    r"thanks|thank you|thx|ty|"
    r"bye|goodbye|see you|later|"
    r"good morning|good night|good evening|"
    r"how are you|what's up|whats up|"
    r"cool|nice|awesome|great|lol|lmao"
    r")\.?\!?$",
    re.IGNORECASE,
)

# Longer social / presence checks (bench q11–q13 style) — must short-circuit before LLM.
_SOCIAL_PHRASE = re.compile(
    r"^(?:"
    r"you\s+there|still\s+there|anyone\s+there|"
    r"hey(?:\s+there)?,?\s+how(?:'s|s|\s+is|\s+are)\s+(?:it|you)(?:\s+going|\s+doing)?|"
    r"how(?:'s|s|\s+is)\s+it\s+going|"
    r"how\s+are\s+you(?:\s+doing)?|"
    r"thanks?,?\s+that\s+helped|"
    r"thank\s+you,?\s+that\s+helped|"
    r"thanks?\s+a\s+lot|thanks?\s+so\s+much|"
    r"you\s+good|all\s+good|"
    r"just\s+checking\s+in"
    r")\.?\!?$",
    re.IGNORECASE,
)

# Clear non-co-watch chit-chat / world knowledge.
_OFF_TOPIC = re.compile(
    r"(?ix)\b("
    r"weather|temperature|forecast|"
    r"stock(?:s)?|crypto|bitcoin|"
    r"news today|who won the (?:game|match|election)|"
    r"order (?:pizza|food)|set (?:a )?timer|"
    r"remind me|call (?:mom|my)|"
    r"python code|write (?:code|a script)|"
    r"chat ?gpt|openai"
    r")\b"
)

_NAVIGATE = re.compile(
    r"(?ix)\b("
    r"go\s+to|jump\s+to|skip\s+to|take\s+me\s+to|"
    r"rewind|fast[\s-]?forward|"
    r"where\s+does|where\s+is|when\s+does|when\s+is|"
    r"show\s+me\s+(?:the|that|when)|"
    r"find\s+(?:the|that|when|where)|"
    r"credits?|post[\s-]?credits?|"
    r"(?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th))\s+"
    r"(?:goal|fight|scene|kiss|moment|point)|"
    r"\d{1,2}:\d{2}(?::\d{2})?"
    r")\b"
)

_CONTENT = re.compile(
    r"(?ix)\b("
    r"what|who|why|when|where|how|"
    r"happen(?:ed|ing)?|going\s+on|talking|doing|said|saying|"
    r"plot|story|scene|episode|show|movie|"
    r"character|actor|actress|cast|director|creator|"
    r"name|kill(?:er|ed)?|murder|reveal|twist|ending|"
    r"before|earlier|again|remember|met|"
    r"relationship|related|know\s+each\s+other|"
    r"on\s+screen|that\s+(?:guy|girl|man|woman|kid|person)|"
    r"is\s+(?:he|she|that|this|they)|"
    r"are\s+they|did\s+(?:he|she|they)|"
    r"tell\s+me|explain|recap|summar(?:y|ize)|"
    r"joke|one[\s-]?liner|funny|laugh"
    r")\b"
)

_GATE_CLASSIFIER_PROMPT = """You classify a viewer's spoken line during a TV show.

Reply with only YES or NO.

YES = meaningful co-watch request about the show, a character, plot so far, cast, a joke about the show, or navigating to a moment.
NO = noise, filler, accident, empty small-talk only, or completely off-topic (weather, coding, etc.).

Reply with only YES or NO."""


@dataclass(frozen=True)
class UtteranceDecision:
    action: GateAction
    reason: str
    reply: str
    speak: bool
    usage: TokenUsage | None = None

    @property
    def short_circuit(self) -> bool:
        """True when the agent should not run tools / full conversation."""
        return self.action in ("ignore", "social", "off_topic")

    @property
    def skip_memory(self) -> bool:
        return self.action in ("ignore",) or (self.action == "off_topic" and not self.reply)


def classify_utterance(
    question: str,
    *,
    settings: Settings,
    completion: CompletionClient | None = None,
    persona_id: str | None = None,
) -> UtteranceDecision:
    """Heuristic gate first; optional mini LLM only when ambiguous."""
    decision = _classify_utterance_inner(
        question, settings=settings, completion=completion
    )
    _record_gate_outcome(decision, persona_id=persona_id)
    return decision


def _record_gate_outcome(
    decision: UtteranceDecision,
    *,
    persona_id: str | None = None,
) -> None:
    reason = decision.reason or ""
    if reason.startswith("gate:prompt_"):
        outcome = "prompt_llm"
    elif decision.short_circuit:
        outcome = "free"
    else:
        # navigate / joke / content / merged_pending — agent path continues
        outcome = "agent"
    try:
        from ai_cowatcher.observability.prometheus_metrics import record_utterance_gate

        record_utterance_gate(
            outcome=outcome,
            action=decision.action,
            persona_id=persona_id or "",
        )
    except Exception:  # noqa: BLE001
        pass
    logger.info(
        json.dumps(
            {
                "event": "utterance_gate",
                "outcome": outcome,
                "action": decision.action,
                "reason": decision.reason,
                "short_circuit": decision.short_circuit,
                "persona_id": persona_id or "",
            },
            separators=(",", ":"),
        )
    )


def _classify_utterance_inner(
    question: str,
    *,
    settings: Settings,
    completion: CompletionClient | None = None,
) -> UtteranceDecision:
    """Heuristic gate first; optional mini LLM only when ambiguous."""
    if not getattr(settings, "utterance_gate_enabled", True):
        return UtteranceDecision(
            action="content",
            reason="gate_disabled",
            reply="",
            speak=True,
        )

    raw = (question or "").strip()
    cleaned = _normalize(raw)

    if not cleaned:
        return UtteranceDecision(
            action="ignore",
            reason="gate:ignore_empty",
            reply=_IGNORE_REPLY,
            speak=False,
        )

    if len(cleaned) < 2:
        return UtteranceDecision(
            action="ignore",
            reason="gate:ignore_too_short",
            reply=_IGNORE_REPLY,
            speak=False,
        )

    if _is_noise_or_filler(cleaned):
        return UtteranceDecision(
            action="ignore",
            reason="gate:ignore_filler",
            reply=_IGNORE_REPLY,
            speak=False,
        )

    if _SOCIAL.fullmatch(cleaned) or _SOCIAL_PHRASE.fullmatch(cleaned):
        return UtteranceDecision(
            action="social",
            reason="gate:social",
            reply=_SOCIAL_REPLY,
            speak=True,
        )

    # Off-topic before content: "what's the weather" should not open the agent.
    if _OFF_TOPIC.search(cleaned):
        return UtteranceDecision(
            action="off_topic",
            reason="gate:off_topic",
            reply=_OFF_TOPIC_REPLY,
            speak=True,
        )

    if is_joke_request(raw):
        return UtteranceDecision(
            action="joke",
            reason="gate:joke",
            reply="",
            speak=True,
        )

    if _NAVIGATE.search(cleaned):
        return UtteranceDecision(
            action="navigate",
            reason="gate:navigate",
            reply="",
            speak=True,
        )

    if _CONTENT.search(cleaned):
        return UtteranceDecision(
            action="content",
            reason="gate:content",
            reply="",
            speak=True,
        )

    # Ambiguous: short or medium phrase without clear markers.
    strategy = getattr(settings, "utterance_gate_strategy", "heuristic")
    if strategy == "merged":
        # Intent + answer share one model call later; do not run a second classifier.
        return UtteranceDecision(
            action="content",
            reason="gate:merged_pending",
            reply="",
            speak=True,
        )
    if strategy == "prompt" and completion is not None:
        yes, reason, usage = _prompt_is_meaningful(cleaned, settings, completion)
        if yes:
            return UtteranceDecision(
                action="content",
                reason=reason,
                reply="",
                speak=True,
                usage=usage,
            )
        return UtteranceDecision(
            action="ignore",
            reason=reason,
            reply=_AMBIGUOUS_IGNORE_REPLY,
            speak=False,
            usage=usage,
        )

    # Pure heuristic: ambiguous short speech is ignored; longer unknown gets a soft reject.
    if len(cleaned) <= 24:
        return UtteranceDecision(
            action="ignore",
            reason="gate:ignore_ambiguous",
            reply=_IGNORE_REPLY,
            speak=False,
        )

    return UtteranceDecision(
        action="off_topic",
        reason="gate:off_topic_ambiguous",
        reply=_OFF_TOPIC_REPLY,
        speak=True,
    )


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[“”\"']", "", text)
    # Collapse punctuation STT often leaves on fillers ("Wait—", "Hmm…")
    text = re.sub(r"[—–\-…]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,!?…")
    return text


def _is_noise_or_filler(cleaned: str) -> bool:
    if _FILLER_ONLY.fullmatch(cleaned):
        return True
    if _FILLER_PHRASE.fullmatch(cleaned):
        return True
    # Mostly non-letters (STT garbage).
    letters = sum(1 for c in cleaned if c.isalpha())
    if letters < 2:
        return True
    if letters / max(len(cleaned), 1) < 0.35 and len(cleaned) < 20:
        return True
    # Repeated single-token babble
    tokens = cleaned.split()
    if len(tokens) <= 3 and all(re.fullmatch(r"(um|uh|hmm|ah|oh|like|so)+", t) for t in tokens):
        return True
    return False


def _prompt_is_meaningful(
    cleaned: str,
    settings: Settings,
    completion: CompletionClient,
) -> tuple[bool, str, TokenUsage | None]:
    result = completion.complete(
        model=settings.conversation_fast_model,
        messages=[
            {"role": "system", "content": _GATE_CLASSIFIER_PROMPT},
            {"role": "user", "content": cleaned},
        ],
        temperature=0.0,
        max_tokens=8,
    )
    answer = (result.content or "").strip().upper()
    if answer.startswith("YES"):
        return True, "gate:prompt_yes", result.usage
    return False, "gate:prompt_no", result.usage
