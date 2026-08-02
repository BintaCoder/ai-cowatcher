"""Grounded fallbacks when the model refuses despite tool evidence."""

from __future__ import annotations

import json
import re
from typing import Any

from ai_cowatcher.agent.brevity import enforce_brief_answer

_REFUSAL_RE = re.compile(
    r"\b("
    r"i'?m not sure"
    r"|not sure( yet)?"
    r"|nothing'?s made that clear"
    r"|don'?t know( yet)?"
    r"|do not know( yet)?"
    r"|i don'?t know"
    r"|no idea"
    r"|can'?t tell"
    r"|unclear"
    r")\b",
    re.IGNORECASE,
)

_WHAT_INTENT = re.compile(
    r"\b(what|happen|happened|happening|going on|talking|doing|said)\b",
    re.I,
)
_WHO_INTENT = re.compile(r"\b(who|name|actor)\b", re.I)
_SPOILER_ENTITY = re.compile(r"\b(killer|murderer|twist|ending)\b", re.I)
_KILLER_IN_TEXT = re.compile(r"\bkiller\b|\bmurderer\b", re.I)


def is_refusal_answer(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    return bool(_REFUSAL_RE.search(cleaned))


def _short_snippet(text: str, max_words: int = 16) -> str:
    words = " ".join(str(text).split()).split()
    if not words:
        return ""
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(",;:") + "…"


def collect_scene_snippets(tool_payloads: list[Any]) -> list[str]:
    snippets: list[str] = []
    for payload in tool_payloads:
        scenes = payload
        if isinstance(payload, str):
            try:
                scenes = json.loads(payload)
            except json.JSONDecodeError:
                continue
        if not isinstance(scenes, list):
            continue
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            text = (scene.get("transcript") or scene.get("caption") or "").strip()
            if text:
                snippets.append(text)
    return snippets


def grounded_fallback_answer(question: str, tool_payloads: list[Any]) -> str | None:
    """Build a short couch-friend line from tool evidence when the LLM refuses.

    Important: do not invent answers for spoiler-sensitive questions (e.g. killer)
    when tool text does not actually contain that information.
    """
    snippets = collect_scene_snippets(tool_payloads)
    if not snippets:
        return None

    lower = (question or "").lower()

    # Spoiler-sensitive entity questions: only if tools mention it.
    if _SPOILER_ENTITY.search(lower):
        for snippet in reversed(snippets):
            if _KILLER_IN_TEXT.search(snippet) or _SPOILER_ENTITY.search(snippet):
                return enforce_brief_answer(_short_snippet(snippet, max_words=18), question)
        return None

    # "What happened?" style — best-effort from recent dialogue.
    if _WHAT_INTENT.search(lower):
        snippet = _short_snippet(snippets[-1], max_words=18)
        if not snippet:
            return None
        draft = snippet if snippet[:1].isupper() else snippet[0].upper() + snippet[1:]
        return enforce_brief_answer(draft, question)

    # "Who is that?" without a proper name in tools — admit name unknown, use dialogue.
    if _WHO_INTENT.search(lower):
        snippet = _short_snippet(snippets[-1], max_words=14)
        if not snippet:
            return None
        draft = f"They haven’t named them yet — {snippet}"
        return enforce_brief_answer(draft, question)

    # Other intents: leave LLM refusal if it's cautious; don't over-apply.
    return None
