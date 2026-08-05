"""Detect mid-watch joke / one-liner requests and soft joke fallbacks."""

from __future__ import annotations

import re
from typing import Any

from ai_cowatcher.agent.brevity import enforce_brief_answer

# Explicit joke asks (whole question or clear phrase). Avoid matching
# "what's the joke about?" style plot questions incorrectly when possible by
# requiring joke intent words without heavy plot-interrogative framing.
_JOKE_PHRASE = re.compile(
    r"(?ix)"
    r"(?:^|\b)"
    r"(?:"
    r"tell\s+me\s+a\s+joke"
    r"|got\s+a\s+(?:good\s+)?joke"
    r"|give\s+me\s+a\s+joke"
    r"|say\s+something\s+funny"
    r"|make\s+me\s+laugh"
    r"|one[\s-]?liner"
    r"|one[\s-]?liners"
    r"|funny\s+(?:line|one|bit)"
    r"|crack\s+a\s+joke"
    r"|hit\s+me\s+with\s+a\s+joke"
    r"|joke\s+please"
    r"|be\s+funny"
    r"|something\s+funny"
    r"|any\s+jokes?"
    r"|entertain\s+me"
    r"|jokes?"
    r")"
    r"(?:\b|$|[!?.…]*)"
)

_PLOT_NOT_JOKE = re.compile(
    r"(?ix)\b("
    r"what(?:'s|\s+is)\s+the\s+joke"
    r"|why\s+(?:is\s+)?(?:that|this)\s+funny"
    r"|joke\s+about"
    r"|running\s+joke"
    r")\b"
)

_SCENE_JOKE_QUERY = "funny dialogue conversation banter what just happened"


def is_joke_request(question: str) -> bool:
    """True when the viewer wants a short content-flavoured gag, not plot Q&A."""
    q = (question or "").strip()
    if not q or len(q) > 120:
        return False
    if _PLOT_NOT_JOKE.search(q):
        return False
    return bool(_JOKE_PHRASE.search(q))


def joke_scene_query(question: str) -> str:
    """Query text that pulls dialogue/banter for a scene-flavoured gag."""
    del question  # reserved for future tuning
    return _SCENE_JOKE_QUERY


def joke_fallback_answer(tool_payloads: list[Any]) -> str | None:
    """Lightweight gag when the model fails but scenes exist."""
    transcript, caption = _first_scene_bits(tool_payloads)
    snip = transcript or caption
    if not snip:
        return (
            "I've got nothing funny yet — the screen's still being coy."
        )
    short = " ".join(str(snip).split())
    words = short.split()
    if len(words) > 12:
        short = " ".join(words[:12]).rstrip(",;:")
    # Playful wrap, not a plot recap.
    line = f"Plot twist for the couch: \"{short}.\""
    return enforce_brief_answer(line, "joke")


def soft_no_scene_joke() -> str:
    return "I've got nothing to riff on yet — rewind a beat and try me again."


def _first_scene_bits(payloads: list[Any]) -> tuple[str, str]:
    for payload in payloads:
        scenes = _as_scene_list(payload)
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            transcript = str(scene.get("transcript") or "").strip()
            caption = str(scene.get("caption") or "").strip()
            if transcript or caption:
                return transcript, caption
    return "", ""


def _as_scene_list(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, str):
        import json

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []
