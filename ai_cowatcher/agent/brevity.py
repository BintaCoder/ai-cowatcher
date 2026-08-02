"""Enforce couch-friend brevity on co-watcher answers."""

from __future__ import annotations

import re

# Hard ceiling for a whisper-during-movie answer (~one breath).
_DEFAULT_MAX_WORDS = 28
_DEFAULT_MAX_SENTENCES = 1
_JOKE_MAX_WORDS = 20
_JOKE_MAX_SENTENCES = 1
_DETAIL_MAX_WORDS = 80
_DETAIL_MAX_SENTENCES = 3

_DETAIL_INTENT = re.compile(
    r"\b("
    r"tell me more|in detail|more detail|everything so far|give me more|"
    r"explain (more|it|that)|go on|elaborate|full story|recap|"
    r"walk me through|what all happened|what's been going on"
    r")\b",
    re.IGNORECASE,
)

# Lazy import avoided: joke detection lives in joke_intent; brevity only needs a flag.

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def wants_detailed_answer(question: str) -> bool:
    return bool(_DETAIL_INTENT.search(question or ""))


def enforce_brief_answer(
    text: str,
    question: str = "",
    *,
    joke_mode: bool = False,
) -> str:
    """Trim chatty model output into a short, speakable co-watch reply."""
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return cleaned

    detail = wants_detailed_answer(question) and not joke_mode
    if joke_mode:
        max_sentences = _JOKE_MAX_SENTENCES
        max_words = _JOKE_MAX_WORDS
    elif detail:
        max_sentences = _DETAIL_MAX_SENTENCES
        max_words = _DETAIL_MAX_WORDS
    else:
        max_sentences = _DEFAULT_MAX_SENTENCES
        max_words = _DEFAULT_MAX_WORDS

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(cleaned) if s.strip()]
    if not sentences:
        return cleaned

    kept: list[str] = []
    word_count = 0
    for sentence in sentences:
        words = sentence.split()
        if not kept:
            # Always keep the start of the first sentence, clip hard if needed.
            if len(words) > max_words:
                clipped = " ".join(words[:max_words]).rstrip(",;:") + "…"
                kept.append(clipped)
                break
            kept.append(sentence)
            word_count = len(words)
            if len(kept) >= max_sentences:
                break
            continue

        if len(kept) >= max_sentences or word_count + len(words) > max_words:
            break
        kept.append(sentence)
        word_count += len(words)

    result = " ".join(kept).strip()
    return result or cleaned
