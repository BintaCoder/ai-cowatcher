"""Extract ordinal index and cleaned query from navigation questions."""

from __future__ import annotations

import re

_ORDINAL_WORDS = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
    "sixth": 6,
    "6th": 6,
    "seventh": 7,
    "7th": 7,
    "eighth": 8,
    "8th": 8,
    "ninth": 9,
    "9th": 9,
    "tenth": 10,
    "10th": 10,
}

_ORDINAL_PREFIX_RE = re.compile(
    r"(?:the\s+)?(?P<word>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"\d{1,2}(?:st|nd|rd|th)?)\s+",
    re.IGNORECASE,
)

_NAV_VERB_RE = re.compile(
    r"^(?:please\s+)?(?:take me to|go to|jump to|skip to|show me|find|where is|where's)\s+",
    re.IGNORECASE,
)


def parse_ordinal(question: str) -> tuple[int | None, str]:
    """Return (1-based ordinal or None, query with nav verbs/ordinal stripped)."""
    text = question.strip()
    text = _NAV_VERB_RE.sub("", text).strip()

    ordinal: int | None = None
    match = _ORDINAL_PREFIX_RE.match(text)
    if match:
        word = match.group("word").lower()
        if word in _ORDINAL_WORDS:
            ordinal = _ORDINAL_WORDS[word]
        elif word.isdigit():
            ordinal = int(word)
        else:
            digits = re.match(r"(\d+)", word)
            if digits:
                ordinal = int(digits.group(1))
        text = text[match.end() :].strip()

    text = re.sub(r"\s+(scene|moment|part)$", "", text, flags=re.IGNORECASE).strip()
    return ordinal, text or question.strip()
