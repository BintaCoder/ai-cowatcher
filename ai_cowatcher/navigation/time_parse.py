"""Parse natural-language and clock-style timestamps into seconds."""

from __future__ import annotations

import re

_TIME_AT_RE = re.compile(
    r"(?:at\s+)?(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(?:in(?:to)?\s+the\s+(?:movie|video|show|film))?",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b",
    re.IGNORECASE,
)
_GO_TO_TIME_RE = re.compile(
    r"\b(?:go|jump|skip|take me)\s+to\s+(\d{1,2}:\d{2}(?::\d{2})?|\d+(?:\.\d+)?\s*(?:hours?|hrs?|h|minutes?|mins?|m))\b",
    re.IGNORECASE,
)


def parse_clock_timestamp(text: str) -> float | None:
    """Parse H:MM, HH:MM:SS, or M:SS from text."""
    match = _TIME_AT_RE.search(text) or re.search(
        r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b", text
    )
    if not match:
        return None
    hours = 0
    if match.lastindex and match.lastindex >= 3 and match.group(3):
        hours, minutes, seconds = int(match.group(1)), int(match.group(2)), int(match.group(3))
    elif int(match.group(1)) > 59:
        hours, minutes, seconds = int(match.group(1)), int(match.group(2)), 0
    else:
        minutes, seconds = int(match.group(1)), int(match.group(2))
    return float(hours * 3600 + minutes * 60 + seconds)


def parse_duration_seconds(text: str) -> float | None:
    """Parse '10 minutes', '1.5 hours', etc."""
    total = 0.0
    found = False
    for match in _DURATION_RE.finditer(text):
        found = True
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith("h"):
            total += value * 3600
        elif unit.startswith("m"):
            total += value * 60
        else:
            total += value
    return total if found else None


def parse_seek_timestamp(question: str) -> float | None:
    """Best-effort absolute seek target in seconds from a navigation question."""
    go_match = _GO_TO_TIME_RE.search(question)
    if go_match:
        fragment = go_match.group(1)
        if ":" in fragment:
            return parse_clock_timestamp(fragment)
        duration = parse_duration_seconds(question)
        if duration is not None:
            return duration

    clock = parse_clock_timestamp(question)
    if clock is not None:
        return clock

    return parse_duration_seconds(question)
