"""Pure helpers for ask-bench sampling (no network, MOCK_MODE-safe)."""

from __future__ import annotations

import random
from typing import Any, Sequence


def parse_cache_source(
    model_name: str | None,
    *,
    model_tier: str | None = None,
    escalation_reason: str | None = None,
) -> str:
    """Derive low-cardinality cache_source from /ask response fields.

    Server stamps cache hits as ``model_name="qa_cache:exact|semantic"`` and
    ``model_tier="cache"``. Agent/LLM answers after a cache miss use a real
    model name → recorded as ``miss``. Free-gate replies use ``gate:free``
    → ``free`` (no QA cache / no LLM). Empty / missing fields → ``none``.
    """
    name = (model_name or "").strip()
    lower = name.lower()
    if lower.startswith("qa_cache:"):
        src = lower.split(":", 1)[1].strip()
        if src in ("exact", "semantic"):
            return src
        if src:
            return src
        return "miss"

    if lower.startswith("gate:") or (model_tier or "").strip().lower() == "gate":
        return "free"

    reason = (escalation_reason or "").strip().lower()
    if reason.startswith("cache:"):
        src = reason.split(":", 1)[1].strip()
        if src in ("exact", "semantic"):
            return src

    tier = (model_tier or "").strip().lower()
    if tier == "cache":
        return "exact"

    if not name:
        return "none"
    return "miss"


def clamp_playhead(current_ts: float, duration_sec: float, *, epsilon: float = 0.05) -> float:
    """Clamp playhead into [0, max(0, duration - epsilon)]."""
    if duration_sec <= 0:
        return 0.0
    upper = max(0.0, float(duration_sec) - float(epsilon))
    ts = float(current_ts)
    if ts < 0.0:
        return 0.0
    if ts > upper:
        return upper
    return ts


def sample_questions(
    questions: Sequence[dict[str, Any]],
    *,
    n: int,
    duration_sec: float,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Pick ``n`` questions uniformly (without replacement when possible).

    Each item is a question dict plus ``current_ts`` drawn uniformly in
    ``[0, duration]`` and clamped.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not questions:
        raise ValueError("question bank is empty")

    rng = random.Random(seed)
    pool = list(questions)
    if n <= len(pool):
        chosen = rng.sample(pool, n)
    else:
        chosen = [pool[rng.randrange(len(pool))] for _ in range(n)]

    samples: list[dict[str, Any]] = []
    for q in chosen:
        raw_ts = rng.uniform(0.0, max(0.0, float(duration_sec)))
        samples.append(
            {
                "id": str(q["id"]),
                "text": str(q["text"]),
                "kind": str(q.get("kind") or ""),
                "current_ts": clamp_playhead(raw_ts, duration_sec),
            }
        )
    return samples
