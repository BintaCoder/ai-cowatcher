"""Gemini token cost estimation and session budget tracking."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from threading import Lock
from typing import Any

logger = logging.getLogger("ai_cowatcher.cost")

# Approx public Gemini Flash-class rates (USD / 1M tokens). Override via Settings.
# Used only for pilot dashboards — not billing.
DEFAULT_INPUT_USD_PER_MTOK = 0.10
DEFAULT_OUTPUT_USD_PER_MTOK = 0.40


def estimate_tokens_from_text(text: str) -> int:
    """Rough token estimate for logging (~4 chars/token English)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += estimate_tokens_from_text(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += estimate_tokens_from_text(part["text"])
                elif isinstance(part, str):
                    total += estimate_tokens_from_text(part)
        tools = message.get("tool_calls")
        if tools:
            total += estimate_tokens_from_text(json.dumps(tools)[:8000])
    return total


@dataclass(frozen=True)
class LlmCallCost:
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    estimated_usd: float
    path: str  # merged | legacy | other


def estimate_usd(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    input_usd_per_mtok: float = DEFAULT_INPUT_USD_PER_MTOK,
    output_usd_per_mtok: float = DEFAULT_OUTPUT_USD_PER_MTOK,
) -> float:
    return (prompt_tokens / 1_000_000.0) * input_usd_per_mtok + (
        completion_tokens / 1_000_000.0
    ) * output_usd_per_mtok


_lock = Lock()
_session_usd: dict[str, float] = {}
_prompt_tokens_samples: list[int] = []


def record_session_spend(user_id: str, amount_usd: float) -> float:
    """Accumulate estimated spend; returns session total."""
    key = (user_id or "anonymous").strip() or "anonymous"
    with _lock:
        _session_usd[key] = _session_usd.get(key, 0.0) + max(0.0, amount_usd)
        return _session_usd[key]


def session_spend(user_id: str) -> float:
    key = (user_id or "anonymous").strip() or "anonymous"
    with _lock:
        return _session_usd.get(key, 0.0)


def note_prompt_tokens(n: int) -> None:
    if n <= 0:
        return
    with _lock:
        _prompt_tokens_samples.append(n)
        if len(_prompt_tokens_samples) > 500:
            del _prompt_tokens_samples[:250]


def average_prompt_tokens() -> float:
    with _lock:
        if not _prompt_tokens_samples:
            return 0.0
        return sum(_prompt_tokens_samples) / len(_prompt_tokens_samples)


def record_llm_call(
    *,
    model_id: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    path: str = "other",
    user_id: str | None = None,
    input_usd_per_mtok: float = DEFAULT_INPUT_USD_PER_MTOK,
    output_usd_per_mtok: float = DEFAULT_OUTPUT_USD_PER_MTOK,
    session_budget_usd: float = 0.50,
) -> LlmCallCost:
    """Log structured cost, update Prometheus + session budget."""
    p = int(prompt_tokens or 0)
    c = int(completion_tokens or 0)
    usd = estimate_usd(
        prompt_tokens=p,
        completion_tokens=c,
        input_usd_per_mtok=input_usd_per_mtok,
        output_usd_per_mtok=output_usd_per_mtok,
    )
    note_prompt_tokens(p)
    call = LlmCallCost(
        model_id=model_id or "unknown",
        prompt_tokens=p,
        completion_tokens=c,
        estimated_usd=usd,
        path=path,
    )
    logger.info(
        json.dumps(
            {
                "event": "llm_call_cost",
                "model_id": call.model_id,
                "input_tokens": call.prompt_tokens,
                "output_tokens": call.completion_tokens,
                "estimated_cost": round(call.estimated_usd, 8),
                "path": call.path,
                "user_id": user_id or "",
            },
            separators=(",", ":"),
        )
    )
    try:
        from ai_cowatcher.observability.prometheus_metrics import (
            record_llm_token_usage,
            record_session_budget_alert,
        )

        record_llm_token_usage(
            model=call.model_id,
            path=call.path,
            prompt_tokens=p,
            completion_tokens=c,
            estimated_usd=usd,
        )
    except Exception:  # noqa: BLE001
        pass

    if user_id:
        total = record_session_spend(user_id, usd)
        if session_budget_usd > 0 and total > session_budget_usd:
            logger.warning(
                json.dumps(
                    {
                        "event": "session_cost_budget_exceeded",
                        "user_id": user_id,
                        "session_usd": round(total, 6),
                        "budget_usd": session_budget_usd,
                    },
                    separators=(",", ":"),
                )
            )
            try:
                from ai_cowatcher.observability.prometheus_metrics import (
                    record_session_budget_alert,
                )

                record_session_budget_alert()
            except Exception:  # noqa: BLE001
                pass
    return call


def is_gemini_model(model: str) -> bool:
    return bool(re.search(r"gemini", model or "", re.I))
