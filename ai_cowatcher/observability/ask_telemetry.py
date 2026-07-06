"""Pilot observability for /ask — structured logs and in-memory rollups."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from threading import Lock

logger = logging.getLogger("ai_cowatcher.ask")

ASK_EVENT = "ask_request"
_UNKNOWN_PHRASE = "don't know yet"

_lock = Lock()
_records: list[AskRecord] = []


@dataclass(frozen=True)
class AskRecord:
    title_id: str
    user_id: str
    current_ts: float
    latency_ms: float
    model_tier: str
    model_name: str
    escalation_reason: str
    used_context: bool
    dont_know: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


def is_dont_know_answer(answer: str) -> bool:
    return _UNKNOWN_PHRASE in answer.lower()


def record_ask_request(record: AskRecord) -> None:
    with _lock:
        _records.append(record)

    payload = {
        "event": ASK_EVENT,
        **asdict(record),
    }
    logger.info(json.dumps(payload, separators=(",", ":")))


def _build_summary(records: list[AskRecord]) -> dict[str, object]:
    if not records:
        return {
            "ask_count": 0,
            "average_latency_ms": 0.0,
            "tier_usage": {"fast": 0, "escalated": 0},
            "escalation_rate": 0.0,
            "dont_know_rate_overall": 0.0,
            "by_title": {},
        }

    ask_count = len(records)
    average_latency_ms = sum(record.latency_ms for record in records) / ask_count
    tier_usage = {
        "fast": sum(1 for record in records if record.model_tier == "fast"),
        "escalated": sum(1 for record in records if record.model_tier == "escalated"),
    }
    escalation_rate = tier_usage["escalated"] / ask_count
    dont_know_count = sum(1 for record in records if record.dont_know)
    dont_know_rate_overall = dont_know_count / ask_count

    by_title: dict[str, dict[str, object]] = {}
    for record in records:
        title_stats = by_title.setdefault(
            record.title_id,
            {
                "ask_count": 0,
                "dont_know_count": 0,
                "dont_know_rate": 0.0,
                "average_latency_ms": 0.0,
                "tier_usage": {"fast": 0, "escalated": 0},
            },
        )
        title_stats["ask_count"] = int(title_stats["ask_count"]) + 1
        if record.dont_know:
            title_stats["dont_know_count"] = int(title_stats["dont_know_count"]) + 1
        tier_key = record.model_tier if record.model_tier in ("fast", "escalated") else "fast"
        title_tier_usage = title_stats["tier_usage"]
        assert isinstance(title_tier_usage, dict)
        title_tier_usage[tier_key] = int(title_tier_usage.get(tier_key, 0)) + 1

    for title_id, title_stats in by_title.items():
        title_ask_count = int(title_stats["ask_count"])
        title_dont_know_count = int(title_stats["dont_know_count"])
        title_stats["dont_know_rate"] = title_dont_know_count / title_ask_count
        title_records = [record for record in records if record.title_id == title_id]
        title_stats["average_latency_ms"] = sum(
            record.latency_ms for record in title_records
        ) / title_ask_count
        del title_id

    return {
        "ask_count": ask_count,
        "average_latency_ms": round(average_latency_ms, 2),
        "tier_usage": tier_usage,
        "escalation_rate": round(escalation_rate, 4),
        "dont_know_rate_overall": round(dont_know_rate_overall, 4),
        "by_title": by_title,
    }


def metrics_lite_summary() -> dict[str, object]:
    with _lock:
        records = list(_records)
    return _build_summary(records)


def conversation_tier_counts() -> dict[str, int]:
    summary = metrics_lite_summary()
    tier_usage = summary.get("tier_usage", {})
    if isinstance(tier_usage, dict):
        return {
            "fast": int(tier_usage.get("fast", 0)),
            "escalated": int(tier_usage.get("escalated", 0)),
        }
    return {"fast": 0, "escalated": 0}


def reset_ask_telemetry() -> None:
    with _lock:
        _records.clear()


def summarize_ask_log_lines(lines: list[str]) -> dict[str, object]:
    """Build the same rollup as /metrics-lite from JSON log lines."""
    records: list[AskRecord] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") != ASK_EVENT:
            continue
        records.append(
            AskRecord(
                title_id=str(payload["title_id"]),
                user_id=str(payload.get("user_id", "")),
                current_ts=float(payload["current_ts"]),
                latency_ms=float(payload["latency_ms"]),
                model_tier=str(payload["model_tier"]),
                model_name=str(payload.get("model_name", "")),
                escalation_reason=str(payload.get("escalation_reason", "")),
                used_context=bool(payload.get("used_context")),
                dont_know=bool(payload.get("dont_know")),
                prompt_tokens=payload.get("prompt_tokens"),
                completion_tokens=payload.get("completion_tokens"),
                total_tokens=payload.get("total_tokens"),
            )
        )

    return _build_summary(records)
