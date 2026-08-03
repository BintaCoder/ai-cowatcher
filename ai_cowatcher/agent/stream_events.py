"""SSE event types for progressive /ask streaming."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


AskEventType = Literal[
    "status", "tool_start", "tool_end", "intent", "token", "done", "error"
]


@dataclass(frozen=True)
class AskStreamEvent:
    type: AskEventType
    message: str | None = None
    tool: str | None = None
    text: str | None = None
    answer: str | None = None
    title_id: str | None = None
    user_id: str | None = None
    current_ts: float | None = None
    model_tier: str | None = None
    model_name: str | None = None
    escalation_reason: str | None = None
    used_context: bool | None = None
    latency_ms: float | None = None
    detail: str | None = None
    speak: bool | None = None
    skip_memory: bool | None = None
    # Merged gate tags: FILLER | SOCIAL | JOKE | NAVIGATE | CONTENT
    intent: str | None = None
    navigate: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}
