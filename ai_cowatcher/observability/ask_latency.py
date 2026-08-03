"""Per-stage latency tracker for /ask and /ask/stream.

Lightweight: context managers + structured JSON logs + Prometheus histograms.
Safe to use in MOCK_MODE; never raises into the ask path.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger("ai_cowatcher.ask.latency")

LATENCY_EVENT = "ask_latency_stages"

# Canonical stage names (histograms labeled by these).
STAGE_GATE = "gate"
STAGE_CACHE_LOOKUP = "cache_lookup"
STAGE_CACHE_EXACT = "cache_exact"
STAGE_CACHE_SEMANTIC = "cache_semantic"
STAGE_SCENE_RETRIEVE = "scene_retrieve"
STAGE_SCENE_PLAYHEAD = "scene_playhead"
STAGE_SCENE_EMBED = "scene_embed"
STAGE_LLM_TTFT = "llm_ttft"
STAGE_LLM_TOTAL = "llm_total"
STAGE_MULTIMODAL = "multimodal"
STAGE_TOTAL = "total"


@dataclass
class AskLatencyTracker:
    """Accumulate wall-clock stage durations for a single ask request."""

    path: str = "ask"  # ask | ask_stream
    _t0: float = field(default_factory=time.perf_counter)
    stages_ms: dict[str, float] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)
    _llm_request_sent: float | None = field(default=None, repr=False)
    _llm_first_token: float | None = field(default=None, repr=False)

    def set_meta(self, **kwargs: object) -> None:
        self.meta.update(kwargs)

    def record_ms(self, stage: str, duration_ms: float) -> None:
        if duration_ms < 0:
            return
        # Prefer first mark for TTFT; sum-safe for nested exact/semantic under cache.
        existing = self.stages_ms.get(stage)
        if existing is None or stage == STAGE_LLM_TTFT:
            self.stages_ms[stage] = round(duration_ms, 2)
        else:
            self.stages_ms[stage] = round(existing + duration_ms, 2)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record_ms(name, (time.perf_counter() - started) * 1000.0)

    def mark_llm_request_sent(self) -> None:
        self._llm_request_sent = time.perf_counter()

    def mark_llm_first_token(self) -> None:
        if self._llm_first_token is not None:
            return
        now = time.perf_counter()
        self._llm_first_token = now
        if self._llm_request_sent is not None:
            self.record_ms(
                STAGE_LLM_TTFT, (now - self._llm_request_sent) * 1000.0
            )

    def mark_llm_stream_complete(self) -> None:
        if self._llm_request_sent is None:
            return
        self.record_ms(
            STAGE_LLM_TOTAL, (time.perf_counter() - self._llm_request_sent) * 1000.0
        )

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000.0, 2)

    def finish(self, **extra: object) -> dict[str, object]:
        """Finalize total, emit log + metrics, return payload dict."""
        self.record_ms(STAGE_TOTAL, self.total_ms())
        if extra:
            self.meta.update(extra)
        payload: dict[str, object] = {
            "event": LATENCY_EVENT,
            "path": self.path,
            "stages_ms": dict(self.stages_ms),
            **self.meta,
        }
        try:
            logger.info(json.dumps(payload, separators=(",", ":")))
        except Exception:  # noqa: BLE001
            logger.exception("Failed to log ask latency stages")
        try:
            from ai_cowatcher.observability.prometheus_metrics import (
                observe_ask_stages,
            )

            observe_ask_stages(self.stages_ms)
        except Exception:  # noqa: BLE001
            pass
        return payload
