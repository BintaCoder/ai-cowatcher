"""Prometheus metrics for the co-watcher pilot."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import Counter, Gauge, Histogram

from ai_cowatcher.observability.ask_telemetry import AskRecord

# ── Real-time /ask ────────────────────────────────────────────────────────────

ASK_REQUEST_DURATION = Histogram(
    "cowatcher_ask_request_duration_seconds",
    "End-to-end latency for POST /ask",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

ASK_REQUESTS_TOTAL = Counter(
    "cowatcher_ask_requests_total",
    "Total /ask requests",
    labelnames=("status",),
)

ASK_DONT_KNOW_TOTAL = Counter(
    "cowatcher_ask_dont_know_total",
    "Answers containing the pilot don't-know phrase",
)

ASK_MODEL_TIER_TOTAL = Counter(
    "cowatcher_ask_model_tier_total",
    "Model tier selected for /ask",
    labelnames=("tier",),
)

ASK_STAGE_DURATION = Histogram(
    "cowatcher_ask_stage_duration_seconds",
    "Per-stage latency for POST /ask and /ask/stream",
    labelnames=("stage", "persona_id"),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# ── Tool calls ────────────────────────────────────────────────────────────────

TOOL_CALL_DURATION = Histogram(
    "cowatcher_tool_call_duration_seconds",
    "Latency of agent tool invocations",
    labelnames=("tool",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

TOOL_CALLS_TOTAL = Counter(
    "cowatcher_tool_calls_total",
    "Agent tool invocations",
    labelnames=("tool", "outcome"),
)

# ── Storage backends ──────────────────────────────────────────────────────────

STORAGE_QUERY_DURATION = Histogram(
    "cowatcher_storage_query_duration_seconds",
    "Latency of storage layer queries",
    labelnames=("backend", "operation"),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# ── Offline ingestion ─────────────────────────────────────────────────────────

INGEST_JOB_DURATION = Histogram(
    "cowatcher_ingest_job_duration_seconds",
    "Duration of a full title ingestion job",
    buckets=(30, 60, 120, 300, 600, 1200, 1800, 3600, 7200),
)

INGEST_JOBS_TOTAL = Counter(
    "cowatcher_ingest_jobs_total",
    "Ingestion jobs processed by the worker",
    labelnames=("status",),
)

INGEST_SCENES_PROCESSED_TOTAL = Counter(
    "cowatcher_ingest_scenes_processed_total",
    "Scenes newly persisted during ingestion",
)

INGEST_QUEUE_DEPTH = Gauge(
    "cowatcher_ingest_queue_depth",
    "Approximate ingest queue depth (broker-specific)",
    labelnames=("broker",),
)

# ── Cost / QA cache ───────────────────────────────────────────────────────────

QA_CACHE_LOOKUPS_TOTAL = Counter(
    "cowatcher_qa_cache_lookups_total",
    "QA cache lookup outcomes",
    labelnames=("result",),  # exact_hit | semantic_hit | miss
)

QA_CACHE_HIT_TOTAL = Counter(
    "cowatcher_qa_cache_hit_total",
    "QA cache hits",
    labelnames=("source",),  # exact | semantic
)

QA_CACHE_MISS_TOTAL = Counter(
    "cowatcher_qa_cache_miss_total",
    "QA cache full misses (exact + semantic miss)",
)

# PromQL for primary dashboards (hit rate is first-class; counters above feed it):
#   sum(rate(cowatcher_qa_cache_hit_total[5m]))
#     / (sum(rate(cowatcher_qa_cache_hit_total[5m]))
#        + sum(rate(cowatcher_qa_cache_miss_total[5m])))
# Split sources: cowatcher_qa_cache_hit_total{source="exact|semantic"}


LEGACY_TOOL_PATH_TOTAL = Counter(
    "cowatcher_legacy_tool_path_total",
    "Full multi-tool LLM agent path invocations (expensive; should be rare in pilot)",
    labelnames=("persona_id",),
)

LLM_PROMPT_TOKENS_TOTAL = Counter(
    "cowatcher_llm_prompt_tokens_total",
    "Prompt tokens sent to the conversation LLM",
    labelnames=("model", "path"),
)

LLM_COMPLETION_TOKENS_TOTAL = Counter(
    "cowatcher_llm_completion_tokens_total",
    "Completion tokens from the conversation LLM",
    labelnames=("model", "path"),
)

LLM_ESTIMATED_COST_USD_TOTAL = Counter(
    "cowatcher_llm_estimated_cost_usd_total",
    "Estimated LLM cost in USD (pilot rates, not billing)",
    labelnames=("model", "path"),
)

LLM_CALLS_TOTAL = Counter(
    "cowatcher_llm_calls_total",
    "Conversation LLM invocations",
    labelnames=("model", "path"),
)

SESSION_COST_BUDGET_ALERTS_TOTAL = Counter(
    "cowatcher_session_cost_budget_alerts_total",
    "Times a user session exceeded the pilot cost budget",
)

PROMPT_TOKENS_PER_ASK = Histogram(
    "cowatcher_ask_prompt_tokens",
    "Estimated or reported prompt tokens for merged/ask path",
    buckets=(50, 100, 200, 400, 800, 1600, 3200, 6400),
)

UTTERANCE_GATE_TOTAL = Counter(
    "cowatcher_utterance_gate_total",
    "Utterance gate outcomes (heuristic free resolve vs agent fallthrough)",
    labelnames=("outcome", "action", "persona_id"),
    # outcome: free (short-circuit, no LLM) | agent (needs merged/legacy agent)
    #          | prompt_llm (legacy YES/NO gate model)
)


def observe_ask_record(record: AskRecord) -> None:
    ASK_REQUESTS_TOTAL.labels(status="success").inc()
    ASK_REQUEST_DURATION.observe(record.latency_ms / 1000.0)
    ASK_MODEL_TIER_TOTAL.labels(tier=record.model_tier).inc()
    if record.dont_know:
        ASK_DONT_KNOW_TOTAL.inc()


def observe_ask_stages(
    stages_ms: dict[str, float],
    *,
    persona_id: str | None = None,
) -> None:
    """Record per-stage durations from AskLatencyTracker."""
    pid = (persona_id or "").strip() or "unknown"
    for stage, ms in stages_ms.items():
        try:
            ASK_STAGE_DURATION.labels(
                stage=str(stage),
                persona_id=pid,
            ).observe(float(ms) / 1000.0)
        except Exception:  # noqa: BLE001
            continue


def record_ask_error() -> None:
    ASK_REQUESTS_TOTAL.labels(status="error").inc()


def record_qa_cache_result(result: str) -> None:
    """result: exact_hit | semantic_hit | miss"""
    QA_CACHE_LOOKUPS_TOTAL.labels(result=result).inc()
    if result == "exact_hit":
        QA_CACHE_HIT_TOTAL.labels(source="exact").inc()
    elif result == "semantic_hit":
        QA_CACHE_HIT_TOTAL.labels(source="semantic").inc()
    elif result == "miss":
        QA_CACHE_MISS_TOTAL.inc()


def record_legacy_tool_path(*, persona_id: str | None = None) -> None:
    LEGACY_TOOL_PATH_TOTAL.labels(persona_id=(persona_id or "").strip() or "unknown").inc()


def record_utterance_gate(
    *,
    outcome: str,
    action: str,
    persona_id: str | None = None,
) -> None:
    """outcome: free | agent | prompt_llm; action: ignore|social|content|…"""
    UTTERANCE_GATE_TOTAL.labels(
        outcome=outcome or "agent",
        action=action or "content",
        persona_id=(persona_id or "").strip() or "unknown",
    ).inc()


def record_llm_token_usage(
    *,
    model: str,
    path: str,
    prompt_tokens: int,
    completion_tokens: int,
    estimated_usd: float,
) -> None:
    m = model or "unknown"
    p = path or "other"
    LLM_CALLS_TOTAL.labels(model=m, path=p).inc()
    if prompt_tokens > 0:
        LLM_PROMPT_TOKENS_TOTAL.labels(model=m, path=p).inc(prompt_tokens)
        PROMPT_TOKENS_PER_ASK.observe(prompt_tokens)
    if completion_tokens > 0:
        LLM_COMPLETION_TOKENS_TOTAL.labels(model=m, path=p).inc(completion_tokens)
    if estimated_usd > 0:
        LLM_ESTIMATED_COST_USD_TOTAL.labels(model=m, path=p).inc(estimated_usd)


def record_session_budget_alert() -> None:
    SESSION_COST_BUDGET_ALERTS_TOTAL.inc()


@contextmanager
def observe_tool_call(tool: str) -> Iterator[None]:
    started = time.perf_counter()
    outcome = "success"
    try:
        yield
    except Exception:
        outcome = "error"
        raise
    finally:
        TOOL_CALL_DURATION.labels(tool=tool).observe(time.perf_counter() - started)
        TOOL_CALLS_TOTAL.labels(tool=tool, outcome=outcome).inc()


@contextmanager
def observe_storage_query(backend: str, operation: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        STORAGE_QUERY_DURATION.labels(backend=backend, operation=operation).observe(
            time.perf_counter() - started
        )


def record_ingest_job(*, status: str, duration_sec: float, scenes_processed: int = 0) -> None:
    INGEST_JOBS_TOTAL.labels(status=status).inc()
    if status == "completed":
        INGEST_JOB_DURATION.observe(duration_sec)
    if scenes_processed > 0:
        INGEST_SCENES_PROCESSED_TOTAL.inc(scenes_processed)


def set_ingest_queue_depth(broker: str, depth: int | None) -> None:
    if depth is None:
        return
    INGEST_QUEUE_DEPTH.labels(broker=broker).set(depth)
