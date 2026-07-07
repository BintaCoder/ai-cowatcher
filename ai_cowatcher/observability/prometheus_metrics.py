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


def observe_ask_record(record: AskRecord) -> None:
    ASK_REQUESTS_TOTAL.labels(status="success").inc()
    ASK_REQUEST_DURATION.observe(record.latency_ms / 1000.0)
    ASK_MODEL_TIER_TOTAL.labels(tier=record.model_tier).inc()
    if record.dont_know:
        ASK_DONT_KNOW_TOTAL.inc()


def record_ask_error() -> None:
    ASK_REQUESTS_TOTAL.labels(status="error").inc()


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
