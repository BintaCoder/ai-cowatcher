"""Lightweight Prometheus metrics for ask-bench runs (optional scrape/push)."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

BENCH_ASK_TOTAL = Counter(
    "cowatcher_bench_ask_total",
    "Ask-bench questions executed",
    labelnames=("title_id", "question_id", "cache_source", "persona_id"),
)

BENCH_ASK_DURATION = Histogram(
    "cowatcher_bench_ask_duration_seconds",
    "Ask-bench client-measured /ask latency",
    labelnames=("title_id", "question_id", "cache_source", "persona_id"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)


def record_bench_ask(
    *,
    title_id: str,
    question_id: str,
    cache_source: str,
    latency_ms: float,
    persona_id: str = "",
) -> None:
    tid = title_id or "unknown"
    qid = question_id or "unknown"
    src = cache_source or "none"
    pid = persona_id or "default"
    BENCH_ASK_TOTAL.labels(
        title_id=tid, question_id=qid, cache_source=src, persona_id=pid
    ).inc()
    if latency_ms >= 0:
        BENCH_ASK_DURATION.labels(
            title_id=tid, question_id=qid, cache_source=src, persona_id=pid
        ).observe(float(latency_ms) / 1000.0)
