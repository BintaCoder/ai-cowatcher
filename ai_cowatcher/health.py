"""Dependency health probes for the /health endpoint."""

from __future__ import annotations

import time
from typing import Any

import psycopg2
import redis
from qdrant_client import QdrantClient

from ai_cowatcher.config import Settings


def _probe(name: str, fn) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        detail = fn()
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return {"status": "ok", "latency_ms": latency_ms, "detail": detail}
    except Exception as exc:  # noqa: BLE001 — health endpoint must surface all failures
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return {"status": "error", "latency_ms": latency_ms, "detail": str(exc)}


def check_postgres(settings: Settings) -> dict[str, Any]:
    def _ping() -> str:
        with psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            dbname=settings.postgres_db,
            connect_timeout=3,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return "connected"

    return _probe("postgres", _ping)


def check_redis(settings: Settings) -> dict[str, Any]:
    def _ping() -> str:
        client = redis.Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=3,
            socket_timeout=3,
            decode_responses=True,
        )
        try:
            pong = client.ping()
            if pong is not True:
                raise RuntimeError(f"unexpected PING response: {pong!r}")
            return "connected"
        finally:
            client.close()

    return _probe("redis", _ping)


def check_qdrant(settings: Settings) -> dict[str, Any]:
    def _ping() -> str:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=3,
        )
        collections = client.get_collections()
        return f"{len(collections.collections)} collection(s)"

    return _probe("qdrant", _ping)


def collect_dependency_health(settings: Settings) -> dict[str, dict[str, Any]]:
    return {
        "postgres": check_postgres(settings),
        "redis": check_redis(settings),
        "qdrant": check_qdrant(settings),
    }


def overall_status(dependencies: dict[str, dict[str, Any]]) -> str:
    if all(dep.get("status") == "ok" for dep in dependencies.values()):
        return "ok"
    return "degraded"
