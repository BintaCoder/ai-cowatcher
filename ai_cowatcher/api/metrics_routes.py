"""Metrics endpoints — pilot rollups and Prometheus scrape."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.observability.ask_telemetry import metrics_lite_summary

router = APIRouter(tags=["metrics"])


def _app_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


@router.get("/metrics-lite")
async def metrics_lite() -> dict[str, object]:
    return metrics_lite_summary()


@router.get("/metrics")
async def prometheus_metrics(request: Request) -> Response:
    settings = _app_settings(request)
    if not settings.prometheus_enabled:
        return Response(status_code=404, content="Prometheus metrics disabled")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
