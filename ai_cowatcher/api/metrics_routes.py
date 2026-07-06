"""Pilot metrics rollup endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from ai_cowatcher.observability.ask_telemetry import metrics_lite_summary

router = APIRouter(tags=["metrics"])


@router.get("/metrics-lite")
async def metrics_lite() -> dict[str, object]:
    return metrics_lite_summary()
