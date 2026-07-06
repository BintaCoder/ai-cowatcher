"""Ingestion API routes — enqueue offline jobs, never run vision captioning inline."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks

from ai_cowatcher.api.schemas import IngestRequest, IngestResponse
from ai_cowatcher.ingestion.pipeline import run_ingestion

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


def _run_ingestion_job(title_id: str, video_path: str, force: bool) -> None:
    try:
        result = run_ingestion(title_id, video_path, force=force)
        logger.info(
            "Background ingestion finished for %s (%s scenes, skipped=%s)",
            result.title_id,
            result.scene_count,
            result.skipped,
        )
    except Exception:
        logger.exception("Background ingestion failed for %s", title_id)


@router.post("", response_model=IngestResponse)
async def enqueue_ingestion(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    background_tasks.add_task(
        _run_ingestion_job,
        request.title_id,
        request.video_path,
        request.force,
    )
    return IngestResponse(
        status="queued",
        title_id=request.title_id,
        message="Offline ingestion job queued. Scene captioning runs only in the batch worker.",
    )
