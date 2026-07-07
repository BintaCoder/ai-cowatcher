"""Ingestion API routes — publish events to the message broker, never run inline."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ai_cowatcher.api.schemas import IngestRequest, IngestResponse
from ai_cowatcher.config import get_settings
from ai_cowatcher.ingestion.catalog import enqueue_title_ingestion

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
async def enqueue_ingestion(request: IngestRequest) -> IngestResponse:
    settings = get_settings()
    event = enqueue_title_ingestion(
        request.title_id,
        request.video_path,
        force=request.force,
        settings=settings,
    )
    return IngestResponse(
        status="queued",
        title_id=request.title_id,
        message=(
            f"Offline ingestion event {event.event_id} published to "
            f"{settings.message_broker}. Scene captioning runs only in the ingest worker."
        ),
    )
