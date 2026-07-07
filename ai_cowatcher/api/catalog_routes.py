"""Catalog API — register new titles and trigger event-driven ingestion."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ai_cowatcher.api.schemas import CatalogTitleRequest, CatalogTitleResponse
from ai_cowatcher.config import get_settings
from ai_cowatcher.ingestion.catalog import enqueue_title_ingestion

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.post("/titles", response_model=CatalogTitleResponse)
async def register_title(request: CatalogTitleRequest) -> CatalogTitleResponse:
    settings = get_settings()
    event = enqueue_title_ingestion(
        request.title_id,
        request.video_path,
        force=request.force,
        display_name=request.display_name,
        settings=settings,
    )
    return CatalogTitleResponse(
        status="queued",
        title_id=request.title_id,
        event_id=event.event_id,
        message=(
            f"Title registered and ingest event published to {settings.message_broker}. "
            "The ingest worker will run scene detection, enrichment, and indexing."
        ),
    )
