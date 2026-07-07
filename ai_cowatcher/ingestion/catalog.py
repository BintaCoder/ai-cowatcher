"""Publish catalog titles to the ingest message broker."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session, sessionmaker

from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.db.base import create_db_engine, init_database
from ai_cowatcher.messaging.base import IngestEventProducer
from ai_cowatcher.messaging.events import IngestTitleEvent
from ai_cowatcher.messaging.publisher import get_ingest_producer
from ai_cowatcher.storage.postgres_store import SceneEventRepository

logger = logging.getLogger(__name__)


def enqueue_title_ingestion(
    title_id: str,
    video_path: str,
    *,
    force: bool = False,
    display_name: str | None = None,
    settings: Settings | None = None,
    producer: IngestEventProducer | None = None,
    session_factory: sessionmaker | None = None,
) -> IngestTitleEvent:
    """Register a title as queued and publish an ingest event for the worker."""
    settings = settings or get_settings()
    if session_factory is None:
        engine = create_db_engine(settings=settings)
        init_database(engine=engine, settings=settings)
        session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    with session_factory() as session:
        _register_queued(session, title_id, video_path, display_name=display_name)

    event = IngestTitleEvent(
        title_id=title_id,
        video_path=video_path,
        force=force,
        display_name=display_name,
    )
    publish = producer or get_ingest_producer(settings)
    publish.publish(event)
    logger.info(
        "Queued ingest event %s for title %s via broker=%s",
        event.event_id,
        title_id,
        settings.message_broker,
    )
    return event


def _register_queued(
    session: Session,
    title_id: str,
    video_path: str,
    *,
    display_name: str | None,
) -> None:
    repo = SceneEventRepository(session)
    repo.register_queued(title_id, video_path, display_name=display_name)
