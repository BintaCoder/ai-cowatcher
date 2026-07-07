"""Consume catalog events and run the resumable offline ingestion pipeline."""

from __future__ import annotations

import logging
import signal
from typing import Callable

from sqlalchemy.orm import sessionmaker

from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.db.base import create_db_engine, init_database
from ai_cowatcher.ingestion.pipeline import run_ingestion
from ai_cowatcher.messaging.base import BrokerMessage, IngestEventConsumer
from ai_cowatcher.messaging.events import IngestTitleEvent
from ai_cowatcher.messaging.factory import build_ingest_consumer
from ai_cowatcher.storage.postgres_store import SceneEventRepository

logger = logging.getLogger(__name__)


class IngestConsumerWorker:
    """Pull ingest events and run Phase 1 pipeline + Phase 2 enrichment."""

    def __init__(
        self,
        settings: Settings,
        consumer: IngestEventConsumer,
        *,
        run_ingestion_fn: Callable[..., object] | None = None,
        session_factory: sessionmaker | None = None,
    ) -> None:
        self._settings = settings
        self._consumer = consumer
        self._run_ingestion = run_ingestion_fn or run_ingestion
        self._session_factory = session_factory
        self._running = True

    def _get_session_factory(self) -> sessionmaker:
        if self._session_factory is not None:
            return self._session_factory
        engine = create_db_engine(settings=self._settings)
        init_database(engine=engine, settings=self._settings)
        return sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def _is_completed(self, title_id: str) -> bool:
        with self._get_session_factory()() as session:
            return SceneEventRepository(session).is_completed(title_id)

    def handle_event(self, event: IngestTitleEvent) -> None:
        if self._is_completed(event.title_id) and not event.force:
            logger.info(
                "Title %s already completed; skipping duplicate event %s",
                event.title_id,
                event.event_id,
            )
            return

        logger.info(
            "Processing ingest event %s for title %s (attempt %s)",
            event.event_id,
            event.title_id,
            event.attempt,
        )
        self._run_ingestion(
            event.title_id,
            event.video_path,
            force=event.force,
            display_name=event.display_name,
        )

    def process_message(self, message: BrokerMessage) -> None:
        self.handle_event(message.event)

    def run_once(self, *, timeout_sec: float = 1.0) -> bool:
        message = self._consumer.poll(timeout_sec=timeout_sec)
        if message is None:
            return False
        try:
            self.process_message(message)
        except Exception:
            logger.exception(
                "Ingest failed for title %s; nacking for retry",
                message.event.title_id,
            )
            self._consumer.nack(message, requeue=True)
            return True
        self._consumer.ack(message)
        return True

    def run_forever(self, *, poll_timeout_sec: float = 1.0) -> None:
        def _stop(_signum: int, _frame: object) -> None:
            logger.info("Shutdown signal received; stopping ingest worker")
            self._running = False

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        logger.info(
            "Ingest worker started (broker=%s)",
            self._settings.message_broker,
        )
        while self._running:
            self.run_once(timeout_sec=poll_timeout_sec)
        self._consumer.close()
        logger.info("Ingest worker stopped")


def run_worker(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    consumer = build_ingest_consumer(settings)
    worker = IngestConsumerWorker(settings, consumer)
    worker.run_forever()
