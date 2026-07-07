"""Tests for event-driven ingestion messaging and worker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import SceneEventRecord
from ai_cowatcher.ingestion.catalog import enqueue_title_ingestion
from ai_cowatcher.ingestion.pipeline import IngestionPipeline
from ai_cowatcher.messaging.consumer_worker import IngestConsumerWorker
from ai_cowatcher.messaging.events import IngestTitleEvent
from ai_cowatcher.messaging.memory import (
    InMemoryIngestConsumer,
    InMemoryIngestProducer,
    reset_shared_queue,
)
from ai_cowatcher.messaging.publisher import reset_ingest_producer
from ai_cowatcher.providers.factory import build_ingestion_providers
from ai_cowatcher.storage.postgres_store import SceneEventRepository
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore


@pytest.fixture
def messaging_settings() -> Settings:
    return Settings(MOCK_MODE=True, MESSAGE_BROKER="memory")


@pytest.fixture
def sqlite_session_factory():
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    from ai_cowatcher.db.base import Base
    from ai_cowatcher.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@pytest.fixture(autouse=True)
def reset_messaging_state():
    reset_shared_queue()
    reset_ingest_producer()
    yield
    reset_shared_queue()
    reset_ingest_producer()


def test_memory_producer_consumer_roundtrip():
    producer = InMemoryIngestProducer()
    consumer = InMemoryIngestConsumer()
    event = IngestTitleEvent(title_id="demo", video_path="/videos/demo.mp4")
    producer.publish(event)

    message = consumer.poll(timeout_sec=0.1)
    assert message is not None
    assert message.event.title_id == "demo"
    assert message.event.video_path == "/videos/demo.mp4"
    consumer.ack(message)


def test_enqueue_title_registers_queued_and_publishes(
    messaging_settings: Settings,
    sqlite_session_factory,
):
    producer = InMemoryIngestProducer()
    event = enqueue_title_ingestion(
        "catalog-title",
        "/data/catalog-title.mp4",
        display_name="Catalog Title",
        settings=messaging_settings,
        producer=producer,
        session_factory=sqlite_session_factory,
    )

    with sqlite_session_factory() as session:
        repo = SceneEventRepository(session)
        title = repo.get_title("catalog-title")
        assert title is not None
        assert title.status == "queued"
        assert title.display_name == "Catalog Title"

    consumer = InMemoryIngestConsumer()
    message = consumer.poll(timeout_sec=0.1)
    assert message is not None
    assert message.event.event_id == event.event_id


def test_consumer_skips_completed_title(
    messaging_settings: Settings,
    sqlite_session_factory,
):
    with sqlite_session_factory() as session:
        repo = SceneEventRepository(session)
        repo.mark_processing("done-title", "/videos/done.mp4")
        repo.mark_completed("done-title", scene_count=3)

    run_ingestion = MagicMock()
    consumer = InMemoryIngestConsumer()
    worker = IngestConsumerWorker(
        messaging_settings,
        consumer,
        run_ingestion_fn=run_ingestion,
        session_factory=sqlite_session_factory,
    )

    event = IngestTitleEvent(title_id="done-title", video_path="/videos/done.mp4")
    consumer._queue.put(event)  # noqa: SLF001 - test helper
    worker.run_once(timeout_sec=0.1)

    run_ingestion.assert_not_called()


def test_consumer_nacks_failed_job_for_retry(
    messaging_settings: Settings,
    sqlite_session_factory,
):
    run_ingestion = MagicMock(side_effect=RuntimeError("worker pod died"))
    consumer = InMemoryIngestConsumer()
    worker = IngestConsumerWorker(
        messaging_settings,
        consumer,
        run_ingestion_fn=run_ingestion,
        session_factory=sqlite_session_factory,
    )

    event = IngestTitleEvent(title_id="retry-title", video_path="/videos/retry.mp4")
    consumer._queue.put(event)  # noqa: SLF001
    worker.run_once(timeout_sec=0.1)

    retry_message = consumer.poll(timeout_sec=0.1)
    assert retry_message is not None
    assert retry_message.event.title_id == "retry-title"
    assert retry_message.event.attempt == 2


def test_worker_resumes_partial_ingest_without_duplicating_scenes(
    tmp_path: Path,
    messaging_settings: Settings,
    sqlite_session_factory,
):
    from qdrant_client import QdrantClient

    video_path = tmp_path / "resume.mp4"
    video_path.write_bytes(b"placeholder")

    with sqlite_session_factory() as session:
        repo = SceneEventRepository(session)
        repo.mark_processing("resume-title", str(video_path))
        repo.save_scene_event(
            SceneEventRecord(
                scene_id="s0000",
                title_id="resume-title",
                start_ts=0.0,
                end_ts=12.5,
                transcript="opening",
                caption="opening scene",
                face_cluster_ids=[],
                speaker_cluster_ids=[],
            )
        )

    qdrant_store = QdrantSceneStore(
        messaging_settings,
        client=QdrantClient(":memory:"),
    )
    pipeline = IngestionPipeline(
        settings=messaging_settings,
        providers=build_ingestion_providers(messaging_settings),
        session_factory=sqlite_session_factory,
        qdrant_store=qdrant_store,
    )

    consumer = InMemoryIngestConsumer()
    worker = IngestConsumerWorker(
        messaging_settings,
        consumer,
        run_ingestion_fn=pipeline.run,
        session_factory=sqlite_session_factory,
    )

    event = IngestTitleEvent(title_id="resume-title", video_path=str(video_path))
    consumer._queue.put(event)  # noqa: SLF001
    worker.run_once(timeout_sec=0.1)

    with sqlite_session_factory() as session:
        repo = SceneEventRepository(session)
        assert repo.is_completed("resume-title") is True
        assert repo.count_scene_events("resume-title") == 3
