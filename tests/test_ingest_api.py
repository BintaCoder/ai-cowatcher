"""Tests for ingest/catalog HTTP endpoints publishing broker events."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_cowatcher.config import Settings
from ai_cowatcher.main import create_app
from ai_cowatcher.messaging.memory import InMemoryIngestConsumer, reset_shared_queue
from ai_cowatcher.messaging.publisher import reset_ingest_producer


@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    return Settings(
        MOCK_MODE=True,
        MESSAGE_BROKER="memory",
        QDRANT_COLLECTION="test_api_segments",
    )


@pytest.fixture
def sqlite_session_factory():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

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


@pytest.fixture
def client(
    api_settings: Settings,
    sqlite_session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    engine = sqlite_session_factory.kw["bind"]

    def _create_db_engine(*, settings=None, **kwargs):
        return engine

    def _init_database(**kwargs):
        return None

    monkeypatch.setattr("ai_cowatcher.ingestion.catalog.create_db_engine", _create_db_engine)
    monkeypatch.setattr("ai_cowatcher.ingestion.catalog.init_database", _init_database)
    monkeypatch.setattr("ai_cowatcher.api.routes.get_settings", lambda: api_settings)
    monkeypatch.setattr("ai_cowatcher.api.catalog_routes.get_settings", lambda: api_settings)
    return TestClient(create_app(api_settings))


def test_post_ingest_publishes_event(client: TestClient):
    response = client.post(
        "/ingest",
        json={
            "title_id": "api-title",
            "video_path": "/videos/api-title.mp4",
            "force": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["title_id"] == "api-title"

    consumer = InMemoryIngestConsumer()
    message = consumer.poll(timeout_sec=0.1)
    assert message is not None
    assert message.event.title_id == "api-title"
    assert message.event.video_path == "/videos/api-title.mp4"


def test_post_catalog_titles_publishes_event(client: TestClient):
    response = client.post(
        "/catalog/titles",
        json={
            "title_id": "catalog-api",
            "video_path": "/videos/catalog-api.mp4",
            "display_name": "Catalog API Title",
            "force": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["title_id"] == "catalog-api"
    assert body["event_id"]

    consumer = InMemoryIngestConsumer()
    message = consumer.poll(timeout_sec=0.1)
    assert message is not None
    assert message.event.display_name == "Catalog API Title"
