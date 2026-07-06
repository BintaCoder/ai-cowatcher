"""Tests for the offline ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from ai_cowatcher.config import Settings
from ai_cowatcher.ingestion.pipeline import IngestionPipeline
from ai_cowatcher.providers.factory import build_ingestion_providers
from ai_cowatcher.storage.postgres_store import SceneEventRepository
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        MOCK_MODE=True,
        QDRANT_COLLECTION="test_title_segments",
    )


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


@pytest.fixture
def qdrant_store(test_settings: Settings) -> QdrantSceneStore:
    client = QdrantClient(":memory:")
    return QdrantSceneStore(test_settings, client=client)


def test_ingestion_writes_scene_events_to_postgres_and_qdrant(
    tmp_path: Path,
    test_settings: Settings,
    sqlite_session_factory,
    qdrant_store: QdrantSceneStore,
):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"not-a-real-video-but-mock-detector-does-not-read-it")

    pipeline = IngestionPipeline(
        settings=test_settings,
        providers=build_ingestion_providers(test_settings),
        session_factory=sqlite_session_factory,
        qdrant_store=qdrant_store,
    )

    result = pipeline.run("pilot-title-001", str(video_path))
    assert result.skipped is False
    assert result.scene_count == 3

    with sqlite_session_factory() as session:
        repo = SceneEventRepository(session)
        assert repo.count_scene_events("pilot-title-001") == 3
        assert repo.is_completed("pilot-title-001") is True

    assert qdrant_store.count_title_scenes("pilot-title-001") == 3


def test_ingestion_is_idempotent_without_force(
    tmp_path: Path,
    test_settings: Settings,
    sqlite_session_factory,
    qdrant_store: QdrantSceneStore,
):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"placeholder")

    pipeline = IngestionPipeline(
        settings=test_settings,
        providers=build_ingestion_providers(test_settings),
        session_factory=sqlite_session_factory,
        qdrant_store=qdrant_store,
    )

    first = pipeline.run("pilot-title-002", str(video_path))
    second = pipeline.run("pilot-title-002", str(video_path))

    assert first.scene_count == 3
    assert second.skipped is True
    assert second.scene_count == 3

    with sqlite_session_factory() as session:
        repo = SceneEventRepository(session)
        assert repo.count_scene_events("pilot-title-002") == 3

    assert qdrant_store.count_title_scenes("pilot-title-002") == 3
