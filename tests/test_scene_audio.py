"""Tests for scene audio object store and multimodal helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from ai_cowatcher.agent.multimodal import (
    build_multimodal_messages,
    collect_audio_keys,
    load_scene_clips,
)
from ai_cowatcher.config import Settings
from ai_cowatcher.ingestion.pipeline import IngestionPipeline
from ai_cowatcher.providers.factory import build_ingestion_providers
from ai_cowatcher.storage.object_store import (
    LocalFilesystemObjectStore,
    scene_audio_object_key,
)
from ai_cowatcher.storage.postgres_store import SceneEventRepository
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore


def test_collect_audio_keys_caps_and_dedupes():
    payloads = [
        [
            {"scene_id": "s1", "audio_object_key": "scenes/t/s1.wav", "transcript": "a"},
            {"scene_id": "s2", "audio_object_key": "scenes/t/s2.wav", "transcript": "b"},
            {"scene_id": "s1b", "audio_object_key": "scenes/t/s1.wav", "transcript": "c"},
        ]
    ]
    keys = collect_audio_keys(payloads, max_clips=2)
    assert keys == ["scenes/t/s1.wav", "scenes/t/s2.wav"]


def test_build_multimodal_messages_includes_input_audio():
    clips = [("scenes/t/s1.wav", b"RIFF....wav")]
    messages = build_multimodal_messages(
        question="What just happened?",
        tool_payloads=[[{"scene_id": "s1", "transcript": "hello", "caption": "room"}]],
        clips=clips,
    )
    assert messages[0]["role"] == "system"
    content = messages[1]["content"]
    assert any(part.get("type") == "input_audio" for part in content)
    assert any(part.get("type") == "text" and "What just happened" in part.get("text", "") for part in content)


def test_local_object_store_roundtrip(tmp_path: Path):
    store = LocalFilesystemObjectStore(tmp_path / "obj")
    key = scene_audio_object_key("demo", "s0001")
    store.put_bytes(key, b"abc-wav", content_type="audio/wav")
    assert store.get_bytes(key) == b"abc-wav"
    assert load_scene_clips([key], store) == [(key, b"abc-wav")]
    assert store.delete_prefix("scenes/demo/") >= 1
    assert store.get_bytes(key) is None


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


def test_ingestion_persists_scene_audio_keys(
    tmp_path: Path,
    sqlite_session_factory,
):
    objects = LocalFilesystemObjectStore(tmp_path / "objects")
    settings = Settings(
        MOCK_MODE=True,
        QDRANT_COLLECTION="test_audio_ingest",
        OBJECT_STORE_BACKEND="local",
        OBJECT_STORE_LOCAL_DIR=str(tmp_path / "objects"),
        SCENE_AUDIO_ENABLED=True,
    )
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"not-video")

    client = QdrantClient(":memory:")
    qdrant = QdrantSceneStore(settings, client=client)
    pipeline = IngestionPipeline(
        settings=settings,
        providers=build_ingestion_providers(settings),
        session_factory=sqlite_session_factory,
        qdrant_store=qdrant,
        object_store=objects,
    )
    result = pipeline.run("audio-title", str(video_path))
    assert result.scene_count == 3

    with sqlite_session_factory() as session:
        repo = SceneEventRepository(session)
        records = repo.list_scene_records("audio-title")
        assert len(records) == 3
        for rec in records:
            assert rec.audio_object_key
            assert objects.get_bytes(rec.audio_object_key)

    # Qdrant payload carries the key for retrieval
    hits = qdrant.search_scenes(
        title_id="audio-title",
        query_vector=[0.0] * 1024,
        current_ts=50.0,
        top_k=5,
    )
    # mock embedder may not rank; just ensure upsert didn't break and at least count works
    assert qdrant.count_title_scenes("audio-title") == 3
