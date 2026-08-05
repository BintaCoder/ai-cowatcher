"""Integration tests for POST /navigate."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy.orm import sessionmaker

from ai_cowatcher.config import Settings
from ai_cowatcher.db.base import create_db_engine, init_database
from ai_cowatcher.db.models import SceneEvent, TitleEvent, TitleIngestion
from ai_cowatcher.domain import SceneEventRecord
from ai_cowatcher.ingestion.event_detection import build_title_events
from ai_cowatcher.main import create_app
from ai_cowatcher.providers.mock import MockTextEmbedder
from ai_cowatcher.realtime.navigation_session import NavigationSession
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore


@pytest.fixture
def navigate_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    title_id = f"nav-{uuid.uuid4().hex[:8]}"
    settings = Settings(MOCK_MODE=True, QDRANT_COLLECTION=f"test_nav_{uuid.uuid4().hex[:8]}")

    engine = create_db_engine(settings=settings)
    init_database(engine=engine, settings=settings)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    scenes = [
        SceneEventRecord("s0000", title_id, 0, 20, "opening dialogue", "living room", []),
        SceneEventRecord("s0001", title_id, 30, 50, "they fight in the yard", "two men fighting", []),
        SceneEventRecord("s0002", title_id, 80, 100, "another fight erupts", "brawl outside", []),
    ]
    events, credits_ts = build_title_events(title_id, scenes)

    with session_factory() as session:
        session.add(
            TitleIngestion(
                title_id=title_id,
                display_name="Nav Demo",
                video_path=str(tmp_path / "clip.mp4"),
                status="completed",
                scene_count=len(scenes),
                credits_start_ts=credits_ts,
            )
        )
        for scene in scenes:
            session.add(
                SceneEvent(
                    scene_id=f"{title_id}:{scene.scene_id}",
                    title_id=title_id,
                    start_ts=scene.start_ts,
                    end_ts=scene.end_ts,
                    transcript=scene.transcript,
                    caption=scene.caption,
                    face_cluster_ids=scene.face_cluster_ids,
                )
            )
        for event in events:
            session.add(
                TitleEvent(
                    event_id=event.event_id,
                    title_id=event.title_id,
                    event_type=event.event_type,
                    ordinal=event.ordinal,
                    start_ts=event.start_ts,
                    end_ts=event.end_ts,
                    scene_id=event.scene_id,
                    label=event.label,
                    event_metadata=event.metadata,
                )
            )
        session.commit()

    embedder = MockTextEmbedder()
    qdrant = QdrantSceneStore(settings, client=QdrantClient(":memory:"))
    qdrant.ensure_collection(embedder.vector_size)
    vectors = embedder.embed_texts([scene.embedding_text for scene in scenes])
    qdrant.upsert_scene_events(scenes, vectors)

    nav_session = NavigationSession(
        settings=settings,
        session_factory=session_factory,
        embedder=embedder,
        qdrant_store=qdrant,
    )
    build_calls: list[int] = []

    def _warm_navigation(*_args, **_kwargs):
        build_calls.append(1)
        return nav_session

    monkeypatch.setattr(
        "ai_cowatcher.main.build_navigation_session",
        _warm_navigation,
    )
    monkeypatch.setattr(
        "ai_cowatcher.main.build_viewing_session",
        lambda *_args, **_kwargs: MagicMock(),
    )
    # Shared lifespan setup still opens a SQLAlchemy engine via create_db_engine;
    # session data lives on the same Postgres/settings as above when using real PG.
    monkeypatch.setattr(
        "ai_cowatcher.main.create_db_engine",
        lambda settings=None, **_kwargs: engine,
    )
    monkeypatch.setattr(
        "ai_cowatcher.main.init_database",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "ai_cowatcher.main._build_embedder",
        lambda _settings: embedder,
    )
    monkeypatch.setattr(
        "ai_cowatcher.main.QdrantSceneStore",
        lambda *_args, **_kwargs: qdrant,
    )

    with TestClient(create_app(settings)) as client:
        yield client, title_id, build_calls


def test_navigate_absolute_time(navigate_client):
    client, title_id, _ = navigate_client
    response = client.post(
        "/navigate",
        json={
            "title_id": title_id,
            "current_ts": 0,
            "question": "go to 1 minute 30 seconds",
            "user_id": "u1",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["seek_to_ts"] == 90.0
    assert body["navigation_mode"] == "absolute_time"


def test_navigate_second_fight(navigate_client):
    client, title_id, _ = navigate_client
    response = client.post(
        "/navigate",
        json={
            "title_id": title_id,
            "current_ts": 0,
            "question": "take me to the second fight",
            "user_id": "u1",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["seek_to_ts"] == 80.0
    assert body["event_type"] == "fight"


def test_navigate_reuses_warm_session(navigate_client):
    client, title_id, build_calls = navigate_client
    assert build_calls == [1]
    assert client.app.state.navigation_session is not None

    for _ in range(3):
        response = client.post(
            "/navigate",
            json={
                "title_id": title_id,
                "current_ts": 0,
                "question": "go to 0:30",
                "user_id": "u1",
            },
        )
        assert response.status_code == 200

    # Built once at lifespan — never per request.
    assert build_calls == [1]
