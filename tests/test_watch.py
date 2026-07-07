"""Tests for watch page routes and range streaming helpers."""

from __future__ import annotations

from pathlib import Path
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from ai_cowatcher.config import Settings
from ai_cowatcher.db.base import create_db_engine, init_database
from ai_cowatcher.db.models import TitleIngestion
from ai_cowatcher.main import create_app
from ai_cowatcher.web.streaming import parse_range_header


def test_parse_range_header_full_file():
    start, end = parse_range_header(None, 1000)
    assert (start, end) == (0, 999)


def test_parse_range_header_suffix():
    start, end = parse_range_header("bytes=500-", 1000)
    assert (start, end) == (500, 999)


def test_parse_range_header_explicit():
    start, end = parse_range_header("bytes=10-19", 1000)
    assert (start, end) == (10, 19)


@pytest.fixture
def watch_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"\x00" * 128)
    title_id = f"demo-web-{uuid.uuid4().hex[:8]}"

    settings = Settings(MOCK_MODE=True)
    engine = create_db_engine(settings=settings)
    init_database(engine=engine, settings=settings)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    with session_factory() as session:
        session.add(
            TitleIngestion(
                title_id=title_id,
                display_name="Demo Clip",
                video_path=str(video_path),
                status="completed",
                scene_count=3,
            )
        )
        session.commit()

    monkeypatch.setattr(
        "ai_cowatcher.api.watch_routes._session_factory",
        session_factory,
    )
    monkeypatch.setattr(
        "ai_cowatcher.api.watch_routes._get_session_factory",
        lambda _settings=None: session_factory,
    )

    app = create_app(settings)
    return TestClient(app), video_path, title_id


def test_watch_page_served(watch_client):
    client, _, _ = watch_client
    response = client.get("/watch")
    assert response.status_code == 200
    assert "Co-watcher" in response.text
    assert "SpeechRecognition" in response.text


def test_titles_endpoint_lists_completed(watch_client):
    client, _, title_id = watch_client
    response = client.get("/titles")
    assert response.status_code == 200
    payload = response.json()
    assert any(item["title_id"] == title_id for item in payload)
    match = next(item for item in payload if item["title_id"] == title_id)
    assert match["display_name"] == "Demo Clip"
    assert match["scene_count"] == 3


def test_video_stream_supports_range(watch_client):
    client, video_path, title_id = watch_client
    full = client.get(f"/video/{title_id}")
    assert full.status_code == 200
    assert full.content == video_path.read_bytes()

    partial = client.get(f"/video/{title_id}", headers={"Range": "bytes=0-15"})
    assert partial.status_code == 206
    assert partial.content == b"\x00" * 16
    assert partial.headers["content-range"] == f"bytes 0-15/{video_path.stat().st_size}"
