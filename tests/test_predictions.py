"""Prediction Mode — store, reveal_ts enforcement, classification sample."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_cowatcher.config import Settings
from ai_cowatcher.db.base import Base
from ai_cowatcher.db.models import TitleEvent, TitleIngestion
from ai_cowatcher.main import create_app
from ai_cowatcher.predictions import (
    PredictionStore,
    PredictionTooEarlyError,
    looks_like_prediction,
    persona_prediction_ack,
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

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@pytest.mark.parametrize(
    "utterance,expected",
    [
        ("I bet she's lying", True),
        ("I think it's him", True),
        ("I reckon they get back together", True),
        ("my guess is the butler", True),
        ("who do you think did it?", True),
        ("will they end up together?", True),
        ("What just happened?", False),
        ("Who is that on the stairs?", False),
        ("Tell me about the detective", False),
        ("Where is the cafe?", False),
        ("joke", False),
        ("hi", False),
    ],
)
def test_looks_like_prediction_sample(utterance: str, expected: bool):
    assert looks_like_prediction(utterance) is expected


def test_prediction_classification_threshold():
    """Labeled sample accuracy for PREDICTION vs CONTENT heuristic (≥80%)."""
    labeled = [
        ("I bet she's lying", True),
        ("I think it's the neighbor", True),
        ("I reckon they'll break up", True),
        ("my guess is Reyes", True),
        ("predict the ending — he escapes", True),
        ("do you think they get married?", True),
        ("who do you think did it?", True),
        ("What just happened on screen?", False),
        ("Who is standing by the door?", False),
        ("Why did she leave the room?", False),
        ("Explain that shot", False),
        ("Where is the hotel?", False),
        ("Tell me about the cast", False),
        ("How did they meet?", False),
        ("hi there", False),
    ]
    correct = sum(
        1 for q, exp in labeled if looks_like_prediction(q) is exp
    )
    assert correct / len(labeled) >= 0.8


def test_early_resolve_blocked(sqlite_session_factory):
    with sqlite_session_factory() as session:
        session.add(
            TitleIngestion(
                title_id="pred-title",
                video_path="/tmp/x.mp4",
                status="completed",
                scene_count=1,
                credits_start_ts=3600.0,
            )
        )
        session.add(
            TitleEvent(
                event_id="pred-title:reveal_001",
                title_id="pred-title",
                event_type="reveal_point",
                ordinal=1,
                start_ts=1800.0,
                end_ts=1810.0,
                label="Killer reveal",
                event_metadata={"prediction_topics": ["killer_reveal"]},
            )
        )
        session.commit()

        store = PredictionStore(session)
        rec = store.create_prediction(
            title_id="pred-title",
            user_id="u1",
            session_id="u1:pred-title",
            guess_text="the butler",
            made_at_ts=100.0,
            topic_tags=["killer_reveal"],
        )
        assert rec.reveal_ts == 1800.0
        assert rec.resolved is False

        with pytest.raises(PredictionTooEarlyError):
            store.try_resolve(
                prediction_id=rec.prediction_id,
                current_ts=500.0,
            )

        resolved = store.try_resolve(
            prediction_id=rec.prediction_id,
            current_ts=1800.0,
            correct=None,
            resolution_text="Time to revisit.",
        )
        assert resolved is not None
        assert resolved.resolved is True


def test_persona_ack_has_no_plot_confirmation():
    ack = persona_prediction_ack("witty_friend", "she's lying")
    lower = ack.lower()
    assert "lying" in lower or "locking" in lower
    assert "wrong" not in lower
    assert "you're right" not in lower


def test_predictions_pending_route_enforces_reveal(sqlite_session_factory, monkeypatch):
    settings = Settings(MOCK_MODE=True, PREDICTION_MODE_ENABLED=True)
    title_id = f"pred-api-{uuid.uuid4().hex[:8]}"

    with sqlite_session_factory() as session:
        session.add(
            TitleIngestion(
                title_id=title_id,
                video_path="/tmp/x.mp4",
                status="completed",
                scene_count=1,
                credits_start_ts=900.0,
            )
        )
        session.commit()
        store = PredictionStore(session)
        store.create_prediction(
            title_id=title_id,
            user_id="web-viewer",
            session_id=f"web-viewer:{title_id}",
            guess_text="they reunite",
            made_at_ts=10.0,
            reveal_ts=500.0,
        )

    monkeypatch.setattr(
        "ai_cowatcher.api.prediction_routes._session_factory",
        sqlite_session_factory,
    )
    monkeypatch.setattr(
        "ai_cowatcher.api.prediction_routes._get_session_factory",
        lambda _settings=None: sqlite_session_factory,
    )

    client = TestClient(create_app(settings))
    early = client.post(
        "/predictions/pending",
        json={
            "title_id": title_id,
            "user_id": "web-viewer",
            "session_id": f"web-viewer:{title_id}",
            "current_ts": 50.0,
            "persona_id": "easygoing_friend",
        },
    )
    assert early.status_code == 200
    assert early.json()["reveals"] == []

    late = client.post(
        "/predictions/pending",
        json={
            "title_id": title_id,
            "user_id": "web-viewer",
            "session_id": f"web-viewer:{title_id}",
            "current_ts": 500.0,
            "persona_id": "easygoing_friend",
        },
    )
    assert late.status_code == 200
    reveals = late.json()["reveals"]
    assert len(reveals) == 1
    assert "message" in reveals[0]
    assert "reunite" in reveals[0]["guess_text"]
