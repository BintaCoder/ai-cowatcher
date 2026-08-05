"""Proactive trivia — spoiler filter, pacing, opt-in hard gate."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_cowatcher.config import Settings
from ai_cowatcher.db.base import Base
from ai_cowatcher.db.models import SceneEvent, SceneTrivia, TitleIngestion
from ai_cowatcher.main import create_app
from ai_cowatcher.trivia import (
    TriviaPacingState,
    TriviaStore,
    flavor_trivia,
    is_plot_spoiler_fact,
    pacing_allows,
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
    "text,spoiler",
    [
        ("Shot on a sound stage in Burbank.", False),
        ("The actor also starred in a 90s sitcom.", False),
        ("Production used practical lighting for night scenes.", False),
        ("She dies in the finale after the twist.", True),
        ("It turns out he was the killer all along.", True),
        ("Plot twist: they were twins.", True),
        ("He will die before the credits.", True),
    ],
)
def test_spoiler_filter(text: str, spoiler: bool):
    assert is_plot_spoiler_fact(text) is spoiler


def test_save_candidates_rejects_spoilers(sqlite_session_factory):
    with sqlite_session_factory() as session:
        store = TriviaStore(session)
        kept = store.save_candidates(
            title_id="t1",
            scene_id="s1",
            candidates=[
                ("Filmed on Stage 12 at Warner Bros.", "filming_location", 0.8),
                ("She dies in the ending after the twist.", "production_fact", 0.9),
            ],
        )
        assert len(kept) == 1
        assert "Warner" in kept[0].fact_text
        rows = list(session.scalars(select(SceneTrivia)))
        assert len(rows) == 2
        rejected = [r for r in rows if r.rejected]
        assert len(rejected) == 1
        assert rejected[0].reject_reason == "plot_spoiler_filter"


def test_pacing_gap_and_cap():
    state = TriviaPacingState(
        surfaced_ids=["a"],
        last_surfaced_ts=100.0,
        last_ask_ts=None,
        count=1,
    )
    assert pacing_allows(
        state, current_ts=200.0, min_gap_sec=720.0, max_per_session=4
    ) is False
    assert pacing_allows(
        state, current_ts=900.0, min_gap_sec=720.0, max_per_session=4
    ) is True

    capped = TriviaPacingState(
        surfaced_ids=["a", "b", "c", "d"],
        last_surfaced_ts=0.0,
        last_ask_ts=None,
        count=4,
    )
    assert pacing_allows(
        capped, current_ts=9999.0, min_gap_sec=10.0, max_per_session=4
    ) is False


def test_ask_cooldown_blocks():
    state = TriviaPacingState.empty()
    state.last_ask_ts = 50.0
    assert pacing_allows(
        state,
        current_ts=60.0,
        min_gap_sec=10.0,
        max_per_session=4,
        ask_cooldown_sec=45.0,
    ) is False


def test_flavor_trivia_personas():
    fact = "Shot in Vancouver."
    witty = flavor_trivia("witty_friend", fact)
    calm = flavor_trivia("calm_scout", fact)
    easy = flavor_trivia("easygoing_friend", fact)
    assert "aside" in witty.lower() or "Tiny" in witty
    assert "Quiet" in calm or "note" in calm.lower()
    assert "fun fact" in easy.lower()
    assert fact in witty and fact in calm and fact in easy


def test_trivia_tick_opt_out_hard_gate(sqlite_session_factory, monkeypatch):
    settings = Settings(MOCK_MODE=True, TRIVIA_MIN_GAP_SEC=1.0, TRIVIA_MAX_PER_SESSION=4)
    title_id = f"triv-{uuid.uuid4().hex[:8]}"

    with sqlite_session_factory() as session:
        session.add(
            TitleIngestion(
                title_id=title_id,
                video_path="/tmp/x.mp4",
                status="completed",
                scene_count=1,
            )
        )
        session.add(
            SceneEvent(
                scene_id=f"{title_id}:s1",
                title_id=title_id,
                start_ts=0.0,
                end_ts=30.0,
                caption="office banter",
                transcript="hello",
                face_cluster_ids=[],
            )
        )
        session.add(
            SceneTrivia(
                trivia_id=f"triv_{uuid.uuid4().hex[:12]}",
                scene_id=f"{title_id}:s1",
                title_id=title_id,
                fact_text="Filmed on a multi-camera sitcom stage.",
                category="production_fact",
                confidence=0.8,
                flagged=False,
                rejected=False,
            )
        )
        session.commit()

    monkeypatch.setattr(
        "ai_cowatcher.api.trivia_routes._session_factory",
        sqlite_session_factory,
    )
    monkeypatch.setattr(
        "ai_cowatcher.api.trivia_routes._get_session_factory",
        lambda _settings=None: sqlite_session_factory,
    )

    client = TestClient(create_app(settings))
    off = client.post(
        "/trivia/tick",
        json={
            "title_id": title_id,
            "current_ts": 10.0,
            "enabled": False,
            "persona_id": "easygoing_friend",
        },
    )
    assert off.status_code == 200
    assert off.json()["trivia"] is None
    assert off.json()["reason"] == "opt_out"

    on = client.post(
        "/trivia/tick",
        json={
            "title_id": title_id,
            "current_ts": 10.0,
            "enabled": True,
            "persona_id": "witty_friend",
            "surfaced_ids": [],
            "count": 0,
        },
    )
    assert on.status_code == 200
    payload = on.json()
    assert payload["trivia"] is not None
    assert "message" in payload["trivia"]
    assert "aside" in payload["trivia"]["message"].lower() or "Tiny" in payload["trivia"]["message"]
