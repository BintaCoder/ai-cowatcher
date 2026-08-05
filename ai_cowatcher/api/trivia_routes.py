"""Proactive trivia tick — cheap playhead check, no Gemini on the hot path."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker

from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.db.base import create_db_engine, init_database
from ai_cowatcher.personas.loader import resolve_persona_id
from ai_cowatcher.trivia import (
    TriviaPacingState,
    TriviaStore,
    flavor_trivia,
    pacing_allows,
)

router = APIRouter(tags=["trivia"])

_session_factory: sessionmaker | None = None


def _get_session_factory(settings: Settings | None = None) -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        settings = settings or get_settings()
        engine = create_db_engine(settings=settings)
        init_database(engine=engine, settings=settings)
        _session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return _session_factory


def get_db_session(settings: Settings = Depends(get_settings)) -> Session:
    factory = _get_session_factory(settings)
    session = factory()
    try:
        yield session
    finally:
        session.close()


class TriviaTickRequest(BaseModel):
    title_id: str = Field(..., min_length=1, max_length=128)
    current_ts: float = Field(..., ge=0.0)
    # Hard gate: client must opt in. Off by default on the UI.
    enabled: bool = False
    persona_id: str | None = Field(default=None, max_length=64)
    surfaced_ids: list[str] = Field(default_factory=list)
    last_surfaced_ts: float | None = None
    last_ask_ts: float | None = None
    count: int = 0


@router.post("/trivia/tick")
async def trivia_tick(
    body: TriviaTickRequest,
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_db_session),
) -> dict:
    """
    Sparse, opt-in trivia push. Returns {trivia: null} or a flavored fact.
    Never calls Gemini — facts are precomputed at ingest.
    """
    if not body.enabled:
        return {"trivia": None, "reason": "opt_out"}

    state = TriviaPacingState(
        surfaced_ids=list(body.surfaced_ids or []),
        last_surfaced_ts=body.last_surfaced_ts,
        last_ask_ts=body.last_ask_ts,
        count=int(body.count or 0),
    )
    if not pacing_allows(
        state,
        current_ts=body.current_ts,
        min_gap_sec=float(settings.trivia_min_gap_sec),
        max_per_session=int(settings.trivia_max_per_session),
        ask_cooldown_sec=float(settings.trivia_ask_cooldown_sec),
    ):
        return {"trivia": None, "reason": "pacing"}

    store = TriviaStore(session)
    fact = store.eligible_near_playhead(
        title_id=body.title_id,
        current_ts=body.current_ts,
        exclude_ids=set(state.surfaced_ids),
    )
    if fact is None:
        return {"trivia": None, "reason": "none_near_playhead"}

    persona_id = resolve_persona_id(
        body.persona_id,
        default_id=getattr(settings, "default_persona_id", None) or "easygoing_friend",
    )
    message = flavor_trivia(persona_id, fact.fact_text)
    payload = fact.to_dict()
    payload["message"] = message
    return {"trivia": payload, "reason": "ok"}
