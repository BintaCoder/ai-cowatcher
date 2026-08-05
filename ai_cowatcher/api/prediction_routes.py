"""Prediction Mode HTTP API — list guesses + spoiler-safe pending reveals."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker

from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.db.base import create_db_engine, init_database
from ai_cowatcher.predictions import (
    PredictionStore,
    PredictionTooEarlyError,
    persona_prediction_reveal,
)
from ai_cowatcher.personas.loader import resolve_persona_id

router = APIRouter(tags=["predictions"])

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


class PredictionListRequest(BaseModel):
    title_id: str = Field(..., min_length=1, max_length=128)
    user_id: str = Field(..., min_length=1, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)


class PredictionPendingRequest(BaseModel):
    title_id: str = Field(..., min_length=1, max_length=128)
    user_id: str = Field(..., min_length=1, max_length=128)
    current_ts: float = Field(..., ge=0.0)
    session_id: str | None = Field(default=None, max_length=128)
    title_ended: bool = False
    persona_id: str | None = Field(default=None, max_length=64)
    # When true, mark pending items resolved (passive reveal). Default true.
    mark_resolved: bool = True


@router.post("/predictions/list")
async def list_predictions(
    body: PredictionListRequest,
    session: Session = Depends(get_db_session),
) -> dict:
    store = PredictionStore(session)
    rows = store.list_for_session(
        title_id=body.title_id,
        user_id=body.user_id,
        session_id=body.session_id,
    )
    return {"predictions": [r.to_dict() for r in rows]}


@router.post("/predictions/pending")
async def pending_predictions(
    body: PredictionPendingRequest,
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_db_session),
) -> dict:
    """
    Return predictions safe to surface at current_ts (or title end).
    Enforces reveal_ts server-side; early resolve attempts are skipped.
    """
    if not settings.prediction_mode_enabled:
        return {"reveals": []}

    store = PredictionStore(session)
    pending = store.pending_reveals(
        title_id=body.title_id,
        user_id=body.user_id,
        current_ts=body.current_ts,
        title_ended=body.title_ended,
        session_id=body.session_id,
    )
    persona_id = resolve_persona_id(
        body.persona_id,
        default_id=getattr(settings, "default_persona_id", None) or "easygoing_friend",
    )
    reveals: list[dict] = []
    for rec in pending:
        message = persona_prediction_reveal(
            persona_id,
            guess_text=rec.guess_text,
            correct=rec.correct,
            resolution_text=rec.resolution_text
            or "Time to revisit that guess — no hard spoilers from me.",
        )
        if body.mark_resolved:
            try:
                store.try_resolve(
                    prediction_id=rec.prediction_id,
                    current_ts=body.current_ts,
                    title_ended=body.title_ended,
                    correct=rec.correct,
                    resolution_text=rec.resolution_text or message,
                )
            except PredictionTooEarlyError:
                # Race: playhead moved back — skip.
                continue
        payload = rec.to_dict()
        payload["message"] = message
        reveals.append(payload)
    return {"reveals": reveals}
