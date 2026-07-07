"""Real-time co-watcher ask endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ai_cowatcher.api.schemas import AskRequest, AskResponse
from ai_cowatcher.observability.prometheus_metrics import record_ask_error
from ai_cowatcher.realtime.viewing_session import build_viewing_session

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    try:
        session = build_viewing_session()
        result = session.ask(
            title_id=request.title_id,
            current_ts=request.current_ts,
            question=request.question,
            user_id=request.user_id,
        )
    except Exception as exc:
        record_ask_error()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(
        answer=result.answer,
        title_id=result.title_id,
        user_id=result.user_id,
        current_ts=result.current_ts,
        model_tier=result.model_tier,
        model_name=result.model_name,
        escalation_reason=result.escalation_reason,
    )
