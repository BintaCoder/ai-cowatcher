"""Real-time co-watcher ask endpoint."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from ai_cowatcher.api.schemas import AskRequest, AskResponse
from ai_cowatcher.config import Settings, get_settings

router = APIRouter(tags=["ask"])


def _get_viewing_session(request: Request) -> ViewingSession:
    session = getattr(request.app.state, "viewing_session", None)
    if session is not None:
        return session
    return build_viewing_session(getattr(request.app.state, "settings", None))


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> AskResponse:
    session = _get_viewing_session(http_request)
    try:
        result = session.ask(
            title_id=request.title_id,
            current_ts=request.current_ts,
            question=request.question,
            user_id=request.user_id,
            persist_memory=False,
        )
    except Exception as exc:
        record_ask_error()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    background_tasks.add_task(
        session.persist_memory,
        user_id=request.user_id,
        title_id=request.title_id,
        question=request.question,
        answer=result.answer,
        current_ts=request.current_ts,
    )

    return AskResponse(
        answer=result.answer,
        title_id=result.title_id,
        user_id=result.user_id,
        current_ts=result.current_ts,
        model_tier=result.model_tier,
        model_name=result.model_name,
        escalation_reason=result.escalation_reason,
    )
