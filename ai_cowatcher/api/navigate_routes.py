"""POST /navigate — jump playback to a moment in the title."""

from __future__ import annotations

from fastapi import APIRouter

from ai_cowatcher.api.schemas import NavigateRequest, NavigateResponseSchema
from ai_cowatcher.realtime.navigation_session import build_navigation_session

router = APIRouter(tags=["navigate"])


@router.post("/navigate", response_model=NavigateResponseSchema)
async def navigate(request: NavigateRequest) -> NavigateResponseSchema:
    session = build_navigation_session()
    result = session.navigate(
        title_id=request.title_id,
        question=request.question,
        current_ts=request.current_ts,
        user_id=request.user_id,
    )
    return NavigateResponseSchema(
        answer=result.answer,
        title_id=result.title_id,
        user_id=result.user_id,
        current_ts=result.current_ts,
        seek_to_ts=result.seek_to_ts,
        scene_id=result.scene_id,
        event_type=result.event_type,
        navigation_mode=result.navigation_mode,
    )
