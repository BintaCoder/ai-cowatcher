"""POST /navigate — jump playback to a moment in the title."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from ai_cowatcher.api.schemas import NavigateRequest, NavigateResponseSchema
from ai_cowatcher.realtime.navigation_session import NavigationSession, build_navigation_session

router = APIRouter(tags=["navigate"])


def _get_navigation_session(request: Request) -> NavigationSession:
    session = getattr(request.app.state, "navigation_session", None)
    if session is not None:
        return session
    return build_navigation_session(getattr(request.app.state, "settings", None))


@router.post("/navigate", response_model=NavigateResponseSchema)
async def navigate(
    body: NavigateRequest,
    http_request: Request,
) -> NavigateResponseSchema:
    session = _get_navigation_session(http_request)
    # Resolver/embed/Qdrant/TMDB are sync — run off the event loop.
    result = await asyncio.to_thread(
        session.navigate,
        title_id=body.title_id,
        question=body.question,
        current_ts=body.current_ts,
        user_id=body.user_id,
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
