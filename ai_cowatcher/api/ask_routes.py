"""Real-time co-watcher ask endpoint."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse

from ai_cowatcher.api.schemas import AskRequest, AskResponse
from ai_cowatcher.agent.stream_events import AskStreamEvent
from ai_cowatcher.observability.prometheus_metrics import record_ask_error
from ai_cowatcher.realtime.viewing_session import ViewingSession, build_viewing_session

router = APIRouter(tags=["ask"])


def _get_viewing_session(request: Request) -> ViewingSession:
    session = getattr(request.app.state, "viewing_session", None)
    if session is not None:
        return session
    return build_viewing_session(getattr(request.app.state, "settings", None))


def _sse_line(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> AskResponse:
    session = _get_viewing_session(http_request)
    try:
        # Sync agent work (LLM + tools + embed) must not block the event loop.
        result = await asyncio.to_thread(
            session.ask,
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


@router.post("/ask/stream")
async def ask_question_stream(
    request: AskRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> StreamingResponse:
    """Server-Sent Events stream: status, tool_*, token, done (or error)."""
    session = _get_viewing_session(http_request)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[AskStreamEvent | BaseException | None] = asyncio.Queue()

    def worker() -> None:
        try:
            for event in session.ask_stream(
                title_id=request.title_id,
                current_ts=request.current_ts,
                question=request.question,
                user_id=request.user_id,
                persist_memory=False,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except BaseException as exc:  # noqa: BLE001 — surface to SSE consumer
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def event_stream() -> AsyncIterator[str]:
        future = loop.run_in_executor(None, worker)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    record_ask_error()
                    yield _sse_line(
                        {
                            "type": "error",
                            "detail": str(item),
                            "message": "Ask failed",
                        }
                    )
                    break
                if item.type == "done" and item.answer:
                    background_tasks.add_task(
                        session.persist_memory,
                        user_id=request.user_id,
                        title_id=request.title_id,
                        question=request.question,
                        answer=item.answer,
                        current_ts=request.current_ts,
                    )
                yield _sse_line(item.to_dict())
        finally:
            await future

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
