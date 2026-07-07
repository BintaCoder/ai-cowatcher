"""Watch page and video streaming for the co-watcher pilot UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker

from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.db.base import create_db_engine, init_database
from ai_cowatcher.storage.postgres_store import SceneEventRepository
from ai_cowatcher.web.streaming import (
    async_iter_file_range,
    guess_video_media_type,
    parse_range_header,
)

router = APIRouter(tags=["watch"])

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_WATCH_HTML = _WEB_DIR / "watch.html"

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


class TitleListItem(BaseModel):
    title_id: str
    display_name: str | None
    scene_count: int


@router.get("/watch", response_class=HTMLResponse)
async def watch_page() -> HTMLResponse:
    if not _WATCH_HTML.is_file():
        raise HTTPException(status_code=500, detail="Watch page asset missing")
    return HTMLResponse(_WATCH_HTML.read_text(encoding="utf-8"))


@router.get("/titles", response_model=list[TitleListItem])
async def list_titles(session: Session = Depends(get_db_session)) -> list[TitleListItem]:
    repo = SceneEventRepository(session)
    titles = repo.list_completed_titles()
    return [
        TitleListItem(
            title_id=title.title_id,
            display_name=title.display_name,
            scene_count=title.scene_count,
        )
        for title in titles
    ]


def _resolve_video_path(title_id: str, session: Session) -> Path:
    repo = SceneEventRepository(session)
    title = repo.get_title(title_id)
    if title is None or title.status != "completed":
        raise HTTPException(status_code=404, detail=f"Title not found or not ready: {title_id}")

    path = Path(title.video_path).expanduser()
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Video file missing on disk for title {title_id}",
        )
    return path


@router.get("/video/{title_id}")
async def stream_video(
    title_id: str,
    request: Request,
    session: Session = Depends(get_db_session),
):
    path = _resolve_video_path(title_id, session)
    file_size = path.stat().st_size
    media_type = guess_video_media_type(path)
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes"},
        )

    try:
        start, end = parse_range_header(range_header, file_size)
    except ValueError as exc:
        raise HTTPException(status_code=416, detail=str(exc)) from exc

    content_length = end - start + 1
    return StreamingResponse(
        async_iter_file_range(path, start, end),
        status_code=206,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(content_length),
        },
    )
