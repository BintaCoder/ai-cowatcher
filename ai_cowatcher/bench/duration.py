"""Resolve title video duration for playhead sampling."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ai_cowatcher.db.models import SceneEvent, TitleIngestion
from ai_cowatcher.web.streaming import resolve_video_file_path

logger = logging.getLogger(__name__)

# Pilot default: human-facing title name (watch UI catalog).
DEFAULT_TITLE_NAME = "Friends Ross"


def resolve_title_id(session: Session | None, title_ref: str) -> str:
    """Map a title id or display name to the ingested ``title_id``.

    Order: exact ``title_id`` → case-insensitive ``display_name`` → snake_case slug.
    """
    ref = (title_ref or "").strip()
    if not ref:
        raise ValueError("title id / name is empty")

    if session is None:
        return ref

    by_id = session.get(TitleIngestion, ref)
    if by_id is not None:
        return by_id.title_id

    row = session.scalar(
        select(TitleIngestion).where(
            func.lower(TitleIngestion.display_name) == ref.lower()
        )
    )
    if row is not None:
        logger.info("Resolved title name %r → title_id=%r", ref, row.title_id)
        return row.title_id

    slug_guess = ref.lower().replace(" ", "_")
    by_slug = session.get(TitleIngestion, slug_guess)
    if by_slug is not None:
        logger.info("Resolved title %r → title_id=%r (slug)", ref, by_slug.title_id)
        return by_slug.title_id

    raise ValueError(
        f"No ingested title matching id or display name {ref!r}. "
        "Check title_ingestions (title_id / display_name) or pass --title-id "
        "with the exact ingest id (e.g. friends_ross)."
    )


def duration_from_scene_max(session: Session, title_id: str) -> float | None:
    """Use max scene end_ts as an approximate video duration."""
    value = session.scalar(
        select(func.max(SceneEvent.end_ts)).where(SceneEvent.title_id == title_id)
    )
    if value is None:
        return None
    try:
        d = float(value)
    except (TypeError, ValueError):
        return None
    return d if d > 0 else None


def video_path_for_title(session: Session, title_id: str) -> Path | None:
    title = session.get(TitleIngestion, title_id)
    if title is None or not title.video_path:
        return None
    return resolve_video_file_path(title.video_path)


def duration_from_ffprobe(video_path: Path) -> float | None:
    """ffprobe -show_entries format=duration (requires ffprobe on PATH)."""
    if not video_path.is_file():
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("ffprobe failed for %s: %s", video_path, exc)
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout or "{}")
        duration = float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return duration if duration > 0 else None


def resolve_duration(
    session: Session | None,
    title_id: str,
    *,
    override_sec: float | None = None,
) -> float:
    """Resolve duration: override → ffprobe video path → scene max end_ts.

    Raises ValueError if nothing works.
    """
    if override_sec is not None and override_sec > 0:
        return float(override_sec)

    path: Path | None = None
    if session is not None:
        path = video_path_for_title(session, title_id)
        if path is not None:
            probed = duration_from_ffprobe(path)
            if probed is not None:
                return probed
        scene_dur = duration_from_scene_max(session, title_id)
        if scene_dur is not None:
            return scene_dur

    raise ValueError(
        f"Could not resolve duration for title_id={title_id!r}. "
        "Pass --duration-sec, ensure the title is ingested with a local video, "
        "or install ffprobe."
    )
