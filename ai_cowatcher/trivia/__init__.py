"""Proactive trivia — ingest-time facts, opt-in sparse delivery."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_cowatcher.db.models import SceneEvent, SceneTrivia

logger = logging.getLogger(__name__)

TRIVIA_CATEGORIES = frozenset(
    {
        "filming_location",
        "cast_filmography",
        "production_fact",
        "true_story_note",
    }
)

# Plot / fate leakage — reject at ingest (not playhead cutoff alone).
_SPOILER_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\b(dies?|killed|murder|ending|finale|spoiler|twist)\b",
        r"\b(reveals? that|turns out|it was actually|secretly)\b",
        r"\b(will (?:die|leave|marry|betray)|is going to)\b",
        r"\b(plot twist|cliffhanger|season finale)\b",
    )
]


@dataclass(frozen=True)
class TriviaFact:
    trivia_id: str
    scene_id: str
    title_id: str
    fact_text: str
    category: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "trivia_id": self.trivia_id,
            "scene_id": self.scene_id,
            "title_id": self.title_id,
            "fact_text": self.fact_text,
            "category": self.category,
            "confidence": self.confidence,
        }


def is_plot_spoiler_fact(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    return any(p.search(t) for p in _SPOILER_PATTERNS)


def flavor_trivia(persona_id: str | None, fact_text: str) -> str:
    """Persona wrapper without an LLM call."""
    fact = (fact_text or "").strip()
    pid = (persona_id or "").lower()
    if "witty" in pid:
        return f"Tiny aside — {fact}"
    if "calm" in pid:
        return f"Quiet note: {fact}"
    return f"Oh, fun fact — {fact}"


def mock_trivia_candidates_for_scene(
    *,
    title_id: str,
    scene_id: str,
    caption: str,
    transcript: str,
) -> list[tuple[str, str, float]]:
    """
    Return list of (fact_text, category, confidence) for MOCK_MODE / offline demos.
    Production ingest can swap in a batched LiteLLM call with the same filter.
    """
    del transcript
    caption = (caption or "").strip()
    base = caption[:80] if caption else "this scene"
    return [
        (
            f"Production note: lighting and framing for '{base}' follow the show's usual sitcom stage style.",
            "production_fact",
            0.55,
        ),
        (
            "Cast filmography: several ensemble regulars appeared together across 1990s network comedies.",
            "cast_filmography",
            0.5,
        ),
    ]


class TriviaStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_candidates(
        self,
        *,
        title_id: str,
        scene_id: str,
        candidates: list[tuple[str, str, float]],
    ) -> list[TriviaFact]:
        saved: list[TriviaFact] = []
        for fact_text, category, confidence in candidates:
            cat = category if category in TRIVIA_CATEGORIES else "production_fact"
            rejected = is_plot_spoiler_fact(fact_text)
            reason = "plot_spoiler_filter" if rejected else None
            if rejected:
                logger.info(
                    "trivia_rejected title_id=%s scene_id=%s reason=%s text=%r",
                    title_id,
                    scene_id,
                    reason,
                    fact_text[:120],
                )
            row = SceneTrivia(
                trivia_id=f"triv_{uuid.uuid4().hex[:16]}",
                scene_id=scene_id,
                title_id=title_id,
                fact_text=fact_text,
                category=cat,
                confidence=float(confidence),
                flagged=rejected,
                rejected=rejected,
                reject_reason=reason,
            )
            self._session.add(row)
            if not rejected:
                saved.append(
                    TriviaFact(
                        trivia_id=row.trivia_id,
                        scene_id=scene_id,
                        title_id=title_id,
                        fact_text=fact_text,
                        category=cat,
                        confidence=float(confidence),
                    )
                )
        self._session.commit()
        return saved

    def eligible_near_playhead(
        self,
        *,
        title_id: str,
        current_ts: float,
        exclude_ids: set[str] | None = None,
        window_sec: float = 90.0,
    ) -> TriviaFact | None:
        exclude = exclude_ids or set()
        # Prefer trivia whose scene overlaps [current_ts - window, current_ts + window]
        scenes = list(
            self._session.scalars(
                select(SceneEvent)
                .where(SceneEvent.title_id == title_id)
                .where(SceneEvent.start_ts <= current_ts + window_sec)
                .where(SceneEvent.end_ts >= max(0.0, current_ts - window_sec))
                .order_by(SceneEvent.start_ts.asc())
            )
        )
        scene_ids = [s.scene_id for s in scenes]
        if not scene_ids:
            return None
        rows = list(
            self._session.scalars(
                select(SceneTrivia)
                .where(SceneTrivia.title_id == title_id)
                .where(SceneTrivia.scene_id.in_(scene_ids))
                .where(SceneTrivia.rejected.is_(False))
                .order_by(SceneTrivia.confidence.desc())
            )
        )
        for row in rows:
            if row.trivia_id in exclude:
                continue
            return TriviaFact(
                trivia_id=row.trivia_id,
                scene_id=row.scene_id,
                title_id=row.title_id,
                fact_text=row.fact_text,
                category=row.category,
                confidence=float(row.confidence),
            )
        return None


@dataclass
class TriviaPacingState:
    """In-memory / client-reported pacing for a watch session."""

    surfaced_ids: list[str]
    last_surfaced_ts: float | None
    last_ask_ts: float | None
    count: int

    @classmethod
    def empty(cls) -> TriviaPacingState:
        return cls(surfaced_ids=[], last_surfaced_ts=None, last_ask_ts=None, count=0)


def pacing_allows(
    state: TriviaPacingState,
    *,
    current_ts: float,
    min_gap_sec: float,
    max_per_session: int,
    ask_cooldown_sec: float = 45.0,
) -> bool:
    if state.count >= max_per_session:
        return False
    if state.last_surfaced_ts is not None:
        if abs(current_ts - state.last_surfaced_ts) < min_gap_sec:
            return False
    if state.last_ask_ts is not None:
        if abs(current_ts - state.last_ask_ts) < ask_cooldown_sec:
            return False
    return True
