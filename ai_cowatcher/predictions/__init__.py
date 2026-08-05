"""Prediction Mode — capture speculative guesses; resolve only after reveal_ts."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_cowatcher.db.models import Prediction, TitleEvent, TitleIngestion

# Hard ceiling used when no credits/reveal metadata exists (end-of-title client flag).
FALLBACK_REVEAL_TS = 1.0e9

_PREDICTION_HINT = re.compile(
    r"\b("
    r"i bet|i think|i reckon|i guess|my guess|predict|prediction|"
    r"who do you think|will they|do you think|going to be the|"
    r"must be|has to be|probably|gonna be"
    r")\b",
    re.IGNORECASE,
)

_QUESTION_HINT = re.compile(
    r"\b(who is|what is|what just|what happened|where is|when does|"
    r"how did|why did|explain|tell me about)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    session_id: str
    title_id: str
    user_id: str
    viewer_label: str
    question_prompt: str
    guess_text: str
    character_id: str | None
    topic_tags: tuple[str, ...]
    made_at_ts: float
    reveal_ts: float
    resolved: bool
    correct: bool | None
    resolution_text: str | None

    def to_dict(self) -> dict:
        return {
            "prediction_id": self.prediction_id,
            "session_id": self.session_id,
            "title_id": self.title_id,
            "user_id": self.user_id,
            "viewer_label": self.viewer_label,
            "question_prompt": self.question_prompt,
            "guess_text": self.guess_text,
            "character_id": self.character_id,
            "topic_tags": list(self.topic_tags),
            "made_at_ts": self.made_at_ts,
            "reveal_ts": self.reveal_ts,
            "resolved": self.resolved,
            "correct": self.correct,
            "resolution_text": self.resolution_text,
        }


def looks_like_prediction(question: str) -> bool:
    """Heuristic for mock LLM + classification tests (not the sole gate)."""
    q = (question or "").strip()
    if not q:
        return False
    if _PREDICTION_HINT.search(q):
        # Prefer CONTENT when clearly asking for facts now.
        if _QUESTION_HINT.search(q) and not re.search(
            r"\b(i bet|i think|i reckon|my guess|predict)\b", q, re.I
        ):
            return False
        return True
    return False


def extract_guess_text(question: str) -> str:
    text = (question or "").strip()
    # Light cleanup of lead-ins; keep free text for the pilot.
    text = re.sub(
        r"^(i bet|i think|i reckon|i guess|my guess is|predict that)\s+",
        "",
        text,
        flags=re.I,
    ).strip()
    return text or (question or "").strip()


def infer_topic_tags(guess_text: str) -> list[str]:
    lower = (guess_text or "").lower()
    tags: list[str] = []
    if any(w in lower for w in ("kill", "murder", "did it", "culprit", "suspect")):
        tags.append("killer_reveal")
    if any(w in lower for w in ("together", "couple", "marry", "dating", "love")):
        tags.append("romance_reveal")
    if not tags:
        tags.append("general")
    return tags


def persona_prediction_ack(persona_id: str | None, guess_text: str) -> str:
    """Template acknowledgment — no plot confirmation (no extra LLM call)."""
    short = (guess_text or "that").strip()
    if len(short) > 48:
        short = short[:45] + "…"
    pid = (persona_id or "").lower()
    if "witty" in pid:
        return f"Locking it in — you called '{short}'. We'll see…"
    if "calm" in pid:
        return f"Noted: '{short}'. I'll hold that until it's fair to check."
    return f"Got it — logged '{short}'. No spoilers until later."


def persona_prediction_reveal(
    persona_id: str | None,
    *,
    guess_text: str,
    correct: bool | None,
    resolution_text: str | None,
) -> str:
    pid = (persona_id or "").lower()
    guess = (guess_text or "your guess").strip()
    if len(guess) > 40:
        guess = guess[:37] + "…"
    if correct is True:
        if "witty" in pid:
            return f"Called it? You said '{guess}' — that tracks."
        if "calm" in pid:
            return f"Your guess '{guess}' looks right from here."
        return f"Nice — '{guess}' checks out."
    if correct is False:
        if "witty" in pid:
            return f"Ooh, close but '{guess}' wasn't it. {resolution_text or ''}".strip()
        return f"Not quite — '{guess}' missed. {resolution_text or ''}".strip()
    # Unresolved content / open reveal without ground truth
    return resolution_text or f"Time to revisit your guess: '{guess}'."


class PredictionStore:
    """Postgres-backed predictions with server-side reveal_ts enforcement."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def resolve_reveal_ts(
        self,
        *,
        title_id: str,
        made_at_ts: float,
        topic_tags: list[str] | None = None,
    ) -> float:
        """Pick earliest eligible reveal_point after made_at_ts; else credits; else fallback."""
        tags = set(topic_tags or [])
        events = list(
            self._session.scalars(
                select(TitleEvent)
                .where(TitleEvent.title_id == title_id)
                .where(TitleEvent.event_type.in_(("reveal_point", "credits")))
                .order_by(TitleEvent.start_ts.asc())
            )
        )
        for ev in events:
            if ev.start_ts < made_at_ts:
                continue
            if ev.event_type == "credits":
                return float(ev.start_ts)
            meta = ev.event_metadata or {}
            topics = meta.get("prediction_topics") or meta.get("topics") or []
            if isinstance(topics, str):
                topics = [topics]
            topic_set = {str(t) for t in topics}
            if not tags or not topic_set or tags & topic_set or "general" in tags:
                return float(ev.start_ts)

        title = self._session.get(TitleIngestion, title_id)
        if title and title.credits_start_ts is not None:
            credits = float(title.credits_start_ts)
            if credits >= made_at_ts:
                return credits
        return FALLBACK_REVEAL_TS

    def create_prediction(
        self,
        *,
        title_id: str,
        user_id: str,
        session_id: str,
        guess_text: str,
        made_at_ts: float,
        question_prompt: str = "",
        viewer_label: str = "you",
        character_id: str | None = None,
        topic_tags: list[str] | None = None,
        reveal_ts: float | None = None,
    ) -> PredictionRecord:
        tags = topic_tags or infer_topic_tags(guess_text)
        reveal = (
            float(reveal_ts)
            if reveal_ts is not None
            else self.resolve_reveal_ts(
                title_id=title_id, made_at_ts=made_at_ts, topic_tags=tags
            )
        )
        # Never allow reveal before the guess was made.
        reveal = max(reveal, float(made_at_ts))
        row = Prediction(
            prediction_id=f"pred_{uuid.uuid4().hex[:16]}",
            session_id=session_id,
            title_id=title_id,
            user_id=user_id,
            viewer_label=viewer_label or "you",
            question_prompt=question_prompt or "",
            guess_text=guess_text,
            character_id=character_id,
            topic_tags=tags,
            made_at_ts=float(made_at_ts),
            reveal_ts=reveal,
            resolved=False,
            correct=None,
            resolution_text=None,
        )
        self._session.add(row)
        self._session.commit()
        self._session.refresh(row)
        return _to_record(row)

    def list_for_session(
        self,
        *,
        title_id: str,
        user_id: str,
        session_id: str | None = None,
    ) -> list[PredictionRecord]:
        stmt = (
            select(Prediction)
            .where(Prediction.title_id == title_id)
            .where(Prediction.user_id == user_id)
            .order_by(Prediction.created_at.asc())
        )
        if session_id:
            stmt = stmt.where(Prediction.session_id == session_id)
        return [_to_record(r) for r in self._session.scalars(stmt)]

    def pending_reveals(
        self,
        *,
        title_id: str,
        user_id: str,
        current_ts: float,
        title_ended: bool = False,
        session_id: str | None = None,
    ) -> list[PredictionRecord]:
        """Unresolved predictions whose reveal_ts has been reached (or title ended)."""
        rows = self.list_for_session(
            title_id=title_id, user_id=user_id, session_id=session_id
        )
        out: list[PredictionRecord] = []
        for rec in rows:
            if rec.resolved:
                continue
            if title_ended or current_ts >= rec.reveal_ts:
                out.append(rec)
        return out

    def try_resolve(
        self,
        *,
        prediction_id: str,
        current_ts: float,
        title_ended: bool = False,
        correct: bool | None = None,
        resolution_text: str | None = None,
        force: bool = False,
    ) -> PredictionRecord | None:
        """
        Resolve a prediction. Blocked before reveal_ts unless title_ended or force.
        `force` is for admin/tests only — production callers must not set it.
        """
        row = self._session.get(Prediction, prediction_id)
        if row is None:
            return None
        if row.resolved:
            return _to_record(row)
        if not force and not title_ended and float(current_ts) < float(row.reveal_ts):
            raise PredictionTooEarlyError(
                prediction_id=prediction_id,
                reveal_ts=float(row.reveal_ts),
                current_ts=float(current_ts),
            )
        row.resolved = True
        row.correct = correct
        row.resolution_text = resolution_text
        row.resolved_at = datetime.now(timezone.utc)
        self._session.commit()
        self._session.refresh(row)
        return _to_record(row)


class PredictionTooEarlyError(Exception):
    def __init__(self, *, prediction_id: str, reveal_ts: float, current_ts: float) -> None:
        self.prediction_id = prediction_id
        self.reveal_ts = reveal_ts
        self.current_ts = current_ts
        super().__init__(
            f"Prediction {prediction_id} cannot resolve at ts={current_ts} "
            f"before reveal_ts={reveal_ts}"
        )


def _to_record(row: Prediction) -> PredictionRecord:
    tags = row.topic_tags or []
    if not isinstance(tags, list):
        tags = list(tags) if tags else []
    return PredictionRecord(
        prediction_id=row.prediction_id,
        session_id=row.session_id,
        title_id=row.title_id,
        user_id=row.user_id,
        viewer_label=row.viewer_label,
        question_prompt=row.question_prompt or "",
        guess_text=row.guess_text,
        character_id=row.character_id,
        topic_tags=tuple(str(t) for t in tags),
        made_at_ts=float(row.made_at_ts),
        reveal_ts=float(row.reveal_ts),
        resolved=bool(row.resolved),
        correct=row.correct,
        resolution_text=row.resolution_text,
    )
