"""Resolve navigation questions to seek timestamps."""

from __future__ import annotations

import re

from ai_cowatcher.domain import NavigateResult, SceneLookupHit, TitleEventRecord
from ai_cowatcher.navigation.ordinal import parse_ordinal
from ai_cowatcher.navigation.time_parse import parse_seek_timestamp
from ai_cowatcher.retrieval.cast_lookup import CastLookupTool
from ai_cowatcher.retrieval.event_lookup import EventLookupTool
from ai_cowatcher.retrieval.scene_navigate import SceneNavigateTool
from ai_cowatcher.storage.postgres_store import SceneEventRepository


_ACTOR_NAV_RE = re.compile(
    r"(?:where|when)\s+(?:does|did|do)\s+(.+?)\s+(?:appear|show up|come on|on screen)",
    re.I,
)


class NavigationResolver:
    """Deterministic navigation resolver (no LLM — fast and cheap)."""

    def __init__(
        self,
        repo: SceneEventRepository,
        scene_navigate: SceneNavigateTool,
        event_lookup: EventLookupTool,
        cast_lookup: CastLookupTool | None = None,
        *,
        title_display_name: str | None = None,
    ):
        self._repo = repo
        self._scene_navigate = scene_navigate
        self._event_lookup = event_lookup
        self._cast_lookup = cast_lookup
        self._title_display_name = title_display_name

    def resolve(self, *, title_id: str, question: str, current_ts: float) -> NavigateResult:
        absolute_ts = parse_seek_timestamp(question)
        if absolute_ts is not None:
            return NavigateResult(
                answer=f"Jumping to {_format_ts(absolute_ts)}.",
                seek_to_ts=absolute_ts,
                navigation_mode="absolute_time",
            )

        lower = question.lower()
        if "credit" in lower or "post-credit" in lower:
            events = self._event_lookup.lookup(title_id=title_id, question=question)
            if events:
                return _from_event(events[0], "credits")

        actor_hit = self._resolve_actor(title_id, question)
        if actor_hit is not None:
            return actor_hit

        ordinal, query = parse_ordinal(question)
        events = self._event_lookup.lookup(
            title_id=title_id,
            question=question,
            ordinal=ordinal,
        )
        if events:
            pick = events[0]
            if ordinal is not None:
                for event in events:
                    if event.ordinal == ordinal:
                        pick = event
                        break
            return _from_event(pick, pick.event_type)

        hits = self._scene_navigate.navigate(
            title_id=title_id,
            query_text=query,
            ordinal=ordinal,
            current_ts=current_ts,
        )
        if hits:
            return _from_scene(hits[0], ordinal)

        return NavigateResult(
            answer="I couldn't find that moment in this title.",
            seek_to_ts=None,
            navigation_mode="not_found",
        )

    def _resolve_actor(self, title_id: str, question: str) -> NavigateResult | None:
        actor_name = _extract_actor_name(question)
        if actor_name is None and self._cast_lookup and self._title_display_name:
            actor_name = _match_cast_in_question(question, self._cast_lookup, self._title_display_name)
        if not actor_name:
            return None

        events = self._event_lookup.lookup(
            title_id=title_id,
            question=question,
            actor_name=actor_name,
        )
        ordinal, _ = parse_ordinal(question)
        if ordinal and ordinal > 0:
            events = self._event_lookup.lookup(
                title_id=title_id,
                question=question,
                actor_name=actor_name,
                ordinal=ordinal,
            )
        if events:
            return _from_event(events[0], "actor_appearance")

        hits = self._scene_navigate.navigate(
            title_id=title_id,
            query_text=actor_name,
            ordinal=ordinal,
        )
        if hits:
            return NavigateResult(
                answer=f"Jumping to where {actor_name} appears.",
                seek_to_ts=hits[0].start_ts,
                scene_id=hits[0].scene_id,
                event_type="actor_appearance",
                navigation_mode="actor_scene",
            )
        return None


def _from_event(event: TitleEventRecord, event_type: str) -> NavigateResult:
    return NavigateResult(
        answer=f"Jumping to {event.label}.",
        seek_to_ts=event.start_ts,
        scene_id=event.scene_id,
        event_type=event_type,
        navigation_mode="indexed_event",
    )


def _from_scene(hit: SceneLookupHit, ordinal: int | None) -> NavigateResult:
    ordinal_text = f" ({ordinal}{_ordinal_suffix(ordinal)})" if ordinal else ""
    snippet = hit.caption or hit.transcript or "that moment"
    return NavigateResult(
        answer=f"Jumping to{ordinal_text}: {snippet[:120]}.",
        seek_to_ts=hit.start_ts,
        scene_id=hit.scene_id,
        navigation_mode="semantic_scene",
    )


def _extract_actor_name(question: str) -> str | None:
    match = _ACTOR_NAV_RE.search(question)
    if match:
        return match.group(1).strip()
    return None


def _match_cast_in_question(
    question: str, cast_lookup: CastLookupTool, title_name: str
) -> str | None:
    result = cast_lookup.lookup(title_name=title_name)
    if "error" in result:
        return None
    lower = question.lower()
    for entry in result.get("cast", []):
        actor = str(entry.get("actor", ""))
        character = str(entry.get("character", ""))
        if actor and actor.lower() in lower:
            return actor
        if character and character.lower() in lower:
            return character
    return None


def _format_ts(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
