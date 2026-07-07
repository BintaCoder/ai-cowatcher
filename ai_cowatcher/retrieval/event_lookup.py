"""event_lookup — query indexed sports/credits/actor title events."""

from __future__ import annotations

import re

from ai_cowatcher.domain import TitleEventRecord
from ai_cowatcher.navigation.ordinal import parse_ordinal
from ai_cowatcher.storage.postgres_store import SceneEventRepository

_EVENT_ALIASES: dict[str, str] = {
    "goal": "goal",
    "goals": "goal",
    "point": "point",
    "points": "point",
    "six": "six",
    "sixer": "six",
    "sixes": "six",
    "wicket": "wicket",
    "wickets": "wicket",
    "touchdown": "touchdown",
    "touchdowns": "touchdown",
    "fight": "fight",
    "fights": "fight",
    "credits": "credits",
    "credit": "credits",
    "post-credits": "credits",
    "post credits": "credits",
    "crash": "crash",
    "airplane": "crash",
    "plane": "crash",
    "actor": "actor_appearance",
}


class EventLookupTool:
    def __init__(self, repo: SceneEventRepository):
        self._repo = repo

    def lookup(
        self,
        *,
        title_id: str,
        question: str,
        event_type: str | None = None,
        ordinal: int | None = None,
        actor_name: str | None = None,
    ) -> list[TitleEventRecord]:
        if actor_name:
            events = self._repo.list_title_events(title_id, event_type="actor_appearance")
            filtered = [
                event
                for event in events
                if actor_name.lower() in event.label.lower()
                or str(event.metadata.get("actor_name", "")).lower() == actor_name.lower()
            ]
            if ordinal is not None and ordinal > 0:
                return [filtered[ordinal - 1]] if ordinal <= len(filtered) else []
            return filtered

        resolved_type = event_type or _infer_event_type(question)
        if resolved_type is None:
            return []

        parsed_ordinal, _ = parse_ordinal(question)
        use_ordinal = ordinal if ordinal is not None else parsed_ordinal

        if resolved_type == "credits":
            credits_ts = self._repo.get_credits_start_ts(title_id)
            if credits_ts is not None:
                return self._repo.list_title_events(title_id, event_type="credits")
            return []

        events = self._repo.list_title_events(title_id, event_type=resolved_type)
        if use_ordinal is not None and use_ordinal > 0:
            match = [event for event in events if event.ordinal == use_ordinal]
            return match
        return events


def _infer_event_type(question: str) -> str | None:
    lower = question.lower()
    for alias, event_type in sorted(_EVENT_ALIASES.items(), key=lambda item: -len(item[0])):
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            return event_type
    return None
