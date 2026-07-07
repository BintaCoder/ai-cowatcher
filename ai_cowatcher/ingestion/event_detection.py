"""Detect navigable title events during offline ingestion."""

from __future__ import annotations

import re

from ai_cowatcher.domain import SceneEventRecord, TitleEventRecord

_SPORTS_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "goal": [
        re.compile(r"\bgoal\b", re.I),
        re.compile(r"\bscores?\b", re.I),
    ],
    "point": [
        re.compile(r"\bpoint\b", re.I),
        re.compile(r"\bscores?\b", re.I),
    ],
    "six": [
        re.compile(r"\bsix\b", re.I),
        re.compile(r"\bsixer\b", re.I),
    ],
    "wicket": [
        re.compile(r"\bwicket\b", re.I),
    ],
    "touchdown": [
        re.compile(r"\btouchdown\b", re.I),
    ],
    "fight": [
        re.compile(r"\bfight(?:ing|s)?\b", re.I),
        re.compile(r"\bpunch(?:es|ed|ing)?\b", re.I),
        re.compile(r"\bbrawl\b", re.I),
    ],
}

_CREDITS_PATTERNS = [
    re.compile(r"\bcredits?\b", re.I),
    re.compile(r"\bdirected by\b", re.I),
    re.compile(r"\bexecutive producer\b", re.I),
    re.compile(r"\bpost[- ]credits?\b", re.I),
]

_CRASH_PATTERNS = [
    re.compile(r"\bairplane\b", re.I),
    re.compile(r"\bplane\b", re.I),
    re.compile(r"\bcrash(?:es|ed|ing)?\b", re.I),
    re.compile(r"\bfalls?\b", re.I),
    re.compile(r"\bexplod", re.I),
]


class _SceneText:
    __slots__ = ("scene_id", "start_ts", "end_ts", "text")

    def __init__(self, scene_id: str, start_ts: float, end_ts: float, text: str):
        self.scene_id = scene_id
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.text = text


def _scene_texts(scenes: list[SceneEventRecord]) -> list[_SceneText]:
    return [
        _SceneText(
            scene.scene_id,
            scene.start_ts,
            scene.end_ts,
            f"{scene.transcript}\n{scene.caption}".strip(),
        )
        for scene in scenes
    ]


def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _make_event(
    *,
    title_id: str,
    event_type: str,
    ordinal: int,
    scene: _SceneText,
    label: str,
    metadata: dict[str, object] | None = None,
) -> TitleEventRecord:
    return TitleEventRecord(
        event_id=f"{title_id}:{event_type}_{ordinal:03d}",
        title_id=title_id,
        event_type=event_type,
        ordinal=ordinal,
        start_ts=scene.start_ts,
        end_ts=scene.end_ts,
        scene_id=scene.scene_id,
        label=label,
        metadata=metadata or {},
    )


def detect_pattern_events(
    title_id: str,
    scenes: list[SceneEventRecord],
    pattern_map: dict[str, list[re.Pattern[str]]],
) -> list[TitleEventRecord]:
    events: list[TitleEventRecord] = []
    counters: dict[str, int] = {}
    for scene in _scene_texts(scenes):
        for event_type, patterns in pattern_map.items():
            if not _matches_any(scene.text, patterns):
                continue
            counters[event_type] = counters.get(event_type, 0) + 1
            ordinal = counters[event_type]
            events.append(
                _make_event(
                    title_id=title_id,
                    event_type=event_type,
                    ordinal=ordinal,
                    scene=scene,
                    label=f"{ordinal}{_ordinal_suffix(ordinal)} {event_type}",
                )
            )
    return events


def detect_credits_start(scenes: list[SceneEventRecord]) -> float | None:
    if not scenes:
        return None
    duration = max(scene.end_ts for scene in scenes)
    tail_start = duration * 0.85
    credits_scenes = [
        scene
        for scene in _scene_texts(scenes)
        if scene.start_ts >= tail_start and _matches_any(scene.text, _CREDITS_PATTERNS)
    ]
    if credits_scenes:
        return min(scene.start_ts for scene in credits_scenes)
    last = max(scenes, key=lambda s: s.start_ts)
    if last.start_ts >= duration * 0.92:
        return last.start_ts
    return None


def detect_actor_appearances(
    title_id: str,
    scenes: list[SceneEventRecord],
    cast_names: list[str],
) -> list[TitleEventRecord]:
    if not cast_names:
        return []

    events: list[TitleEventRecord] = []
    for name in cast_names:
        if not name or len(name) < 3:
            continue
        pattern = re.compile(rf"\b{re.escape(name)}\b", re.I)
        ordinal = 0
        for scene in _scene_texts(scenes):
            if not pattern.search(scene.text):
                continue
            ordinal += 1
            events.append(
                _make_event(
                    title_id=title_id,
                    event_type="actor_appearance",
                    ordinal=ordinal,
                    scene=scene,
                    label=f"{name} on screen",
                    metadata={"actor_name": name},
                )
            )
    return events


def build_title_events(
    title_id: str,
    scenes: list[SceneEventRecord],
    cast_names: list[str] | None = None,
) -> tuple[list[TitleEventRecord], float | None]:
    sports = detect_pattern_events(title_id, scenes, _SPORTS_PATTERNS)
    crashes = detect_pattern_events(title_id, scenes, {"crash": _CRASH_PATTERNS})
    actors = detect_actor_appearances(title_id, scenes, cast_names or [])
    credits_ts = detect_credits_start(scenes)

    credits_events: list[TitleEventRecord] = []
    if credits_ts is not None:
        credits_events.append(
            TitleEventRecord(
                event_id=f"{title_id}:credits_001",
                title_id=title_id,
                event_type="credits",
                ordinal=1,
                start_ts=credits_ts,
                end_ts=max(scene.end_ts for scene in scenes),
                scene_id=None,
                label="Credits / post-credits",
                metadata={"credits_start_ts": credits_ts},
            )
        )

    all_events = sports + crashes + actors + credits_events
    all_events.sort(key=lambda event: (event.start_ts, event.event_type, event.ordinal))
    return all_events, credits_ts


def _ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
