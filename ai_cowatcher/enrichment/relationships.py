"""Build timestamped appearance history and character relationships.

Every appearance and relationship carries the timestamp at which it becomes
true on screen. ``known_since_ts`` on a relationship is the spoiler anchor: the
earliest playback position where the relationship is revealed (either the first
scene the two characters share, or an earlier/later scene whose dialogue
explicitly states the relationship).
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

from ai_cowatcher.domain import (
    CharacterAppearance,
    CharacterIdentity,
    CharacterRelationship,
    SceneEventRecord,
)

# Ordered by specificity — the first matching keyword wins for a scene.
_RELATIONSHIP_KEYWORDS: list[tuple[str, str]] = [
    (r"\b(brother|sister|sibling|siblings)\b", "sibling"),
    (r"\b(husband|wife|spouse|married|wedding)\b", "spouse"),
    (r"\b(father|mother|dad|mom|son|daughter|parent|child)\b", "family"),
    (r"\b(love|lovers?|dating|girlfriend|boyfriend|kiss(?:ed|ing)?)\b", "romantic"),
    (r"\b(enemy|enemies|hate|betray(?:ed|al)?|kill(?:ed|er)?)\b", "adversary"),
    (r"\b(friend|friends|friendship|buddy|pal)\b", "friend"),
    (r"\b(boss|colleague|partner|coworker|co-worker)\b", "colleague"),
]

_COMPILED_KEYWORDS = [(re.compile(pattern, re.I), rel_type) for pattern, rel_type in _RELATIONSHIP_KEYWORDS]


def _clusters_of(character: CharacterIdentity) -> set[str]:
    return set(character.face_cluster_ids) | set(character.speaker_cluster_ids)


def build_appearances(
    characters: list[CharacterIdentity],
    scenes: list[SceneEventRecord],
) -> list[CharacterAppearance]:
    ordered = sorted(scenes, key=lambda s: s.start_ts)
    appearances: list[CharacterAppearance] = []
    for scene in ordered:
        active = set(scene.face_cluster_ids) | set(scene.speaker_cluster_ids)
        for character in characters:
            if _clusters_of(character) & active:
                appearances.append(
                    CharacterAppearance(
                        character_id=character.character_id,
                        scene_id=scene.scene_id,
                        start_ts=scene.start_ts,
                        end_ts=scene.end_ts,
                    )
                )
    return appearances


def build_relationships(
    title_id: str,
    characters: list[CharacterIdentity],
    scenes: list[SceneEventRecord],
) -> list[CharacterRelationship]:
    ordered = sorted(scenes, key=lambda s: s.start_ts)
    names = {c.character_id: c.name for c in characters}

    scene_members: list[tuple[SceneEventRecord, set[str]]] = []
    for scene in ordered:
        active = set(scene.face_cluster_ids) | set(scene.speaker_cluster_ids)
        members = {
            character.character_id
            for character in characters
            if _clusters_of(character) & active
        }
        scene_members.append((scene, members))

    co_counts: dict[tuple[str, str], int] = defaultdict(int)
    first_together: dict[tuple[str, str], SceneEventRecord] = {}
    reveal: dict[tuple[str, str], tuple[str, SceneEventRecord]] = {}

    for scene, members in scene_members:
        rel_type = _relationship_type_in_text(f"{scene.transcript}\n{scene.caption}")
        for pair in combinations(sorted(members), 2):
            co_counts[pair] += 1
            first_together.setdefault(pair, scene)
            if rel_type is not None and pair not in reveal:
                reveal[pair] = (rel_type, scene)

    relationships: list[CharacterRelationship] = []
    for pair, count in co_counts.items():
        source_id, target_id = pair
        if pair in reveal:
            rel_type, reveal_scene = reveal[pair]
            known_since_ts = reveal_scene.start_ts
            scene_id = reveal_scene.scene_id
            summary = _reveal_summary(rel_type, names.get(source_id), names.get(target_id))
        else:
            first_scene = first_together[pair]
            rel_type = "acquainted"
            known_since_ts = first_scene.start_ts
            scene_id = first_scene.scene_id
            summary = _acquainted_summary(
                names.get(source_id), names.get(target_id), count
            )
        relationships.append(
            CharacterRelationship(
                title_id=title_id,
                source_id=source_id,
                target_id=target_id,
                rel_type=rel_type,
                summary=summary,
                known_since_ts=known_since_ts,
                scene_id=scene_id,
                source_name=names.get(source_id),
                target_name=names.get(target_id),
            )
        )
    relationships.sort(key=lambda r: (r.known_since_ts, r.source_id, r.target_id))
    return relationships


def _relationship_type_in_text(text: str) -> str | None:
    for pattern, rel_type in _COMPILED_KEYWORDS:
        if pattern.search(text):
            return rel_type
    return None


def _label(name: str | None, character_id: str = "") -> str:
    return name or character_id or "someone"


def _reveal_summary(rel_type: str, a: str | None, b: str | None) -> str:
    left = _label(a)
    right = _label(b)
    templates = {
        "sibling": f"{left} and {right} are siblings.",
        "spouse": f"{left} and {right} are married.",
        "family": f"{left} and {right} are family.",
        "romantic": f"{left} and {right} are romantically involved.",
        "adversary": f"{left} and {right} are in conflict.",
        "friend": f"{left} and {right} are friends.",
        "colleague": f"{left} and {right} work together.",
    }
    return templates.get(rel_type, f"{left} and {right} know each other.")


def _acquainted_summary(a: str | None, b: str | None, count: int) -> str:
    scenes_word = "scene" if count == 1 else "scenes"
    return f"{_label(a)} and {_label(b)} have shared {count} {scenes_word} so far."
