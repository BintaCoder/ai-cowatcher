"""Best-effort character name resolution from cast/credits metadata.

Mapping an anonymous face/voice cluster to a real character name is inherently
noisy without labelled reference photos. We use a conservative, deterministic
heuristic: attribute a cast character name to the identity whose scenes mention
that name most, requiring a unique winner. Unmatched identities keep ``name=None``
rather than guessing, so the co-watcher never asserts a wrong name.
"""

from __future__ import annotations

import re
from dataclasses import replace

from ai_cowatcher.domain import CharacterAppearance, CharacterIdentity, SceneEventRecord


def resolve_names(
    characters: list[CharacterIdentity],
    appearances: list[CharacterAppearance],
    scenes: list[SceneEventRecord],
    cast_names: list[str],
) -> list[CharacterIdentity]:
    candidates = [name.strip() for name in cast_names if name and len(name.strip()) >= 2]
    if not candidates:
        return characters

    scene_text = {scene.scene_id: f"{scene.transcript}\n{scene.caption}" for scene in scenes}
    appearances_by_char: dict[str, list[str]] = {}
    for appearance in appearances:
        appearances_by_char.setdefault(appearance.character_id, []).append(
            appearance.scene_id
        )

    # name -> {character_id: mention_count}
    name_hits: dict[str, dict[str, int]] = {name: {} for name in candidates}
    patterns = {name: re.compile(rf"\b{re.escape(name)}\b", re.I) for name in candidates}

    for character in characters:
        text = " ".join(
            scene_text.get(scene_id, "")
            for scene_id in appearances_by_char.get(character.character_id, [])
        )
        for name in candidates:
            count = len(patterns[name].findall(text))
            if count:
                name_hits[name][character.character_id] = count

    assigned: dict[str, str] = {}
    used_names: set[str] = set()
    # Assign each name to the single character that mentions it most (unique winner).
    for name, hits in name_hits.items():
        if not hits:
            continue
        ranked = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)
        top_char, top_count = ranked[0]
        if len(ranked) > 1 and ranked[1][1] == top_count:
            continue  # ambiguous — skip rather than guess
        if top_char in assigned or name in used_names:
            continue
        assigned[top_char] = name
        used_names.add(name)

    return [
        replace(character, name=assigned.get(character.character_id, character.name))
        for character in characters
    ]
