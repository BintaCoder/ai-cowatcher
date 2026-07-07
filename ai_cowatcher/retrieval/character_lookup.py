"""character_lookup tool — spoiler-safe character intelligence from Neo4j.

Mirrors scene_lookup's spoiler principle: the character store only returns
appearances/relationships whose timestamp is ``<= current_ts``, so nothing the
viewer hasn't watched yet can leak into an answer.
"""

from __future__ import annotations

from ai_cowatcher.storage.character_store import CharacterStore


class CharacterLookupTool:
    def __init__(self, store: CharacterStore):
        self._store = store

    def lookup(
        self, *, title_id: str, character: str | None, current_ts: float
    ) -> dict[str, object]:
        result = self._store.character_lookup(title_id, character or None, current_ts)
        if result is None:
            return {
                "found": False,
                "message": (
                    "No character intelligence is available for this moment yet."
                ),
            }
        payload = result.to_tool_dict()
        payload["found"] = True
        return payload
