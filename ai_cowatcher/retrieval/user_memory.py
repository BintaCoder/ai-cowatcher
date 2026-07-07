"""user_memory tool — this viewer's own prior conversation for a title.

Returns recent turns or a short summary so the agent can maintain continuity
("as I mentioned earlier...") without stuffing full history into every prompt.

Scoped strictly to (user_id, title_id). The orchestrator injects user_id from
the authenticated request context — the LLM never supplies it.
"""

from __future__ import annotations

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import ConversationTurnRecord
from ai_cowatcher.storage.user_memory_store import UserMemoryStore


class UserMemoryTool:
    def __init__(self, store: UserMemoryStore, settings: Settings):
        self._store = store
        self._settings = settings

    def lookup(
        self,
        *,
        user_id: str,
        title_id: str,
        mode: str = "turns",
        max_turns: int | None = None,
    ) -> dict[str, object]:
        limit = max_turns or self._settings.user_memory_max_turns
        turns = self._store.get_recent_turns(user_id, title_id, max_turns=limit)
        if not turns:
            return {"found": False, "turns": [], "summary": ""}

        payload_turns = [turn.to_dict() for turn in turns]
        summary = _summarize_turns(turns) if mode == "summary" else ""
        return {
            "found": True,
            "turn_count": len(turns),
            "turns": payload_turns,
            "summary": summary or _summarize_turns(turns),
        }


def _summarize_turns(turns: list[ConversationTurnRecord]) -> str:
    """Deterministic short summary of recent Q&A pairs (no extra LLM call)."""
    parts: list[str] = []
    index = 0
    while index < len(turns):
        turn = turns[index]
        if turn.role == "user":
            question = turn.content.strip()
            answer = ""
            if index + 1 < len(turns) and turns[index + 1].role == "assistant":
                answer = turns[index + 1].content.strip()
                index += 2
            else:
                index += 1
            if question and answer:
                parts.append(f'You asked "{_clip(question)}" and I said "{_clip(answer)}".')
            elif question:
                parts.append(f'You asked "{_clip(question)}".')
        else:
            index += 1
    return " ".join(parts[-3:])


def _clip(text: str, max_len: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
