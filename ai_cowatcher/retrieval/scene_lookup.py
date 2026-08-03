"""scene_lookup tool — spoiler-safe semantic retrieval over ingested scenes.

Spoiler rule: only scenes with ``start_ts <= current_ts`` (already started,
including the scene currently playing). Scenes that start after the playhead
are never returned.
"""

from __future__ import annotations

import re

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import SceneLookupHit
from ai_cowatcher.interfaces import TextEmbedder
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore

# Visual / "now" questions — better answered from the playhead window than full ANN.
_PLAYHEAD_QUESTION = re.compile(
    r"(?ix)\b("
    r"on\s+(?:the\s+)?screen|"
    r"(?:who|what)(?:'s|\s+is)\s+(?:that|this|he|she|they)|"
    r"that\s+(?:guy|girl|man|woman|person|kid)|"
    r"what\s+just\s+happen|"
    r"what(?:'s|\s+is)\s+happen|"
    r"what(?:'s|\s+is)\s+going\s+on"
    r")\b"
)


def is_playhead_local_question(question: str) -> bool:
    return bool(_PLAYHEAD_QUESTION.search(question or ""))


class SceneLookupTool:
    """Query Qdrant for matching scenes at (or before) the viewer's playhead."""

    def __init__(
        self,
        embedder: TextEmbedder,
        qdrant_store: QdrantSceneStore,
        settings: Settings,
    ):
        self._embedder = embedder
        self._qdrant = qdrant_store
        self._top_k = settings.retrieval_top_k

    def lookup(
        self,
        *,
        title_id: str,
        query_text: str,
        current_ts: float,
        top_k: int | None = None,
    ) -> list[SceneLookupHit]:
        k = top_k or self._top_k
        # Fast path: skip embedding for current-moment questions.
        if is_playhead_local_question(query_text):
            hits = self._qdrant.playhead_scenes(
                title_id=title_id, current_ts=current_ts, limit=min(k, 3)
            )
            if hits:
                return hits
        vector = self._embedder.embed_texts([query_text])[0]
        return self._qdrant.search_scenes(
            title_id=title_id,
            query_vector=vector,
            current_ts=current_ts,
            top_k=k,
        )

    def lookup_playhead(
        self,
        *,
        title_id: str,
        current_ts: float,
        limit: int = 3,
    ) -> list[SceneLookupHit]:
        return self._qdrant.playhead_scenes(
            title_id=title_id, current_ts=current_ts, limit=limit
        )
