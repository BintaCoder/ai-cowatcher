"""scene_navigate — full-title semantic search for playback jumping."""

from __future__ import annotations

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import SceneLookupHit
from ai_cowatcher.interfaces import TextEmbedder
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore


class SceneNavigateTool:
    """Search scenes across the full title (navigation mode — no spoiler filter)."""

    def __init__(
        self,
        embedder: TextEmbedder,
        qdrant_store: QdrantSceneStore,
        settings: Settings,
    ):
        self._embedder = embedder
        self._qdrant = qdrant_store
        self._top_k = settings.navigation_top_k

    def navigate(
        self,
        *,
        title_id: str,
        query_text: str,
        ordinal: int | None = None,
        current_ts: float = 0.0,
    ) -> list[SceneLookupHit]:
        vector = self._embedder.embed_texts([query_text])[0]
        fetch_k = self._top_k if ordinal is None else max(self._top_k, ordinal * 3)
        hits = self._qdrant.search_scenes(
            title_id=title_id,
            query_vector=vector,
            current_ts=current_ts,
            top_k=fetch_k,
            spoiler_safe=False,
        )
        hits.sort(key=lambda hit: (hit.start_ts, hit.scene_id))
        if ordinal is not None and ordinal > 0:
            if ordinal <= len(hits):
                return [hits[ordinal - 1]]
            return []
        return hits
