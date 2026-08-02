"""scene_lookup tool — spoiler-safe semantic retrieval over ingested scenes."""

from __future__ import annotations

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import SceneLookupHit
from ai_cowatcher.interfaces import TextEmbedder
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore


class SceneLookupTool:
    """Query Qdrant for matching scenes visible at the viewer's current timestamp."""

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
        vector = self._embedder.embed_texts([query_text])[0]
        return self._qdrant.search_scenes(
            title_id=title_id,
            query_vector=vector,
            current_ts=current_ts,
            top_k=top_k or self._top_k,
        )
