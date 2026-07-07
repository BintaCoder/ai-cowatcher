"""knowledge_search tool — RAG over a curated, non-spoiler knowledge base.

This is the **only** retrieval tool in the system with **no** playback-position
(`current_ts`) filter. By design it answers public, spoiler-insensitive facts
(actor biographies, crew/creator info, sports statistics, production trivia)
from a knowledge base we control and vet offline — not the live open web, so
answers stay consistent and cannot pull plot spoilers from external wikis or
recaps.

scene_lookup and character_lookup are spoiler-safe (timestamp-filtered).
knowledge_search is intentionally not.
"""

from __future__ import annotations

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import KnowledgeSearchHit
from ai_cowatcher.interfaces import TextEmbedder
from ai_cowatcher.storage.qdrant_knowledge_store import QdrantKnowledgeStore


class KnowledgeSearchTool:
    """Semantic search over curated title knowledge (BGE-M3 + Qdrant)."""

    def __init__(
        self,
        embedder: TextEmbedder,
        knowledge_store: QdrantKnowledgeStore,
        settings: Settings,
    ):
        self._embedder = embedder
        self._store = knowledge_store
        self._top_k = settings.knowledge_top_k

    def search(
        self,
        *,
        title_id: str,
        query_text: str,
        category: str | None = None,
        top_k: int | None = None,
    ) -> list[KnowledgeSearchHit]:
        """Search curated knowledge for a title. No current_ts — not spoiler-filtered."""
        vector = self._embedder.embed_texts([query_text])[0]
        return self._store.search_knowledge(
            title_id=title_id,
            query_vector=vector,
            top_k=top_k or self._top_k,
            category=category,
        )
