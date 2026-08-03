"""scene_lookup tool — spoiler-safe semantic retrieval over ingested scenes.

Spoiler rule: only scenes with ``start_ts <= current_ts`` (already started,
including the scene currently playing). Scenes that start after the playhead
are never returned.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import OrderedDict

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import SceneLookupHit
from ai_cowatcher.interfaces import TextEmbedder
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore

logger = logging.getLogger(__name__)

# Visual / "now" questions — better answered from the playhead window than full ANN.
_PLAYHEAD_QUESTION = re.compile(
    r"(?ix)\b("
    r"on\s+(?:the\s+)?screen|"
    r"(?:who|what)(?:'s|\s+is)\s+(?:that|this|he|she|they)|"
    r"who(?:'s|\s+is)\s+on\s+(?:the\s+)?screen|"
    r"that\s+(?:guy|girl|man|woman|person|kid)|"
    r"what\s+just\s+happen(?:ed|ing)?|"
    r"what(?:'s|\s+is)\s+happen(?:ing)?|"
    r"what(?:'s|\s+is)\s+going\s+on"
    r")\b"
)


def is_playhead_local_question(question: str) -> bool:
    return bool(_PLAYHEAD_QUESTION.search(question or ""))


class _QueryEmbeddingCache:
    """Process-local TTL cache for query embeddings (BGE skip when repeated)."""

    def __init__(self, *, ttl_sec: float, max_entries: int) -> None:
        self._ttl = max(float(ttl_sec), 0.0)
        self._max = max(int(max_entries), 1)
        self._items: OrderedDict[str, tuple[float, list[float]]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> list[float] | None:
        if self._ttl <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, vector = item
            if now > expires_at:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return list(vector)

    def put(self, key: str, vector: list[float]) -> None:
        if self._ttl <= 0:
            return
        expires_at = time.monotonic() + self._ttl
        with self._lock:
            if key in self._items:
                del self._items[key]
            self._items[key] = (expires_at, list(vector))
            while len(self._items) > self._max:
                self._items.popitem(last=False)


class SceneLookupTool:
    """Query Qdrant for matching scenes at (or before) the viewer's playhead."""

    def __init__(
        self,
        embedder: TextEmbedder,
        qdrant_store: QdrantSceneStore,
        settings: Settings,
        *,
        embedding_cache: _QueryEmbeddingCache | None = None,
    ):
        self._embedder = embedder
        self._qdrant = qdrant_store
        self._top_k = settings.retrieval_top_k
        self._embedding_cache = embedding_cache or _QueryEmbeddingCache(
            ttl_sec=float(
                getattr(settings, "query_embedding_cache_ttl_sec", 90.0) or 0.0
            ),
            max_entries=int(
                getattr(settings, "query_embedding_cache_max", 256) or 256
            ),
        )

    def _cache_key(self, query_text: str) -> str:
        return re.sub(r"\s+", " ", (query_text or "").strip().lower())

    def _embed_query(
        self,
        query_text: str,
        *,
        query_vector: list[float] | None = None,
    ) -> tuple[list[float], str]:
        """Return (vector, source) where source is reuse|cache|embed."""
        if query_vector is not None:
            return query_vector, "reuse"
        key = self._cache_key(query_text)
        cached = self._embedding_cache.get(key) if key else None
        if cached is not None:
            return cached, "cache"
        vector = self._embedder.embed_texts([query_text])[0]
        if key:
            self._embedding_cache.put(key, vector)
        return vector, "embed"

    def lookup(
        self,
        *,
        title_id: str,
        query_text: str,
        current_ts: float,
        top_k: int | None = None,
        query_vector: list[float] | None = None,
    ) -> list[SceneLookupHit]:
        k = top_k or self._top_k
        # Fast path: skip embedding for current-moment questions.
        if is_playhead_local_question(query_text) and query_vector is None:
            hits = self._qdrant.playhead_scenes(
                title_id=title_id, current_ts=current_ts, limit=min(k, 3)
            )
            if hits:
                logger.info(
                    json.dumps(
                        {
                            "event": "scene_lookup_path",
                            "path": "playhead",
                            "skip_bge": True,
                            "title_id": title_id,
                            "current_ts": current_ts,
                            "hit_count": len(hits),
                        },
                        separators=(",", ":"),
                    )
                )
                return hits
            logger.info(
                json.dumps(
                    {
                        "event": "scene_lookup_path",
                        "path": "playhead_miss_embed",
                        "skip_bge": False,
                        "title_id": title_id,
                        "current_ts": current_ts,
                    },
                    separators=(",", ":"),
                )
            )

        vector, embed_source = self._embed_query(
            query_text, query_vector=query_vector
        )
        logger.info(
            json.dumps(
                {
                    "event": "scene_lookup_path",
                    "path": "embed",
                    "skip_bge": False,
                    "embed_source": embed_source,
                    "title_id": title_id,
                    "current_ts": current_ts,
                },
                separators=(",", ":"),
            )
        )
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
