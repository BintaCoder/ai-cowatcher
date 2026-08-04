"""Two-tier Q&A cache for the co-watcher real-time path.

Tier 1 (exact):    Redis (or in-memory) lookup on normalized (title_id, ts_bucket, question)
Tier 2 (semantic): Qdrant similarity search against a small ``qa_cache`` collection

Both tiers reuse Redis + Qdrant already in the stack. Synchronous API matches the
agent path (``asyncio.to_thread`` on /ask).

On a full miss, ``last_query_embedding`` is stashed so callers can reuse it for
store() without re-embedding.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ai_cowatcher.config import Settings
from ai_cowatcher.interfaces import TextEmbedder

logger = logging.getLogger(__name__)

DEFAULT_QA_CACHE_COLLECTION = "qa_cache"


class ExactKV(Protocol):
    def get(self, key: str) -> str | None: ...

    def setex(self, key: str, ttl: int, value: str) -> None: ...


class InMemoryExactKV:
    """Process-local exact cache for tests and when Redis is unavailable."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, str]] = {}  # key -> (expires_at, value)

    def get(self, key: str) -> str | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at and time.time() > expires_at:
            del self._data[key]
            return None
        return value

    def setex(self, key: str, ttl: int, value: str) -> None:
        expires_at = time.time() + max(ttl, 1)
        self._data[key] = (expires_at, value)


def ts_bucket(current_ts: float, bucket_sec: int) -> int:
    return int(current_ts // max(bucket_sec, 1))


def normalize_question(question: str) -> str:
    q = (question or "").strip().lower()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def exact_key(
    title_id: str,
    current_ts: float,
    question: str,
    *,
    bucket_sec: int,
    persona_id: str = "",
) -> str:
    norm = normalize_question(question)
    bucket = ts_bucket(current_ts, bucket_sec)
    persona = (persona_id or "").strip() or "_"
    raw = f"{title_id}:{bucket}:{persona}:{norm}"
    return "qa_exact:" + hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    audio_object_key: str | None = None
    title_id: str = ""
    ts_bucket: int = 0
    created_at: float = 0.0
    source: str = ""  # "exact" | "semantic"


class QACache:
    """Two-tier cache: Redis exact match, then Qdrant semantic near-duplicates."""

    def __init__(
        self,
        *,
        exact_kv: ExactKV,
        qdrant: QdrantClient,
        embedder: TextEmbedder,
        settings: Settings,
    ) -> None:
        self._exact = exact_kv
        self._qdrant = qdrant
        self._embedder = embedder
        self._settings = settings
        self._collection = settings.qa_cache_collection
        self._bucket_sec = settings.qa_cache_ts_bucket_sec
        self._exact_ttl = settings.qa_cache_exact_ttl_sec
        self._semantic_ttl = settings.qa_cache_semantic_ttl_sec
        self._threshold = settings.qa_cache_semantic_threshold
        self._top_k = settings.qa_cache_semantic_top_k
        self.last_query_embedding: list[float] | None = None
        self.ensure_collection()

    def ensure_collection(self) -> None:
        size = getattr(self._embedder, "vector_size", None) or len(
            self._embedder.embed_texts(["_"])[0]
        )
        try:
            if self._qdrant.collection_exists(self._collection):
                return
            self._qdrant.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=int(size), distance=qmodels.Distance.COSINE
                ),
            )
            logger.info("Created Qdrant QA cache collection %s (dim=%s)", self._collection, size)
        except Exception:  # noqa: BLE001 — cache must not break ask
            logger.exception("QA cache ensure_collection failed")

    # ---------- Tier 1: exact ----------

    def _exact_lookup(
        self,
        title_id: str,
        current_ts: float,
        question: str,
        *,
        persona_id: str = "",
    ) -> CachedAnswer | None:
        key = exact_key(
            title_id,
            current_ts,
            question,
            bucket_sec=self._bucket_sec,
            persona_id=persona_id,
        )
        try:
            raw = self._exact.get(key)
        except Exception:  # noqa: BLE001
            logger.exception("QA exact lookup failed")
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return CachedAnswer(
            answer=str(data.get("answer", "")),
            audio_object_key=data.get("audio_object_key"),
            title_id=str(data.get("title_id", title_id)),
            ts_bucket=int(data.get("ts_bucket", 0)),
            created_at=float(data.get("created_at", 0.0)),
            source="exact",
        )

    def _exact_store(
        self,
        title_id: str,
        current_ts: float,
        question: str,
        answer: str,
        audio_object_key: str | None,
        *,
        persona_id: str = "",
    ) -> None:
        key = exact_key(
            title_id,
            current_ts,
            question,
            bucket_sec=self._bucket_sec,
            persona_id=persona_id,
        )
        payload = {
            "answer": answer,
            "audio_object_key": audio_object_key,
            "title_id": title_id,
            "ts_bucket": ts_bucket(current_ts, self._bucket_sec),
            "created_at": time.time(),
            "persona_id": (persona_id or "").strip() or "_",
        }
        try:
            self._exact.setex(key, self._exact_ttl, json.dumps(payload))
        except Exception:  # noqa: BLE001
            logger.exception("QA exact store failed")

    # ---------- Tier 2: semantic ----------

    def _semantic_lookup(
        self,
        title_id: str,
        current_ts: float,
        question_embedding: list[float],
        *,
        persona_id: str = "",
    ) -> CachedAnswer | None:
        if not self._qdrant.collection_exists(self._collection):
            return None
        bucket = ts_bucket(current_ts, self._bucket_sec)
        persona = (persona_id or "").strip() or "_"
        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="title_id", match=qmodels.MatchValue(value=title_id)
                ),
                qmodels.FieldCondition(
                    key="ts_bucket", match=qmodels.MatchValue(value=bucket)
                ),
                qmodels.FieldCondition(
                    key="persona_id", match=qmodels.MatchValue(value=persona)
                ),
            ]
        )
        try:
            results = self._qdrant.query_points(
                collection_name=self._collection,
                query=question_embedding,
                query_filter=query_filter,
                limit=self._top_k,
                with_payload=True,
            ).points
        except Exception:  # noqa: BLE001
            logger.exception("QA semantic lookup failed")
            return None
        if not results:
            return None

        best = results[0]
        score = float(best.score or 0.0)
        if score < self._threshold:
            return None
        payload = best.payload or {}
        created_at = float(payload.get("created_at", 0.0))
        if created_at and time.time() - created_at > self._semantic_ttl:
            return None
        answer = str(payload.get("answer", "")).strip()
        if not answer:
            return None
        return CachedAnswer(
            answer=answer,
            audio_object_key=(
                str(payload["audio_object_key"])
                if payload.get("audio_object_key")
                else None
            ),
            title_id=str(payload.get("title_id", title_id)),
            ts_bucket=int(payload.get("ts_bucket", bucket)),
            created_at=created_at,
            source="semantic",
        )

    def _semantic_store(
        self,
        title_id: str,
        current_ts: float,
        question: str,
        question_embedding: list[float],
        answer: str,
        audio_object_key: str | None,
        *,
        persona_id: str = "",
    ) -> None:
        if not self._qdrant.collection_exists(self._collection):
            self.ensure_collection()
        if not self._qdrant.collection_exists(self._collection):
            return
        bucket = ts_bucket(current_ts, self._bucket_sec)
        norm = normalize_question(question)
        persona = (persona_id or "").strip() or "_"
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL, f"qa:{title_id}:{bucket}:{persona}:{norm}"
            )
        )
        payload: dict[str, Any] = {
            "answer": answer,
            "audio_object_key": audio_object_key,
            "title_id": title_id,
            "ts_bucket": bucket,
            "created_at": time.time(),
            "question_norm": norm,
            "persona_id": persona,
        }
        try:
            self._qdrant.upsert(
                collection_name=self._collection,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=question_embedding,
                        payload=payload,
                    )
                ],
            )
        except Exception:  # noqa: BLE001
            logger.exception("QA semantic store failed")

    # ---------- Public API ----------

    def lookup(
        self,
        title_id: str,
        current_ts: float,
        question: str,
        *,
        persona_id: str = "",
    ) -> CachedAnswer | None:
        """Exact first (near-zero cost), then semantic. None on full miss.

        Exact hits never call the embedder. Semantic embed runs only on exact miss.
        Stage timings are logged as ``qa_cache_lookup`` for latency dashboards.
        Cache keys isolate by persona_id so different tones never cross-hit.
        """
        self.last_query_embedding = None
        t0 = time.perf_counter()
        persona = (persona_id or "").strip()

        t_exact = time.perf_counter()
        hit = self._exact_lookup(
            title_id, current_ts, question, persona_id=persona
        )
        exact_ms = (time.perf_counter() - t_exact) * 1000.0
        if hit and hit.answer.strip():
            self._record_cache_metric("exact_hit")
            self._log_lookup_stages(
                result="exact_hit",
                exact_ms=exact_ms,
                embed_ms=0.0,
                semantic_ms=0.0,
                total_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return hit

        # Semantic tier only: embed after exact miss (never on exact hit path).
        embed_ms = 0.0
        semantic_ms = 0.0
        try:
            t_embed = time.perf_counter()
            embedding = self._embedder.embed_texts([question])[0]
            embed_ms = (time.perf_counter() - t_embed) * 1000.0
        except Exception:  # noqa: BLE001
            logger.exception("QA cache embed failed")
            self._record_cache_metric("miss")
            self._log_lookup_stages(
                result="miss",
                exact_ms=exact_ms,
                embed_ms=embed_ms,
                semantic_ms=0.0,
                total_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return None

        t_sem = time.perf_counter()
        hit = self._semantic_lookup(
            title_id, current_ts, embedding, persona_id=persona
        )
        semantic_ms = (time.perf_counter() - t_sem) * 1000.0
        if hit:
            self._record_cache_metric("semantic_hit")
            self._log_lookup_stages(
                result="semantic_hit",
                exact_ms=exact_ms,
                embed_ms=embed_ms,
                semantic_ms=semantic_ms,
                total_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return hit

        self.last_query_embedding = embedding
        self._record_cache_metric("miss")
        self._log_lookup_stages(
            result="miss",
            exact_ms=exact_ms,
            embed_ms=embed_ms,
            semantic_ms=semantic_ms,
            total_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return None

    @staticmethod
    def _log_lookup_stages(
        *,
        result: str,
        exact_ms: float,
        embed_ms: float,
        semantic_ms: float,
        total_ms: float,
    ) -> None:
        try:
            logger.info(
                json.dumps(
                    {
                        "event": "qa_cache_lookup",
                        "result": result,
                        "exact_ms": round(exact_ms, 2),
                        "embed_ms": round(embed_ms, 2),
                        "semantic_ms": round(semantic_ms, 2),
                        "total_ms": round(total_ms, 2),
                        "exact_only": result == "exact_hit",
                    },
                    separators=(",", ":"),
                )
            )
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _record_cache_metric(result: str) -> None:
        try:
            from ai_cowatcher.observability.prometheus_metrics import (
                record_qa_cache_result,
            )

            record_qa_cache_result(result)
        except Exception:  # noqa: BLE001
            pass

    def store(
        self,
        title_id: str,
        current_ts: float,
        question: str,
        answer: str,
        *,
        question_embedding: list[float] | None = None,
        audio_object_key: str | None = None,
        persona_id: str = "",
    ) -> None:
        """Write-through to both tiers after a fresh answer."""
        text = (answer or "").strip()
        if not text:
            return
        persona = (persona_id or "").strip()
        self._exact_store(
            title_id,
            current_ts,
            question,
            text,
            audio_object_key,
            persona_id=persona,
        )

        embedding = question_embedding or self.last_query_embedding
        if embedding is None:
            try:
                embedding = self._embedder.embed_texts([question])[0]
            except Exception:  # noqa: BLE001
                logger.exception("QA cache embed for store failed")
                return
        self._semantic_store(
            title_id,
            current_ts,
            question,
            embedding,
            text,
            audio_object_key,
            persona_id=persona,
        )


def should_cache_answer(
    *,
    answer: str,
    speak: bool,
    skip_memory: bool,
    escalation_reason: str | None,
    navigate: bool = False,
) -> bool:
    """Decide whether a generated answer is safe to cache for future replays."""
    if navigate:
        return False
    if skip_memory and not (answer or "").strip():
        return False
    text = (answer or "").strip()
    if not text:
        return False
    if not speak:
        return False
    reason = (escalation_reason or "").lower()
    if "filler" in reason or reason.endswith(":ignore") or "ignore_" in reason:
        return False
    if "gate:social" in reason or reason == "gate:social":
        # Canned social is persona-static; skip cache pollution / false "miss" noise.
        return False
    if "navigate" in reason:
        return False
    # Avoid poisoning with soft refusals
    lower = text.lower()
    if lower.startswith("not sure yet") or "nothing's made that clear" in lower:
        return False
    return True


def build_qa_cache(
    settings: Settings,
    *,
    embedder: TextEmbedder,
    qdrant_client: QdrantClient | None = None,
    exact_kv: ExactKV | None = None,
) -> QACache | None:
    if not getattr(settings, "qa_cache_enabled", False):
        return None

    if exact_kv is None:
        exact_kv = _build_exact_kv(settings)

    client = qdrant_client or QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )
    return QACache(
        exact_kv=exact_kv,
        qdrant=client,
        embedder=embedder,
        settings=settings,
    )


def _build_exact_kv(settings: Settings) -> ExactKV:
    if settings.mock_mode:
        return InMemoryExactKV()
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        return _RedisExactKV(client)
    except Exception:  # noqa: BLE001
        logger.warning("Redis unavailable for QA cache — using in-memory exact tier")
        return InMemoryExactKV()


class _RedisExactKV:
    def __init__(self, client: Any) -> None:
        self._client = client

    def get(self, key: str) -> str | None:
        raw = self._client.get(key)
        if raw is None:
            return None
        return str(raw)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._client.setex(key, ttl, value)
