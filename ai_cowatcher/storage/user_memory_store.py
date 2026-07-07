"""Redis + Postgres storage for per-user conversation memory."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Protocol

import redis
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from ai_cowatcher.config import Settings
from ai_cowatcher.db.models import UserConversationTurn
from ai_cowatcher.domain import ConversationTurnRecord
from ai_cowatcher.observability.prometheus_metrics import observe_storage_query

logger = logging.getLogger(__name__)


class MemoryCache(Protocol):
    def get_recent(self, user_id: str, title_id: str) -> list[ConversationTurnRecord] | None:
        ...

    def set_recent(self, user_id: str, title_id: str, turns: list[ConversationTurnRecord]) -> None:
        ...

    def append_turn(self, user_id: str, title_id: str, turn: ConversationTurnRecord) -> None:
        ...


class InMemoryMemoryCache:
    """Process-local cache for tests and mock mode."""

    def __init__(self) -> None:
        self._data: dict[str, list[ConversationTurnRecord]] = {}

    def _key(self, user_id: str, title_id: str) -> str:
        return f"{user_id}:{title_id}"

    def get_recent(self, user_id: str, title_id: str) -> list[ConversationTurnRecord] | None:
        turns = self._data.get(self._key(user_id, title_id))
        if turns is None:
            return None
        return list(turns)

    def set_recent(self, user_id: str, title_id: str, turns: list[ConversationTurnRecord]) -> None:
        self._data[self._key(user_id, title_id)] = list(turns)

    def append_turn(self, user_id: str, title_id: str, turn: ConversationTurnRecord) -> None:
        key = self._key(user_id, title_id)
        self._data.setdefault(key, []).append(turn)


class RedisMemoryCache:
    """Low-latency cache of recent turns for the active viewing session."""

    def __init__(self, settings: Settings, client: redis.Redis | None = None):
        self._settings = settings
        self._client = client or redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
        self._ttl = settings.user_memory_redis_ttl_sec
        self._max_cached = settings.user_memory_cache_turns

    def _key(self, user_id: str, title_id: str) -> str:
        return f"cowatcher:memory:{user_id}:{title_id}"

    def get_recent(self, user_id: str, title_id: str) -> list[ConversationTurnRecord] | None:
        raw = self._client.get(self._key(user_id, title_id))
        if raw is None:
            return None
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return [_record_from_dict(item, user_id, title_id) for item in items]

    def set_recent(self, user_id: str, title_id: str, turns: list[ConversationTurnRecord]) -> None:
        payload = json.dumps([turn.to_dict() for turn in turns[-self._max_cached :]])
        self._client.setex(self._key(user_id, title_id), self._ttl, payload)

    def append_turn(self, user_id: str, title_id: str, turn: ConversationTurnRecord) -> None:
        existing = self.get_recent(user_id, title_id) or []
        existing.append(turn)
        self.set_recent(user_id, title_id, existing[-self._max_cached :])


def _record_from_dict(item: dict, user_id: str, title_id: str) -> ConversationTurnRecord:
    return ConversationTurnRecord(
        turn_id=str(item.get("turn_id", "")),
        user_id=user_id,
        title_id=title_id,
        role=str(item.get("role", "")),
        content=str(item.get("content", "")),
        current_ts=float(item.get("current_ts", 0.0)),
    )


class UserMemoryRepository:
    """Postgres persistence for conversation turns — scoped by user_id + title_id."""

    def __init__(self, session: Session):
        self._session = session

    def append_turn(
        self,
        *,
        user_id: str,
        title_id: str,
        role: str,
        content: str,
        current_ts: float,
        turn_id: str | None = None,
    ) -> ConversationTurnRecord:
        record = ConversationTurnRecord(
            turn_id=turn_id or uuid.uuid4().hex,
            user_id=user_id,
            title_id=title_id,
            role=role,
            content=content,
            current_ts=current_ts,
        )
        self._session.add(
            UserConversationTurn(
                turn_id=record.turn_id,
                user_id=record.user_id,
                title_id=record.title_id,
                role=record.role,
                content=record.content,
                current_ts=record.current_ts,
            )
        )
        self._session.commit()
        return record

    def list_recent_turns(
        self, user_id: str, title_id: str, *, limit: int
    ) -> list[ConversationTurnRecord]:
        stmt = (
            select(UserConversationTurn)
            .where(
                UserConversationTurn.user_id == user_id,
                UserConversationTurn.title_id == title_id,
            )
            .order_by(UserConversationTurn.created_at.desc())
            .limit(limit)
        )
        rows = list(self._session.scalars(stmt).all())
        rows.reverse()
        return [_row_to_record(row) for row in rows]


def _row_to_record(row: UserConversationTurn) -> ConversationTurnRecord:
    return ConversationTurnRecord(
        turn_id=row.turn_id,
        user_id=row.user_id,
        title_id=row.title_id,
        role=row.role,
        content=row.content,
        current_ts=row.current_ts,
    )


class UserMemoryStore:
    """Coordinates Postgres (source of truth) and Redis (session cache)."""

    def __init__(
        self,
        session_factory: sessionmaker,
        cache: MemoryCache,
        settings: Settings,
    ):
        self._session_factory = session_factory
        self._cache = cache
        self._settings = settings

    def append_turn(
        self,
        *,
        user_id: str,
        title_id: str,
        role: str,
        content: str,
        current_ts: float,
    ) -> ConversationTurnRecord:
        with self._session_factory() as session:
            record = UserMemoryRepository(session).append_turn(
                user_id=user_id,
                title_id=title_id,
                role=role,
                content=content,
                current_ts=current_ts,
            )
        try:
            self._cache.append_turn(user_id, title_id, record)
        except Exception:  # noqa: BLE001 — cache failure must not drop the turn
            logger.exception("Failed to update user memory cache for %s/%s", user_id, title_id)
        return record

    def get_recent_turns(
        self, user_id: str, title_id: str, *, max_turns: int | None = None
    ) -> list[ConversationTurnRecord]:
        limit = max_turns or self._settings.user_memory_max_turns
        with observe_storage_query("redis", "get_recent_turns"):
            try:
                cached = self._cache.get_recent(user_id, title_id)
                if cached is not None:
                    return cached[-limit:]
            except Exception:  # noqa: BLE001
                logger.exception("User memory cache read failed for %s/%s", user_id, title_id)

        with observe_storage_query("postgres", "list_recent_turns"):
            with self._session_factory() as session:
                turns = UserMemoryRepository(session).list_recent_turns(
                    user_id, title_id, limit=limit
                )
        try:
            self._cache.set_recent(user_id, title_id, turns)
        except Exception:  # noqa: BLE001
            logger.exception("User memory cache refresh failed for %s/%s", user_id, title_id)
        return turns


_IN_MEMORY_CACHE: InMemoryMemoryCache | None = None


def build_memory_cache(settings: Settings) -> MemoryCache:
    if settings.mock_mode:
        global _IN_MEMORY_CACHE
        if _IN_MEMORY_CACHE is None:
            _IN_MEMORY_CACHE = InMemoryMemoryCache()
        return _IN_MEMORY_CACHE
    return RedisMemoryCache(settings)


def build_user_memory_store(
    settings: Settings,
    session_factory: sessionmaker,
    cache: MemoryCache | None = None,
) -> UserMemoryStore:
    return UserMemoryStore(
        session_factory=session_factory,
        cache=cache or build_memory_cache(settings),
        settings=settings,
    )
