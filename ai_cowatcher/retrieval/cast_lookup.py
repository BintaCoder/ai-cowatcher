"""cast_lookup tool — public cast/actor metadata (ingest cache first, TMDB fallback).

Cast lists are public information and are not plot spoilers. Prefer the
per-title cast extracted during offline ingest (Postgres + optional Redis).
Live TMDB is a fallback for titles that have not been re-ingested yet.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Protocol

import httpx

from ai_cowatcher.config import Settings

logger = logging.getLogger(__name__)

_CAST_REDIS_PREFIX = "cowatcher:cast:"


class CastCacheReader(Protocol):
    def get_cast_cache(self, title_id: str) -> dict | None: ...


class CastCacheWriter(Protocol):
    def save_cast_cache(self, title_id: str, cast_payload: dict) -> None: ...


class CastRedisCache:
    """Optional hot cache in front of Postgres for real-time /ask."""

    def __init__(self, client: Any, *, ttl_sec: int = 7 * 24 * 3600) -> None:
        self._client = client
        self._ttl = max(ttl_sec, 60)

    def _key(self, title_id: str) -> str:
        return f"{_CAST_REDIS_PREFIX}{title_id}"

    def get(self, title_id: str) -> dict | None:
        try:
            raw = self._client.get(self._key(title_id))
        except Exception:  # noqa: BLE001
            logger.exception("Cast Redis get failed for %s", title_id)
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def set(self, title_id: str, payload: dict) -> None:
        try:
            self._client.setex(self._key(title_id), self._ttl, json.dumps(payload))
        except Exception:  # noqa: BLE001
            logger.exception("Cast Redis set failed for %s", title_id)

    def delete(self, title_id: str) -> None:
        try:
            self._client.delete(self._key(title_id))
        except Exception:  # noqa: BLE001
            logger.exception("Cast Redis delete failed for %s", title_id)


class InMemoryCastRedis:
    """Test double for CastRedisCache."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        del ttl
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)


class CastLookupTool:
    """Resolve cast for a title: Redis → Postgres ingest cache → TMDB."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        *,
        store: CastCacheReader | None = None,
        redis_cache: CastRedisCache | None = None,
    ):
        self._settings = settings
        self._client = client
        self._store = store
        self._redis = redis_cache

    def _http(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(timeout=self._settings.tmdb_timeout_sec)

    def _get_with_retry(self, client: httpx.Client, url: str, params: dict) -> httpx.Response:
        """GET with retry on transient connection failures (TLS resets, timeouts)."""
        attempts = max(1, self._settings.tmdb_max_retries)
        backoff = self._settings.tmdb_retry_backoff_sec
        last_exc: httpx.TransportError | None = None
        for attempt in range(attempts):
            try:
                response = client.get(url, params=params)
                response.raise_for_status()
                return response
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt + 1 < attempts:
                    wait = backoff * (2**attempt)
                    logger.warning(
                        "TMDB request failed (attempt %d/%d): %s; retrying in %.1fs",
                        attempt + 1,
                        attempts,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    def lookup(
        self,
        *,
        title_name: str = "",
        year: int | None = None,
        title_id: str | None = None,
    ) -> dict[str, object]:
        """Return cast dict or ``{"error": str}``.

        Prefer ingest cache when ``title_id`` is known; fall back to live TMDB
        search when needed.
        """
        cached = self._lookup_cached(title_id)
        if cached is not None:
            return cached

        # No cache or miss — live TMDB (or soft error if unconfigured).
        if not title_name or not str(title_name).strip():
            if title_id:
                return {
                    "error": (
                        "No cast cached for this title yet. "
                        "Re-ingest with a display name, or pass title_name."
                    )
                }
            return {"error": "No title name provided to search."}

        live = self._lookup_tmdb(title_name=str(title_name).strip(), year=year)
        if "cast" in live and title_id and self._store is not None:
            # Best-effort write-through so subsequent asks hit cache.
            try:
                writer = self._store
                if hasattr(writer, "save_cast_cache"):
                    writer.save_cast_cache(title_id, live)  # type: ignore[attr-defined]
                if self._redis is not None:
                    self._redis.set(title_id, live)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to write-through cast cache for %s", title_id)
        return live

    def _lookup_cached(self, title_id: str | None) -> dict[str, object] | None:
        if not title_id:
            return None
        if self._redis is not None:
            hit = self._redis.get(title_id)
            if hit and hit.get("cast"):
                result = dict(hit)
                result["source"] = "redis"
                return result
        if self._store is not None:
            try:
                hit = self._store.get_cast_cache(title_id)
            except Exception:  # noqa: BLE001
                logger.exception("Cast Postgres cache read failed for %s", title_id)
                hit = None
            if hit and hit.get("cast"):
                result = dict(hit)
                result.setdefault("source", "ingest")
                if self._redis is not None:
                    self._redis.set(title_id, result)
                return result
        return None

    def _lookup_tmdb(
        self, *, title_name: str, year: int | None = None
    ) -> dict[str, object]:
        if self._settings.mock_mode:
            return {
                "error": "Cast lookup uses offline mocks in MOCK_MODE (no TMDB call)."
            }
        if not self._settings.tmdb_api_key:
            return {"error": "Cast lookup is not configured (missing TMDB API key)."}

        params: dict[str, str] = {
            "api_key": self._settings.tmdb_api_key,
            "query": title_name,
        }
        if year is not None:
            params["year"] = str(year)

        base = self._settings.tmdb_base_url.rstrip("/")
        close_after = self._client is None
        client = self._http()
        try:
            search = self._get_with_retry(client, f"{base}/search/multi", params)
            results = [
                item
                for item in search.json().get("results", [])
                if item.get("media_type") in ("movie", "tv")
            ]
            if not results:
                return {"error": f'No TMDB match found for "{title_name}".'}

            best = results[0]
            media_type = best["media_type"]
            tmdb_id = best["id"]
            display_title = best.get("title") or best.get("name") or title_name

            credits = self._get_with_retry(
                client,
                f"{base}/{media_type}/{tmdb_id}/credits",
                {"api_key": self._settings.tmdb_api_key},
            )
            cast_entries = credits.json().get("cast", [])[
                : self._settings.tmdb_max_cast
            ]
            cast = [
                {
                    "actor": entry.get("name", ""),
                    "character": entry.get("character", ""),
                }
                for entry in cast_entries
                if entry.get("name")
            ]
            return {
                "title": display_title,
                "media_type": media_type,
                "tmdb_id": tmdb_id,
                "cast": cast,
                "source": "tmdb",
            }
        except httpx.TransportError as exc:
            logger.warning("TMDB cast lookup network failure for %s: %s", title_name, exc)
            return {
                "error": (
                    "Couldn't reach the cast database right now (network issue reaching TMDB). "
                    "This can happen on networks that block TMDB; please try again."
                )
            }
        except httpx.HTTPError as exc:
            logger.warning("TMDB cast lookup failed for %s: %s", title_name, exc)
            return {"error": f"Cast lookup failed: {exc}"}
        finally:
            if close_after:
                client.close()


def extract_cast_for_title(
    settings: Settings,
    *,
    title_name: str,
    year: int | None = None,
    client: httpx.Client | None = None,
) -> dict[str, object]:
    """One-shot TMDB extract used by the ingest pipeline (always live/API or error)."""
    tool = CastLookupTool(settings, client=client)
    return tool._lookup_tmdb(title_name=title_name, year=year)


def build_cast_redis_cache(settings: Settings) -> CastRedisCache | None:
    if settings.mock_mode:
        return None
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        ttl = getattr(settings, "cast_cache_redis_ttl_sec", 7 * 24 * 3600)
        return CastRedisCache(client, ttl_sec=int(ttl))
    except Exception:  # noqa: BLE001
        logger.warning("Redis unavailable for cast cache — Postgres only")
        return None


def actor_names_from_payload(payload: dict | None) -> list[str]:
    if not payload or "cast" not in payload:
        return []
    names: list[str] = []
    for entry in payload.get("cast") or []:
        if not isinstance(entry, dict):
            continue
        actor = str(entry.get("actor") or "").strip()
        if actor:
            names.append(actor)
    return names
