"""cast_lookup tool — public cast/actor metadata from TMDB.

Cast lists are public information and are not plot spoilers, so this tool is
safe to expose to the real-time co-watcher without the spoiler guard that
scene_lookup uses.
"""

from __future__ import annotations

import logging
import time

import httpx

from ai_cowatcher.config import Settings

logger = logging.getLogger(__name__)


class CastLookupTool:
    """Search TMDB for a title and return its top-billed cast."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self._settings = settings
        self._client = client

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
                # Network-level failure (e.g. TMDB TLS reset by ISP/middlebox). Retry.
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

    def lookup(self, *, title_name: str, year: int | None = None) -> dict[str, object]:
        """Return {"title": str, "media_type": str, "cast": [{actor, character}]} or {"error": str}."""
        if not self._settings.tmdb_api_key:
            return {"error": "Cast lookup is not configured (missing TMDB API key)."}
        if not title_name or not title_name.strip():
            return {"error": "No title name provided to search."}

        params = {"api_key": self._settings.tmdb_api_key, "query": title_name.strip()}
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
            cast_entries = credits.json().get("cast", [])[: self._settings.tmdb_max_cast]
            cast = [
                {
                    "actor": entry.get("name", ""),
                    "character": entry.get("character", ""),
                }
                for entry in cast_entries
                if entry.get("name")
            ]
            return {"title": display_title, "media_type": media_type, "cast": cast}
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
