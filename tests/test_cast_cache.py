"""Cast extract-at-ingest and cache for real-time lookups."""

from __future__ import annotations

import uuid

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ai_cowatcher.config import Settings
from ai_cowatcher.db.base import init_database
from ai_cowatcher.retrieval.cast_lookup import (
    CastLookupTool,
    CastRedisCache,
    InMemoryCastRedis,
    actor_names_from_payload,
)
from ai_cowatcher.storage.postgres_store import SceneEventRepository


@pytest.fixture
def repo() -> SceneEventRepository:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    settings = Settings(MOCK_MODE=True)
    init_database(engine=engine, settings=settings)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = factory()
    r = SceneEventRepository(session)
    r.mark_processing("demo", "/tmp/demo.mp4", display_name="Demo Show")
    return r


def test_save_and_get_cast_cache(repo: SceneEventRepository):
    payload = {
        "title": "Demo Show",
        "media_type": "tv",
        "cast": [
            {"actor": "Jane Doe", "character": "Detective Reyes"},
            {"actor": "John Smith", "character": "Bartender"},
        ],
        "source": "ingest",
    }
    repo.save_cast_cache("demo", payload)
    got = repo.get_cast_cache("demo")
    assert got is not None
    assert len(got["cast"]) == 2
    assert actor_names_from_payload(got) == ["Jane Doe", "John Smith"]


def test_lookup_prefers_postgres_cache(repo: SceneEventRepository):
    payload = {
        "title": "Demo Show",
        "cast": [{"actor": "Cached Actor", "character": "Lead"}],
        "source": "ingest",
    }
    repo.save_cast_cache("demo", payload)

    # Transport that would fail if TMDB were called.
    def fail_send(request: httpx.Request) -> httpx.Response:
        raise AssertionError("TMDB should not be called when cache hits")

    transport = httpx.MockTransport(fail_send)
    client = httpx.Client(transport=transport)
    tool = CastLookupTool(
        Settings(MOCK_MODE=False, TMDB_API_KEY="secret"),
        client=client,
        store=repo,
    )
    result = tool.lookup(title_id="demo", title_name="ignored")
    assert "cast" in result
    assert result["cast"][0]["actor"] == "Cached Actor"
    assert result["source"] in ("ingest", "redis")


def test_redis_layers_over_postgres(repo: SceneEventRepository):
    payload = {
        "title": "Demo Show",
        "cast": [{"actor": "Redis Actor", "character": "X"}],
        "source": "ingest",
    }
    repo.save_cast_cache("demo", payload)
    redis_backend = InMemoryCastRedis()
    cache = CastRedisCache(redis_backend, ttl_sec=3600)
    tool = CastLookupTool(
        Settings(MOCK_MODE=True),
        store=repo,
        redis_cache=cache,
    )
    first = tool.lookup(title_id="demo")
    assert first["source"] == "ingest" or first["cast"][0]["actor"] == "Redis Actor"
    # Second hit should be redis once warmed
    second = tool.lookup(title_id="demo")
    assert second.get("source") == "redis"
    assert second["cast"][0]["actor"] == "Redis Actor"


def test_tmdb_fallback_and_write_through(repo: SceneEventRepository):
    def handler(request: httpx.Request) -> httpx.Response:
        if "search/multi" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 99,
                            "media_type": "movie",
                            "title": "Demo Show",
                        }
                    ]
                },
            )
        if "/credits" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "cast": [
                        {"name": "Live Actor", "character": "Hero"},
                        {"name": "Sidekick", "character": "Friend"},
                    ]
                },
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tool = CastLookupTool(
        Settings(MOCK_MODE=False, TMDB_API_KEY="key", TMDB_MAX_RETRIES=1),
        client=client,
        store=repo,
    )
    result = tool.lookup(title_id="demo", title_name="Demo Show")
    assert result["source"] == "tmdb"
    assert result["cast"][0]["actor"] == "Live Actor"
    # Write-through
    cached = repo.get_cast_cache("demo")
    assert cached is not None
    assert cached["cast"][0]["actor"] == "Live Actor"


def test_mock_ingest_extract_on_pipeline_method():
    from ai_cowatcher.ingestion.pipeline import IngestionPipeline

    engine = create_engine("sqlite+pysqlite:///:memory:")
    settings = Settings(MOCK_MODE=True, QA_CACHE_ENABLED=False)
    init_database(engine=engine, settings=settings)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    pipeline = IngestionPipeline(settings=settings, session_factory=factory)
    tid = f"t-{uuid.uuid4().hex[:6]}"
    with factory() as session:
        repo = SceneEventRepository(session)
        repo.mark_processing(tid, "/tmp/x.mp4", display_name="Friends")
        payload = pipeline._extract_and_cache_cast(tid, repo)
        assert payload is not None
        assert len(payload["cast"]) >= 1
        assert repo.get_cast_cache(tid) is not None
