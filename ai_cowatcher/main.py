"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy.orm import sessionmaker

from ai_cowatcher import __version__
from ai_cowatcher.api.catalog_routes import router as catalog_router
from ai_cowatcher.api.ask_routes import router as ask_router
from ai_cowatcher.api.navigate_routes import router as navigate_router
from ai_cowatcher.api.watch_routes import router as watch_router
from ai_cowatcher.api.metrics_routes import router as metrics_router
from ai_cowatcher.api.routes import router as ingest_router
from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.db.base import create_db_engine, init_database
from ai_cowatcher.agent.metrics import conversation_tier_counts, metrics_lite_summary
from ai_cowatcher.health import collect_dependency_health, overall_status
from ai_cowatcher.providers.litellm_env import configure_litellm_env
from ai_cowatcher.realtime.navigation_session import build_navigation_session
from ai_cowatcher.realtime.viewing_session import _build_embedder, build_viewing_session
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_litellm_env(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # One engine + embedder + Qdrant client shared by /ask and /navigate.
        engine = create_db_engine(settings=settings)
        init_database(engine=engine, settings=settings)
        session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        embedder = _build_embedder(settings)
        qdrant = QdrantSceneStore(settings)

        logger.info("Warming real-time viewing + navigation sessions (shared embedder)")
        settings.validate_pilot_latency_config()
        app.state.viewing_session = build_viewing_session(
            settings,
            session_factory=session_factory,
            embedder=embedder,
            qdrant_store=qdrant,
        )
        app.state.navigation_session = build_navigation_session(
            settings,
            session_factory=session_factory,
            embedder=embedder,
            qdrant_store=qdrant,
        )
        yield

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Pay-TV co-watcher pilot API",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.include_router(catalog_router)
    app.include_router(ingest_router)
    app.include_router(ask_router)
    app.include_router(navigate_router)
    app.include_router(watch_router)
    app.include_router(metrics_router)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "service": settings.app_name,
            "version": __version__,
            "status": "running",
        }

    @app.get("/health")
    async def health() -> JSONResponse:
        dependencies = collect_dependency_health(settings)
        status = overall_status(dependencies)
        body = {
            "status": status,
            "mock_mode": settings.mock_mode,
            "app_env": settings.app_env,
            "dependencies": dependencies,
            "latency_path": {
                "pilot_low_latency": settings.pilot_low_latency,
                "utterance_gate_strategy": settings.utterance_gate_strategy,
                "legacy_multi_tool_reachable": settings.legacy_multi_tool_path_reachable(),
                "evidence_max_scenes": settings.evidence_max_scenes,
                "evidence_max_chars_per_field": settings.evidence_max_chars_per_field,
                "qa_cache_enabled": settings.qa_cache_enabled,
                "qa_cache_semantic_threshold": settings.qa_cache_semantic_threshold,
                "session_cost_budget_usd": settings.session_cost_budget_usd,
            },
            "llm": {
                "active_model": settings.active_llm_model,
                "tier_fast_model": settings.conversation_fast_model,
                "tier_escalated_model": settings.conversation_escalated_model,
                "escalation_strategy": settings.llm_escalation_strategy,
                "escalation_min_chars": settings.llm_escalation_min_chars,
                "primary_model": settings.llm_primary_model,
                "fallback_model": settings.llm_fallback_model,
                "tier_counts": conversation_tier_counts(),
            },
            "metrics_lite": metrics_lite_summary(),
            "vision_model": settings.active_vision_model,
            "whisper": {
                "model_size": settings.whisper_model_size,
                "compute_type": settings.whisper_compute_type,
                "device": settings.whisper_device,
            },
        }
        return JSONResponse(status_code=200 if status == "ok" else 503, content=body)

    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    uvicorn.run(
        "ai_cowatcher.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "development",
    )


if __name__ == "__main__":
    run()
