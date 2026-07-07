"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ai_cowatcher import __version__
from ai_cowatcher.api.ask_routes import router as ask_router
from ai_cowatcher.api.watch_routes import router as watch_router
from ai_cowatcher.api.metrics_routes import router as metrics_router
from ai_cowatcher.api.routes import router as ingest_router
from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.db.base import init_database
from ai_cowatcher.agent.metrics import conversation_tier_counts, metrics_lite_summary
from ai_cowatcher.health import collect_dependency_health, overall_status
from ai_cowatcher.providers.litellm_env import configure_litellm_env

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_litellm_env(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        init_database(settings=settings)
        yield

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Pay-TV co-watcher pilot API",
        lifespan=lifespan,
    )

    app.include_router(ingest_router)
    app.include_router(ask_router)
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
