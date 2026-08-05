"""SQLAlchemy base and session helpers."""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from ai_cowatcher.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


def create_db_engine(database_url: str | None = None, settings: Settings | None = None):
    settings = settings or get_settings()
    url = database_url or settings.postgres_dsn
    return create_engine(url, pool_pre_ping=True)


def create_session_factory(engine=None, settings: Settings | None = None):
    if engine is None:
        engine = create_db_engine(settings=settings)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_database(engine=None, settings: Settings | None = None) -> None:
    from ai_cowatcher.db import models  # noqa: F401 — register models

    if engine is None:
        engine = create_db_engine(settings=settings)
    Base.metadata.create_all(bind=engine)
    _apply_lightweight_migrations(engine)


def _apply_lightweight_migrations(engine) -> None:
    """Pilot-safe additive migrations (no Alembic yet)."""
    dialect = engine.dialect.name
    with engine.begin() as conn:
        if dialect == "sqlite":
            # SQLite: no IF NOT EXISTS for ADD COLUMN on many versions — try/ignore.
            for stmt in (
                "ALTER TABLE title_ingestions ADD COLUMN display_name VARCHAR(512)",
                "ALTER TABLE title_ingestions ADD COLUMN credits_start_ts FLOAT",
                "ALTER TABLE title_ingestions ADD COLUMN cast_cache TEXT",
                "ALTER TABLE title_ingestions ADD COLUMN cast_cached_at DATETIME",
                "ALTER TABLE scene_events ADD COLUMN speaker_cluster_ids TEXT DEFAULT '[]'",
                "ALTER TABLE scene_events ADD COLUMN audio_object_key VARCHAR(512)",
            ):
                try:
                    conn.execute(text(stmt))
                except Exception:  # noqa: BLE001
                    pass
            return

        conn.execute(
            text(
                "ALTER TABLE title_ingestions "
                "ADD COLUMN IF NOT EXISTS display_name VARCHAR(512)"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE title_ingestions "
                "ADD COLUMN IF NOT EXISTS credits_start_ts DOUBLE PRECISION"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE scene_events "
                "ADD COLUMN IF NOT EXISTS speaker_cluster_ids JSONB DEFAULT '[]'::jsonb"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE scene_events "
                "ADD COLUMN IF NOT EXISTS audio_object_key VARCHAR(512)"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE title_ingestions "
                "ADD COLUMN IF NOT EXISTS cast_cache JSONB"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE title_ingestions "
                "ADD COLUMN IF NOT EXISTS cast_cached_at TIMESTAMPTZ"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE bench_ask_result "
                "ADD COLUMN IF NOT EXISTS persona_id VARCHAR(64) NOT NULL DEFAULT ''"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE bench_ask_result "
                "ADD COLUMN IF NOT EXISTS companion_gender VARCHAR(16) NOT NULL DEFAULT ''"
            )
        )
