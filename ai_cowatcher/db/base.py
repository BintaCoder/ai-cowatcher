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
        if dialect == "sqlite":
            # SQLite lacks IF NOT EXISTS for ADD COLUMN on older versions; ignore duplicates.
            try:
                conn.execute(
                    text(
                        "ALTER TABLE scene_events "
                        "ADD COLUMN speaker_cluster_ids TEXT DEFAULT '[]'"
                    )
                )
            except Exception:  # noqa: BLE001
                pass
            try:
                conn.execute(
                    text(
                        "ALTER TABLE scene_events "
                        "ADD COLUMN audio_object_key VARCHAR(512)"
                    )
                )
            except Exception:  # noqa: BLE001
                pass
        else:
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
