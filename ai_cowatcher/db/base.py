"""SQLAlchemy base and session helpers."""

from __future__ import annotations

from sqlalchemy import create_engine
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
