"""Lazy singleton ingest event producer for the API process."""

from __future__ import annotations

from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.messaging.base import IngestEventProducer
from ai_cowatcher.messaging.factory import build_ingest_producer

_producer: IngestEventProducer | None = None


def get_ingest_producer(settings: Settings | None = None) -> IngestEventProducer:
    global _producer
    if _producer is None:
        _producer = build_ingest_producer(settings or get_settings())
    return _producer


def reset_ingest_producer() -> None:
    """Close and clear the cached producer (tests only)."""
    global _producer
    if _producer is not None:
        _producer.close()
    _producer = None
