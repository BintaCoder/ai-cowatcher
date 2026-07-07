"""Event-driven ingestion messaging (Kafka / RabbitMQ / in-memory)."""

from ai_cowatcher.messaging.events import IngestTitleEvent
from ai_cowatcher.messaging.factory import build_ingest_consumer, build_ingest_producer

__all__ = [
    "IngestTitleEvent",
    "build_ingest_consumer",
    "build_ingest_producer",
]
