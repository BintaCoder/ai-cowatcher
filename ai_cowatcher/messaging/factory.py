"""Build messaging adapters from application settings."""

from __future__ import annotations

from ai_cowatcher.config import Settings
from ai_cowatcher.messaging.base import IngestEventConsumer, IngestEventProducer
from ai_cowatcher.messaging.kafka import KafkaIngestConsumer, KafkaIngestProducer
from ai_cowatcher.messaging.memory import InMemoryIngestConsumer, InMemoryIngestProducer
from ai_cowatcher.messaging.rabbitmq import RabbitIngestConsumer, RabbitIngestProducer


def build_ingest_producer(settings: Settings) -> IngestEventProducer:
    broker = settings.message_broker.lower()
    if broker == "kafka":
        return KafkaIngestProducer(settings)
    if broker == "rabbitmq":
        return RabbitIngestProducer(settings)
    if broker == "memory":
        return InMemoryIngestProducer()
    raise ValueError(f"Unsupported message_broker: {settings.message_broker}")


def build_ingest_consumer(settings: Settings) -> IngestEventConsumer:
    broker = settings.message_broker.lower()
    if broker == "kafka":
        return KafkaIngestConsumer(settings)
    if broker == "rabbitmq":
        return RabbitIngestConsumer(settings)
    if broker == "memory":
        return InMemoryIngestConsumer()
    raise ValueError(f"Unsupported message_broker: {settings.message_broker}")
