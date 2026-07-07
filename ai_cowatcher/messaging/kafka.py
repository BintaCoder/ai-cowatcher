"""Kafka producer/consumer for larger-scale deployments."""

from __future__ import annotations

import logging
from typing import Any

from ai_cowatcher.config import Settings
from ai_cowatcher.messaging.base import BrokerMessage, IngestEventConsumer, IngestEventProducer
from ai_cowatcher.messaging.events import IngestTitleEvent

logger = logging.getLogger(__name__)


def _import_kafka() -> Any:
    try:
        from kafka import KafkaConsumer, KafkaProducer
    except ImportError as exc:
        raise ImportError(
            "kafka-python is required for Kafka. Install with: pip install 'ai-cowatcher[kafka]'"
        ) from exc
    return KafkaConsumer, KafkaProducer


class KafkaIngestProducer(IngestEventProducer):
    def __init__(self, settings: Settings) -> None:
        _, KafkaProducer = _import_kafka()
        self._topic = settings.ingest_topic
        self._producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
            acks="all",
            retries=3,
        )

    def publish(self, event: IngestTitleEvent) -> None:
        future = self._producer.send(
            self._topic,
            key=event.title_id.encode("utf-8"),
            value=event.to_json().encode("utf-8"),
            headers=[("event_id", event.event_id.encode("utf-8"))],
        )
        future.get(timeout=30)
        logger.info("Published ingest event %s for title %s", event.event_id, event.title_id)

    def close(self) -> None:
        self._producer.flush()
        self._producer.close()


class KafkaIngestConsumer(IngestEventConsumer):
    def __init__(self, settings: Settings) -> None:
        KafkaConsumer, _ = _import_kafka()
        self._consumer = KafkaConsumer(
            settings.ingest_topic,
            bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
            group_id=settings.kafka_consumer_group,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            consumer_timeout_ms=1000,
        )

    def poll(self, *, timeout_sec: float = 1.0) -> BrokerMessage | None:
        records = self._consumer.poll(timeout_ms=int(timeout_sec * 1000))
        if not records:
            return None
        for _tp, messages in records.items():
            if not messages:
                continue
            record = messages[0]
            event = IngestTitleEvent.from_json(record.value)
            return BrokerMessage(event=event, delivery=record)
        return None

    def ack(self, message: BrokerMessage) -> None:
        self._consumer.commit()

    def nack(self, message: BrokerMessage, *, requeue: bool = True) -> None:
        # Kafka redelivery is implicit when offset is not committed.
        if not requeue:
            self._consumer.commit()

    def close(self) -> None:
        self._consumer.close()
