"""RabbitMQ producer/consumer for smaller-scale deployments."""

from __future__ import annotations

import logging
from typing import Any

from ai_cowatcher.config import Settings
from ai_cowatcher.messaging.base import BrokerMessage, IngestEventConsumer, IngestEventProducer
from ai_cowatcher.messaging.events import IngestTitleEvent

logger = logging.getLogger(__name__)


def _import_pika() -> Any:
    try:
        import pika
    except ImportError as exc:
        raise ImportError(
            "pika is required for RabbitMQ. Install with: pip install 'ai-cowatcher[rabbitmq]'"
        ) from exc
    return pika


class RabbitIngestProducer(IngestEventProducer):
    def __init__(self, settings: Settings) -> None:
        pika = _import_pika()
        self._queue_name = settings.ingest_queue_name
        params = pika.URLParameters(settings.rabbitmq_url)
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.queue_declare(queue=self._queue_name, durable=True)

    def publish(self, event: IngestTitleEvent) -> None:
        self._channel.basic_publish(
            exchange="",
            routing_key=self._queue_name,
            body=event.to_json().encode("utf-8"),
            properties=self._import_pika().BasicProperties(
                delivery_mode=2,
                content_type="application/json",
                message_id=event.event_id,
            ),
        )
        logger.info("Published ingest event %s for title %s", event.event_id, event.title_id)

    @staticmethod
    def _import_pika() -> Any:
        return _import_pika()

    def close(self) -> None:
        if self._connection.is_open:
            self._connection.close()


class RabbitIngestConsumer(IngestEventConsumer):
    def __init__(self, settings: Settings) -> None:
        pika = _import_pika()
        self._queue_name = settings.ingest_queue_name
        params = pika.URLParameters(settings.rabbitmq_url)
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.queue_declare(queue=self._queue_name, durable=True)
        self._channel.basic_qos(prefetch_count=1)
        self._pending: BrokerMessage | None = None

    def poll(self, *, timeout_sec: float = 1.0) -> BrokerMessage | None:
        if self._pending is not None:
            return self._pending
        method, properties, body = self._channel.basic_get(queue=self._queue_name, auto_ack=False)
        if method is None:
            self._connection.sleep(timeout_sec)
            return None
        event = IngestTitleEvent.from_json(body)
        message = BrokerMessage(event=event, delivery=method.delivery_tag)
        self._pending = message
        return message

    def ack(self, message: BrokerMessage) -> None:
        self._channel.basic_ack(delivery_tag=message.delivery)
        if self._pending is message:
            self._pending = None

    def nack(self, message: BrokerMessage, *, requeue: bool = True) -> None:
        self._channel.basic_nack(delivery_tag=message.delivery, requeue=requeue)
        if self._pending is message:
            self._pending = None

    def close(self) -> None:
        if self._connection.is_open:
            self._connection.close()
