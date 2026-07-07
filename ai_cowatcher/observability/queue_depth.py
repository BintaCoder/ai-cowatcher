"""Probe message-broker queue depth for Prometheus gauges."""

from __future__ import annotations

import logging

from ai_cowatcher.config import Settings

logger = logging.getLogger(__name__)


def probe_ingest_queue_depth(settings: Settings) -> int | None:
    """Return approximate pending ingest events, or None if unavailable."""
    broker = settings.message_broker.lower()
    if broker == "memory":
        from ai_cowatcher.messaging.memory import _get_shared_queue

        return _get_shared_queue().qsize()
    if broker == "rabbitmq":
        return _rabbitmq_queue_depth(settings)
    if broker == "kafka":
        return _kafka_consumer_lag(settings)
    return None


def _rabbitmq_queue_depth(settings: Settings) -> int | None:
    try:
        import pika
    except ImportError:
        return None
    try:
        params = pika.URLParameters(settings.rabbitmq_url)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        result = channel.queue_declare(queue=settings.ingest_queue_name, passive=True)
        depth = int(result.method.message_count)
        connection.close()
        return depth
    except Exception:
        logger.debug("Could not probe RabbitMQ queue depth", exc_info=True)
        return None


def _kafka_consumer_lag(settings: Settings) -> int | None:
    """Best-effort lag estimate for the ingest consumer group."""
    try:
        from kafka import KafkaConsumer, TopicPartition
    except ImportError:
        return None
    consumer = None
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=settings.kafka_bootstrap_servers.split(","),
            group_id=settings.kafka_consumer_group,
            enable_auto_commit=False,
        )
        partitions = consumer.partitions_for_topic(settings.ingest_topic)
        if not partitions:
            return 0
        topic_partitions = [TopicPartition(settings.ingest_topic, p) for p in partitions]
        end_offsets = consumer.end_offsets(topic_partitions)
        lag = 0
        for tp in topic_partitions:
            end = end_offsets.get(tp, 0)
            pos = consumer.committed(tp)
            if pos is None:
                pos = 0
            lag += max(0, end - pos)
        return lag
    except Exception:
        logger.debug("Could not probe Kafka consumer lag", exc_info=True)
        return None
    finally:
        if consumer is not None:
            consumer.close()
