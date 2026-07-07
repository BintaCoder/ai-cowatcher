"""In-process queue for tests and local mock-mode development."""

from __future__ import annotations

import queue
import threading
from typing import Any

from ai_cowatcher.messaging.base import BrokerMessage, IngestEventConsumer, IngestEventProducer
from ai_cowatcher.messaging.events import IngestTitleEvent

_shared_queue: queue.Queue[IngestTitleEvent] | None = None
_queue_lock = threading.Lock()


def _get_shared_queue() -> queue.Queue[IngestTitleEvent]:
    global _shared_queue
    with _queue_lock:
        if _shared_queue is None:
            _shared_queue = queue.Queue()
        return _shared_queue


def reset_shared_queue() -> None:
    """Clear the shared in-memory queue (tests only)."""
    global _shared_queue
    with _queue_lock:
        _shared_queue = queue.Queue()


class InMemoryIngestProducer(IngestEventProducer):
    def __init__(self, *, q: queue.Queue[IngestTitleEvent] | None = None) -> None:
        self._queue = q or _get_shared_queue()

    def publish(self, event: IngestTitleEvent) -> None:
        self._queue.put(event)

    def close(self) -> None:
        return None


class InMemoryIngestConsumer(IngestEventConsumer):
    def __init__(self, *, q: queue.Queue[IngestTitleEvent] | None = None) -> None:
        self._queue = q or _get_shared_queue()
        self._pending: dict[str, BrokerMessage] = {}

    def poll(self, *, timeout_sec: float = 1.0) -> BrokerMessage | None:
        try:
            event = self._queue.get(timeout=timeout_sec)
        except queue.Empty:
            return None
        message = BrokerMessage(event=event, delivery=event.event_id)
        self._pending[event.event_id] = message
        return message

    def ack(self, message: BrokerMessage) -> None:
        self._pending.pop(message.event.event_id, None)

    def nack(self, message: BrokerMessage, *, requeue: bool = True) -> None:
        self._pending.pop(message.event.event_id, None)
        if requeue:
            retry = IngestTitleEvent(
                title_id=message.event.title_id,
                video_path=message.event.video_path,
                force=message.event.force,
                display_name=message.event.display_name,
                event_id=message.event.event_id,
                event_type=message.event.event_type,
                attempt=message.event.attempt + 1,
            )
            self._queue.put(retry)

    def close(self) -> None:
        return None
