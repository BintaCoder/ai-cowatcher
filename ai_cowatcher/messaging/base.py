"""Shared producer/consumer interfaces for ingestion events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ai_cowatcher.messaging.events import IngestTitleEvent


@dataclass
class BrokerMessage:
    """Broker-agnostic consumed message wrapper."""

    event: IngestTitleEvent
    delivery: Any


class IngestEventProducer(Protocol):
    def publish(self, event: IngestTitleEvent) -> None:
        ...

    def close(self) -> None:
        ...


class IngestEventConsumer(Protocol):
    def poll(self, *, timeout_sec: float = 1.0) -> BrokerMessage | None:
        ...

    def ack(self, message: BrokerMessage) -> None:
        ...

    def nack(self, message: BrokerMessage, *, requeue: bool = True) -> None:
        ...

    def close(self) -> None:
        ...
