"""Catalog / ingestion event payloads."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class IngestTitleEvent:
    """Published when a new title lands in the catalog and needs offline processing."""

    title_id: str
    video_path: str
    force: bool = False
    display_name: str | None = None
    event_id: str = ""
    event_type: str = "title.cataloged"
    attempt: int = 1

    def __post_init__(self) -> None:
        if not self.event_id:
            object.__setattr__(self, "event_id", uuid.uuid4().hex)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | bytes) -> IngestTitleEvent:
        data: dict[str, Any] = json.loads(raw)
        return cls(
            title_id=str(data["title_id"]),
            video_path=str(data["video_path"]),
            force=bool(data.get("force", False)),
            display_name=data.get("display_name"),
            event_id=str(data.get("event_id") or uuid.uuid4().hex),
            event_type=str(data.get("event_type", "title.cataloged")),
            attempt=int(data.get("attempt", 1)),
        )
