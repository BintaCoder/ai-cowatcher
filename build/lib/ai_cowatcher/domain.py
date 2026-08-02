"""Shared domain types for offline ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SceneBoundary:
    """A detected scene window within a title."""

    index: int
    start_ts: float
    end_ts: float

    @property
    def scene_id(self) -> str:
        return f"s{self.index:04d}"


@dataclass
class SceneEventRecord:
    """Fully enriched scene ready for persistence."""

    scene_id: str
    title_id: str
    start_ts: float
    end_ts: float
    transcript: str
    caption: str
    face_cluster_ids: list[str] = field(default_factory=list)

    @property
    def embedding_text(self) -> str:
        return f"{self.transcript}\n\n{self.caption}".strip()


@dataclass(frozen=True)
class SceneLookupHit:
    """A scene returned by semantic retrieval (spoiler-safe window)."""

    scene_id: str
    title_id: str
    start_ts: float
    end_ts: float
    transcript: str
    caption: str
    face_cluster_ids: tuple[str, ...] = ()
    score: float = 0.0

    def to_tool_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "title_id": self.title_id,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "transcript": self.transcript,
            "caption": self.caption,
            "face_cluster_ids": list(self.face_cluster_ids),
            "score": self.score,
        }
