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
    speaker_cluster_ids: list[str] = field(default_factory=list)

    @property
    def embedding_text(self) -> str:
        return f"{self.transcript}\n\n{self.caption}".strip()


@dataclass(frozen=True)
class SpeakerSegment:
    """A diarized span of audio attributed to a single speaker label."""

    start_ts: float
    end_ts: float
    speaker_label: str


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
    speaker_cluster_ids: tuple[str, ...] = ()
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
            "speaker_cluster_ids": list(self.speaker_cluster_ids),
            "score": self.score,
        }


@dataclass(frozen=True)
class TitleEventRecord:
    """Indexed navigable moment (sports, credits, actor appearance, etc.)."""

    event_id: str
    title_id: str
    event_type: str
    ordinal: int
    start_ts: float
    end_ts: float
    scene_id: str | None
    label: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_tool_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "title_id": self.title_id,
            "event_type": self.event_type,
            "ordinal": self.ordinal,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "scene_id": self.scene_id,
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class NavigateResult:
    """Resolved navigation target for the watch UI."""

    answer: str
    seek_to_ts: float | None
    scene_id: str | None = None
    event_type: str | None = None
    navigation_mode: str = "semantic"


@dataclass(frozen=True)
class CharacterIdentity:
    """A unified character linking face cluster(s) and speaker cluster(s)."""

    character_id: str
    title_id: str
    name: str | None
    face_cluster_ids: tuple[str, ...]
    speaker_cluster_ids: tuple[str, ...]
    first_ts: float

    def to_dict(self) -> dict[str, object]:
        return {
            "character_id": self.character_id,
            "title_id": self.title_id,
            "name": self.name,
            "face_cluster_ids": list(self.face_cluster_ids),
            "speaker_cluster_ids": list(self.speaker_cluster_ids),
            "first_ts": self.first_ts,
        }


@dataclass(frozen=True)
class CharacterAppearance:
    """A timestamped scene appearance for a character."""

    character_id: str
    scene_id: str
    start_ts: float
    end_ts: float

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
        }


@dataclass(frozen=True)
class CharacterRelationship:
    """A timestamped relationship edge between two characters.

    ``known_since_ts`` is the earliest playback position at which this
    relationship is established/revealed on screen. It is the spoiler anchor:
    the relationship must only surface once ``current_ts >= known_since_ts``.
    """

    title_id: str
    source_id: str
    target_id: str
    rel_type: str
    summary: str
    known_since_ts: float
    scene_id: str | None = None
    source_name: str | None = None
    target_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "target_id": self.target_id,
            "target_name": self.target_name,
            "rel_type": self.rel_type,
            "summary": self.summary,
            "known_since_ts": self.known_since_ts,
            "scene_id": self.scene_id,
        }


@dataclass(frozen=True)
class CharacterLookupResult:
    """Spoiler-safe view of a character at the viewer's current timestamp."""

    character: CharacterIdentity
    appearances: tuple[CharacterAppearance, ...]
    relationships: tuple[CharacterRelationship, ...]
    spoiler_filtered: bool = False

    def to_tool_dict(self) -> dict[str, object]:
        return {
            "character": {
                "character_id": self.character.character_id,
                "name": self.character.name,
                "face_cluster_ids": list(self.character.face_cluster_ids),
                "speaker_cluster_ids": list(self.character.speaker_cluster_ids),
            },
            "seen_before": bool(self.appearances),
            "appearance_count": len(self.appearances),
            "appearances": [item.to_dict() for item in self.appearances],
            "relationships": [item.to_dict() for item in self.relationships],
            "spoiler_filtered": self.spoiler_filtered,
        }


@dataclass(frozen=True)
class KnowledgeChunkRecord:
    """A curated, non-spoiler knowledge chunk for RAG retrieval."""

    chunk_id: str
    title_id: str
    category: str
    text: str
    source: str = "curated"

    @property
    def embedding_text(self) -> str:
        return f"{self.category}: {self.text}".strip()


@dataclass(frozen=True)
class KnowledgeSearchHit:
    """A knowledge chunk returned by semantic search (no playback filter)."""

    chunk_id: str
    title_id: str
    category: str
    text: str
    source: str
    score: float = 0.0

    def to_tool_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "title_id": self.title_id,
            "category": self.category,
            "text": self.text,
            "source": self.source,
            "score": self.score,
        }
