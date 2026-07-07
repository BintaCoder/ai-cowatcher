"""Map speaker-diarization segments onto scene windows.

Runs alongside face clustering: for each scene we record which speaker
clusters were active, mirroring how ``face_cluster_ids`` records which faces
were on screen. These per-scene face+speaker signals are what the offline
character-graph enrichment later links into unified character identities.
"""

from __future__ import annotations

from ai_cowatcher.domain import SceneBoundary, SpeakerSegment
from ai_cowatcher.interfaces import SpeakerDiarizer


def speaker_cluster_id(title_id: str, speaker_label: str) -> str:
    """Stable, title-scoped id for a diarized speaker, e.g. ``demo-sc-SPEAKER_00``."""
    normalized = speaker_label.strip().replace(" ", "_") or "UNKNOWN"
    return f"{title_id}-sc-{normalized}"


def diarize_title(diarizer: SpeakerDiarizer, audio_path: str) -> list[SpeakerSegment]:
    return diarizer.diarize(audio_path)


def speaker_clusters_for_scenes(
    segments: list[SpeakerSegment],
    scenes: list[SceneBoundary],
    title_id: str,
) -> list[list[str]]:
    """Return, per scene, the sorted list of speaker-cluster ids active in it."""
    return [
        _clusters_for_window(segments, scene.start_ts, scene.end_ts, title_id)
        for scene in scenes
    ]


def _clusters_for_window(
    segments: list[SpeakerSegment],
    start_ts: float,
    end_ts: float,
    title_id: str,
) -> list[str]:
    labels: set[str] = set()
    for segment in segments:
        if segment.end_ts > start_ts and segment.start_ts < end_ts:
            labels.add(speaker_cluster_id(title_id, segment.speaker_label))
    return sorted(labels)
