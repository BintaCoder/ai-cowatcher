"""Map full-audio transcription segments onto scene windows."""

from __future__ import annotations

from dataclasses import dataclass

from ai_cowatcher.domain import SceneBoundary
from ai_cowatcher.interfaces import Transcriber


@dataclass(frozen=True)
class TranscriptSegment:
    start_ts: float
    end_ts: float
    text: str


def transcripts_for_scenes(
    transcriber: Transcriber,
    audio_path: str,
    scenes: list[SceneBoundary],
) -> list[str]:
    if hasattr(transcriber, "transcribe_full"):
        segments = transcriber.transcribe_full(audio_path)
        return [_text_for_window(segments, scene.start_ts, scene.end_ts) for scene in scenes]

    return [
        transcriber.transcribe_window(audio_path, scene.start_ts, scene.end_ts)
        for scene in scenes
    ]


def _text_for_window(
    segments: list[TranscriptSegment],
    start_ts: float,
    end_ts: float,
) -> str:
    texts = [
        segment.text
        for segment in segments
        if segment.end_ts > start_ts and segment.start_ts < end_ts and segment.text
    ]
    return " ".join(texts).strip()
