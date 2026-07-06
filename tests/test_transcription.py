"""Tests for single-pass transcription slicing."""

from __future__ import annotations

from ai_cowatcher.domain import SceneBoundary
from ai_cowatcher.ingestion.transcription import TranscriptSegment, transcripts_for_scenes


class _FakeTranscriber:
    def transcribe_full(self, audio_path: str) -> list[TranscriptSegment]:
        del audio_path
        return [
            TranscriptSegment(0.0, 5.0, "Hello there."),
            TranscriptSegment(5.0, 12.0, "Something happens next."),
            TranscriptSegment(12.0, 20.0, "The ending."),
        ]


def test_transcripts_for_scenes_slices_full_transcription_once():
    scenes = [
        SceneBoundary(index=0, start_ts=0.0, end_ts=6.0),
        SceneBoundary(index=1, start_ts=6.0, end_ts=15.0),
    ]
    transcripts = transcripts_for_scenes(_FakeTranscriber(), "audio.wav", scenes)
    assert "Hello there" in transcripts[0]
    assert "Something happens" in transcripts[1]
