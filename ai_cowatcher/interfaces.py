"""Provider interfaces — each has mock and real implementations."""

from __future__ import annotations

from typing import Protocol

from ai_cowatcher.domain import SceneBoundary, SpeakerSegment


class SceneDetector(Protocol):
    def detect_scenes(self, video_path: str) -> list[SceneBoundary]:
        ...


class AudioExtractor(Protocol):
    def extract_audio(self, video_path: str, output_path: str) -> str:
        ...


class Transcriber(Protocol):
    def transcribe_window(self, audio_path: str, start_ts: float, end_ts: float) -> str:
        ...


class SpeakerDiarizer(Protocol):
    def diarize(self, audio_path: str) -> list[SpeakerSegment]:
        ...


class FaceAnalyzer(Protocol):
    def detect_face_clusters(
        self, video_path: str, title_id: str, scene: SceneBoundary
    ) -> list[str]:
        ...


class SceneCaptioner(Protocol):
    def caption_scenes(self, video_path: str, scenes: list[SceneBoundary]) -> list[str]:
        ...


class TextEmbedder(Protocol):
    @property
    def vector_size(self) -> int:
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...
