"""Mock AI/AV providers for local development and tests."""

from __future__ import annotations

import hashlib

from ai_cowatcher.domain import SceneBoundary


class MockSceneDetector:
    def detect_scenes(self, video_path: str) -> list[SceneBoundary]:
        return [
            SceneBoundary(index=0, start_ts=0.0, end_ts=12.5),
            SceneBoundary(index=1, start_ts=12.5, end_ts=28.0),
            SceneBoundary(index=2, start_ts=28.0, end_ts=45.0),
        ]


class MockAudioExtractor:
    def extract_audio(self, video_path: str, output_path: str) -> str:
        with open(output_path, "wb") as handle:
            handle.write(b"mock-audio")
        return output_path


class MockTranscriber:
    def transcribe_window(self, audio_path: str, start_ts: float, end_ts: float) -> str:
        return f"Mock transcript from {start_ts:.1f}s to {end_ts:.1f}s."


class MockFaceAnalyzer:
    def detect_face_clusters(
        self, video_path: str, title_id: str, scene: SceneBoundary
    ) -> list[str]:
        if scene.index % 2 == 0:
            return [f"{title_id}-fc-0", f"{title_id}-fc-1"]
        return [f"{title_id}-fc-0"]


class MockSceneCaptioner:
    def caption_scenes(self, video_path: str, scenes: list[SceneBoundary]) -> list[str]:
        return [
            f"Mock caption for scene {scene.index} ({scene.start_ts:.1f}-{scene.end_ts:.1f}s)."
            for scene in scenes
        ]


class MockTextEmbedder:
    vector_size = 1024

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = [
                ((digest[index % len(digest)] / 255.0) * 2.0) - 1.0
                for index in range(self.vector_size)
            ]
            vectors.append(values)
        return vectors
