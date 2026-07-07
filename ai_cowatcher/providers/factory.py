"""Select mock or real provider implementations from settings."""

from __future__ import annotations

from dataclasses import dataclass

from ai_cowatcher.config import Settings
from ai_cowatcher.interfaces import (
    AudioExtractor,
    FaceAnalyzer,
    SceneCaptioner,
    SceneDetector,
    SpeakerDiarizer,
    TextEmbedder,
    Transcriber,
)
from ai_cowatcher.providers import mock, real


@dataclass(frozen=True)
class IngestionProviders:
    scene_detector: SceneDetector
    audio_extractor: AudioExtractor
    transcriber: Transcriber
    speaker_diarizer: SpeakerDiarizer
    face_analyzer: FaceAnalyzer
    captioner: SceneCaptioner
    embedder: TextEmbedder


def build_ingestion_providers(settings: Settings) -> IngestionProviders:
    if settings.mock_mode:
        return IngestionProviders(
            scene_detector=mock.MockSceneDetector(),
            audio_extractor=mock.MockAudioExtractor(),
            transcriber=mock.MockTranscriber(),
            speaker_diarizer=mock.MockSpeakerDiarizer(),
            face_analyzer=mock.MockFaceAnalyzer(),
            captioner=mock.MockSceneCaptioner(),
            embedder=mock.MockTextEmbedder(),
        )

    return IngestionProviders(
        scene_detector=real.PySceneDetectDetector(),
        audio_extractor=real.FFmpegAudioExtractor(settings),
        transcriber=real.FasterWhisperTranscriber(settings),
        speaker_diarizer=real.PyannoteDiarizer(settings),
        face_analyzer=real.InsightFaceAnalyzer(settings),
        captioner=real.LiteLLMSceneCaptioner(settings),
        embedder=real.BgeM3Embedder(settings),
    )
