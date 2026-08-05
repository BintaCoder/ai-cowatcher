"""Select mock or real provider implementations from settings."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionProviders:
    scene_detector: SceneDetector
    audio_extractor: AudioExtractor
    transcriber: Transcriber
    speaker_diarizer: SpeakerDiarizer
    face_analyzer: FaceAnalyzer
    captioner: SceneCaptioner
    embedder: TextEmbedder


def _build_speaker_diarizer(settings: Settings) -> SpeakerDiarizer:
    """Prefer pyannote when installed and enabled; otherwise no-op (ingest still works)."""
    if not getattr(settings, "diarization_enabled", True):
        logger.info("Speaker diarization disabled (DIARIZATION_ENABLED=false)")
        return real.NoOpSpeakerDiarizer()
    try:
        return real.PyannoteDiarizer(settings)
    except ImportError as exc:
        logger.warning(
            "Speaker diarization unavailable (%s). "
            "Continuing without speaker clusters. "
            "Optional install: pip install '.[diarization]' (+ HUGGINGFACE_TOKEN).",
            exc,
        )
        return real.NoOpSpeakerDiarizer()
    except Exception:  # noqa: BLE001 — model/HF/token issues should not hard-fail ingest init
        logger.exception(
            "Failed to load pyannote diarizer; continuing without speaker clusters"
        )
        return real.NoOpSpeakerDiarizer()


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
        speaker_diarizer=_build_speaker_diarizer(settings),
        face_analyzer=real.InsightFaceAnalyzer(settings),
        captioner=real.LiteLLMSceneCaptioner(settings),
        embedder=real.BgeM3Embedder(settings),
    )
