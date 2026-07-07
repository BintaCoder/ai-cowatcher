"""Real AI/AV provider implementations for offline ingestion."""

from __future__ import annotations

import base64
import logging
import platform
import subprocess
import time
from pathlib import Path

import cv2
import litellm
import numpy as np
from litellm.exceptions import RateLimitError
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import SceneBoundary, SpeakerSegment
from ai_cowatcher.ingestion.transcription import TranscriptSegment

logger = logging.getLogger(__name__)


class PySceneDetectDetector:
    def detect_scenes(self, video_path: str) -> list[SceneBoundary]:
        video = open_video(video_path)
        manager = SceneManager()
        manager.add_detector(ContentDetector())
        manager.detect_scenes(video)
        scene_list = manager.get_scene_list()
        if not scene_list:
            duration = video.duration.get_seconds() if video.duration else 0.0
            return [SceneBoundary(index=0, start_ts=0.0, end_ts=max(duration, 0.1))]

        boundaries: list[SceneBoundary] = []
        for index, (start, end) in enumerate(scene_list):
            boundaries.append(
                SceneBoundary(
                    index=index,
                    start_ts=start.get_seconds(),
                    end_ts=end.get_seconds(),
                )
            )
        return boundaries


class FFmpegAudioExtractor:
    def __init__(self, settings: Settings):
        self._ffmpeg = settings.ffmpeg_bin

    def extract_audio(self, video_path: str, output_path: str) -> str:
        cmd = [
            self._ffmpeg,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path


class FasterWhisperTranscriber:
    def __init__(self, settings: Settings):
        from faster_whisper import WhisperModel

        self._device = settings.whisper_device
        logger.info(
            "Loading Whisper model=%s device=%s compute_type=%s",
            settings.whisper_model_size,
            settings.whisper_device,
            settings.whisper_compute_type,
        )
        self._model = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            num_workers=settings.whisper_num_workers,
        )

    def transcribe_full(self, audio_path: str) -> list[TranscriptSegment]:
        logger.info("Transcribing full audio once (device=%s)", self._device)
        segments, info = self._model.transcribe(audio_path)
        if info.language:
            logger.info(
                "Whisper language=%s probability=%.2f",
                info.language,
                info.language_probability or 0.0,
            )
        return [
            TranscriptSegment(
                start_ts=float(segment.start),
                end_ts=float(segment.end),
                text=segment.text.strip(),
            )
            for segment in segments
            if segment.text.strip()
        ]

    def transcribe_window(self, audio_path: str, start_ts: float, end_ts: float) -> str:
        segments, _ = self._model.transcribe(
            audio_path,
            clip_timestamps=(start_ts, end_ts),
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class PyannoteDiarizer:
    """Speaker diarization via pyannote.audio (offline pipeline only).

    Loaded lazily because it pulls in torch. Requires a Hugging Face access
    token with access to the gated diarization model.
    """

    def __init__(self, settings: Settings):
        from pyannote.audio import Pipeline

        token = settings.huggingface_token or None
        logger.info("Loading pyannote diarization pipeline=%s", settings.diarization_model)
        self._pipeline = Pipeline.from_pretrained(
            settings.diarization_model,
            use_auth_token=token,
        )

    def diarize(self, audio_path: str) -> list[SpeakerSegment]:
        annotation = self._pipeline(audio_path)
        segments: list[SpeakerSegment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                SpeakerSegment(
                    start_ts=float(turn.start),
                    end_ts=float(turn.end),
                    speaker_label=str(speaker),
                )
            )
        segments.sort(key=lambda seg: (seg.start_ts, seg.end_ts))
        return segments


class InsightFaceAnalyzer:
    def __init__(self, settings: Settings):
        from insightface.app import FaceAnalysis

        providers = _insightface_providers()
        if providers:
            logger.info("InsightFace ONNX providers=%s", providers)
            self._app = FaceAnalysis(
                name=settings.insightface_model,
                providers=providers,
            )
        else:
            self._app = FaceAnalysis(name=settings.insightface_model)
        self._app.prepare(ctx_id=settings.insightface_ctx_id, det_size=(640, 640))
        self._similarity_threshold = 0.45

    def detect_face_clusters(
        self, video_path: str, title_id: str, scene: SceneBoundary
    ) -> list[str]:
        midpoint = (scene.start_ts + scene.end_ts) / 2.0
        frame = _read_frame_at(video_path, midpoint)
        if frame is None:
            return []

        faces = self._app.get(frame)
        if not faces:
            return []

        cluster_centroids: list[np.ndarray] = []
        cluster_ids: list[str] = []
        next_cluster = 0

        for face in faces:
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            assigned = False
            for idx, centroid in enumerate(cluster_centroids):
                if float(np.dot(embedding, centroid)) >= self._similarity_threshold:
                    cluster_ids.append(f"{title_id}-fc-{idx}")
                    assigned = True
                    break
            if not assigned:
                cluster_centroids.append(embedding)
                cluster_ids.append(f"{title_id}-fc-{next_cluster}")
                next_cluster += 1

        return sorted(set(cluster_ids))


class LiteLLMSceneCaptioner:
    """Offline-only batched scene captioning. Not exposed on the realtime API."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def caption_scenes(self, video_path: str, scenes: list[SceneBoundary]) -> list[str]:
        captions: list[str] = []
        total = len(scenes)
        for index, scene in enumerate(scenes):
            midpoint = (scene.start_ts + scene.end_ts) / 2.0
            frame = _read_frame_at(video_path, midpoint)
            if frame is None:
                captions.append("")
                continue

            frame = _resize_frame_for_vision(frame, self._settings.vision_frame_max_size)
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                captions.append("")
                continue

            image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
            prompt = (
                "Describe this TV scene in one or two concise sentences for a viewer "
                "co-watcher assistant. Focus on visible action, characters, and setting."
            )
            logger.info(
                "Captioning scene %s (%d/%d) via %s",
                scene.scene_id,
                index + 1,
                total,
                self._settings.active_vision_model,
            )
            response = _vision_completion_with_retry(
                model=self._settings.active_vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=self._settings.vision_max_tokens,
                max_retries=self._settings.vision_caption_max_retries,
            )
            captions.append(response.choices[0].message.content.strip())
            if index + 1 < total and self._settings.vision_caption_delay_sec > 0:
                time.sleep(self._settings.vision_caption_delay_sec)
        return captions


class BgeM3Embedder:
    vector_size = 1024

    def __init__(self, settings: Settings):
        from FlagEmbedding import BGEM3FlagModel

        use_fp16 = settings.embedding_device in ("cuda", "mps")
        logger.info(
            "Loading BGE-M3 model=%s device=%s use_fp16=%s",
            settings.embedding_model,
            settings.embedding_device,
            use_fp16,
        )
        self._model = BGEM3FlagModel(
            settings.embedding_model,
            use_fp16=use_fp16,
            device=settings.embedding_device,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        output = self._model.encode(texts, return_dense=True, return_sparse=False)
        dense = output["dense_vecs"]
        return [vector.tolist() for vector in dense]


def _resize_frame_for_vision(frame, max_size: int):
    height, width = frame.shape[:2]
    if max(height, width) <= max_size:
        return frame
    scale = max_size / max(height, width)
    new_width, new_height = int(width * scale), int(height * scale)
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _vision_completion_with_retry(
    *,
    model: str,
    messages: list,
    max_tokens: int,
    max_retries: int,
):
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return litellm.completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
        except RateLimitError as exc:
            last_error = exc
            wait_sec = min(60.0, 1.0 * (2**attempt))
            logger.warning(
                "Vision caption rate limited (attempt %d/%d), sleeping %.1fs",
                attempt + 1,
                max_retries,
                wait_sec,
            )
            time.sleep(wait_sec)
    assert last_error is not None
    raise last_error


def _insightface_providers() -> list[str] | None:
    if platform.system() != "Darwin":
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    available = ort.get_available_providers()
    if "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return None


def _read_frame_at(video_path: str, timestamp_s: float):
    capture = cv2.VideoCapture(video_path)
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
        ok, frame = capture.read()
        if ok:
            return frame
        return None
    finally:
        capture.release()
