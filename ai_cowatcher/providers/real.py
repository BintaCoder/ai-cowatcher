"""Real AI/AV provider implementations for offline ingestion."""

from __future__ import annotations

import base64
import logging
import subprocess
from pathlib import Path

import cv2
import litellm
import numpy as np
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import SceneBoundary

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

        self._model = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            num_workers=settings.whisper_num_workers,
        )

    def transcribe_window(self, audio_path: str, start_ts: float, end_ts: float) -> str:
        segments, _ = self._model.transcribe(
            audio_path,
            clip_timestamps=(start_ts, end_ts),
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class InsightFaceAnalyzer:
    def __init__(self, settings: Settings):
        from insightface.app import FaceAnalysis

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
        for scene in scenes:
            midpoint = (scene.start_ts + scene.end_ts) / 2.0
            frame = _read_frame_at(video_path, midpoint)
            if frame is None:
                captions.append("")
                continue

            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                captions.append("")
                continue

            image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
            prompt = (
                "Describe this TV scene in one or two concise sentences for a viewer "
                "co-watcher assistant. Focus on visible action, characters, and setting."
            )
            response = litellm.completion(
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
            )
            captions.append(response.choices[0].message.content.strip())
        return captions


class BgeM3Embedder:
    vector_size = 1024

    def __init__(self, settings: Settings):
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(
            settings.embedding_model,
            use_fp16=False,
            device=settings.embedding_device,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        output = self._model.encode(texts, return_dense=True, return_sparse=False)
        dense = output["dense_vecs"]
        return [vector.tolist() for vector in dense]


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
