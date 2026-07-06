"""Offline once-per-title ingestion pipeline."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.db.base import create_db_engine, init_database
from ai_cowatcher.domain import SceneBoundary, SceneEventRecord
from ai_cowatcher.ingestion.transcription import transcripts_for_scenes
from ai_cowatcher.providers.factory import IngestionProviders, build_ingestion_providers
from ai_cowatcher.providers.litellm_env import configure_litellm_env
from ai_cowatcher.storage.postgres_store import SceneEventRepository
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    title_id: str
    scene_count: int
    skipped: bool = False


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        providers: IngestionProviders | None = None,
        session_factory: sessionmaker | None = None,
        qdrant_store: QdrantSceneStore | None = None,
    ):
        self._settings = settings or get_settings()
        self._providers = providers or build_ingestion_providers(self._settings)
        if session_factory is None:
            engine = create_db_engine(settings=self._settings)
            init_database(engine=engine, settings=self._settings)
            session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        self._session_factory = session_factory
        self._qdrant = qdrant_store or QdrantSceneStore(self._settings)

    def run(self, title_id: str, video_path: str, *, force: bool = False) -> IngestionResult:
        configure_litellm_env(self._settings)
        video = Path(video_path)
        if not video.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        with self._session_factory() as session:
            repo = SceneEventRepository(session)
            if repo.is_completed(title_id) and not force:
                logger.info("Title %s already ingested; skipping", title_id)
                return IngestionResult(
                    title_id=title_id,
                    scene_count=repo.count_scene_events(title_id),
                    skipped=True,
                )

            if force:
                repo.delete_title_data(title_id)
                self._qdrant.delete_title(title_id)

            repo.mark_processing(title_id, str(video))

            try:
                events = self._process_video(title_id, str(video))
                vectors = self._providers.embedder.embed_texts(
                    [event.embedding_text for event in events]
                )
                self._qdrant.ensure_collection(self._providers.embedder.vector_size)
                repo.save_scene_events(events)
                self._qdrant.upsert_scene_events(events, vectors)
                repo.mark_completed(title_id, len(events))
                return IngestionResult(title_id=title_id, scene_count=len(events))
            except Exception as exc:
                logger.exception("Ingestion failed for title %s", title_id)
                repo.mark_failed(title_id, str(exc))
                raise

    def _process_video(self, title_id: str, video_path: str) -> list[SceneEventRecord]:
        scenes = self._providers.scene_detector.detect_scenes(video_path)
        if not scenes:
            raise ValueError("No scenes detected")

        with tempfile.TemporaryDirectory(prefix="cowatcher-audio-") as tmpdir:
            audio_path = str(Path(tmpdir) / "title_audio.wav")
            self._providers.audio_extractor.extract_audio(video_path, audio_path)
            transcripts = transcripts_for_scenes(
                self._providers.transcriber,
                audio_path,
                scenes,
            )

        face_clusters = [
            self._providers.face_analyzer.detect_face_clusters(video_path, title_id, scene)
            for scene in scenes
        ]
        captions = self._providers.captioner.caption_scenes(video_path, scenes)

        return [
            _build_scene_event(title_id, scene, transcript, caption, clusters)
            for scene, transcript, caption, clusters in zip(
                scenes, transcripts, captions, face_clusters, strict=True
            )
        ]


def _build_scene_event(
    title_id: str,
    scene: SceneBoundary,
    transcript: str,
    caption: str,
    face_cluster_ids: list[str],
) -> SceneEventRecord:
    return SceneEventRecord(
        scene_id=scene.scene_id,
        title_id=title_id,
        start_ts=scene.start_ts,
        end_ts=scene.end_ts,
        transcript=transcript,
        caption=caption,
        face_cluster_ids=face_cluster_ids,
    )


def run_ingestion(title_id: str, video_path: str, *, force: bool = False) -> IngestionResult:
    return IngestionPipeline().run(title_id, video_path, force=force)
