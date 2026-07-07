"""Offline once-per-title ingestion pipeline."""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.db.base import create_db_engine, init_database
from ai_cowatcher.domain import SceneBoundary, SceneEventRecord
from ai_cowatcher.ingestion.diarization import diarize_title, speaker_clusters_for_scenes
from ai_cowatcher.ingestion.event_detection import build_title_events
from ai_cowatcher.ingestion.knowledge_index import index_title_knowledge
from ai_cowatcher.ingestion.transcription import transcripts_for_scenes
from ai_cowatcher.providers.factory import IngestionProviders, build_ingestion_providers
from ai_cowatcher.providers.litellm_env import configure_litellm_env
from ai_cowatcher.retrieval.cast_lookup import CastLookupTool
from ai_cowatcher.storage.postgres_store import SceneEventRepository
from ai_cowatcher.storage.qdrant_knowledge_store import QdrantKnowledgeStore
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    title_id: str
    scene_count: int
    skipped: bool = False
    resumed: bool = False
    newly_processed: int = 0


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

    def run(
        self,
        title_id: str,
        video_path: str,
        *,
        force: bool = False,
        display_name: str | None = None,
    ) -> IngestionResult:
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

            existing_scene_ids = repo.get_existing_scene_ids(title_id)
            resuming = bool(existing_scene_ids)
            if resuming:
                logger.info(
                    "Resuming ingest for title %s; %d scenes already persisted",
                    title_id,
                    len(existing_scene_ids),
                )

            repo.mark_processing(title_id, str(video), display_name=display_name)

            try:
                newly_processed = self._process_video(
                    title_id, str(video), repo, existing_scene_ids
                )
                self._index_navigation_events(title_id, repo)
                self._index_character_graph(title_id, repo)
                self._index_title_knowledge(title_id)
                total = repo.count_scene_events(title_id)
                repo.mark_completed(title_id, total)
                return IngestionResult(
                    title_id=title_id,
                    scene_count=total,
                    resumed=resuming,
                    newly_processed=newly_processed,
                )
            except Exception as exc:
                logger.exception("Ingestion failed for title %s", title_id)
                repo.mark_failed(title_id, str(exc))
                raise

    def _process_video(
        self,
        title_id: str,
        video_path: str,
        repo: SceneEventRepository,
        existing_scene_ids: set[str],
    ) -> int:
        scenes = self._providers.scene_detector.detect_scenes(video_path)
        if not scenes:
            raise ValueError("No scenes detected")

        pending = [scene for scene in scenes if scene.scene_id not in existing_scene_ids]
        if not pending:
            logger.info("All %d scenes already persisted for title %s", len(scenes), title_id)
            return 0

        logger.info(
            "Processing %d/%d scenes for title %s (%d already done)",
            len(pending),
            len(scenes),
            title_id,
            len(scenes) - len(pending),
        )

        with tempfile.TemporaryDirectory(prefix="cowatcher-audio-") as tmpdir:
            audio_path = str(Path(tmpdir) / "title_audio.wav")
            self._providers.audio_extractor.extract_audio(video_path, audio_path)
            transcripts = transcripts_for_scenes(
                self._providers.transcriber,
                audio_path,
                pending,
            )
            speaker_segments = diarize_title(self._providers.speaker_diarizer, audio_path)
            speaker_clusters = speaker_clusters_for_scenes(
                speaker_segments, pending, title_id
            )

        self._qdrant.ensure_collection(self._providers.embedder.vector_size)

        delay = self._settings.vision_caption_delay_sec
        processed = 0
        for scene, transcript, speakers in zip(
            pending, transcripts, speaker_clusters, strict=True
        ):
            clusters = self._providers.face_analyzer.detect_face_clusters(
                video_path, title_id, scene
            )
            caption = self._providers.captioner.caption_scenes(video_path, [scene])[0]
            event = _build_scene_event(
                title_id, scene, transcript, caption, clusters, speakers
            )

            vector = self._providers.embedder.embed_texts([event.embedding_text])[0]
            self._qdrant.upsert_scene_events([event], [vector])
            repo.save_scene_event(event)

            processed += 1
            logger.info(
                "Persisted scene %s (%d/%d) for title %s",
                scene.scene_id,
                processed,
                len(pending),
                title_id,
            )
            if processed < len(pending) and delay > 0:
                time.sleep(delay)

        return processed

    def _index_navigation_events(self, title_id: str, repo: SceneEventRepository) -> None:
        scenes = repo.list_scene_records(title_id)
        if not scenes:
            return

        cast_names = self._cast_names(title_id, repo)

        events, credits_ts = build_title_events(title_id, scenes, cast_names=cast_names)
        repo.replace_title_events(title_id, events)
        repo.set_credits_start_ts(title_id, credits_ts)
        logger.info(
            "Indexed %d navigation events for title %s (credits_start_ts=%s)",
            len(events),
            title_id,
            credits_ts,
        )

    def _cast_names(self, title_id: str, repo: SceneEventRepository) -> list[str]:
        display_name = repo.get_display_name(title_id)
        if not (display_name and self._settings.cast_lookup_enabled):
            return []
        cast_result = CastLookupTool(self._settings).lookup(title_name=display_name)
        if "cast" not in cast_result:
            return []
        return [
            str(entry.get("actor", ""))
            for entry in cast_result["cast"]
            if entry.get("actor")
        ]

    def _index_character_graph(self, title_id: str, repo: SceneEventRepository) -> None:
        """Offline character-intelligence enrichment (LangGraph -> Neo4j)."""
        if not self._settings.character_graph_enabled:
            logger.info(
                "Character graph disabled (set NEO4J_URI to enable); skipping for %s",
                title_id,
            )
            return

        scenes = repo.list_scene_records(title_id)
        if not scenes:
            return

        from ai_cowatcher.enrichment.graph import run_character_enrichment
        from ai_cowatcher.storage.character_store import build_character_store

        cast_names = self._cast_names(title_id, repo)
        store = build_character_store(self._settings)
        try:
            result = run_character_enrichment(
                self._settings,
                title_id=title_id,
                scenes=scenes,
                cast_names=cast_names,
                store=store,
            )
            logger.info(
                "Character graph for %s: %d characters, %d appearances, %d relationships",
                title_id,
                len(result.characters),
                len(result.appearances),
                len(result.relationships),
            )
        except Exception:  # noqa: BLE001 - enrichment must not fail the ingest
            logger.exception("Character graph enrichment failed for title %s", title_id)
        finally:
            store.close()

    def _index_title_knowledge(self, title_id: str) -> None:
        """Index curated knowledge files (if present) into the knowledge collection."""
        try:
            knowledge_store = QdrantKnowledgeStore(self._settings)
            result = index_title_knowledge(
                title_id,
                settings=self._settings,
                embedder=self._providers.embedder,
                knowledge_store=knowledge_store,
            )
            if result.chunk_count:
                logger.info(
                    "Indexed %d knowledge chunks for title %s",
                    result.chunk_count,
                    title_id,
                )
        except Exception:  # noqa: BLE001 - knowledge indexing must not fail ingest
            logger.exception("Knowledge indexing failed for title %s", title_id)


def _build_scene_event(
    title_id: str,
    scene: SceneBoundary,
    transcript: str,
    caption: str,
    face_cluster_ids: list[str],
    speaker_cluster_ids: list[str],
) -> SceneEventRecord:
    return SceneEventRecord(
        scene_id=scene.scene_id,
        title_id=title_id,
        start_ts=scene.start_ts,
        end_ts=scene.end_ts,
        transcript=transcript,
        caption=caption,
        face_cluster_ids=face_cluster_ids,
        speaker_cluster_ids=speaker_cluster_ids,
    )


def run_ingestion(
    title_id: str,
    video_path: str,
    *,
    force: bool = False,
    display_name: str | None = None,
) -> IngestionResult:
    return IngestionPipeline().run(
        title_id,
        video_path,
        force=force,
        display_name=display_name,
    )
