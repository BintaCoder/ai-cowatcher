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
from ai_cowatcher.retrieval.cast_lookup import (
    CastLookupTool,
    CastRedisCache,
    actor_names_from_payload,
    build_cast_redis_cache,
    extract_cast_for_title,
)
from ai_cowatcher.storage.object_store import (
    ObjectStore,
    build_object_store,
    scene_audio_object_key,
)
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
        object_store: ObjectStore | None = None,
        cast_redis: CastRedisCache | None = None,
    ):
        self._settings = settings or get_settings()
        self._providers = providers or build_ingestion_providers(self._settings)
        if session_factory is None:
            engine = create_db_engine(settings=self._settings)
            init_database(engine=engine, settings=self._settings)
            session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        self._session_factory = session_factory
        self._qdrant = qdrant_store or QdrantSceneStore(self._settings)
        self._object_store = object_store or build_object_store(self._settings)
        self._cast_redis = (
            cast_redis
            if cast_redis is not None
            else build_cast_redis_cache(self._settings)
        )

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
                try:
                    self._object_store.delete_prefix(f"scenes/{title_id}/")
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to delete scene audio for title %s", title_id)
                if self._cast_redis is not None:
                    self._cast_redis.delete(title_id)

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
                # Cast once per title — used by nav indexing, character graph, and /ask.
                self._extract_and_cache_cast(title_id, repo)
                self._index_navigation_events(title_id, repo)
                self._index_scene_trivia(title_id, repo)
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
                audio_key = self._store_scene_audio(
                    title_id=title_id,
                    scene=scene,
                    title_audio_path=audio_path,
                    tmpdir=tmpdir,
                )
                event = _build_scene_event(
                    title_id,
                    scene,
                    transcript,
                    caption,
                    clusters,
                    speakers,
                    audio_object_key=audio_key,
                )

                vector = self._providers.embedder.embed_texts([event.embedding_text])[0]
                self._qdrant.upsert_scene_events([event], [vector])
                repo.save_scene_event(event)

                processed += 1
                logger.info(
                    "Persisted scene %s (%d/%d) for title %s audio=%s",
                    scene.scene_id,
                    processed,
                    len(pending),
                    title_id,
                    audio_key or "none",
                )
                if processed < len(pending) and delay > 0:
                    time.sleep(delay)

            return processed

    def _store_scene_audio(
        self,
        *,
        title_id: str,
        scene: SceneBoundary,
        title_audio_path: str,
        tmpdir: str,
    ) -> str | None:
        if not self._settings.scene_audio_enabled:
            return None
        try:
            clip_path = str(Path(tmpdir) / f"{scene.scene_id}.wav")
            self._providers.audio_extractor.extract_audio_window(
                title_audio_path,
                clip_path,
                start_ts=scene.start_ts,
                end_ts=scene.end_ts,
            )
            data = Path(clip_path).read_bytes()
            if not data:
                return None
            key = scene_audio_object_key(title_id, scene.scene_id)
            return self._object_store.put_bytes(key, data, content_type="audio/wav")
        except Exception:  # noqa: BLE001 — audio optional; text still usable
            logger.exception(
                "Scene audio extract/upload failed title=%s scene=%s",
                title_id,
                scene.scene_id,
            )
            return None

    def _extract_and_cache_cast(
        self, title_id: str, repo: SceneEventRepository
    ) -> dict | None:
        """Fetch public cast during ingest and store for fast real-time lookup."""
        display_name = repo.get_display_name(title_id)
        search_name = display_name or self._settings.effective_search_title(title_id)
        if not search_name:
            logger.info("No display/search name for cast extract title=%s", title_id)
            return None

        # Allow re-ingest force path to refresh; still skip duplicate TMDB if present
        # unless force wiped the row (delete_title_data clears cast with the title).
        existing = repo.get_cast_cache(title_id)
        if existing and existing.get("cast") and not self._settings.mock_mode:
            # Warm Redis even when DB already has it (resume path).
            if self._cast_redis is not None:
                self._cast_redis.set(title_id, existing)
            logger.info(
                "Cast already cached for %s (%d actors)",
                title_id,
                len(existing.get("cast") or []),
            )
            return existing

        if self._settings.mock_mode:
            # Deterministic mock cast so offline pilots still exercise the cache path.
            payload = {
                "title": search_name,
                "media_type": "movie",
                "tmdb_id": None,
                "cast": [
                    {"actor": "Mock Actor One", "character": "Lead"},
                    {"actor": "Mock Actor Two", "character": "Supporting"},
                ],
                "source": "mock_ingest",
            }
            repo.save_cast_cache(title_id, payload)
            if self._cast_redis is not None:
                self._cast_redis.set(title_id, payload)
            logger.info("Cached mock cast for title %s", title_id)
            return payload

        if not self._settings.tmdb_api_key:
            logger.warning(
                "TMDB_API_KEY missing — skipping cast extract for title %s", title_id
            )
            return None

        try:
            result = extract_cast_for_title(
                self._settings, title_name=search_name
            )
        except Exception:  # noqa: BLE001 — cast is optional; don't fail ingest
            logger.exception("Cast extract failed for title %s", title_id)
            return None

        if "error" in result or not result.get("cast"):
            logger.warning(
                "Cast extract empty for title %s name=%r: %s",
                title_id,
                search_name,
                result.get("error"),
            )
            return None

        payload = dict(result)
        payload["source"] = "ingest"
        payload["search_name"] = search_name
        repo.save_cast_cache(title_id, payload)
        if self._cast_redis is not None:
            self._cast_redis.set(title_id, payload)
        logger.info(
            "Cached cast for title %s (%d actors) from TMDB match %r",
            title_id,
            len(payload.get("cast") or []),
            payload.get("title"),
        )
        return payload

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

    def _index_scene_trivia(self, title_id: str, repo: SceneEventRepository) -> None:
        """Precompute spoiler-filtered trivia once per title (no ask-path LLM)."""
        if not getattr(self._settings, "trivia_ingest_enabled", True):
            return
        scenes = repo.list_scene_records(title_id)
        if not scenes:
            return
        # Sparse sample: every Nth scene to keep ingest light (pilot).
        stride = max(1, len(scenes) // 12) if len(scenes) > 12 else 1
        sampled = scenes[::stride][:16]
        from ai_cowatcher.trivia import TriviaStore, mock_trivia_candidates_for_scene

        saved = 0
        rejected = 0
        with self._session_factory() as session:
            store = TriviaStore(session)
            for scene in sampled:
                candidates = mock_trivia_candidates_for_scene(
                    title_id=title_id,
                    scene_id=scene.scene_id,
                    caption=scene.caption,
                    transcript=scene.transcript,
                )
                before = len(candidates)
                kept = store.save_candidates(
                    title_id=title_id,
                    scene_id=scene.scene_id,
                    candidates=candidates,
                )
                saved += len(kept)
                rejected += before - len(kept)
        logger.info(
            "Indexed scene trivia for title %s (saved=%d rejected=%d sampled=%d)",
            title_id,
            saved,
            rejected,
            len(sampled),
        )

    def _cast_names(self, title_id: str, repo: SceneEventRepository) -> list[str]:
        payload = repo.get_cast_cache(title_id)
        if payload:
            return actor_names_from_payload(payload)
        # Fallback: extract now if not yet cached (older resumed path).
        payload = self._extract_and_cache_cast(title_id, repo)
        return actor_names_from_payload(payload)

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
    audio_object_key: str | None = None,
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
        audio_object_key=audio_object_key,
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
