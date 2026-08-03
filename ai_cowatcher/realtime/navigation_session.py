"""Real-time navigation orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import sessionmaker

from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.db.base import create_db_engine, init_database
from ai_cowatcher.interfaces import TextEmbedder
from ai_cowatcher.navigation.resolver import NavigationResolver
from ai_cowatcher.providers import mock
from ai_cowatcher.providers.real import BgeM3Embedder
from ai_cowatcher.retrieval.cast_lookup import CastLookupTool, build_cast_redis_cache
from ai_cowatcher.retrieval.event_lookup import EventLookupTool
from ai_cowatcher.retrieval.scene_navigate import SceneNavigateTool
from ai_cowatcher.storage.postgres_store import SceneEventRepository
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore


class _NavCastStore:
    def __init__(self, repo: SceneEventRepository) -> None:
        self._repo = repo

    def get_cast_cache(self, title_id: str) -> dict | None:
        return self._repo.get_cast_cache(title_id)

    def save_cast_cache(self, title_id: str, cast_payload: dict) -> None:
        self._repo.save_cast_cache(title_id, cast_payload)


@dataclass
class NavigateResponse:
    answer: str
    title_id: str
    user_id: str
    current_ts: float
    seek_to_ts: float | None
    scene_id: str | None
    event_type: str | None
    navigation_mode: str


class NavigationSession:
    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker,
        embedder: TextEmbedder,
        qdrant_store: QdrantSceneStore,
    ):
        self._settings = settings
        self._session_factory = session_factory
        self._embedder = embedder
        self._qdrant = qdrant_store

    def navigate(
        self,
        *,
        title_id: str,
        question: str,
        current_ts: float,
        user_id: str,
    ) -> NavigateResponse:
        with self._session_factory() as session:
            repo = SceneEventRepository(session)
            display_name = repo.get_display_name(title_id)
            cast_lookup = CastLookupTool(
                self._settings,
                store=_NavCastStore(repo),
                redis_cache=build_cast_redis_cache(self._settings),
            )
            resolver = NavigationResolver(
                repo=repo,
                scene_navigate=SceneNavigateTool(self._embedder, self._qdrant, self._settings),
                event_lookup=EventLookupTool(repo),
                cast_lookup=cast_lookup,
                title_display_name=display_name,
            )
            result = resolver.resolve(title_id=title_id, question=question, current_ts=current_ts)

        return NavigateResponse(
            answer=result.answer,
            title_id=title_id,
            user_id=user_id,
            current_ts=current_ts,
            seek_to_ts=result.seek_to_ts,
            scene_id=result.scene_id,
            event_type=result.event_type,
            navigation_mode=result.navigation_mode,
        )


def _build_embedder(settings: Settings) -> TextEmbedder:
    if settings.mock_mode:
        return mock.MockTextEmbedder()
    return BgeM3Embedder(settings)


def build_navigation_session(
    settings: Settings | None = None,
    *,
    session_factory: sessionmaker | None = None,
    embedder: TextEmbedder | None = None,
    qdrant_store: QdrantSceneStore | None = None,
) -> NavigationSession:
    """Build a navigation orchestrator.

    Prefer injecting shared ``session_factory`` / ``embedder`` / ``qdrant_store``
    (warmed once at app startup) so /navigate does not re-load models per request.
    """
    settings = settings or get_settings()
    if session_factory is None:
        engine = create_db_engine(settings=settings)
        init_database(engine=engine, settings=settings)
        session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return NavigationSession(
        settings=settings,
        session_factory=session_factory,
        embedder=embedder or _build_embedder(settings),
        qdrant_store=qdrant_store or QdrantSceneStore(settings),
    )
