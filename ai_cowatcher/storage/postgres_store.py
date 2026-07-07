"""Postgres persistence for title ingestion and scene events."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ai_cowatcher.db.models import SceneEvent, TitleEvent, TitleIngestion
from ai_cowatcher.domain import SceneEventRecord, TitleEventRecord


class SceneEventRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_title(self, title_id: str) -> TitleIngestion | None:
        return self._session.get(TitleIngestion, title_id)

    def get_display_name(self, title_id: str) -> str | None:
        title = self.get_title(title_id)
        if title is None or not title.display_name:
            return None
        return title.display_name

    def set_display_name(self, title_id: str, display_name: str) -> None:
        title = self.get_title(title_id)
        if title is None:
            raise ValueError(f"Unknown title_id {title_id}")
        title.display_name = display_name
        self._session.commit()

    def is_completed(self, title_id: str) -> bool:
        title = self.get_title(title_id)
        return title is not None and title.status == "completed"

    def mark_processing(self, title_id: str, video_path: str) -> TitleIngestion:
        title = self.get_title(title_id)
        if title is None:
            title = TitleIngestion(title_id=title_id, video_path=video_path, status="processing")
            self._session.add(title)
        else:
            title.video_path = video_path
            title.status = "processing"
            title.error_message = None
        self._session.commit()
        return title

    def mark_completed(self, title_id: str, scene_count: int) -> None:
        title = self.get_title(title_id)
        if title is None:
            raise ValueError(f"Unknown title_id {title_id}")
        title.status = "completed"
        title.scene_count = scene_count
        title.ingested_at = datetime.now(UTC)
        title.error_message = None
        self._session.commit()

    def mark_failed(self, title_id: str, message: str) -> None:
        title = self.get_title(title_id)
        if title is None:
            title = TitleIngestion(title_id=title_id, video_path="", status="failed")
            self._session.add(title)
        title.status = "failed"
        title.error_message = message
        self._session.commit()

    def delete_title_data(self, title_id: str) -> None:
        self._session.execute(delete(SceneEvent).where(SceneEvent.title_id == title_id))
        self._session.execute(delete(TitleEvent).where(TitleEvent.title_id == title_id))
        title = self.get_title(title_id)
        if title is not None:
            self._session.delete(title)
        self._session.commit()

    def save_scene_events(self, events: list[SceneEventRecord]) -> None:
        for event in events:
            self._merge_event(event)
        self._session.commit()

    def save_scene_event(self, event: SceneEventRecord) -> None:
        """Persist a single enriched scene and commit immediately (resumable ingest)."""
        self._merge_event(event)
        self._session.commit()

    def _merge_event(self, event: SceneEventRecord) -> None:
        row = SceneEvent(
            scene_id=f"{event.title_id}:{event.scene_id}",
            title_id=event.title_id,
            start_ts=event.start_ts,
            end_ts=event.end_ts,
            transcript=event.transcript,
            caption=event.caption,
            face_cluster_ids=event.face_cluster_ids,
            speaker_cluster_ids=event.speaker_cluster_ids,
        )
        self._session.merge(row)

    def get_existing_scene_ids(self, title_id: str) -> set[str]:
        """Return raw scene ids (without the title prefix) already persisted for a title."""
        stmt = select(SceneEvent.scene_id).where(SceneEvent.title_id == title_id)
        prefix = f"{title_id}:"
        return {
            row[len(prefix):] if row.startswith(prefix) else row
            for row in self._session.scalars(stmt).all()
        }

    def count_scene_events(self, title_id: str) -> int:
        stmt = select(SceneEvent).where(SceneEvent.title_id == title_id)
        return len(self._session.scalars(stmt).all())

    def list_completed_titles(self) -> list[TitleIngestion]:
        stmt = (
            select(TitleIngestion)
            .where(TitleIngestion.status == "completed")
            .order_by(TitleIngestion.title_id)
        )
        return list(self._session.scalars(stmt).all())

    def list_scene_records(self, title_id: str) -> list[SceneEventRecord]:
        stmt = (
            select(SceneEvent)
            .where(SceneEvent.title_id == title_id)
            .order_by(SceneEvent.start_ts)
        )
        prefix = f"{title_id}:"
        records: list[SceneEventRecord] = []
        for row in self._session.scalars(stmt).all():
            raw_scene_id = row.scene_id
            if raw_scene_id.startswith(prefix):
                raw_scene_id = raw_scene_id[len(prefix) :]
            records.append(
                SceneEventRecord(
                    scene_id=raw_scene_id,
                    title_id=row.title_id,
                    start_ts=row.start_ts,
                    end_ts=row.end_ts,
                    transcript=row.transcript,
                    caption=row.caption,
                    face_cluster_ids=list(row.face_cluster_ids or []),
                    speaker_cluster_ids=list(row.speaker_cluster_ids or []),
                )
            )
        return records

    def set_credits_start_ts(self, title_id: str, credits_start_ts: float | None) -> None:
        title = self.get_title(title_id)
        if title is None:
            raise ValueError(f"Unknown title_id {title_id}")
        title.credits_start_ts = credits_start_ts
        self._session.commit()

    def get_credits_start_ts(self, title_id: str) -> float | None:
        title = self.get_title(title_id)
        if title is None:
            return None
        return title.credits_start_ts

    def replace_title_events(self, title_id: str, events: list[TitleEventRecord]) -> None:
        self._session.execute(delete(TitleEvent).where(TitleEvent.title_id == title_id))
        for event in events:
            self._session.merge(
                TitleEvent(
                    event_id=event.event_id,
                    title_id=event.title_id,
                    event_type=event.event_type,
                    ordinal=event.ordinal,
                    start_ts=event.start_ts,
                    end_ts=event.end_ts,
                    scene_id=event.scene_id,
                    label=event.label,
                    event_metadata=event.metadata,
                )
            )
        self._session.commit()

    def list_title_events(
        self, title_id: str, *, event_type: str | None = None
    ) -> list[TitleEventRecord]:
        stmt = select(TitleEvent).where(TitleEvent.title_id == title_id)
        if event_type is not None:
            stmt = stmt.where(TitleEvent.event_type == event_type)
        stmt = stmt.order_by(TitleEvent.start_ts, TitleEvent.ordinal)
        return [_title_event_to_record(row) for row in self._session.scalars(stmt).all()]

    def get_title_event(
        self, title_id: str, event_type: str, ordinal: int
    ) -> TitleEventRecord | None:
        stmt = select(TitleEvent).where(
            TitleEvent.title_id == title_id,
            TitleEvent.event_type == event_type,
            TitleEvent.ordinal == ordinal,
        )
        row = self._session.scalars(stmt).first()
        if row is None:
            return None
        return _title_event_to_record(row)


def _title_event_to_record(row: TitleEvent) -> TitleEventRecord:
    return TitleEventRecord(
        event_id=row.event_id,
        title_id=row.title_id,
        event_type=row.event_type,
        ordinal=row.ordinal,
        start_ts=row.start_ts,
        end_ts=row.end_ts,
        scene_id=row.scene_id,
        label=row.label,
        metadata=dict(row.event_metadata or {}),
    )
