"""Relational models for offline ingestion."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from ai_cowatcher.db.base import Base


class TitleIngestion(Base):
    __tablename__ = "title_ingestions"

    title_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    credits_start_ts: Mapped[float | None] = mapped_column(Float, nullable=True)
    video_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    scene_count: Mapped[int] = mapped_column(nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    scene_events: Mapped[list["SceneEvent"]] = relationship(
        back_populates="title",
        cascade="all, delete-orphan",
    )
    title_events: Mapped[list["TitleEvent"]] = relationship(
        back_populates="title",
        cascade="all, delete-orphan",
    )


class TitleEvent(Base):
    __tablename__ = "title_events"

    event_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    title_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("title_ingestions.title_id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    ordinal: Mapped[int] = mapped_column(nullable=False, default=1)
    start_ts: Mapped[float] = mapped_column(Float, nullable=False)
    end_ts: Mapped[float] = mapped_column(Float, nullable=False)
    scene_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    label: Mapped[str] = mapped_column(Text, nullable=False, default="")
    event_metadata: Mapped[dict] = mapped_column(
        "metadata", JSON().with_variant(JSONB, "postgresql"), default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    title: Mapped[TitleIngestion] = relationship(back_populates="title_events")


class SceneEvent(Base):
    __tablename__ = "scene_events"

    scene_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    title_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("title_ingestions.title_id", ondelete="CASCADE"), index=True
    )
    start_ts: Mapped[float] = mapped_column(Float, nullable=False)
    end_ts: Mapped[float] = mapped_column(Float, nullable=False)
    transcript: Mapped[str] = mapped_column(Text, nullable=False, default="")
    caption: Mapped[str] = mapped_column(Text, nullable=False, default="")
    face_cluster_ids: Mapped[list] = mapped_column(JSON().with_variant(JSONB, "postgresql"))
    speaker_cluster_ids: Mapped[list] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=list
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    title: Mapped[TitleIngestion] = relationship(back_populates="scene_events")


class UserConversationTurn(Base):
    """Per-user conversation history for a title (source of truth in Postgres)."""

    __tablename__ = "user_conversation_turns"

    turn_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    title_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    current_ts: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
