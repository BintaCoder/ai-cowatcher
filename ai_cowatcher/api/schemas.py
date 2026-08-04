"""API schemas for ingestion and real-time ask endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CompanionGender = Literal["male", "female", "neutral"]


class IngestRequest(BaseModel):
    title_id: str = Field(..., min_length=1, max_length=128)
    video_path: str = Field(..., min_length=1)
    force: bool = False


class CatalogTitleRequest(BaseModel):
    title_id: str = Field(..., min_length=1, max_length=128)
    video_path: str = Field(..., min_length=1)
    display_name: str | None = Field(default=None, max_length=512)
    force: bool = False


class CatalogTitleResponse(BaseModel):
    status: str
    title_id: str
    event_id: str
    message: str


class IngestResponse(BaseModel):
    status: str
    title_id: str
    message: str


class AskRequest(BaseModel):
    title_id: str = Field(..., min_length=1, max_length=128)
    current_ts: float = Field(..., ge=0.0)
    question: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1, max_length=128)
    # Companion personality (tone). Defaults to Settings.DEFAULT_PERSONA_ID server-side.
    persona_id: str | None = Field(default=None, max_length=64)
    # Preferred TTS / delivery gender (client may also apply voice selection local-only).
    companion_gender: CompanionGender | None = None


class AskResponse(BaseModel):
    answer: str
    title_id: str
    user_id: str
    current_ts: float
    model_tier: str
    model_name: str
    escalation_reason: str
    speak: bool = True
    skip_memory: bool = False


class NavigateRequest(BaseModel):
    title_id: str = Field(..., min_length=1, max_length=128)
    current_ts: float = Field(..., ge=0.0)
    question: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1, max_length=128)


class NavigateResponseSchema(BaseModel):
    answer: str
    title_id: str
    user_id: str
    current_ts: float
    seek_to_ts: float | None = None
    scene_id: str | None = None
    event_type: str | None = None
    navigation_mode: str

