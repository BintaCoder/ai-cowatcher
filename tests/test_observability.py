"""Tests for pilot observability: JSON ask logs and /metrics-lite."""

from __future__ import annotations

import io
import json
import logging
import sys
import uuid

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ai_cowatcher.agent.completion import MockCompletionClient
from ai_cowatcher.agent.metrics import metrics_lite_summary
from ai_cowatcher.config import Settings
from ai_cowatcher.main import create_app
from ai_cowatcher.observability.summarize_logs import main as summarize_logs_main
from ai_cowatcher.providers.mock import MockTextEmbedder
from ai_cowatcher.realtime.viewing_session import build_viewing_session
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore

SPOILER_QUESTION = "Who is the killer?"


@pytest.fixture
def observability_settings() -> Settings:
    return Settings(MOCK_MODE=True, QDRANT_COLLECTION="test_observability")


@pytest.fixture
def seeded_qdrant(observability_settings: Settings) -> QdrantSceneStore:
    embedder = MockTextEmbedder()
    client = QdrantClient(":memory:")
    store = QdrantSceneStore(observability_settings, client=client)
    store.ensure_collection(embedder.vector_size)

    title_id = "thriller-001"
    query_vector = embedder.embed_texts([SPOILER_QUESTION])[0]
    filler_vector = embedder.embed_texts(["crime scene investigation"])[0]

    def upsert(
        scene_id: str,
        start_ts: float,
        end_ts: float,
        transcript: str,
        vector: list[float],
    ) -> None:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{title_id}:{scene_id}"))
        client.upsert(
            collection_name=store._collection,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "title_id": title_id,
                        "scene_id": scene_id,
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "transcript": transcript,
                        "caption": "",
                        "face_cluster_ids": [],
                    },
                )
            ],
        )

    upsert(
        "s0000",
        start_ts=0.0,
        end_ts=20.0,
        transcript="Detectives arrive at the crime scene.",
        vector=filler_vector,
    )
    upsert(
        "s0001",
        start_ts=50.0,
        end_ts=70.0,
        transcript="The killer is Marcus and the room goes silent.",
        vector=query_vector,
    )
    return store


@pytest.fixture
def viewing_session(observability_settings: Settings, seeded_qdrant: QdrantSceneStore):
    return build_viewing_session(
        observability_settings,
        qdrant_store=seeded_qdrant,
        embedder=MockTextEmbedder(),
        completion_client=MockCompletionClient(),
    )


def test_ask_emits_structured_json_log(
    viewing_session,
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level(logging.INFO, logger="ai_cowatcher.ask"):
        viewing_session.ask(
            title_id="thriller-001",
            current_ts=25.0,
            question="Who is the killer?",
            user_id="viewer-42",
        )

    ask_records = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "ai_cowatcher.ask"
    ]
    assert len(ask_records) == 1
    payload = ask_records[0]
    assert payload["event"] == "ask_request"
    assert payload["title_id"] == "thriller-001"
    assert payload["current_ts"] == 25.0
    assert payload["model_tier"] in {"fast", "escalated"}
    assert isinstance(payload["latency_ms"], (int, float))
    assert payload["dont_know"] is True
    assert payload["used_context"] is True
    assert payload["total_tokens"] is not None


def test_metrics_lite_reports_pilot_kpis(viewing_session, observability_settings: Settings):
    viewing_session.ask(
        title_id="thriller-001",
        current_ts=25.0,
        question="Who is the killer?",
        user_id="viewer-42",
    )
    viewing_session.ask(
        title_id="thriller-001",
        current_ts=75.0,
        question="Who is the killer?",
        user_id="viewer-42",
    )
    viewing_session.ask(
        title_id="thriller-001",
        current_ts=75.0,
        question="Explain why the detective lied to his partner about the alibi.",
        user_id="viewer-42",
    )

    summary = metrics_lite_summary()
    assert summary["ask_count"] == 3
    assert summary["average_latency_ms"] > 0
    assert summary["tier_usage"]["fast"] >= 1
    assert summary["tier_usage"]["escalated"] >= 1
    assert 0.0 < summary["escalation_rate"] < 1.0
    assert summary["dont_know_rate_overall"] > 0.0

    by_title = summary["by_title"]["thriller-001"]
    assert by_title["ask_count"] == 3
    assert by_title["dont_know_rate"] > 0.0
    assert by_title["average_latency_ms"] > 0


def test_metrics_lite_endpoint(viewing_session, observability_settings: Settings):
    viewing_session.ask(
        title_id="thriller-001",
        current_ts=25.0,
        question="Who is the killer?",
        user_id="viewer-42",
    )

    app = create_app(observability_settings)
    client = TestClient(app)
    response = client.get("/metrics-lite")
    assert response.status_code == 200
    body = response.json()
    assert body["ask_count"] == 1
    assert "average_latency_ms" in body
    assert "escalation_rate" in body
    assert "dont_know_rate_overall" in body
    assert "thriller-001" in body["by_title"]


def test_summarize_logs_script_from_stdin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    log_line = json.dumps(
        {
            "event": "ask_request",
            "title_id": "demo",
            "user_id": "u1",
            "current_ts": 12.5,
            "latency_ms": 120.0,
            "model_tier": "fast",
            "model_name": "mock-llm-fast",
            "escalation_reason": "default_fast_tier",
            "used_context": False,
            "dont_know": True,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
    )

    monkeypatch.setattr(sys, "stdin", io.StringIO(log_line + "\n"))
    assert summarize_logs_main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ask_count"] == 1
    assert output["dont_know_rate_overall"] == 1.0
    assert output["by_title"]["demo"]["dont_know_rate"] == 1.0
