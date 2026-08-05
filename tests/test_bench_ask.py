"""Unit tests for ask-bench pure logic (no live Gemini / network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_cowatcher.bench.duration import DEFAULT_TITLE_NAME, resolve_title_id
from ai_cowatcher.bench.sampling import clamp_playhead, parse_cache_source, sample_questions
from ai_cowatcher.db.models import TitleIngestion

QUESTIONS_FILE = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "friends_ross_questions.json"
)


@pytest.fixture(scope="module")
def friends_questions() -> list[dict]:
    data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) == 20
    return data


def test_friends_ross_bank_ids(friends_questions: list[dict]) -> None:
    ids = [q["id"] for q in friends_questions]
    assert ids == [f"q{i:02d}" for i in range(1, 21)]
    for q in friends_questions:
        assert q["text"].strip()
        assert q.get("kind")


@pytest.mark.parametrize(
    "model_name,model_tier,escalation,expected",
    [
        ("qa_cache:exact", "cache", "cache:exact", "exact"),
        ("qa_cache:semantic", "cache", "cache:semantic", "semantic"),
        ("gemini/gemini-2.0-flash", "fast", "none", "miss"),
        ("", "", "", "none"),
        (None, None, None, "none"),
        ("QA_CACHE:Exact", "cache", "", "exact"),
        ("some-model", "cache", "", "exact"),  # tier=cache fallback
        ("", "fast", "cache:semantic", "semantic"),
        ("gate:free", "gate", "gate:social", "free"),
        ("", "gate", "gate:ignore_filler", "free"),
    ],
)
def test_parse_cache_source(model_name, model_tier, escalation, expected) -> None:
    assert (
        parse_cache_source(
            model_name,
            model_tier=model_tier,
            escalation_reason=escalation,
        )
        == expected
    )


def test_clamp_playhead() -> None:
    assert clamp_playhead(-5.0, 100.0) == 0.0
    assert clamp_playhead(50.0, 100.0) == 50.0
    assert clamp_playhead(200.0, 100.0) == pytest.approx(99.95)
    assert clamp_playhead(10.0, 0.0) == 0.0
    assert clamp_playhead(0.01, 0.02, epsilon=0.05) == 0.0


def test_sample_questions_seed_reproducible(friends_questions: list[dict]) -> None:
    a = sample_questions(friends_questions, n=5, duration_sec=120.0, seed=42)
    b = sample_questions(friends_questions, n=5, duration_sec=120.0, seed=42)
    assert a == b
    assert len(a) == 5
    ids = {row["id"] for row in a}
    assert len(ids) == 5
    for row in a:
        assert 0.0 <= row["current_ts"] <= 120.0
        assert row["text"]


def test_sample_questions_different_seeds(friends_questions: list[dict]) -> None:
    a = sample_questions(friends_questions, n=5, duration_sec=90.0, seed=1)
    b = sample_questions(friends_questions, n=5, duration_sec=90.0, seed=2)
    # Extremely unlikely equal under different seeds for both ids and timestamps
    assert ([(r["id"], r["current_ts"]) for r in a] != [(r["id"], r["current_ts"]) for r in b])


def test_sample_questions_n_validates(friends_questions: list[dict]) -> None:
    with pytest.raises(ValueError):
        sample_questions(friends_questions, n=0, duration_sec=10.0, seed=0)
    with pytest.raises(ValueError):
        sample_questions([], n=1, duration_sec=10.0, seed=0)


def test_sample_playhead_clamped_to_duration(friends_questions: list[dict]) -> None:
    rows = sample_questions(friends_questions, n=5, duration_sec=0.01, seed=7)
    for row in rows:
        assert row["current_ts"] == 0.0


def test_http_error_helpers() -> None:
    from ai_cowatcher.bench.ask_runner import (
        _http_error_detail,
        _is_retryable_ask_status,
    )

    class _R:
        def __init__(self, body):
            self._body = body
            self.text = json.dumps(body) if isinstance(body, dict) else str(body)
            self.reason_phrase = "Error"

        def json(self):
            return self._body

    detail = _http_error_detail(
        _R({"detail": 'GeminiException - "message": "high demand"'})
    )
    assert "high demand" in detail
    assert _is_retryable_ask_status(500, detail)
    assert _is_retryable_ask_status(503, "unavailable")
    assert not _is_retryable_ask_status(400, "bad request")


def test_post_ask_payload_includes_persona(monkeypatch) -> None:
    """post_ask must send persona_id + companion_gender (QA cache / tone)."""
    from ai_cowatcher.bench import ask_runner

    captured: dict = {}

    class FakeResp:
        is_success = True
        status_code = 200
        request = None

        def json(self):
            return {
                "answer": "ok",
                "model_name": "mock",
                "model_tier": "fast",
                "escalation_reason": "merged:CONTENT",
            }

    class FakeClient:
        def post(self, url, json=None):  # noqa: A002
            captured["url"] = url
            captured["json"] = json
            return FakeResp()

    data, _ms = ask_runner.post_ask(
        FakeClient(),  # type: ignore[arg-type]
        base_url="http://localhost:8000",
        title_id="friends_ross",
        question="Who is that?",
        current_ts=40.0,
        user_id="bench",
        persona_id="witty_friend",
        companion_gender="female",
        retries=0,
    )
    assert data["answer"] == "ok"
    assert captured["json"]["persona_id"] == "witty_friend"
    assert captured["json"]["companion_gender"] == "female"


def test_bench_cli_persona_flags() -> None:
    from ai_cowatcher.bench.ask_runner import build_parser

    ns = build_parser().parse_args(
        ["--persona-id", "witty_friend", "--companion-gender", "male", "--all-personas"]
    )
    assert ns.persona_id == "witty_friend"
    assert ns.companion_gender == "male"
    assert ns.all_personas is True


def test_default_title_name() -> None:
    assert DEFAULT_TITLE_NAME == "Friends Ross"


def test_resolve_title_id_by_display_name_and_slug() -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from ai_cowatcher.db.base import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add(
            TitleIngestion(
                title_id="friends_ross",
                video_path="/tmp/x.mp4",
                status="completed",
                display_name="Friends Ross",
            )
        )
        session.commit()
        assert resolve_title_id(session, "Friends Ross") == "friends_ross"
        assert resolve_title_id(session, "friends_ross") == "friends_ross"
        assert resolve_title_id(session, "FRIENDS ROSS") == "friends_ross"


def test_row_from_jsonl_preserves_created_at() -> None:
    from datetime import datetime, timezone

    from ai_cowatcher.bench.store import row_from_jsonl_dict

    row = row_from_jsonl_dict(
        {
            "run_id": "abc123",
            "title_id": "friends_ross",
            "question_id": "q01",
            "question": "What just happened?",
            "current_ts": 12.5,
            "answer": "A joke beat.",
            "latency_ms": 900.0,
            "cache_source": "miss",
            "model_name": "gemini/x",
            "model_tier": "fast",
            "persona_id": "witty_friend",
            "companion_gender": "female",
            "status": "ok",
            "created_at": "2026-08-04T10:28:25.703427+00:00",
        }
    )
    assert row.run_id == "abc123"
    assert row.persona_id == "witty_friend"
    assert row.created_at == datetime(2026, 8, 4, 10, 28, 25, 703427, tzinfo=timezone.utc)


def test_reimport_jsonl_skips_existing_run(tmp_path: Path) -> None:
    from sqlalchemy import create_engine, event, func, select
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from ai_cowatcher.bench.store import (
        BenchResultRow,
        insert_bench_result,
        reimport_jsonl_file,
    )
    from ai_cowatcher.db.base import Base
    from ai_cowatcher.db.models import BenchAskResult

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _pragma(dbapi_connection, _):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    path = tmp_path / "run1.jsonl"
    path.write_text(
        json.dumps(
            {
                "run_id": "run1",
                "title_id": "t",
                "question_id": "q01",
                "question": "Hi",
                "current_ts": 1.0,
                "answer": "Hello",
                "latency_ms": 10.0,
                "cache_source": "free",
                "model_name": "gate:free",
                "model_tier": "gate",
                "persona_id": "easygoing_friend",
                "companion_gender": "neutral",
                "status": "ok",
                "created_at": "2026-08-04T12:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with factory() as session:
        n, status = reimport_jsonl_file(session, path)
        assert status == "ok" and n == 1
        n2, status2 = reimport_jsonl_file(session, path)
        assert status2 == "skipped" and n2 == 0
        count = session.scalar(select(func.count()).select_from(BenchAskResult))
        assert count == 1


def test_reimport_jsonl_dir(tmp_path: Path) -> None:
    from sqlalchemy import create_engine, event, func, select
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from ai_cowatcher.bench.store import reimport_jsonl_dir
    from ai_cowatcher.db.base import Base
    from ai_cowatcher.db.models import BenchAskResult

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _pragma(dbapi_connection, _):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    for rid in ("aaa", "bbb"):
        (tmp_path / f"{rid}.jsonl").write_text(
            json.dumps(
                {
                    "run_id": rid,
                    "title_id": "t",
                    "question_id": "q01",
                    "question": "Q",
                    "current_ts": 1.0,
                    "answer": "A",
                    "latency_ms": 5.0,
                    "cache_source": "miss",
                    "model_name": "m",
                    "model_tier": "fast",
                    "status": "ok",
                    "created_at": "2026-08-04T12:00:00+00:00",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    with factory() as session:
        stats = reimport_jsonl_dir(session, tmp_path)
        assert stats["files"] == 2
        assert stats["inserted"] == 2
        assert stats["skipped"] == 0
        stats2 = reimport_jsonl_dir(session, tmp_path)
        assert stats2["skipped"] == 2
        assert stats2["inserted"] == 0
        assert session.scalar(select(func.count()).select_from(BenchAskResult)) == 2
