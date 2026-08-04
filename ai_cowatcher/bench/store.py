"""Persist ask-bench rows to Postgres (+ optional JSONL)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.db.base import create_db_engine, init_database
from ai_cowatcher.db.models import BenchAskResult


@dataclass
class BenchResultRow:
    run_id: str
    title_id: str
    question_id: str
    question: str
    current_ts: float
    answer: str
    latency_ms: float
    cache_source: str
    model_name: str
    model_tier: str
    used_context: bool | None = None
    skip_memory: bool | None = None
    error: str | None = None
    kind: str | None = None
    status: str = "ok"

    def to_jsonl_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
        return payload


def ensure_bench_schema(settings: Settings | None = None) -> sessionmaker:
    settings = settings or get_settings()
    engine = create_db_engine(settings=settings)
    init_database(engine=engine, settings=settings)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def insert_bench_result(session: Session, row: BenchResultRow) -> None:
    session.add(
        BenchAskResult(
            run_id=row.run_id,
            title_id=row.title_id,
            question_id=row.question_id,
            question=row.question,
            current_ts=row.current_ts,
            answer=row.answer,
            latency_ms=row.latency_ms,
            cache_source=row.cache_source,
            model_name=row.model_name,
            model_tier=row.model_tier,
            used_context=row.used_context,
            skip_memory=row.skip_memory,
            error=row.error,
            kind=row.kind,
            status=row.status,
        )
    )
    session.commit()


def append_jsonl(path: Path, row: BenchResultRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row.to_jsonl_dict(), ensure_ascii=False) + "\n")
