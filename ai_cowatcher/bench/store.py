"""Persist ask-bench rows to Postgres (+ optional JSONL)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from ai_cowatcher.config import Settings, get_settings
from ai_cowatcher.db.base import create_db_engine, init_database
from ai_cowatcher.db.models import BenchAskResult

logger = logging.getLogger(__name__)

# Default archive window used by Grafana panels + docs.
BENCH_RETENTION_DAYS = 30


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
    persona_id: str = ""
    companion_gender: str = ""
    used_context: bool | None = None
    skip_memory: bool | None = None
    error: str | None = None
    kind: str | None = None
    status: str = "ok"
    created_at: datetime | None = None

    def to_jsonl_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        created = self.created_at or datetime.now(timezone.utc)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        payload["created_at"] = created.isoformat()
        return payload


def ensure_bench_schema(settings: Settings | None = None) -> sessionmaker:
    settings = settings or get_settings()
    engine = create_db_engine(settings=settings)
    init_database(engine=engine, settings=settings)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def insert_bench_result(session: Session, row: BenchResultRow) -> None:
    kwargs: dict[str, Any] = {
        "run_id": row.run_id,
        "title_id": row.title_id,
        "question_id": row.question_id,
        "question": row.question,
        "current_ts": row.current_ts,
        "answer": row.answer,
        "latency_ms": row.latency_ms,
        "cache_source": row.cache_source,
        "model_name": row.model_name,
        "model_tier": row.model_tier,
        "persona_id": row.persona_id or "",
        "companion_gender": row.companion_gender or "",
        "used_context": row.used_context,
        "skip_memory": row.skip_memory,
        "error": row.error,
        "kind": row.kind,
        "status": row.status,
    }
    if row.created_at is not None:
        created = row.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        kwargs["created_at"] = created
    session.add(BenchAskResult(**kwargs))
    session.commit()


def append_jsonl(path: Path, row: BenchResultRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row.to_jsonl_dict(), ensure_ascii=False) + "\n")


def _parse_created_at(raw: Any) -> datetime | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    text = str(raw).strip()
    if not text:
        return None
    # Support trailing Z from ISO dumps.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def row_from_jsonl_dict(payload: dict[str, Any]) -> BenchResultRow:
    return BenchResultRow(
        run_id=str(payload.get("run_id") or ""),
        title_id=str(payload.get("title_id") or ""),
        question_id=str(payload.get("question_id") or ""),
        question=str(payload.get("question") or ""),
        current_ts=float(payload.get("current_ts") or 0.0),
        answer=str(payload.get("answer") or ""),
        latency_ms=float(payload.get("latency_ms") if payload.get("latency_ms") is not None else -1.0),
        cache_source=str(payload.get("cache_source") or "none"),
        model_name=str(payload.get("model_name") or ""),
        model_tier=str(payload.get("model_tier") or ""),
        persona_id=str(payload.get("persona_id") or ""),
        companion_gender=str(payload.get("companion_gender") or ""),
        used_context=payload.get("used_context"),
        skip_memory=payload.get("skip_memory"),
        error=payload.get("error"),
        kind=payload.get("kind"),
        status=str(payload.get("status") or "ok"),
        created_at=_parse_created_at(payload.get("created_at")),
    )


def run_id_row_count(session: Session, run_id: str) -> int:
    return int(
        session.scalar(
            select(func.count())
            .select_from(BenchAskResult)
            .where(BenchAskResult.run_id == run_id)
        )
        or 0
    )


def run_id_exists(session: Session, run_id: str) -> bool:
    return run_id_row_count(session, run_id) > 0


def delete_run_rows(session: Session, run_id: str) -> int:
    result = session.execute(
        delete(BenchAskResult).where(BenchAskResult.run_id == run_id)
    )
    session.commit()
    return int(result.rowcount or 0)


def reimport_jsonl_file(
    session: Session,
    path: Path,
    *,
    skip_existing_run: bool = True,
    replace_existing: bool = False,
) -> tuple[int, str]:
    """
    Re-insert rows from a JSONL archive into Postgres.
    Returns (inserted_count, status) where status is ok|skipped|empty|error|replaced.
    """
    if not path.is_file():
        return 0, "error"
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return 0, "empty"

    first = json.loads(lines[0])
    run_id = str(first.get("run_id") or path.stem)
    expected = len(lines)
    db_count = run_id_row_count(session, run_id)

    if replace_existing and db_count > 0:
        delete_run_rows(session, run_id)
        db_count = 0
    elif skip_existing_run and db_count >= expected:
        logger.info(
            "bench reimport skip complete run_id=%s path=%s db=%d jsonl=%d",
            run_id,
            path.name,
            db_count,
            expected,
        )
        return 0, "skipped"
    elif skip_existing_run and db_count > 0:
        # Partial Postgres row set — replace so Grafana gets full Q&A text.
        logger.info(
            "bench reimport replacing partial run_id=%s path=%s db=%d jsonl=%d",
            run_id,
            path.name,
            db_count,
            expected,
        )
        delete_run_rows(session, run_id)

    inserted = 0
    for line in lines:
        payload = json.loads(line)
        row = row_from_jsonl_dict(payload)
        if not row.run_id:
            row.run_id = run_id
        insert_bench_result(session, row)
        inserted += 1
    status = "replaced" if db_count > 0 else "ok"
    return inserted, status


def reimport_jsonl_dir(
    session: Session,
    directory: Path,
    *,
    skip_existing_run: bool = True,
    replace_existing: bool = False,
) -> dict[str, int]:
    """Import all *.jsonl under directory. Returns counters."""
    stats = {
        "files": 0,
        "inserted": 0,
        "skipped": 0,
        "empty": 0,
        "error": 0,
        "replaced": 0,
    }
    if not directory.is_dir():
        return stats
    for path in sorted(directory.glob("*.jsonl")):
        stats["files"] += 1
        try:
            n, status = reimport_jsonl_file(
                session,
                path,
                skip_existing_run=skip_existing_run,
                replace_existing=replace_existing,
            )
        except Exception:  # noqa: BLE001
            logger.exception("bench reimport failed path=%s", path)
            stats["error"] += 1
            continue
        stats["inserted"] += n
        if status == "skipped":
            stats["skipped"] += 1
        elif status == "empty":
            stats["empty"] += 1
        elif status == "error":
            stats["error"] += 1
        elif status == "replaced":
            stats["replaced"] += 1
    return stats
