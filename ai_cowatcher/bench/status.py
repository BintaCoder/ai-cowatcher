"""Print ask-bench Postgres vs JSONL archive health (for Grafana troubleshooting)."""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import func, select

from ai_cowatcher.config import get_settings
from ai_cowatcher.db.models import BenchAskResult


def _jsonl_line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose bench_ask_result vs JSONL archives.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="JSONL archive directory",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    print(
        f"Postgres target: {settings.postgres_host}:{settings.postgres_port}/"
        f"{settings.postgres_db} (user={settings.postgres_user})"
    )
    print("Grafana reads the same DB via docker host postgres:5432 when using make up-core.")

    factory = ensure_bench_schema()
    with factory() as session:
        total = session.scalar(select(func.count()).select_from(BenchAskResult)) or 0
        print(f"\nbench_ask_result total rows: {total}")

        if total:
            runs = session.execute(
                select(
                    BenchAskResult.run_id,
                    func.min(BenchAskResult.created_at),
                    func.max(BenchAskResult.created_at),
                    func.count(),
                )
                .group_by(BenchAskResult.run_id)
                .order_by(func.min(BenchAskResult.created_at).desc())
                .limit(20)
            )
            print("\nLatest runs in Postgres:")
            for run_id, started, ended, n in runs:
                print(f"  {run_id}  rows={n}  from={started}  to={ended}")

    archive_dir = args.dir
    if not archive_dir.is_dir():
        print(f"\nNo archive dir: {archive_dir}")
        return 1

    files = sorted(archive_dir.glob("*.jsonl"))
    print(f"\nJSONL archives ({len(files)} files in {archive_dir}):")
    missing = 0
    with factory() as session:
        for path in files:
            lines = _jsonl_line_count(path)
            run_id = path.stem
            db_count = session.scalar(
                select(func.count())
                .select_from(BenchAskResult)
                .where(BenchAskResult.run_id == run_id)
            ) or 0
            if db_count >= lines:
                status = "ok"
            elif db_count > 0:
                status = "partial"
                missing += 1
            else:
                status = "missing"
                missing += 1
            print(f"  {path.name}: jsonl={lines} postgres={db_count} [{status}]")

    if missing:
        print(
            "\nTo restore missing/partial runs:\n"
            "  make bench-reimport REPLACE=1\n"
            "  docker compose restart grafana"
        )
        return 1

    print("\nArchives match Postgres — if Grafana is empty, restart Grafana and hard-refresh Ask Bench.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
