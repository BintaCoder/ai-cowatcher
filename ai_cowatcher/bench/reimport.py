"""Re-import ask-bench JSONL archives into Postgres for Grafana."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_cowatcher.config import get_settings
from ai_cowatcher.bench.store import ensure_bench_schema, reimport_jsonl_dir, reimport_jsonl_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-import benchmarks/results/*.jsonl into Postgres bench_ask_result "
            "(idempotent: skips run_ids already present)."
        )
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory of *.jsonl archives (default: benchmarks/results)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Import a single JSONL file instead of a directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Insert even if run_id already exists (can duplicate rows)",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete each run_id in Postgres before re-importing from JSONL",
    )
    args = parser.parse_args(argv)

    factory = ensure_bench_schema()
    settings = get_settings()
    print(
        f"Postgres: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )
    skip = not args.force and not args.replace_existing
    replace = args.replace_existing or args.force
    with factory() as session:
        if args.file is not None:
            n, status = reimport_jsonl_file(
                session,
                args.file,
                skip_existing_run=skip,
                replace_existing=replace,
            )
            print(f"{args.file}: inserted={n} status={status}")
            return 0 if status in ("ok", "skipped", "empty", "replaced") else 1

        stats = reimport_jsonl_dir(
            session,
            args.dir,
            skip_existing_run=skip,
            replace_existing=replace,
        )
        print(
            "Reimport complete: "
            f"files={stats['files']} inserted={stats['inserted']} "
            f"replaced={stats['replaced']} skipped={stats['skipped']} "
            f"empty={stats['empty']} error={stats['error']}"
        )
        return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
