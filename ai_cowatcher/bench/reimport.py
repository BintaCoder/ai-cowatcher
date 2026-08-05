"""Re-import ask-bench JSONL archives into Postgres for Grafana."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    args = parser.parse_args(argv)

    factory = ensure_bench_schema()
    skip = not args.force
    with factory() as session:
        if args.file is not None:
            n, status = reimport_jsonl_file(session, args.file, skip_existing_run=skip)
            print(f"{args.file}: inserted={n} status={status}")
            return 0 if status in ("ok", "skipped", "empty") else 1

        stats = reimport_jsonl_dir(session, args.dir, skip_existing_run=skip)
        print(
            "Reimport complete: "
            f"files={stats['files']} inserted={stats['inserted']} "
            f"skipped={stats['skipped']} empty={stats['empty']} error={stats['error']}"
        )
        return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
