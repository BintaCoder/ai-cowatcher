"""CLI entrypoint for offline title ingestion."""

from __future__ import annotations

import argparse
import logging
import sys

from ai_cowatcher.config import get_settings
from ai_cowatcher.ingestion.pipeline import run_ingestion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a title into the offline co-watcher index")
    parser.add_argument("--title-id", required=True, help="Unique title identifier")
    parser.add_argument("--video", required=True, help="Path to the source video file")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if this title was already processed",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    try:
        result = run_ingestion(args.title_id, args.video, force=args.force)
    except Exception:
        logging.exception("Ingestion failed")
        return 1

    if result.skipped:
        print(f"Skipped existing title {result.title_id} ({result.scene_count} scenes)")
    else:
        print(f"Ingested {result.title_id} with {result.scene_count} scenes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
