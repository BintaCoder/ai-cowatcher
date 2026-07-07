"""CLI entrypoint for the ingest event consumer worker."""

from __future__ import annotations

import logging

from ai_cowatcher.config import get_settings
from ai_cowatcher.messaging.consumer_worker import run_worker


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    run_worker(settings)


if __name__ == "__main__":
    main()
