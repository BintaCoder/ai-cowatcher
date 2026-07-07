"""CLI entrypoint for the ingest event consumer worker."""

from __future__ import annotations

import logging

from prometheus_client import start_http_server

from ai_cowatcher.config import get_settings
from ai_cowatcher.messaging.consumer_worker import run_worker


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    if settings.prometheus_enabled:
        start_http_server(settings.worker_metrics_port)
    run_worker(settings)


if __name__ == "__main__":
    main()
