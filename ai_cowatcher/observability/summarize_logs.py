"""CLI entrypoint to summarize ask_request JSON logs from stdin."""

from __future__ import annotations

import json
import sys

from ai_cowatcher.observability.ask_telemetry import summarize_ask_log_lines


def main() -> int:
    summary = summarize_ask_log_lines(sys.stdin.readlines())
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
