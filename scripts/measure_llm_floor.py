#!/usr/bin/env python3
"""Measure approximate fresh-generation latency floor for the fast-tier model.

Calls the conversation completion client with a minimal system+user prompt and
tiny max_tokens (no scene retrieve, no tools). Use this to decide whether the
~800ms /ask/stream average is achievable per-request or only as a blended
cache-hit / cache-miss average.

Usage:
  MOCK_MODE=false GEMINI_API_KEY=... PYTHONPATH=. \\
    python scripts/measure_llm_floor.py --runs 5

With MOCK_MODE=true the script still runs but records mock-client floor only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_cowatcher.agent.completion import (  # noqa: E402
    LiteLLMCompletionClient,
    MockCompletionClient,
)
from ai_cowatcher.config import get_settings  # noqa: E402
from ai_cowatcher.providers.litellm_env import configure_litellm_env  # noqa: E402


MINIMAL_SYSTEM = (
    "You are a brief co-watch friend. Reply in one short sentence only."
)
MINIMAL_USER = "Who is talking right now? (no tools — answer from this line alone: Ross speaks.)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure LLM fresh-gen latency floor")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Completion budget for floor probe (keep small)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_litellm_env(settings)
    model = settings.conversation_fast_model
    effort = (settings.llm_reasoning_effort or "").strip() or "minimal"

    if settings.mock_mode:
        client = MockCompletionClient()
    else:
        client = LiteLLMCompletionClient(settings)

    messages = [
        {"role": "system", "content": MINIMAL_SYSTEM},
        {"role": "user", "content": MINIMAL_USER},
    ]

    latencies: list[float] = []
    texts: list[str] = []
    total_runs = max(0, args.warmup) + max(1, args.runs)
    for i in range(total_runs):
        t0 = time.perf_counter()
        result = client.complete(
            model=model,
            messages=messages,
            tools=None,
            temperature=0.2,
            max_tokens=args.max_tokens,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        if i >= args.warmup:
            latencies.append(ms)
            texts.append((result.content or "").strip()[:200])

    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = (
        latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))]
        if latencies_sorted
        else 0.0
    )
    mean = statistics.mean(latencies) if latencies else 0.0

    # If floor already near/above 800ms on fresh gen, 800ms avg needs cache blend.
    floor_statement: str
    if settings.mock_mode:
        floor_statement = (
            "MOCK_MODE floor only — not representative of Gemini. "
            "Re-run with MOCK_MODE=false to measure the real fast-tier floor."
        )
    elif mean >= 800:
        floor_statement = (
            f"Fresh-generation floor mean≈{mean:.0f}ms (p50={p50:.0f}, p95={p95:.0f}) "
            "is already near or above the 800ms average target. Achieving ~800ms "
            "average requires a blended mix of cache hits (exact/semantic/free-gate) "
            "and cache misses — not a per-request guarantee on every fresh answer."
        )
    elif mean >= 600:
        floor_statement = (
            f"Fresh-generation floor mean≈{mean:.0f}ms leaves little headroom under "
            "the 800ms average once retrieve/gate overhead is added. Treat 800ms as a "
            "blended traffic average; fresh misses will often land ~floor + ~100–400ms."
        )
    else:
        floor_statement = (
            f"Fresh-generation floor mean≈{mean:.0f}ms is below 800ms; the remaining "
            "gap is retrieve + evidence + gate + network. Per-request ~800ms is possible "
            "on thin prompts but still benefits from cache hits for a stable average."
        )

    summary = {
        "model": model,
        "mock_mode": settings.mock_mode,
        "reasoning_effort": effort,
        "max_tokens": args.max_tokens,
        "runs": len(latencies),
        "mean_ms": round(mean, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "min_ms": round(min(latencies), 2) if latencies else None,
        "max_ms": round(max(latencies), 2) if latencies else None,
        "sample_answers": texts[:3],
        "floor_statement": floor_statement,
        "target_avg_ms": 800,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
