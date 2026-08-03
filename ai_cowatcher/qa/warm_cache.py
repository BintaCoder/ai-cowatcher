"""Pre-warm Redis + Qdrant QA cache for demo titles."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from ai_cowatcher.config import get_settings
from ai_cowatcher.providers.litellm_env import configure_litellm_env
from ai_cowatcher.realtime.viewing_session import _build_embedder
from ai_cowatcher.storage.qdrant_store import QdrantSceneStore
from ai_cowatcher.storage.qa_cache import build_qa_cache

logger = logging.getLogger(__name__)

# Demo Q&A pairs for first-hit cache on scripted walkthroughs.
# Answers are spoiler-safe within the stated current_ts for short pilot clips.
DEFAULT_DEMO_PAIRS: dict[str, list[dict[str, Any]]] = {
    "friends_ross": [
        {
            "question": "Who is that?",
            "answer": "That's Ross — one of the friends.",
            "current_ts": 45.0,
        },
        {
            "question": "What's going on?",
            "answer": "They're in the middle of a messy apartment conversation.",
            "current_ts": 60.0,
        },
        {
            "question": "What just happened?",
            "answer": "Someone just walked into an awkward beat with the group.",
            "current_ts": 90.0,
        },
        {
            "question": "Make a joke about this",
            "answer": "If sitcom timing had a union, this pause would be steward.",
            "current_ts": 90.0,
        },
    ],
    "pilot-title-001": [
        {
            "question": "Who's on screen?",
            "answer": "A character from the pilot clip — follow what they've just said.",
            "current_ts": 30.0,
        },
        {
            "question": "What happened?",
            "answer": "A short setup scene just played; stick to what you've heard so far.",
            "current_ts": 45.0,
        },
    ],
    "pilot-title-002": [
        {
            "question": "Who is talking?",
            "answer": "Whoever just spoke in the pilot — the transcript will match it.",
            "current_ts": 25.0,
        },
        {
            "question": "What's the joke?",
            "answer": "A light beat just passed; laugh at the timing, not a later spoiler.",
            "current_ts": 40.0,
        },
    ],
}


def load_pairs(path: Path | None, title_id: str | None) -> list[tuple[str, dict[str, Any]]]:
    """Return list of (title_id, pair_dict)."""
    if path is not None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            # [{"title_id", "question", "answer", "current_ts"}, ...]
            out: list[tuple[str, dict[str, Any]]] = []
            for row in data:
                tid = str(row.get("title_id") or title_id or "")
                if not tid:
                    raise SystemExit("Each row needs title_id or pass --title-id")
                out.append((tid, row))
            return out
        if isinstance(data, dict):
            # {title_id: [pairs...]}
            out = []
            for tid, pairs in data.items():
                for row in pairs:
                    out.append((str(tid), row))
            return out
        raise SystemExit("JSON must be a list of pairs or a title_id map")

    if title_id:
        pairs = DEFAULT_DEMO_PAIRS.get(title_id)
        if not pairs:
            raise SystemExit(
                f"No built-in pairs for {title_id!r}. "
                f"Known: {', '.join(sorted(DEFAULT_DEMO_PAIRS))} or pass --file."
            )
        return [(title_id, p) for p in pairs]

    out = []
    for tid, pairs in DEFAULT_DEMO_PAIRS.items():
        for row in pairs:
            out.append((tid, row))
    return out


def warm(
    pairs: list[tuple[str, dict[str, Any]]],
    *,
    dry_run: bool = False,
) -> int:
    settings = get_settings()
    configure_litellm_env(settings)
    if dry_run:
        for tid, row in pairs:
            print(f"[dry-run] {tid} ts={row.get('current_ts')} q={row.get('question')!r}")
        return len(pairs)

    embedder = _build_embedder(settings)
    qdrant = QdrantSceneStore(settings)
    cache = build_qa_cache(
        settings,
        embedder=embedder,
        qdrant_client=getattr(qdrant, "_client", None),
    )
    if cache is None:
        raise SystemExit("QA cache disabled (QA_CACHE_ENABLED=false)")

    n = 0
    for tid, row in pairs:
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        current_ts = float(row.get("current_ts", 0.0))
        if not question or not answer:
            logger.warning("Skipping incomplete pair for %s", tid)
            continue
        cache.store(tid, current_ts, question, answer)
        n += 1
        print(f"Warm store: title={tid} ts={current_ts:.1f} q={question[:60]!r}")
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-populate exact + semantic QA cache for demo titles"
    )
    parser.add_argument(
        "--title-id",
        default=None,
        help="Warm only this title (uses built-in demo pairs if --file omitted)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="JSON file of pairs (list or title_id map)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pairs only; do not write cache",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=get_settings().log_level)
    try:
        pairs = load_pairs(args.file, args.title_id)
        n = warm(pairs, dry_run=args.dry_run)
    except SystemExit:
        raise
    except Exception:
        logger.exception("warm_qa_cache failed")
        return 1
    print(f"Warmed {n} Q&A pair(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
