"""Pre-warm Redis + Qdrant QA cache for demo titles and common real-pattern Q&A."""

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

# Common real-traffic phrasing patterns (quality-review / bench seeds), not only
# scripted demos. Answers are spoiler-safe within the stated current_ts window.
COMMON_PATTERN_TEMPLATES: list[dict[str, Any]] = [
    {
        "question": "What's going on?",
        "answer": "They're mid-scene right now — stick with what just played.",
        "current_ts": 60.0,
    },
    {
        "question": "what's going on in this scene?",
        "answer": "This beat is still unfolding on screen; nothing later is locked yet.",
        "current_ts": 60.0,
    },
    {
        "question": "Who is that?",
        "answer": "Whoever just spoke or held the frame — match them to the dialogue you heard.",
        "current_ts": 45.0,
    },
    {
        "question": "Who's on screen?",
        "answer": "The person holding the frame right now — follow their line.",
        "current_ts": 45.0,
    },
    {
        "question": "Who is that guy talking right now?",
        "answer": "The guy speaking in this beat — name them only from what you've seen so far.",
        "current_ts": 45.0,
    },
    {
        "question": "What just happened?",
        "answer": "Something just landed in this beat — keep it to the last few seconds.",
        "current_ts": 30.0,
    },
    {
        "question": "What happened?",
        "answer": "A beat just passed; stay with the lines and reaction you just heard.",
        "current_ts": 30.0,
    },
    {
        "question": "Why is he upset?",
        "answer": "From what we've seen so far, the mood just tightened — no later reveal.",
        "current_ts": 90.0,
    },
    {
        "question": "Why does he look so upset?",
        "answer": "Something in this beat clearly hit a nerve on screen.",
        "current_ts": 90.0,
    },
    {
        "question": "Who just walked in?",
        "answer": "Whoever just entered the frame in this beat.",
        "current_ts": 75.0,
    },
    {
        "question": "What did they just say?",
        "answer": "Match the last line you heard on the track — no paraphrase from later.",
        "current_ts": 50.0,
    },
]

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
            "question": "Who is that guy talking right now?",
            "answer": "That's Ross talking in this beat.",
            "current_ts": 45.0,
        },
        {
            "question": "What's going on?",
            "answer": "They're in the middle of a messy apartment conversation.",
            "current_ts": 60.0,
        },
        {
            "question": "What's going on in this scene?",
            "answer": "Apartment hangout energy — follow the last awkward exchange.",
            "current_ts": 60.0,
        },
        {
            "question": "What just happened?",
            "answer": "Someone just walked into an awkward beat with the group.",
            "current_ts": 90.0,
        },
        {
            "question": "Why does he look so upset?",
            "answer": "Ross is wound up about how the building sees him.",
            "current_ts": 120.0,
        },
        {
            "question": "Why is everyone reacting that way?",
            "answer": "The group's reaction matches the awkward beat that just landed.",
            "current_ts": 120.0,
        },
        {
            "question": "Who just walked in?",
            "answer": "Whoever just entered Ross's place in this beat.",
            "current_ts": 50.0,
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
            "question": "Who is that?",
            "answer": "The character holding the frame in this pilot beat.",
            "current_ts": 30.0,
        },
        {
            "question": "What's going on?",
            "answer": "A short pilot setup is playing; stick to what you've heard so far.",
            "current_ts": 40.0,
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
            "question": "What's going on?",
            "answer": "A light pilot beat is in progress; no later plot yet.",
            "current_ts": 35.0,
        },
        {
            "question": "What's the joke?",
            "answer": "A light beat just passed; laugh at the timing, not a later spoiler.",
            "current_ts": 40.0,
        },
    ],
}


def merge_common_patterns(
    title_pairs: dict[str, list[dict[str, Any]]],
    *,
    include_common: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Merge COMMON_PATTERN_TEMPLATES into each title without duplicating questions."""
    if not include_common:
        return {tid: list(pairs) for tid, pairs in title_pairs.items()}
    out: dict[str, list[dict[str, Any]]] = {}
    for tid, pairs in title_pairs.items():
        existing_q = {
            str(row.get("question") or "").strip().lower()
            for row in pairs
            if str(row.get("question") or "").strip()
        }
        merged = list(pairs)
        for tmpl in COMMON_PATTERN_TEMPLATES:
            q = str(tmpl.get("question") or "").strip()
            if not q or q.lower() in existing_q:
                continue
            merged.append(dict(tmpl))
            existing_q.add(q.lower())
        out[tid] = merged
    return out


def load_pairs_from_question_set(
    path: Path,
    *,
    title_id: str,
    current_ts: float = 60.0,
) -> list[tuple[str, dict[str, Any]]]:
    """Seed warm pairs from a quality-review / bench question JSON list.

    Accepts either:
    - [{id, text|question, ...}] (friends_ross_questions style)
    - persona_eval.json-style {"questions": [{question, current_ts, ...}]}
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]]
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        rows = list(data["questions"])
    elif isinstance(data, list):
        rows = list(data)
    else:
        raise SystemExit(f"Unsupported question set shape in {path}")

    out: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("kind") or row.get("intent") or "").lower()
        # Skip pure social/filler — those hit the free gate, not the QA cache.
        if kind in {"social", "filler"}:
            continue
        question = str(row.get("question") or row.get("text") or "").strip()
        if not question:
            continue
        ts = float(row.get("current_ts", current_ts))
        # Placeholder spoiler-safe answer; real demos should override via --file.
        answer = (
            str(row.get("answer") or "").strip()
            or f"Stay with the on-screen beat for: {question.rstrip('?')}."
        )
        out.append(
            (
                str(row.get("title_id") or title_id),
                {
                    "question": question,
                    "answer": answer,
                    "current_ts": ts,
                    **(
                        {"persona_id": row["persona_id"]}
                        if row.get("persona_id")
                        else {}
                    ),
                },
            )
        )
    return out


def load_pairs(
    path: Path | None,
    title_id: str | None,
    *,
    include_common: bool = True,
    question_set: Path | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Return list of (title_id, pair_dict)."""
    if question_set is not None:
        tid = title_id or "friends_ross"
        pairs = load_pairs_from_question_set(question_set, title_id=tid)
        if include_common:
            common_as_map = merge_common_patterns({tid: []}, include_common=True)
            # Prepend common patterns first so demos prioritize them.
            seeded: list[tuple[str, dict[str, Any]]] = [
                (tid, row) for row in common_as_map.get(tid, [])
            ]
            seen = {str(r.get("question") or "").strip().lower() for _, r in seeded}
            for pair_tid, row in pairs:
                q = str(row.get("question") or "").strip().lower()
                if q and q not in seen:
                    seeded.append((pair_tid, row))
                    seen.add(q)
            return seeded
        return pairs

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
            merged = merge_common_patterns(
                {str(k): list(v) for k, v in data.items()},
                include_common=include_common,
            )
            out = []
            for tid, pairs in merged.items():
                for row in pairs:
                    out.append((str(tid), row))
            return out
        raise SystemExit("JSON must be a list of pairs or a title_id map")

    demos = merge_common_patterns(DEFAULT_DEMO_PAIRS, include_common=include_common)
    if title_id:
        pairs = demos.get(title_id)
        if not pairs:
            raise SystemExit(
                f"No built-in pairs for {title_id!r}. "
                f"Known: {', '.join(sorted(demos))} or pass --file / --question-set."
            )
        return [(title_id, p) for p in pairs]

    out = []
    for tid, pairs in demos.items():
        for row in pairs:
            out.append((tid, row))
    return out


def warm(
    pairs: list[tuple[str, dict[str, Any]]],
    *,
    dry_run: bool = False,
    persona_ids: list[str] | None = None,
) -> int:
    settings = get_settings()
    configure_litellm_env(settings)
    from ai_cowatcher.personas.loader import list_personas, resolve_persona_id

    default_persona = resolve_persona_id(
        None, default_id=getattr(settings, "default_persona_id", None)
    )
    if persona_ids:
        personas = [resolve_persona_id(p, default_id=default_persona) for p in persona_ids]
    else:
        # Seed every known persona so demo cache hits match watch UI selection.
        personas = [p.persona_id for p in list_personas()] or [default_persona]

    if dry_run:
        for tid, row in pairs:
            for pid in personas:
                print(
                    f"[dry-run] {tid} persona={pid} ts={row.get('current_ts')} "
                    f"q={row.get('question')!r}"
                )
        return len(pairs) * len(personas)

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
        # Optional per-row persona override; otherwise seed all personas.
        row_personas = personas
        if row.get("persona_id"):
            row_personas = [
                resolve_persona_id(str(row["persona_id"]), default_id=default_persona)
            ]
        for pid in row_personas:
            # Light per-persona phrasing only when answer has no custom persona already.
            seeded_answer = answer
            if "persona_id" not in row and pid == "witty_friend" and "—" not in answer:
                seeded_answer = answer.rstrip(".!") + " — neat beat."
            cache.store(tid, current_ts, question, seeded_answer, persona_id=pid)
            n += 1
            print(
                f"Warm store: title={tid} persona={pid} ts={current_ts:.1f} "
                f"q={question[:60]!r}"
            )
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
        "--question-set",
        type=Path,
        default=None,
        help=(
            "Quality-review / bench question JSON (e.g. benchmarks/friends_ross_questions.json) "
            "to seed common real phrasing"
        ),
    )
    parser.add_argument(
        "--no-common-patterns",
        action="store_true",
        help="Skip merging COMMON_PATTERN_TEMPLATES into built-in / file maps",
    )
    parser.add_argument(
        "--persona-id",
        action="append",
        default=None,
        help="Persona id to seed (repeatable). Default: all packaged personas.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pairs only; do not write cache",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=get_settings().log_level)
    try:
        pairs = load_pairs(
            args.file,
            args.title_id,
            include_common=not args.no_common_patterns,
            question_set=args.question_set,
        )
        n = warm(pairs, dry_run=args.dry_run, persona_ids=args.persona_id)
    except SystemExit:
        raise
    except Exception:
        logger.exception("warm_qa_cache failed")
        return 1
    print(f"Warmed {n} Q&A pair(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
