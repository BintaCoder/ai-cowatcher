"""Offline estimate of evidence-size impact on merged prompt tokens (no Gemini calls)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from ai_cowatcher.agent.intent_tags import MERGED_SYSTEM_PROMPT, build_merged_user_turn
from ai_cowatcher.observability.llm_cost import (
    DEFAULT_INPUT_USD_PER_MTOK,
    DEFAULT_OUTPUT_USD_PER_MTOK,
    estimate_messages_tokens,
    estimate_usd,
)
from ai_cowatcher.retrieval.evidence import scene_evidence_json

# Fixed completion budget assumption for spoken short answers.
ASSUMED_COMPLETION_TOKENS = 80


def _fake_scenes(n: int, transcript_chars: int) -> list[dict]:
    body = "x" * transcript_chars
    return [
        {
            "scene_id": f"s{i:04d}",
            "start_ts": float(i * 10),
            "end_ts": float(i * 10 + 8),
            "transcript": body,
            "caption": body,
            "score": 0.9 - i * 0.05,
        }
        for i in range(n)
    ]


def _messages(scene_json: str) -> list[dict]:
    user = build_merged_user_turn(
        title_id="bench_title",
        current_ts=120.0,
        question="What just happened on screen?",
        scene_evidence=scene_json,
    )
    return [
        {"role": "system", "content": MERGED_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _stats(xs: list[int], *, questions: int) -> dict:
    mean_t = statistics.mean(xs)
    cost = (
        estimate_usd(
            prompt_tokens=int(mean_t),
            completion_tokens=ASSUMED_COMPLETION_TOKENS,
            input_usd_per_mtok=DEFAULT_INPUT_USD_PER_MTOK,
            output_usd_per_mtok=DEFAULT_OUTPUT_USD_PER_MTOK,
        )
        * questions
    )
    return {
        "mean_prompt_tokens": round(mean_t, 1),
        "p50_prompt_tokens": int(statistics.median(xs)),
        "est_usd_for_n_asks": round(cost, 6),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare merged prompt tokens before/after evidence trim"
    )
    parser.add_argument("--questions", type=int, default=50, help="Simulated questions")
    parser.add_argument("--titles", type=int, default=3, help="Title count (for reporting)")
    parser.add_argument(
        "--raw-scenes", type=int, default=5, help="Scenes before trim (legacy)"
    )
    parser.add_argument(
        "--raw-chars", type=int, default=800, help="Transcript chars per field before"
    )
    parser.add_argument("--max-scenes", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=200)
    parser.add_argument(
        "--candidates",
        type=str,
        default="",
        help=(
            "Optional comma list of max_scenes:max_chars candidates to compare, "
            "e.g. '3:280,2:200,2:160' (defaults alone if empty)"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON summary path",
    )
    args = parser.parse_args(argv)

    candidates: list[tuple[int, int]] = []
    if args.candidates.strip():
        for part in args.candidates.split(","):
            part = part.strip()
            if not part:
                continue
            scenes_s, chars_s = part.split(":", 1)
            candidates.append((int(scenes_s), int(chars_s)))
    if not candidates:
        candidates = [(args.max_scenes, args.max_chars)]

    before_tokens: list[int] = []
    candidate_tokens: dict[str, list[int]] = {
        f"scenes={s}_chars={c}": [] for s, c in candidates
    }
    for i in range(args.questions):
        # mild variance by index
        raw = _fake_scenes(args.raw_scenes + (i % 2), args.raw_chars + (i % 50))
        before = json.dumps(raw, ensure_ascii=False)
        before_tokens.append(estimate_messages_tokens(_messages(before)))
        for s, c in candidates:
            key = f"scenes={s}_chars={c}"
            after = scene_evidence_json(
                [raw], max_scenes=s, max_chars_per_field=c
            )
            candidate_tokens[key].append(estimate_messages_tokens(_messages(after)))

    before_s = _stats(before_tokens, questions=args.questions)
    after_map = {
        key: _stats(vals, questions=args.questions)
        for key, vals in candidate_tokens.items()
    }
    preferred = f"scenes={args.max_scenes}_chars={args.max_chars}"
    primary_key = preferred if preferred in after_map else next(iter(after_map))
    after_s = after_map[primary_key]
    reduction = 1.0 - (
        after_s["mean_prompt_tokens"] / before_s["mean_prompt_tokens"]
        if before_s["mean_prompt_tokens"]
        else 0.0
    )
    summary = {
        "questions": args.questions,
        "titles_reported": args.titles,
        "assumed_completion_tokens": ASSUMED_COMPLETION_TOKENS,
        "before_trim": before_s,
        "after_trim": after_s,
        "candidate_trim": after_map,
        "prompt_token_reduction_fraction": round(reduction, 4),
        "note": (
            "Token counts are ~4-char estimates for offline bench; "
            "run live /ask with metrics for provider-reported usage. "
            "Evidence trim does not touch spoiler filter (start_ts<=current_ts). "
            "Compare candidates with --candidates 3:280,2:200,2:160."
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
