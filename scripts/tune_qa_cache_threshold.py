#!/usr/bin/env python3
"""Sweep QA_CACHE_SEMANTIC_THRESHOLD on labeled near-dup / different pairs.

Usage (offline mock vectors for CI):
  PYTHONPATH=. python scripts/tune_qa_cache_threshold.py --mock

Usage (real BGE embeddings — preferred for shipping threshold):
  PYTHONPATH=. MOCK_MODE=false python scripts/tune_qa_cache_threshold.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_cowatcher.qa.threshold_eval import (  # noqa: E402
    evaluate_threshold,
    load_labeled_pairs,
    pick_threshold,
    report_to_dict,
    score_pairs,
)


class _MockClusterEmbedder:
    """Deterministic vectors: near-dups share a cluster id for high cosine."""

    vector_size = 32

    def __init__(self, pairs_path: Path) -> None:
        from ai_cowatcher.qa.threshold_eval import load_labeled_pairs

        self._cluster: dict[str, int] = {}
        next_id = 0
        for p in load_labeled_pairs(pairs_path):
            if p.same_intent:
                a = p.question_a.strip().lower()
                b = p.question_b.strip().lower()
                if a not in self._cluster and b not in self._cluster:
                    self._cluster[a] = next_id
                    self._cluster[b] = next_id
                    next_id += 1
                elif a in self._cluster:
                    self._cluster[b] = self._cluster[a]
                else:
                    self._cluster[a] = self._cluster[b]
            else:
                for q in (p.question_a, p.question_b):
                    key = q.strip().lower()
                    if key not in self._cluster:
                        self._cluster[key] = next_id
                        next_id += 1

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            cid = self._cluster.get(text.strip().lower(), 999)
            vec = [0.0] * self.vector_size
            # Cluster-aligned unit axis so same-intent cosine ≈ 1.0
            idx = cid % self.vector_size
            vec[idx] = 1.0
            # Small orthogonal noise from text length so identical text still ~1.
            noise_idx = (cid * 7 + len(text)) % self.vector_size
            if noise_idx != idx:
                vec[noise_idx] = 0.02
            # normalize
            n = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / n for v in vec])
        return out


def _build_embedder(*, mock: bool, pairs_path: Path):
    if mock:
        return _MockClusterEmbedder(pairs_path)
    from ai_cowatcher.config import get_settings
    from ai_cowatcher.realtime.viewing_session import _build_embedder as build

    return build(get_settings())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune QA semantic cache threshold")
    parser.add_argument(
        "--pairs",
        type=Path,
        default=ROOT / "benchmarks" / "qa_cache_threshold_pairs.json",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.84,
        help="Lowest threshold in sweep",
    )
    parser.add_argument(
        "--stop",
        type=float,
        default=0.94,
        help="Highest threshold in sweep (inclusive-ish)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.02,
        help="Step size (prompt 6: e.g. 0.02)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use cluster mock embedder (CI) instead of BGE",
    )
    parser.add_argument(
        "--max-fpr",
        type=float,
        default=0.0,
        help="Max allowed false-positive rate on different pairs",
    )
    parser.add_argument(
        "--min-near-hit",
        type=float,
        default=0.8,
        help="Min near-dup hit rate required",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    pairs = load_labeled_pairs(args.pairs)
    if not pairs:
        print("No labeled pairs found", file=sys.stderr)
        return 1

    embedder = _build_embedder(mock=args.mock, pairs_path=args.pairs)
    scored = score_pairs(pairs, embedder)

    thresholds: list[float] = []
    t = args.start
    # inclusive stop within float noise
    while t <= args.stop + 1e-9:
        thresholds.append(round(t, 4))
        t += args.step

    reports = [evaluate_threshold(scored, thr) for thr in thresholds]
    chosen = pick_threshold(
        reports,
        min_near_dup_hit_rate=args.min_near_hit,
        max_false_positive_rate=args.max_fpr,
    )

    detail_scores = [
        {
            "question_a": s.pair.question_a,
            "question_b": s.pair.question_b,
            "same_intent": s.pair.same_intent,
            "score": round(s.score, 4),
            "note": s.pair.note,
        }
        for s in scored
    ]
    payload = {
        "pairs_file": str(args.pairs),
        "mock_embedder": args.mock,
        "pair_scores": detail_scores,
        "sweep": [report_to_dict(r) for r in reports],
        "recommended": report_to_dict(chosen) if chosen else None,
        "note": (
            "Prefer the lowest threshold that keeps false_positive_rate ≤ max-fpr "
            "and near_dup_hit_rate ≥ min-near-hit. Default shipping value lives in "
            "QA_CACHE_SEMANTIC_THRESHOLD / Settings.qa_cache_semantic_threshold."
        ),
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
