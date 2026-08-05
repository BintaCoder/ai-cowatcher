"""Evaluate QA-cache semantic cosine thresholds on labeled question pairs.

Near-duplicate pairs (same intent, different phrasing) should score ≥ threshold.
Genuinely different pairs should score < threshold (false-positive control).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


@dataclass(frozen=True)
class LabeledPair:
    question_a: str
    question_b: str
    same_intent: bool
    note: str = ""


@dataclass(frozen=True)
class PairScore:
    pair: LabeledPair
    score: float


@dataclass(frozen=True)
class ThresholdReport:
    threshold: float
    true_positive: int
    false_negative: int
    true_negative: int
    false_positive: int
    near_dup_count: int
    different_count: int

    @property
    def near_dup_hit_rate(self) -> float:
        if self.near_dup_count == 0:
            return 0.0
        return self.true_positive / self.near_dup_count

    @property
    def false_positive_rate(self) -> float:
        if self.different_count == 0:
            return 0.0
        return self.false_positive / self.different_count

    def ok(
        self,
        *,
        min_near_dup_hit_rate: float = 0.8,
        max_false_positive_rate: float = 0.0,
    ) -> bool:
        return (
            self.near_dup_hit_rate >= min_near_dup_hit_rate
            and self.false_positive_rate <= max_false_positive_rate
        )


def load_labeled_pairs(path: Path) -> list[LabeledPair]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("pairs") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"Expected list of pairs in {path}")
    out: list[LabeledPair] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        a = str(row.get("question_a") or row.get("a") or "").strip()
        b = str(row.get("question_b") or row.get("b") or "").strip()
        if not a or not b:
            continue
        same = bool(row.get("same_intent", row.get("near_dup", False)))
        out.append(
            LabeledPair(
                question_a=a,
                question_b=b,
                same_intent=same,
                note=str(row.get("note") or ""),
            )
        )
    return out


def score_pairs(
    pairs: Sequence[LabeledPair],
    embedder: Embedder,
) -> list[PairScore]:
    texts: list[str] = []
    for p in pairs:
        texts.append(p.question_a)
        texts.append(p.question_b)
    vectors = embedder.embed_texts(texts)
    if len(vectors) != len(texts):
        raise RuntimeError("embedder returned wrong vector count")
    scored: list[PairScore] = []
    for i, pair in enumerate(pairs):
        va = vectors[2 * i]
        vb = vectors[2 * i + 1]
        scored.append(PairScore(pair=pair, score=cosine(va, vb)))
    return scored


def evaluate_threshold(
    scored: Sequence[PairScore],
    threshold: float,
) -> ThresholdReport:
    tp = fn = tn = fp = 0
    near = 0
    different = 0
    for item in scored:
        hit = item.score >= threshold
        if item.pair.same_intent:
            near += 1
            if hit:
                tp += 1
            else:
                fn += 1
        else:
            different += 1
            if hit:
                fp += 1
            else:
                tn += 1
    return ThresholdReport(
        threshold=threshold,
        true_positive=tp,
        false_negative=fn,
        true_negative=tn,
        false_positive=fp,
        near_dup_count=near,
        different_count=different,
    )


def sweep_thresholds(
    scored: Sequence[PairScore],
    thresholds: Sequence[float],
    *,
    min_near_dup_hit_rate: float = 0.8,
    max_false_positive_rate: float = 0.0,
) -> list[ThresholdReport]:
    reports = [evaluate_threshold(scored, t) for t in thresholds]
    return reports


def pick_threshold(
    reports: Sequence[ThresholdReport],
    *,
    min_near_dup_hit_rate: float = 0.8,
    max_false_positive_rate: float = 0.0,
) -> ThresholdReport | None:
    """Prefer the lowest safe threshold (more hits) among reports that pass quality gates."""
    ok = [
        r
        for r in reports
        if r.ok(
            min_near_dup_hit_rate=min_near_dup_hit_rate,
            max_false_positive_rate=max_false_positive_rate,
        )
    ]
    if not ok:
        return None
    # Lowest threshold among safe ones → highest legitimate hit rate.
    return min(ok, key=lambda r: r.threshold)


def report_to_dict(report: ThresholdReport) -> dict[str, Any]:
    return {
        "threshold": report.threshold,
        "true_positive": report.true_positive,
        "false_negative": report.false_negative,
        "true_negative": report.true_negative,
        "false_positive": report.false_positive,
        "near_dup_count": report.near_dup_count,
        "different_count": report.different_count,
        "near_dup_hit_rate": round(report.near_dup_hit_rate, 4),
        "false_positive_rate": round(report.false_positive_rate, 4),
        "ok": report.ok(),
    }
