"""Compact scene tool payloads for LLM context (cost control)."""

from __future__ import annotations

import json
from typing import Any


def _clip(text: str, max_chars: int) -> str:
    raw = (text or "").strip()
    if max_chars <= 0 or len(raw) <= max_chars:
        return raw
    return raw[: max(0, max_chars - 1)].rstrip() + "…"


def compact_scene_dict(
    scene: dict[str, Any],
    *,
    max_chars_per_field: int = 200,
) -> dict[str, Any]:
    """Copy scene tool dict with truncated transcript/caption; keep spoiler times."""
    out = dict(scene)
    out["transcript"] = _clip(str(scene.get("transcript") or ""), max_chars_per_field)
    out["caption"] = _clip(str(scene.get("caption") or ""), max_chars_per_field)
    # Drop bulky optional fields not needed for short spoken answers
    out.pop("face_cluster_ids", None)
    out.pop("speaker_cluster_ids", None)
    return out


def compact_scene_payloads(
    payloads: list[Any],
    *,
    max_scenes: int = 2,
    max_chars_per_field: int = 200,
) -> list[list[dict[str, Any]]]:
    """
    Payloads from scene_lookup are typically a list of scene dicts (or nested lists
    from multi-tool rounds). Flatten one list of scenes, take top-k by order,
    compact text fields.
    """
    scenes: list[dict[str, Any]] = []
    for item in payloads:
        if isinstance(item, list):
            for scene in item:
                if isinstance(scene, dict):
                    scenes.append(scene)
        elif isinstance(item, dict):
            scenes.append(item)
    limited = scenes[: max(1, max_scenes)] if max_scenes > 0 else scenes
    compacted = [
        compact_scene_dict(s, max_chars_per_field=max_chars_per_field) for s in limited
    ]
    return [compacted] if compacted else []


def scene_evidence_json(
    payloads: list[Any],
    *,
    max_scenes: int = 2,
    max_chars_per_field: int = 200,
) -> str:
    compacted = compact_scene_payloads(
        payloads, max_scenes=max_scenes, max_chars_per_field=max_chars_per_field
    )
    if not compacted or not compacted[0]:
        return "(no scene matches)"
    return json.dumps(compacted[0], ensure_ascii=False)
