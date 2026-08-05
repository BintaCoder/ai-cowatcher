"""Build multimodal messages: text question + retrieved scene audio clips."""

from __future__ import annotations

import base64
import logging
from typing import Any

from ai_cowatcher.config import Settings
from ai_cowatcher.storage.object_store import ObjectStore

logger = logging.getLogger(__name__)

_MULTIMODAL_SYSTEM = """You are the viewer's friend on the couch. Answer from the scene audio \
and text provided only. One short spoken sentence for a mid-watch reply. No spoilers beyond \
what the clips/text make clear. No bullet points. Names only if spoken/written in the material."""


def collect_audio_keys(tool_payloads: list[Any], *, max_clips: int) -> list[str]:
    """Extract unique audio_object_key values from scene_lookup tool payloads."""
    keys: list[str] = []
    seen: set[str] = set()
    for payload in tool_payloads:
        scenes = payload if isinstance(payload, list) else []
        if isinstance(payload, dict) and "audio_object_key" in payload:
            scenes = [payload]
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            key = scene.get("audio_object_key")
            if not key or not isinstance(key, str):
                continue
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
            if len(keys) >= max_clips:
                return keys
    return keys


def load_scene_clips(
    keys: list[str],
    object_store: ObjectStore,
) -> list[tuple[str, bytes]]:
    clips: list[tuple[str, bytes]] = []
    for key in keys:
        data = object_store.get_bytes(key)
        if data:
            clips.append((key, data))
        else:
            logger.warning("Missing scene audio object key=%s", key)
    return clips


def build_multimodal_messages(
    *,
    question: str,
    tool_payloads: list[Any],
    clips: list[tuple[str, bytes]],
) -> list[dict[str, Any]]:
    """LiteLLM/OpenAI-style multimodal messages with input_audio parts (WAV base64)."""
    text_bits: list[str] = [
        f"Viewer question: {question}",
        "",
        "Scene text evidence (spoiler-safe retrieval):",
    ]
    for payload in tool_payloads:
        scenes = payload if isinstance(payload, list) else []
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            if "transcript" not in scene and "caption" not in scene:
                continue
            sid = scene.get("scene_id", "?")
            text_bits.append(
                f"- [{sid}] transcript: {scene.get('transcript', '')} | "
                f"caption: {scene.get('caption', '')}"
            )

    content: list[dict[str, Any]] = [
        {"type": "text", "text": "\n".join(text_bits)},
    ]
    for key, data in clips:
        b64 = base64.b64encode(data).decode("ascii")
        content.append({"type": "text", "text": f"(scene audio: {key})"})
        content.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": b64,
                    "format": "wav",
                },
            }
        )

    return [
        {"role": "system", "content": _MULTIMODAL_SYSTEM},
        {"role": "user", "content": content},
    ]


def should_use_multimodal(settings: Settings, *, joke_mode: bool = False) -> bool:
    if joke_mode:
        return False
    if settings.mock_mode:
        return False
    return bool(settings.multimodal_scene_audio_enabled)
