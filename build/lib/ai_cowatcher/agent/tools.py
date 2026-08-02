"""Tool schemas exposed to the conversation agent."""

from __future__ import annotations

SCENE_LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "scene_lookup",
        "description": (
            "Search ingested scene events for a title up to the viewer's current playback "
            "position. Returns matching scenes in chronological order. "
            "Only scenes that have already aired are visible."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": "Natural-language search query describing what to look for.",
                },
            },
            "required": ["query_text"],
        },
    },
}

AGENT_TOOLS = [SCENE_LOOKUP_TOOL]
