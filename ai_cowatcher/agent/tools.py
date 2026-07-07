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

CAST_LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "cast_lookup",
        "description": (
            "Look up the public cast/actor list for the title the viewer is watching. "
            "Use this when the viewer asks who an actor is, who plays a character, or "
            "for the names of the actors. Cast lists are public information and are not "
            "plot spoilers. Returns actors and the characters they play."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title_name": {
                    "type": "string",
                    "description": (
                        "The movie or show name to search for. Use the known title of "
                        "what the viewer is watching if provided in context."
                    ),
                },
                "year": {
                    "type": "integer",
                    "description": "Optional release year to disambiguate the title.",
                },
            },
            "required": ["title_name"],
        },
    },
}

AGENT_TOOLS = [SCENE_LOOKUP_TOOL]
