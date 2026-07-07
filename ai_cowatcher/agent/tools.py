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

CHARACTER_LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "character_lookup",
        "description": (
            "Look up in-story character intelligence for the title the viewer is "
            "watching, restricted to what has aired up to their current position. "
            "Use this for questions about who a person on screen is, whether the "
            "viewer has seen them before, how two characters know each other, or "
            "what their relationship is so far. Results are spoiler-safe: only "
            "appearances and relationships already revealed by the current timestamp "
            "are returned, so future reveals are never leaked. Returns the resolved "
            "character, their prior appearances, and known relationships."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "character": {
                    "type": "string",
                    "description": (
                        "The character's name or id if known. Leave empty (or omit) "
                        "to refer to the person currently on screen, e.g. for "
                        "'have I seen him before?'."
                    ),
                },
            },
            "required": [],
        },
    },
}

KNOWLEDGE_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "knowledge_search",
        "description": (
            "Search the curated knowledge base for public, spoiler-insensitive facts "
            "about the title: actor biographies, crew/creator info, sports statistics, "
            "and general production trivia. Use this when the viewer asks about real-world "
            "people or production details that do NOT depend on how far they have watched. "
            "Unlike scene_lookup and character_lookup, this tool has NO playback-position "
            "filter — it is safe by design because the knowledge base is vetted offline "
            "and does not include plot spoilers or live web content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": "Natural-language search query for public facts.",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Optional category filter, e.g. actor_biography, crew, "
                        "sports_statistics, production."
                    ),
                },
            },
            "required": ["query_text"],
        },
    },
}

AGENT_TOOLS = [SCENE_LOOKUP_TOOL]

USER_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "user_memory",
        "description": (
            "Retrieve this viewer's own prior questions and answers for the title "
            "they are currently watching. Use when they refer to something they "
            "said earlier ('as I mentioned', 'what did I ask before', continuity). "
            "Returns only this viewer's history for this title — never other users. "
            "Does not use playback position; it is personal chat history, not plot data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["turns", "summary"],
                    "description": (
                        "'turns' returns the last N messages; 'summary' returns a "
                        "short recap of recent exchanges."
                    ),
                },
                "max_turns": {
                    "type": "integer",
                    "description": "How many recent turns to return (default 10).",
                },
            },
            "required": [],
        },
    },
}
