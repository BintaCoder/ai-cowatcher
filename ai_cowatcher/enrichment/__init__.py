"""Offline character-graph enrichment (LangGraph).

Runs once per title after scene ingest, off the request path. It links the
per-scene face and speaker clusters into unified character identities,
resolves names from cast metadata where possible, builds a timestamped
appearance + relationship graph, and persists it to Neo4j.
"""

from ai_cowatcher.enrichment.graph import (
    CharacterGraphResult,
    build_character_graph,
    run_character_enrichment,
)

__all__ = [
    "CharacterGraphResult",
    "build_character_graph",
    "run_character_enrichment",
]
