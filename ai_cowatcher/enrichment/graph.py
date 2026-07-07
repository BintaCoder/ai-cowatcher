"""LangGraph offline enrichment: scene clusters -> character graph in Neo4j.

Multi-step and offline (never on the request path), so a small LangGraph
``StateGraph`` is a natural fit and keeps the stages explicit:

    load_scenes -> link_identities -> resolve_names
                -> build_relationships -> persist_graph

Each node is a thin wrapper over a pure function in this package; only the
final ``persist_graph`` node has side effects (writing to the character store).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import (
    CharacterAppearance,
    CharacterIdentity,
    CharacterRelationship,
    SceneEventRecord,
)
from ai_cowatcher.enrichment.identity import link_identities
from ai_cowatcher.enrichment.naming import resolve_names
from ai_cowatcher.enrichment.relationships import build_appearances, build_relationships
from ai_cowatcher.storage.character_store import CharacterStore

logger = logging.getLogger(__name__)

SceneLoader = Callable[[str], tuple[list[SceneEventRecord], list[str]]]


@dataclass
class CharacterGraphResult:
    title_id: str
    characters: list[CharacterIdentity]
    appearances: list[CharacterAppearance]
    relationships: list[CharacterRelationship]
    persisted: bool = False


class _State(TypedDict, total=False):
    title_id: str
    scenes: list[SceneEventRecord]
    cast_names: list[str]
    characters: list[CharacterIdentity]
    appearances: list[CharacterAppearance]
    relationships: list[CharacterRelationship]
    persisted: bool


def build_character_graph(
    store: CharacterStore,
    *,
    min_cooccur: int = 1,
    loader: SceneLoader | None = None,
):
    """Compile the LangGraph enrichment graph. Imports langgraph lazily."""
    from langgraph.graph import END, START, StateGraph

    def load_scenes(state: _State) -> _State:
        if state.get("scenes") is not None:
            return {}
        if loader is None:
            raise ValueError("No scenes provided and no loader configured")
        scenes, cast_names = loader(state["title_id"])
        return {"scenes": scenes, "cast_names": cast_names}

    def link_identities_node(state: _State) -> _State:
        characters = link_identities(
            state["title_id"], state.get("scenes", []), min_cooccur=min_cooccur
        )
        return {"characters": characters}

    def resolve_names_node(state: _State) -> _State:
        scenes = state.get("scenes", [])
        characters = state.get("characters", [])
        appearances = build_appearances(characters, scenes)
        named = resolve_names(
            characters, appearances, scenes, state.get("cast_names", []) or []
        )
        return {"characters": named, "appearances": appearances}

    def relationships_node(state: _State) -> _State:
        scenes = state.get("scenes", [])
        characters = state.get("characters", [])
        appearances = state.get("appearances") or build_appearances(characters, scenes)
        relationships = build_relationships(state["title_id"], characters, scenes)
        return {"appearances": appearances, "relationships": relationships}

    def persist_graph(state: _State) -> _State:
        store.replace_title_characters(
            state["title_id"],
            state.get("characters", []),
            state.get("appearances", []),
            state.get("relationships", []),
        )
        return {"persisted": True}

    graph = StateGraph(_State)
    graph.add_node("load_scenes", load_scenes)
    graph.add_node("link_identities", link_identities_node)
    graph.add_node("resolve_names", resolve_names_node)
    graph.add_node("build_relationships", relationships_node)
    graph.add_node("persist_graph", persist_graph)

    graph.add_edge(START, "load_scenes")
    graph.add_edge("load_scenes", "link_identities")
    graph.add_edge("link_identities", "resolve_names")
    graph.add_edge("resolve_names", "build_relationships")
    graph.add_edge("build_relationships", "persist_graph")
    graph.add_edge("persist_graph", END)
    return graph.compile()


def run_character_enrichment(
    settings: Settings,
    *,
    title_id: str,
    scenes: list[SceneEventRecord],
    cast_names: list[str],
    store: CharacterStore,
) -> CharacterGraphResult:
    """Run the enrichment graph for one title and persist to the store."""
    compiled = build_character_graph(
        store, min_cooccur=settings.character_link_min_cooccur
    )
    final: dict[str, Any] = compiled.invoke(
        {"title_id": title_id, "scenes": scenes, "cast_names": cast_names}
    )
    return CharacterGraphResult(
        title_id=title_id,
        characters=final.get("characters", []),
        appearances=final.get("appearances", []),
        relationships=final.get("relationships", []),
        persisted=bool(final.get("persisted")),
    )
