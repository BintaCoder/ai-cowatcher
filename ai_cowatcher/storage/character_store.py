"""Character-intelligence graph persistence (Neo4j) with a spoiler-safe query.

The store holds three kinds of records per title:

* ``CharacterIdentity`` — a unified character linking face + speaker clusters.
* ``CharacterAppearance`` — a timestamped scene appearance.
* ``CharacterRelationship`` — a timestamped edge whose ``known_since_ts`` is the
  earliest playback position at which the relationship is revealed on screen.

``character_lookup`` enforces the **same spoiler principle as scene_lookup**:
only appearances/relationships with timestamp ``<= current_ts`` are ever
returned, so a viewer can never learn something that hasn't aired yet.
"""

from __future__ import annotations

import logging
from typing import Protocol

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import (
    CharacterAppearance,
    CharacterIdentity,
    CharacterLookupResult,
    CharacterRelationship,
)

logger = logging.getLogger(__name__)


class CharacterStore(Protocol):
    def replace_title_characters(
        self,
        title_id: str,
        characters: list[CharacterIdentity],
        appearances: list[CharacterAppearance],
        relationships: list[CharacterRelationship],
    ) -> None:
        ...

    def character_lookup(
        self, title_id: str, key: str | None, current_ts: float
    ) -> CharacterLookupResult | None:
        ...

    def close(self) -> None:
        ...


def _resolve_character(
    characters: list[CharacterIdentity],
    appearances: list[CharacterAppearance],
    key: str | None,
    current_ts: float,
) -> CharacterIdentity | None:
    """Resolve a character by id/name/cluster, else the most recent on screen.

    An empty/None key models a pronoun like "him"/"her": we return whoever is
    (most recently) on screen at ``current_ts``.
    """
    if key:
        needle = key.strip().lower()
        for character in characters:
            if character.character_id.lower() == needle:
                return character
        for character in characters:
            if character.name and character.name.lower() == needle:
                return character
        for character in characters:
            if character.name and needle in character.name.lower():
                return character
        for character in characters:
            clusters = {c.lower() for c in character.face_cluster_ids}
            clusters |= {c.lower() for c in character.speaker_cluster_ids}
            if needle in clusters:
                return character

    visible = [ap for ap in appearances if ap.start_ts <= current_ts]
    if not visible:
        return None
    latest = max(visible, key=lambda ap: ap.start_ts)
    by_id = {character.character_id: character for character in characters}
    return by_id.get(latest.character_id)


def resolve_lookup(
    title_id: str,
    characters: list[CharacterIdentity],
    appearances: list[CharacterAppearance],
    relationships: list[CharacterRelationship],
    key: str | None,
    current_ts: float,
) -> CharacterLookupResult | None:
    """Pure, spoiler-safe resolution shared by the in-memory store and tests."""
    character = _resolve_character(characters, appearances, key, current_ts)
    if character is None:
        return None

    own_appearances = [
        ap for ap in appearances if ap.character_id == character.character_id
    ]
    visible_appearances = tuple(
        sorted(
            (ap for ap in own_appearances if ap.start_ts <= current_ts),
            key=lambda ap: ap.start_ts,
        )
    )

    names = {c.character_id: c.name for c in characters}
    own_relationships = [
        rel
        for rel in relationships
        if character.character_id in (rel.source_id, rel.target_id)
    ]
    visible_relationships = tuple(
        _with_names(rel, names)
        for rel in sorted(own_relationships, key=lambda r: r.known_since_ts)
        if rel.known_since_ts <= current_ts
    )

    hidden = any(ap.start_ts > current_ts for ap in own_appearances) or any(
        rel.known_since_ts > current_ts for rel in own_relationships
    )

    return CharacterLookupResult(
        character=character,
        appearances=visible_appearances,
        relationships=visible_relationships,
        spoiler_filtered=hidden,
    )


def _with_names(
    rel: CharacterRelationship, names: dict[str, str | None]
) -> CharacterRelationship:
    return CharacterRelationship(
        title_id=rel.title_id,
        source_id=rel.source_id,
        target_id=rel.target_id,
        rel_type=rel.rel_type,
        summary=rel.summary,
        known_since_ts=rel.known_since_ts,
        scene_id=rel.scene_id,
        source_name=rel.source_name or names.get(rel.source_id),
        target_name=rel.target_name or names.get(rel.target_id),
    )


class InMemoryCharacterStore:
    """Process-local character store. Used in mock mode, demos, and tests."""

    def __init__(self) -> None:
        self._characters: dict[str, list[CharacterIdentity]] = {}
        self._appearances: dict[str, list[CharacterAppearance]] = {}
        self._relationships: dict[str, list[CharacterRelationship]] = {}

    def replace_title_characters(
        self,
        title_id: str,
        characters: list[CharacterIdentity],
        appearances: list[CharacterAppearance],
        relationships: list[CharacterRelationship],
    ) -> None:
        self._characters[title_id] = list(characters)
        self._appearances[title_id] = list(appearances)
        self._relationships[title_id] = list(relationships)

    def character_lookup(
        self, title_id: str, key: str | None, current_ts: float
    ) -> CharacterLookupResult | None:
        return resolve_lookup(
            title_id,
            self._characters.get(title_id, []),
            self._appearances.get(title_id, []),
            self._relationships.get(title_id, []),
            key,
            current_ts,
        )

    def close(self) -> None:  # noqa: D401 - nothing to release
        return None


class Neo4jCharacterStore:
    """Neo4j-backed character graph with spoiler filtering enforced in Cypher."""

    def __init__(self, settings: Settings, driver=None):
        self._settings = settings
        self._database = settings.neo4j_database
        if driver is not None:
            self._driver = driver
        else:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )

    def replace_title_characters(
        self,
        title_id: str,
        characters: list[CharacterIdentity],
        appearances: list[CharacterAppearance],
        relationships: list[CharacterRelationship],
    ) -> None:
        with self._driver.session(database=self._database) as session:
            session.execute_write(self._write_graph, title_id, characters, appearances, relationships)

    @staticmethod
    def _write_graph(tx, title_id, characters, appearances, relationships) -> None:
        tx.run(
            "MATCH (c:Character {title_id: $title_id}) DETACH DELETE c",
            title_id=title_id,
        )
        tx.run(
            "MATCH (s:Scene {title_id: $title_id}) DETACH DELETE s",
            title_id=title_id,
        )
        for character in characters:
            tx.run(
                """
                MERGE (c:Character {character_id: $character_id})
                SET c.title_id = $title_id,
                    c.name = $name,
                    c.face_cluster_ids = $face_cluster_ids,
                    c.speaker_cluster_ids = $speaker_cluster_ids,
                    c.first_ts = $first_ts
                """,
                character_id=character.character_id,
                title_id=character.title_id,
                name=character.name,
                face_cluster_ids=list(character.face_cluster_ids),
                speaker_cluster_ids=list(character.speaker_cluster_ids),
                first_ts=character.first_ts,
            )
        for appearance in appearances:
            tx.run(
                """
                MERGE (s:Scene {title_id: $title_id, scene_id: $scene_id})
                SET s.start_ts = $start_ts, s.end_ts = $end_ts
                WITH s
                MATCH (c:Character {character_id: $character_id})
                MERGE (c)-[a:APPEARS_IN]->(s)
                SET a.start_ts = $start_ts, a.end_ts = $end_ts
                """,
                title_id=title_id,
                scene_id=appearance.scene_id,
                start_ts=appearance.start_ts,
                end_ts=appearance.end_ts,
                character_id=appearance.character_id,
            )
        for rel in relationships:
            tx.run(
                """
                MATCH (a:Character {character_id: $source_id})
                MATCH (b:Character {character_id: $target_id})
                MERGE (a)-[r:RELATIONSHIP {rel_type: $rel_type}]->(b)
                SET r.summary = $summary,
                    r.known_since_ts = $known_since_ts,
                    r.scene_id = $scene_id
                """,
                source_id=rel.source_id,
                target_id=rel.target_id,
                rel_type=rel.rel_type,
                summary=rel.summary,
                known_since_ts=rel.known_since_ts,
                scene_id=rel.scene_id,
            )

    def character_lookup(
        self, title_id: str, key: str | None, current_ts: float
    ) -> CharacterLookupResult | None:
        with self._driver.session(database=self._database) as session:
            return session.execute_read(
                self._read_lookup, title_id, key, current_ts
            )

    @staticmethod
    def _read_lookup(tx, title_id, key, current_ts) -> CharacterLookupResult | None:
        node = None
        if key:
            record = tx.run(
                """
                MATCH (c:Character {title_id: $title_id})
                WHERE toLower(c.character_id) = toLower($key)
                   OR toLower(coalesce(c.name, '')) = toLower($key)
                   OR toLower(coalesce(c.name, '')) CONTAINS toLower($key)
                   OR any(x IN c.face_cluster_ids WHERE toLower(x) = toLower($key))
                   OR any(x IN c.speaker_cluster_ids WHERE toLower(x) = toLower($key))
                RETURN c LIMIT 1
                """,
                title_id=title_id,
                key=key,
            ).single()
            if record is not None:
                node = record["c"]
        if node is None:
            record = tx.run(
                """
                MATCH (c:Character {title_id: $title_id})-[a:APPEARS_IN]->(:Scene)
                WHERE a.start_ts <= $current_ts
                RETURN c ORDER BY a.start_ts DESC LIMIT 1
                """,
                title_id=title_id,
                current_ts=current_ts,
            ).single()
            if record is not None:
                node = record["c"]
        if node is None:
            return None

        character = _node_to_identity(node)

        appearance_records = tx.run(
            """
            MATCH (c:Character {character_id: $character_id})-[a:APPEARS_IN]->(s:Scene)
            RETURN s.scene_id AS scene_id, a.start_ts AS start_ts, a.end_ts AS end_ts,
                   (a.start_ts <= $current_ts) AS visible
            ORDER BY a.start_ts
            """,
            character_id=character.character_id,
            current_ts=current_ts,
        )
        appearances: list[CharacterAppearance] = []
        hidden = False
        for row in appearance_records:
            if row["visible"]:
                appearances.append(
                    CharacterAppearance(
                        character_id=character.character_id,
                        scene_id=row["scene_id"],
                        start_ts=float(row["start_ts"]),
                        end_ts=float(row["end_ts"]),
                    )
                )
            else:
                hidden = True

        relationship_records = tx.run(
            """
            MATCH (c:Character {character_id: $character_id})-[r:RELATIONSHIP]-(o:Character)
            RETURN r.rel_type AS rel_type, r.summary AS summary,
                   r.known_since_ts AS known_since_ts, r.scene_id AS scene_id,
                   startNode(r).character_id AS source_id,
                   startNode(r).name AS source_name,
                   endNode(r).character_id AS target_id,
                   endNode(r).name AS target_name,
                   (r.known_since_ts <= $current_ts) AS visible
            ORDER BY r.known_since_ts
            """,
            character_id=character.character_id,
            current_ts=current_ts,
        )
        relationships: list[CharacterRelationship] = []
        for row in relationship_records:
            if not row["visible"]:
                hidden = True
                continue
            relationships.append(
                CharacterRelationship(
                    title_id=title_id,
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    rel_type=row["rel_type"],
                    summary=row["summary"],
                    known_since_ts=float(row["known_since_ts"]),
                    scene_id=row["scene_id"],
                    source_name=row["source_name"],
                    target_name=row["target_name"],
                )
            )

        return CharacterLookupResult(
            character=character,
            appearances=tuple(appearances),
            relationships=tuple(relationships),
            spoiler_filtered=hidden,
        )

    def close(self) -> None:
        self._driver.close()


def _node_to_identity(node) -> CharacterIdentity:
    return CharacterIdentity(
        character_id=node["character_id"],
        title_id=node["title_id"],
        name=node.get("name"),
        face_cluster_ids=tuple(node.get("face_cluster_ids") or ()),
        speaker_cluster_ids=tuple(node.get("speaker_cluster_ids") or ()),
        first_ts=float(node.get("first_ts") or 0.0),
    )


_IN_MEMORY_SINGLETON: InMemoryCharacterStore | None = None


def build_character_store(settings: Settings) -> CharacterStore:
    """Neo4j store when configured, else a process-local in-memory store.

    The in-memory fallback is a singleton so that, within a single process
    (e.g. tests or a combined demo), enrichment writes are visible to lookups.
    """
    if settings.neo4j_enabled:
        return Neo4jCharacterStore(settings)
    global _IN_MEMORY_SINGLETON
    if _IN_MEMORY_SINGLETON is None:
        _IN_MEMORY_SINGLETON = InMemoryCharacterStore()
    return _IN_MEMORY_SINGLETON
