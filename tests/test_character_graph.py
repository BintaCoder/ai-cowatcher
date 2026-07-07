"""Unit tests for the offline character-intelligence enrichment.

Covers speaker-cluster mapping, face+speaker identity linking, timestamped
appearance/relationship building, the LangGraph enrichment graph, and the
spoiler-safe character store query.
"""

from __future__ import annotations

import pytest

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import SceneBoundary, SceneEventRecord, SpeakerSegment
from ai_cowatcher.enrichment.identity import link_identities
from ai_cowatcher.enrichment.relationships import build_appearances, build_relationships
from ai_cowatcher.ingestion.diarization import speaker_clusters_for_scenes
from ai_cowatcher.storage.character_store import InMemoryCharacterStore

TITLE = "drama-unit"


def _scene(
    scene_id: str,
    start: float,
    end: float,
    transcript: str,
    faces: list[str],
    speakers: list[str],
) -> SceneEventRecord:
    return SceneEventRecord(
        scene_id=scene_id,
        title_id=TITLE,
        start_ts=start,
        end_ts=end,
        transcript=transcript,
        caption="",
        face_cluster_ids=faces,
        speaker_cluster_ids=speakers,
    )


def _reveal_scenes() -> list[SceneEventRecord]:
    return [
        _scene("s0000", 0.0, 20.0, "A man walks in.", [f"{TITLE}-fc-0"], [f"{TITLE}-sc-SPEAKER_00"]),
        _scene("s0001", 30.0, 50.0, "He sits alone.", [f"{TITLE}-fc-0"], [f"{TITLE}-sc-SPEAKER_00"]),
        _scene(
            "s0002",
            120.0,
            140.0,
            "You are my sister and I never knew.",
            [f"{TITLE}-fc-0", f"{TITLE}-fc-1"],
            [f"{TITLE}-sc-SPEAKER_00", f"{TITLE}-sc-SPEAKER_01"],
        ),
    ]


def test_speaker_clusters_for_scenes_maps_overlaps():
    segments = [
        SpeakerSegment(0.0, 15.0, "SPEAKER_00"),
        SpeakerSegment(15.0, 40.0, "SPEAKER_01"),
    ]
    scenes = [
        SceneBoundary(index=0, start_ts=0.0, end_ts=12.0),
        SceneBoundary(index=1, start_ts=12.0, end_ts=30.0),
    ]
    mapped = speaker_clusters_for_scenes(segments, scenes, TITLE)
    assert mapped[0] == [f"{TITLE}-sc-SPEAKER_00"]
    assert set(mapped[1]) == {f"{TITLE}-sc-SPEAKER_00", f"{TITLE}-sc-SPEAKER_01"}


def test_link_identities_links_face_and_speaker():
    characters = link_identities(TITLE, _reveal_scenes(), min_cooccur=1)
    assert len(characters) == 2
    first = characters[0]
    assert first.face_cluster_ids == (f"{TITLE}-fc-0",)
    assert f"{TITLE}-sc-SPEAKER_00" in first.speaker_cluster_ids
    assert first.first_ts == 0.0


def test_relationship_reveal_is_timestamped_at_dialogue():
    scenes = _reveal_scenes()
    characters = link_identities(TITLE, scenes, min_cooccur=1)
    relationships = build_relationships(TITLE, characters, scenes)
    assert len(relationships) == 1
    rel = relationships[0]
    assert rel.rel_type == "sibling"
    assert rel.known_since_ts == 120.0
    assert "sibling" in rel.summary.lower()


def test_store_filters_relationship_before_reveal():
    scenes = _reveal_scenes()
    characters = link_identities(TITLE, scenes, min_cooccur=1)
    appearances = build_appearances(characters, scenes)
    relationships = build_relationships(TITLE, characters, scenes)

    store = InMemoryCharacterStore()
    store.replace_title_characters(TITLE, characters, appearances, relationships)

    before = store.character_lookup(TITLE, None, current_ts=40.0)
    assert before is not None
    assert before.appearances  # seen before
    assert before.relationships == ()  # sibling reveal (120s) is hidden
    assert before.spoiler_filtered is True

    after = store.character_lookup(TITLE, None, current_ts=130.0)
    assert after is not None
    assert any(rel.rel_type == "sibling" for rel in after.relationships)


def test_enrichment_graph_populates_store():
    langgraph = pytest.importorskip("langgraph")
    assert langgraph is not None
    from ai_cowatcher.enrichment.graph import run_character_enrichment

    store = InMemoryCharacterStore()
    result = run_character_enrichment(
        Settings(MOCK_MODE=True),
        title_id=TITLE,
        scenes=_reveal_scenes(),
        cast_names=[],
        store=store,
    )
    assert result.persisted is True
    assert len(result.characters) == 2
    assert any(rel.rel_type == "sibling" for rel in result.relationships)
    lookup = store.character_lookup(TITLE, None, current_ts=130.0)
    assert lookup is not None
    assert lookup.relationships
