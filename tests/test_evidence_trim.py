"""Evidence payload trim for LLM cost control."""

from ai_cowatcher.retrieval.evidence import (
    compact_scene_dict,
    compact_scene_payloads,
    scene_evidence_json,
)


def test_compact_clips_fields_and_drops_clusters():
    scene = {
        "scene_id": "s1",
        "transcript": "a" * 500,
        "caption": "b" * 500,
        "face_cluster_ids": [1, 2],
        "speaker_cluster_ids": ["x"],
        "start_ts": 10.0,
    }
    out = compact_scene_dict(scene, max_chars_per_field=50)
    assert len(out["transcript"]) <= 50
    assert len(out["caption"]) <= 50
    assert "face_cluster_ids" not in out
    assert out["start_ts"] == 10.0


def test_scene_evidence_top_k():
    scenes = [
        {"scene_id": f"s{i}", "transcript": f"line {i} " * 40, "caption": ""}
        for i in range(6)
    ]
    payload = scene_evidence_json([scenes], max_scenes=2, max_chars_per_field=40)
    assert "s0" in payload and "s1" in payload
    assert "s2" not in payload
    compacted = compact_scene_payloads([scenes], max_scenes=2, max_chars_per_field=40)
    assert len(compacted[0]) == 2
