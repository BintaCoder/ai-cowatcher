"""Tests for grounded refusal fallbacks."""

from __future__ import annotations

from ai_cowatcher.agent.grounding import (
    grounded_fallback_answer,
    is_refusal_answer,
)


def test_detects_not_sure_refusal():
    assert is_refusal_answer("Not sure yet — nothing's made that clear so far.")
    assert not is_refusal_answer("They're starting a lightning-round game.")


def test_grounded_fallback_from_scene_payload():
    payload = [
        {
            "scene_id": "s1",
            "start_ts": 1.0,
            "end_ts": 3.0,
            "transcript": "The lightning round begins. Stop it!",
            "caption": "",
            "score": 0.5,
        }
    ]
    answer = grounded_fallback_answer("What just happened?", [payload])
    assert answer
    assert "lightning" in answer.lower()
    assert "not sure" not in answer.lower()


def test_who_fallback_uses_dialogue():
    payload = [
        {
            "transcript": "Joey had an imaginary childhood friend named Maurice.",
            "caption": "",
        }
    ]
    answer = grounded_fallback_answer("Who is that guy?", [payload])
    assert answer
    assert "name" in answer.lower() or "maurice" in answer.lower() or "joey" in answer.lower()
