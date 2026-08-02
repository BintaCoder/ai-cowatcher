"""Tests for couch-friend answer brevity."""

from __future__ import annotations

from ai_cowatcher.agent.brevity import enforce_brief_answer, wants_detailed_answer


def test_default_answer_is_one_short_sentence():
    long = (
        "Based on what has aired so far, the detectives arrive at the crime scene "
        "and then they talk about the knife and afterward they interview the neighbor "
        "who claims to have seen nothing."
    )
    brief = enforce_brief_answer(long, "What just happened?")
    assert brief.count(".") <= 1
    assert len(brief.split()) <= 30
    assert "Based on what has aired" not in brief or len(brief.split()) <= 28


def test_detail_request_allows_more():
    assert wants_detailed_answer("tell me more about that")
    text = (
        "They found the knife. Then they questioned the neighbor. "
        "Finally the detective left the room quietly."
    )
    out = enforce_brief_answer(text, "tell me more in detail")
    assert "knife" in out.lower()
    assert len(out.split()) <= 80


def test_already_short_unchanged():
    short = "That's Marcus at the door."
    assert enforce_brief_answer(short, "Who is that?") == short
