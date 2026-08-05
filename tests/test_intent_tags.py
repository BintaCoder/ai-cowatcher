"""Tests for merged intent tags (gate + answer in one completion)."""

from __future__ import annotations

from ai_cowatcher.agent.intent_tags import (
    IntentStreamParser,
    parse_tagged_answer,
    social_body_or_default,
)


def test_parse_tagged_filler():
    parsed = parse_tagged_answer("[FILLER]")
    assert parsed.tag == "FILLER"
    assert parsed.body == ""


def test_parse_tagged_content():
    raw = "[CONTENT]\n\nThat's Detective Reyes in the foyer."
    parsed = parse_tagged_answer(raw)
    assert parsed.tag == "CONTENT"
    assert "Detective" in parsed.body


def test_parse_malformed_defaults_to_content():
    parsed = parse_tagged_answer("Just a normal answer without tags.")
    assert parsed.tag == "CONTENT"
    assert "normal answer" in parsed.body


def test_stream_parser_emits_tag_then_body():
    parser = IntentStreamParser()
    events: list[tuple[str, str]] = []
    for piece in ["[CON", "TENT]", "\n\n", "Hello ", "there"]:
        events.extend(parser.feed(piece))
    events.extend(parser.finish())
    assert ("tag", "CONTENT") in events
    body = "".join(v for k, v in events if k == "text")
    assert body == "Hello there"


def test_stream_parser_filler_only():
    parser = IntentStreamParser()
    events: list[tuple[str, str]] = []
    for piece in ["[FILLER]"]:
        events.extend(parser.feed(piece))
    events.extend(parser.finish())
    assert events[0] == ("tag", "FILLER")
    assert not any(k == "text" for k, _ in events)


def test_social_default():
    assert "show" in social_body_or_default("").lower()


def test_parse_tagged_prediction():
    raw = "[PREDICTION]\n\nGot it — logged 'she's lying'. No spoilers until later."
    parsed = parse_tagged_answer(raw)
    assert parsed.tag == "PREDICTION"
    assert "logged" in parsed.body.lower() or "lying" in parsed.body.lower()


def test_stream_parser_prediction():
    parser = IntentStreamParser()
    events: list[tuple[str, str]] = []
    for piece in ["[PRED", "ICTION]", "\n\n", "Locking it in."]:
        events.extend(parser.feed(piece))
    events.extend(parser.finish())
    assert ("tag", "PREDICTION") in events
    body = "".join(v for k, v in events if k == "text")
    assert "Locking" in body
