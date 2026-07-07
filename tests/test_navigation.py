"""Tests for navigation time parsing, ordinals, and event detection."""

from __future__ import annotations

from ai_cowatcher.domain import SceneEventRecord
from ai_cowatcher.ingestion.event_detection import build_title_events, detect_credits_start
from ai_cowatcher.navigation.ordinal import parse_ordinal
from ai_cowatcher.navigation.time_parse import parse_seek_timestamp


def test_parse_seek_ten_minutes():
    assert parse_seek_timestamp("take me to 10 minutes") == 600.0


def test_parse_seek_clock_time():
    assert parse_seek_timestamp("go to 1:30") == 90.0


def test_parse_seek_hours_minutes():
    assert parse_seek_timestamp("jump to 1 hour 5 minutes") == 3900.0


def test_parse_ordinal_second_fight():
    ordinal, query = parse_ordinal("take me to the second fight")
    assert ordinal == 2
    assert "fight" in query.lower()


def test_detect_fight_and_credits_events():
    scenes = [
        SceneEventRecord("s0000", "demo", 0, 10, "they start to fight", "two men fighting", []),
        SceneEventRecord("s0001", "demo", 40, 50, "another brawl begins", "punching scene", []),
        SceneEventRecord("s0002", "demo", 170, 180, "credits roll", "directed by someone", []),
    ]
    events, credits_ts = build_title_events("demo", scenes)
    fight_events = [event for event in events if event.event_type == "fight"]
    assert len(fight_events) == 2
    assert fight_events[1].ordinal == 2
    assert credits_ts is not None


def test_detect_credits_tail():
    scenes = [
        SceneEventRecord("s0000", "demo", 0, 80, "dialogue", "kitchen scene", []),
        SceneEventRecord("s0001", "demo", 170, 180, "end credits", "cast list scrolls", []),
    ]
    assert detect_credits_start(scenes) == 170.0
