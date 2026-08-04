"""Tests for companion personality packs and AskRequest persona fields."""

from __future__ import annotations

from ai_cowatcher.api.schemas import AskRequest
from ai_cowatcher.agent.intent_tags import (
    build_merged_system_prompt,
    social_body_or_default,
)
from ai_cowatcher.config import Settings
from ai_cowatcher.personas.loader import (
    get_persona,
    list_personas,
    normalize_companion_gender,
    resolve_persona_id,
)


def test_personas_load_from_package():
    personas = list_personas()
    ids = {p.persona_id for p in personas}
    assert "easygoing_friend" in ids
    assert "witty_friend" in ids
    assert len(personas) >= 2
    witty = get_persona("witty_friend")
    assert witty.canned_social_reply
    assert witty.tts.rate > 0
    assert witty.traits.humor >= witty.traits.formality


def test_resolve_default_persona_from_settings():
    settings = Settings(MOCK_MODE=True, DEFAULT_PERSONA_ID="witty_friend")
    resolved = resolve_persona_id(None, default_id=settings.default_persona_id)
    assert resolved == "witty_friend"
    bad = resolve_persona_id("not_a_real_persona", default_id="easygoing_friend")
    assert bad == "easygoing_friend"


def test_merged_system_prompt_includes_persona_block():
    persona = get_persona("easygoing_friend")
    prompt = build_merged_system_prompt(persona, companion_gender="female")
    assert "Companion persona" in prompt
    assert persona.canned_social_reply in prompt
    assert "female companion friend" in prompt
    # Tone guidance must not invent facts
    assert "Never change facts" in prompt or "tone only" in prompt.lower()


def test_social_body_prefers_persona_canned():
    witty = get_persona("witty_friend")
    assert social_body_or_default("", canned=witty.canned_social_reply) == witty.canned_social_reply
    assert social_body_or_default("Hi there!", canned=witty.canned_social_reply) == "Hi there!"


def test_ask_request_accepts_persona_and_gender():
    req = AskRequest(
        title_id="demo",
        current_ts=10.0,
        question="Who is that?",
        user_id="u1",
        persona_id="witty_friend",
        companion_gender="male",
    )
    assert req.persona_id == "witty_friend"
    assert req.companion_gender == "male"

    bare = AskRequest(
        title_id="demo",
        current_ts=10.0,
        question="Hi",
        user_id="u1",
    )
    assert bare.persona_id is None
    assert bare.companion_gender is None


def test_normalize_companion_gender():
    assert normalize_companion_gender("Female") == "female"
    assert normalize_companion_gender("neutral") == "neutral"
    assert normalize_companion_gender("other") is None
