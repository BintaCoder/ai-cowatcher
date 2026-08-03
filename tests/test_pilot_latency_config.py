"""Pilot latency / prod config guards for cost control."""

import pytest

from ai_cowatcher.config import Settings


def test_prod_requires_low_latency():
    s = Settings(
        APP_ENV="production",
        PILOT_LOW_LATENCY=False,
        UTTERANCE_GATE_STRATEGY="merged",
        MOCK_MODE=True,
    )
    with pytest.raises(ValueError, match="PILOT_LOW_LATENCY"):
        s.validate_pilot_latency_config()


def test_prod_requires_merged_gate():
    s = Settings(
        APP_ENV="production",
        PILOT_LOW_LATENCY=True,
        UTTERANCE_GATE_STRATEGY="prompt",
        MOCK_MODE=True,
    )
    with pytest.raises(ValueError, match="UTTERANCE_GATE_STRATEGY"):
        s.validate_pilot_latency_config()


def test_legacy_reachable_when_low_latency_off():
    s = Settings(
        APP_ENV="development",
        PILOT_LOW_LATENCY=False,
        UTTERANCE_GATE_STRATEGY="prompt",
        MOCK_MODE=True,
    )
    assert s.legacy_multi_tool_path_reachable() is True
    s.validate_pilot_latency_config()  # warn only in non-prod


def test_prod_ok_defaults():
    s = Settings(
        APP_ENV="production",
        PILOT_LOW_LATENCY=True,
        UTTERANCE_GATE_STRATEGY="merged",
        MOCK_MODE=True,
    )
    s.validate_pilot_latency_config()
    assert s.legacy_multi_tool_path_reachable() is False
