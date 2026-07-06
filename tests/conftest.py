"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from ai_cowatcher.agent.metrics import reset_ask_telemetry


@pytest.fixture(autouse=True)
def reset_pilot_metrics():
    reset_ask_telemetry()
    yield
    reset_ask_telemetry()
