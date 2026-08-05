"""Push Settings API keys into os.environ for LiteLLM providers."""

from __future__ import annotations

import os

from ai_cowatcher.config import Settings


def configure_litellm_env(settings: Settings) -> None:
    """LiteLLM reads provider keys from the process environment, not Settings."""
    _set_env_if_missing("OPENAI_API_KEY", settings.openai_api_key)
    _set_env_if_missing("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    if settings.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    elif settings.gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key
    # Local Ollama (LiteLLM reads OLLAMA_API_BASE).
    base = (settings.ollama_api_base or "").strip()
    if base:
        os.environ["OLLAMA_API_BASE"] = base
    # Shared HTTP client for connection reuse across Gemini/LiteLLM calls.
    if not settings.mock_mode:
        try:
            from ai_cowatcher.agent.completion import ensure_litellm_http_pool

            ensure_litellm_http_pool()
        except Exception:  # noqa: BLE001 — never block app start
            pass


def _set_env_if_missing(name: str, value: str) -> None:
    if value:
        os.environ[name] = value
