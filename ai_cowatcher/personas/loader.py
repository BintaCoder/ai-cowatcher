"""Load companion personas from packaged JSON files."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

CompanionGender = Literal["male", "female", "neutral"]

VALID_GENDERS: frozenset[str] = frozenset({"male", "female", "neutral"})
DEFAULT_PERSONA_ID = "easygoing_friend"
_KNOWN_FALLBACK_ORDER = ("easygoing_friend", "witty_friend", "calm_scout")

_PERSONAS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PersonaTraits:
    humor: float = 0.5
    formality: float = 0.3
    warmth: float = 0.7
    verbosity: float = 0.3


@dataclass(frozen=True)
class PersonaTts:
    rate: float = 1.0
    pitch: float = 1.0


@dataclass(frozen=True)
class Persona:
    persona_id: str
    display_name: str
    traits: PersonaTraits
    style_notes: str
    avoid: tuple[str, ...]
    canned_social_reply: str
    tts: PersonaTts

    def prompt_block(self, *, companion_gender: CompanionGender | None = None) -> str:
        """Tone-only persona block for system prompts (never facts / spoilers)."""
        traits = self.traits
        avoid = "; ".join(self.avoid) if self.avoid else "(none listed)"
        gender_line = ""
        if companion_gender in ("male", "female", "neutral"):
            gender_line = (
                f"\nVoice delivery: speak as a {companion_gender} companion friend "
                "(delivery only — never rewrite facts or invent plot)."
            )
        return (
            f"Companion persona: {self.display_name} ({self.persona_id})\n"
            f"Tone traits (0–1 scale): humor={traits.humor:.2f}, "
            f"formality={traits.formality:.2f}, warmth={traits.warmth:.2f}, "
            f"verbosity={traits.verbosity:.2f}.\n"
            f"Style: {self.style_notes}\n"
            f"Avoid: {avoid}\n"
            f"SOCIAL canned reply when tag is SOCIAL (prefer this wording): "
            f'"{self.canned_social_reply}"\n'
            "Persona rules: shape tone only. Never change facts, spoilers, don't-know "
            "wording rules, or invent plot outside tool evidence."
            f"{gender_line}"
        )


def _parse_persona(data: dict) -> Persona:
    traits_raw = data.get("traits") or {}
    tts_raw = data.get("tts") or {}
    avoid_raw = data.get("avoid") or []
    if isinstance(avoid_raw, str):
        avoid: tuple[str, ...] = (avoid_raw,)
    else:
        avoid = tuple(str(item) for item in avoid_raw)
    return Persona(
        persona_id=str(data["persona_id"]).strip(),
        display_name=str(data.get("display_name") or data["persona_id"]).strip(),
        traits=PersonaTraits(
            humor=float(traits_raw.get("humor", 0.5)),
            formality=float(traits_raw.get("formality", 0.3)),
            warmth=float(traits_raw.get("warmth", 0.7)),
            verbosity=float(traits_raw.get("verbosity", 0.3)),
        ),
        style_notes=str(data.get("style_notes") or "").strip(),
        avoid=avoid,
        canned_social_reply=str(
            data.get("canned_social_reply")
            or "Right here with you — ask about the show anytime."
        ).strip(),
        tts=PersonaTts(
            rate=float(tts_raw.get("rate", 1.0)),
            pitch=float(tts_raw.get("pitch", 1.0)),
        ),
    )


@lru_cache(maxsize=1)
def _load_all() -> dict[str, Persona]:
    loaded: dict[str, Persona] = {}
    for path in sorted(_PERSONAS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                logger.warning("Persona file %s is not an object; skipping", path.name)
                continue
            persona = _parse_persona(data)
            loaded[persona.persona_id] = persona
        except Exception:  # noqa: BLE001 — bad file must not break ask
            logger.exception("Failed to load persona from %s", path)
    if not loaded:
        # Hardcoded emergency fallback so ask never crashes on empty package data.
        loaded[DEFAULT_PERSONA_ID] = Persona(
            persona_id=DEFAULT_PERSONA_ID,
            display_name="Easygoing Friend",
            traits=PersonaTraits(),
            style_notes="Warm, relaxed friend on the couch.",
            avoid=("forced jokes",),
            canned_social_reply="Right here with you — ask anytime about what's on screen.",
            tts=PersonaTts(),
        )
    return loaded


def list_personas() -> list[Persona]:
    all_p = _load_all()
    ordered: list[Persona] = []
    for pid in _KNOWN_FALLBACK_ORDER:
        if pid in all_p:
            ordered.append(all_p[pid])
    for pid, persona in sorted(all_p.items()):
        if pid not in _KNOWN_FALLBACK_ORDER:
            ordered.append(persona)
    return ordered


def resolve_persona_id(
    persona_id: str | None,
    *,
    default_id: str | None = None,
) -> str:
    all_p = _load_all()
    preferred = (persona_id or "").strip() or (default_id or DEFAULT_PERSONA_ID).strip()
    if preferred in all_p:
        return preferred
    if default_id and default_id in all_p:
        return default_id
    if DEFAULT_PERSONA_ID in all_p:
        return DEFAULT_PERSONA_ID
    return next(iter(all_p))


def get_persona(
    persona_id: str | None = None,
    *,
    default_id: str | None = None,
) -> Persona:
    resolved = resolve_persona_id(persona_id, default_id=default_id)
    return _load_all()[resolved]


def normalize_companion_gender(
    value: str | None,
) -> CompanionGender | None:
    if value is None:
        return None
    g = str(value).strip().lower()
    if g in VALID_GENDERS:
        return g  # type: ignore[return-value]
    return None
