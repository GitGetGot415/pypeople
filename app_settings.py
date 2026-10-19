"""Persistent preferences that are independent of model-provider settings."""

import json
from pathlib import Path

from character_data import NPC_TRAIT_MODES


SETTINGS_PATH = Path(__file__).resolve().parent / "app_settings.json"
DEFAULT_NPC_TRAIT_MODE = "normal"


def load_settings():
    try:
        settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        settings = {}
    if not isinstance(settings, dict):
        settings = {}
    mode = settings.get("npc_trait_mode")
    return {
        "npc_trait_mode": mode if mode in NPC_TRAIT_MODES else DEFAULT_NPC_TRAIT_MODE,
        "voice_rewrites": settings.get("voice_rewrites") if isinstance(settings.get("voice_rewrites"), bool) else True,
        "dialogue_validation": settings.get("dialogue_validation") if isinstance(settings.get("dialogue_validation"), bool) else True,
    }


def save_settings(npc_trait_mode, voice_rewrites, dialogue_validation):
    if npc_trait_mode not in NPC_TRAIT_MODES:
        raise ValueError(f"Trait mode must be one of: {', '.join(NPC_TRAIT_MODES)}.")
    settings = {"npc_trait_mode": npc_trait_mode, "voice_rewrites": voice_rewrites,
                "dialogue_validation": dialogue_validation}
    temporary = SETTINGS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    temporary.replace(SETTINGS_PATH)
