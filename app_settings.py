"""Persistent preferences that are independent of model-provider settings."""

import json
from pathlib import Path

from character_data import NPC_TRAIT_MODES


SETTINGS_PATH = Path(__file__).resolve().parent / "app_settings.json"
DEFAULT_NPC_TRAIT_MODE = "normal"


def load_npc_trait_mode():
    try:
        settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_NPC_TRAIT_MODE
    mode = settings.get("npc_trait_mode") if isinstance(settings, dict) else None
    return mode if mode in NPC_TRAIT_MODES else DEFAULT_NPC_TRAIT_MODE


def save_npc_trait_mode(mode):
    if mode not in NPC_TRAIT_MODES:
        raise ValueError(f"Trait mode must be one of: {', '.join(NPC_TRAIT_MODES)}.")
    settings = {"npc_trait_mode": mode}
    temporary = SETTINGS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    temporary.replace(SETTINGS_PATH)
