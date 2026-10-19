"""Persistent preferences that are independent of model-provider settings."""

import json
from pathlib import Path

from character_data import NPC_TRAIT_MODES
from constants import MAX_CONFIGURABLE_LLM_REQUESTS, MAX_LLM_REQUESTS


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
    limit = settings.get("max_llm_requests")
    return {
        "npc_trait_mode": mode if mode in NPC_TRAIT_MODES else DEFAULT_NPC_TRAIT_MODE,
        "voice_rewrites": settings.get("voice_rewrites") if isinstance(settings.get("voice_rewrites"), bool) else True,
        "dialogue_validation": settings.get("dialogue_validation") if isinstance(settings.get("dialogue_validation"), bool) else True,
        "max_llm_requests": limit if type(limit) is int and 1 <= limit <= MAX_CONFIGURABLE_LLM_REQUESTS else MAX_LLM_REQUESTS,
    }


def save_settings(npc_trait_mode, voice_rewrites, dialogue_validation, *, max_llm_requests=MAX_LLM_REQUESTS):
    if npc_trait_mode not in NPC_TRAIT_MODES:
        raise ValueError(f"Trait mode must be one of: {', '.join(NPC_TRAIT_MODES)}.")
    if type(max_llm_requests) is not int or not 1 <= max_llm_requests <= MAX_CONFIGURABLE_LLM_REQUESTS:
        raise ValueError(f"Request limit must be between 1 and {MAX_CONFIGURABLE_LLM_REQUESTS}.")
    settings = {"npc_trait_mode": npc_trait_mode, "voice_rewrites": voice_rewrites,
                "dialogue_validation": dialogue_validation, "max_llm_requests": max_llm_requests}
    temporary = SETTINGS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    temporary.replace(SETTINGS_PATH)
