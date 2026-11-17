"""Persistent application preferences; API keys are never saved here."""

from paths import APP_SETTINGS_PATH
import json
from persistence.characters import NPC_TRAIT_MODES
from constants import MAX_CONFIGURABLE_LLM_REQUESTS, MAX_LLM_REQUESTS
from ai.online import PROVIDERS, load_profiles, profile_settings


SETTINGS_PATH = APP_SETTINGS_PATH
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
    online_provider = settings.get("online_provider")
    return {
        "npc_trait_mode": mode if mode in NPC_TRAIT_MODES else DEFAULT_NPC_TRAIT_MODE,
        "voice_rewrites": settings.get("voice_rewrites") if isinstance(settings.get("voice_rewrites"), bool) else True,
        "dialogue_validation": settings.get("dialogue_validation") if isinstance(settings.get("dialogue_validation"), bool) else True,
        "max_llm_requests": limit if type(limit) is int and 1 <= limit <= MAX_CONFIGURABLE_LLM_REQUESTS else MAX_LLM_REQUESTS,
        "provider": "ollama" if settings.get("provider") == "ollama" else "online",
        "online_provider": online_provider if isinstance(online_provider, str) and online_provider in PROVIDERS else "gemini",
        "online_profiles": load_profiles(settings.get("online_profiles")),
    }


def save_settings(npc_trait_mode, voice_rewrites, dialogue_validation, *, max_llm_requests=MAX_LLM_REQUESTS,
                  provider="online", online_provider="gemini", online_profiles=None):
    if npc_trait_mode not in NPC_TRAIT_MODES:
        raise ValueError(f"Trait mode must be one of: {', '.join(NPC_TRAIT_MODES)}.")
    if type(max_llm_requests) is not int or not 1 <= max_llm_requests <= MAX_CONFIGURABLE_LLM_REQUESTS:
        raise ValueError(f"Request limit must be between 1 and {MAX_CONFIGURABLE_LLM_REQUESTS}.")
    if provider not in ("online", "ollama") or online_provider not in PROVIDERS:
        raise ValueError("Choose a dialogue provider in Settings.")
    settings = {"npc_trait_mode": npc_trait_mode, "voice_rewrites": voice_rewrites,
                "dialogue_validation": dialogue_validation, "max_llm_requests": max_llm_requests,
                "provider": provider, "online_provider": online_provider,
                "online_profiles": profile_settings(online_profiles if online_profiles is not None else load_profiles(None))}
    temporary = SETTINGS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    temporary.replace(SETTINGS_PATH)
