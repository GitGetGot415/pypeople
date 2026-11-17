"""Project resources and writable data, independent of source package locations."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ASSET_DIRECTORY = PROJECT_ROOT / "assets"
CHARACTER_DIRECTORY = PROJECT_ROOT / "characters"
SAVE_DIRECTORY = PROJECT_ROOT / "saves"
LOG_DIRECTORY = PROJECT_ROOT / "logs"
MODEL_LOG_DIRECTORY = LOG_DIRECTORY / "model_requests"
APP_SETTINGS_PATH = PROJECT_ROOT / "app_settings.json"
OLLAMA_SETTINGS_PATH = PROJECT_ROOT / "ollama_settings.json"
