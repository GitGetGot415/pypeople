import httpx
import webbrowser
import subprocess
import platform
import shutil
import json
from pathlib import Path

DEFAULT_OLLAMA_MODEL = "llama3"
SETTINGS_PATH = Path(__file__).resolve().parent / "ollama_settings.json"

_ollama_ready = False


def load_model():
    """Keep the default when preferences are missing or invalid."""
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        model = data.get("model") if isinstance(data, dict) else None
        if isinstance(model, str) and model.strip():
            return model.strip()
    except (OSError, ValueError):
        pass
    return DEFAULT_OLLAMA_MODEL


def save_model(model):
    """Replace preferences atomically so an interrupted save keeps the old choice."""
    temporary = SETTINGS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"model": model}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(SETTINGS_PATH)


def mark_ollama_ready():
    """Remember successful discovery or validated generation for this app session."""
    global _ollama_ready
    _ollama_ready = True


def list_models():
    """Map installed model names (including tags) to their size in bytes, if known."""
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        response.raise_for_status()
        models = response.json()["models"]
        if not isinstance(models, list):
            raise ValueError
        sizes = {}
        for model in models:
            name = model["name"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError
            size = model.get("size")
            sizes[name.strip()] = size if type(size) is int and size >= 0 else None
    except httpx.HTTPError:
        raise ValueError("Cannot list models. Make sure Ollama is running, then Refresh.") from None
    except (ValueError, KeyError, TypeError):
        raise ValueError("Ollama returned an invalid model list. Try Refresh again.") from None
    mark_ollama_ready()
    return {name: sizes[name] for name in sorted(sizes, key=str.casefold)}


def check_ollama(*, force_refresh=False):
    """Reuse a successful check for this app session unless explicitly refreshed."""
    global _ollama_ready
    if _ollama_ready and not force_refresh:
        return True

    _ollama_ready = False
    try:
        print("Checking if your Ollama is running...")
        response = httpx.get(
            "http://localhost:11434/api/tags",
            timeout=2.0
        )
        _ollama_ready = response.status_code == 200
        return _ollama_ready

    except httpx.RequestError:
        return False

def open_ollama_download():
    return webbrowser.open("https://ollama.com/download")


def download_model():
    system = platform.system()

    try:
        if system == "Windows":
            subprocess.Popen(
                [
                    "cmd", "/k",
                    "ollama pull llama3 && "
                    "(echo Llama 3 download complete! You may close this window.) || "
                    "(echo Llama 3 download failed. Check the error above and try again.)"
                ],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )

        elif system == "Darwin":
            command = (
                "ollama pull llama3 && "
                "echo 'Llama 3 download complete! You may close this window.' || "
                "echo 'Llama 3 download failed. Check the error above and try again.'"
            )

            subprocess.Popen([
                "osascript", "-e",
                f'tell application "Terminal" to do script "{command}"'
            ])

        elif system == "Linux":
            command = (
                "ollama pull llama3 && "
                "echo 'Llama 3 download complete! You may close this window.' || "
                "echo 'Llama 3 download failed. Check the error above and try again.'; "
                "echo 'Press Enter to close.'; read -r"
            )

            if shutil.which("gnome-terminal"):
                subprocess.Popen([
                    "gnome-terminal", "--",
                    "bash", "-c", command
                ])
            elif shutil.which("konsole"):
                subprocess.Popen([
                    "konsole", "-e",
                    "bash", "-c", command
                ])
            elif shutil.which("xterm"):
                subprocess.Popen([
                    "xterm", "-e",
                    "bash", "-c", command
                ])
            else:
                return False

        else:
            return False

        return True

    except OSError:
        return False
