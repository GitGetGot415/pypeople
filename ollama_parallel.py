"""Read and edit local Ollama parallelism, and inspect loaded runner capacity."""

from dataclasses import dataclass, field
import os
import platform
import re
import subprocess
import shlex
import shutil

import httpx
from constants import MAX_CONFIGURABLE_LLM_REQUESTS


def set_parallel_limit(limit):
    """Set the OS launch preference; desktop Ollama applies it on its next launch."""
    if type(limit) is not int or not 1 <= limit <= MAX_CONFIGURABLE_LLM_REQUESTS:
        raise ValueError(f"Choose a request limit from 1 to {MAX_CONFIGURABLE_LLM_REQUESTS}.")
    value = str(limit)
    system = platform.system()
    try:
        if system == "Darwin":
            if _command(["launchctl", "setenv", "OLLAMA_NUM_PARALLEL", value]) is None:
                raise ValueError("Could not update the macOS Ollama launch setting.")
        elif system == "Windows":
            import ctypes
            import winreg
            with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "OLLAMA_NUM_PARALLEL", 0, winreg.REG_SZ, value)
            # Notify desktop apps about the changed user environment.
            ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, ctypes.c_wchar_p("Environment"), 2, 1000, None)
        elif system == "Linux":
            helper = shutil.which("pkexec")
            if not helper:
                raise ValueError("Editing the Ollama system service needs a graphical administrator prompt (pkexec).")
            directory = "/etc/systemd/system/ollama.service.d"
            # The only inserted value is a validated integer. Authorization is
            # requested by the desktop, without opening a terminal.
            script = (f"set -e\nmkdir -p {directory}\n"
                      f"printf '%s\\n' '[Service]' 'Environment=\"OLLAMA_NUM_PARALLEL={value}\"' > {directory}/pypeople-parallel.conf.tmp\n"
                      f"mv {directory}/pypeople-parallel.conf.tmp {directory}/pypeople-parallel.conf\n"
                      "systemctl daemon-reload\nsystemctl restart ollama.service\n")
            result = subprocess.run([helper, "/bin/sh", "-c", script], capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise ValueError("Ollama service update failed or administrator authorization was cancelled.")
        else:
            raise ValueError("Editing Ollama's launch setting is not supported on this operating system.")
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Could not update Ollama's launch setting. Check your account permissions and try again.") from exc
    os.environ["OLLAMA_NUM_PARALLEL"] = value
    return ("Ollama service restarted with the new limit." if system == "Linux" else
            "Limit saved. Quit and reopen Ollama to apply it.")


@dataclass
class ParallelStatus:
    launch_value: str = "Unavailable"
    launch_source: str = ""
    slots: dict[str, int] = field(default_factory=dict)
    running_note: str = "Unconfirmed; local process details unavailable."

    def lines(self, model, game_limit):
        source = f" ({self.launch_source})" if self.launch_source else ""
        launch = f"Ollama launch parallelism: {self.launch_value}{source}"
        slots = self.slots.get(model, self.slots.get(f"{model}:latest"))
        if slots is None:
            return launch, f"Loaded model slots: {self.running_note}", f"Game limit: {game_limit}. Refresh to check again."
        usable = min(game_limit, slots)
        note = f"This game can use {usable} simultaneous request{'s' if usable != 1 else ''} (game limit: {game_limit})."
        if self.launch_value.isdigit() and int(self.launch_value) != slots:
            note = "Launch setting differs; Ollama may need a restart."
        return launch, f"Loaded model slots: {slots} confirmed for the selected model.", note


def _command(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=2,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if platform.system() == "Windows" else 0)
    except (OSError, UnicodeError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _setting(value):
    value = str(value).strip()
    if not value:
        return "Not set"
    if value.isascii() and value.isdigit() and int(value) > 0:
        return str(int(value))
    return "Invalid value"


def _launch_setting():
    system = platform.system()
    if system == "Darwin":
        value = _command(["launchctl", "getenv", "OLLAMA_NUM_PARALLEL"])
        return (_setting(value), "macOS launch setting") if value is not None else ("Unavailable", "macOS launch setting")
    if system == "Windows":
        import winreg
        for root, path, source in (
            (winreg.HKEY_CURRENT_USER, r"Environment", "Windows user setting"),
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", "Windows system setting"),
        ):
            try:
                with winreg.OpenKey(root, path) as key:
                    value, _ = winreg.QueryValueEx(key, "OLLAMA_NUM_PARALLEL")
                return _setting(value), source
            except FileNotFoundError:
                continue
            except OSError:
                return "Unavailable", source
        return "Not set", "Windows launch setting"
    if system == "Linux":
        value = _command(["systemctl", "show", "ollama.service", "--property=Environment", "--value"])
        if value is not None:
            try:
                for entry in shlex.split(value):
                    if entry.startswith("OLLAMA_NUM_PARALLEL="):
                        return _setting(entry.partition("=")[2]), "Ollama systemd service"
            except ValueError:
                pass
        # Shell-launched servers and EnvironmentFile settings may differ.
    if "OLLAMA_NUM_PARALLEL" in os.environ:
        return _setting(os.environ["OLLAMA_NUM_PARALLEL"]), "game environment; server unconfirmed"
    return "Unavailable", "server launch setting not exposed"


def _runner_slots():
    if platform.system() == "Windows":
        commands = _command([
            "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name = 'ollama.exe' OR Name = 'llama-server.exe' OR Name = 'ollama_llama_server.exe'\" "
            "| Select-Object -ExpandProperty CommandLine",
        ])
    else:
        commands = _command(["ps", "-axww", "-o", "command="])
    if commands is None:
        return {}
    slots = {}
    for line in commands.splitlines():
        # Match Ollama's runner binaries, including older `ollama runner` builds.
        if not re.match(r'^\s*(?:"[^"]*(?:ollama|llama-server|ollama_llama_server)(?:\.exe)?"|\S*(?:ollama|llama-server|ollama_llama_server)(?:\.exe)?)\s', line):
            continue
        digest = re.search(r'\bsha256-([a-fA-F0-9]{64})\b', line)
        parallel = re.search(r'(?:^|\s)(?:--parallel|-np)(?:=|\s+)([1-9][0-9]*)(?=\s|$)', line)
        if digest and parallel:
            slots.setdefault(digest.group(1).lower(), set()).add(int(parallel.group(1)))
    # Multiple local servers may load the same model with different capacities.
    return {digest: next(iter(values)) for digest, values in slots.items() if len(values) == 1}


def _model_blob(name):
    # /api/ps identifies the manifest, while runner --model paths identify its
    # weight blob. /api/show resolves that difference without loading the model.
    try:
        response = httpx.post("http://localhost:11434/api/show", json={"model": name}, timeout=2.0)
        response.raise_for_status()
        modelfile = response.json()["modelfile"]
        for line in modelfile.splitlines():
            if line.startswith("FROM "):
                digest = re.search(r'\bsha256-([a-fA-F0-9]{64})\b', line)
                return digest.group(1).lower() if digest else None
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return None


def inspect_parallelism():
    """Only report active capacity when a loaded model matches a local runner."""
    status = ParallelStatus()
    status.launch_value, status.launch_source = _launch_setting()
    try:
        response = httpx.get("http://localhost:11434/api/ps", timeout=2.0)
        response.raise_for_status()
        models = response.json()["models"]
        if not isinstance(models, list):
            raise ValueError
        if not models:
            status.running_note = "No model loaded; run a conversation, then Refresh."
            return status
        runners = _runner_slots()
        for model in models:
            name, digest = model["name"], model["digest"]
            if not isinstance(name, str) or not isinstance(digest, str):
                raise ValueError
            count = runners.get(digest.removeprefix("sha256:").lower())
            if count is None and runners:
                count = runners.get(_model_blob(name))
            if count is not None:
                status.slots[name] = count
        status.running_note = "Unconfirmed for the selected model; try Refresh after use."
    except httpx.HTTPError:
        status.running_note = "Unavailable; cannot connect to Ollama."
    except (ValueError, KeyError, TypeError):
        status.slots.clear()
        status.running_note = "Unavailable; Ollama returned invalid model details."
    return status
