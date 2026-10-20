"""Read and edit local Ollama parallelism, and inspect loaded runner capacity."""

from dataclasses import dataclass, field
import json
import os
import platform
import re
import subprocess
import shlex
import shutil
import signal
from pathlib import Path
from time import monotonic, sleep

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


def _stop_process(pid):
    """Give a known local Ollama process time to shut down cleanly."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = monotonic() + 10
    while monotonic() < deadline:
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        sleep(0.1)
    raise ValueError("Ollama did not stop. Close it and try Apply again.")


def _restart_mac():
    listeners = _command(["/usr/sbin/lsof", "-t", "-iTCP:11434", "-sTCP:LISTEN"])
    # lsof returns a nonzero status when no server is listening.
    server_pids = set((listeners or "").split())
    processes = _command(["ps", "-axww", "-o", "pid=,command="])
    if processes is None:
        raise ValueError("Cannot inspect the local Ollama process to restart it.")
    commands = {}
    for line in processes.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            commands[parts[0]] = parts[1]
    bundle = None
    executable = shutil.which("ollama")
    for pid in server_pids:
        command = commands.get(pid, "")
        server = re.match(r"^(.*/ollama)\s+serve(?:\s|$)", command)
        if server is None:
            raise ValueError("Port 11434 is not owned by a recognizable Ollama server; it was left running.")
        executable = server.group(1)
        app = re.match(r"^(.+/Ollama\.app)/Contents/", command)
        if app:
            bundle = app.group(1)
    if bundle is None and not server_pids:
        for command in commands.values():
            app = re.match(r"^(.+/Ollama\.app)/Contents/MacOS/Ollama$", command)
            if app:
                bundle = app.group(1)
                break
        if bundle is None:
            for candidate in (Path("/Applications/Ollama.app"), Path.home() / "Applications/Ollama.app"):
                if candidate.is_dir():
                    bundle = str(candidate)
                    break
    if bundle:
        for pid, command in commands.items():
            if command == f"{bundle}/Contents/MacOS/Ollama":
                _stop_process(int(pid))
    for pid in server_pids:
        # Recheck identity in case exiting the desktop app also stopped its server.
        command = _command(["ps", "-p", pid, "-o", "command="])
        if command and command == commands[pid]:
            _stop_process(int(pid))
    if bundle:
        executable = str(Path(bundle) / "Contents/Resources/ollama")
    if executable:
        # Run just the API server, so restarting never opens or activates the
        # desktop app over the game window.
        subprocess.Popen([executable, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True, env=os.environ.copy())
    else:
        raise ValueError("Ollama is not installed or its executable could not be found.")


def _restart_windows():
    # Target the local API listener and its desktop app, rather than every model
    # process on the machine. Return its executable for a direct hidden launch.
    script = r'''
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$listeners = @(Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique)
$executable = (Get-Command ollama.exe -ErrorAction SilentlyContinue).Source
foreach ($serverId in $listeners) {
    $server = Get-CimInstance Win32_Process -Filter "ProcessId = $serverId"
    if ($server.Name -ne 'ollama.exe' -or $server.CommandLine -notmatch '\bserve\b') { throw 'Unexpected process on the Ollama port.' }
    $executable = $server.ExecutablePath
}
if (-not $executable) {
    $candidate = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { $executable = $candidate }
}
if (-not $executable) { throw 'Ollama executable was not found.' }
$directory = Split-Path $executable
Get-CimInstance Win32_Process -Filter "Name = 'ollama app.exe'" | Where-Object {
    $_.ExecutablePath -and (Split-Path $_.ExecutablePath) -eq $directory
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -ErrorAction Stop
    Wait-Process -Id $_.ProcessId -Timeout 10 -ErrorAction SilentlyContinue
}
foreach ($serverId in $listeners) {
    $server = Get-CimInstance Win32_Process -Filter "ProcessId = $serverId"
    if ($server -and $server.ExecutablePath -eq $executable) {
        Stop-Process -Id $serverId -ErrorAction Stop
        Wait-Process -Id $serverId -Timeout 10 -ErrorAction SilentlyContinue
    }
}
Write-Output $executable
'''
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True, encoding="utf-8", timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    executable = result.stdout.strip()
    if result.returncode or not executable:
        raise ValueError("Could not restart the local Ollama server. Close it and try Apply again.")
    # Avoid PowerShell child-process pipe inheritance and preserve paths with
    # spaces. Pass the updated process environment explicitly to the API server.
    subprocess.Popen([executable, "serve"], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), env=os.environ.copy())


def apply_parallel_limit(limit, model, progress=None):
    """Apply a limit, reload the selected model, and confirm its actual slots."""
    report = progress or (lambda message: None)
    report("Saving the launch setting...")
    set_parallel_limit(limit)
    report("Restarting Ollama; other local Ollama requests will be interrupted...")
    try:
        system = platform.system()
        if system == "Darwin":
            _restart_mac()
        elif system == "Windows":
            _restart_windows()
        # The Linux setter already restarts the systemd service.
        deadline = monotonic() + 20
        while True:
            try:
                response = httpx.get("http://localhost:11434/api/version", timeout=1.0)
                response.raise_for_status()
                break
            except httpx.HTTPError:
                if monotonic() >= deadline:
                    raise ValueError("Ollama did not become ready after restarting. Reopen it and try Apply again.")
                sleep(0.25)
        report("Loading the selected model with the new limit...")
        response = httpx.post("http://localhost:11434/api/chat", json={
            "model": model, "messages": [], "stream": False, "keep_alive": "15m",
            "options": {"num_ctx": 16384},
        }, timeout=120.0)
        response.raise_for_status()
        try:
            loaded = response.json()
        except ValueError as exc:
            raise ValueError("Ollama returned an invalid model-loading response. Refresh and try Apply again.") from exc
        if not isinstance(loaded, dict) or loaded.get("error"):
            raise ValueError("Ollama restarted but could not load the selected model. Try a lower request limit.")
        report("Checking the loaded model's parallel slots...")
        status = inspect_parallelism()
        slots = status.slots.get(model, status.slots.get(f"{model}:latest"))
        if slots is None:
            raise ValueError("Ollama restarted, but its loaded slots could not be confirmed. Refresh Ollama to inspect its state.")
        if slots != limit or status.launch_value != str(limit):
            raise ValueError(f"Requested {limit} slots, but Ollama loaded {slots}. Try a lower limit or check available memory.")
        return status
    except (OSError, subprocess.TimeoutExpired, httpx.HTTPError, KeyError, TypeError) as exc:
        raise ValueError("Launch setting saved, but Ollama could not restart or reload the model. Try a lower limit and Apply again.") from exc


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


def _command(arguments, *, timeout=2):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout,
                                encoding="utf-8" if platform.system() == "Windows" else None,
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
    windows = platform.system() == "Windows"
    if windows:
        commands = _command([
            "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
            "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
            "Get-CimInstance Win32_Process -Filter \"Name = 'ollama.exe' OR Name = 'llama-server.exe' OR Name = 'ollama_llama_server.exe'\" "
            "| Select-Object Name,CommandLine | ConvertTo-Json -Compress",
        ], timeout=10)
        # Windows command lines may leave executable paths containing spaces
        # unquoted. Use the OS-reported process name instead of guessing it.
        try:
            processes = json.loads(commands or "[]")
            if isinstance(processes, dict):
                processes = [processes]
            commands = "\n".join(
                process["CommandLine"] for process in processes
                if process["Name"].lower() in ("ollama.exe", "llama-server.exe", "ollama_llama_server.exe")
                and isinstance(process["CommandLine"], str)
            )
        except (ValueError, KeyError, TypeError, AttributeError):
            return {}
    else:
        commands = _command(["ps", "-axww", "-o", "command="])
    if commands is None:
        return {}
    slots = {}
    for line in commands.splitlines():
        # Match Ollama's runner binaries, including older `ollama runner` builds.
        if not windows and not re.match(r'^\s*(?:"[^"]*(?:ollama|llama-server|ollama_llama_server)(?:\.exe)?"|\S*(?:ollama|llama-server|ollama_llama_server)(?:\.exe)?)\s', line):
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
