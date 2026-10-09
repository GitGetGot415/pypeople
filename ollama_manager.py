import httpx
import webbrowser
import subprocess
import platform
import shutil

def check_ollama():
    try:
        print("Checking if your Ollama is running...")
        response = httpx.get(
            "http://localhost:11434/api/tags",
            timeout=2.0
        )
        return response.status_code == 200

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
