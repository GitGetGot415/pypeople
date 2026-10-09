"""Native JSON file picker shared by character creation and game setup."""

import subprocess
import sys


def character_file_dialog(save):
    if sys.platform == "darwin":
        # Creating even a hidden Tk root can abort Python in some macOS
        # Tcl/Tk builds. Use macOS's native picker instead.
        if save:
            script = 'POSIX path of (choose file name with prompt "Export character" default name "character.json")'
        else:
            script = 'POSIX path of (choose file with prompt "Import character" of type {"public.json"})'
        result = subprocess.run(
            ("/usr/bin/osascript", "-e", script),
            capture_output=True,
            text=True,
            check=False,
        )
        # AppleScript returns a nonzero status when the user cancels.
        return result.stdout.strip() if result.returncode == 0 else ""

    import tkinter as tk
    from tkinter import filedialog

    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise OSError("File dialog is unavailable.") from error
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        try:
            if save:
                return filedialog.asksaveasfilename(
                    parent=root,
                    title="Export character",
                    defaultextension=".json",
                    filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
                    initialfile="character.json",
                )
            return filedialog.askopenfilename(
                parent=root,
                title="Import character",
                filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            )
        except tk.TclError as error:
            raise OSError("File dialog is unavailable.") from error
    finally:
        root.destroy()

