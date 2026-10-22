"""Native JSON file picker shared by character creation and game setup."""

import subprocess
import sys
import json


def character_file_dialog(save, *, title=None, multiple=False):
    title = title or ("Export character" if save else "Import character")
    if sys.platform == "darwin":
        # Creating even a hidden Tk root can abort Python in some macOS
        # Tcl/Tk builds. Use macOS's native picker instead.
        if save:
            script = f'POSIX path of (choose file name with prompt {json.dumps(title)} default name "character.json")'
        elif multiple:
            script = (
                f'set selectedFiles to choose file with prompt {json.dumps(title)} '
                'of type {"public.json"} with multiple selections allowed\n'
                'set selectedPaths to ""\n'
                'repeat with selectedFile in selectedFiles\n'
                'set selectedPaths to selectedPaths & (POSIX path of selectedFile) & linefeed\n'
                'end repeat\nreturn selectedPaths'
            )
        else:
            script = f'POSIX path of (choose file with prompt {json.dumps(title)} of type {{"public.json"}})'
        result = subprocess.run(
            ("/usr/bin/osascript", "-e", script),
            capture_output=True,
            text=True,
            check=False,
        )
        # AppleScript returns a nonzero status when the user cancels.
        if multiple and not save:
            return tuple(result.stdout.strip().splitlines()) if result.returncode == 0 else ()
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
                    title=title,
                    defaultextension=".json",
                    filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
                    initialfile="character.json",
                )
            picker = filedialog.askopenfilenames if multiple else filedialog.askopenfilename
            return picker(
                parent=root,
                title=title,
                filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            )
        except tk.TclError as error:
            raise OSError("File dialog is unavailable.") from error
    finally:
        root.destroy()

