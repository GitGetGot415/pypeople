import sys

def main():
    if sys.version_info < (3, 11):
        print("Pypeople requires Python 3.11 or newer.")
        return 1
    try:
        from ui.app import run
    except ImportError:
        print("Pypeople needs pygame, google-genai, pydantic 2, and pyperclip.")
        print("In the project folder, run: python -m pip install --user -r requirements.txt")
        return 1
    return run()

if __name__ == "__main__":
    raise SystemExit(main())
