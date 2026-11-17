import sys


def main():
    if sys.version_info < (3, 11):
        print("Pypeople requires Python 3.11 or newer.")
        return 1
    try:
        from application.app import run
    except ImportError as exc:
        print(f"Pypeople could not import {exc.name or 'a required package'}.")
        print("In the project folder, run: python -m pip install --user -r requirements.txt")
        return 1
    return run()

if __name__ == "__main__":
    raise SystemExit(main())
