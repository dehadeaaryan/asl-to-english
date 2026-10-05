"""Compatibility entry point for the local ASL recognition app.

The original IMAGE-mode landmark demo has been replaced by the tracked,
trained-recognizer workflow in app.py. Keep this filename usable for existing
local muscle memory.
"""

from app import main


if __name__ == "__main__":
    main()
