"""Runtime configuration. Values can be overridden with environment variables
or entries in the project-root .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL_NAME = os.getenv("WEREWOLF_MODEL", "gemini-3.5-flash-lite")
# "1" -> AI players use a random offline backend (no API calls); for tests/dev.
FAKE_LLM = os.getenv("WEREWOLF_FAKE_LLM", "0") == "1"

LLM_TIMEOUT_S = float(os.getenv("WEREWOLF_LLM_TIMEOUT", "60"))
LLM_MAX_ATTEMPTS = 3

# The board ("屠邊" win condition, no sheriff): 2 wolves, seer, witch, 2 villagers.
ROLE_SETUP = ["werewolf", "werewolf", "seer", "witch", "villager", "villager"]
NUM_PLAYERS = len(ROLE_SETUP)
MAX_DAYS = 10  # safety cap; reaching it ends the game as a draw
MAX_SPEECH_CHARS = 200
# After each AI speech the human can see, the game waits until the page has typed it out
# (TYPE_S_PER_CHAR per character, at most TYPE_MAX_S: the same speed as static/app.js) and then
# SPEECH_PAUSE_S more before anything else happens.
SPEECH_PAUSE_S = float(os.getenv("WEREWOLF_SPEECH_PAUSE", "2"))
TYPE_S_PER_CHAR = 0.055
TYPE_MAX_S = 6.0

# Players must give this password to connect (the page asks for it, or takes it from /?key=…).
# Empty = no password.
PASSWORD = os.getenv("WEREWOLF_PASSWORD", "")

# Every finished game is saved here as one JSON file: the god's-eye view of the post-game review.
GAMES_DIR = Path(os.getenv("WEREWOLF_GAMES_DIR") or ROOT / "games")


def get_api_key() -> str | None:
    return os.getenv("gemini_key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
