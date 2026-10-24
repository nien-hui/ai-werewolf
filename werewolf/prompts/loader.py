"""Fixed prompt wording, kept as Markdown under text/ — edit those files to change what the AIs read.

text/rules.md may use {num_players} and {max_days}; the other files are used as-is.
Personas (text/personas/*.md) are loaded by werewolf/personas.py.
"""
from __future__ import annotations

from pathlib import Path

from .. import config

TEXT_DIR = Path(__file__).with_name("text")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _read_dir(name: str) -> dict[str, str]:
    """{file stem: text} for every .md in text/<name>/."""
    return {p.stem: _read(p) for p in sorted((TEXT_DIR / name).glob("*.md"))}


_RULES = _read(TEXT_DIR / "rules.md")
OUTPUT_RULES = _read(TEXT_DIR / "output_rules.md")
ROLE_GUIDES = _read_dir("roles")


def rules_text() -> str:
    return _RULES.format(num_players=config.NUM_PLAYERS, max_days=config.MAX_DAYS)


def role_guide(role: str) -> str:
    return ROLE_GUIDES.get(role, "")
