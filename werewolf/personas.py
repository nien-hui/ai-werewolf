"""AI player personalities, given out at setup (see rules.assign_players).

Each persona is one file, prompts/text/personas/<key>.md: a 「# 名稱」 line, then the style.
`style` is injected into that AI's system prompt (an empty style, as in plain.md, adds no
personality section at all); `name` is only shown in the
post-game review (players are always displayed as 「N號」 during the game).
"""
from __future__ import annotations

from pathlib import Path

PERSONA_DIR = Path(__file__).parent / "prompts" / "text" / "personas"


def _load(path: Path) -> dict[str, str]:
    title, _, style = path.read_text(encoding="utf-8").strip().partition("\n")
    if not title.startswith("# "):
        raise ValueError(f"{path}: the first line must be 「# 個性名稱」")
    return {"name": title[2:].strip(), "style": style.strip()}


PERSONAS: dict[str, dict[str, str]] = {p.stem: _load(p) for p in sorted(PERSONA_DIR.glob("*.md"))}

PERSONA_KEYS: list[str] = list(PERSONAS)


def persona_name(key: str) -> str:
    return PERSONAS.get(key, {}).get("name", "")


def persona_style(key: str) -> str:
    return PERSONAS.get(key, {}).get("style", "")
