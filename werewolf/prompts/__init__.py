"""Prompts for the AI players.

text/*.md     fixed wording: rules, role guides, output rules, personas
loader.py     reads text/*.md
knowledge.py  what one seat may know; renders its game log and situation (INFORMATION SAFETY)
builders.py   assembles the messages: system prompt, per-step tasks
"""
from __future__ import annotations

from .builders import (
    all_prompt_text,
    build_messages,
    system_prompt,
    task_exile_vote,
    task_seer_check,
    task_speak,
    task_witch,
    task_wolf_target,
    user_prompt,
)
from .knowledge import private_knowledge, public_deaths, render_log, saved_kill, situation_text, speaking_order
from .loader import OUTPUT_RULES, ROLE_GUIDES, role_guide, rules_text

__all__ = [
    "rules_text",
    "role_guide",
    "system_prompt",
    "render_log",
    "situation_text",
    "private_knowledge",
    "public_deaths",
    "speaking_order",
    "task_speak",
    "task_wolf_target",
    "task_witch",
    "task_seer_check",
    "task_exile_vote",
    "saved_kill",
    "user_prompt",
    "build_messages",
    "all_prompt_text",
    "OUTPUT_RULES",
    "ROLE_GUIDES",
]
