"""Pure game-rule functions (no LLM, no LangGraph).

Seats are 1..N ints. Players are `state.Player` dicts. Nothing here mutates
its arguments.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from typing import Any, Iterable, Optional

from . import config
from .personas import PERSONA_KEYS
from .state import GOD_ROLES, NightState, Player

WITCH_ACTIONS = ("save", "poison", "none")
# Said in place of an empty speech (the human's 「過」 button). AI fallbacks use the same
# phrases, so a pass looks like any other short speech instead of a human-only marker.
PASS_SPEECHES = {
    "discussion": ("我先聽聽大家的。", "目前資訊還不多，我再觀察一下。", "我沒什麼特別的想法，先過。"),
    "last_words": ("我沒什麼好說的，大家加油。", "好人加油，認真找狼。"),
    "wolf_chat": ("我都可以，照你的想法刀吧。", "今晚隨便挑一個吧，我跟。"),
}
_LINE_BREAK_RE = re.compile(r"\s*[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]\s*")


# ------------------------------------------------------------------ setup
def assign_players(
    rng: random.Random, human_role: Optional[str] = None, all_ai: bool = False, persona: Optional[str] = None,
    persona_pool: Optional[list[str]] = None,
) -> list[Player]:
    """Shuffle config.ROLE_SETUP onto seats 1..N and pick the human seat.

    human_role (test hook) forces the human onto a seat holding that role.
    all_ai: no human seat, every seat is an AI (self-play; human_role is ignored).
    persona: give every AI this persona key instead of a mix.
    persona_pool: mix only these persona keys (default: all of PERSONA_KEYS).
    Each AI gets a persona key, all different while there are enough of them, then each about
    as often as the others; the human's persona is "".
    """
    board = list(config.ROLE_SETUP)
    if human_role is not None and human_role not in board:
        raise ValueError(f"unknown human_role: {human_role!r}")
    if persona is not None and persona not in PERSONA_KEYS:
        raise ValueError(f"unknown persona: {persona!r}")
    pool = list(persona_pool or PERSONA_KEYS)
    if not set(pool) <= set(PERSONA_KEYS):
        raise ValueError(f"unknown persona in pool: {pool!r}")
    roles = board
    rng.shuffle(roles)
    n = len(roles)
    human_idx: Optional[int] = None
    if not all_ai:
        candidates = [i for i, r in enumerate(roles) if human_role is None or r == human_role]
        human_idx = rng.choice(candidates)
    num_ai = n if all_ai else n - 1
    picks: list[str] = [persona] * num_ai if persona is not None else []
    while len(picks) < num_ai:  # one round of different personas after another
        picks += rng.sample(pool, min(len(pool), num_ai - len(picks)))
    personas = iter(picks)
    return [
        {
            "seat": i + 1,
            "role": role,
            "alive": True,
            "is_human": i == human_idx,
            "persona": "" if i == human_idx else next(personas),
        }
        for i, role in enumerate(roles)
    ]


def wolf_teammates(players: list[Player], seat: int) -> list[int]:
    """Other wolves' seats (dead included) for a wolf; [] for non-wolves."""
    me = next(p for p in players if p["seat"] == seat)
    if me["role"] != "werewolf":
        return []
    return [p["seat"] for p in players if p["role"] == "werewolf" and p["seat"] != seat]


# ------------------------------------------------------------------ winner
def check_winner(players: list[Player]) -> tuple[Optional[str], str]:
    """屠邊 rule. Returns (winner, reason_zh); (None, "") while the game goes on.

    Wolves-dead is checked first, so a simultaneous wipe-out favours good.
    """
    alive = [p for p in players if p["alive"]]
    if not any(p["role"] == "werewolf" for p in alive):
        return "good", "所有狼人都已出局"
    if not any(p["role"] in GOD_ROLES for p in alive):
        return "wolf", "神職（預言家、女巫）已全部出局，狼人屠邊成功"
    if not any(p["role"] == "villager" for p in alive):
        return "wolf", "村民已全部出局，狼人屠邊成功"
    return None, ""


def draw_reason() -> str:
    return f"已達 {config.MAX_DAYS} 天上限，平局"


def mark_dead(players: list[Player], seats: Iterable[int]) -> list[Player]:
    """Copy of players with `seats` set to alive=False."""
    dead = set(seats)
    return [{**p, "alive": False} if p["seat"] in dead else dict(p) for p in players]


# ------------------------------------------------------------------- night
def resolve_night(night: NightState) -> list[int]:
    """Seats that die at dawn: wolf target (unless saved) plus poison target."""
    dead: set[int] = set()
    if night.get("wolf_target") is not None and not night.get("saved"):
        dead.add(night["wolf_target"])
    if night.get("poisoned") is not None:
        dead.add(night["poisoned"])
    return sorted(dead)


def wolf_consensus(picks: dict[int, int]) -> int:
    """The AI wolves' kill from {wolf seat: target}: the most picked target; on a tie, the pick
    of the smallest-seat wolf among those who picked a tied target. picks must be non-empty."""
    counts = Counter(picks.values())
    best = max(counts.values())
    return next(picks[w] for w in sorted(picks) if counts[picks[w]] == best)


def wolf_kill_options(players: list[Player]) -> list[int]:
    """Legal kill targets: alive non-wolves (no 自刀, no 空刀)."""
    return [p["seat"] for p in players if p["alive"] and p["role"] != "werewolf"]


def alive_others(players: list[Player], seat: int) -> list[int]:
    """Alive seats except `seat` (seer check / exile vote / poison options)."""
    return [p["seat"] for p in players if p["alive"] and p["seat"] != seat]


# ------------------------------------------------------------------- day
def discussion_order(
    alive: Iterable[int],
    last_night_deaths: Iterable[int],
    day: int,
    num_players: int = config.NUM_PLAYERS,
) -> list[int]:
    """Clockwise speaking order for the day.

    Starts at the next alive seat after the smallest seat that died last
    night; with no deaths, at seat ((day-1) % N)+1 (or the next alive seat).
    """
    alive_set = set(alive)
    if not alive_set:
        return []
    deaths = list(last_night_deaths)
    first = (min(deaths) % num_players) + 1 if deaths else ((day - 1) % num_players) + 1
    order = [((first - 1 + k) % num_players) + 1 for k in range(num_players)]
    return [s for s in order if s in alive_set]


def _tally(targets: Iterable[Optional[int]]) -> tuple[dict[str, int], list[int]]:
    counts = Counter(t for t in targets if t is not None)
    tally = {str(s): counts[s] for s in sorted(counts)}
    if not counts:
        return tally, []
    best = max(counts.values())
    return tally, sorted(s for s, c in counts.items() if c == best)


def tally_votes(votes: list[dict]) -> tuple[dict[str, int], list[int], Optional[int]]:
    """Exile vote. votes: [{"voter", "target"}], target None = abstain.

    Returns (tally with str keys, top seats, exiled seat or None on tie/no votes).
    """
    tally, top = _tally(v.get("target") for v in votes)
    return tally, top, top[0] if len(top) == 1 else None


# ------------------------------------------------- resume-value coercion
def as_seat(value: Any) -> Optional[int]:
    """int-like value -> int, else None (bools rejected)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def one_line(text: str) -> str:
    """Trimmed text with every line break (and the spaces around it) turned into one space.

    Everything players say is stored as one line: AI output always is, so a line break
    would out the human, and in the AIs' log it could pass for a host or private line.
    """
    return _LINE_BREAK_RE.sub(" ", text).strip()


def coerce_speech(value: Any) -> str:
    """{"text": str} (or a bare str) -> one trimmed line, max MAX_SPEECH_CHARS; "" if empty."""
    text = value.get("text") if isinstance(value, dict) else value
    return one_line(text)[: config.MAX_SPEECH_CHARS].strip() if isinstance(text, str) else ""


def pass_speech(kind: str, rng: random.Random, said: Iterable[str] = ()) -> str:
    """What an empty speech becomes: a PASS_SPEECHES phrase, one this player hasn't said yet if possible."""
    phrases = PASS_SPEECHES.get(kind, PASS_SPEECHES["discussion"])
    used = set(said)
    return rng.choice([p for p in phrases if p not in used] or list(phrases))


def coerce_target(
    value: Any, options: list[int], allow_skip: bool, rng: random.Random
) -> Optional[int]:
    """{"target": n} -> n if legal; else None if skipping is allowed, else a random option."""
    raw = value.get("target") if isinstance(value, dict) else value
    seat = as_seat(raw)
    if seat in options:
        return seat
    if allow_skip or not options:
        return None
    return rng.choice(options)


def coerce_witch(
    value: Any, killed: Optional[int], can_save: bool, poison_options: list[int]
) -> tuple[str, Optional[int]]:
    """{"action", "target"} -> (action, target); anything invalid -> ("none", None)."""
    if not isinstance(value, dict):
        return "none", None
    action = value.get("action")
    if action == "save" and can_save and killed is not None:
        return "save", killed
    if action == "poison":
        target = as_seat(value.get("target"))
        if target in poison_options:
            return "poison", target
    return "none", None
