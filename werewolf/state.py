"""Shared game-state types, constants and event helpers.

Everything stored in the LangGraph state must stay JSON/msgpack friendly:
plain dicts, lists, str/int/bool/None. Never use int dict keys.
"""
from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

# ---------------------------------------------------------------- constants
ROLE_ZH = {"werewolf": "狼人", "seer": "預言家", "witch": "女巫", "villager": "村民"}
ROLE_TEAM = {"werewolf": "wolf", "seer": "good", "witch": "good", "villager": "good"}
TEAM_ZH = {"wolf": "狼人陣營", "good": "好人陣營", "draw": "平局"}
GOD_ROLES = ("seer", "witch")

# Event visibility values:
#   "all"      – everyone (public table talk / announcements)
#   "wolves"   – players whose role is werewolf
#   "seat:<n>" – only seat n (private info, e.g. seer result)
#   "god"      – nobody during the game; shown only in the post-game review.
#                An AI may additionally see its *own* god events (its thoughts).
VIS_ALL = "all"
VIS_WOLVES = "wolves"
VIS_GOD = "god"


def vis_seat(seat: int) -> str:
    return f"seat:{seat}"


# -------------------------------------------------------------------- types
class Player(TypedDict):
    seat: int  # 1..N
    role: str  # key of ROLE_ZH
    alive: bool
    is_human: bool
    persona: str  # persona key for AI players, "" for the human


class Event(TypedDict):
    id: int  # unique, increasing (index in state["events"])
    day: int  # night N and day N share the number N; 0 = before first night
    phase: str  # "setup" | "night" | "day" | "end"
    kind: str  # system | death | speech | last_words | thought | vote_result | wolf_chat | ...
    seat: Optional[int]  # acting / speaking seat, None for system events
    text: str  # human-readable zh-TW text
    visibility: str  # see VIS_* above
    data: dict  # kind-specific structured payload


class NightState(TypedDict):
    wolf_target: Optional[int]
    saved: bool
    poisoned: Optional[int]


class GameState(TypedDict, total=False):
    game_id: str
    seed: Optional[int]
    human_role: Optional[str]  # optional test hook: force the human's role
    all_ai: bool  # start input: no human seat, every seat is an AI (self-play)
    persona: Optional[str]  # start input: every AI gets this persona key (self-play), else a mix
    persona_pool: Optional[list[str]]  # start input: the mix draws only these keys (self-play)
    players: list[Player]
    human_seat: Optional[int]  # None in an all-AI game
    day: int
    phase: str  # "setup" | "night" | "day" | "end"
    stage: str  # speech-loop routing: wolf_chat | night_last_words | discussion | exile_last_words
    speech_queue: list[int]
    events: Annotated[list[Event], operator.add]  # nodes return ONLY new events
    # wolf_pick / cast_vote run once per voter, all at the same time (one Send each): "voter" is only
    # in the Send's input; each run adds one {"seat", "value"} (raw answer / AI result) to "ballots",
    # and the node that counts them (wolf_kill / count_votes) clears it
    voter: int
    ballots: Annotated[list[dict], operator.add]
    night: NightState
    antidote: bool  # witch still has the antidote
    poison: bool  # witch still has the poison
    seer_checks: list[dict]  # [{"day": int, "target": int, "is_wolf": bool}]
    last_night_deaths: list[int]
    winner: Optional[str]  # "wolf" | "good" | "draw"
    win_reason: str
    token_usage: Optional[dict]  # set by game_over: the backend's usage for this game (None without one)


# ------------------------------------------------------------------ helpers
def get_player(state: GameState, seat: int) -> Player:
    for p in state["players"]:
        if p["seat"] == seat:
            return p
    raise KeyError(f"no seat {seat}")


def all_seats(state: GameState) -> list[int]:
    return [p["seat"] for p in state["players"]]


def alive_seats(state: GameState) -> list[int]:
    return [p["seat"] for p in state["players"] if p["alive"]]


def seats_with_role(state: GameState, role: str, alive_only: bool = True) -> list[int]:
    return [
        p["seat"]
        for p in state["players"]
        if p["role"] == role and (p["alive"] or not alive_only)
    ]


def is_wolf(state: GameState, seat: int) -> bool:
    return get_player(state, seat)["role"] == "werewolf"


class EventBuilder:
    """Collects new events for one node run, assigning sequential ids.

    Usage inside a node:
        eb = EventBuilder(state)               # day/phase default from state
        eb.add("system", "天黑請閉眼")
        return {"events": eb.events, ...}

    Ids continue from len(state["events"]); this is safe because graph nodes
    run strictly one at a time (no parallel branches write events).
    """

    def __init__(self, state: GameState, *, day: Optional[int] = None, phase: Optional[str] = None):
        self._next_id = len(state.get("events") or [])
        self.day = state.get("day", 0) if day is None else day
        self.phase = state.get("phase", "setup") if phase is None else phase
        self.events: list[Event] = []

    def add(
        self,
        kind: str,
        text: str,
        *,
        visibility: str = VIS_ALL,
        seat: Optional[int] = None,
        data: Optional[dict] = None,
        day: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> Event:
        ev: Event = {
            "id": self._next_id,
            "day": self.day if day is None else day,
            "phase": self.phase if phase is None else phase,
            "kind": kind,
            "seat": seat,
            "text": text,
            "visibility": visibility,
            "data": data or {},
        }
        self._next_id += 1
        self.events.append(ev)
        return ev


def visible_to(
    event: Event,
    seat: int,
    role: str,
    *,
    include_god: bool = False,
    include_own_god: bool = False,
) -> bool:
    """Whether `seat` (holding `role`) may see `event`.

    include_god:      reveal every god event (post-game review only).
    include_own_god:  reveal god events whose event["seat"] == seat
                      (an AI may re-read its own earlier thoughts).
    """
    v = event["visibility"]
    if v == VIS_ALL:
        return True
    if v == VIS_GOD:
        return include_god or (include_own_god and event.get("seat") == seat)
    if v == VIS_WOLVES:
        return include_god or role == "werewolf"
    if v.startswith("seat:"):
        return include_god or int(v[5:]) == seat
    return include_god


def events_visible_to(state: GameState, seat: int, *, include_own_god: bool = False) -> list[Event]:
    role = get_player(state, seat)["role"]
    return [
        e
        for e in state.get("events") or []
        if visible_to(e, seat, role, include_own_god=include_own_god)
    ]
