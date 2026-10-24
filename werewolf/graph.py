"""LangGraph StateGraph for one werewolf game.

Interrupt rule: a node that calls interrupt() is re-executed from the top on
resume, so every node calls interrupt() first (at most once) and only then
does backend calls or randomness.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from collections import Counter
from typing import Any, Awaitable, Iterable, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite, Send, interrupt

from . import config, rules
from .personas import persona_name
from .state import (
    ROLE_ZH,
    TEAM_ZH,
    VIS_ALL,
    VIS_GOD,
    VIS_WOLVES,
    EventBuilder,
    GameState,
    alive_seats,
    get_player,
    seats_with_role,
    vis_seat,
)

log = logging.getLogger(__name__)

# One game is far below this many supersteps; LangGraph's default (25) is per
# astream() call and would trip while a dead human watches the AIs play on.
RECURSION_LIMIT = 2000

# stage -> speech kind, and where the speech loop goes once the queue is empty
STAGE_KIND = {
    "wolf_chat": "wolf_chat",
    "night_last_words": "last_words",
    "discussion": "discussion",
    "exile_last_words": "last_words",
}
STAGE_NEXT = {
    "wolf_chat": "wolf_pick",
    "night_last_words": "day_start",
    "discussion": "cast_vote",
    "exile_last_words": "after_exile",
}
# speech kind -> (event kind, visibility, thought action)
SPEECH_EVENT = {
    "discussion": ("speech", VIS_ALL, "speech"),
    "last_words": ("last_words", VIS_ALL, "last_words"),
    "wolf_chat": ("wolf_chat", VIS_WOLVES, "wolf_chat"),
}

SPEAK_PROMPTS = {
    "discussion": "輪到你發言了。",
    "last_words": "你出局了，請發表遺言。",
    "wolf_chat": "狼人夜聊時間：每隻狼只講一次，說出你今晚想刀誰、白天怎麼打（只有狼人看得到）。",
}
CHOOSE_PROMPTS = {
    "wolf_kill": "請選擇今晚要擊殺的玩家（你的選擇就是狼隊的最終決定）。",
    "seer_check": "預言家請睜眼，選擇今晚要查驗身份的玩家。",
    "exile_vote": "請投票選出要放逐的玩家，也可以選擇棄票。",
}
NIGHT_STATUS = "夜晚行動中…"
VOTE_STATUS = "投票中…"
NO_THOUGHT = "（沒有留下想法）"
BUDGET_REASON = "AI 的 token 額度用完了，遊戲提前結束"
# Night steps that don't happen (wolf-chat turns of dead wolves, dead or idle witch, dead
# seer) still wait about as long as the AI calls they replace, so the length of the night
# doesn't tell the human which roles are still alive. The wait is the running average AI call
# time (cap per sample) times a random factor.
PAD_JITTER = (0.7, 1.3)
LATENCY_SAMPLE_CAP_S = 6.0


# ------------------------------------------------------------- helpers
def _seats(seats: Iterable[int]) -> str:
    return "、".join(f"{s}號" for s in seats)


def _target_str(target: Optional[int]) -> str:
    return "棄票" if target is None else f"{target}號"


def _rng(state: GameState, salt: str) -> random.Random:
    """Per-step RNG for fallback choices; reproducible when the game has a seed."""
    seed = state.get("seed")
    if seed is None:
        return random.Random()
    return random.Random(f"{seed}:{len(state.get('events') or [])}:{salt}")


def _status(text: str) -> None:
    get_stream_writer()({"type": "status", "text": text})


def _no_update() -> dict:
    # never return {} from a node: LangGraph then streams `None` as the node's update
    return {"events": []}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _human_role(state: GameState) -> Optional[str]:
    """The human's role; None in an all-AI game (nobody is watching)."""
    human = state.get("human_seat")
    return None if human is None else get_player(state, human)["role"]


def _ballot(seat: int, value: Any) -> dict:
    return {"ballots": [{"seat": seat, "value": value}]}


def _ballots(state: GameState) -> dict[int, Any]:
    """seat -> the raw answer / AI result its wolf_pick or cast_vote run left."""
    return {b["seat"]: b["value"] for b in state.get("ballots") or []}


def _add_thought(eb: EventBuilder, state: GameState, seat: int, action: str, res: dict) -> None:
    """Emit the god-only thought explaining the event emitted right after it."""
    eb.add("thought", _text(res.get("thought")) or NO_THOUGHT, visibility=VIS_GOD, seat=seat, data={"action": action})


def _reveal(eb: EventBuilder, players: list) -> None:
    eb.add(
        "reveal",
        "身份公開："
        + "、".join(
            f"{p['seat']}號 {ROLE_ZH[p['role']]}（{'真人' if p['is_human'] else 'AI・' + persona_name(p['persona'])}）"
            for p in players
        ),
        data={
            "roles": [
                {"seat": p["seat"], "role": p["role"], "is_human": p["is_human"], "persona": p["persona"]}
                for p in players
            ],
        },
    )


# ----------------------------------------------------- interrupt payloads
def speak_payload(kind: str, day: int) -> dict:
    return {"type": "speak", "kind": kind, "day": day, "prompt": SPEAK_PROMPTS[kind]}


def choose_payload(action: str, day: int, options: list[int], allow_skip: bool = False) -> dict:
    return {
        "type": "choose",
        "action": action,
        "day": day,
        "prompt": CHOOSE_PROMPTS[action],
        "options": list(options),
        "allow_skip": allow_skip,
    }


def _witch_info_text(killed: Optional[int], witch: int, antidote: bool) -> str:
    if not antidote:
        return "你已用過解藥，無法得知今晚死訊。"
    if killed is None:
        return "今晚沒有人被狼人殺害。"
    if killed == witch:
        return f"今晚 {killed}號（你自己）被狼人殺害，女巫不能自救。"
    return f"今晚 {killed}號 被狼人殺害。"


def witch_payload(
    day: int, witch: int, killed: Optional[int], antidote: bool, can_save: bool, poison_options: list[int]
) -> dict:
    choices = []
    if can_save:
        choices.append(f"用解藥救 {killed}號")
    if poison_options:
        choices.append("對一名玩家使用毒藥")
    choices.append("不使用藥水")
    prompt = (
        f"女巫請睜眼。{_witch_info_text(killed, witch, antidote)}"
        f"你可以選擇：{'、'.join(choices)}（每晚最多只能用一瓶藥）。"
    )
    return {
        "type": "witch",
        "day": day,
        "prompt": prompt,
        "killed": killed,
        "can_save": can_save,
        "poison_options": list(poison_options),
    }


# ----------------------------------------------------------------- nodes
class _Nodes:
    def __init__(self, backend: Any):
        self.backend = backend
        # running average seconds per AI call kind (shared by every game of this graph)
        self.latency: dict[str, float] = {}
        # game_id -> time.monotonic() when the human has read the last AI speech (see _pace)
        self.read_until: dict[str, float] = {}

    async def _ai(self, call: Awaitable[Any], kind: str) -> dict:
        """Await one backend call. The contract says backends never raise; stay alive anyway."""
        start = time.monotonic()
        try:
            res = await call
        except Exception:  # noqa: BLE001
            log.exception("AI backend call failed")
            return {}
        finally:
            took = min(time.monotonic() - start, LATENCY_SAMPLE_CAP_S)
            prev = self.latency.get(kind)
            self.latency[kind] = took if prev is None else 0.8 * prev + 0.2 * took
        return res if isinstance(res, dict) else {}

    def over_budget(self, state: GameState) -> bool:
        """Whether the backend has used up this game's token budget (backends without one never do)."""
        check = getattr(self.backend, "over_budget", None)
        if check is None:
            return False
        try:
            return bool(check(state.get("game_id")))
        except Exception:  # noqa: BLE001
            log.exception("backend over_budget check failed")
            return False

    def _read(self, state: GameState, text: str) -> None:
        """The human sees an AI speech now: nothing else happens until the page has typed it out
        and config.SPEECH_PAUSE_S more has passed."""
        n = len(text)
        typing = min(n * config.TYPE_S_PER_CHAR, config.TYPE_MAX_S) if n >= 2 else 0.0
        self.read_until[state.get("game_id") or ""] = time.monotonic() + typing + config.SPEECH_PAUSE_S

    async def _pace(self, state: GameState) -> None:
        """Wait until the human has read the last AI speech (see _read). Every node starts with
        this (see build_graph; speech_turn does it itself); re-running a node after an interrupt
        waits no more, since its prompt was only shown after the wait."""
        left = self.read_until.get(state.get("game_id") or "", 0.0) - time.monotonic()
        if left > 0:
            await asyncio.sleep(left)

    def paced(self, node: Any) -> Any:
        """`node` (sync or async) preceded by _pace."""
        async def run(state: GameState) -> dict:
            await self._pace(state)
            out = node(state)
            return await out if inspect.isawaitable(out) else out
        return run

    def _finish(self, state: GameState) -> Optional[dict]:
        """Tell the backend the game's AI calls are over (it logs and forgets the token usage).
        Returns the game's token usage (None from a backend that doesn't count it)."""
        self.read_until.pop(state.get("game_id") or "", None)
        done = getattr(self.backend, "finish_game", None)
        if done is not None:
            try:
                usage = done(state.get("game_id"))
                return usage if isinstance(usage, dict) else None
            except Exception:  # noqa: BLE001
                log.exception("backend finish_game failed")
        return None

    async def _pad(self, state: GameState, role: str, kind: str, calls: int = 1) -> None:
        """Stand-in for `calls` skipped AI calls of `kind` by `role` (see PAD_JITTER): the latency of
        `kind`, else the average. Never before an interrupt; skipped when the human holds `role`
        (they know what happened anyway) or nobody is human."""
        latency = self.latency.get(kind) or (
            sum(self.latency.values()) / len(self.latency) if self.latency else 0)
        human_role = _human_role(state)
        if human_role is not None and human_role != role and latency > 0:
            await asyncio.sleep(calls * latency * random.uniform(*PAD_JITTER))

    # setup ---------------------------------------------------------------
    def setup(self, state: GameState) -> dict:
        board = config.ROLE_SETUP
        human_role = state.get("human_role")
        if human_role not in board:
            human_role = None
        players = rules.assign_players(random.Random(state.get("seed")), human_role,
                                       all_ai=bool(state.get("all_ai")), persona=state.get("persona"),
                                       persona_pool=state.get("persona_pool"))
        human = next((p["seat"] for p in players if p["is_human"]), None)

        counts = Counter(board)
        board = "、".join(f"{counts[r]} {ROLE_ZH[r]}" for r in ROLE_ZH if counts[r])
        eb = EventBuilder(state, day=0, phase="setup")
        eb.add(
            "system",
            f"遊戲開始！本局共 {len(players)} 名玩家：{board}（屠邊規則，無警長）。",
        )
        for p in players:
            mates = rules.wolf_teammates(players, p["seat"])
            text = f"你是 {p['seat']}號，你的身份是：{ROLE_ZH[p['role']]}。"
            if mates:
                text += f"你的狼隊友是 {_seats(mates)}。"
            eb.add(
                "role_assign",
                text,
                visibility=vis_seat(p["seat"]),
                seat=p["seat"],
                data={"role": p["role"], "teammates": mates},
            )
        return {
            "players": players,
            "human_seat": human,
            "day": 0,
            "phase": "setup",
            "stage": "",
            "speech_queue": [],
            "night": {"wolf_target": None, "saved": False, "poisoned": None},
            "antidote": True,
            "poison": True,
            "seer_checks": [],
            "last_night_deaths": [],
            "winner": None,
            "win_reason": "",
            "events": eb.events,
        }

    # night ---------------------------------------------------------------
    # The only night status is the one sent here (plus one right after each of the human's
    # own night actions): how many statuses a night has must not depend on who is alive.
    def night_start(self, state: GameState) -> dict:
        day = state["day"] + 1
        _status(NIGHT_STATUS)
        eb = EventBuilder(state, day=day, phase="night")
        eb.add("system", f"第{day}夜，天黑請閉眼。")
        wolves = seats_with_role(state, "werewolf")
        chat = len(wolves) >= 2
        _rng(state, "wolf_chat").shuffle(wolves)  # who opens the chat shouldn't depend on seat number
        return {
            "day": day,
            "phase": "night",
            "night": {"wolf_target": None, "saved": False, "poisoned": None},
            "stage": "wolf_chat" if chat else "",
            "speech_queue": wolves if chat else [],
            "events": eb.events,
        }

    async def speech_turn(self, state: GameState) -> dict:
        queue = list(state.get("speech_queue") or [])
        if not queue:
            return _no_update()
        seat, rest = queue[0], queue[1:]
        kind = STAGE_KIND[state["stage"]]
        ev_kind, visibility, action = SPEECH_EVENT[kind]

        if seat == state["human_seat"]:
            await self._pace(state)
            text = rules.coerce_speech(interrupt(speak_payload(kind, state["day"])))
            if kind == "wolf_chat":
                _status(NIGHT_STATUS)
            eb = EventBuilder(state)
        else:
            # the AI thinks while the human is still reading the last speech
            call = asyncio.ensure_future(
                self._ai(self.backend.speak(state, seat, kind), "wolf_chat" if kind == "wolf_chat" else "speak"))
            try:
                await self._pace(state)
                if kind == "last_words":
                    _status(f"{seat}號 正在發表遺言…")
                elif kind != "wolf_chat":  # night: see night_start
                    _status(f"{seat}號 正在發言…")
                res = await call
            finally:
                call.cancel()  # only does something if the run was cancelled
            eb = EventBuilder(state)
            _add_thought(eb, state, seat, action, res)
            text = rules.coerce_speech({"text": res.get("speech")})
        if not text:  # the human passed (or a broken backend said nothing)
            said = [e["text"] for e in state["events"] if e.get("seat") == seat and e["kind"] == ev_kind]
            text = rules.pass_speech(kind, _rng(state, "pass"), said)
        eb.add(ev_kind, text, visibility=visibility, seat=seat)
        human_role = _human_role(state)  # nobody human: nobody to wait for
        if human_role is not None and seat != state["human_seat"] and (kind != "wolf_chat" or human_role == "werewolf"):
            self._read(state, text)
            if kind != "wolf_chat":  # nobody is speaking during the pause (night keeps its status)
                _status("")
        return {"speech_queue": rest, "events": eb.events}

    # Ballots: wolf_pick / cast_vote run once per voter, all at the same time (one Send each, see
    # _goto). The human's run waits for the human's answer while the AIs' runs call the backend;
    # on resume LangGraph replays the finished AI runs instead of running them again, so the AIs
    # decide while the human thinks. Each run only leaves a ballot: the node that counts them
    # (wolf_kill / count_votes) writes every event, since event ids need one writer at a time.
    async def wolf_pick(self, state: GameState) -> dict:
        w = state["voter"]
        options = rules.wolf_kill_options(state["players"])
        if w == state["human_seat"]:
            value = interrupt(choose_payload("wolf_kill", state["day"], options))
            _status(NIGHT_STATUS)
            return _ballot(w, value)
        return _ballot(w, await self._ai(self.backend.wolf_target(state, w, options), "wolf_target"))

    async def wolf_kill(self, state: GameState) -> dict:
        """The wolves' ballots -> tonight's target: a human wolf's pick is final, else the AI wolves'
        consensus."""
        human = state["human_seat"]
        wolves = seats_with_role(state, "werewolf")
        options = rules.wolf_kill_options(state["players"])
        if not wolves or not options:  # unreachable: the game would be over
            return {"ballots": Overwrite([]), "events": []}
        # every wolf of the board gets a chat turn while 2+ are alive: wait out the missing turns
        chat_turns = len(wolves) if len(wolves) >= 2 else 0
        missing = len(seats_with_role(state, "werewolf", alive_only=False)) - chat_turns
        if missing > 0:
            await self._pad(state, "werewolf", "wolf_chat", calls=missing)

        ballots = _ballots(state)
        rng = _rng(state, "wolf_kill")
        human_wolf = human in wolves
        human_pick = rules.coerce_target(ballots.get(human), options, False, rng) if human_wolf else None
        ai_wolves = [w for w in wolves if w != human]
        eb = EventBuilder(state)
        picks: dict[int, int] = {}
        for w in wolves:
            if w == human:
                pick = human_pick
            else:
                res = ballots.get(w) or {}
                _add_thought(eb, state, w, "wolf_kill", res)
                pick = rules.coerce_target(res, options, False, rng)
            picks[w] = pick
            eb.add("wolf_pick", f"{w}號 選擇擊殺 {pick}號", visibility=VIS_WOLVES, seat=w, data={"target": pick})
        final = human_pick if human_wolf else rules.wolf_consensus({w: picks[w] for w in ai_wolves})
        eb.add("wolf_kill", f"狼隊決定擊殺 {final}號", visibility=VIS_WOLVES, data={"target": final})
        return {"night": {**state["night"], "wolf_target": final}, "ballots": Overwrite([]), "events": eb.events}

    async def witch(self, state: GameState) -> dict:
        witches = seats_with_role(state, "witch")
        antidote, poison = state["antidote"], state["poison"]
        if not witches or not (antidote or poison):
            await self._pad(state, "witch", "witch")
            return _no_update()
        w = witches[0]
        night = state["night"]
        killed = night["wolf_target"] if antidote else None
        can_save = antidote and killed is not None and killed != w
        poison_options = rules.alive_others(state["players"], w) if poison else []
        has_action = can_save or bool(poison_options)
        is_human = w == state["human_seat"]

        if is_human and has_action:
            value = interrupt(witch_payload(state["day"], w, killed, antidote, can_save, poison_options))
            action, target = rules.coerce_witch(value, killed, can_save, poison_options)
            _status(NIGHT_STATUS)

        eb = EventBuilder(state)
        eb.add("witch_info", _witch_info_text(killed, w, antidote), visibility=vis_seat(w), seat=w,
               data={"killed": killed})
        if not has_action:
            await self._pad(state, "witch", "witch")
            return {"events": eb.events}
        if not is_human:
            res = await self._ai(self.backend.witch(state, w, killed, can_save, poison_options), "witch")
            _add_thought(eb, state, w, "witch", res)
            action, target = rules.coerce_witch(res, killed, can_save, poison_options)

        update: dict = {"events": eb.events}
        if action == "save":
            text = f"女巫使用解藥，救活了 {target}號。"
            update |= {"antidote": False, "night": {**night, "saved": True}}
        elif action == "poison":
            text = f"女巫對 {target}號 使用了毒藥。"
            update |= {"poison": False, "night": {**night, "poisoned": target}}
        else:
            text = "女巫今晚沒有使用藥水。"
        eb.add("witch_action", text, visibility=vis_seat(w), seat=w, data={"action": action, "target": target})
        return update

    async def seer(self, state: GameState) -> dict:
        seers = seats_with_role(state, "seer")
        if not seers:
            await self._pad(state, "seer", "seer_check")
            return _no_update()
        s = seers[0]
        options = rules.alive_others(state["players"], s)
        if not options:
            return _no_update()
        if s == state["human_seat"]:
            value = interrupt(choose_payload("seer_check", state["day"], options))
            target = rules.coerce_target(value, options, False, _rng(state, "seer"))
            _status(NIGHT_STATUS)
            eb = EventBuilder(state)
        else:
            res = await self._ai(self.backend.seer_check(state, s, options), "seer_check")
            eb = EventBuilder(state)
            _add_thought(eb, state, s, "seer_check", res)
            target = rules.coerce_target(res, options, False, _rng(state, "seer"))
        wolf = get_player(state, target)["role"] == "werewolf"
        eb.add("seer_result", f"{target}號 的身份是：{'狼人' if wolf else '好人'}", visibility=vis_seat(s),
               seat=s, data={"target": target, "is_wolf": wolf})
        checks = [*(state.get("seer_checks") or []), {"day": state["day"], "target": target, "is_wolf": wolf}]
        return {"seer_checks": checks, "events": eb.events}

    # day -----------------------------------------------------------------
    def dawn(self, state: GameState) -> dict:
        deaths = rules.resolve_night(state["night"])
        players = rules.mark_dead(state["players"], deaths)
        eb = EventBuilder(state, phase="day")
        eb.add("death", f"天亮了，昨晚 {_seats(deaths)} 死亡。" if deaths else "天亮了，昨晚是平安夜。",
               data={"seats": deaths})
        update: dict = {"phase": "day", "players": players, "last_night_deaths": deaths,
                        "stage": "", "speech_queue": [], "events": eb.events}
        winner, reason = rules.check_winner(players)
        last_words = deaths if state["day"] == 1 else []
        if winner:
            update |= {"winner": winner, "win_reason": reason}
        elif last_words:
            eb.add("system", f"請 {_seats(last_words)} 發表遺言。")
            update |= {"stage": "night_last_words", "speech_queue": last_words}
        return update

    def day_start(self, state: GameState) -> dict:
        day = state["day"]
        order = rules.discussion_order(alive_seats(state), state.get("last_night_deaths") or [], day,
                                       len(state["players"]))
        eb = EventBuilder(state)
        eb.add("system", f"第{day}天，請依序發言。發言順序：{' → '.join(f'{s}號' for s in order)}")
        return {"stage": "discussion", "speech_queue": order, "events": eb.events}

    async def cast_vote(self, state: GameState) -> dict:
        v, human = state["voter"], state["human_seat"]
        voters = alive_seats(state)
        options = rules.alive_others(state["players"], v)
        if v == human:
            value = interrupt(choose_payload("exile_vote", state["day"], options, allow_skip=True))
            if len(voters) > 1:
                _status(VOTE_STATUS)
            return _ballot(v, value)
        if human not in voters and v == voters[0]:  # nobody is asked: one status for the whole vote
            _status(VOTE_STATUS)
        return _ballot(v, await self._ai(self.backend.exile_vote(state, v, options), "exile_vote"))

    def count_votes(self, state: GameState) -> dict:
        human, day = state["human_seat"], state["day"]
        ballots = _ballots(state)
        rng = _rng(state, "exile")
        eb = EventBuilder(state)
        votes = []
        for v in alive_seats(state):
            if v != human:
                _add_thought(eb, state, v, "exile_vote", ballots.get(v) or {})
            target = rules.coerce_target(ballots.get(v), rules.alive_others(state["players"], v), True, rng)
            votes.append({"voter": v, "target": target})
        tally, top, exiled = rules.tally_votes(votes)

        listing = "、".join(f"{v['voter']}號 → {_target_str(v['target'])}" for v in votes)
        if exiled is not None:
            result = f"{exiled}號 以 {tally[str(exiled)]} 票被放逐出局，請發表遺言。"
        elif top:
            result = f"{_seats(top)} 平票（各 {tally[str(top[0])]} 票），本輪無人出局。"
        else:
            result = "全員棄票，本輪無人出局。"
        eb.add("vote_result", f"放逐投票：{listing}。{result}",
               data={"votes": votes, "tally": tally, "exiled": exiled})

        update: dict = {"ballots": Overwrite([]), "events": eb.events}
        if exiled is not None:
            update |= {"players": rules.mark_dead(state["players"], [exiled]),
                       "stage": "exile_last_words", "speech_queue": [exiled]}
        else:
            update |= {"stage": "", "speech_queue": []}
            if day >= config.MAX_DAYS:
                update |= {"winner": "draw", "win_reason": rules.draw_reason()}
        return update

    def after_exile(self, state: GameState) -> dict:
        winner, reason = rules.check_winner(state["players"])
        if winner:
            return {"winner": winner, "win_reason": reason}
        if state["day"] >= config.MAX_DAYS:
            return {"winner": "draw", "win_reason": rules.draw_reason()}
        return _no_update()

    # end -----------------------------------------------------------------
    def game_over(self, state: GameState) -> dict:
        winner, reason = state.get("winner"), state.get("win_reason")
        if not winner and self.over_budget(state):  # cut short: keep a result the board already shows
            winner, reason = rules.check_winner(state["players"])
            if not winner:
                winner, reason = "draw", BUDGET_REASON
        winner = winner or "draw"
        reason = reason or rules.draw_reason()
        eb = EventBuilder(state, phase="end")
        text = f"遊戲結束！{reason}。" if winner == "draw" else f"遊戲結束！{reason}，{TEAM_ZH[winner]}獲勝！"
        eb.add("game_over", text, data={"winner": winner, "reason": reason})
        _reveal(eb, state["players"])
        usage = self._finish(state)
        return {"phase": "end", "winner": winner, "win_reason": reason, "stage": "", "speech_queue": [],
                "token_usage": usage, "events": eb.events}


# --------------------------------------------------------------- routing
# ballot node -> (who runs it, the node that counts the ballots)
BALLOTS = {
    "wolf_pick": (lambda state: seats_with_role(state, "werewolf"), "wolf_kill"),
    "cast_vote": (alive_seats, "count_votes"),
}


def _goto(state: GameState, node: str) -> Any:
    """`node`; for a ballot node, one Send per voter, which all run at the same time."""
    if node not in BALLOTS:
        return node
    voters, count = BALLOTS[node]
    return [Send(node, {**state, "voter": v}) for v in voters(state)] or count  # no voters: unreachable


def _after_night_start(state: GameState) -> Any:
    return "speech_turn" if state.get("speech_queue") else _goto(state, "wolf_pick")


def _after_speech(state: GameState) -> Any:
    return "speech_turn" if state.get("speech_queue") else _goto(state, STAGE_NEXT[state["stage"]])


def _after_dawn(state: GameState) -> str:
    if state.get("winner"):
        return "game_over"
    return "speech_turn" if state.get("speech_queue") else "day_start"


def _after_count_votes(state: GameState) -> str:
    if state.get("speech_queue"):
        return "speech_turn"
    return "game_over" if state.get("winner") else "night_start"


def _after_exile(state: GameState) -> str:
    return "game_over" if state.get("winner") else "night_start"


# ----------------------------------------------------------------- build
def build_graph(backend: Any = None, checkpointer: Any = None):
    """Compile and return the werewolf StateGraph.
    backend None -> werewolf.ai.get_backend(); checkpointer None -> InMemorySaver()."""
    if backend is None:
        from .ai import get_backend

        backend = get_backend()
    n = _Nodes(backend)
    g = StateGraph(GameState)
    for name in ("setup", "night_start", "speech_turn", "wolf_pick", "wolf_kill", "witch", "seer", "dawn",
                 "day_start", "cast_vote", "count_votes", "after_exile", "game_over"):
        node = getattr(n, name)
        g.add_node(name, node if name == "speech_turn" else n.paced(node))  # speech_turn paces itself

    def after_ai(node: str, route, targets: list[str]) -> None:
        """Edges out of a node that calls the AI: once the game is over its token budget, end it."""
        g.add_conditional_edges(node, lambda s: "game_over" if n.over_budget(s) else route(s),
                                sorted({*targets, "game_over"}))

    g.add_edge(START, "setup")
    g.add_edge("setup", "night_start")
    g.add_conditional_edges("night_start", _after_night_start, ["speech_turn", "wolf_pick", "wolf_kill"])
    after_ai("speech_turn", _after_speech,
             ["speech_turn", "wolf_pick", "wolf_kill", "day_start", "cast_vote", "count_votes", "after_exile"])
    g.add_edge("wolf_pick", "wolf_kill")  # once every wolf's run is done
    after_ai("wolf_kill", lambda s: "witch", ["witch"])
    after_ai("witch", lambda s: "seer", ["seer"])
    after_ai("seer", lambda s: "dawn", ["dawn"])
    g.add_conditional_edges("dawn", _after_dawn, ["game_over", "speech_turn", "day_start"])
    g.add_edge("day_start", "speech_turn")
    g.add_edge("cast_vote", "count_votes")  # once every voter's run is done
    after_ai("count_votes", _after_count_votes, ["speech_turn", "game_over", "night_start"])
    g.add_conditional_edges("after_exile", _after_exile, ["game_over", "night_start"])
    g.add_edge("game_over", END)
    graph = g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())
    return graph.with_config(recursion_limit=RECURSION_LIMIT)
