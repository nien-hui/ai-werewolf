"""What one seat may know, and the game log / situation it is shown.

INFORMATION SAFETY: everything here is built only from what one seat may know:
its own player record, public table state (alive/dead, speaking order) and
`events_visible_to(state, seat, include_own_god=True)`. Never read
`human_seat`, `is_human`, other players' roles/personas or others' thoughts.
"""
from __future__ import annotations

from typing import Optional

from .. import rules
from ..state import ROLE_ZH, Event, GameState, alive_seats, events_visible_to, get_player

# ------------------------------------------------------------- private knowledge


def _seats_str(seats, me: Optional[int] = None) -> str:
    if not seats:
        return "無"
    return "、".join(f"{s}號" + ("（你）" if s == me else "") for s in seats)


def _wolf_word(is_wolf: bool) -> str:
    return "狼人" if is_wolf else "好人"


def private_knowledge(state: GameState, seat: int, events: Optional[list[Event]] = None) -> dict:
    """What `seat` privately knows, derived from its own record + visible events."""
    me = get_player(state, seat)
    role = me["role"]
    if events is None:
        events = events_visible_to(state, seat, include_own_god=True)
    k: dict = {"role": role, "teammates": [], "seer_checks": [], "wolf_kills": [], "witch_log": []}

    if role == "werewolf":
        mates = None
        for e in events:
            if e["kind"] == "role_assign" and e.get("seat") == seat:
                mates = e["data"].get("teammates")
        if mates is None:  # fallback: wolves legitimately know each other
            mates = [p["seat"] for p in state["players"] if p["role"] == "werewolf" and p["seat"] != seat]
        k["teammates"] = sorted(int(m) for m in mates if int(m) != seat)
        k["wolf_kills"] = [
            {"day": e["day"], "target": e["data"].get("target")}
            for e in events
            if e["kind"] == "wolf_kill" and e["data"].get("target") is not None
        ]

    if role == "seer":
        seen = set()
        for e in events:
            if e["kind"] == "seer_result":
                c = {"day": e["day"], "target": e["data"].get("target"), "is_wolf": bool(e["data"].get("is_wolf"))}
                if (c["day"], c["target"]) not in seen:
                    seen.add((c["day"], c["target"]))
                    k["seer_checks"].append(c)
        for c in state.get("seer_checks") or []:  # own results; fallback if events are missing
            if (c.get("day"), c.get("target")) not in seen:
                seen.add((c.get("day"), c.get("target")))
                k["seer_checks"].append({"day": c.get("day"), "target": c.get("target"), "is_wolf": bool(c.get("is_wolf"))})
        k["seer_checks"].sort(key=lambda c: (c["day"] or 0))

    if role == "witch":
        k["antidote"] = bool(state.get("antidote", True))
        k["poison"] = bool(state.get("poison", True))
        for e in events:
            if e["kind"] == "witch_info":
                k["witch_log"].append(f"第{e['day']}夜：" + _witch_info_line(e))
            elif e["kind"] == "witch_action":
                k["witch_log"].append(f"第{e['day']}夜：" + _witch_action_line(e))
    return k


def _witch_info_line(e: Event) -> str:
    killed = e["data"].get("killed")
    if killed is not None:
        return f"{killed}號 被狼人殺害"
    return e.get("text") or "你無法得知今晚死訊"


def _witch_action_line(e: Event) -> str:
    action, target = e["data"].get("action"), e["data"].get("target")
    if action == "save":
        return f"你用解藥救了 {target}號" if target is not None else "你使用了解藥"
    if action == "poison":
        return f"你對 {target}號 用了毒藥"
    return "你沒有用藥"


def public_deaths(state: GameState, events: list[Event]) -> dict[int, str]:
    """seat -> how they died, from public events only."""
    out: dict[int, str] = {}
    for e in events:
        if e["visibility"] != "all":
            continue
        if e["kind"] == "death":
            for s in e["data"].get("seats") or []:
                out.setdefault(int(s), f"第{e['day']}夜死亡")
        elif e["kind"] == "vote_result" and e["data"].get("exiled") is not None:
            out.setdefault(int(e["data"]["exiled"]), f"第{e['day']}天被放逐")
    for p in state["players"]:
        if not p["alive"]:
            out.setdefault(p["seat"], "已死亡")
    return out


# -------------------------------------------------------------------- context

def _phase_header(day: int, phase: str) -> str:
    if phase == "night":
        return f"【第{day}夜】"
    if phase == "day":
        return f"【第{day}天】"
    if phase == "end":
        return "【遊戲結束】"
    return "【遊戲開始】"


def _strip_seat_prefix(text: str, seat: Optional[int]) -> str:
    text = (text or "").strip()
    if seat is not None:
        for sep in ("：", ":"):
            p = f"{seat}號{sep}"
            if text.startswith(p):
                return text[len(p):].strip()
    return text


# A player's own 「」 become 『』 and 〔〕 (the host-line marker) become 【】.
_QUOTE_MARKS = str.maketrans("「」〔〕", "『』【】")


def _quoted(text: str, seat: Optional[int]) -> str:
    """What a player typed, as one line inside 「」 (see _QUOTE_MARKS): nothing a player types
    can pass for a host announcement or a private line in the log."""
    return "「" + rules.one_line(_strip_seat_prefix(text, seat)).translate(_QUOTE_MARKS) + "」"


def _render_event(e: Event, me: int) -> Optional[str]:
    kind, seat, data = e["kind"], e.get("seat"), e.get("data") or {}
    tag = f"{seat}號（你）" if seat == me else f"{seat}號"
    text = e.get("text") or ""
    if kind == "reveal":  # names the human; the game is over anyway
        return None
    if kind == "thought":
        if seat != me:
            return None
        return f"（你當時的想法：{text}）"
    if kind == "role_assign":
        role = data.get("role")
        line = f"（你的私密資訊）你的身份是{ROLE_ZH.get(role, role)}"
        if data.get("teammates"):
            line += f"，狼隊友：{_seats_str(data['teammates'])}"
        return line
    if kind == "speech":
        return f"{tag}：{_quoted(text, seat)}"
    if kind == "last_words":
        return f"{tag}（遺言）：{_quoted(text, seat)}"
    if kind == "wolf_chat":
        return f"（狼隊頻道）{tag}：{_quoted(text, seat)}"
    if kind == "wolf_pick":
        return f"（狼隊頻道）{tag} 想刀 {data.get('target')}號"
    if kind == "wolf_kill":
        return f"（狼隊頻道）狼隊決定擊殺 {data.get('target')}號"
    if kind == "witch_info":
        return f"（你的私密資訊）{_witch_info_line(e)}"
    if kind == "witch_action":
        return f"（你的私密資訊）{_witch_action_line(e)}"
    if kind == "seer_result":
        return f"（你的私密資訊）查驗結果：{data.get('target')}號 是{_wolf_word(bool(data.get('is_wolf')))}"
    if kind == "vote_result" and "votes" in data:
        votes = "、".join(
            f"{v['voter']}號→" + (f"{v['target']}號" if v.get("target") is not None else "棄票")
            for v in data["votes"]
        )
        exiled = data.get("exiled")
        result = f"{exiled}號 被放逐" if exiled is not None else "沒有人出局"
        return f"〔投票〕{votes}。結果：{result}"
    # system / death / game_over / reveal(roles only) / anything else visible
    return f"〔主持人〕{text}" if text else None


def render_log(state: GameState, seat: int, events: Optional[list[Event]] = None) -> str:
    if events is None:
        events = events_visible_to(state, seat, include_own_god=True)
    lines: list[str] = []
    cur = None
    for e in events:
        line = _render_event(e, seat)
        if line is None:
            continue
        key = (e.get("day", 0), e.get("phase", "setup"))
        if key != cur:
            cur = key
            lines.append("")
            lines.append(_phase_header(*key))
        lines.append(line)
    return "\n".join(lines).strip() or "（目前還沒有任何事件）"


def _phase_zh(state: GameState) -> str:
    day, phase = state.get("day", 0), state.get("phase", "setup")
    if phase == "night":
        return f"第{day}夜"
    if phase == "day":
        return f"第{day}天白天"
    if phase == "end":
        return "遊戲已結束"
    return "遊戲開始前"


def speaking_order(state: GameState, events: list[Event]) -> tuple[list[int], list[int]]:
    """(today's full discussion order, seats that already spoke today). Public info."""
    day = state.get("day", 0)
    spoken = [
        e["seat"] for e in events
        if e["kind"] == "speech" and e.get("day") == day and e.get("phase") == "day" and e["visibility"] == "all"
    ]
    order = list(spoken) + [s for s in state.get("speech_queue") or [] if s not in spoken]
    return order, spoken


def situation_text(state: GameState, seat: int, events: Optional[list[Event]] = None) -> str:
    if events is None:
        events = events_visible_to(state, seat, include_own_god=True)
    me = get_player(state, seat)
    k = private_knowledge(state, seat, events)
    role = k["role"]
    alive = alive_seats(state)
    deaths = public_deaths(state, events)
    lines = [
        "【目前狀況】",
        f"- 現在：{_phase_zh(state)}",
        f"- 你是 {seat}號，身份：{ROLE_ZH.get(role, role)}" + ("" if me["alive"] else "（你已出局）"),
        f"- 存活：{_seats_str(alive, seat)}",
        "- 出局：" + ("、".join(f"{s}號（{why}）" for s, why in sorted(deaths.items())) if deaths else "無"),
    ]
    priv: list[str] = []
    if role == "werewolf":
        mates = [f"{m}號" + ("" if get_player(state, m)["alive"] else "（已出局）") for m in k["teammates"]]
        priv.append(f"狼隊友：{'、'.join(mates) or '無'}")
        for w in k["wolf_kills"]:
            priv.append(f"第{w['day']}夜 狼隊擊殺 {w['target']}號")
    if role == "seer":
        for c in k["seer_checks"]:
            priv.append(f"第{c['day']}夜 查驗 {c['target']}號：{_wolf_word(c['is_wolf'])}")
        if not k["seer_checks"]:
            priv.append("還沒有查驗結果")
    if role == "witch":
        priv.append(f"解藥：{'還在' if k['antidote'] else '已用掉'}；毒藥：{'還在' if k['poison'] else '已用掉'}")
        priv.extend(k["witch_log"])
    if priv:
        lines.append("- 你的私密資訊：" + "；".join(priv))
    if state.get("phase") == "day" and state.get("stage") == "discussion":
        order, spoken = speaking_order(state, events)
        if seat in alive and seat not in order:
            order.insert(len(spoken), seat)
        if order:
            line = f"- 今天的發言順序：{' → '.join(f'{s}號' for s in order)}"
            if seat in order and seat not in spoken:
                idx = order.index(seat)
                pos = "第一個" if idx == 0 else ("最後一個" if idx == len(order) - 1 else f"第 {idx + 1} 個")
                line += f"；你是{pos}發言"
            lines.append(line)
    return "\n".join(lines)


def saved_kill(state: GameState, seat: int) -> Optional[int]:
    """For a wolf: last night's kill target if they survived (the witch saved them), else None."""
    if get_player(state, seat)["role"] != "werewolf":
        return None
    alive = set(alive_seats(state))
    for w in private_knowledge(state, seat)["wolf_kills"]:
        if w["day"] == state.get("day") and w["target"] in alive:
            return w["target"]
    return None
