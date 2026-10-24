"""Assemble the AI players' messages: system prompt (fixed text from text/*.md) + game log + task.

INFORMATION SAFETY: see knowledge.py — build only from what the seat may know.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..personas import persona_style
from ..state import ROLE_TEAM, ROLE_ZH, TEAM_ZH, GameState, events_visible_to, get_player
from .knowledge import _seats_str, private_knowledge, render_log, saved_kill, situation_text, speaking_order
from .loader import OUTPUT_RULES, role_guide, rules_text

# --------------------------------------------------------------------- system


def system_prompt(state: GameState, seat: int) -> str:
    me = get_player(state, seat)
    k = private_knowledge(state, seat)
    role = me["role"]
    who = f"你是 {seat}號 玩家，身份是「{ROLE_ZH.get(role, role)}」（{TEAM_ZH[ROLE_TEAM.get(role, 'good')]}）。"
    if role == "werewolf":
        who += f"你的狼隊友是：{_seats_str(k['teammates'])}。"
    style = persona_style(me.get("persona", ""))
    persona = f"【你的說話個性】\n{style}\n個性只影響語氣，不影響你的判斷力。" if style else ""
    parts = [
        f"你正在玩一場線上文字狼人殺，共 {len(state['players'])} 位玩家。{who}",
        rules_text(),
        role_guide(role),
        persona,
        OUTPUT_RULES,
    ]
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------- tasks

_THOUGHT_HINT = "thought 寫你的內心真實想法（2~4 句：局勢判斷、你的打算和理由），只有你自己看得到。"


def task_speak(state: GameState, seat: int, kind: str) -> str:
    day = state.get("day", 0)
    role = get_player(state, seat)["role"]
    saved = saved_kill(state, seat) if kind != "wolf_chat" else None
    warn = (
        f"注意：昨晚狼隊刀的是 {saved}號，但他沒死（被女巫救了）。刀口只有狼人和女巫知道，好人只知道是平安夜；"
        f"除非女巫已經公開說過，公開發言絕對不要暗示 {saved}號 昨晚被刀、沒死、命大或被救。\n"
        if saved is not None else ""
    )
    if kind == "wolf_chat":
        body = (
            f"現在是第{day}夜的狼隊頻道，只有狼人看得到，可以講真話。夜聊每隻狼只講一次，"
            "所以不要提問、不要等隊友回答，直接講你的主張：今晚刀誰（說出座號和理由）、"
            "白天要不要有人悍跳預言家、怎麼互相配合。"
        )
        mate_spoke = any(
            e["kind"] == "wolf_chat" and e.get("day") == day and e.get("seat") != seat
            for e in events_visible_to(state, seat)
        )
        if mate_spoke:
            body += "隊友已經講過了，你是最後一個：同意就順著前面的提議補充，不同意就直接給出你的版本和理由，把今晚的計畫定下來。"
        else:
            body += "你先講，隊友會接著回應你的提議。"
    elif kind == "last_words":
        body = (
            "你已經出局了，這是你的遺言，說完就不能再發言。"
            "可以公開身份或資訊、留下你的懷疑、給隊友方向。"
        )
        if role == "seer":
            body += "你是預言家，記得把所有查驗結果都報出來。"
        elif role == "werewolf":
            body += "你是狼人，可以繼續偽裝成好人，幫隊友混淆視聽。"
    else:
        body = f"現在是第{day}天白天討論，輪到你發言，大家發言完會投票放逐一人。"
        order, spoken = speaking_order(state, events_visible_to(state, seat))
        if not spoken:
            body += "你是第一個發言，資訊還很少，可以先表態或分享你的想法。"
        if role == "werewolf":
            body += "記得你是狼，要像好人一樣發言。"
    return f"{warn}{body}\n{_THOUGHT_HINT}\nspeech 是你要說出口的話（1~4 句）。"


def task_wolf_target(state: GameState, seat: int, options: list[int]) -> str:
    return (
        f"現在是第{state.get('day', 0)}夜，請選擇今晚要擊殺的玩家，可選：{options}。"
        "盡量配合狼隊夜聊的共識。\n" + _THOUGHT_HINT + "\ntarget 填座號。"
    )


def task_witch(state: GameState, seat: int, killed: Optional[int], can_save: bool, poison_options: list[int]) -> str:
    lines = [f"現在是第{state.get('day', 0)}夜，女巫請行動（同一晚只能用一瓶藥）。"]
    if killed is not None:
        lines.append(f"今晚 {killed}號 被狼人殺害。")
    elif not state.get("antidote", True):
        lines.append("你已經用過解藥，無法得知今晚死訊。")
    else:
        lines.append("今晚沒有人被狼人殺害。")
    if can_save and killed is not None:
        lines.append(f"你可以用解藥救 {killed}號（action=save）。")
    elif killed is not None and killed == seat:
        lines.append("女巫不能自救，這晚不能用解藥。")
    elif not state.get("antidote", True):
        lines.append("你的解藥已經用掉了。")
    else:
        lines.append("你現在不能用解藥。")
    if poison_options:
        lines.append(f"你可以用毒藥毒一個人（action=poison，target 填座號），可選：{poison_options}。")
    else:
        lines.append("你的毒藥已經用掉了。")
    lines.append("什麼都不做就選 action=none。")
    lines.append(_THOUGHT_HINT)
    return "\n".join(lines)


def task_seer_check(state: GameState, seat: int, options: list[int]) -> str:
    hint = (
        "第一晚還沒有人發言，隨便選一個就好，不要編造理由。"
        if state.get("day", 0) <= 1
        else "通常優先驗還沒驗過、發言可疑或對局勢關鍵的人。"
    )
    return (
        f"現在是第{state.get('day', 0)}夜，請選擇今晚要查驗的玩家，可選：{options}。{hint}\n"
        + _THOUGHT_HINT + "\ntarget 填座號。"
    )


def task_exile_vote(state: GameState, seat: int, options: list[int]) -> str:
    return (
        f"第{state.get('day', 0)}天的發言結束了，請投票放逐一位玩家，可選：{options}，"
        "也可以棄票（target 填 null）。平票或全部棄票就沒人出局。\n"
        "好人：投你目前最懷疑是狼的人，資訊少也要給一票。棄票保護不了任何人，狼人的票集中就能放逐好人，"
        "尤其要小心叫大家「今天先別投」的人；神職只剩一個時，他被放逐狼人就直接贏了。\n"
        "狼人：別投隊友，盡量把票推到好人身上。\n" + _THOUGHT_HINT
    )


# ------------------------------------------------------------------- messages


def user_prompt(state: GameState, seat: int, task: str) -> str:
    events = events_visible_to(state, seat, include_own_god=True)
    return (
        "【到目前為止你看得到的紀錄】\n"
        f"{render_log(state, seat, events)}\n\n"
        f"{situation_text(state, seat, events)}\n\n"
        f"【你的任務】\n{task}"
    )


def build_messages(state: GameState, seat: int, task: str) -> list[BaseMessage]:
    return [SystemMessage(system_prompt(state, seat)), HumanMessage(user_prompt(state, seat, task))]


def all_prompt_text(state: GameState, seat: int, task: str) -> str:
    """Concatenated prompt text (used by tests to check for leaks)."""
    return "\n".join(str(m.content) for m in build_messages(state, seat, task))
