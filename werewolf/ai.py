"""AI player backends.

Every public backend method NEVER raises and ALWAYS returns legal values.
GeminiBackend: LLM with structured output, retries with feedback, strict
validation, random legal fallback, per-game token accounting and budget.
FakeBackend: offline seeded-random player.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from . import config, prompts, rules
from .state import GameState, events_visible_to, get_player

log = logging.getLogger(__name__)

SPEAK_KINDS = ("discussion", "last_words", "wolf_chat")
MAX_THOUGHT_CHARS = 400
# Calls normally take 1-3 s but the API occasionally hangs; cut such attempts
# short and retry instead of waiting the full config.LLM_TIMEOUT_S.
ATTEMPT_TIMEOUT_S = float(os.getenv("WEREWOLF_LLM_ATTEMPT_TIMEOUT", "20"))
# Measured on gemini-3.5-flash-lite with these prompts: the default/"minimal"
# thinking level runs away (~63k reasoning tokens, ~2 min, sometimes derailing
# into unrelated English) on ~40% of calls; "low" on 0/64 short-state calls and
# ~3-5% of calls over full games. The output cap turns those into a ~4 s retry.
THINKING_LEVEL = os.getenv("WEREWOLF_THINKING_LEVEL", "low")  # "" -> model default
MAX_OUTPUT_TOKENS = 1200  # normal replies use 100-600 tokens incl. reasoning
RETRY_BACKOFF_S = 2.0  # sleep before retrying after an API error (x attempt, max 5 s)
# Per-game cap on Gemini tokens: input + output (output includes reasoning), over every attempt.
# A normal 6-player game uses ~150-400k input and ~25k output tokens. A game that reaches it
# makes no more calls and the graph ends it early (see over_budget). 0 = no cap.
TOKEN_BUDGET = int(os.getenv("WEREWOLF_TOKEN_BUDGET", "1200000"))
MAX_TRACKED_GAMES = 500  # beyond this, the oldest unfinished games' usage is forgotten


def attempt_timeout() -> float:
    return min(config.LLM_TIMEOUT_S, ATTEMPT_TIMEOUT_S)


class AIBackend(Protocol):
    async def speak(self, state: GameState, seat: int, kind: str) -> dict: ...
    async def wolf_target(self, state: GameState, seat: int, options: list[int]) -> dict: ...
    async def witch(self, state: GameState, seat: int, killed: Optional[int], can_save: bool,
                    poison_options: list[int]) -> dict: ...
    async def seer_check(self, state: GameState, seat: int, options: list[int]) -> dict: ...
    async def exile_vote(self, state: GameState, seat: int, options: list[int]) -> dict: ...


# ------------------------------------------------------------- output models
_THOUGHT_DESC = "你的內心真實想法（2~4句，繁體中文）：局勢判斷、你的打算和理由。只有你自己看得到。"


class SpeechOut(BaseModel):
    thought: str = Field(description=_THOUGHT_DESC)
    speech: str = Field(description="你要公開說的話：1~4句繁體中文台灣口語，像在群組打字，開頭不要加「N號：」。")


class WolfKillOut(BaseModel):
    thought: str = Field(description=_THOUGHT_DESC)
    target: int = Field(description="今晚要擊殺的座號，必須是可選名單裡的數字。")


class SeerCheckOut(BaseModel):
    thought: str = Field(description=_THOUGHT_DESC)
    target: int = Field(description="今晚要查驗的座號，必須是可選名單裡的數字。")


class ExileOut(BaseModel):
    thought: str = Field(description=_THOUGHT_DESC)
    target: Optional[int] = Field(description="要放逐的座號，必須在可選名單裡；棄票填 null。")


class WitchOut(BaseModel):
    thought: str = Field(description=_THOUGHT_DESC)
    action: Literal["save", "poison", "none"] = Field(
        description="save=用解藥救今晚被殺的人；poison=用毒藥毒人；none=不用藥。"
    )
    target: Optional[int] = Field(
        default=None, description="只有 action=poison 時填要毒的座號（必須在可選名單裡），其他情況填 null。"
    )


# ------------------------------------------------------------ text cleanup
_TAG_RE = re.compile(r"^\s*[【\[〔(（]?\s*(?:狼隊頻道|遺言|發言)\s*[】\]〕)）]\s*[：:]?\s*")  # copied log tags
_LABEL_RE = re.compile(r"^\s*(?:發言|遺言|speech)\s*[：:]\s*", re.IGNORECASE)


def _seat_prefix_re(seat: Optional[int]) -> re.Pattern:
    num = str(seat) if seat is not None else r"\d{1,2}"
    # "3號：" "【3號】：" "3號（你）：" "玩家3號:"; only the speaker's own seat when known
    return re.compile(
        rf"^\s*[【\[〔(（「]?\s*(?:玩家\s*)?{num}\s*號\s*(?:[（(][^）)]{{0,6}}[）)])?\s*[】\]〕)）」]?\s*[：:﹕]\s*"
    )


_QUOTES = {"「": "」", "『": "』", '"': '"', "“": "”"}

# Safety net only (prompts demand 繁體 and live samples had none): common
# simplified-only characters with an unambiguous traditional form in this game.
_S2T = str.maketrans({a: b for a, b in (
    "这這 们們 说說 么麼 为為 时時 话話 选選 杀殺 预預 验驗 题題 让讓 给給 还還 进進 过過 对對 没沒 关關 现現 开開 间間 问問 长長 头頭 实實 买買 东東 车車 边邊 体體 爱愛 亲親 视視 觉覺 认認 识識 读讀 写寫 谁誰 两兩 号號 会會 个個 吗嗎 应應 该該 样樣 "
    "怀懷 场場 决決 药藥 带帶 风風 点點 节節 发發 脸臉 种種 队隊 员員 刚剛 终終 惊驚 吓嚇 烦煩 战戰 乐樂 学學 习習 书書 处處 备備 杂雜 岁歲 冲衝 动動 欢歡 从從 众眾 优優 伤傷 传傳 价價 儿兒 军軍 农農 减減 则則 创創 别別 劳勞 势勢 区區 华華 单單 卫衛 历歷 压壓 "
    "请請 帮幫 纠糾 错錯 润潤"
).split()})
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
# Something any on-topic answer mentions. Seen live (~0.3% of calls): fluent Chinese about an
# unrelated task ("請提供您需要修改的文本…", a second-hand gadget listing) that passed
# _check_chinese; both used 您, which these players never say.
_ON_TOPIC_RE = re.compile(r"\d\s*號|狼|預言家|女巫|村民|神職|好人|真人|AI|驗|查|投|票|刀|毒|救|藥|跳|夜|晚|局|發言|遺言|出局")


def clean_text(text: Any, limit: int) -> str:
    if text is None:
        return ""
    s = rules.one_line(str(text)).translate(_S2T)
    return s[:limit].strip()


def _check_chinese(*texts: str) -> None:
    """Reject derailed outputs: not Chinese (seen live: an English product blurb as `thought`)
    or Chinese about something else."""
    s = "".join(t for t in texts if t)
    letters = sum(1 for c in s if c.isalpha())
    if letters >= 20 and len(_CJK_RE.findall(s)) < 0.5 * letters:
        raise InvalidOutput("請全部用繁體中文回答，內容要跟這場狼人殺有關")
    if "您" in s or (len(s) >= 20 and not _ON_TOPIC_RE.search(s)):
        raise InvalidOutput("內容要跟這場狼人殺有關，請用繁體中文重新回答")


_LEAK_WORDS = "沒死|救|被刀|命大|幸運|死裡逃生|逃過"


def _hints_saved(text: str, saved: int) -> bool:
    """Whether `text` hints that seat `saved` was attacked last night (wolf-only knowledge)."""
    seat = rf"(?<!\d){saved}\s*號"
    return re.search(rf"{seat}.{{0,12}}(?:{_LEAK_WORDS})|(?:{_LEAK_WORDS}).{{0,12}}{seat}", text) is not None


def _saved_kill_leak(state: GameState, seat: int, kind: str, speech: str) -> Optional[int]:
    """For a wolf's public speech: the saved kill target it gives away, unless someone already
    talked about that seat being saved today (e.g. the witch claimed it)."""
    saved = prompts.saved_kill(state, seat) if kind != "wolf_chat" else None
    if saved is None or not _hints_saved(speech, saved):
        return None
    for e in events_visible_to(state, seat):
        if (e["kind"] in ("speech", "last_words") and e.get("day") == state.get("day")
                and e.get("seat") != seat and _hints_saved(e.get("text") or "", saved)):
            return None
    return saved


def clean_speech(text: Any, seat: Optional[int] = None, limit: int = config.MAX_SPEECH_CHARS) -> str:
    """Single-line speech without stray "3號：" / log-tag prefixes or wrapping quotes."""
    s = clean_text(text, 10_000)
    seat_re = _seat_prefix_re(seat)
    for _ in range(4):
        new = s
        for rx in (_TAG_RE, _LABEL_RE, seat_re):
            new = rx.sub("", new, count=1)
        new = new.strip()
        if new == s:
            break
        s = new
    if len(s) >= 2 and s[0] in _QUOTES and s[-1] == _QUOTES[s[0]] and s.count(s[0]) == 1:
        s = s[1:-1].strip()
    return s[:limit].strip()


def _as_dict(raw: Any) -> dict:
    if isinstance(raw, dict) and {"raw", "parsed", "parsing_error"} <= raw.keys():  # include_raw=True
        meta = getattr(raw["raw"], "response_metadata", None) or {}
        finish = str(meta.get("finish_reason") or "")
        if finish and not finish.endswith("STOP"):  # e.g. MAX_TOKENS: JSON was cut off mid-way
            raise InvalidOutput(f"回答被截斷了（{finish}），請簡短思考、直接給出完整答案")
        if raw["parsing_error"] is not None:
            raise InvalidOutput(f"回答不是正確的 JSON 格式：{str(raw['parsing_error'])[:200]}")
        raw = raw["parsed"]
    if raw is None:
        raise InvalidOutput("沒有回傳任何內容")
    if isinstance(raw, BaseModel):
        return raw.model_dump()
    if isinstance(raw, dict):
        return raw
    raise InvalidOutput(f"回傳格式錯誤：{type(raw).__name__}")


def _to_int(v: Any) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else None


class InvalidOutput(ValueError):
    """Model output parsed but broke a rule; message (zh-TW) is fed back on retry."""


# --------------------------------------------------------- fallback helpers
GENERIC_SPEECH = rules.PASS_SPEECHES  # the same phrases a human's 「過」 turns into
FALLBACK_THOUGHT = "（這次沒想好，隨便決定了）"


def _known_wolves_and_mates(state: GameState, seat: int) -> tuple[set[int], set[int], set[int]]:
    """(teammates if wolf, seer-known wolves, seer-known good) — own knowledge only."""
    try:
        k = prompts.private_knowledge(state, seat)
    except Exception:
        return set(), set(), set()
    wolves = {c["target"] for c in k["seer_checks"] if c["is_wolf"]}
    good = {c["target"] for c in k["seer_checks"] if not c["is_wolf"]}
    return set(k["teammates"]), wolves, good


def heuristic_target(rng: random.Random, state: GameState, seat: int, options: list[int], action: str) -> int:
    """A random but not-stupid legal choice from `options` (non-empty)."""
    opts = list(options)
    mates, wolves, good = _known_wolves_and_mates(state, seat)
    if action == "exile_vote":
        if wolves & set(opts):
            return rng.choice(sorted(wolves & set(opts)))
        pref = [o for o in opts if o not in mates and o not in good]
        if pref:
            return rng.choice(pref)
    elif action == "seer_check":
        checked = wolves | good
        pref = [o for o in opts if o not in checked]
        if pref:
            return rng.choice(pref)
    return rng.choice(opts)


def fallback_speak(rng: random.Random, kind: str) -> dict:
    return {
        "speech": rng.choice(GENERIC_SPEECH.get(kind, GENERIC_SPEECH["discussion"])),
        "thought": FALLBACK_THOUGHT,
    }


def fallback_target(rng: random.Random, state: GameState, seat: int, options: list[int], action: str) -> dict:
    if not options:  # should never happen; nothing legal to pick
        return {"target": None, "thought": FALLBACK_THOUGHT}
    try:
        target = heuristic_target(rng, state, seat, options, action)
    except Exception:
        target = rng.choice(list(options))
    return {"target": target, "thought": FALLBACK_THOUGHT}


def fallback_witch(killed: Optional[int], can_save: bool) -> dict:
    if can_save and killed is not None:
        return {"action": "save", "target": None, "thought": FALLBACK_THOUGHT}
    return {"action": "none", "target": None, "thought": FALLBACK_THOUGHT}


def fallback_exile(rng: random.Random, state: GameState, seat: int, options: list[int]) -> dict:
    if not options:
        return {"target": None, "thought": FALLBACK_THOUGHT}
    return fallback_target(rng, state, seat, options, "exile_vote")


# ------------------------------------------------------------------- Gemini


def _tokens(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else 0


@dataclass
class TokenUsage:
    """One game's Gemini usage over every attempt (retries and rejected answers included)."""

    input: int = 0
    output: int = 0  # includes reasoning
    reasoning: int = 0
    requests: int = 0  # attempts sent
    metered: int = 0  # attempts whose reply reported usage (not timeouts / API errors)
    skipped: int = 0  # calls answered by a fallback because the game was over budget
    budget: int = field(default=0, compare=False)  # this game's cap (0 = none)

    @property
    def total(self) -> int:
        return self.input + self.output

    def add(self, raw: Any) -> None:
        """Count the usage_metadata of one include_raw=True reply, if it has any."""
        msg = raw.get("raw") if isinstance(raw, dict) else None
        meta = getattr(msg, "usage_metadata", None)
        if not isinstance(meta, dict):
            return
        details = meta.get("output_token_details")
        self.input += _tokens(meta.get("input_tokens"))
        self.output += _tokens(meta.get("output_tokens"))
        self.reasoning += _tokens(details.get("reasoning")) if isinstance(details, dict) else 0
        self.metered += 1

    def as_dict(self) -> dict:
        return {"total": self.total, "input": self.input, "output": self.output, "reasoning": self.reasoning,
                "requests": self.requests, "metered": self.metered, "skipped": self.skipped,
                "budget": self.budget}

    def summary(self) -> str:
        cap = f" of {self.budget}" if self.budget > 0 else ""
        return (f"{self.total}{cap} tokens (input {self.input}, output {self.output} incl. reasoning "
                f"{self.reasoning}); {self.requests} requests, {self.requests - self.metered} without usage, "
                f"{self.skipped} calls skipped over budget")


class GeminiBackend:
    """Gemini via langchain-google-genai with structured output.

    Keeps per-game token totals (keyed by state["game_id"]; one instance serves every game) and
    stops calling the API for a game once it reaches `token_budget`."""

    def __init__(self, llm: Any = None, *, model: Optional[str] = None, seed: Optional[int] = None,
                 token_budget: Optional[int] = None):
        if llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            # No temperature: this model family ignores it (and warns).
            extra = {"thinking_level": THINKING_LEVEL} if THINKING_LEVEL else {}
            llm = ChatGoogleGenerativeAI(
                model=model or config.MODEL_NAME,
                google_api_key=config.get_api_key(),
                max_retries=2,
                timeout=attempt_timeout(),
                max_output_tokens=MAX_OUTPUT_TOKENS,
                **extra,
            )
        self.llm = llm
        self._rng = random.Random(seed)
        self._runnables: dict[type, Any] = {}
        self.token_budget = TOKEN_BUDGET if token_budget is None else token_budget
        self._usage: dict[str, TokenUsage] = {}  # game_id -> usage, oldest game first

    def _structured(self, schema: type[BaseModel]) -> Any:
        r = self._runnables.get(schema)
        if r is None:
            r = self._runnables[schema] = self.llm.with_structured_output(schema, include_raw=True)
        return r

    # -- token accounting -----------------------------------------------------
    def usage(self, game_id: Optional[str]) -> TokenUsage:
        """This game's running totals, started on first use."""
        key = game_id or ""
        u = self._usage.get(key)
        if u is None:
            u = self._usage[key] = TokenUsage(budget=self.token_budget)
            while len(self._usage) > MAX_TRACKED_GAMES:  # games that never ended
                old = next(iter(self._usage))
                log.info("game %s: forgetting the token usage of an unfinished game: %s",
                         old, self._usage.pop(old).summary())
        return u

    def over_budget(self, game_id: Optional[str]) -> bool:
        """Whether this game has used up its token budget: no more API calls for it, and the
        graph ends the game after the current step."""
        u = self._usage.get(game_id or "")
        return u is not None and u.budget > 0 and u.total >= u.budget

    def finish_game(self, game_id: Optional[str]) -> Optional[dict]:
        """Log the game's token usage and forget it (the graph calls this when the game ends).
        Returns the usage (saved in the game record), None if the game made no calls."""
        u = self._usage.pop(game_id or "", None)
        if u is None:
            return None
        log.info("game %s: token usage %s", game_id, u.summary())
        return u.as_dict()

    def _count(self, game_id: str, raw: Any) -> None:
        u = self.usage(game_id)
        was_over = self.over_budget(game_id)
        u.add(raw)
        if not was_over and self.over_budget(game_id):
            log.warning("game %s: reached its token budget (%s); no more Gemini calls, the game ends early",
                        game_id, u.summary())

    def _skip_over_budget(self, game_id: str, seat: int, what: str) -> bool:
        if not self.over_budget(game_id):
            return False
        self.usage(game_id).skipped += 1
        log.info("game %s seat %s %s: over the token budget, using fallback", game_id, seat, what)
        return True

    async def _ask(self, schema: type[BaseModel], state: GameState, seat: int, task: str, validate, what: str):
        """Call the model with retries; returns validate(output), or None after all attempts fail
        or once the game is over its token budget."""
        game_id = state.get("game_id") or ""
        if self._skip_over_budget(game_id, seat, what):
            return None
        messages = prompts.build_messages(state, seat, task)
        runnable = self._structured(schema)
        msgs = messages
        timeout = attempt_timeout()
        for attempt in range(1, config.LLM_MAX_ATTEMPTS + 1):
            if attempt > 1 and self._skip_over_budget(game_id, seat, what):
                return None
            self.usage(game_id).requests += 1
            try:
                raw = await asyncio.wait_for(runnable.ainvoke(msgs), timeout=timeout)
                self._count(game_id, raw)
                return validate(_as_dict(raw))
            except InvalidOutput as e:
                log.warning("seat %s %s: invalid output (attempt %d): %s", seat, what, attempt, e)
                msgs = messages + [HumanMessage(f"你上一次的回答不符合規則：{e}。請修正後重新回答。")]
            except asyncio.TimeoutError:
                log.warning("seat %s %s: timeout after %.0fs (attempt %d)", seat, what, timeout, attempt)
            except Exception as e:  # API / parse errors
                log.warning("seat %s %s: %s: %s (attempt %d)", seat, what, type(e).__name__, e, attempt)
                if attempt < config.LLM_MAX_ATTEMPTS:
                    await asyncio.sleep(min(RETRY_BACKOFF_S * attempt, 5.0))
        log.error("seat %s %s: all %d attempts failed, using fallback", seat, what, config.LLM_MAX_ATTEMPTS)
        return None

    # -- public API (never raises) ------------------------------------------
    async def speak(self, state: GameState, seat: int, kind: str) -> dict:
        try:
            def validate(d: dict) -> dict:
                speech = clean_speech(d.get("speech"), seat)
                if not speech:
                    raise InvalidOutput("speech 是空的，請寫出你要說的話")
                _check_chinese(speech, str(d.get("thought") or ""))
                leak = _saved_kill_leak(state, seat, kind, speech)
                if leak is not None:
                    raise InvalidOutput(
                        f"好人不知道昨晚刀的是 {leak}號；公開發言不要暗示他昨晚被刀、沒死、命大或被救，請重寫 speech"
                    )
                return {"speech": speech, "thought": clean_text(d.get("thought"), MAX_THOUGHT_CHARS)}

            out = await self._ask(SpeechOut, state, seat, prompts.task_speak(state, seat, kind), validate, f"speak/{kind}")
            return out or fallback_speak(self._rng, kind)
        except Exception:
            log.exception("speak failed for seat %s", seat)
            return fallback_speak(self._rng, kind)

    async def _choose(self, schema, state, seat, options, task_fn, action: str) -> dict:
        try:
            def validate(d: dict) -> dict:
                t = _to_int(d.get("target"))
                if t not in options:
                    raise InvalidOutput(f"target={d.get('target')!r} 不在可選名單 {options} 裡")
                _check_chinese(str(d.get("thought") or ""))
                return {"target": t, "thought": clean_text(d.get("thought"), MAX_THOUGHT_CHARS)}

            out = await self._ask(schema, state, seat, task_fn(state, seat, options), validate, action)
            return out or fallback_target(self._rng, state, seat, options, action)
        except Exception:
            log.exception("%s failed for seat %s", action, seat)
            return fallback_target(self._rng, state, seat, options, action)

    async def wolf_target(self, state: GameState, seat: int, options: list[int]) -> dict:
        return await self._choose(WolfKillOut, state, seat, list(options), prompts.task_wolf_target, "wolf_kill")

    async def seer_check(self, state: GameState, seat: int, options: list[int]) -> dict:
        return await self._choose(SeerCheckOut, state, seat, list(options), prompts.task_seer_check, "seer_check")

    async def witch(self, state: GameState, seat: int, killed: Optional[int], can_save: bool,
                    poison_options: list[int]) -> dict:
        poison_options = list(poison_options)
        can_save = bool(can_save and killed is not None and killed != seat)
        try:
            def validate(d: dict) -> dict:
                action = d.get("action")
                _check_chinese(str(d.get("thought") or ""))
                thought = clean_text(d.get("thought"), MAX_THOUGHT_CHARS)
                if action == "save":
                    if not can_save:
                        raise InvalidOutput("你現在不能用解藥（action 不能是 save）")
                    return {"action": "save", "target": None, "thought": thought}
                if action == "poison":
                    t = _to_int(d.get("target"))
                    if not poison_options:
                        raise InvalidOutput("你的毒藥已經用掉了（action 不能是 poison）")
                    if t not in poison_options:
                        raise InvalidOutput(f"毒藥 target={d.get('target')!r} 不在可選名單 {poison_options} 裡")
                    return {"action": "poison", "target": t, "thought": thought}
                if action == "none":
                    return {"action": "none", "target": None, "thought": thought}
                raise InvalidOutput(f"action={action!r} 必須是 save、poison 或 none")

            task = prompts.task_witch(state, seat, killed, can_save, poison_options)
            out = await self._ask(WitchOut, state, seat, task, validate, "witch")
            return out or fallback_witch(killed, can_save)
        except Exception:
            log.exception("witch failed for seat %s", seat)
            return fallback_witch(killed, can_save)

    async def exile_vote(self, state: GameState, seat: int, options: list[int]) -> dict:
        options = list(options)
        try:
            def validate(d: dict) -> dict:
                raw = d.get("target")
                t = _to_int(raw)
                if raw is not None and t not in options:
                    raise InvalidOutput(f"target={raw!r} 不在可選名單 {options} 裡（棄票請填 null）")
                _check_chinese(str(d.get("thought") or ""))
                return {"target": t, "thought": clean_text(d.get("thought"), MAX_THOUGHT_CHARS)}

            out = await self._ask(ExileOut, state, seat, prompts.task_exile_vote(state, seat, options), validate, "exile_vote")
            return out or fallback_exile(self._rng, state, seat, options)
        except Exception:
            log.exception("exile_vote failed for seat %s", seat)
            return fallback_exile(self._rng, state, seat, options)


# --------------------------------------------------------------------- Fake

_FLAVOR = {
    "analyst": ("首先，", "所以我先這樣判斷。"),
    "leader": ("聽我的，", "今天就這樣決定。"),
    "skeptic": ("你們不覺得怪嗎？", "是不是太刻意了？"),
    "chill": ("欸", "好啦隨便啦"),
}
_FAKE_DISCUSS = [
    "我覺得{x}號剛剛的發言有點怪，先記一下。",
    "目前沒什麼資訊，我比較想聽{x}號怎麼說。",
    "{x}號的邏輯我不太認同，今天可以考慮出他。",
    "我是好人啦，{x}號你可以解釋一下嗎？",
]
_FAKE_LAST = ["我是好人，{x}號很可疑，大家注意一下。", "好人加油，我覺得{x}號有問題。"]
_FAKE_WOLF = ["今晚刀{x}號吧，他看起來像神。", "我想刀{x}號，他發言太穩了，留著是威脅。", "{x}號帶節奏很強，先處理掉。"]


class FakeBackend:
    """Offline backend: seeded-random legal choices + short template speeches."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def _flavor(self, state: GameState, seat: int) -> tuple[str, str]:
        try:
            return _FLAVOR.get(get_player(state, seat).get("persona", ""), ("", ""))
        except Exception:
            return ("", "")

    def _others(self, state: GameState, seat: int, alive_only: bool = True) -> list[int]:
        return [p["seat"] for p in state.get("players") or [] if p["seat"] != seat and (p["alive"] or not alive_only)]

    async def speak(self, state: GameState, seat: int, kind: str) -> dict:
        try:
            others = self._others(state, seat) or self._others(state, seat, alive_only=False)
            mates, wolves, good = _known_wolves_and_mates(state, seat)
            x = self.rng.choice([o for o in others if o not in mates] or others) if others else seat
            pre, post = self._flavor(state, seat)
            if kind == "wolf_chat":
                body = self.rng.choice(_FAKE_WOLF).format(x=x)
            elif kind == "last_words":
                body = self.rng.choice(_FAKE_LAST).format(x=x)
            elif wolves:  # seer reports a found wolf
                body = f"我是預言家，查到{min(wolves)}號是狼人，今天出他。"
            else:
                body = self.rng.choice(_FAKE_DISCUSS).format(x=x)
            speech = f"{pre}{body}" + (post if self.rng.random() < 0.5 else "")
            return {"speech": clean_speech(speech, seat), "thought": f"（測試用）隨便說說，懷疑{x}號。"}
        except Exception:
            log.exception("FakeBackend.speak failed")
            return fallback_speak(self.rng, kind)

    async def wolf_target(self, state: GameState, seat: int, options: list[int]) -> dict:
        return {**fallback_target(self.rng, state, seat, list(options), "wolf_kill"), "thought": "（測試用）隨機挑一個刀。"}

    async def seer_check(self, state: GameState, seat: int, options: list[int]) -> dict:
        return {**fallback_target(self.rng, state, seat, list(options), "seer_check"), "thought": "（測試用）驗一個還沒驗過的人。"}

    async def witch(self, state: GameState, seat: int, killed: Optional[int], can_save: bool,
                    poison_options: list[int]) -> dict:
        r = self.rng.random()
        if can_save and killed is not None and r < 0.6:
            return {"action": "save", "target": None, "thought": f"（測試用）救 {killed}號。"}
        if poison_options and r > 0.85:
            t = self.rng.choice(list(poison_options))
            return {"action": "poison", "target": t, "thought": f"（測試用）毒 {t}號。"}
        return {"action": "none", "target": None, "thought": "（測試用）今晚不用藥。"}

    async def exile_vote(self, state: GameState, seat: int, options: list[int]) -> dict:
        options = list(options)
        if not options or self.rng.random() < 0.1:
            return {"target": None, "thought": "（測試用）棄票。"}
        return {**fallback_exile(self.rng, state, seat, options), "thought": "（測試用）隨便投一個。"}


def get_backend() -> AIBackend:
    if config.FAKE_LLM:
        return FakeBackend()
    if not config.get_api_key():
        log.error("No Gemini API key (.env gemini_key); falling back to FakeBackend")
        return FakeBackend()
    return GeminiBackend()
