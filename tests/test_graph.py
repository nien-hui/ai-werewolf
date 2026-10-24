"""End-to-end graph tests with an in-file stub backend (no werewolf.ai, no LLM)."""
import asyncio
import random
import re
from collections import Counter
from dataclasses import dataclass, field

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from werewolf import config, rules
from werewolf.graph import BUDGET_REASON, build_graph
from werewolf.state import GOD_ROLES, ROLE_ZH, get_player

PASS = object()  # expectation: an empty speech became one of rules.PASS_SPEECHES[kind]
ROLES = [None, "werewolf", "seer", "witch", "villager"]
SPEECH_ACTION = {"discussion": "speech", "last_words": "last_words", "wolf_chat": "wolf_chat"}
STAGE_KIND = {"wolf_chat": "wolf_chat", "night_last_words": "last_words",
              "discussion": "discussion", "exile_last_words": "last_words"}
PHRASES = ["我是村民啦", "我覺得 3號 怪怪的", "先聽聽大家", "嗯…我沒什麼資訊", "今天出 2號 吧"]
CAUSE_WORDS = ("毒", "狼", "殺", "女巫", "救", "刀")
STATUS_RE = re.compile(r"^(\d+號 正在發言…|\d+號 正在發表遺言…|夜晚行動中…|投票中…|)$")  # "" clears

VISIBILITY = {
    "system": "all", "death": "all", "speech": "all", "last_words": "all", "vote_result": "all",
    "game_over": "all", "reveal": "all",
    "wolf_chat": "wolves", "wolf_pick": "wolves", "wolf_kill": "wolves",
    "role_assign": "self", "witch_info": "self", "witch_action": "self", "seer_result": "self",
    "thought": "god",
}
NO_SEAT_KINDS = {"system", "death", "vote_result", "game_over", "reveal", "wolf_kill"}
PAYLOAD_KEYS = {
    "speak": {"type", "kind", "day", "prompt"},
    "choose": {"type", "action", "day", "prompt", "options", "allow_skip"},
    "witch": {"type", "day", "prompt", "killed", "can_save", "poison_options"},
}
DATA_KEYS = {
    "system": set(), "speech": set(), "last_words": set(), "wolf_chat": set(),
    "role_assign": {"role", "teammates"}, "wolf_pick": {"target"}, "wolf_kill": {"target"},
    "witch_info": {"killed"}, "witch_action": {"action", "target"}, "seer_result": {"target", "is_wolf"},
    "death": {"seats"}, "vote_result": {"votes", "tally", "exiled"}, "game_over": {"winner", "reason"},
    "reveal": {"roles"},
    "thought": {"action"},
}


# ------------------------------------------------------------ stub backend
class StubBackend:
    """Random but legal AI. Records every call and any contract violation it sees."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.calls = Counter()  # (seat, thought action) -> n
        self.ncalls = 0
        self.problems = []

    def _record(self, state, seat, action, alive=True):
        self.ncalls += 1
        self.calls[(seat, action)] += 1
        if seat == state["human_seat"]:
            self.problems.append(f"backend called for the human seat {seat} ({action})")
        if get_player(state, seat)["alive"] != alive:
            self.problems.append(f"{action} by seat {seat} with alive={not alive}")

    def _expect(self, cond, msg):
        if not cond:
            self.problems.append(msg)

    def _thought(self, seat):
        return f"{seat}號 的內心想法 #{self.ncalls}"

    async def speak(self, state, seat, kind):
        self._record(state, seat, SPEECH_ACTION[kind], alive=kind != "last_words")
        if kind == "wolf_chat":
            self._expect(get_player(state, seat)["role"] == "werewolf", "non-wolf in wolf chat")
        await asyncio.sleep(0)
        return {"speech": self.rng.choice(PHRASES), "thought": self._thought(seat)}

    async def wolf_target(self, state, seat, options):
        self._record(state, seat, "wolf_kill")
        self._expect(get_player(state, seat)["role"] == "werewolf", "non-wolf picks a kill")
        self._expect(options == rules.wolf_kill_options(state["players"]), f"bad wolf options {options}")
        await asyncio.sleep(0)
        return {"target": self.rng.choice(options), "thought": self._thought(seat)}

    async def witch(self, state, seat, killed, can_save, poison_options):
        self._record(state, seat, "witch")
        self._expect(get_player(state, seat)["role"] == "witch", "non-witch acts as witch")
        exp_killed = state["night"]["wolf_target"] if state["antidote"] else None
        self._expect(killed == exp_killed, f"witch killed {killed} != {exp_killed}")
        self._expect(can_save == (state["antidote"] and killed is not None and killed != seat), "bad can_save")
        exp_poison = rules.alive_others(state["players"], seat) if state["poison"] else []
        self._expect(poison_options == exp_poison, f"bad poison options {poison_options}")
        self._expect(can_save or poison_options, "witch asked with nothing to do")
        await asyncio.sleep(0)
        legal = [("none", None)] + ([("save", None)] if can_save else []) + [("poison", t) for t in poison_options]
        action, target = self.rng.choice(legal)
        return {"action": action, "target": target, "thought": self._thought(seat)}

    async def seer_check(self, state, seat, options):
        self._record(state, seat, "seer_check")
        self._expect(get_player(state, seat)["role"] == "seer", "non-seer checks")
        self._expect(options == rules.alive_others(state["players"], seat), f"bad seer options {options}")
        await asyncio.sleep(0)
        return {"target": self.rng.choice(options), "thought": self._thought(seat)}

    async def exile_vote(self, state, seat, options):
        self._record(state, seat, "exile_vote")
        self._expect(options == rules.alive_others(state["players"], seat), f"bad exile options {options}")
        await asyncio.sleep(0)
        r = self.rng.random()  # bias towards a common target so exiles actually happen
        target = options[0] if r < 0.5 else (None if r > 0.85 else self.rng.choice(options))
        return {"target": target, "thought": self._thought(seat)}


class BrokenBackend(StubBackend):
    """Violates the backend contract (raises / garbage / illegal values); the graph must cope."""

    async def _maybe_break(self, good):
        r = self.rng.random()
        if r < 0.15:
            raise RuntimeError("boom")
        if r < 0.25:
            return None
        if r < 0.45:
            return {"target": 99, "action": "both", "speech": "", "thought": None}
        return good

    async def speak(self, state, seat, kind):
        return await self._maybe_break(await super().speak(state, seat, kind))

    async def wolf_target(self, state, seat, options):
        good = await super().wolf_target(state, seat, options)
        if self.rng.random() < 0.3:  # try 自刀
            good["target"] = seat
        return await self._maybe_break(good)

    async def witch(self, state, seat, killed, can_save, poison_options):
        good = await super().witch(state, seat, killed, can_save, poison_options)
        if self.rng.random() < 0.3:
            good = {"action": "save" if not can_save else "poison", "target": seat, "thought": "x"}
        return await self._maybe_break(good)

    async def seer_check(self, state, seat, options):
        return await self._maybe_break(await super().seer_check(state, seat, options))

    async def exile_vote(self, state, seat, options):
        good = await super().exile_vote(state, seat, options)
        if self.rng.random() < 0.2:
            good["target"] = seat  # self vote -> abstain
        return await self._maybe_break(good)


# ------------------------------------------------------------------ driver
@dataclass
class Prompt:
    payload: dict
    expect: tuple
    n_events: int  # len(events) when interrupted = id of the resumed node's first event


@dataclass
class Game:
    values: dict
    events: list
    streamed: list
    statuses: list
    prompts: list = field(default_factory=list)


def check_payload(payload, v):
    """Assert the interrupt payload matches the protocol for the current state `v`."""
    human = v["human_seat"]
    me = get_player(v, human)
    assert set(payload) == PAYLOAD_KEYS[payload["type"]], payload
    assert payload["day"] == v["day"]
    assert isinstance(payload["prompt"], str) and payload["prompt"]
    players = v["players"]
    t = payload["type"]
    if t == "speak":
        assert payload["kind"] == STAGE_KIND[v["stage"]]
        assert v["speech_queue"][0] == human
        if payload["kind"] == "last_words":
            assert not me["alive"]
            assert v["stage"] == "exile_last_words" or (v["day"] == 1 and human in v["last_night_deaths"])
        else:
            assert me["alive"]
            if payload["kind"] == "wolf_chat":
                assert me["role"] == "werewolf" and v["phase"] == "night"
    elif t == "choose":
        action = payload["action"]
        assert payload["allow_skip"] is (action == "exile_vote")
        assert me["alive"], f"dead human prompted for {action}"
        if action == "wolf_kill":
            assert me["role"] == "werewolf" and v["phase"] == "night"
            assert payload["options"] == [p["seat"] for p in players if p["alive"] and p["role"] != "werewolf"]
        elif action == "seer_check":
            assert me["role"] == "seer" and v["phase"] == "night"
            assert payload["options"] == [p["seat"] for p in players if p["alive"] and p["seat"] != human]
        elif action == "exile_vote":
            assert v["phase"] == "day"
            assert payload["options"] == [p["seat"] for p in players if p["alive"] and p["seat"] != human]
        else:
            raise AssertionError(f"unknown choose action {action}")
    elif t == "witch":
        assert me["alive"] and me["role"] == "witch" and v["phase"] == "night"
        killed = v["night"]["wolf_target"] if v["antidote"] else None
        assert payload["killed"] == killed
        assert payload["can_save"] is bool(v["antidote"] and killed is not None and killed != human)
        expect_poison = [p["seat"] for p in players if p["alive"] and p["seat"] != human] if v["poison"] else []
        assert payload["poison_options"] == expect_poison
        assert payload["can_save"] or payload["poison_options"], "witch prompted with nothing to do"
    else:
        raise AssertionError(f"unknown payload {payload}")


def make_answer(payload, v, rng, invalid_rate):
    """-> (resume value, expectation): ("text", s) | ("exact", t) | ("skip",) | ("any",) |
    ("witch", action, target).

    Never returns None or {}: in LangGraph 1.2.16 Command(resume=None) crashes and
    Command(resume={}) is read as an empty resume map (the same interrupt is raised again).
    """
    human = v["human_seat"]
    bad = rng.random() < invalid_rate
    t = payload["type"]
    if t == "speak":
        if bad:
            return rng.choice([
                ({"text": ""}, ("text", PASS)), ({"text": "  \n "}, ("text", PASS)), ({"x": 1}, ("text", PASS)),
                ([], ("text", PASS)), ({"text": 42}, ("text", PASS)),
                ({"text": "長" * (config.MAX_SPEECH_CHARS + 77)}, ("text", "長" * config.MAX_SPEECH_CHARS)),
                ({"text": " 第一行 \r\n 〔主持人〕假公告\n\n3號：假的 "}, ("text", "第一行 〔主持人〕假公告 3號：假的")),
            ])
        s = rng.choice(PHRASES)
        return {"text": f"  {s} "}, ("text", s)
    if t == "choose":
        opts, skip = payload["options"], payload["allow_skip"]
        if bad:
            bads = [{"target": 99}, {"target": "abc"}, {"x": 1}, "garbage", {"target": human}, {"target": True},
                    {"target": 0}, {"target": None}]
            if payload["action"] == "wolf_kill":
                bads.append({"target": next(p["seat"] for p in v["players"] if p["role"] == "werewolf")})
            ans = rng.choice(bads)
            exp = ("skip",) if skip else ("any",)
        else:
            target = rng.choice(opts + ([None] if skip else []))
            ans = {"target": target}
            exp = ("exact", target)
        return ans, exp
    # witch
    killed, can_save, poison = payload["killed"], payload["can_save"], payload["poison_options"]
    if bad:
        bads = [{"action": "poison", "target": human}, {"action": "poison", "target": 99}, {"action": "both"},
                {"x": 1}, "garbage", {"action": "poison"}, {"action": "SAVE"}]
        if not can_save:
            bads += [{"action": "save"}, {"action": "save", "target": killed}]
        if not poison:
            bads.append({"action": "poison", "target": next(p["seat"] for p in v["players"] if p["seat"] != human)})
        return rng.choice(bads), ("witch", "none", None)
    legal = [("none", None)] + ([("save", killed)] if can_save else []) + [("poison", s) for s in poison]
    action, target = rng.choice(legal)
    ans = {"action": action} | ({"target": target} if action == "poison" else {})
    return ans, ("witch", action, target)


async def play(seed, human_role=None, backend=None, invalid_rate=0.25, checkpointer=None, all_ai=False):
    backend = backend if backend is not None else StubBackend(seed)
    graph = build_graph(backend=backend, checkpointer=checkpointer)
    game_id = f"g-{seed}-{human_role}"
    cfg = {"configurable": {"thread_id": game_id}}
    rng = random.Random(seed * 7919 + 13)
    inp = {"game_id": game_id, "seed": seed, "events": []}
    if human_role is not None:
        inp["human_role"] = human_role
    if all_ai:
        inp["all_ai"] = True
    streamed, statuses, prompts = [], [], []
    for _ in range(500):
        async for mode, chunk in graph.astream(inp, cfg, stream_mode=["updates", "custom"]):
            if mode == "custom":
                statuses.append(chunk)
                continue
            for node, update in chunk.items():
                if node != "__interrupt__":
                    assert isinstance(update, dict), f"{node} streamed {update!r}"  # server does .get()
                    streamed.extend(update.get("events", []))
        snap = await graph.aget_state(cfg)
        if not snap.interrupts:
            assert not snap.next
            break
        assert len(snap.interrupts) == 1
        payload = snap.interrupts[0].value
        check_payload(payload, snap.values)
        answer, expect = make_answer(payload, snap.values, rng, invalid_rate)
        prompts.append(Prompt(payload, expect, len(snap.values["events"])))
        inp = Command(resume=answer)
    else:
        raise AssertionError("game did not finish")
    return Game(snap.values, snap.values["events"], streamed, statuses, prompts)


# ------------------------------------------------------------ verification
def next_non_thought(events, i):
    j = i + 1
    while events[j]["kind"] == "thought":
        j += 1
    return events[j]


def check_events_shape(g):
    v, events = g.values, g.events
    role = {p["seat"]: p["role"] for p in v["players"]}
    assert [e["id"] for e in events] == list(range(len(events)))
    assert g.streamed == events, "streamed node updates must equal the final event log"
    for e in events:
        assert set(e) == {"id", "day", "phase", "kind", "seat", "text", "visibility", "data"}
        assert e["kind"] in VISIBILITY, e
        assert set(e["data"]) == DATA_KEYS[e["kind"]], e
        assert isinstance(e["text"], str) and e["text"], e
        if e["kind"] in NO_SEAT_KINDS:
            assert e["seat"] is None, e
        else:
            assert e["seat"] in role, e
        vis = VISIBILITY[e["kind"]]
        assert e["visibility"] == (f"seat:{e['seat']}" if vis == "self" else vis), e
        assert e["phase"] in ("setup", "night", "day", "end")
    # setup
    assigns = [e for e in events if e["kind"] == "role_assign"]
    assert [e["seat"] for e in assigns] == sorted(role) and all(e["day"] == 0 for e in assigns)
    for e in assigns:
        mates = [s for s, r in role.items() if r == "werewolf" and s != e["seat"]] if role[e["seat"]] == "werewolf" else []
        assert e["data"] == {"role": role[e["seat"]], "teammates": mates}
        assert e["text"].startswith(f"你是 {e['seat']}號，你的身份是：{ROLE_ZH[role[e['seat']]]}。")
        if mates:
            assert f"你的狼隊友是 {mates[0]}號" in e["text"]
    # ending
    kinds = [e["kind"] for e in events]
    assert kinds.count("game_over") == 1 and kinds.count("reveal") == 1
    assert kinds[-2:] == ["game_over", "reveal"]
    assert events[-1]["data"]["roles"] == [
        {"seat": p["seat"], "role": p["role"], "is_human": p["is_human"], "persona": p["persona"]}
        for p in v["players"]
    ]


def check_thoughts(g, backend):
    events, human = g.events, g.values["human_seat"]
    thoughts = [e for e in events if e["kind"] == "thought"]
    assert len(thoughts) == backend.ncalls, "every backend call -> exactly one thought (no re-execution)"
    assert Counter((e["seat"], e["data"]["action"]) for e in thoughts) == backend.calls
    for i, e in enumerate(events):
        if e["kind"] == "thought":
            assert e["seat"] != human
            action = e["data"]["action"]
            if action == "exile_vote":
                nxt = next_non_thought(events, i)
                assert nxt["kind"] == "vote_result"
                assert e["seat"] in [vv["voter"] for vv in nxt["data"]["votes"]]
            else:
                expect_kind = {"speech": "speech", "last_words": "last_words", "wolf_chat": "wolf_chat",
                               "wolf_kill": "wolf_pick", "witch": "witch_action", "seer_check": "seer_result"}[action]
                nxt = events[i + 1]
                assert (nxt["kind"], nxt["seat"]) == (expect_kind, e["seat"]), (e, nxt)
        elif e["kind"] in ("speech", "last_words", "wolf_chat", "wolf_pick", "witch_action", "seer_result"):
            prev_is_thought = i > 0 and events[i - 1]["kind"] == "thought" and events[i - 1]["seat"] == e["seat"]
            assert prev_is_thought == (e["seat"] != human), e


def check_rules_replay(g):
    """Replay the event log and check every rule of the game."""
    v, events = g.values, g.events
    human = v["human_seat"]
    role = {p["seat"]: p["role"] for p in v["players"]}
    wolves = {s for s, r in role.items() if r == "werewolf"}
    witch = next(s for s, r in role.items() if r == "witch")
    seer = next(s for s, r in role.items() if r == "seer")
    alive = set(role)
    antidote = poison = True
    night = None
    last_deaths = []
    may_last_words = set()
    winner_seen = None
    order = None
    spoken = []

    def replay_players():
        return [{"seat": s, "role": r, "alive": s in alive, "is_human": s == human, "persona": ""}
                for s, r in sorted(role.items())]

    for i, e in enumerate(events):
        k, d = e["kind"], e["data"]
        if k == "system" and e["text"].startswith(f"第{e['day']}夜"):
            assert winner_seen is None
            assert e["phase"] == "night"
            night = {"picks": {}, "target": None, "saved": False, "poisoned": None, "witch": 0,
                     "alive_wolves": sorted(alive & wolves)}
            if len(night["alive_wolves"]) >= 2:
                chat = [x["seat"] for x in events[i + 1:] if x["kind"] == "wolf_chat" and x["day"] == e["day"]]
                assert sorted(chat) == night["alive_wolves"]
            else:
                assert not any(x["kind"] == "wolf_chat" and x["day"] == e["day"] for x in events)
        elif k == "wolf_chat":
            assert e["seat"] in alive and e["seat"] in wolves and e["phase"] == "night"
        elif k == "wolf_pick":
            assert e["seat"] in alive & wolves
            assert d["target"] in alive and d["target"] not in wolves, "wolves never target wolves"
            night["picks"][e["seat"]] = d["target"]
        elif k == "wolf_kill":
            assert d["target"] in alive and d["target"] not in wolves
            assert sorted(night["picks"]) == night["alive_wolves"]
            ai_wolves = [w for w in night["alive_wolves"] if w != human]
            if human in night["alive_wolves"]:
                assert d["target"] == night["picks"][human]
            else:  # the most picked target; a tie goes to the smallest-seat wolf's pick
                counts = Counter(night["picks"].values())
                top = max(counts.values())
                assert counts[d["target"]] == top
                assert d["target"] == next(night["picks"][w] for w in ai_wolves if counts[night["picks"][w]] == top)
            night["target"] = d["target"]
        elif k == "witch_info":
            assert e["seat"] == witch and witch in alive
            assert d["killed"] == (night["target"] if antidote else None)
        elif k == "witch_action":
            assert e["seat"] == witch and witch in alive and events[i - 1]["kind"] in ("witch_info", "thought")
            night["witch"] += 1
            assert night["witch"] == 1, "at most one potion per night"
            if d["action"] == "save":
                assert antidote and d["target"] == night["target"] and d["target"] != witch, "no self-save"
                antidote, night["saved"] = False, True
            elif d["action"] == "poison":
                assert poison and d["target"] in alive and d["target"] != witch
                poison, night["poisoned"] = False, d["target"]
            else:
                assert d == {"action": "none", "target": None}
        elif k == "seer_result":
            assert e["seat"] == seer and seer in alive
            assert d["target"] in alive and d["target"] != seer
            assert d["is_wolf"] == (role[d["target"]] == "werewolf")
            assert d["target"] in [c["target"] for c in v["seer_checks"]]
        elif k == "death":
            dead = sorted(({night["target"]} - ({night["target"]} if night["saved"] else set()))
                          | ({night["poisoned"]} if night["poisoned"] else set()))
            assert d["seats"] == dead
            assert not any(w in e["text"] for w in CAUSE_WORDS), e["text"]
            assert e["phase"] == "day"
            alive -= set(dead)
            last_deaths = dead
            winner_seen = rules.check_winner(replay_players())[0]
            # 遺言 only after night 1, and not when the game is already decided
            may_last_words = set(dead) if e["day"] == 1 and not winner_seen else set()
            if winner_seen:
                assert next_non_thought(events, i)["kind"] == "game_over"
        elif k == "last_words":
            assert e["seat"] in may_last_words and e["seat"] not in alive
            may_last_words.discard(e["seat"])
        elif k == "system" and e["text"].startswith(f"第{e['day']}天"):
            assert winner_seen is None and not may_last_words, "all night last words come first"
            order = rules.discussion_order(sorted(alive), last_deaths, e["day"], len(role))
            spoken = []
        elif k == "speech":
            assert e["seat"] in alive and e["phase"] == "day"
            spoken.append(e["seat"])
        elif k == "vote_result":
            assert spoken == order, "discussion follows seat order"
            voters = [x["voter"] for x in d["votes"]]
            assert voters == sorted(alive)
            for x in d["votes"]:
                assert set(x) == {"voter", "target"}
                assert x["target"] is None or (x["target"] in alive and x["target"] != x["voter"])
            tally, top, exiled = rules.tally_votes(d["votes"])
            assert (d["tally"], d["exiled"]) == (tally, exiled)
            if exiled is not None:
                alive.discard(exiled)
                may_last_words = {exiled}
                assert next_non_thought(events, i)["kind"] == "last_words"
                assert next_non_thought(events, i)["seat"] == exiled
            else:
                may_last_words = set()
            winner_seen = rules.check_winner(replay_players())[0]
        elif k == "game_over":
            assert not may_last_words
            w = d["winner"]
            assert w == v["winner"] and d["reason"] == v["win_reason"]
            if w == "draw":
                assert e["day"] >= config.MAX_DAYS and rules.check_winner(replay_players())[0] is None
            else:
                assert rules.check_winner(replay_players()) == (w, d["reason"])
                assert rules.check_winner(v["players"])[0] == w
    # end state
    assert {p["seat"] for p in v["players"] if p["alive"]} == alive
    assert v["phase"] == "end" and v["antidote"] == antidote and v["poison"] == poison
    assert v["day"] <= config.MAX_DAYS


def check_human_prompts(g):
    """Every human action came through exactly one interrupt, with coerced resume values."""
    events, human = g.events, g.values["human_seat"]
    for p in g.prompts:
        pl, exp, n = p.payload, p.expect, p.n_events
        later = events[n:]
        if pl["type"] == "speak":
            e = events[n]
            assert (e["kind"], e["seat"]) == ({"discussion": "speech"}.get(pl["kind"], pl["kind"]), human)
            if exp[1] is PASS:
                assert e["text"] in rules.PASS_SPEECHES[pl["kind"]]
            else:
                assert e["text"] == exp[1]
        elif pl["type"] == "witch":
            assert events[n]["kind"] == "witch_info" and events[n]["data"]["killed"] == pl["killed"]
            e = events[n + 1]
            assert e["kind"] == "witch_action" and e["seat"] == human
            assert (e["data"]["action"], e["data"]["target"]) == exp[1:]
        else:
            action = pl["action"]
            if action == "wolf_kill":
                e = next(x for x in later if x["kind"] == "wolf_pick" and x["seat"] == human)
                got = e["data"]["target"]
                assert next(x for x in later if x["kind"] == "wolf_kill")["data"]["target"] == got
            elif action == "seer_check":
                e = events[n]
                assert (e["kind"], e["seat"]) == ("seer_result", human)
                got = e["data"]["target"]
            elif action == "exile_vote":
                e = next(x for x in later if x["kind"] == "vote_result")
                got = next(x["target"] for x in e["data"]["votes"] if x["voter"] == human)
            else:
                raise AssertionError(f"unexpected prompt {pl}")
            if exp[0] == "exact":
                assert got == exp[1]
            elif exp[0] == "skip":
                assert got is None
            else:
                assert got in pl["options"]
    # counts: each human-made event <-> one prompt
    n_prompts = Counter((p.payload["type"], p.payload.get("kind") or p.payload.get("action")) for p in g.prompts)
    mine = Counter()
    for e in events:
        if e["seat"] == human and e["kind"] in ("speech", "last_words", "wolf_chat"):
            mine[("speak", {"speech": "discussion"}.get(e["kind"], e["kind"]))] += 1
        elif e["seat"] == human and e["kind"] == "wolf_pick":
            mine[("choose", "wolf_kill")] += 1
        elif e["seat"] == human and e["kind"] == "seer_result":
            mine[("choose", "seer_check")] += 1
        elif e["seat"] == human and e["kind"] == "witch_action":
            mine[("witch", None)] += 1
        elif e["kind"] == "vote_result" and human in [x["voter"] for x in e["data"]["votes"]]:
            mine[("choose", "exile_vote")] += 1
    assert n_prompts == mine


def check_statuses(g):
    texts = [s["text"] for s in g.statuses]
    assert all(s.get("type") == "status" and set(s) == {"type", "text"} for s in g.statuses)
    assert all(STATUS_RE.match(t) for t in texts), texts
    human = g.values["human_seat"]
    ai_speech = sum(1 for e in g.events if e["kind"] == "speech" and e["seat"] != human)
    ai_last = sum(1 for e in g.events if e["kind"] == "last_words" and e["seat"] != human)
    assert sum(t.endswith("正在發言…") for t in texts) == ai_speech
    assert sum(t.endswith("正在發表遺言…") for t in texts) == ai_last
    assert texts.count("") == ai_speech + ai_last  # cleared once each AI day speech is out
    # night: one status per night plus one after each of the human's own night actions, so the
    # count can't tell the human how many wolves / gods are still acting
    nights = sum(1 for e in g.events if e["kind"] == "system" and e["text"].endswith("天黑請閉眼。"))
    own = sum(1 for p in g.prompts if p.payload["type"] == "witch" or p.payload.get("kind") == "wolf_chat"
              or p.payload.get("action") in ("wolf_kill", "seer_check"))
    assert texts.count("夜晚行動中…") == nights + own


def check_game(g, backend):
    assert backend.problems == [], backend.problems[:5]
    check_events_shape(g)
    check_thoughts(g, backend)
    check_rules_replay(g)
    check_human_prompts(g)
    check_statuses(g)


# ------------------------------------------------------------------ tests
@pytest.mark.parametrize("human_role", ROLES)
@pytest.mark.parametrize("seed", range(48))
async def test_full_games(seed, human_role):
    backend = StubBackend(seed)
    g = await play(seed, human_role, backend)
    check_game(g, backend)
    if human_role:
        assert get_player(g.values, g.values["human_seat"])["role"] == human_role


@pytest.mark.parametrize("seed", range(12))
async def test_all_ai_games(seed):
    """Self-play: no human seat, so the game runs to the end without one interrupt or pacing wait."""
    backend = StubBackend(seed)
    g = await play(seed, backend=backend, all_ai=True)
    assert g.prompts == [] and g.values["human_seat"] is None
    assert not any(p["is_human"] for p in g.values["players"]) and all(p["persona"] for p in g.values["players"])
    check_events_shape(g)
    check_thoughts(g, backend)
    check_rules_replay(g)
    assert g.values["winner"] in ("wolf", "good", "draw") and g.events[-1]["kind"] == "reveal"
    assert g.values["token_usage"] is None  # the stub doesn't count tokens
    assert g.values["token_usage"] is None  # the stub doesn't count tokens


async def test_games_cover_interesting_cases():
    """Across many games every branch actually happens (guards against vacuous tests)."""
    seen = Counter()
    for seed in range(120):
        role = ROLES[seed % len(ROLES)]
        backend = StubBackend(seed)
        g = await play(seed, role, backend, invalid_rate=0.3)
        check_game(g, backend)
        ev = g.events
        human = g.values["human_seat"]
        seen[g.values["winner"]] += 1
        seen.update(f"prompt:{p.payload['type']}:{p.payload.get('kind') or p.payload.get('action')}"
                    for p in g.prompts)
        seen.update(f"witch:{e['data']['action']}" for e in ev if e["kind"] == "witch_action")
        seen.update("night_last_words" for e in ev if e["kind"] == "last_words" and e["phase"] == "day"
                    and e["day"] == 1 and not any(x["kind"] == "vote_result" and x["day"] == 1
                                                  for x in ev[: e["id"]]))
        seen.update("tie" for e in ev if e["kind"] == "vote_result" and e["data"]["exiled"] is None
                    and e["data"]["tally"])
        seen.update("peaceful" for e in ev if e["kind"] == "death" and not e["data"]["seats"])
        seen.update("double_death" for e in ev if e["kind"] == "death" and len(e["data"]["seats"]) == 2)
        for day in {e["day"] for e in ev if e["kind"] == "wolf_chat"}:
            chat = [e["seat"] for e in ev if e["kind"] == "wolf_chat" and e["day"] == day]
            seen["wolf_chat_low_first" if chat[0] < chat[1] else "wolf_chat_high_first"] += 1
        if not get_player(g.values, human)["alive"]:
            seen["human_dead"] += 1
    for key in ["good", "wolf", "prompt:speak:discussion", "prompt:speak:last_words",
                "prompt:speak:wolf_chat", "prompt:choose:wolf_kill", "prompt:choose:seer_check",
                "prompt:choose:exile_vote", "prompt:witch:None",
                "witch:save", "witch:poison", "witch:none", "night_last_words", "tie", "peaceful",
                "double_death", "human_dead", "wolf_chat_low_first", "wolf_chat_high_first"]:
        assert seen[key] > 0, f"never saw {key}: {seen}"


@pytest.mark.parametrize("max_days", [1, 2])
async def test_draw_at_max_days(monkeypatch, max_days):
    monkeypatch.setattr(config, "MAX_DAYS", max_days)
    draws = 0
    for seed in range(40):
        backend = StubBackend(seed)
        g = await play(seed, ROLES[seed % len(ROLES)], backend)
        check_game(g, backend)
        assert g.values["day"] <= max_days
        if g.values["winner"] == "draw":
            draws += 1
            assert rules.check_winner(g.values["players"])[0] is None
            assert f"{max_days} 天" in g.values["win_reason"]
    assert draws > 0


@pytest.mark.parametrize("seed", range(30))
async def test_broken_backend_is_survivable(seed):
    backend = BrokenBackend(seed)
    g = await play(seed, ROLES[seed % len(ROLES)], backend)
    check_game(g, backend)


async def test_abstain_everything_is_peaceful_day(monkeypatch):
    class Abstain(StubBackend):
        async def exile_vote(self, state, seat, options):
            await super().exile_vote(state, seat, options)
            return {"target": None, "thought": "棄票"}

    backend = Abstain(5)
    g = await play(5, "villager", backend, invalid_rate=0.0)
    check_game(g, backend)
    for e in g.events:
        if e["kind"] == "vote_result" and all(x["target"] is None for x in e["data"]["votes"]):
            assert e["data"] == {**e["data"], "tally": {}, "exiled": None}


class BudgetBackend(StubBackend):
    """Stub whose token budget runs out after `limit` AI calls."""

    def __init__(self, seed, limit):
        super().__init__(seed)
        self.limit = limit
        self.cut_at = None  # ncalls when the graph first saw the budget gone
        self.finished = []

    def over_budget(self, game_id):
        if self.ncalls >= self.limit and self.cut_at is None:
            self.cut_at = self.ncalls
        return self.ncalls >= self.limit

    def finish_game(self, game_id):
        self.finished.append(game_id)
        return {"total": self.ncalls}
        return {"total": self.ncalls}


async def test_token_budget_ends_game_early():
    """Right after the AI step that used up the budget the game ends: no more AI calls, a draw
    unless the board already has a winner, and the backend hears that the game is over."""
    cut_after, outcomes = Counter(), Counter()
    for seed in range(16):
        for limit in (1, 3, 7, 12, 20, 35, 10_000):
            role = ROLES[seed % len(ROLES)]
            backend = BudgetBackend(seed, limit)
            g = await play(seed, role, backend)
            v, events = g.values, g.events
            assert backend.finished == [f"g-{seed}-{role}"]
            assert v["token_usage"] == {"total": backend.ncalls}  # what finish_game returned
            assert v["token_usage"] == {"total": backend.ncalls}  # what finish_game returned
            if backend.cut_at is None:
                check_game(g, backend)
                outcomes["finished"] += 1
                continue
            outcomes["cut"] += 1
            assert backend.problems == [], backend.problems[:5]
            assert backend.ncalls == backend.cut_at, "AI called after the budget ran out"
            assert [e["id"] for e in events] == list(range(len(events))) and g.streamed == events
            check_thoughts(g, backend)
            kinds = [e["kind"] for e in events]
            assert kinds[-2:] == ["game_over", "reveal"] and kinds.count("game_over") == kinds.count("reveal") == 1
            assert set(events[-1]["data"]["roles"][0]) == {"seat", "role", "is_human", "persona"}
            assert v["phase"] == "end"
            winner, reason = rules.check_winner(v["players"])
            assert (v["winner"], v["win_reason"]) == ((winner, reason) if winner else ("draw", BUDGET_REASON))
            assert events[-2]["data"] == {"winner": v["winner"], "reason": v["win_reason"]}
            cut_after[next(e["data"]["action"] for e in reversed(events) if e["kind"] == "thought")] += 1
    assert outcomes["cut"] and outcomes["finished"], outcomes
    assert {"wolf_chat", "wolf_kill", "speech", "exile_vote"} <= set(cut_after), cut_after


async def test_deterministic_with_seed():
    a = await play(11, None, StubBackend(11))
    b = await play(11, None, StubBackend(11))
    assert a.events == b.events


async def test_seed_none_and_bad_human_role():
    backend = StubBackend(1)
    graph = build_graph(backend=backend)
    cfg = {"configurable": {"thread_id": "x"}}
    inp = {"game_id": "x", "seed": None, "events": [], "human_role": "sheriff"}
    async for _ in graph.astream(inp, cfg, stream_mode=["updates", "custom"]):
        pass
    snap = await graph.aget_state(cfg)
    assert snap.interrupts or not snap.next
    assert sum(p["is_human"] for p in snap.values["players"]) == 1


async def test_shared_checkpointer_resumes_in_new_graph():
    """A game can be continued by another compiled graph sharing the checkpointer."""
    saver = InMemorySaver()
    backend = StubBackend(3)
    g1 = build_graph(backend=backend, checkpointer=saver)
    cfg = {"configurable": {"thread_id": "shared"}}
    async for _ in g1.astream({"game_id": "shared", "seed": 3, "events": [], "human_role": "villager"}, cfg):
        pass
    snap = await g1.aget_state(cfg)
    assert snap.interrupts
    g2 = build_graph(backend=backend, checkpointer=saver)
    snap2 = await g2.aget_state(cfg)
    assert snap2.values["events"] == snap.values["events"]
    assert snap2.interrupts[0].value == snap.interrupts[0].value


def test_board_has_every_side():
    assert {"werewolf", "villager"} <= set(config.ROLE_SETUP) and set(GOD_ROLES) <= set(config.ROLE_SETUP)
    assert set(config.ROLE_SETUP) <= set(ROLE_ZH)


async def test_langgraph_resume_quirks():
    """Documents LangGraph behaviour the server must respect: resume={} re-raises the same
    interrupt; a proper value then resumes normally."""
    backend = StubBackend(2)
    graph = build_graph(backend=backend)
    cfg = {"configurable": {"thread_id": "quirk"}}
    async for _ in graph.astream({"game_id": "quirk", "seed": 2, "events": [], "human_role": "villager"}, cfg):
        pass
    first = (await graph.aget_state(cfg)).interrupts[0]
    async for _ in graph.astream(Command(resume={}), cfg):
        pass
    again = (await graph.aget_state(cfg)).interrupts[0]
    assert again.value == first.value
    async for _ in graph.astream(Command(resume={"text": "我是 AI"}), cfg):
        pass
    snap = await graph.aget_state(cfg)
    assert any(e["text"] == "我是 AI" for e in snap.values["events"])


class FixedLatency(StubBackend):
    """Every AI call takes `delay` seconds (speak calls: `speak_delay`, default `delay`); night 1:
    wolves kill 1 and the witch saves; day 1: all AIs vote `exile`
    (seed 0 with human_role=villager: 1村 2預 3狼 4狼 5村* 6女巫)."""

    def __init__(self, exile, delay, speak_delay=None):
        super().__init__(0)
        self.exile, self.delay = exile, delay
        self.speak_delay = delay if speak_delay is None else speak_delay

    async def _slow(self, coro, delay=None):
        await asyncio.sleep(self.delay if delay is None else delay)
        return await coro

    async def speak(self, state, seat, kind):
        return await self._slow(super().speak(state, seat, kind), self.speak_delay)

    async def wolf_target(self, state, seat, options):
        res = await self._slow(super().wolf_target(state, seat, options))
        return {**res, "target": 1 if 1 in options else options[0]}

    async def witch(self, state, seat, killed, can_save, poison_options):
        res = await self._slow(super().witch(state, seat, killed, can_save, poison_options))
        return {**res, "action": "save" if can_save else "none", "target": None}

    async def seer_check(self, state, seat, options):
        return await self._slow(super().seer_check(state, seat, options))

    async def exile_vote(self, state, seat, options):
        res = await self._slow(super().exile_vote(state, seat, options))
        return {**res, "target": self.exile if state["day"] == 1 and self.exile in options else None}


async def night2_duration(exile, delay, speak_delay=None):
    graph = build_graph(backend=FixedLatency(exile, delay, speak_delay))
    cfg = {"configurable": {"thread_id": f"t{exile}"}}
    inp = {"game_id": f"t{exile}", "seed": 0, "events": [], "human_role": "villager"}
    start = end = None
    while end is None:
        async for mode, chunk in graph.astream(inp, cfg, stream_mode=["updates", "custom"]):
            for node, upd in (chunk.items() if mode == "updates" else ()):
                for e in (upd or {}).get("events", []) if node != "__interrupt__" else ():
                    if e["kind"] == "system" and e["text"].startswith("第2夜"):
                        start = asyncio.get_running_loop().time()
                    elif e["kind"] == "death" and e["day"] == 2:
                        end = asyncio.get_running_loop().time()
        snap = await graph.aget_state(cfg)
        assert snap.values["players"][4]["is_human"]
        p = snap.interrupts[0].value
        inp = Command(resume={"text": "嗨"} if p["type"] == "speak" else
                      {"target": exile if p["action"] == "exile_vote" and exile in p["options"] else None})
    return end - start


@pytest.mark.parametrize("speak_factor", [1, 2])
async def test_night_length_does_not_reveal_who_is_alive(monkeypatch, speak_factor):
    """Exiling a wolf (no wolf chat on night 2), the seer or the witch used to make night 2 visibly
    shorter than exiling a villager; skipped steps now wait about as long as the AI calls would.
    speak_factor=2: speak calls are slower than the others (as live), so pads must use the latency
    of the skipped call kind, not one average over all calls."""
    import werewolf.graph as graph_mod

    monkeypatch.setattr(graph_mod.random, "uniform", lambda a, b: (a + b) / 2)
    delay = 0.04
    durations = {who: await night2_duration(seat, delay, speak_factor * delay) for who, seat in
                 [("villager", 1), ("wolf", 3), ("seer", 2), ("witch", 6)]}
    base = durations["villager"]  # 2 wolf-chat turns, wolf picks, witch, seer: 5 sequential calls
    assert base >= 5 * delay
    for d in durations.values():
        assert abs(d - base) < 0.1 * base, durations


async def timed_game(seed, human_role, pause, monkeypatch, per_char=0.0, backend=None):
    """Play a game with config.SPEECH_PAUSE_S = pause and TYPE_S_PER_CHAR = per_char;
    -> (final values, [(time, node, events)] per node update, [(time resumed, time of the next
    update)] for every human answer)."""
    monkeypatch.setattr(config, "SPEECH_PAUSE_S", pause)
    monkeypatch.setattr(config, "TYPE_S_PER_CHAR", per_char)
    graph = build_graph(backend=backend or StubBackend(seed))
    cfg = {"configurable": {"thread_id": f"p{seed}"}}
    inp = {"game_id": f"p{seed}", "seed": seed, "events": [], "human_role": human_role}
    loop = asyncio.get_running_loop()
    updates, resumes = [], []
    while True:
        start = loop.time()
        first = None
        async for chunk in graph.astream(inp, cfg, stream_mode="updates"):
            for node, upd in chunk.items():
                if node != "__interrupt__":
                    updates.append((loop.time(), node, upd.get("events", [])))
                    first = first or loop.time()
        if isinstance(inp, Command) and first is not None:
            resumes.append((start, first))
        snap = await graph.aget_state(cfg)
        if not snap.interrupts:
            return snap.values, updates, resumes
        p = snap.interrupts[0].value
        inp = Command(resume={"text": "嗨"} if p["type"] == "speak" else
                      {"target": p["options"][0]} if p["type"] == "choose" else {"action": "none"})


@pytest.mark.parametrize("human_role", ["villager", "werewolf"])
async def test_pause_after_ai_speeches_the_human_sees(monkeypatch, human_role):
    """After every AI speech / last words the human can see (and AI wolf chat, if the human is a
    wolf) nothing happens until the page has typed it out (TYPE_S_PER_CHAR per character) and
    SPEECH_PAUSE_S more; after the human's own speeches and hidden wolf chat the game goes
    straight on, and answering a prompt doesn't wait again."""
    pause, per_char = 0.05, 0.004
    values, updates, resumes = await timed_game(3, human_role, pause, monkeypatch, per_char=per_char)
    human = values["human_seat"]
    wolf_human = human_role == "werewolf"
    paused = quick = 0
    for (t, node, evs), (t2, _, _) in zip(updates, updates[1:]):
        spoken = [e for e in evs if e["kind"] in ("speech", "last_words", "wolf_chat")]
        if not spoken:
            continue
        e = spoken[-1]
        seen = e["seat"] != human and (e["kind"] != "wolf_chat" or wolf_human)
        if seen:
            wait = pause + min(len(e["text"]) * per_char, config.TYPE_MAX_S)
            assert wait * 0.95 <= t2 - t < wait + 0.04, (e, t2 - t, wait)
            paused += 1
        else:
            assert t2 - t < pause * 0.8, (e, t2 - t)
            quick += 1
    assert paused and quick
    assert all(first - start < pause * 0.8 for start, first in resumes)


async def test_next_ai_thinks_during_the_pause(monkeypatch):
    """The next AI speaker's call runs while the human reads: two AI speeches are max(pause,
    call time) apart, not pause + call time."""
    class Slow(StubBackend):
        async def speak(self, state, seat, kind):
            await asyncio.sleep(0.06)
            return await super().speak(state, seat, kind)

    pause = 0.1
    values, updates, _ = await timed_game(3, "villager", pause, monkeypatch, backend=Slow(3))
    human = values["human_seat"]
    gaps = []
    for (t, _, evs), (t2, _, evs2) in zip(updates, updates[1:]):
        if (any(e["kind"] == "speech" and e["seat"] != human for e in evs)
                and any(e["kind"] == "speech" and e["seat"] != human for e in evs2)):
            gaps.append(t2 - t)
    assert gaps and all(pause * 0.95 <= g < pause + 0.045 for g in gaps), gaps


@pytest.mark.parametrize("human_role,action,kind", [("villager", "exile_vote", "vote_result"),
                                                    ("werewolf", "wolf_kill", "wolf_kill")])
async def test_ais_decide_while_the_human_does(human_role, action, kind):
    """wolf_pick / cast_vote run once per voter at the same time (Send): the human's prompt streams
    while the AIs are still deciding, and the answer is counted right away, without asking them again."""
    delay = 0.3

    class SlowPicks(StubBackend):
        async def wolf_target(self, state, seat, options):
            await asyncio.sleep(delay)
            return await super().wolf_target(state, seat, options)

        async def exile_vote(self, state, seat, options):
            await asyncio.sleep(delay)
            return await super().exile_vote(state, seat, options)

    loop = asyncio.get_running_loop()
    for seed in range(10):  # until the human is asked for `action` (a villager may die first)
        backend = SlowPicks(seed)
        graph = build_graph(backend=backend)
        cfg = {"configurable": {"thread_id": f"s{seed}"}}
        inp = {"game_id": f"s{seed}", "seed": seed, "events": [], "human_role": human_role}
        while True:
            asked = None
            async for chunk in graph.astream(inp, cfg, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    asked = loop.time()
            snap = await graph.aget_state(cfg)
            if not snap.interrupts or snap.interrupts[0].value.get("action") == action:
                break
            p = snap.interrupts[0].value
            inp = Command(resume={"text": "嗨"} if p["type"] == "speak" else
                          {"target": p["options"][0]} if p["type"] == "choose" else {"action": "none"})
        if snap.interrupts:
            break
    else:
        raise AssertionError(f"the human was never asked for {action}")
    assert loop.time() - asked >= delay * 0.8, "the prompt should come while the AIs still decide"

    calls, start, counted = backend.ncalls, loop.time(), None
    target = snap.interrupts[0].value["options"][0]
    async for chunk in graph.astream(Command(resume={"target": target}), cfg, stream_mode="updates"):
        for upd in chunk.values():
            if counted is None and isinstance(upd, dict) and any(e["kind"] == kind for e in upd.get("events", [])):
                counted = (loop.time() - start, backend.ncalls)
    assert counted is not None and counted[0] < delay / 2, counted
    assert counted[1] == calls, "the AIs were asked again"
