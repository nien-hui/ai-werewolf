"""Server tests: full games through the /ws WebSocket (TestClient) with an in-file stub backend."""
import asyncio
import contextlib
import json
import random

import anyio
import pytest
from fastapi.testclient import TestClient

from werewolf import config, rules, server
from werewolf.personas import persona_name
from werewolf.server import ClientError, create_app, human_tags, validate_action
from werewolf.state import ROLE_ZH, VIS_ALL, VIS_GOD, VIS_WOLVES, visible_to

TIMEOUT = 30  # seconds per receive; a hang fails instead of blocking forever
PHRASES = ["我是村民啦", "我覺得 3號 怪怪的", "先聽聽大家", "嗯…我沒什麼資訊", "今天出 2號 吧"]
PAYLOAD_KEYS = {
    "speak": {"type", "kind", "day", "prompt"},
    "choose": {"type", "action", "day", "prompt", "options", "allow_skip"},
    "witch": {"type", "day", "prompt", "killed", "can_save", "poison_options"},
}
REVIEW_KEYS = {"players", "events", "winner", "win_reason", "human_seat"}
REVIEW_PLAYER_KEYS = {"seat", "role", "role_zh", "is_human", "persona", "persona_name", "alive"}
SPEECH_KINDS = {"discussion": "speech", "last_words": "last_words", "wolf_chat": "wolf_chat"}


def _seeds_by_role(n=3):
    """Seeds whose human gets each role (setup draws roles with random.Random(seed))."""
    found = {r: [] for r in ROLE_ZH}
    for seed in range(1000):
        players = rules.assign_players(random.Random(seed))
        role = next(p["role"] for p in players if p["is_human"])
        if len(found[role]) < n:
            found[role].append(seed)
    return [(role, s) for role, seeds in found.items() for s in seeds]


ROLE_SEEDS = _seeds_by_role()


# ------------------------------------------------------------ stub backend
class StubBackend:
    """Random but legal AI answers. `gate` (asyncio.Event) blocks every call until set."""

    def __init__(self, seed=0, gate=None):
        self.rng = random.Random(seed)
        self.gate = gate
        self.calls = 0

    async def _enter(self):
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        await asyncio.sleep(0)

    def _others(self, state, seat):
        return [p["seat"] for p in state["players"] if p["seat"] != seat]

    async def speak(self, state, seat, kind):
        await self._enter()
        return {"speech": self.rng.choice(PHRASES), "thought": f"{seat}號 的祕密想法"}

    async def wolf_target(self, state, seat, options):
        await self._enter()
        return {"target": self.rng.choice(options), "thought": "刀一個"}

    async def witch(self, state, seat, killed, can_save, poison_options):
        await self._enter()
        legal = [("none", None)] + ([("save", None)] if can_save else []) + [("poison", t) for t in poison_options]
        action, target = self.rng.choice(legal)
        return {"action": action, "target": target, "thought": "用藥"}

    async def seer_check(self, state, seat, options):
        await self._enter()
        return {"target": self.rng.choice(options), "thought": "驗一個"}

    async def exile_vote(self, state, seat, options):
        await self._enter()
        r = self.rng.random()
        target = options[0] if r < 0.5 else (None if r > 0.85 else self.rng.choice(options))
        return {"target": target, "thought": "投票"}


# ---------------------------------------------------------- socket client
class Conn:
    """Wraps a TestClient WebSocket session with a receive timeout."""

    def __init__(self, ws):
        self.ws = ws

    def send(self, msg):
        self.ws.send_json(msg)

    def send_text(self, text):
        self.ws.send_text(text)

    def recv(self, timeout=TIMEOUT):
        async def get():
            with anyio.fail_after(timeout):
                return await self.ws._send_rx.receive()

        message = self.ws.portal.call(get)
        assert message["type"] == "websocket.send", message
        return json.loads(message["text"])

    def call(self, fn, *args):
        """Run fn in the server's event loop (shared TestClient portal)."""
        return self.ws.portal.call(fn, *args)


def expected_tags(events, seat):
    """Independent re-implementation of what the human knows about other seats."""
    tags = {}
    for e in events:
        if e["kind"] == "role_assign" and e["seat"] == seat:
            tags.update({m: "狼隊友" for m in e["data"]["teammates"]})
        if e["kind"] == "seer_result" and e["seat"] == seat:
            tags[e["data"]["target"]] = "查驗：狼人" if e["data"]["is_wolf"] else "查驗：好人"
    return tags


class Human:
    """Plays the human through the socket and checks every message it receives."""

    def __init__(self, rng, invalid_rate=0.3):
        self.rng = rng
        self.invalid_rate = invalid_rate
        self.game_id = None
        self.you = None
        self.events = []
        self.players = []  # every players message
        self.statuses = []
        self.prompts = []
        self.answers = []  # (prompt, forwarded value)
        self.errors = []
        self.review = None
        self.pending = None
        self.message_types = []

    # message checks -------------------------------------------------------
    def check_event(self, ev):
        assert self.you is not None, "event before game_started"
        seat, role = self.you["seat"], self.you["role"]
        assert visible_to(ev, seat, role, include_god=False), f"leaked event {ev}"
        assert ev["visibility"] != VIS_GOD and ev["kind"] != "thought", f"god event leaked {ev}"
        if ev["visibility"] == VIS_WOLVES:
            assert role == "werewolf", f"wolf channel leaked to {role}: {ev}"
        if ev["visibility"].startswith("seat:"):
            assert ev["visibility"] == f"seat:{seat}", f"private event of another seat leaked: {ev}"
        if self.events:
            assert ev["id"] > self.events[-1]["id"], "events out of order / duplicated"

    def check_players(self, msg):
        assert set(msg) == {"type", "day", "phase", "players"}
        assert [p["seat"] for p in msg["players"]] == list(range(1, config.NUM_PLAYERS + 1))
        me = self.you["seat"]
        tags = expected_tags(self.events, me)
        for p in msg["players"]:
            assert set(p) == {"seat", "alive", "is_you", "tag"}
            assert p["is_you"] == (p["seat"] == me)
            assert p["tag"] == tags.get(p["seat"]), (p, tags)
        if self.you["role"] not in ("werewolf", "seer"):
            assert all(p["tag"] is None for p in msg["players"])

    def check_prompt(self, payload):
        assert set(payload) == PAYLOAD_KEYS[payload["type"]], payload
        assert payload["prompt"]

    def on_message(self, msg):
        t = msg["type"]
        self.message_types.append(t)
        if t == "game_started":
            assert set(msg) == {"type", "game_id", "you", "num_players"}
            assert msg["num_players"] == config.NUM_PLAYERS
            you = msg["you"]
            assert you["role_zh"] == ROLE_ZH[you["role"]]
            if self.you is not None:
                assert you == self.you and msg["game_id"] == self.game_id
            self.game_id, self.you = msg["game_id"], you
        elif t == "players":
            assert self.you is not None, "players before game_started"
            self.check_players(msg)
            self.players.append(msg)
        elif t == "event":
            self.check_event(msg["event"])
            self.events.append(msg["event"])
        elif t == "status":
            assert isinstance(msg["text"], str)
            self.statuses.append(msg["text"])
        elif t == "prompt":
            self.check_prompt(msg["prompt"])
            assert self.pending is None, "two prompts without an answer"
            self.prompts.append(msg["prompt"])
            self.pending = msg["prompt"]
        elif t == "game_end":
            self.review = msg["review"]
        elif t == "error":
            raise AssertionError(f"unexpected error: {msg}")
        else:
            raise AssertionError(f"unknown message {msg}")

    # answers --------------------------------------------------------------
    def legal(self, p):
        rng = self.rng
        if p["type"] == "speak":
            return {"text": rng.choice(PHRASES + ["", f"  {rng.choice(PHRASES)}  "])}
        if p["type"] == "choose":
            options = p["options"] + ([None] if p["allow_skip"] else [])
            return {"target": rng.choice(options)}
        legal = [{"action": "none", "target": None}]
        if p["can_save"]:
            legal.append({"action": "save", "target": rng.choice([None, p["killed"]])})
        legal += [{"action": "poison", "target": t} for t in p["poison_options"]]
        return rng.choice(legal)

    def invalid(self, p):
        bad = [3, None, [], "hello"]
        if p["type"] == "speak":
            bad += [{"text": 5}, {}, {"text": None}, {"text": "字" * (config.MAX_SPEECH_CHARS + 1)}]
        elif p["type"] == "choose":
            opts = p["options"]
            bad += [{"target": 99}, {"target": str(opts[0])}, {}, {"target": self.you["seat"]},
                    {"target": 1.5}]
            if 1 in opts:
                bad.append({"target": True})
            if not p["allow_skip"]:
                bad.append({"target": None})
        else:
            bad += [{}, {"action": "fly"}, {"action": "poison", "target": 99}, {"action": "poison"},
                    {"action": "none", "target": 2}]
            if not p["can_save"]:
                bad.append({"action": "save", "target": None})
            else:
                others = [s for s in range(1, 7) if s != p["killed"]]
                bad.append({"action": "save", "target": others[0]})
            if not p["poison_options"]:
                bad.append({"action": "poison", "target": 1})
        return self.rng.choice(bad)

    def answer(self, conn):
        p, self.pending = self.pending, None
        while self.rng.random() < self.invalid_rate:
            conn.send({"type": "action", "value": self.invalid(p)})
            msg = conn.recv()  # graph is paused: the error must be the very next message
            assert msg["type"] == "error" and msg["message"], msg
            self.errors.append(msg["message"])
        value = self.legal(p)
        conn.send({"type": "action", "value": value})
        self.answers.append((p, value))

    def play(self, conn, max_prompts=None):
        """Handle messages until game_end (or until `max_prompts` prompts are pending)."""
        while self.review is None:
            if self.pending is None:
                self.on_message(conn.recv())
            if self.pending is not None:
                if max_prompts is not None and len(self.prompts) >= max_prompts:
                    return
                self.answer(conn)


async def graph_values(app, game_id):
    snap = await app.state.manager.graph.aget_state({"configurable": {"thread_id": game_id}})
    return snap.values


def final_values(app, game_id):
    """The finished game's state. Its graph checkpoints are deleted at game end, so this comes
    from the session (built from the streamed updates; the review comes from the graph state)."""
    s = app.state.manager.sessions[game_id]
    assert s.review is not None
    return {"events": s.events, "players": s.players, "human_seat": s.human_seat}


def check_finished_game(h, values, answers=None):
    """After game_end: review completeness, nothing hidden from the human leaked, answers applied.
    `answers`: every (prompt, value) the human sent in this game (default h.answers)."""
    answers = h.answers if answers is None else answers
    rv = h.review
    me, role = h.you["seat"], h.you["role"]
    assert set(rv) == REVIEW_KEYS
    all_events = values["events"]
    assert rv["events"] == all_events, "review must contain ALL events"
    assert [e["id"] for e in all_events] == list(range(len(all_events)))
    assert any(e["visibility"] == VIS_GOD for e in all_events)  # thoughts are in the review
    # the human got exactly the events visible to them, in order, renumbered 0, 1, 2, ...
    # (global ids would show how many hidden events happened in between)
    visible = [e for e in all_events if visible_to(e, me, role, include_god=False)]
    assert h.events == [{**e, "id": i} for i, e in enumerate(visible)]
    assert all(e["visibility"] != VIS_GOD for e in h.events)
    if role != "werewolf":
        assert all(e["visibility"] != VIS_WOLVES for e in h.events)

    assert rv["human_seat"] == me == values["human_seat"]
    assert [p["seat"] for p in rv["players"] if p["is_human"]] == [me]
    for p, sp in zip(rv["players"], values["players"], strict=True):
        assert set(p) == REVIEW_PLAYER_KEYS
        assert (p["seat"], p["role"], p["is_human"], p["persona"], p["alive"]) == \
               (sp["seat"], sp["role"], sp["is_human"], sp["persona"], sp["alive"])
        assert p["role_zh"] == ROLE_ZH[p["role"]]
        assert p["persona_name"] == persona_name(p["persona"])
        assert bool(p["persona_name"]) != p["is_human"]
    assert next(p["role"] for p in rv["players"] if p["seat"] == me) == role

    over = next(e for e in all_events if e["kind"] == "game_over")
    assert (rv["winner"], rv["win_reason"]) == (over["data"]["winner"], over["data"]["reason"])
    reveal = [e for e in all_events if e["kind"] == "reveal"]
    assert len(reveal) == 1 and [r["seat"] for r in reveal[0]["data"]["roles"] if r["is_human"]] == [me]

    # every forwarded answer landed in the events
    speeches = iter(e for e in all_events if e["seat"] == me and e["kind"] in SPEECH_KINDS.values())
    for p, value in answers:
        if p["type"] == "speak":
            ev = next(speeches)
            assert ev["kind"] == SPEECH_KINDS[p["kind"]]
            text = rules.coerce_speech(value)
            assert ev["text"] == text if text else ev["text"] in rules.PASS_SPEECHES[p["kind"]]
    assert next(speeches, None) is None

    # the last players message reflects the final board and the human's knowledge
    last = h.players[-1]
    assert [p["alive"] for p in last["players"]] == [p["alive"] for p in values["players"]]
    tags = {p["seat"]: p["tag"] for p in last["players"] if p["tag"]}
    if role == "werewolf":
        mates = {p["seat"] for p in values["players"] if p["role"] == "werewolf" and p["seat"] != me}
        assert {s for s, t in tags.items() if t == "狼隊友"} == mates
    elif role == "seer":
        checks = {e["data"]["target"]: e["data"]["is_wolf"] for e in all_events
                  if e["kind"] == "seer_result" and e["seat"] == me}
        assert tags == {s: "查驗：狼人" if w else "查驗：好人" for s, w in checks.items()}
    else:
        assert tags == {}


@pytest.fixture
def backend():
    return StubBackend(1234)


@pytest.fixture
def app(backend):
    return create_app(backend=backend)


@pytest.fixture
def client(app):
    # one portal/event loop shared by every socket (games outlive sockets); sockets opened with
    # open_ws() are closed before the client, so a failing test cannot hang the teardown
    with TestClient(app) as c, contextlib.ExitStack() as stack:
        c.sockets = stack
        yield c


def open_ws(client):
    """Open a socket outside a `with` block; close early with close_ws()."""
    ws = client.websocket_connect("/ws")
    ws.__enter__()
    client.sockets.callback(close_ws, ws)
    return ws


def close_ws(ws):
    ws.__exit__(None, None, None)  # waits until the server handler has finished; idempotent


def take_replay(conn, h, n_events, tail):
    """Receive a resume_game replay: game_started, players, n_events events, then `tail`."""
    msgs = [conn.recv() for _ in range(2 + n_events + (1 if tail else 0))]
    assert [m["type"] for m in msgs] == ["game_started", "players"] + ["event"] * n_events + ([tail] if tail else [])
    # the replayed players tags reflect all replayed events, so check that message after them
    for m in [msgs[0], *msgs[2:2 + n_events], msgs[1], *msgs[2 + n_events:]]:
        h.on_message(m)
    return msgs


def new_game(client, seed):
    ws = open_ws(client)
    conn = Conn(ws)
    conn.send({"type": "new_game", "seed": seed})
    return ws, conn


# ----------------------------------------------------------------- tests
@pytest.mark.parametrize("role,seed", ROLE_SEEDS)
def test_full_game_via_socket(client, app, role, seed):
    with client.websocket_connect("/ws") as ws:
        conn = Conn(ws)
        conn.send({"type": "new_game", "seed": seed})
        h = Human(random.Random(seed))
        h.play(conn)
        assert h.you["role"] == role
        assert h.message_types[0] == "game_started"
        assert h.statuses, "status messages are forwarded"
        check_finished_game(h, final_values(app, h.game_id))
        # nothing more is pending: an action now is an error
        conn.send({"type": "action", "value": {"target": 1}})
        assert conn.recv()["type"] == "error"


@pytest.mark.parametrize("seed", [None, None, 777])
def test_full_game_random_seed(client, app, seed):
    with client.websocket_connect("/ws") as ws:
        conn = Conn(ws)
        conn.send({"type": "new_game"} if seed is None else {"type": "new_game", "seed": seed})
        h = Human(random.Random(seed), invalid_rate=0.0)
        h.play(conn)
        check_finished_game(h, final_values(app, h.game_id))


def test_same_seed_same_board(client):
    boards = []
    for _ in range(2):
        with client.websocket_connect("/ws") as ws:
            conn = Conn(ws)
            conn.send({"type": "new_game", "seed": 42})
            h = Human(random.Random(0))
            while not h.events or h.events[-1]["kind"] != "role_assign":
                h.on_message(conn.recv())
            boards.append(h.you)
    assert boards[0] == boards[1]


@pytest.mark.parametrize("close_first", [True, False])
@pytest.mark.parametrize("role,seed", [ROLE_SEEDS[0], ROLE_SEEDS[4]])
def test_resume_on_new_socket(client, app, role, seed, close_first):
    rng = random.Random(seed)
    h1 = Human(rng)
    ws1, c1 = new_game(client, seed)
    h1.play(c1, max_prompts=2)
    assert h1.pending is not None
    if close_first:  # a disconnect must not kill the game
        close_ws(ws1)

    with client.websocket_connect("/ws") as ws2:
        c2 = Conn(ws2)
        c2.send({"type": "resume_game", "game_id": h1.game_id})
        h2 = Human(rng)
        take_replay(c2, h2, len(h1.events), "prompt")
        assert h2.you == h1.you and h2.game_id == h1.game_id
        assert h2.events == h1.events
        assert h2.prompts == [h1.pending]
        assert h2.players[0]["players"] == h1.players[-1]["players"]
        h2.play(c2)
        values = final_values(app, h1.game_id)
        check_finished_game(h2, values, h1.answers + h2.answers)

        if not close_first:  # the first socket received the same live stream
            while h1.review is None:
                h1.pending = None  # h2 answers the prompts
                h1.on_message(c1.recv())
            assert h1.events == h2.events
            assert h1.review == h2.review
            close_ws(ws1)

    # resuming a finished game re-sends history and game_end
    with client.websocket_connect("/ws") as ws3:
        c3 = Conn(ws3)
        c3.send({"type": "resume_game", "game_id": h1.game_id})
        h3 = Human(rng)
        take_replay(c3, h3, len(h2.events), "game_end")
        assert h3.events == h2.events and h3.review == h2.review


def test_disconnect_while_running_then_resume(client, app):
    backend = app.state.manager._backend
    backend.gate = asyncio.Event()  # every AI call blocks until released
    rng = random.Random(5)
    ws1, c1 = new_game(client, 5)
    h1 = Human(rng)
    # night_start's status, event and players were sent => the graph is inside the first AI call
    while not (any(e["kind"] == "system" and e["phase"] == "night" for e in h1.events)
               and h1.message_types[-1] == "players"):
        h1.on_message(c1.recv())
        if h1.pending is not None:
            h1.answer(c1)
    assert h1.statuses == ["夜晚行動中…"]
    # graph running, nothing pending -> action is rejected
    c1.send({"type": "action", "value": {"text": "插話"}})
    err = c1.recv()
    assert err["type"] == "error"
    close_ws(ws1)

    with client.websocket_connect("/ws") as ws2:
        c2 = Conn(ws2)
        c2.send({"type": "resume_game", "game_id": h1.game_id})
        h2 = Human(rng)
        msgs = take_replay(c2, h2, len(h1.events), "status")
        assert msgs[-1]["text"] == h1.statuses[-1] and h2.events == h1.events
        c2.call(backend.gate.set)
        h2.play(c2)
        check_finished_game(h2, final_values(app, h1.game_id), h1.answers + h2.answers)


def test_invalid_messages(client):
    with client.websocket_connect("/ws") as ws:
        conn = Conn(ws)
        bad = [
            "not json {",
            json.dumps([1, 2]),
            json.dumps({"type": "dance"}),
            json.dumps({"no": "type"}),
            json.dumps({"type": "action", "value": {"text": "hi"}}),  # no game yet
            json.dumps({"type": "resume_game", "game_id": "nope"}),
            json.dumps({"type": "resume_game", "game_id": 5}),
            json.dumps({"type": "new_game", "seed": "abc"}),
            json.dumps({"type": "new_game", "seed": True}),
            json.dumps({"type": "new_game", "seed": 1.5}),
            json.dumps({"type": "new_game", "seed": 2**60}),
        ]
        for text in bad:
            conn.send_text(text)
            msg = conn.recv()
            assert msg["type"] == "error" and isinstance(msg["message"], str) and msg["message"], (text, msg)
        # the socket still works afterwards
        conn.send({"type": "new_game", "seed": 3})
        assert conn.recv()["type"] == "game_started"


def test_new_game_on_same_socket_switches_game(client, app):
    with client.websocket_connect("/ws") as ws:
        conn = Conn(ws)
        conn.send({"type": "new_game", "seed": 1})
        h1 = Human(random.Random(1))
        h1.play(conn, max_prompts=1)
        conn.send({"type": "new_game", "seed": 2})
        h2 = Human(random.Random(2), invalid_rate=0.0)
        h2.play(conn)
        assert h2.game_id != h1.game_id
        check_finished_game(h2, final_values(app, h2.game_id))
        # the old game is still paused at its prompt and can be resumed
        conn.send({"type": "resume_game", "game_id": h1.game_id})
        h3 = Human(random.Random(1))
        take_replay(conn, h3, len(h1.events), "prompt")
        assert h3.prompts == [h1.pending] and h3.events == h1.events


def test_runner_error_is_reported(client, app, monkeypatch):
    def boom(update):
        raise RuntimeError("boom")

    with client.websocket_connect("/ws") as ws:
        conn = Conn(ws)
        with monkeypatch.context() as m:  # undoes only this patch, not conftest's test defaults
            m.setattr(server.GameSession, "_on_update", lambda self, update: boom(update))
            conn.send({"type": "new_game", "seed": 9})
            msgs = [conn.recv(), conn.recv()]
        assert msgs[0] == {"type": "status", "text": ""}
        assert msgs[1]["type"] == "error" and msgs[1]["message"] == server.RUN_ERROR
        session = next(iter(app.state.manager.sessions.values()))
        assert session.failed and not session.running
        # a resume retries the run; with the bug gone the game continues normally
        conn.send({"type": "resume_game", "game_id": session.game_id})
        h = Human(random.Random(9), invalid_rate=0.0)
        h.play(conn)
        assert h.you["seat"] == session.human_seat


def test_backend_build_failure_is_reported(monkeypatch):
    def broken_build(**kwargs):
        raise RuntimeError("no backend")

    monkeypatch.setattr(server, "build_graph", broken_build)
    app = create_app()
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        conn = Conn(ws)
        conn.send({"type": "new_game"})
        msg = conn.recv()
        assert msg["type"] == "error" and "AI" in msg["message"]
        assert app.state.manager.sessions == {} and app.state.manager._graph is None


def test_health_and_static(client, monkeypatch):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "model": config.MODEL_NAME, "fake": config.FAKE_LLM}
    monkeypatch.setattr(config, "FAKE_LLM", not config.FAKE_LLM)
    assert client.get("/api/health").json()["fake"] == config.FAKE_LLM
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert r.text == (config.ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert r.headers["cache-control"] == "no-cache"
    for name in ("app.js", "style.css"):
        if (config.ROOT / "static" / name).exists():
            r = client.get(f"/static/{name}")
            assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
            # revalidation still works: an unchanged file is a cheap 304
            again = client.get(f"/static/{name}", headers={"if-none-match": r.headers["etag"]})
            assert again.status_code == 304 and again.headers["cache-control"] == "no-cache"
    assert client.get("/static/nope.js").status_code == 404


def test_password(client, monkeypatch):
    from starlette.websockets import WebSocketDisconnect
    monkeypatch.setattr(config, "PASSWORD", "秘密pw")
    for url in ("/ws", "/ws?key=wrong", "/ws?key=" + "秘密"):
        with client.websocket_connect(url) as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_text()
            assert exc.value.code == server.AUTH_CLOSE
    with client.websocket_connect("/ws?key=" + "秘密pw") as ws:
        conn = Conn(ws)
        conn.send({"type": "new_game", "seed": 1})
        assert conn.recv()["type"] == "game_started"


def test_app_does_not_build_backend_at_import():
    assert server.app.state.manager._graph is None
    assert create_app().state.manager._graph is None


def test_default_backend_fake_llm_game(monkeypatch):
    pytest.importorskip("werewolf.ai")
    monkeypatch.setattr(config, "FAKE_LLM", True)
    app = create_app()  # backend=None -> werewolf.ai.get_backend() on first game
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        conn = Conn(ws)
        conn.send({"type": "new_game", "seed": 11})
        h = Human(random.Random(11), invalid_rate=0.1)
        h.play(conn)
        assert app.state.manager._backend is None and app.state.manager._graph is not None
        thoughts = [e for e in h.review["events"] if e["kind"] == "thought"]
        assert thoughts and all("測試用" in e["text"] for e in thoughts)  # werewolf.ai.FakeBackend played
        check_finished_game(h, final_values(app, h.game_id))


# ------------------------------------------------------------ unit tests
SPEAK = {"type": "speak", "kind": "discussion", "day": 1, "prompt": "p"}
CHOOSE = {"type": "choose", "action": "seer_check", "day": 1, "prompt": "p", "options": [1, 3], "allow_skip": False}
EXILE = {**CHOOSE, "action": "exile_vote", "allow_skip": True}
WITCH = {"type": "witch", "day": 1, "prompt": "p", "killed": 3, "can_save": True, "poison_options": [1, 2, 4]}
WITCH_NO_SAVE = {**WITCH, "killed": None, "can_save": False}
WITCH_NO_POISON = {**WITCH, "poison_options": []}


@pytest.mark.parametrize("prompt,value,expected", [
    (SPEAK, {"text": "嗨"}, {"text": "嗨"}),
    (SPEAK, {"text": ""}, {"text": ""}),
    (SPEAK, {"text": " 嗨 ", "extra": 1}, {"text": " 嗨 "}),
    (CHOOSE, {"target": 3}, {"target": 3}),
    (CHOOSE, {"target": 3, "reason": "x"}, {"target": 3}),  # extra fields are dropped
    (EXILE, {"target": None}, {"target": None}),
    (WITCH, {"action": "save", "target": None}, {"action": "save"}),
    (WITCH, {"action": "save", "target": 3}, {"action": "save"}),
    (WITCH, {"action": "save"}, {"action": "save"}),
    (WITCH, {"action": "poison", "target": 2}, {"action": "poison", "target": 2}),
    (WITCH, {"action": "none", "target": None}, {"action": "none"}),
    (WITCH_NO_SAVE, {"action": "poison", "target": 4}, {"action": "poison", "target": 4}),
])
def test_validate_action_ok(prompt, value, expected):
    assert validate_action(prompt, value) == expected


@pytest.mark.parametrize("prompt,value", [
    (SPEAK, None), (SPEAK, "嗨"), (SPEAK, {}), (SPEAK, {"text": 1}),
    (SPEAK, {"text": "a" * (config.MAX_SPEECH_CHARS + 1)}),
    (CHOOSE, {}), (CHOOSE, {"target": None}), (CHOOSE, {"target": 2}), (CHOOSE, {"target": "3"}),
    (CHOOSE, {"target": True}), (CHOOSE, {"target": 3.0}), (EXILE, {"target": 9}),
    (WITCH, {}), (WITCH, {"action": "SAVE"}), (WITCH, {"action": "save", "target": 1}),
    (WITCH, {"action": "save", "target": True}),
    (WITCH, {"action": "poison"}), (WITCH, {"action": "poison", "target": 3}),
    (WITCH, {"action": "poison", "target": None}), (WITCH, {"action": "none", "target": 1}),
    (WITCH_NO_SAVE, {"action": "save"}), (WITCH_NO_POISON, {"action": "poison", "target": 1}),
    ({"type": "mystery"}, {"text": "x"}),
])
def test_validate_action_rejects(prompt, value):
    with pytest.raises(ClientError) as exc:
        validate_action(prompt, value)
    assert str(exc.value)


def test_human_tags_only_from_own_knowledge():
    def ev(kind, seat, data, vis):
        return {"id": 0, "day": 1, "phase": "night", "kind": kind, "seat": seat, "text": "", "visibility": vis,
                "data": data}

    events = [
        ev("role_assign", 2, {"role": "werewolf", "teammates": [5]}, "seat:2"),
        ev("seer_result", 2, {"target": 4, "is_wolf": False}, "seat:2"),
        ev("seer_result", 3, {"target": 1, "is_wolf": True}, "seat:3"),  # someone else's
        ev("speech", 4, {}, VIS_ALL),
    ]
    assert human_tags(events, 2) == {5: "狼隊友", 4: "查驗：好人"}
    assert human_tags(events, 3) == {1: "查驗：狼人"}
    assert human_tags(events, 6) == {}


# ------------------------------------------------------- hardening tests
class ScriptedNights(StubBackend):
    """seed 5 (1狼 2狼 3女巫 4村民* 5預言家 6村民): night 1 wolves kill 6 and the witch saves;
    on day 1 every AI votes `exile`."""

    def __init__(self, exile):
        super().__init__(0)
        self.exile = exile

    async def wolf_target(self, state, seat, options):
        await self._enter()
        return {"target": 6 if state["day"] == 1 else options[0], "thought": "刀"}

    async def witch(self, state, seat, killed, can_save, poison_options):
        await self._enter()
        return {"action": "save" if can_save else "none", "target": None, "thought": "救"}

    async def exile_vote(self, state, seat, options):
        await self._enter()
        return {"target": self.exile if state["day"] == 1 and self.exile in options else None, "thought": "投"}


@pytest.mark.parametrize("exile", [1, 6, 3, 5])  # wolf, villager, witch, seer
def test_event_ids_do_not_count_hidden_events(exile):
    """The id gap across night 2 used to equal the number of hidden events that night, which told
    a villager whether the day-1 exile was a wolf, a villager, the witch or the seer."""
    app = create_app(backend=ScriptedNights(exile))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        conn = Conn(ws)
        conn.send({"type": "new_game", "seed": 5})
        h = Human(random.Random(0), invalid_rate=0.0)
        h.play(conn)
        assert h.you == {"seat": 4, "role": "villager", "role_zh": "村民"}
        ids = [e["id"] for e in h.events]
        assert ids == list(range(len(ids)))
        night2 = next(e for e in h.events if e["text"].startswith("第2夜"))
        dawn2 = next(e for e in h.events if e["kind"] == "death" and e["day"] == 2)
        assert dawn2["id"] == night2["id"] + 1


@pytest.mark.parametrize("bad", ["\ud83d", "前\udfff後"])
def test_lone_surrogate_is_rejected(client, bad):
    """A lone UTF-16 surrogate used to be stored, then every send to the client failed for good."""
    _ws, conn = new_game(client, 7)
    h = Human(random.Random(7), invalid_rate=0.0)
    while True:
        h.play(conn, max_prompts=len(h.prompts) + 1)
        p = h.pending
        if p["type"] == "speak":
            break
        h.answer(conn)
    value = {"text": bad}
    conn.send_text(json.dumps({"type": "action", "value": value}))
    msg = conn.recv()
    assert msg["type"] == "error" and "字元" in msg["message"]
    h.answer(conn)
    h.play(conn)
    assert h.review is not None


async def test_client_writer_escapes_unencodable_text_and_closes_dead_socket():
    class FakeWS:
        def __init__(self, fail=False):
            self.sent, self.closed, self.fail = [], False, fail

        async def send_text(self, text):
            if self.fail:
                raise RuntimeError("socket gone")
            text.encode("utf-8")  # what a real socket does; raises on a lone surrogate
            self.sent.append(text)

        async def close(self):
            self.closed = True

    ws = FakeWS()
    c = server.Client(ws)
    c.send({"type": "event", "event": {"text": "壞\ud83d字"}})
    c.send({"type": "status", "text": "還活著"})
    for _ in range(20):
        await asyncio.sleep(0)
    assert [json.loads(t) for t in ws.sent] == [{"type": "event", "event": {"text": "壞\ud83d字"}},
                                                {"type": "status", "text": "還活著"}]
    assert not c.closed
    c.close()

    dead = FakeWS(fail=True)
    c = server.Client(dead)
    c.send({"type": "status", "text": "x"})
    for _ in range(20):
        await asyncio.sleep(0)
    assert c.closed and dead.closed  # the browser sees the close, reconnects and resumes


def test_resend_after_failed_run_retries_it(client, monkeypatch):
    """After RUN_ERROR, sending again (the panel is re-enabled) retries the run instead of saying
    「請稍候」 forever; the answer given before is kept in the checkpoint."""
    real = rules.coerce_speech
    boom = {"left": 1}

    def flaky(value):
        if isinstance(value, dict) and value.get("text") == "BOOMONCE" and boom["left"]:
            boom["left"] -= 1
            raise RuntimeError("node bug")
        return real(value)

    monkeypatch.setattr(rules, "coerce_speech", flaky)
    _ws, conn = new_game(client, 7)
    h = Human(random.Random(7), invalid_rate=0.0)
    while True:
        h.play(conn, max_prompts=len(h.prompts) + 1)
        if h.pending["type"] == "speak":
            break
        h.answer(conn)
    h.pending = None
    conn.send({"type": "action", "value": {"text": "BOOMONCE"}})
    msgs = [conn.recv(), conn.recv()]
    assert msgs == [{"type": "status", "text": ""}, {"type": "error", "message": server.RUN_ERROR}]
    conn.send({"type": "action", "value": {"text": "this value is ignored"}})
    h.play(conn)
    mine = [e["text"] for e in h.events if e["seat"] == h.you["seat"] and e["kind"] in SPEECH_KINDS.values()]
    assert mine.count("BOOMONCE") == 1 and "this value is ignored" not in mine


def test_finished_game_drops_checkpoints(client, app):
    with client.websocket_connect("/ws") as ws:
        conn = Conn(ws)
        conn.send({"type": "new_game", "seed": 3})
        h = Human(random.Random(3), invalid_rate=0.0)
        h.play(conn)
        assert conn.call(graph_values, app, h.game_id) == {}
        assert app.state.manager.sessions[h.game_id].review == h.review  # still resumable


def test_idle_sessions_are_evicted(client, app, monkeypatch):
    monkeypatch.setattr(server, "MAX_SESSIONS", 2)
    manager = app.state.manager
    ids = []
    for seed in (1, 2):
        ws, conn = new_game(client, seed)
        h = Human(random.Random(seed))
        h.play(conn, max_prompts=1)  # paused at a prompt
        ids.append(h.game_id)
        close_ws(ws)
    assert set(manager.sessions) == set(ids)
    ws, conn = new_game(client, 3)  # third game: the least recently used idle one goes
    h = Human(random.Random(3))
    h.play(conn, max_prompts=1)
    assert set(manager.sessions) == {ids[1], h.game_id}
    assert conn.call(graph_values, app, ids[0]) == {}
    conn.send({"type": "resume_game", "game_id": ids[0]})
    assert conn.recv()["type"] == "error"

    monkeypatch.setattr(server, "MAX_SESSIONS", 100)
    monkeypatch.setattr(server, "SESSION_TTL_S", 0)  # every idle game is now expired
    close_ws(ws)
    ws, conn = new_game(client, 4)
    msg = conn.recv()
    assert msg["type"] == "game_started" and set(manager.sessions) == {msg["game_id"]}


class GatedVotes(StubBackend):
    """AI exile votes wait until `vote_gate` is set; the other calls don't."""

    def __init__(self, seed=0):
        super().__init__(seed)
        self.vote_gate = asyncio.Event()

    async def exile_vote(self, state, seat, options):
        await self.vote_gate.wait()
        return await super().exile_vote(state, seat, options)


@pytest.mark.parametrize("backend", [GatedVotes(1234)])
def test_vote_answered_before_the_ai_votes_are_in(client, app, backend):
    """The vote prompt comes while the AI voters are still deciding; an answer sent then is kept
    until they are done and counted with their votes."""
    _ws, conn = new_game(client, 10)  # a villager who votes on day 1
    h = Human(random.Random(10), invalid_rate=0.0)
    while True:
        h.play(conn, max_prompts=len(h.prompts) + 1)
        assert h.pending is not None, "the human was never asked to vote"
        if h.pending.get("action") == "exile_vote":
            break
        h.answer(conn)
    session = app.state.manager.sessions[h.game_id]
    day = h.pending["day"]
    h.answer(conn)
    vote = h.answers[-1][1]["target"]

    async def queued():
        with anyio.fail_after(TIMEOUT):
            while session.queued is None:
                await asyncio.sleep(0.01)

    conn.call(queued)
    assert session.running and session.pending is None
    conn.call(backend.vote_gate.set)
    h.play(conn)
    check_finished_game(h, final_values(app, h.game_id))
    result = next(e for e in h.events if e["kind"] == "vote_result" and e["day"] == day)
    mine = next(v["target"] for v in result["data"]["votes"] if v["voter"] == h.you["seat"])
    assert mine == vote


def play_to_the_end(client, app, seed):
    """Play one game through a socket; -> (Human, session) once its run (and the saving) is done."""
    with client.websocket_connect("/ws") as ws:
        conn = Conn(ws)
        conn.send({"type": "new_game", "seed": seed})
        h = Human(random.Random(seed), invalid_rate=0.0)
        h.play(conn)
        session = app.state.manager.sessions[h.game_id]

        async def run_done():
            with anyio.fail_after(TIMEOUT):
                await session.task

        conn.call(run_done)
    return h, session


def test_finished_game_is_saved(client, app):
    """One JSON file per finished game: the god's-eye view the review shows, plus how it was played."""
    h, session = play_to_the_end(client, app, 3)
    files = list(config.GAMES_DIR.glob("*.json"))
    assert len(files) == 1 and files[0].name.endswith(f"_{h.game_id}.json")
    record = json.loads(files[0].read_text(encoding="utf-8"))
    assert {k: record[k] for k in REVIEW_KEYS} == h.review
    assert (record["game_id"], record["seed"]) == (h.game_id, 3)
    assert record["started_at"] <= record["ended_at"]
    assert "token_usage" in record  # None here: the test backend doesn't count tokens
    assert "token_usage" in record  # None here: the test backend doesn't count tokens
    assert any(e["kind"] == "thought" for e in record["events"])  # every AI's thoughts too


def test_game_ends_normally_when_it_cannot_be_saved(client, app, monkeypatch, tmp_path):
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x")
    monkeypatch.setattr(config, "GAMES_DIR", blocker)
    h, session = play_to_the_end(client, app, 4)
    assert h.review is not None and not session.failed
