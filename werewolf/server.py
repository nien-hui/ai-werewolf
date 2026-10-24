"""FastAPI app: static frontend + WebSocket game sessions.

One GameSession per game id, one shared compiled graph (built lazily on the
first new game, so importing this module never builds the AI backend). Each
session runs the graph in a single background task; outgoing messages go
through a per-socket queue so broadcasting never blocks and per-socket order
is preserved (a resume replays history atomically before any live message).
With config.PASSWORD set, a socket must carry it (/ws?key=…) or it is closed
with AUTH_CLOSE before it can do anything.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command

from . import config
from .graph import build_graph
from .personas import persona_name
from .rules import WITCH_ACTIONS
from .state import ROLE_ZH, Event, Player, visible_to

log = logging.getLogger(__name__)

STATIC_DIR = config.ROOT / "static"
# The page and its JS/CSS are revalidated on every load (ETag/Last-Modified → 304), so a browser
# never pairs a heuristically cached index.html with a newer app.js / style.css.
NO_CACHE = {"Cache-Control": "no-cache"}
MAX_SEED = 2**53 - 1  # JS safe integer; also keeps the seed msgpack-friendly
PLAYER_KEYS = ("players", "day", "phase")  # node-update keys that trigger a `players` message
SESSION_TTL_S = 6 * 3600  # a game nobody is connected to is forgotten after this long idle
MAX_SESSIONS = 100  # beyond this, the least recently used idle games are forgotten first
AUTH_CLOSE = 4401  # websocket close code: missing or wrong password

RUN_ERROR = "遊戲執行時發生錯誤，請再送出一次，或重新整理頁面後按「繼續上一局」。"
TAG_MATE = "狼隊友"
TAG_WOLF = "查驗：狼人"
TAG_GOOD = "查驗：好人"


class ClientError(ValueError):
    """Bad client message/action; str(error) is shown to the player (zh-TW)."""


def _is_seat(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_text(text: str) -> None:
    """Reject text the socket could never send back (e.g. a lone UTF-16 surrogate)."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise ClientError("內容含有無法辨識的字元，請重新輸入。") from None


def _dumps(msg: dict, indent: Optional[int] = None) -> str:
    text = json.dumps(msg, ensure_ascii=False, indent=indent)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate (e.g. odd model output): send it \u-escaped
        text = json.dumps(msg, indent=indent)
    return text


def _write_json(path: Any, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dumps(data, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------ pure helpers
def validate_action(prompt: dict, value: Any) -> dict:
    """Check an action against the pending interrupt payload.

    Returns the cleaned, non-empty resume value to forward to the graph
    (never `{}`: LangGraph treats an empty dict as an empty resume map).
    Raises ClientError when the action is not legal for this prompt.
    """
    if not isinstance(value, dict):
        raise ClientError("行動格式錯誤。")
    kind = prompt.get("type")
    if kind == "speak":
        text = value.get("text")
        if not isinstance(text, str):
            raise ClientError("發言內容必須是文字。")
        if len(text.strip()) > config.MAX_SPEECH_CHARS:
            raise ClientError(f"發言太長了（最多 {config.MAX_SPEECH_CHARS} 字）。")
        _check_text(text)
        return {"text": text}

    if kind == "choose":
        if "target" not in value:
            raise ClientError("請選擇一位玩家。")
        target = value["target"]
        if target is None:
            if not prompt.get("allow_skip"):
                raise ClientError("這次不能棄票，請選擇一位玩家。")
        elif not _is_seat(target) or target not in (prompt.get("options") or []):
            raise ClientError("這個選項不合法，請從可選的玩家中選擇。")
        return {"target": target}

    if kind == "witch":
        action, target = value.get("action"), value.get("target")
        if action not in WITCH_ACTIONS:
            raise ClientError("無效的女巫行動。")
        if action == "save":
            killed = prompt.get("killed")
            if not prompt.get("can_save") or killed is None:
                raise ClientError("你現在不能使用解藥。")
            if target is not None and not (_is_seat(target) and target == killed):
                raise ClientError("解藥只能用在今晚被殺害的玩家身上。")
            return {"action": "save"}
        if action == "poison":
            options = prompt.get("poison_options") or []
            if not options:
                raise ClientError("你的毒藥已經用完了。")
            if not _is_seat(target) or target not in options:
                raise ClientError("請選擇一位可以下毒的玩家。")
            return {"action": "poison", "target": target}
        if target is not None:
            raise ClientError("不使用藥水時不能指定目標。")
        return {"action": "none"}

    raise ClientError("無法辨識目前的行動類型。")


def human_tags(visible_events: list[Event], human_seat: int) -> dict[int, str]:
    """Seat tags from what the human knows: wolf teammates, own seer results."""
    tags: dict[int, str] = {}
    for e in visible_events:
        if e.get("seat") != human_seat:
            continue
        data = e.get("data") or {}
        if e["kind"] == "role_assign":
            for mate in data.get("teammates") or []:
                tags[mate] = TAG_MATE
        elif e["kind"] == "seer_result" and _is_seat(data.get("target")):
            tags[data["target"]] = TAG_WOLF if data.get("is_wolf") else TAG_GOOD
    return tags


def build_review(values: dict) -> dict:
    """game_end review: god's-eye view of the finished game."""
    players: list[Player] = values.get("players") or []
    return {
        "players": [
            {
                "seat": p["seat"],
                "role": p["role"],
                "role_zh": ROLE_ZH.get(p["role"], p["role"]),
                "is_human": p["is_human"],
                "persona": p["persona"],
                "persona_name": persona_name(p["persona"]),
                "alive": p["alive"],
            }
            for p in players
        ],
        "events": list(values.get("events") or []),
        "winner": values.get("winner") or "draw",
        "win_reason": values.get("win_reason") or "",
        "human_seat": values.get("human_seat"),
    }


# ------------------------------------------------------------------ client
class Client:
    """One WebSocket. send() only enqueues; a writer task drains the queue."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.session: Optional[GameSession] = None
        self.closed = False
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._writer = asyncio.create_task(self._write_loop())

    def send(self, msg: dict) -> None:
        if not self.closed:
            self._queue.put_nowait(msg)

    async def _write_loop(self) -> None:
        try:
            while True:
                msg = await self._queue.get()
                await self.ws.send_text(_dumps(msg))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - dead socket
            log.info("websocket send failed (%s); dropping client", type(exc).__name__)
            self.close()
            with contextlib.suppress(Exception):  # so the browser notices, reconnects and resumes
                await self.ws.close()

    def attach(self, session: GameSession) -> None:
        if self.session is not session:
            self.detach()
            self.session = session
        session.clients.add(self)
        session.touch()

    def detach(self) -> None:
        if self.session is not None:
            self.session.clients.discard(self)
            self.session = None

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.detach()
        if self._writer is not asyncio.current_task():
            self._writer.cancel()


def build_record(game_id: str, seed: Optional[int], values: dict, review: dict,
                 started_at: datetime, ended_at: datetime) -> dict:
    """The saved JSON of a finished game: the review (the god's-eye view: every role, persona,
    event and AI thought), when / how it was played and the AIs' token usage."""
    return {
        "game_id": game_id,
        "seed": seed,
        "model": "fake" if config.FAKE_LLM else config.MODEL_NAME,
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds"),
        "token_usage": values.get("token_usage"),
        **{k: review.get(k) for k in ("winner", "win_reason", "human_seat", "players", "events")},
    }


def record_path(games_dir: Path, game_id: str, ended_at: datetime) -> Path:
    return games_dir / f"{ended_at:%Y%m%d-%H%M%S}_{game_id}.json"


# ----------------------------------------------------------------- session
class GameSession:
    """One game: graph thread, human info, pending prompt, all events, sockets."""

    def __init__(self, manager: GameManager, game_id: str, seed: Optional[int]):
        self.manager = manager
        self.game_id = game_id
        self.seed = seed
        self.cfg = {"configurable": {"thread_id": game_id}}
        self.clients: set[Client] = set()
        self.events: list[Event] = []  # every event (incl. god/private), id order
        # What the human has been sent, with ids renumbered 0, 1, 2, ...: the global ids also
        # count hidden events (wolf chat, thoughts, ...), so their gaps would give away how
        # many hidden events each night had, i.e. which roles are still alive.
        self.visible: list[Event] = []
        self._ids: set[int] = set()
        self.players: list[Player] = []
        self.human_seat: Optional[int] = None
        self.human_role: Optional[str] = None
        self.day = 0
        self.phase = "setup"
        self.pending: Optional[dict] = None  # interrupt payload waiting for the human
        self.queued: Any = None  # Command(resume=...) answered before the run that asked was over
        self.review: Optional[dict] = None  # set once the game is over
        self.final: dict = {}  # the graph's final values, once the game is over (save_record)
        self.status = ""  # last transient status while the graph runs
        self.failed = False  # last run raised; the next action or resume_game retries it
        self.task: Optional[asyncio.Task] = None
        self.touched = time.monotonic()  # last activity (eviction, see GameManager.evict)
        self.started_at = datetime.now().astimezone()
        self._lock = asyncio.Lock()
        self._last_players: Optional[dict] = None

    # state -------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def can_see(self, ev: Event) -> bool:
        if self.human_seat is None:
            return False
        return visible_to(ev, self.human_seat, self.human_role or "", include_god=False)

    def touch(self) -> None:
        self.touched = time.monotonic()

    # messages ----------------------------------------------------------
    def broadcast(self, msg: dict) -> None:
        for c in list(self.clients):
            if c.closed:
                self.clients.discard(c)
            else:
                c.send(msg)

    def started_msg(self) -> dict:
        return {
            "type": "game_started",
            "game_id": self.game_id,
            "you": {"seat": self.human_seat, "role": self.human_role,
                    "role_zh": ROLE_ZH.get(self.human_role or "", "")},
            "num_players": len(self.players),
        }

    def players_msg(self) -> dict:
        tags = human_tags(self.visible, self.human_seat) if self.human_seat is not None else {}
        return {
            "type": "players",
            "day": self.day,
            "phase": self.phase,
            "players": [
                {"seat": p["seat"], "alive": p["alive"], "is_you": p["seat"] == self.human_seat,
                 "tag": tags.get(p["seat"])}
                for p in self.players
            ],
        }

    def replay(self, client: Client) -> None:
        """Re-send everything the human may know (resume_game). Call right after attach()."""
        if self.human_seat is not None:
            client.send(self.started_msg())
            client.send(self.players_msg())
        for ev in self.visible:
            client.send({"type": "event", "event": ev})
        if self.review is not None:
            client.send({"type": "game_end", "review": self.review})
        elif self.pending is not None:
            client.send({"type": "prompt", "prompt": self.pending})
        elif self.running and self.status:
            client.send({"type": "status", "text": self.status})

    # actions -----------------------------------------------------------
    def act(self, value: Any) -> None:
        if self.review is not None:
            raise ClientError("這局遊戲已經結束了。")
        if self.failed and not self.running:
            # the last run crashed (RUN_ERROR asks to send again): retry it. The checkpoint
            # already holds the answer given before, so this value is not needed.
            log.info("game %s: retrying after a failed run", self.game_id)
            self.start(None)
            return
        if self.pending is None:
            raise ClientError("現在還沒輪到你行動，請稍候。")
        resume = validate_action(self.pending, value)
        self.pending = None
        if self.running:  # asked while other voters' runs were still going: resume once they are done
            self.queued = Command(resume=resume)
        else:
            self.start(Command(resume=resume))

    # graph runner ------------------------------------------------------
    def start(self, inp: Any) -> None:
        if self.running:
            raise RuntimeError(f"game {self.game_id} is already running")
        self.failed = False
        self.touch()
        self.task = asyncio.create_task(self._run(inp), name=f"werewolf-{self.game_id}")

    async def _run(self, inp: Any) -> None:
        async with self._lock:  # never two graph runs for one game
            graph = self.manager.graph
            try:
                if inp is None:  # retry after a failed run: catch up from the checkpoint first
                    self._sync((await graph.aget_state(self.cfg)).values or {})
                while True:
                    stream = graph.astream(inp, self.cfg, stream_mode=["updates", "custom"])
                    async with contextlib.aclosing(stream):
                        async for mode, chunk in stream:
                            if mode == "custom":
                                self._on_custom(chunk)
                            elif isinstance(chunk, dict):
                                for node, update in chunk.items():
                                    if node == "__interrupt__":
                                        self._on_interrupt(update)
                                    # skip "__metadata__" and any non-dict update
                                    elif not node.startswith("__") and isinstance(update, dict):
                                        self._on_update(update)
                    self._finish(await graph.aget_state(self.cfg))
                    inp, self.queued = self.queued, None  # answered during the run: go on with it
                    if inp is None:
                        break
                if self.review is not None:  # everything needed from now on is in the review
                    await self.drop_checkpoints()
                    await self.save_record()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("game %s: graph run failed", self.game_id)
                self.failed = True
                self.pending = self.queued = None  # the retry asks again
                self.status = ""
                self.broadcast({"type": "status", "text": ""})
                self.broadcast({"type": "error", "message": RUN_ERROR})

    def _on_interrupt(self, interrupts: Any) -> None:
        """A node asks the human: send the prompt now. The rest of its step (the AI voters' runs)
        may still be running; an answer given meanwhile waits in `queued` (see act)."""
        if self.pending is None and self.queued is None and isinstance(interrupts, tuple) and interrupts:
            self.pending = interrupts[0].value
            self.broadcast({"type": "prompt", "prompt": self.pending})

    def _on_custom(self, chunk: Any) -> None:
        if isinstance(chunk, dict) and chunk.get("type") == "status" and isinstance(chunk.get("text"), str):
            self.status = chunk["text"]
            self.broadcast({"type": "status", "text": self.status})

    def _apply_fields(self, update: dict) -> None:
        if isinstance(update.get("players"), list):
            self.players = update["players"]
            if self.human_seat is None:
                human = next((p for p in self.players if p.get("is_human")), None)
                if human is not None:
                    self.human_seat, self.human_role = human["seat"], human["role"]
                    self.broadcast(self.started_msg())
        if "day" in update:
            self.day = update["day"]
        if "phase" in update:
            self.phase = update["phase"]

    async def drop_checkpoints(self) -> None:
        """Free the graph thread (every superstep's checkpoint copies the whole event list)."""
        try:
            await self.manager.graph.checkpointer.adelete_thread(self.game_id)
        except Exception:  # noqa: BLE001
            log.exception("game %s: could not delete its checkpoints", self.game_id)

    async def save_record(self) -> None:
        """Write the finished game to config.GAMES_DIR, one JSON file per game (see build_record).
        A failure is only logged."""
        ended = datetime.now().astimezone()
        record = build_record(self.game_id, self.seed, self.final, self.review or {}, self.started_at, ended)
        path = record_path(config.GAMES_DIR, self.game_id, ended)
        try:
            await asyncio.to_thread(_write_json, path, record)
            log.info("game %s: saved to %s", self.game_id, path)
        except Exception:  # noqa: BLE001
            log.exception("game %s: could not save the game record", self.game_id)

    def _add_event(self, ev: Any) -> None:
        if not isinstance(ev, dict) or ev.get("id") in self._ids:
            return
        self._ids.add(ev.get("id"))
        self.events.append(ev)
        if self.can_see(ev):
            out = {**ev, "id": len(self.visible)}
            self.visible.append(out)
            self.broadcast({"type": "event", "event": out})

    def _maybe_send_players(self, force: bool) -> None:
        if self.human_seat is None:
            return
        msg = self.players_msg()
        if force or msg != self._last_players:  # also when a tag changes (e.g. new seer result)
            self._last_players = msg
            self.broadcast(msg)

    def _on_update(self, update: dict) -> None:
        self._apply_fields(update)
        for ev in update.get("events") or []:
            self._add_event(ev)
        self._maybe_send_players(force=any(k in update for k in PLAYER_KEYS))

    def _sync(self, values: dict) -> None:
        """Catch up with checkpointed state (normally a no-op; heals a run that died mid-stream)."""
        self._apply_fields(values)
        for ev in values.get("events") or []:
            self._add_event(ev)
        self._maybe_send_players(force=False)

    def _finish(self, snap: Any) -> None:
        values = snap.values or {}
        self._sync(values)
        self.status = ""
        self.touch()
        if snap.interrupts:
            if len(snap.interrupts) > 1:
                log.warning("game %s: %d pending interrupts, using the first", self.game_id, len(snap.interrupts))
            if self.pending is None and self.queued is None:  # else already sent (or answered)
                self.pending = snap.interrupts[0].value
                self.broadcast({"type": "prompt", "prompt": self.pending})
        elif not snap.next and values.get("players"):
            self.final = values
            self.review = build_review(values)
            self.broadcast({"type": "game_end", "review": self.review})
        else:
            log.error("game %s: graph stopped without an interrupt (next=%s)", self.game_id, snap.next)
            self.failed = True
            self.broadcast({"type": "error", "message": RUN_ERROR})


# ----------------------------------------------------------------- manager
class GameManager:
    """All sessions plus the shared compiled graph (built on first use)."""

    def __init__(self, backend: Any = None):
        self._backend = backend
        self._graph: Any = None
        self._build_lock = asyncio.Lock()
        self.sessions: dict[str, GameSession] = {}

    @property
    def graph(self) -> Any:
        if self._graph is None:
            self._graph = build_graph(backend=self._backend)  # backend None -> werewolf.ai.get_backend()
        return self._graph

    async def ensure_graph(self) -> Any:
        """Build the graph off the event loop (importing/creating the Gemini client takes ~1 s)."""
        if self._graph is None:
            async with self._build_lock:
                if self._graph is None:
                    self._graph = await asyncio.to_thread(build_graph, backend=self._backend)
        return self._graph

    async def handle(self, client: Client, msg: Any) -> None:
        if not isinstance(msg, dict):
            raise ClientError("無法辨識的訊息格式。")
        kind = msg.get("type")
        if kind == "new_game":
            await self.new_game(client, msg.get("seed"))
        elif kind == "resume_game":
            self.resume_game(client, msg.get("game_id"))
        elif kind == "action":
            if client.session is None:
                raise ClientError("請先開始一局遊戲。")
            client.session.act(msg.get("value"))
        else:
            raise ClientError("無法辨識的訊息類型。")

    async def new_game(self, client: Client, seed: Any) -> None:
        if seed is not None and not (_is_seat(seed) and abs(seed) <= MAX_SEED):
            raise ClientError("seed 必須是整數。")
        try:
            await self.ensure_graph()
        except Exception:  # noqa: BLE001
            log.exception("could not build the game graph / AI backend")
            raise ClientError("無法啟動 AI 玩家，請檢查伺服器設定。") from None
        await self.evict()
        game_id = uuid.uuid4().hex
        session = GameSession(self, game_id, seed)
        self.sessions[game_id] = session
        client.attach(session)
        log.info("new game %s (seed=%s)", game_id, seed)
        session.start({"game_id": game_id, "seed": seed, "events": []})

    def resume_game(self, client: Client, game_id: Any) -> None:
        session = self.sessions.get(game_id) if isinstance(game_id, str) else None
        if session is None:
            raise ClientError("找不到這局遊戲（伺服器可能已重新啟動）。")
        client.attach(session)
        session.replay(client)  # no await between attach and replay: history precedes live messages
        if session.failed and not session.running:
            log.info("game %s: retrying after a failed run", game_id)
            session.start(None)

    async def evict(self) -> None:
        """Forget games nobody is connected to that are not running: those idle for SESSION_TTL_S,
        and the least recently used ones while there are MAX_SESSIONS or more games."""
        now = time.monotonic()
        idle = sorted((s for s in self.sessions.values() if not s.running and not s.clients),
                      key=lambda s: s.touched)
        for s in idle:
            if now - s.touched < SESSION_TTL_S and len(self.sessions) < MAX_SESSIONS:
                break
            if self.sessions.get(s.game_id) is not s or s.running or s.clients:
                continue  # changed while an earlier eviction was awaiting
            del self.sessions[s.game_id]
            log.info("game %s: evicted (idle %.0f s)", s.game_id, now - s.touched)
            if s.review is None:  # finished games already dropped theirs
                await s.drop_checkpoints()

    async def shutdown(self) -> None:
        tasks = [s.task for s in self.sessions.values() if s.running]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# --------------------------------------------------------------------- app
def create_app(backend: Any = None) -> FastAPI:
    """backend None -> werewolf.ai.get_backend(), resolved when the first game starts."""
    manager = GameManager(backend)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await manager.shutdown()

    app = FastAPI(title="AI 狼人殺", lifespan=lifespan)
    app.state.manager = manager

    @app.get("/", include_in_schema=False)
    async def index():
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            return PlainTextResponse("static/index.html not found", status_code=404)
        return FileResponse(page, headers=NO_CACHE)

    @app.get("/api/health")
    async def health():
        return {"ok": True, "model": config.MODEL_NAME, "fake": config.FAKE_LLM}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        key = ws.query_params.get("key", "")
        if config.PASSWORD and not secrets.compare_digest(key.encode("utf-8", "replace"),
                                                          config.PASSWORD.encode("utf-8")):
            await ws.close(code=AUTH_CLOSE)
            return
        client = Client(ws)
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                raw = message.get("text")
                if raw is None and message.get("bytes") is not None:
                    raw = message["bytes"].decode("utf-8", "replace")
                try:
                    msg = json.loads(raw) if raw is not None else None
                except ValueError:
                    msg = None
                try:
                    await manager.handle(client, msg)
                except ClientError as exc:
                    client.send({"type": "error", "message": str(exc)})
                except Exception:  # noqa: BLE001
                    log.exception("websocket message handling failed")
                    client.send({"type": "error", "message": "伺服器處理訊息時發生錯誤。"})
        except Exception as exc:  # noqa: BLE001 - abrupt disconnect; the game keeps running
            log.debug("websocket closed: %r", exc)
        finally:
            client.close()

    app.mount("/static", FrontendFiles(directory=STATIC_DIR), name="static")
    return app


class FrontendFiles(StaticFiles):
    """StaticFiles that sends NO_CACHE (see above) with every file and 304."""

    async def get_response(self, path: str, scope: Any) -> Any:
        response = await super().get_response(path, scope)
        response.headers.update(NO_CACHE)
        return response


app = create_app()
