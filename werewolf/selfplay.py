"""`uv run python -m werewolf.selfplay [--games N] [--concurrency C] [--name NAME] [--persona P] [--fake]`

All-AI self-play: every seat is an AI (no human seat, no waiting for the page's typing), several
games at a time. By default every AI plays without a personality (persona NO_PERSONA); --persona
gives every AI one persona key, or `mix` deals the real personalities out like a web game. Each run gets its own folder, games/ai_only/<NAME or start time>/: a copy of the
prompt files it played with (prompts/) and every finished game, saved like a web game (see
server.build_record). A summary line is printed per game.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from . import config

PROMPTS_DIR = Path(__file__).parent / "prompts" / "text"
NO_PERSONA = "plain"  # prompts/text/personas/plain.md: no personality section in the prompt
MIX = "mix"  # --persona mix: every persona but NO_PERSONA, all different within a game while they last
log = logging.getLogger("werewolf.selfplay")


async def play_one(graph: Any, seed: Optional[int], out_dir: Path, persona: str = NO_PERSONA) -> dict:
    """Play one all-AI game to the end and save it. Returns its record."""
    from .personas import PERSONA_KEYS
    from .server import _write_json, build_record, build_review, record_path

    if persona == MIX:
        deal = {"persona": None, "persona_pool": [k for k in PERSONA_KEYS if k != NO_PERSONA]}
    else:
        deal = {"persona": persona}

    game_id = uuid.uuid4().hex
    cfg = {"configurable": {"thread_id": game_id}}
    started = datetime.now().astimezone()
    try:
        await graph.ainvoke(
            {"game_id": game_id, "seed": seed, "all_ai": True, **deal, "events": []}, cfg)
        snap = await graph.aget_state(cfg)
    finally:
        await graph.checkpointer.adelete_thread(game_id)
    values = snap.values or {}
    if snap.next or snap.interrupts or not values.get("winner"):
        raise RuntimeError(f"game {game_id} stopped before the end (next={snap.next})")
    ended = datetime.now().astimezone()
    record = build_record(game_id, seed, values, build_review(values), started, ended)
    path = record_path(out_dir, game_id, ended)
    await asyncio.to_thread(_write_json, path, record)
    record["path"] = str(path)
    record["days"] = values.get("day", 0)
    return record


async def run(games: int, concurrency: int, out_dir: Path, seed: Optional[int], persona: str = NO_PERSONA) -> int:
    """Play `games` games, `concurrency` at a time (game i gets seed+i, or a random seed when seed is
    None) and save them to out_dir. Returns 0 if none failed."""
    from .graph import build_graph

    graph = build_graph()
    sem = asyncio.Semaphore(concurrency)
    winners: Counter = Counter()
    tokens: list[int] = []
    failed = 0
    done = 0
    t0 = time.monotonic()

    async def one(i: int) -> None:
        nonlocal failed, done
        async with sem:
            try:
                rec = await play_one(graph, None if seed is None else seed + i, out_dir, persona)
            except Exception:  # noqa: BLE001
                failed += 1
                log.exception("game #%d failed", i + 1)
                return
            done += 1
            winners[rec["winner"]] += 1
            usage = rec.get("token_usage") or {}
            if usage.get("total"):
                tokens.append(usage["total"])
            print(f"[{done + failed}/{games}] #{i + 1} winner={rec['winner']} days={rec['days']} "
                  f"tokens={usage.get('total', '-')} ({rec['win_reason']}) → {rec['path']}", flush=True)

    await asyncio.gather(*(one(i) for i in range(games)))
    mins = (time.monotonic() - t0) / 60
    print(f"\nfinished {done}/{games} games in {mins:.1f} min ({failed} failed): "
          + ", ".join(f"{k} {v}" for k, v in sorted(winners.items())), flush=True)
    if tokens:
        print(f"tokens: total {sum(tokens)}, per game avg {sum(tokens) // len(tokens)}, max {max(tokens)}",
              flush=True)
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m werewolf.selfplay", description="全 AI 狼人殺 自我對戰")
    parser.add_argument("--games", type=int, default=50, help="how many games (default: 50)")
    parser.add_argument("--concurrency", type=int, default=5, help="games played at the same time (default: 5)")
    parser.add_argument("--name", help="this run's folder under games/ai_only/ (default: the start time)")
    parser.add_argument("--seed", type=int, default=None, help="game i gets seed+i (default: random)")
    parser.add_argument("--persona", default=NO_PERSONA,
                        help=f"every AI's persona key, or {MIX!r} for a mix of the personalities (default: {NO_PERSONA})")
    parser.add_argument("--fake", action="store_true", help="offline random AI players (no Gemini calls)")
    args = parser.parse_args(argv)

    from .personas import PERSONA_KEYS
    if args.persona != MIX and args.persona not in PERSONA_KEYS:
        sys.exit(f"unknown --persona {args.persona!r}: pick one of {', '.join(PERSONA_KEYS)} or {MIX}")
    if args.fake:  # must happen before the graph builds the backend
        os.environ["WEREWOLF_FAKE_LLM"] = "1"
        config.FAKE_LLM = True
    elif not config.get_api_key():
        sys.exit("No Gemini API key (set gemini_key in .env), or use --fake")
    out = config.GAMES_DIR / "ai_only" / (args.name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    if out.exists():
        sys.exit(f"{out} already exists: pick another --name")
    shutil.copytree(PROMPTS_DIR, out / "prompts")  # what the AIs read in this run
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(f"self-play: {args.games} games, {config.NUM_PLAYERS} AI players, {args.concurrency} at a time, "
          f"persona {args.persona}, model {'fake' if config.FAKE_LLM else config.MODEL_NAME} → {out}", flush=True)
    sys.exit(asyncio.run(run(args.games, max(1, args.concurrency), out, args.seed, args.persona)))


if __name__ == "__main__":
    main()
