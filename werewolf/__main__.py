"""`uv run python -m werewolf [--host H] [--port P] [--fake]` → uvicorn."""
from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from . import config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m werewolf", description="AI 狼人殺 伺服器")
    parser.add_argument("--host", default="127.0.0.1", help="listen address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="listen port (default: 8000)")
    parser.add_argument("--fake", action="store_true",
                        help="offline random AI players (no Gemini calls); same as WEREWOLF_FAKE_LLM=1")
    args = parser.parse_args(argv)

    if args.fake:  # must happen before the first game builds the backend
        os.environ["WEREWOLF_FAKE_LLM"] = "1"
        config.FAKE_LLM = True
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from .server import app

    if config.FAKE_LLM:
        mode = "FakeBackend (offline)"
    elif config.get_api_key():
        mode = config.MODEL_NAME
    else:
        mode = "FakeBackend (no Gemini API key: set gemini_key in .env)"
    print(f"AI 狼人殺 AI players: {mode} → http://{args.host}:{args.port}/", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
