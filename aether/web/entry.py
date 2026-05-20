"""Command-line entrypoint for the local Aether web console."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from aether.web.app import create_app


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9120


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aether-web", description="Run the Aether web console")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"host to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to bind (default: {DEFAULT_PORT})")
    parser.add_argument("--reload", action="store_true", help="enable uvicorn reload")
    parser.add_argument("--web-dist", type=Path, default=None, help="path to built web/dist assets")
    parser.add_argument("--no-open", action="store_true", help="reserved for future browser opening behavior")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    app = create_app(bound_host=args.host, web_dist=args.web_dist)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
