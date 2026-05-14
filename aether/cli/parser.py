"""Argument parser shared by the Python CLI entry points."""

from __future__ import annotations

import argparse
import os
from typing import Final

LOG_LEVELS: Final[tuple[str, ...]] = (
    "debug",
    "info",
    "warning",
    "warn",
    "error",
    "critical",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aether",
        description="Aether - interactive AI agent harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    chat = subparsers.add_parser("chat", help="Start interactive TUI (default)")
    add_chat_args(chat, suppress_defaults=True)

    _ = subparsers.add_parser("providers", help="List available providers")
    _ = subparsers.add_parser("version", help="Show version + runtime info")

    add_chat_args(parser)
    return parser


def _default(value: object, *, suppress: bool) -> object:
    return argparse.SUPPRESS if suppress else value


def add_chat_args(
    p: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
) -> None:
    _ = p.add_argument(
        "-p",
        "--provider",
        metavar="NAME",
        default=_default(None, suppress=suppress_defaults),
        help="Provider to use: claude, openai, codex (default: openai or $AETHER_PROVIDER)",
    )
    _ = p.add_argument(
        "-m",
        "--model",
        metavar="MODEL",
        default=_default(None, suppress=suppress_defaults),
        help="Model name override (default: provider default or $AETHER_MODEL)",
    )
    _ = p.add_argument(
        "--api-key",
        metavar="KEY",
        default=_default(None, suppress=suppress_defaults),
        help="API key (overrides env var for the selected provider)",
    )
    _ = p.add_argument(
        "--base-url",
        metavar="URL",
        default=_default(None, suppress=suppress_defaults),
        help="Base URL override for OpenAI-compatible providers",
    )
    _ = p.add_argument(
        "--system",
        metavar="PROMPT",
        default=_default(None, suppress=suppress_defaults),
        help="System prompt text",
    )
    _ = p.add_argument(
        "--system-file",
        metavar="FILE",
        default=_default(None, suppress=suppress_defaults),
        help="Path to a file containing the system prompt",
    )
    _ = p.add_argument(
        "--session",
        metavar="ID",
        default=_default(None, suppress=suppress_defaults),
        help="Tag a specific session UUID for the new conversation",
    )
    _ = p.add_argument(
        "--resume",
        metavar="ID",
        nargs="?",
        const="__pick__",
        default=_default(None, suppress=suppress_defaults),
        help=(
            "Resume a previously saved session. Pass an id (or unique prefix) "
            "to jump straight in, or no value for an interactive picker."
        ),
    )
    _ = p.add_argument(
        "--max-iterations",
        type=int,
        default=_default(32, suppress=suppress_defaults),
        metavar="N",
        help="Max agent loop iterations per turn (default: 32)",
    )
    _ = p.add_argument(
        "--temperature",
        type=float,
        default=_default(None, suppress=suppress_defaults),
        metavar="T",
        help="Sampling temperature",
    )
    _ = p.add_argument(
        "--max-tokens",
        type=int,
        default=_default(None, suppress=suppress_defaults),
        metavar="N",
        help="Max tokens per response",
    )
    _ = p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=_default(False, suppress=suppress_defaults),
        help="Show engine metadata (status, iterations, exit reason) after each turn",
    )
    _ = p.add_argument(
        "--no-banner",
        action="store_true",
        default=_default(False, suppress=suppress_defaults),
        help="Skip the welcome banner on startup",
    )
    _ = p.add_argument(
        "--no-builtin-tools",
        action="store_true",
        default=_default(False, suppress=suppress_defaults),
        help=(
            "Disable the bundled tool kit (shell, read_file, write_file, "
            "list_dir, grep, glob) for this session."
        ),
    )
    _ = p.add_argument(
        "--log-level",
        metavar="LEVEL",
        default=_default(os.getenv("AETHER_LOG_LEVEL", "warning"), suppress=suppress_defaults),
        choices=list(LOG_LEVELS),
        help="Gateway log level for interactive runs (default: warning)",
    )
