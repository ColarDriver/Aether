"""Python CLI entry point for non-interactive commands and TS TUI launch."""

from __future__ import annotations

import argparse
from typing import cast

from aether.cli.info import cmd_providers, cmd_version
from aether.cli.parser import build_parser


def cmd_chat(args: argparse.Namespace) -> None:
    from aether.cli.launcher import launch_tui

    raise SystemExit(launch_tui(args))


def _command(args: argparse.Namespace) -> str | None:
    value = cast(object, getattr(args, "command", None))
    return value if isinstance(value, str) else None


def _build_parser() -> argparse.ArgumentParser:
    """Compatibility wrapper for callers that import the historical helper."""
    return build_parser()


def dispatch_noninteractive(args: argparse.Namespace) -> None:
    command = _command(args)
    if command == "providers":
        cmd_providers(args)
        return
    if command == "version":
        cmd_version(args)
        return
    raise ValueError(f"unsupported non-interactive command: {command!r}")

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if _command(args) in (None, "chat"):
        from aether.cli.launcher import main as launch_tui_main

        raise SystemExit(launch_tui_main(argv))

    dispatch_noninteractive(args)


if __name__ == "__main__":
    main()
