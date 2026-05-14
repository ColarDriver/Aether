"""Thin launcher for the TypeScript Ink TUI."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

from aether.cli.info import cmd_providers, cmd_version
from aether.cli.parser import build_parser


# Mapping from argparse attr names to the env vars consumed by the TS TUI.
_STRING_FLAG_TO_ENV: dict[str, str] = {
    "provider": "AETHER_PROVIDER",
    "model": "AETHER_MODEL",
    "api_key": "AETHER_API_KEY",
    "base_url": "AETHER_BASE_URL",
    "system": "AETHER_SYSTEM",
    "system_file": "AETHER_SYSTEM_FILE",
    "session": "AETHER_SESSION_ID",
    "log_level": "AETHER_LOG_LEVEL",
}

_INT_FLAG_TO_ENV: dict[str, str] = {
    "max_iterations": "AETHER_MAX_ITERATIONS",
    "max_tokens": "AETHER_MAX_TOKENS",
}

_FLOAT_FLAG_TO_ENV: dict[str, str] = {
    "temperature": "AETHER_TEMPERATURE",
}

_BOOL_FLAG_TO_ENV: dict[str, str] = {
    "verbose": "AETHER_VERBOSE",
    "no_banner": "AETHER_NO_BANNER",
    "no_builtin_tools": "AETHER_NO_BUILTIN_TOOLS",
}


def _namespace_value(args: argparse.Namespace, attr: str) -> object:
    return cast(object, getattr(args, attr, None))


def _command(args: argparse.Namespace) -> str | None:
    value = _namespace_value(args, "command")
    return value if isinstance(value, str) else None


def _build_env(args: argparse.Namespace) -> dict[str, str]:
    """Translate parsed argparse namespace into env vars for the TS child."""
    env = dict(os.environ)
    _ = env.setdefault("AETHER_WORKSPACE_CWD", os.getcwd())

    for attr, key in _STRING_FLAG_TO_ENV.items():
        value = _namespace_value(args, attr)
        if value:
            env[key] = str(value)

    for attr, key in _INT_FLAG_TO_ENV.items():
        value = _namespace_value(args, attr)
        if isinstance(value, int) and not isinstance(value, bool) and key not in env:
            env[key] = str(value)

    for attr, key in _FLOAT_FLAG_TO_ENV.items():
        value = _namespace_value(args, attr)
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            env[key] = str(value)

    for attr, key in _BOOL_FLAG_TO_ENV.items():
        if _namespace_value(args, attr) is True:
            env[key] = "1"

    resume = _namespace_value(args, "resume")
    if resume == "__pick__":
        _ = env.setdefault("AETHER_RESUME", "1")
    elif isinstance(resume, str) and resume:
        _ = env.setdefault("AETHER_RESUME", resume)

    return env


def _resolve_tui_entry() -> tuple[list[str], Path] | None:
    """Locate the TS entry point, preferring compiled output over dev mode."""
    here = Path(__file__).resolve().parent.parent.parent
    tui_dir = here / "tui"
    if not tui_dir.is_dir():
        return None

    dist_entry = tui_dir / "dist" / "entry.js"
    if dist_entry.is_file():
        node = shutil.which("node")
        if node:
            return ([node, str(dist_entry)], tui_dir)

    npx = shutil.which("npx")
    if npx:
        return ([npx, "--yes", "tsx", "src/entry.tsx"], tui_dir)
    return None


def launch_tui(args: argparse.Namespace, unknown: Iterable[str] = ()) -> int:
    """Spawn the TS TUI with env translated from parsed CLI flags."""
    env = _build_env(args)
    resolved = _resolve_tui_entry()
    if resolved is None:
        message = "".join(
            [
                "tui not found. Run `(cd tui && npm install)` first, or build with ",
                "`(cd tui && npm run build)` for production launches.\n",
            ]
        )
        _ = sys.stderr.write(message)
        return 127

    cmd, cwd = resolved
    cmd = [*cmd, *unknown]
    try:
        proc = subprocess.Popen(cmd, env=env, cwd=str(cwd))
        return proc.wait()
    except KeyboardInterrupt:
        return 130
    except FileNotFoundError as exc:
        _ = sys.stderr.write(f"failed to launch TUI: {exc}\n")
        return 127


def _dispatch_noninteractive(args: argparse.Namespace) -> None:
    command = _command(args)
    if command == "providers":
        cmd_providers(args)
        return
    if command == "version":
        cmd_version(args)
        return
    raise ValueError(f"unsupported non-interactive command: {command!r}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if _command(args) not in (None, "chat"):
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        _dispatch_noninteractive(args)
        return 0
    return launch_tui(args, unknown)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
