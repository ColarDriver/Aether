"""Thin launcher for the TypeScript Ink TUI.

Re-uses the existing ``aether/cli/main.py`` argparse surface so users get
the same flag vocabulary as the prompt_toolkit TUI, then translates
those into environment variables consumed by ``tui/src/entry.tsx``
before spawning the TS process.

Invocation:

    python -m aether.cli.launcher --provider claude --temperature 0.2

The launcher does NOT bring up the Python prompt_toolkit TUI. It is a
forwarder onto the TS entry point that runs as a child process under the
user's shell so Ctrl-C / Ctrl-Z propagate naturally.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from aether.cli.main import _build_parser


# Mapping from argparse attr (with underscores) to the env var the TS side
# reads. Kept intentionally short — every flag in ``main.py:_add_chat_args``
# that the TS TUI honours gets a row here. Anything missing means the flag
# is parsed by argparse but ignored by the TS path (the user will see the
# flag accepted without effect, which is the same behaviour Python's
# argparse already gives for unknown attrs on the namespace).
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


def _build_env(args) -> dict[str, str]:
    """Translate parsed argparse namespace into env vars for the TS child."""
    env = dict(os.environ)
    env.setdefault("AETHER_WORKSPACE_CWD", os.getcwd())
    for attr, key in _STRING_FLAG_TO_ENV.items():
        value = getattr(args, attr, None)
        if value:
            env[key] = str(value)
    for attr, key in _INT_FLAG_TO_ENV.items():
        value = getattr(args, attr, None)
        # argparse default for max_iterations is 32. Forward it unless the
        # caller already provided the env var so the TS path matches Python
        # ``cmd_chat``'s EngineConfig default.
        if value is not None and key not in env:
            env[key] = str(value)
    for attr, key in _FLOAT_FLAG_TO_ENV.items():
        value = getattr(args, attr, None)
        if value is not None:
            env[key] = str(value)
    for attr, key in _BOOL_FLAG_TO_ENV.items():
        if getattr(args, attr, False):
            env[key] = "1"

    # --resume has a special tri-state: id, "__pick__" (interactive picker),
    # or None. Translate "__pick__" to AETHER_RESUME=1 so the TS side opens
    # the boot-time picker.
    resume = getattr(args, "resume", None)
    if resume == "__pick__":
        env.setdefault("AETHER_RESUME", "1")
    elif resume:
        env.setdefault("AETHER_RESUME", str(resume))

    return env


def _resolve_tui_entry() -> tuple[list[str], Path] | None:
    """Locate the TS entry. Prefer compiled `dist/entry.js`, fall back to
    `tsx` dev mode. Returns the spawn command list + working directory."""
    here = Path(__file__).resolve().parent.parent.parent  # repo root
    tui_dir = here / "tui"
    if not tui_dir.is_dir():
        return None
    dist_entry = tui_dir / "dist" / "entry.js"
    if dist_entry.is_file():
        node = shutil.which("node")
        if node:
            return ([node, str(dist_entry)], tui_dir)
    # Dev fallback — requires `npm install` has run inside tui/.
    npx = shutil.which("npx")
    if npx:
        return ([npx, "--yes", "tsx", "src/entry.tsx"], tui_dir)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    env = _build_env(args)

    resolved = _resolve_tui_entry()
    if resolved is None:
        sys.stderr.write(
            "tui not found. Run `(cd tui && npm install)` first, or "
            "build with `(cd tui && npm run build)` for production launches.\n"
        )
        return 127
    cmd, cwd = resolved
    cmd = [*cmd, *unknown]
    try:
        proc = subprocess.Popen(cmd, env=env, cwd=str(cwd))
        return proc.wait()
    except KeyboardInterrupt:
        # The child handles its own SIGINT — we just observe and exit.
        return 130
    except FileNotFoundError as exc:
        sys.stderr.write(f"failed to launch TUI: {exc}\n")
        return 127


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
