"""Aether CLI — main entry point.

Usage:
    aether                          # Interactive REPL (default)
    aether chat                     # Interactive REPL (explicit)
    aether chat -p openai           # Use OpenAI-compatible provider
    aether chat -p claude           # Use Anthropic / Claude provider
    aether chat -p codex            # Use Codex provider
    aether chat -m gpt-5.4-mini     # Override model
    aether chat --system "..."      # Set system prompt
    aether chat --verbose           # Show engine metadata per turn
    aether chat --session <id>      # Tag a specific session UUID
    aether chat --resume            # Pick a past session to continue
    aether chat --resume <id>       # Resume a specific session by id/prefix
    aether chat --no-banner         # Skip the welcome banner
    aether chat --log-level info    # Raise engine log verbosity
    aether providers                # List available providers
    aether version                  # Print version + build info
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

# Best-effort version detection (falls back when the package isn't installed
# via pip / hatch — we ship from source during development).
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        AETHER_VERSION = _pkg_version("aether-harness")
    except PackageNotFoundError:
        AETHER_VERSION = "1.0.0"
except Exception:  # pragma: no cover - paranoia
    AETHER_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def _setup_logging(level_name: str) -> None:
    level = _LOG_LEVELS.get((level_name or "warning").lower(), logging.WARNING)
    logging.basicConfig(
        format="%(levelname)s %(name)s: %(message)s",
        level=level,
        stream=sys.stderr,
    )
    if level > logging.DEBUG:
        # The httpx/httpcore stack is extremely chatty at INFO/DEBUG and
        # tends to drown out anything useful when the user just wants
        # engine logs.  Pin them one notch above whatever is requested.
        for noisy in ("httpx", "httpcore", "asyncio", "urllib3"):
            logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


# ---------------------------------------------------------------------------
# History file
# ---------------------------------------------------------------------------

def _default_history_file() -> Path:
    base = Path(os.getenv("AETHER_HOME", Path.home() / ".aether"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "repl_history"


# ---------------------------------------------------------------------------
# Environment context (injected into the system prompt)
# ---------------------------------------------------------------------------
#
# Without this block the model has no idea where the harness is running.
# When the user asks "查看当前文件夹" / "what's in this directory?" the
# model can only guess or apologise, since it doesn't know its CWD,
# whether it's in a git repo, the platform, or the current date.
#
# We prepend a small ``<environment>...</environment>`` block to whatever
# system prompt the user supplied (or stand alone if none was given) so
# every turn has the same baseline context the user takes for granted.

def _build_environment_context() -> str:
    """Return a small block describing where + when the harness is running.

    Environment system-prompt section: working directory, platform,
    shell, git status, current date. Kept short so
    it doesn't dominate the system prompt or burn the prefix cache.
    """
    import platform as _platform
    import time as _time

    cwd = Path.cwd()

    # Walk up looking for a ``.git`` directory — running from a sub-folder
    # of a repo (e.g. ``backend/harness/aether``) shouldn't make the
    # model think it's outside version control.
    git_root: Path | None = None
    for candidate in (cwd, *cwd.parents):
        if (candidate / ".git").exists():
            git_root = candidate
            break

    git_branch: str | None = None
    if git_root is not None:
        head = git_root / ".git" / "HEAD"
        try:
            head_text = head.read_text(encoding="utf-8").strip()
            if head_text.startswith("ref: refs/heads/"):
                git_branch = head_text[len("ref: refs/heads/"):]
        except OSError:
            pass

    lines = [
        "<environment>",
        f"working_directory: {cwd}",
        f"platform: {_platform.system().lower()} ({_platform.machine()})",
        f"is_git_repository: {'yes' if git_root is not None else 'no'}",
    ]
    if git_root is not None and git_root != cwd:
        lines.append(f"git_root: {git_root}")
    if git_branch:
        lines.append(f"git_branch: {git_branch}")
    lines.append(f"shell: {os.environ.get('SHELL', 'unknown')}")
    lines.append(f"date: {_time.strftime('%Y-%m-%d')}")
    lines.append("</environment>")
    return "\n".join(lines)


def _augment_system_prompt(user_prompt: str | None) -> str:
    """Prepend the environment block to *user_prompt* (or stand alone).

    Always returns a non-empty string — every session gets at least the
    environment context so the model knows its working directory.
    """
    env_block = _build_environment_context()
    if user_prompt:
        return f"{env_block}\n\n{user_prompt}"
    return env_block


# ---------------------------------------------------------------------------
# cmd_chat
# ---------------------------------------------------------------------------

def cmd_chat(args: argparse.Namespace) -> None:
    _setup_logging(args.log_level)

    # Local imports keep ``aether providers`` / ``aether version`` startup fast
    # — no Rich/Console import for those subcommands.
    from aether.agents.core.agent import AgentEngine
    from aether.cli.providers import build_provider
    from aether.cli.repl import run_repl
    from aether.cli.sessions import SessionRecord
    from aether.config.schema import EngineConfig, ModelCallConfig
    from aether.subagents import SubagentManager

    from aether.cli import prefs as _prefs

    provider_name: str = args.provider or os.getenv("AETHER_PROVIDER", "openai")

    # Model precedence (highest → lowest):
    #   1. ``--model`` flag                       (this run only)
    #   2. ``AETHER_MODEL`` env var               (this shell)
    #   3. persisted last choice for this provider (across runs)
    #   4. provider default                       (fallback in build_provider)
    #
    # Resume cases override #3 explicitly later — see ``resume_record``.
    explicit_model = args.model or os.getenv("AETHER_MODEL")
    remembered_model: str | None = None
    if not explicit_model:
        remembered_model = _prefs.get_last_model(provider_name)

    try:
        provider = build_provider(
            provider_name,
            model=explicit_model or remembered_model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    except ValueError as exc:
        _print_startup_error(str(exc))
        sys.exit(1)

    # If the user passed ``--model`` (or ``AETHER_MODEL``) on this run,
    # treat that as a deliberate update to the persisted preference so
    # subsequent ``aether chat`` (without the flag) starts on the same
    # model.  When only the remembered value was used, no rewrite —
    # the file already says what we just loaded.
    if explicit_model:
        active_model = getattr(provider, "model", explicit_model)
        if active_model:
            _prefs.set_last_model(provider_name, str(active_model))

    engine_config = EngineConfig(
        max_iterations=args.max_iterations,
        fail_on_tool_error=False,
        fail_on_unknown_tool=False,
        use_builtin_tools=not args.no_builtin_tools,
    )
    # Wire a SubagentManager so the built-in ``task`` / ``task_stop`` tools
    # can dispatch isolated child agents.  Without this the parent fails with
    # "AgentTool not wired: SubagentManager is not configured on the parent
    # agent" the moment the model tries to delegate.
    subagent_manager = SubagentManager()
    engine = AgentEngine(
        provider=provider,
        config=engine_config,
        subagent_manager=subagent_manager,
    )

    model_config = ModelCallConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    system_prompt: str | None = args.system
    if not system_prompt and args.system_file:
        try:
            system_prompt = Path(args.system_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            _print_startup_error(f"could not read system prompt file: {exc}")
            sys.exit(1)

    # Always prepend the environment block so the model knows its
    # working directory, platform, git state, and current date — even
    # when the user didn't supply a system prompt.  Without this the
    # model has to guess (or apologise) when asked about the workspace.
    system_prompt = _augment_system_prompt(system_prompt)

    # ---- session resume -----------------------------------------------------
    resume_record: SessionRecord | None = _resolve_resume_target(
        getattr(args, "resume", None)
    )
    if resume_record is not None:
        # Honour the saved model unless the user explicitly overrode it.
        if resume_record.model and not args.model:
            try:
                provider.set_model(resume_record.model)
            except Exception:  # noqa: BLE001
                pass

    session_id: str = (
        resume_record.session_id if resume_record else (args.session or str(uuid.uuid4()))
    )

    run_repl(
        engine,
        provider_name=provider_name,
        system_prompt=system_prompt,
        model_config=model_config,
        history_file=_default_history_file(),
        session_id=session_id,
        verbose=args.verbose,
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL"),
        version=AETHER_VERSION,
        show_banner=not args.no_banner,
        resume_record=resume_record,
    )


def _resolve_resume_target(value: str | None):
    """Decode the ``--resume`` flag, possibly opening the picker.

    * ``None``                — no resume requested.
    * ``"__pick__"`` sentinel — show the interactive picker.
    * ``<id-or-prefix>``      — load the matching record, or exit on miss.
    """
    if value is None:
        return None

    from aether.cli import sessions as session_store

    if value == "__pick__":
        records = session_store.list_sessions()
        if not records:
            _print_startup_error(
                "No saved sessions to resume — run a turn first to create one."
            )
            sys.exit(1)
        return _interactive_pick_session(records)

    record = session_store.load_session(value)
    if record is not None:
        return record

    # Allow a unique prefix for convenience (e.g. ``--resume abc123``).
    records = session_store.list_sessions()
    matches = [r for r in records if r.session_id.startswith(value)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        _print_startup_error(f"session not found: {value}")
    else:
        _print_startup_error(
            f"ambiguous session prefix {value!r}: matches {len(matches)} records — "
            "use a longer prefix."
        )
    sys.exit(1)


def _interactive_pick_session(records):
    """Drop into the picker before the REPL takes over the screen."""
    from aether.cli import sessions as session_store
    from aether.cli.picker import PickerItem, pick

    items = [
        PickerItem(
            value=r.session_id,
            label=f"{r.session_id[:8]}  ·  {session_store.format_relative_time(r.updated_at)}",
            description=session_store.format_session_preview(r, max_chars=80),
        )
        for r in records
    ]
    selected = pick(items, title=f"Resume session  ·  {len(records)} available")
    if not selected:
        _print_startup_error("resume cancelled.")
        sys.exit(0)
    return next((r for r in records if r.session_id == selected), None)


# ---------------------------------------------------------------------------
# cmd_providers
# ---------------------------------------------------------------------------

def cmd_providers(_args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.table import Table

    from aether.cli.providers import _DEFAULTS  # type: ignore[attr-defined]
    from aether.cli.providers import PROVIDER_ALIASES, list_providers
    from aether.cli.theme import (
        AETHER_ACCENT,
        AETHER_BORDER,
        AETHER_DIM,
        AETHER_PRIMARY,
        AETHER_TEXT,
        THEME,
    )

    console = Console(theme=THEME)
    table = Table(
        title=f"[bold {AETHER_PRIMARY}]Aether providers[/]",
        border_style=AETHER_BORDER,
        header_style=f"bold {AETHER_DIM}",
    )
    table.add_column("name", style=f"bold {AETHER_ACCENT}")
    table.add_column("default model", style=AETHER_TEXT)
    table.add_column("notes", style=AETHER_DIM)

    notes = {
        "claude": "Anthropic Claude — set ANTHROPIC_API_KEY",
        "openai": "OpenAI-compatible — set OPENAI_API_KEY [+ OPENAI_BASE_URL]",
        "codex": "OpenAI Codex — set CODEX_ACCESS_TOKEN",
    }
    for name in list_providers():
        defaults = _DEFAULTS.get(name, {})
        table.add_row(name, str(defaults.get("model", "?")), notes.get(name, ""))

    console.print(table)
    if PROVIDER_ALIASES:
        console.print()
        console.print(f"[{AETHER_DIM}]aliases:[/] " + ", ".join(
            f"{alias} → {target}" for alias, target in PROVIDER_ALIASES.items()
        ))


# ---------------------------------------------------------------------------
# cmd_version
# ---------------------------------------------------------------------------

def cmd_version(_args: argparse.Namespace) -> None:
    from rich.console import Console

    from aether.cli.theme import AETHER_DIM, AETHER_PRIMARY, THEME, icon

    console = Console(theme=THEME)
    console.print(
        f"[bold {AETHER_PRIMARY}]{icon('logo')} Aether[/] "
        f"[{AETHER_DIM}]v{AETHER_VERSION}[/]"
    )
    console.print(
        f"[{AETHER_DIM}]python {sys.version.split()[0]}  ·  "
        f"platform {sys.platform}[/]"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_startup_error(message: str) -> None:
    """Render a startup-time error using Rich if possible."""
    try:
        from rich.console import Console

        from aether.cli.theme import THEME, icon

        console = Console(theme=THEME, stderr=True)
        console.print(f"[aether.error]{icon('error')} {message}[/]")
    except Exception:
        print(f"error: {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aether",
        description="Aether — interactive AI agent harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- chat (default) ---
    chat = subparsers.add_parser("chat", help="Start interactive REPL (default)")
    _add_chat_args(chat)
    chat.set_defaults(func=cmd_chat)

    # --- providers ---
    prov = subparsers.add_parser("providers", help="List available providers")
    prov.set_defaults(func=cmd_providers)

    # --- version ---
    ver = subparsers.add_parser("version", help="Show version + runtime info")
    ver.set_defaults(func=cmd_version)

    # Attach chat args to the root parser too so bare `aether` works
    _add_chat_args(parser)
    parser.set_defaults(func=cmd_chat)

    return parser


def _add_chat_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-p", "--provider",
        metavar="NAME",
        help="Provider to use: claude, openai, codex (default: openai or $AETHER_PROVIDER)",
    )
    p.add_argument(
        "-m", "--model",
        metavar="MODEL",
        help="Model name override (default: provider default or $AETHER_MODEL)",
    )
    p.add_argument(
        "--api-key",
        metavar="KEY",
        help="API key (overrides env var for the selected provider)",
    )
    p.add_argument(
        "--base-url",
        metavar="URL",
        help="Base URL override for OpenAI-compatible providers",
    )
    p.add_argument(
        "--system",
        metavar="PROMPT",
        help="System prompt text",
    )
    p.add_argument(
        "--system-file",
        metavar="FILE",
        help="Path to a file containing the system prompt",
    )
    p.add_argument(
        "--session",
        metavar="ID",
        help="Tag a specific session UUID for the new conversation",
    )
    p.add_argument(
        "--resume",
        metavar="ID",
        nargs="?",
        const="__pick__",
        default=None,
        help=(
            "Resume a previously saved session.  Pass an id (or unique prefix) "
            "to jump straight in, or no value for an interactive picker."
        ),
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=32,
        metavar="N",
        help="Max agent loop iterations per turn (default: 32)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        metavar="T",
        help="Sampling temperature",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Max tokens per response",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show engine metadata (status, iterations, exit reason) after each turn",
    )
    p.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip the welcome banner on startup",
    )
    p.add_argument(
        "--no-builtin-tools",
        action="store_true",
        help=(
            "Disable the bundled tool kit (shell, read_file, write_file, "
            "list_dir, grep, glob) for this session."
        ),
    )
    p.add_argument(
        "--log-level",
        metavar="LEVEL",
        default=os.getenv("AETHER_LOG_LEVEL", "warning"),
        choices=list(_LOG_LEVELS.keys()),
        help="Python log level for engine/provider output (default: warning)",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    func = getattr(args, "func", cmd_chat)
    func(args)


if __name__ == "__main__":
    main()
