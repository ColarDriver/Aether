"""Slash command handlers for the Aether REPL."""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from rich.table import Table
from rich.text import Text

from aether.cli.theme import (
    AETHER_ACCENT,
    AETHER_BORDER,
    AETHER_DIM,
    AETHER_PRIMARY,
    AETHER_TEXT,
    icon,
)

if TYPE_CHECKING:
    from aether.cli.repl import ReplState
    from aether.cli.sessions import SessionRecord


# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SlashCommand:
    name: str
    description: str
    handler: Callable[["ReplState", list[str]], "CommandResult"]


@dataclass(slots=True)
class CommandResult:
    """Outcome of a slash command. Drives the REPL's next move."""

    handled: bool = True
    exit: bool = False


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _cmd_help(state: "ReplState", _args: list[str]) -> CommandResult:
    table = Table(
        title=Text("Slash commands", style=f"bold {AETHER_PRIMARY}"),
        border_style=AETHER_BORDER,
        header_style=f"bold {AETHER_DIM}",
        show_lines=False,
    )
    table.add_column("command", style=f"bold {AETHER_ACCENT}")
    table.add_column("description", style=f"{AETHER_TEXT}")
    for cmd in sorted(REGISTRY.values(), key=lambda c: c.name):
        table.add_row(cmd.name, cmd.description)
    state.ui.console.print(table)
    return CommandResult()


def _cmd_exit(_state: "ReplState", _args: list[str]) -> CommandResult:
    return CommandResult(exit=True)


def _cmd_new(state: "ReplState", _args: list[str]) -> CommandResult:
    # Persist whatever we have on disk before walking away from it.
    from aether.cli.repl import persist_session, reset_session_record

    persist_session(state)
    state.session_id = str(uuid.uuid4())
    state.messages = []
    reset_session_record(state)
    state.ui.success(f"Started new session {state.session_id[:8]}…")
    return CommandResult()


def _cmd_session(state: "ReplState", _args: list[str]) -> CommandResult:
    state.ui.info(f"session: {state.session_id}")
    state.ui.info(f"turns:   {sum(1 for m in state.messages if m.get('role') == 'assistant')}")
    return CommandResult()


def _cmd_clear(state: "ReplState", _args: list[str]) -> CommandResult:
    state.messages = []
    state.ui.success("Cleared in-memory conversation history.")
    return CommandResult()


def _cmd_system(state: "ReplState", args: list[str]) -> CommandResult:
    if not args:
        if state.system_prompt:
            state.ui.info("Current system prompt:")
            state.ui.console.print(
                Text(state.system_prompt, style=f"italic {AETHER_DIM}")
            )
        else:
            state.ui.info("No system prompt configured.")
        return CommandResult()
    state.system_prompt = " ".join(args).strip() or None
    state.messages = []
    if state.system_prompt:
        state.ui.success("System prompt updated; cleared conversation history.")
    else:
        state.ui.success("Cleared system prompt.")
    return CommandResult()


def _cmd_tools(state: "ReplState", _args: list[str]) -> CommandResult:
    descriptors = state.engine.services.tool_registry.list_descriptors()
    state.ui.render_tool_table(descriptors)
    return CommandResult()


def _cmd_verbose(state: "ReplState", _args: list[str]) -> CommandResult:
    state.verbose = not state.verbose
    state.ui.verbose = state.verbose
    state.ui.success(f"verbose: {'on' if state.verbose else 'off'}")
    return CommandResult()


def _cmd_model(state: "ReplState", args: list[str]) -> CommandResult:
    """Show, switch, or pick the active model.

    /model            → fetch /v1/models, show interactive picker
    /model <name>     → switch directly without the picker
    /model list       → print the catalog as a static table
    """
    provider = state.engine.services.provider
    current = getattr(provider, "model", None)

    # ---- /model <name> (direct switch) -------------------------------------
    if args and args[0] != "list":
        target = args[0]
        return _switch_model(state, target, current)

    # ---- /model list (read-only listing) -----------------------------------
    if args and args[0] == "list":
        ids = _safe_list_models(state)
        if not ids:
            state.ui.warn("Provider returned no model catalog.")
            return CommandResult()
        for mid in ids:
            mark = "  " + (icon("success") if mid == current else " ")
            state.ui.console.print(f"{mark} {mid}")
        return CommandResult()

    # ---- /model (interactive picker) ---------------------------------------
    state.ui.set_status("Fetching model catalog…")
    try:
        ids = _safe_list_models(state)
    finally:
        state.ui.clear_status()

    if not ids:
        state.ui.warn(
            f"provider {state.provider_name!r} did not return a model "
            f"catalog (current: {current or 'unknown'})."
        )
        return CommandResult()

    from aether.cli.picker import PickerItem, pick

    pick_items = [PickerItem(value=mid, label=mid) for mid in ids]
    selected = pick(
        pick_items,
        title=f"Select model  ·  {state.provider_name}  ·  {len(ids)} available",
        current=current,
    )
    if not selected:
        state.ui.info("model selection cancelled.")
        return CommandResult()

    return _switch_model(state, selected, current)


def _safe_list_models(state: "ReplState") -> list[str]:
    """Best-effort GET /v1/models; returns empty on failure."""
    try:
        return list(state.engine.services.provider.list_models())
    except Exception as exc:  # noqa: BLE001 — provider errors must not crash REPL
        state.ui.warn(f"could not fetch model catalog: {exc}")
        return []


def _switch_model(state: "ReplState", target: str, current: str | None) -> CommandResult:
    if target == current:
        state.ui.info(f"already on model {target!r}.")
        return CommandResult()
    try:
        state.engine.services.provider.set_model(target)
    except Exception as exc:  # noqa: BLE001
        state.ui.error(f"failed to switch model: {exc}")
        return CommandResult()

    new_current = getattr(state.engine.services.provider, "model", target)
    state.ui.success(f"model switched: {current or 'unknown'} → {new_current}")

    # Persist the choice so the next ``aether chat`` (without --model /
    # --resume) starts on this model instead of falling back to the
    # provider default.  Best-effort: prefs failures must not break the
    # REPL, so this is a no-op on disk-full / read-only fs.
    try:
        from aether.cli import prefs as _prefs  # local import keeps startup lean

        _prefs.set_last_model(state.provider_name, str(new_current))
    except Exception:        # noqa: BLE001 - prefs are best-effort
        pass

    return CommandResult()


def _cmd_interrupt(state: "ReplState", _args: list[str]) -> CommandResult:
    state.engine.interrupt(session_id=state.session_id, reason="slash-interrupt")
    state.ui.warn("Sent interrupt to the active turn (if any).")
    return CommandResult()


# ---------------------------------------------------------------------------
# /resume + /sessions  (persistent conversation history)
# ---------------------------------------------------------------------------

def _cmd_resume(state: "ReplState", args: list[str]) -> CommandResult:
    """Pick a previous session from disk and continue the conversation.

    Forms:
        /resume               → interactive picker (latest session first)
        /resume <id-or-prefix>→ jump straight to that session
        /resume list          → static listing (delegates to /sessions)
    """
    from aether.cli import sessions as session_store

    if args and args[0] == "list":
        return _cmd_sessions(state, args[1:])

    records = session_store.list_sessions()
    # Don't offer to "resume" the conversation we're already in.
    records = [r for r in records if r.session_id != state.session_id or r.messages]

    if not records:
        state.ui.info("No saved sessions yet — they're created after your first turn.")
        return CommandResult()

    # ---- /resume <id-or-prefix> ----------------------------------------
    if args:
        target = args[0]
        match = _match_session(records, target)
        if match is None:
            state.ui.error(f"no session matched: {target}")
            return CommandResult()
        return _apply_resume(state, match)

    # ---- /resume (interactive picker) ----------------------------------
    from aether.cli.picker import PickerItem, pick

    items = [
        PickerItem(
            value=r.session_id,
            label=_session_picker_label(r),
            description=session_store.format_session_preview(r, max_chars=80),
        )
        for r in records
    ]
    selected = pick(
        items,
        title=f"Resume session  ·  {len(records)} available",
        current=None,
    )
    if not selected:
        state.ui.info("resume cancelled.")
        return CommandResult()

    record = next((r for r in records if r.session_id == selected), None)
    if record is None:
        state.ui.error("internal error: selected session vanished from cache.")
        return CommandResult()
    return _apply_resume(state, record)


def _cmd_sessions(state: "ReplState", _args: list[str]) -> CommandResult:
    """Static, scrollback-friendly listing of every persisted session."""
    from aether.cli import sessions as session_store

    records = session_store.list_sessions()
    if not records:
        state.ui.info("No saved sessions yet.")
        return CommandResult()

    table = Table(
        title=Text(f"Saved sessions  ·  {len(records)}", style=f"bold {AETHER_PRIMARY}"),
        border_style=AETHER_BORDER,
        header_style=f"bold {AETHER_DIM}",
        show_lines=False,
    )
    table.add_column("id", style=f"bold {AETHER_ACCENT}", no_wrap=True)
    table.add_column("when", style=AETHER_DIM, no_wrap=True)
    table.add_column("turns", style=AETHER_TEXT, justify="right", no_wrap=True)
    table.add_column("model", style=AETHER_DIM, no_wrap=True)
    table.add_column("preview", style=AETHER_TEXT, overflow="ellipsis")

    for r in records:
        marker = icon("success") if r.session_id == state.session_id else " "
        preview = (r.first_user_message or "(no messages yet)").replace("\n", " ")
        if len(preview) > 60:
            preview = preview[:60] + "…"
        table.add_row(
            f"{marker} {r.session_id[:8]}",
            session_store.format_relative_time(r.updated_at),
            str(r.turn_count),
            f"{r.provider}/{r.model}".strip("/"),
            preview,
        )
    state.ui.console.print(table)
    state.ui.info("Use /resume to switch into one of these sessions.")
    return CommandResult()


def _apply_resume(state: "ReplState", record: "SessionRecord") -> CommandResult:
    """Swap *state* over to the conversation captured in *record*."""
    from aether.cli.repl import persist_session

    # Save the in-progress conversation first so we don't lose it.
    persist_session(state)

    state.session_id = record.session_id
    state.messages = list(record.messages)
    state.system_prompt = record.system_prompt
    state.session_record = record

    # Best-effort: re-align the live provider with the model the session
    # was using.  Users can /model afterwards to override.
    if record.model:
        try:
            state.engine.services.provider.set_model(record.model)
        except Exception:  # noqa: BLE001
            pass

    from aether.cli import sessions as session_store

    summary = session_store.format_session_preview(record, max_chars=80)
    state.ui.success(
        f"Resumed session {record.session_id[:8]}  ·  {summary}"
    )

    last = _last_assistant_text(record.messages)
    if last:
        snippet = last.strip()
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        state.ui.console.print(
            Text(snippet, style=f"italic {AETHER_DIM}")
        )
    return CommandResult()


def _match_session(
    records: list["SessionRecord"], target: str
) -> "SessionRecord | None":
    """Resolve *target* to a session: exact id, then unique prefix."""
    for r in records:
        if r.session_id == target:
            return r
    candidates = [r for r in records if r.session_id.startswith(target)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _session_picker_label(record: "SessionRecord") -> str:
    from aether.cli import sessions as session_store

    when = session_store.format_relative_time(record.updated_at)
    return f"{record.session_id[:8]}  ·  {when}"


def _last_assistant_text(messages: list[dict]) -> str:
    """Last assistant utterance as plain text (for the 'resumed' preview)."""
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return str(part.get("text", ""))
    return ""


def _cmd_stats(state: "ReplState", _args: list[str]) -> CommandResult:
    s = state.ui.stats
    state.ui.info(
        f"last turn: iter={s.iterations} tool_calls={s.tool_calls} "
        f"errors={s.tool_errors} chars={s.streamed_chars} "
        f"elapsed={s.elapsed_sec:.2f}s"
    )
    return CommandResult()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, SlashCommand] = {
    "/help": SlashCommand("/help", "Show this help table", _cmd_help),
    "/exit": SlashCommand("/exit", "Exit the REPL", _cmd_exit),
    "/new": SlashCommand("/new", "Start a new session (saves the current one first)", _cmd_new),
    "/clear": SlashCommand("/clear", "Clear conversation history (keep session)", _cmd_clear),
    "/session": SlashCommand("/session", "Show current session info", _cmd_session),
    "/sessions": SlashCommand("/sessions", "List previously saved sessions", _cmd_sessions),
    "/resume": SlashCommand("/resume", "Resume a previous session: /resume [<id|prefix>]", _cmd_resume),
    "/system": SlashCommand("/system", "Show or set the system prompt: /system <text>", _cmd_system),
    "/tools": SlashCommand("/tools", "List registered tools", _cmd_tools),
    "/model": SlashCommand("/model", "Show, switch, or pick the active model", _cmd_model),
    "/verbose": SlashCommand("/verbose", "Toggle per-turn verbose output", _cmd_verbose),
    "/stats": SlashCommand("/stats", "Show stats from the last turn", _cmd_stats),
    "/interrupt": SlashCommand("/interrupt", "Interrupt the active turn", _cmd_interrupt),
}


# Same alphabet open-claude-code uses for ``looksLikeCommand`` —
# slash command names are identifier-shaped so anything containing '/',
# '.', '~', whitespace, CJK characters, etc. in the head token is
# almost certainly a file path or prose input that should reach the
# model untouched.
_COMMAND_NAME_RE = re.compile(r"^[a-zA-Z0-9:_-]+$")


def _looks_like_command_name(head: str) -> bool:
    """Whether *head* (the token after a leading '/') is shaped like a slash command.

    Mirrors open-claude-code's ``looksLikeCommand`` (see
    ``src/utils/processUserInput/processSlashCommand.tsx``).
    """
    return bool(_COMMAND_NAME_RE.fullmatch(head))


def is_slash(line: str) -> bool:
    """Return True only when *line* should be routed to the slash-command dispatcher.

    Three gates, in order — modelled directly on open-claude-code's
    slash-vs-prose disambiguation so users can paste paths like
    ``/workspace/hermes-agent 帮我看下这个项目`` without the REPL
    swallowing them as ``Unknown command``:

    1. Trimmed line must start with ``/`` and have at least one char
       after it.
    2. The token between ``/`` and the first whitespace must look like
       an identifier (``[a-zA-Z0-9:_-]+``).  Anything else (slash,
       dot, tilde, CJK, …) → almost certainly a path or prose, hand
       it to the model.
    3. ``"/" + head`` must not exist on disk.  Filters absolute paths
       whose first segment happens to be identifier-shaped (``/var``,
       ``/tmp``, ``/workspace``, ``/Users``, …).

    Genuine typos like ``/halp`` still pass all three gates, so the
    dispatcher continues to surface ``Unknown command`` for them.
    """
    stripped = line.strip()
    if not stripped.startswith("/") or len(stripped) < 2:
        return False
    head = stripped[1:].split(maxsplit=1)[0]
    if not head or not _looks_like_command_name(head):
        return False
    try:
        if os.path.exists("/" + head):
            return False
    except OSError:
        # ``os.path.exists`` is documented as never raising, but a
        # broken symlink or permission edge case shouldn't crash the
        # REPL — fall through and treat as a command attempt.
        pass
    return True


# ---------------------------------------------------------------------------
# Completer (drives the popup that appears when the user types '/')
# ---------------------------------------------------------------------------

class SlashCompleter(Completer):
    """Yield completions for the slash-command palette.

    The popup only appears while the buffer is a single line and starts
    with ``/`` — that keeps it from interfering with normal prose.
    """

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterator[Completion]:
        text = document.text_before_cursor

        if "\n" in document.text or not text.startswith("/"):
            return

        # First word completion: '/foo'.  Only one token before cursor.
        first_token = text.split(" ", 1)[0]
        if " " not in text:
            for cmd in sorted(REGISTRY.values(), key=lambda c: c.name):
                if cmd.name.startswith(first_token):
                    yield Completion(
                        cmd.name,
                        start_position=-len(first_token),
                        display=cmd.name,
                        display_meta=cmd.description,
                    )


def dispatch(state: "ReplState", line: str) -> CommandResult:
    """Run *line* as a slash command. Returns ``handled=False`` if unknown."""
    parts = line.strip().split()
    if not parts:
        return CommandResult(handled=False)
    name = parts[0].lower()
    args = parts[1:]
    cmd = REGISTRY.get(name)
    if cmd is None:
        state.ui.warn(f"Unknown command: {name}  (try /help)")
        return CommandResult()
    try:
        return cmd.handler(state, args)
    except Exception as exc:  # noqa: BLE001 — REPL must never crash on /cmd
        state.ui.error(f"/{name[1:]} failed: {exc}")
        return CommandResult()
