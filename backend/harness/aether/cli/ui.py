"""Rich-based UI controller for the Aether CLI.

Centralises streaming token output, status spinners, and tool-call /
tool-result rendering so that the REPL and middleware bridge can simply
fire high-level events.

The streaming path uses ``rich.live.Live`` driving a ``rich.markdown.Markdown``
renderable so that tables, code blocks, lists, and inline formatting are
laid out properly as the model produces them.  We also strip out any
``<tool_call>…</tool_call>`` / ``<tool_result>…</tool_result>`` blocks the
model emits in its prose — those are pure narration noise on top of the
real structured tool calls (which the middleware renders as panels).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, ConsoleRenderable, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from aether.cli.theme import (
    AETHER_ACCENT,
    AETHER_BORDER,
    AETHER_DIM,
    AETHER_ERROR,
    AETHER_PRIMARY,
    AETHER_PRIMARY_DIM,
    AETHER_SUCCESS,
    AETHER_TEXT,
    AETHER_WARNING,
    TOOL_ACCENT,
    TOOL_DIM,
    icon,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_TOOL_RESULT_PREVIEW = 1200
_MAX_TOOL_ARGS_PREVIEW = 800

# Tags some models emit inline to "narrate" their tool use.  These are
# stripped before rendering — the structured tool_calls in the message
# envelope already drive our compact tool lines, so the inline copies are
# pure noise.  Anthropic-style ``<function_calls>`` namespaces are
# tolerated via the ``(?:[\w-]+:)?`` prefix in the regex.
#
#   <tool_call>{...}</tool_call>           <tool_calls>...</tool_calls>
#   <tool_result>...</tool_result>         <tool_results>...</tool_results>
#   <tool_use>...</tool_use>               <tool_response>...</tool_response>
#   <function_call>...</function_call>     <function_calls>...</function_calls>
#   <function_result>...</function_result> <function_response>...</function_response>
#   <invoke>...</invoke>                   <thinking>...</thinking>
_TOOL_TAGS = (
    "tool_call", "tool_calls",
    "tool_result", "tool_results",
    "tool_use", "tool_uses",
    "tool_response", "tool_responses",
    "function_call", "function_calls",
    "function_result", "function_results",
    "function_response", "function_responses",
    "invoke", "invokes",
    "thinking",
)
# A named back-reference (``(?P=tag)``) keeps the open and close tag
# names in lock-step.  ``(?:[\w-]+:)?`` lets either side carry an XML
# namespace prefix (e.g. ``<function_calls>``).
_TAG_NAME = "(?P<tag>" + "|".join(_TOOL_TAGS) + ")"
_TOOL_BLOCK_RE = re.compile(
    r"<\s*(?:[\w-]+:)?" + _TAG_NAME + r"\b[^>]*>"
    r".*?"
    r"</\s*(?:[\w-]+:)?(?P=tag)\s*>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_OPEN_RE = re.compile(
    r"<\s*(?:[\w-]+:)?(?:" + "|".join(_TOOL_TAGS) + r")\b[^>]*>",
    re.IGNORECASE,
)


def strip_tool_blocks(text: str) -> str:
    """Remove inline ``<tool_call>…</tool_call>``-style blocks from *text*.

    The function is conservative: complete blocks are deleted outright;
    a partial open tag (no matching close yet) hides everything from the
    open tag onward, so the user never sees half-rendered JSON before the
    closing tag arrives in the stream.  Trailing blank-line runs are
    collapsed to keep the prose tidy.
    """
    cleaned = _TOOL_BLOCK_RE.sub("", text)
    open_match = _TOOL_OPEN_RE.search(cleaned)
    if open_match is not None:
        cleaned = cleaned[: open_match.start()]
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _format_args(args: dict[str, Any]) -> str:
    """Pretty-print tool arguments, truncating overly long values."""
    if not args:
        return "{}"
    try:
        text = json.dumps(args, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = repr(args)
    if len(text) > _MAX_TOOL_ARGS_PREVIEW:
        text = text[:_MAX_TOOL_ARGS_PREVIEW] + "\n  …(truncated)"
    return text


def _truncate_for_preview(content: str, limit: int = _MAX_TOOL_RESULT_PREVIEW) -> tuple[str, bool]:
    if not content:
        return "", False
    if len(content) <= limit:
        return content, False
    return content[:limit] + "\n…(truncated)", True


# ---------------------------------------------------------------------------
# Compact tool-call rendering — "● Reading file" + "└ <detail>"
# ---------------------------------------------------------------------------

# Verb mapping used to convert raw tool names into human-readable action
# phrases.  Tools whose name isn't in the table fall through to the
# heuristic in :func:`_verb_for_tool` below.
_TOOL_VERBS: dict[str, str] = {
    "read_file": "Reading file",
    "Read": "Reading file",
    "view_file": "Reading file",
    "ViewFile": "Reading file",
    "list_directory": "Listing directory",
    "ListDirectory": "Listing directory",
    "ls": "Listing directory",
    "list": "Listing directory",
    "write_file": "Writing file",
    "WriteFile": "Writing file",
    "Write": "Writing file",
    "create_file": "Creating file",
    "edit_file": "Editing file",
    "Edit": "Editing file",
    "EditFile": "Editing file",
    "patch": "Patching file",
    "apply_patch": "Applying patch",
    "delete_file": "Deleting file",
    "DeleteFile": "Deleting file",
    "run_bash": "Running command",
    "Bash": "Running command",
    "bash": "Running command",
    "shell": "Running command",
    "execute": "Running command",
    "Exec": "Running command",
    "search": "Searching",
    "grep": "Searching",
    "Grep": "Searching",
    "search_code": "Searching code",
    "find": "Searching",
    "Glob": "Finding files",
    "glob": "Finding files",
    "WebFetch": "Fetching URL",
    "fetch_url": "Fetching URL",
    "WebSearch": "Searching the web",
    "Task": "Spawning subagent",
}

# Argument keys we'll surface as the "detail" line under the action verb.
# Order matters — the first match wins.
_TOOL_DETAIL_KEYS: tuple[str, ...] = (
    "path", "file_path", "filename", "filepath",
    "relative_workspace_path", "target_file", "directory",
    "command", "cmd", "script",
    "pattern", "query", "search_term",
    "url",
    "name",
)


def _verb_for_tool(name: str) -> str:
    """Map a raw tool name to a human-readable verb phrase."""
    if name in _TOOL_VERBS:
        return _TOOL_VERBS[name]
    lname = name.lower()
    for prefix, verb in (
        ("read_", "Reading"), ("get_", "Reading"), ("view_", "Reading"),
        ("write_", "Writing"), ("create_", "Creating"), ("save_", "Writing"),
        ("edit_", "Editing"), ("update_", "Editing"), ("patch_", "Patching"),
        ("delete_", "Deleting"), ("remove_", "Deleting"),
        ("list_", "Listing"), ("show_", "Showing"),
        ("run_", "Running"), ("exec_", "Running"), ("execute_", "Running"),
        ("search_", "Searching"), ("find_", "Searching"), ("grep_", "Searching"),
        ("fetch_", "Fetching"), ("download_", "Fetching"),
    ):
        if lname.startswith(prefix):
            tail = lname[len(prefix):].replace("_", " ")
            return f"{verb} {tail}".strip()
    return f"Calling {name}"


def _detail_for_args(args: dict[str, Any]) -> str:
    """Pick the most informative argument value to display as a one-liner."""
    if not args:
        return ""
    for key in _TOOL_DETAIL_KEYS:
        if key in args and args[key] is not None:
            value = args[key]
            if not isinstance(value, str):
                value = str(value)
            value = value.strip()
            if value:
                return value
    # Fallback: first short string value, otherwise compact json.
    for value in args.values():
        if isinstance(value, str) and 0 < len(value) <= 200:
            return value
    try:
        return json.dumps(args, ensure_ascii=False, default=str)
    except Exception:
        return ""


def _truncate_inline(text: str, limit: int = 96) -> str:
    text = text.replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _format_size(n_bytes: int) -> str:
    """Human-readable byte count: ``412 B`` / ``1.2 KB`` / ``3.4 MB``."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes / 1024 / 1024:.1f} MB"


# ---------------------------------------------------------------------------
# CLIUI
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _StreamState:
    active: bool = False
    char_count: int = 0
    line_count: int = 0
    # Number of characters dropped by ``strip_tool_blocks`` during this
    # stream.  Used at end_stream to detect "phantom" tool calls — i.e.
    # cases where the model wrote ``<function_calls>…</function_calls>``
    # in plain text rather than emitting structured ``tool_calls`` in the
    # message envelope.  When that happens the engine never dispatches
    # anything, the model later complains "tool result didn't return", and
    # the user is left wondering what went wrong.  We surface a hint.
    stripped_chars: int = 0
    tool_calls_at_start: int = 0


@dataclass(slots=True)
class TurnStats:
    iterations: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    streamed_chars: int = 0
    elapsed_sec: float = 0.0
    started_at: float = field(default_factory=time.monotonic)


class CLIUI:
    """High-level UI surface for the Aether REPL.

    All output flows through this class so that:
      * spinners are stopped before tokens or panels are written;
      * tool-call panels share a consistent visual style;
      * the streaming flag is correctly reset across turns.
    """

    def __init__(self, console: Console, *, verbose: bool = False) -> None:
        self.console = console
        self.verbose = verbose
        self._status: Status | None = None
        self._stream = _StreamState()
        self._stream_buffer: list[str] = []
        self._live: Live | None = None
        self.stats = TurnStats()
        # Rotated every time a turn begins so the spinner doesn't feel
        # mechanical.  Mirrors Claude Code's "Forging / Channeling /
        # Pondering …" vibe.
        self._verb_cursor = 0

    # ------------------------------ status ---------------------------------

    def set_status(self, message: str, *, spinner: str = "dots") -> None:
        """Show or update the spinner with *message*."""
        if self._status is not None:
            try:
                self._status.update(status=f"[aether.status]{message}[/]", spinner=spinner)
            except Exception:
                self._status.update(status=f"[aether.status]{message}[/]")
            return
        self._status = self.console.status(
            f"[aether.status]{message}[/]",
            spinner=spinner,
            spinner_style=f"{AETHER_PRIMARY}",
        )
        self._status.start()

    def clear_status(self) -> None:
        if self._status is not None:
            try:
                self._status.stop()
            finally:
                self._status = None

    # ------------------------------ stream ---------------------------------

    def begin_stream(self) -> None:
        """Stop any active spinner and open the live markdown surface."""
        if self._stream.active:
            return
        self.clear_status()
        self.console.print()  # blank separator above the assistant block

        self._stream_buffer = []
        self._live = Live(
            self._render_assistant(""),
            console=self.console,
            refresh_per_second=10,
            transient=False,        # keep the final render in scrollback
            vertical_overflow="visible",
        )
        try:
            self._live.start()
        except Exception:           # noqa: BLE001 — Live is best-effort
            self._live = None

        self._stream.active = True
        self._stream.char_count = 0
        self._stream.line_count = 1
        self._stream.stripped_chars = 0
        self._stream.tool_calls_at_start = self.stats.tool_calls

    def stream_delta(self, delta: str) -> None:
        if not delta:
            return
        if not self._stream.active:
            self.begin_stream()

        self._stream_buffer.append(delta)
        self._stream.char_count += len(delta)
        self.stats.streamed_chars += len(delta)
        if "\n" in delta:
            self._stream.line_count += delta.count("\n")

        if self._live is not None:
            try:
                raw = "".join(self._stream_buffer)
                cleaned = strip_tool_blocks(raw)
                self._stream.stripped_chars = max(
                    0, len(raw) - len(cleaned)
                )
                self._live.update(self._render_assistant(cleaned))
            except Exception:        # noqa: BLE001 — never let render kill the stream
                pass
        else:
            # Fallback path when Live failed to start (e.g. no TTY).  Just
            # echo the cleaned delta to stdout.
            cleaned_delta = strip_tool_blocks(delta)
            if cleaned_delta:
                self.console.out(cleaned_delta, highlight=False, end="")

    def end_stream(self) -> None:
        if not self._stream.active:
            return

        stripped = self._stream.stripped_chars
        no_real_tool_dispatched = (
            self.stats.tool_calls == self._stream.tool_calls_at_start
        )

        if self._live is not None:
            cleaned = strip_tool_blocks("".join(self._stream_buffer))
            try:
                self._live.update(self._render_assistant(cleaned), refresh=True)
                self._live.stop()
            except Exception:        # noqa: BLE001
                pass
            self._live = None
        else:
            self.console.print()

        # Diagnostic: the model narrated tool calls in prose but the
        # engine never saw structured tool_calls.  Without this hint the
        # user sees the model say "tool didn't return" with no clue why.
        if stripped >= 80 and no_real_tool_dispatched:
            self._render_phantom_tool_hint(stripped)

        self._stream = _StreamState()
        self._stream_buffer = []

    def _render_phantom_tool_hint(self, stripped_chars: int) -> None:
        """Warn the user when ``<function_calls>`` was inline-only.

        Tells them in plain language: the model wrote XML tool tags into
        its prose but didn't actually populate the ``tool_calls`` field,
        so nothing ran.  Suggests a workaround.
        """
        line = Text()
        line.append("  ", style="")
        line.append(f"{icon('warn')} ", style=f"bold {AETHER_WARNING}")
        line.append("model emitted inline tool tags ", style=AETHER_WARNING)
        line.append(
            f"({stripped_chars} chars stripped)",
            style=f"dim {AETHER_DIM}",
        )
        line.append(
            "  — no structured tool_calls were dispatched.",
            style=AETHER_WARNING,
        )
        self.console.print(line)
        hint = Text(
            "    try /system to ask the model to use the structured "
            "tool_calls field, or pick a model that supports native tool use.",
            style=f"dim {AETHER_DIM}",
        )
        self.console.print(hint)

    # ---------------------------- conversation -----------------------------

    def render_user_echo(self, message: str) -> None:
        """Re-render the user's message in a styled block (multi-line form)."""
        head = Text()
        head.append(f"{icon('user')} ", style=f"bold {AETHER_ACCENT}")
        head.append("you", style=f"bold {AETHER_ACCENT}")
        head.append("  ", style="")
        head.append(message.replace("\n", "\n  "), style=AETHER_TEXT)
        self.console.print(head)

    def render_input_echo(self, message: str) -> None:
        """Compact one-shot echo of what the user just submitted.

        Called immediately after the input frame is erased so the scrollback
        keeps a readable trail of "what was typed → what came back".
        Multi-line input is preserved with a ``› `` hanging indent.

        We always emit a leading blank line so the user turn is visually
        separated from whatever the assistant produced before it — without
        this each new prompt felt glued to the previous response.
        """
        self.console.print()
        text = Text()
        text.append(f"{icon('user')} ", style=f"bold {AETHER_PRIMARY}")
        text.append(message.replace("\n", "\n  "), style=f"{AETHER_TEXT}")
        self.console.print(text)

    def render_assistant_block(self, content: str) -> None:
        """Render a non-streamed final assistant message (fallback path)."""
        cleaned = strip_tool_blocks(content or "")
        if not cleaned.strip():
            return
        self.console.print()
        self.console.print(self._render_assistant(cleaned))

    # ---- assistant body builder (shared by streaming + fallback) ---------

    def _render_assistant(self, text: str) -> ConsoleRenderable:
        """Build the ``● <markdown body>`` two-column renderable.

        The bullet sits in a 2-char left column so multi-line markdown
        wraps cleanly underneath it (matches Claude Code's transcript
        style).  An empty *text* still emits the bullet so the spinner
        clears even if the model produced nothing yet.
        """
        bullet = icon("assistant") or "●"

        if not text.strip():
            return Text(bullet, style=f"bold {AETHER_PRIMARY}")

        try:
            body: ConsoleRenderable = Markdown(
                text.strip(),
                code_theme="ansi_dark",
                inline_code_theme="ansi_dark",
            )
        except Exception:  # noqa: BLE001 — fall back to plain text
            body = Text(text, style=AETHER_TEXT)

        grid = Table.grid(expand=True, padding=(0, 1, 0, 0))
        grid.add_column(width=1, no_wrap=True, style=f"bold {AETHER_PRIMARY}")
        grid.add_column(overflow="fold")
        grid.add_row(bullet, body)
        return grid

    # ------------------------------ tools ----------------------------------
    #
    # Tool calls are rendered as a compact two-line block, Claude-Code style:
    #
    #     ● Reading file
    #       └ backend/harness/aether/runtime/recovery.py
    #
    # On success we stay silent — the call line above is the visual
    # record.  On error we drop one short error line indented under the
    # detail.  This keeps long tool sequences readable without drowning
    # the assistant prose in JSON panels.

    def render_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        *,
        tool_call_id: str | None = None,
    ) -> None:
        """Print the compact ``● <verb>`` + ``└ <detail>`` two-line block."""
        self.clear_status()
        if self._stream.active:
            self.end_stream()

        verb = _verb_for_tool(name)
        detail = _truncate_inline(_detail_for_args(args)) if args else ""

        head = Text()
        head.append(f"{icon('assistant')} ", style=f"bold {AETHER_PRIMARY}")
        head.append(verb, style=f"bold {AETHER_TEXT}")
        self.console.print(head)

        if detail:
            sub = Text()
            sub.append("  └ ", style=AETHER_DIM)
            sub.append(detail, style=f"dim {AETHER_TEXT}")
            self.console.print(sub)

        # Track id silently for /stats; we don't render it inline anymore.
        self.stats.tool_calls += 1
        if tool_call_id:
            # Stash for potential debug rendering; not user-visible.
            pass

    def render_tool_result(
        self,
        name: str,
        content: str,
        *,
        is_error: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Print a one-line indented summary of how the tool finished.

        Shape (always indented under the ``└`` detail line above):

            ✓ 412 B  · 23 lines        ← success with content
            ⚠ empty result             ← success but no content
            ✗ error  <reason…>         ← failure
        """
        self.clear_status()

        line = Text()
        line.append("    ", style="")  # indent under "  └ " detail column

        if is_error:
            self.stats.tool_errors += 1
            msg = _truncate_inline(content or "(no message)", limit=200)
            line.append(f"{icon('error')} ", style=f"bold {AETHER_ERROR}")
            line.append("error  ", style=f"bold {AETHER_ERROR}")
            line.append(msg, style=AETHER_ERROR)
            self.console.print(line)
            return

        # ---- success path ------------------------------------------------
        if not content:
            # An empty success is suspicious — the model often complains
            # "the tool didn't return" when this happens, so make it loud
            # enough that the user can see something is off.
            line.append(f"{icon('warn')} ", style=f"bold {AETHER_WARNING}")
            line.append("empty result", style=AETHER_WARNING)
            self.console.print(line)
            return

        size = _format_size(len(content))
        nlines = content.count("\n") + (0 if content.endswith("\n") else 1)
        line.append(f"{icon('success')} ", style=AETHER_SUCCESS)
        line.append(size, style=f"dim {AETHER_DIM}")
        if nlines > 1:
            line.append(f"  {icon('dot')} {nlines} lines", style=f"dim {AETHER_DIM}")
        self.console.print(line)

    def render_verbose_tool_result(
        self,
        name: str,
        content: str,
        *,
        is_error: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Old-school full panel for ``/verbose`` debugging.

        Not wired in by default — kept around so future ``/inspect``
        commands or unit tests can reach the verbose rendering path.
        """
        head = Text()
        if is_error:
            head.append(f"{icon('tool_error')} ", style=f"bold {AETHER_ERROR}")
            head.append("error  ", style=f"bold {AETHER_ERROR}")
        else:
            head.append(f"{icon('tool_done')} ", style=f"bold {AETHER_SUCCESS}")
            head.append("ok     ", style=f"bold {AETHER_SUCCESS}")
        head.append(name, style=f"{AETHER_TEXT}")
        head.append(f"  {icon('dot')} ", style=AETHER_DIM)
        head.append(f"{len(content)} chars", style=f"dim {AETHER_DIM}")

        preview, truncated = _truncate_for_preview(content)
        body = Text(preview, style=AETHER_DIM if not is_error else AETHER_ERROR)
        if truncated:
            body.append("\n[truncated]", style=f"dim {AETHER_DIM}")
        if not preview:
            body = Text("(empty result)", style=f"italic {AETHER_DIM}")

        self.console.print(head)
        self.console.print(body)
        self.console.print()

    # ----------------------------- messages --------------------------------

    def info(self, message: str) -> None:
        line = Text()
        line.append(f"{icon('info')} ", style=AETHER_ACCENT)
        line.append(message, style=AETHER_DIM)
        self.console.print(line)

    def warn(self, message: str) -> None:
        line = Text()
        line.append(f"{icon('warn')} ", style=AETHER_WARNING)
        line.append(message, style=AETHER_WARNING)
        self.console.print(line)

    def error(self, message: str) -> None:
        line = Text()
        line.append(f"{icon('error')} ", style=AETHER_ERROR)
        line.append(message, style=AETHER_ERROR)
        self.console.print(line)

    def success(self, message: str) -> None:
        line = Text()
        line.append(f"{icon('success')} ", style=AETHER_SUCCESS)
        line.append(message, style=AETHER_SUCCESS)
        self.console.print(line)

    # --------------------------- turn lifecycle ----------------------------

    # Pool of evocative verbs we rotate through across turns; each one
    # gets a trailing ellipsis when shown in the spinner so it reads as
    # an active state.
    _SPINNER_VERBS: tuple[str, ...] = (
        "Thinking", "Pondering", "Forging", "Channeling",
        "Reasoning", "Deliberating", "Synthesising", "Conjuring",
    )

    def _next_spinner_verb(self) -> str:
        verb = self._SPINNER_VERBS[self._verb_cursor % len(self._SPINNER_VERBS)]
        self._verb_cursor += 1
        return verb

    def begin_turn(self) -> None:
        self.stats = TurnStats()
        self.set_status(f"{self._next_spinner_verb()}…")

    def end_turn(self, *, status: str, exit_reason: str, iterations: int, error: str | None = None) -> None:
        self.clear_status()
        if self._stream.active:
            self.end_stream()

        self.stats.iterations = iterations
        self.stats.elapsed_sec = max(0.0, time.monotonic() - self.stats.started_at)

        # Detect "did the agent actually do anything?".  When the model
        # produced a short reply with zero tool dispatches we surface a
        # diagnostic — that's the signature of a model that *meant* to
        # call a tool (markdown ```bash``` block, "I'll take a look at
        # your project. ls", etc.) but never populated the structured
        # ``tool_calls`` field, so the engine had nothing to run.
        if (
            status == "COMPLETED"
            and self.stats.tool_calls == 0
            and self.stats.streamed_chars > 0
            and self.stats.streamed_chars < 200
        ):
            self._render_idle_turn_hint()

        # Render the always-on one-line turn footer — Claude-Code style.
        # We used to be silent on success which made it impossible to
        # tell "model finished" from "REPL ate the response".
        line = Text()
        line.append("  ", style="")
        if status == "COMPLETED":
            line.append(f"{icon('success')} done  ", style=f"bold {AETHER_SUCCESS}")
        elif status == "INTERRUPTED":
            line.append(f"{icon('interrupt')} interrupted  ", style=f"bold {AETHER_WARNING}")
        elif status == "MAX_ITERATIONS":
            line.append(f"{icon('warn')} max iterations  ", style=f"bold {AETHER_WARNING}")
        else:
            line.append(f"{icon('error')} failed  ", style=f"bold {AETHER_ERROR}")

        line.append(f"{icon('iter')}{iterations}  ", style=AETHER_DIM)
        if self.stats.tool_calls:
            line.append(
                f"{icon('tool')}{self.stats.tool_calls}"
                f"{'/' + str(self.stats.tool_errors) + ' err' if self.stats.tool_errors else ''}  ",
                style=AETHER_DIM,
            )
        line.append(f"{self.stats.elapsed_sec:.1f}s", style=AETHER_DIM)
        # ``exit_reason`` is technical jargon ("natural_completion",
        # "tool_calls", …) that's only useful when debugging the engine —
        # show it in verbose mode and on non-success terminal states.
        if self.verbose or status != "COMPLETED":
            line.append(
                f"  {icon('dot')} {exit_reason.lower().replace('_', ' ')}",
                style=f"dim {AETHER_DIM}",
            )

        self.console.print(line)
        if error:
            self.error(error)

    def _render_idle_turn_hint(self) -> None:
        """Warn when the model said something but never dispatched a tool.

        Triggered by short replies with zero tool calls — the canonical
        case is a model writing "I'll take a look" then a markdown code
        block of the command it would run, instead of populating the
        structured ``tool_calls`` field.  Without this hint the user only
        sees a few words then the prompt comes back, with no clue why
        "nothing happened".
        """
        warn = Text()
        warn.append("  ", style="")
        warn.append(f"{icon('warn')} ", style=f"bold {AETHER_WARNING}")
        warn.append(
            "no tools were dispatched this turn",
            style=AETHER_WARNING,
        )
        warn.append(
            "  — if you expected one, the model likely typed the command "
            "into prose instead of calling a tool.",
            style=f"dim {AETHER_DIM}",
        )
        self.console.print(warn)

    # ------------------------------ utility --------------------------------

    def hr(self) -> None:
        self.console.rule(style=AETHER_BORDER)

    def blank(self) -> None:
        self.console.print()

    def make_stream_callback(self):
        """Return a closure suitable for ``EngineRequest.stream_callback``."""
        def _cb(delta: str) -> None:
            self.stream_delta(delta)
        return _cb

    # ---------------------------- /tools listing ---------------------------

    def render_tool_table(self, descriptors: list[Any]) -> None:
        if not descriptors:
            self.info("No tools registered.")
            return
        table = Table(
            title=Text("Available tools", style=f"bold {AETHER_PRIMARY}"),
            border_style=AETHER_BORDER,
            header_style=f"bold {AETHER_DIM}",
            show_lines=False,
        )
        table.add_column("name", style=f"{AETHER_TEXT}")
        table.add_column("description", style=f"{AETHER_DIM}")
        for desc in descriptors:
            name = getattr(desc, "name", "?")
            description = getattr(desc, "description", "") or ""
            if len(description) > 80:
                description = description[:77] + "…"
            table.add_row(name, description)
        self.console.print(table)

    # ------------------------ banner / boot helpers ------------------------

    def boot_line(self, message: str) -> None:
        line = Text()
        line.append(f"{icon('spark')} ", style=AETHER_PRIMARY)
        line.append(message, style=AETHER_DIM)
        self.console.print(line)
