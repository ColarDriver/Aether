"""Coalesced tool-call group renderer for the Aether REPL.

Mirrors claude-code's ``CollapsedReadSearchContent`` (see
``/workspace/open-claude-code/src/components/messages/CollapsedReadSearchContent.tsx``):
consecutive tool dispatches inside a single LLM iteration are bucketed by
category and summarised on a single rolling line:

  ● Searching for 2 patterns, reading 1 file, listing 1 directory…
    ⎿ pattern in src/

When the iteration boundary is crossed (``before_llm`` fires for the next
iteration, or the turn ends), the headline is rewritten in past tense and
flushed to scrollback:

  ● Searched for 2 patterns, read 1 file, listed 1 directory

The tracker is intentionally engine-thread-mutable + Live-thread-readable:
the worst case for a missing lock is one stale frame, which is benign.
Python's GIL guarantees atomic primitive reads/writes which is enough for
the small dict / int / str fields we touch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from rich.console import ConsoleRenderable
from rich.text import Text

from aether.cli.theme import (
    AETHER_DIM,
    AETHER_PRIMARY,
    AETHER_TEXT,
    AETHER_WARNING,
    icon,
)


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

class ToolCategory(str, Enum):
    """Stable bucket id used for verb selection + headline composition.

    Ordering here drives the order in which categories appear in the
    rendered headline (``Searching for X, reading Y, listing Z``).
    """

    SEARCH = "search"
    READ = "read"
    LIST = "list"
    WRITE = "write"
    EDIT = "edit"
    BASH = "bash"
    WEB = "web"
    SUBAGENT = "subagent"
    MCP = "mcp"
    OTHER = "other"


# Map raw tool names → category.  The tool registry uses many spellings
# (``read_file`` vs. ``Read`` vs. ``mcp__filesystem__read_file``), so we
# keep an explicit table plus a prefix-based heuristic in
# ``_category_for``.
_NAME_TO_CATEGORY: dict[str, ToolCategory] = {
    # --- read ---
    "read_file": ToolCategory.READ,
    "read": ToolCategory.READ,
    "Read": ToolCategory.READ,
    "view_file": ToolCategory.READ,
    "ViewFile": ToolCategory.READ,
    "cat": ToolCategory.READ,
    # --- search ---
    "search": ToolCategory.SEARCH,
    "grep": ToolCategory.SEARCH,
    "Grep": ToolCategory.SEARCH,
    "search_code": ToolCategory.SEARCH,
    "find": ToolCategory.SEARCH,
    "Glob": ToolCategory.SEARCH,
    "glob": ToolCategory.SEARCH,
    "rg": ToolCategory.SEARCH,
    # --- list ---
    "list_directory": ToolCategory.LIST,
    "ListDirectory": ToolCategory.LIST,
    "ls": ToolCategory.LIST,
    "list": ToolCategory.LIST,
    "tree": ToolCategory.LIST,
    # --- write ---
    "write_file": ToolCategory.WRITE,
    "WriteFile": ToolCategory.WRITE,
    "Write": ToolCategory.WRITE,
    "create_file": ToolCategory.WRITE,
    "save_file": ToolCategory.WRITE,
    # --- edit ---
    "edit_file": ToolCategory.EDIT,
    "Edit": ToolCategory.EDIT,
    "EditFile": ToolCategory.EDIT,
    "patch": ToolCategory.EDIT,
    "apply_patch": ToolCategory.EDIT,
    "delete_file": ToolCategory.EDIT,
    "DeleteFile": ToolCategory.EDIT,
    "str_replace": ToolCategory.EDIT,
    "StrReplace": ToolCategory.EDIT,
    # --- bash ---
    "run_bash": ToolCategory.BASH,
    "Bash": ToolCategory.BASH,
    "bash": ToolCategory.BASH,
    "shell": ToolCategory.BASH,
    "execute": ToolCategory.BASH,
    "Exec": ToolCategory.BASH,
    "execute_command": ToolCategory.BASH,
    "run_command": ToolCategory.BASH,
    # --- web ---
    "WebFetch": ToolCategory.WEB,
    "fetch_url": ToolCategory.WEB,
    "WebSearch": ToolCategory.WEB,
    # --- subagent ---
    "Task": ToolCategory.SUBAGENT,
    "spawn_agent": ToolCategory.SUBAGENT,
}


# Verb pair per category: (first-position present, follow-on present).
# Mirrors ``CollapsedReadSearchContent.tsx`` lines 345-413 where the first
# verb in the headline gets capitalised and follow-ons stay lowercase.
_CATEGORY_VERBS_PRESENT: dict[ToolCategory, tuple[str, str]] = {
    ToolCategory.SEARCH:   ("Searching for",  "searching for"),
    ToolCategory.READ:     ("Reading",        "reading"),
    ToolCategory.LIST:     ("Listing",        "listing"),
    ToolCategory.WRITE:    ("Writing",        "writing"),
    ToolCategory.EDIT:     ("Editing",        "editing"),
    ToolCategory.BASH:     ("Running",        "running"),
    ToolCategory.WEB:      ("Fetching",       "fetching"),
    ToolCategory.SUBAGENT: ("Spawning",       "spawning"),
    ToolCategory.MCP:      ("Querying",       "querying"),
    ToolCategory.OTHER:    ("Calling",        "calling"),
}

_CATEGORY_VERBS_PAST: dict[ToolCategory, tuple[str, str]] = {
    ToolCategory.SEARCH:   ("Searched for",   "searched for"),
    ToolCategory.READ:     ("Read",           "read"),
    ToolCategory.LIST:     ("Listed",         "listed"),
    ToolCategory.WRITE:    ("Wrote",          "wrote"),
    ToolCategory.EDIT:     ("Edited",         "edited"),
    ToolCategory.BASH:     ("Ran",            "ran"),
    ToolCategory.WEB:      ("Fetched",        "fetched"),
    ToolCategory.SUBAGENT: ("Spawned",        "spawned"),
    ToolCategory.MCP:      ("Queried",        "queried"),
    ToolCategory.OTHER:    ("Called",         "called"),
}

# (singular, plural) noun for each category.
_CATEGORY_NOUN: dict[ToolCategory, tuple[str, str]] = {
    ToolCategory.SEARCH:   ("pattern",   "patterns"),
    ToolCategory.READ:     ("file",      "files"),
    ToolCategory.LIST:     ("directory", "directories"),
    ToolCategory.WRITE:    ("file",      "files"),
    ToolCategory.EDIT:     ("file",      "files"),
    ToolCategory.BASH:     ("command",   "commands"),
    ToolCategory.WEB:      ("URL",       "URLs"),
    ToolCategory.SUBAGENT: ("subagent",  "subagents"),
    ToolCategory.MCP:      ("call",      "calls"),
    ToolCategory.OTHER:    ("call",      "calls"),
}


def _canonical_tool_name(name: str) -> str:
    """Strip ``mcp__server__`` / ``namespace.`` / ``ns:`` prefixes."""
    normalized = (name or "").strip()
    if not normalized:
        return normalized
    if "__" in normalized:
        normalized = normalized.split("__")[-1]
    if "." in normalized:
        normalized = normalized.split(".")[-1]
    if ":" in normalized:
        normalized = normalized.split(":")[-1]
    return normalized


def category_for(name: str) -> ToolCategory:
    """Determine the :class:`ToolCategory` for a raw tool name."""
    canonical = _canonical_tool_name(name)
    if canonical in _NAME_TO_CATEGORY:
        return _NAME_TO_CATEGORY[canonical]
    # Names that originate from an MCP server (``mcp__server__tool``) are
    # bucketed as MCP only when the canonical (post-strip) name didn't
    # match a more specific category — that way ``mcp__filesystem__
    # read_file`` still reads as READ, not MCP.
    if name and "mcp__" in name.lower():
        return ToolCategory.MCP

    lname = canonical.lower()
    for prefix, category in (
        ("read_", ToolCategory.READ),    ("get_", ToolCategory.READ),
        ("view_", ToolCategory.READ),
        ("write_", ToolCategory.WRITE),  ("create_", ToolCategory.WRITE),
        ("save_", ToolCategory.WRITE),
        ("edit_", ToolCategory.EDIT),    ("update_", ToolCategory.EDIT),
        ("patch_", ToolCategory.EDIT),   ("delete_", ToolCategory.EDIT),
        ("remove_", ToolCategory.EDIT),
        ("list_", ToolCategory.LIST),    ("show_", ToolCategory.LIST),
        ("run_", ToolCategory.BASH),     ("exec_", ToolCategory.BASH),
        ("execute_", ToolCategory.BASH),
        ("search_", ToolCategory.SEARCH),("find_", ToolCategory.SEARCH),
        ("grep_", ToolCategory.SEARCH),
        ("fetch_", ToolCategory.WEB),    ("download_", ToolCategory.WEB),
    ):
        if lname.startswith(prefix):
            return category
    return ToolCategory.OTHER


# ---------------------------------------------------------------------------
# Hint formatting — "⎿ <hint>" line shown only while the call is in flight.
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int = 80) -> str:
    text = (text or "").replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def hint_for_call(name: str, args: dict[str, Any]) -> str:
    """Return the indented hint string shown beneath an active group line.

    Each category has its own preferred shape so the user can tell at a
    glance *what* the model is currently looking at:

      bash:    ``$ ls -la``
      search:  ``"foo" in src/``
      read:    ``backend/harness/aether/cli/ui.py``
      web:     ``https://example.com/foo``
    """
    if not args:
        return ""
    cat = category_for(name)

    if cat == ToolCategory.BASH:
        cmd = args.get("command") or args.get("cmd") or args.get("script")
        if cmd:
            return f"$ {_truncate(str(cmd))}"

    if cat == ToolCategory.SEARCH:
        pattern = (
            args.get("pattern")
            or args.get("query")
            or args.get("search_term")
        )
        path = args.get("path") or args.get("directory") or args.get("file")
        if pattern and path:
            return f'"{_truncate(str(pattern), 40)}" in {_truncate(str(path), 40)}'
        if pattern:
            return f'"{_truncate(str(pattern))}"'
        if path:
            return _truncate(str(path))

    if cat == ToolCategory.WEB:
        url = args.get("url")
        if url:
            return _truncate(str(url))

    if cat == ToolCategory.SUBAGENT:
        desc = (
            args.get("description")
            or args.get("prompt")
            or args.get("name")
        )
        if desc:
            return _truncate(str(desc))

    # Fallback: prefer path-like keys, then short string values.
    for key in (
        "path", "file_path", "filename", "filepath",
        "relative_workspace_path", "target_file", "directory",
        "url", "name", "command", "cmd", "script",
        "pattern", "query",
    ):
        if key in args and args[key] is not None:
            value = str(args[key]).strip()
            if value:
                return _truncate(value)

    for value in args.values():
        if isinstance(value, str) and 0 < len(value) <= 80:
            return _truncate(value)

    return ""


# ---------------------------------------------------------------------------
# ToolGroup — single coalesced row's worth of state
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ToolGroup:
    """Aggregated state for one or more dispatches inside a single iteration.

    Survives until ``ToolGroupTracker.flush_active`` is called, which
    happens at iteration boundaries (``before_llm``) and on turn end.
    """

    counts: dict[ToolCategory, int] = field(default_factory=dict)
    last_hint: str = ""
    last_category: ToolCategory = ToolCategory.OTHER
    in_flight: int = 0
    has_error: bool = False
    total_calls: int = 0

    def add_call(self, name: str, args: dict[str, Any]) -> None:
        cat = category_for(name)
        self.counts[cat] = self.counts.get(cat, 0) + 1
        self.last_category = cat
        hint = hint_for_call(name, args)
        if hint:
            self.last_hint = hint
        self.in_flight += 1
        self.total_calls += 1

    def finish_call(self, *, is_error: bool = False) -> None:
        if self.in_flight > 0:
            self.in_flight -= 1
        if is_error:
            self.has_error = True

    @property
    def is_active(self) -> bool:
        return self.in_flight > 0

    def render_headline(self, *, active: bool) -> Text:
        verbs = _CATEGORY_VERBS_PRESENT if active else _CATEGORY_VERBS_PAST
        nouns = _CATEGORY_NOUN
        line = Text()
        bullet = icon("assistant") or "●"
        line.append(f"{bullet} ", style=f"bold {AETHER_PRIMARY}")

        ordered = [c for c in ToolCategory if self.counts.get(c, 0) > 0]
        for i, cat in enumerate(ordered):
            count = self.counts[cat]
            verb_first, verb_more = verbs[cat]
            verb = verb_first if i == 0 else verb_more
            noun = nouns[cat][0] if count == 1 else nouns[cat][1]
            if i:
                line.append(", ", style=AETHER_DIM)
            line.append(f"{verb} ", style=AETHER_TEXT)
            line.append(str(count), style=f"bold {AETHER_TEXT}")
            line.append(f" {noun}", style=AETHER_TEXT)

        if active:
            line.append("…", style=AETHER_DIM)
        elif self.has_error:
            line.append(f"  {icon('warn')} ", style=f"bold {AETHER_WARNING}")
            line.append("with errors", style=AETHER_WARNING)
        return line

    def render_hint(self) -> Text | None:
        if not self.last_hint:
            return None
        line = Text()
        line.append("  ⎿ ", style=AETHER_DIM)
        line.append(self.last_hint, style=f"dim {AETHER_TEXT}")
        return line


# ---------------------------------------------------------------------------
# ToolGroupTracker — owned by CLIUI
# ---------------------------------------------------------------------------

class ToolGroupTracker:
    """Bucket tool dispatches per LLM iteration into a single rolling line.

    The tracker owns at most one :class:`ToolGroup` at a time:

      * ``begin_iteration`` flushes the current group's past-tense
        headline to scrollback (via the *sink* callback) and starts a
        clean slate for the next iteration.
      * ``start_call`` / ``finish_call`` append to the active group.
      * ``flush_active`` is called explicitly from ``end_turn`` so the
        last iteration's summary always lands in the transcript.
      * ``discard_active`` is the error-recovery path — a fatal during
        an iteration should not print a half-baked past-tense line.
    """

    def __init__(
        self,
        *,
        sink: Callable[[ConsoleRenderable], None],
    ) -> None:
        self._sink = sink
        self._active: ToolGroup | None = None

    @property
    def active(self) -> ToolGroup | None:
        return self._active

    def has_active(self) -> bool:
        return self._active is not None

    # --- lifecycle ----------------------------------------------------------

    def begin_iteration(self) -> None:
        """Flush the previous iteration's group (if any)."""
        self.flush_active()

    def start_call(self, name: str, args: dict[str, Any]) -> None:
        if self._active is None:
            self._active = ToolGroup()
        self._active.add_call(name, args)

    def finish_call(self, *, is_error: bool = False) -> None:
        if self._active is None:
            return
        self._active.finish_call(is_error=is_error)

    def flush_active(self) -> None:
        if self._active is None:
            return
        if self._active.total_calls > 0:
            try:
                self._sink(self._active.render_headline(active=False))
            except Exception:  # noqa: BLE001 — never let render kill engine
                pass
        self._active = None

    def discard_active(self) -> None:
        self._active = None
