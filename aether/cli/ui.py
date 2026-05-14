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

from rich.console import Console, ConsoleOptions, ConsoleRenderable, Group, RenderResult
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text

from aether.cli.activity import (
    ActivityBar,
    MODE_THINKING,
    MODE_TOOL_USE,
    TurnState,
)
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
    icon,
)
from aether.cli.tool_groups import ToolCategory, ToolGroupTracker, category_for


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_TOOL_RESULT_PREVIEW = 1200
_MAX_TOOL_ARGS_PREVIEW = 800
_MAX_RAW_TOOL_PREVIEW = 1600

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
    "reasoning_effort",
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
_BRACKET_TOOL_LINE_RE = re.compile(
    r"(?mi)^[ \t]*\[tool:\s*([^\]\n]+?)\][ \t]*(?:\n|$)"
)
_BRACKET_TOOL_OPEN_RE = re.compile(
    r"(?i)\[tool:\s*[^\]\n]*$"
)
# A non-standard "function-equals" inline syntax some Kimi-style
# models emit when they fail to populate the structured ``tool_calls``
# field: ``<function=execute_command> <cmd>`` (with ``=`` instead of
# ``name="…"``, *and* no closing tag).  The body extends until the
# next ``<function=`` or end-of-text.  We capture (name, body) so the
# diagnostic can surface the attempted command and ``strip_tool_blocks``
# can drop both the tag and the prose body that follows it.
_FUNCTION_EQ_TAG_RE = re.compile(
    r"(?is)<function=([^>\s/]+)>(.*?)(?=<function=|</function|\Z)"
)
_FUNCTION_EQ_OPEN_RE = re.compile(r"(?i)<function=[^>\s/]+>")
# Partial open tag at the very end of the buffer (e.g. ``<function=`` while
# the model is mid-token) — anchor with ``$`` so we only strip when it's
# the trailing fragment, not an embedded ``<funct`` inside prose.
_FUNCTION_EQ_PARTIAL_RE = re.compile(r"(?i)<function=?[^>]*$")
_INLINE_INVOKE_OPEN_RE = re.compile(
    r'(?is)<invoke\b[^>]*name="([^"]+)"[^>]*>'
)
_INLINE_PARAMETER_RE = re.compile(
    r'(?is)<parameter\b[^>]*name="([^"]+)"[^>]*>([^<\n]*)'
)
_PARTIAL_INLINE_TAG_NAMES = tuple(
    sorted(set(_TOOL_TAGS + ("parameter",)), key=len, reverse=True)
)


def _strip_bracket_tool_lines(text: str) -> str:
    cleaned = _BRACKET_TOOL_LINE_RE.sub("", text)
    open_match = _BRACKET_TOOL_OPEN_RE.search(cleaned)
    if open_match is not None:
        cleaned = cleaned[: open_match.start()]
    return cleaned


def _strip_partial_inline_xml_tag(text: str) -> str:
    lt_index = text.rfind("<")
    if lt_index < 0:
        return text

    tail = text[lt_index + 1 :]
    if ">" in tail:
        return text

    normalized = tail.strip().lower().lstrip("/")
    if ":" in normalized:
        normalized = normalized.split(":")[-1]
    normalized = re.split(r"[^a-z0-9_-]", normalized, maxsplit=1)[0]
    if not normalized:
        return text[:lt_index]

    if any(tag.startswith(normalized) for tag in _PARTIAL_INLINE_TAG_NAMES):
        return text[:lt_index]
    return text


def _extract_bracket_tool_names(text: str) -> list[str]:
    names: list[str] = []
    for match in _BRACKET_TOOL_LINE_RE.finditer(text):
        name = str(match.group(1) or "").strip()
        if name:
            names.append(name)
    return names


def _canonical_tool_name(name: str) -> str:
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


def _inline_activity_from_invoke(text: str) -> str:
    matches = list(_INLINE_INVOKE_OPEN_RE.finditer(text))
    if not matches:
        return ""

    latest = matches[-1]
    invoke_name = _canonical_tool_name(str(latest.group(1) or "").strip())
    if not invoke_name:
        return ""

    tail = text[latest.end() :]
    args: dict[str, Any] = {}
    for match in _INLINE_PARAMETER_RE.finditer(tail):
        key = str(match.group(1) or "").strip()
        value = str(match.group(2) or "").strip()
        if key and value:
            args[key] = value

    verb = _verb_for_tool(invoke_name)
    detail = _truncate_inline(_detail_for_args(args)) if args else ""
    if detail:
        return f"{verb}…  {detail}"
    return f"{verb}…"


def _inline_activity_from_text(text: str) -> str:
    invoke_activity = _inline_activity_from_invoke(text)
    if invoke_activity:
        return invoke_activity
    names = _extract_bracket_tool_names(text)
    if names:
        return f"{_verb_for_tool(_canonical_tool_name(names[-1]))}…"
    # Last resort: pick up the latest ``<function=NAME>`` so the bar
    # shows the right verb while the user watches the model "type" the
    # attempted invocation in prose.  Without this they'd see a generic
    # "Pondering…" while a clearly-actionable command was on screen.
    eq_matches = list(_FUNCTION_EQ_TAG_RE.finditer(text))
    if eq_matches:
        last = eq_matches[-1]
        name = _canonical_tool_name((last.group(1) or "").strip())
        body = (last.group(2) or "").strip()
        if name:
            verb = _verb_for_tool(name)
            if body:
                detail = _truncate_inline(" ".join(body.split()))
                return f"{verb}…  {detail}"
            return f"{verb}…"
    return ""


def _has_xml_style_inline_tool_markup(text: str) -> bool:
    return _TOOL_OPEN_RE.search(text) is not None


def strip_tool_blocks(text: str) -> str:
    """Remove inline ``<tool_call>…</tool_call>``-style blocks from *text*.

    The function is conservative: complete blocks are deleted outright;
    a partial open tag (no matching close yet) hides everything from the
    open tag onward, so the user never sees half-rendered JSON before the
    closing tag arrives in the stream.  Trailing blank-line runs are
    collapsed to keep the prose tidy.

    Also strips the non-standard ``<function=NAME> <body>`` syntax
    Kimi-class models occasionally emit.  These tags have *no* closing
    pair, so each ``<function=…>`` consumes everything up to the next
    ``<function=`` or end-of-text — we erase the whole run so the user
    sees clean prose instead of a half-rendered tool intent.
    """
    cleaned = _TOOL_BLOCK_RE.sub("", text)
    open_match = _TOOL_OPEN_RE.search(cleaned)
    if open_match is not None:
        cleaned = cleaned[: open_match.start()]
    cleaned = _strip_bracket_tool_lines(cleaned)
    cleaned = _FUNCTION_EQ_TAG_RE.sub("", cleaned)
    cleaned = _FUNCTION_EQ_PARTIAL_RE.sub("", cleaned)
    cleaned = _strip_partial_inline_xml_tag(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


# Match ``<thinking>…</thinking>`` (with any namespace prefix) so we can
# route it to the reasoning channel.  This is the *one* tag inside
# ``_TOOL_TAGS`` that carries human-readable narration — surfacing its
# contents in the activity bar rescues "↑ N tokens but body is blank"
# turns where the model burned its budget thinking out loud.
_THINKING_BLOCK_RE = re.compile(
    r"<\s*(?:[\w-]+:)?thinking\b[^>]*>(.*?)</\s*(?:[\w-]+:)?thinking\s*>",
    re.DOTALL | re.IGNORECASE,
)


def extract_thinking_blocks(text: str) -> str:
    """Return the concatenated content of any ``<thinking>…</thinking>`` blocks.

    Used by :class:`CLIUI` to populate the reasoning excerpt under the
    activity bar so the user can see the model is producing chain-of-
    thought even when the visible body is empty.  Returns an empty string
    when the text carries no thinking blocks.
    """
    if not text or "thinking" not in text.lower():
        return ""
    parts: list[str] = []
    for match in _THINKING_BLOCK_RE.finditer(text):
        body = (match.group(1) or "").strip()
        if body:
            parts.append(body)
    return "\n".join(parts)


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


def _inline_tool_preview(text: str, limit: int = _MAX_RAW_TOOL_PREVIEW) -> tuple[str, bool]:
    """Return a compact preview of raw inline tool-tag output."""
    if not text:
        return "", False

    collapsed = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not collapsed:
        return "", False

    return _truncate_for_preview(collapsed, limit=limit)


def _should_render_inline_tool_preview(cleaned_text: str, stripped_chars: int) -> bool:
    """Show raw preview only when stripped tool-tag output dominates."""
    visible_chars = len(cleaned_text.strip())
    if stripped_chars < 80:
        return False
    if visible_chars == 0:
        return True
    return stripped_chars >= max(160, visible_chars * 3)


# ---------------------------------------------------------------------------
# Compact tool-call rendering — "● Reading file" + "└ <detail>"
# ---------------------------------------------------------------------------

# Verb mapping used to convert raw tool names into human-readable action
# phrases.  Tools whose name isn't in the table fall through to the
# heuristic in :func:`_verb_for_tool` below.
#
# Single-word verbs (claude / codex style) so the call header reads as
# one line: ``● Read backend/harness/aether/cli/ui.py``.  Past-tense
# base form works as both "currently doing this" and "this happened",
# which lets us print the header at dispatch time without a verb-
# rewrite when the result lands.
_TOOL_VERBS: dict[str, str] = {
    "read_file": "Read",
    "read": "Read",
    "Read": "Read",
    "view_file": "Read",
    "ViewFile": "Read",
    "list_directory": "List",
    "execute_command": "Bash",
    "ListDirectory": "List",
    "ls": "List",
    "list": "List",
    "write_file": "Write",
    "WriteFile": "Write",
    "Write": "Write",
    "create_file": "Write",
    "edit_file": "Edit",
    "Edit": "Edit",
    "EditFile": "Edit",
    "patch": "Edit",
    "apply_patch": "Edit",
    "delete_file": "Delete",
    "DeleteFile": "Delete",
    "run_bash": "Bash",
    "Bash": "Bash",
    "bash": "Bash",
    "shell": "Bash",
    "execute": "Bash",
    "Exec": "Bash",
    "search": "Search",
    "grep": "Search",
    "Grep": "Search",
    "search_code": "Search",
    "find": "Search",
    "Glob": "Glob",
    "glob": "Glob",
    "WebFetch": "Fetch",
    "fetch_url": "Fetch",
    "WebSearch": "Search",
    "Task": "Task",
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
    """Map a raw tool name to a single-word action verb.

    Falls through to a prefix heuristic for tools we don't know
    explicitly (``read_secret_dossier`` → ``Read``,
    ``execute_workflow`` → ``Bash``, …) and finally to the raw
    name for fully unknown tools.
    """
    name = _canonical_tool_name(name)
    if name in _TOOL_VERBS:
        return _TOOL_VERBS[name]
    lname = name.lower()
    for prefix, verb in (
        ("read_", "Read"), ("get_", "Read"), ("view_", "Read"),
        ("write_", "Write"), ("create_", "Write"), ("save_", "Write"),
        ("edit_", "Edit"), ("update_", "Edit"), ("patch_", "Edit"),
        ("delete_", "Delete"), ("remove_", "Delete"),
        ("list_", "List"), ("show_", "List"),
        ("run_", "Bash"), ("exec_", "Bash"), ("execute_", "Bash"),
        ("search_", "Search"), ("find_", "Search"), ("grep_", "Search"),
        ("fetch_", "Fetch"), ("download_", "Fetch"),
    ):
        if lname.startswith(prefix):
            return verb
    return name or "Call"


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
# Tool-result preview — codex-style "    ⎿ <line>" tree under each call
# ---------------------------------------------------------------------------

# Strip ANSI CSI sequences before display.  Many bash tools (git, grep,
# ls --color=auto …) emit colour codes that Rich would otherwise render
# as visible ``\x1b[31m`` literals, which is uglier than just losing the
# colour.  We deliberately keep the strip narrow (CSI only) so any other
# control bytes the tool meant to send round-trip unchanged.
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# Output preview budgets.  The whole point of the per-call "    ⎿ …"
# block is to give a quick glance at what the tool returned without
# drowning the transcript in long stdout dumps — the user can always
# re-run with the actual command if they want the full text.
_OUTPUT_PREVIEW_MAX_LINES = 10
_OUTPUT_PREVIEW_MAX_CHARS = 1200
_OUTPUT_LINE_MAX_CHARS = 200

# Indent strings shared by success + error rendering of the result tree.
_RESULT_PREFIX = "    ⎿ "      # 4 spaces + ⎿ + 1 space → content at col 6
_RESULT_CONT = "      "         # 6 spaces — aligns continuation lines


def _strip_ansi(text: str) -> str:
    if not text:
        return text
    return _ANSI_CSI_RE.sub("", text)


def _format_output_preview(content: str) -> tuple[list[str], str]:
    """Return ``(lines, summary)`` for rendering tool output as a tree.

    ``lines`` are the verbatim rows to print under the ``  ⎿ `` prefix
    (already truncated to ``_OUTPUT_PREVIEW_MAX_LINES`` and clipped at
    ``_OUTPUT_LINE_MAX_CHARS`` per row).  ``summary`` is an optional
    trailing dim row like ``… (+12 more lines, 4.5 KB)`` indicating
    how much was hidden — empty string when nothing was truncated.

    Empty / whitespace-only content collapses to ``([], "")`` so the
    caller can stay silent when the tool produced no output.
    """
    if not content:
        return [], ""
    cleaned = _strip_ansi(content).rstrip()
    if not cleaned:
        return [], ""

    full_chars = len(cleaned)
    full_lines = cleaned.split("\n")
    total_lines = len(full_lines)

    # Line-count truncation — most common shape for `git log`, `ls -la`,
    # large `grep` dumps, etc.
    if total_lines > _OUTPUT_PREVIEW_MAX_LINES:
        display_lines = full_lines[:_OUTPUT_PREVIEW_MAX_LINES]
        more_lines = total_lines - _OUTPUT_PREVIEW_MAX_LINES
        summary = (
            f"… (+{more_lines} more line{'s' if more_lines != 1 else ''}, "
            f"{_format_size(full_chars)})"
        )
    # Character-count truncation — single very long line (eg. base64
    # blob, json dump).  Cut at a line boundary if possible so the
    # truncation marker doesn't land mid-word.
    elif full_chars > _OUTPUT_PREVIEW_MAX_CHARS:
        truncated = cleaned[:_OUTPUT_PREVIEW_MAX_CHARS]
        last_nl = truncated.rfind("\n")
        if last_nl > 0:
            truncated = truncated[:last_nl]
        display_lines = truncated.split("\n")
        summary = f"… (truncated, {_format_size(full_chars)} total)"
    else:
        display_lines = full_lines
        summary = ""

    # Per-line clip so a single 5KB line doesn't blow the terminal.
    clipped: list[str] = []
    for line in display_lines:
        if len(line) > _OUTPUT_LINE_MAX_CHARS:
            clipped.append(line[: _OUTPUT_LINE_MAX_CHARS - 1] + "…")
        else:
            clipped.append(line)
    return clipped, summary


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
    inline_activity: str = ""
    # Reasoning-channel buffer — captures content the provider routes
    # through the model's "thinking" / "reasoning" channel, plus any
    # ``<thinking>…</thinking>`` blocks that ``strip_tool_blocks`` peels
    # off the visible body.  Never flushed to scrollback; the activity
    # bar surfaces a dim sub-line of the most recent excerpt so the user
    # can tell the model is producing reasoning even before the first
    # content token lands.
    reasoning_chars: int = 0
    reasoning_excerpt: str = ""


@dataclass(slots=True)
class TurnStats:
    iterations: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    streamed_chars: int = 0
    elapsed_sec: float = 0.0
    started_at: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Idle-warning heuristics
# ---------------------------------------------------------------------------
#
# The "no tools were dispatched this turn" warning previously fired for any
# short reply (<200 chars, 0 tool calls), which produced false positives for
# greetings, clarifications, and short answers.  We now only fire when there
# is positive evidence the model *intended* to invoke a tool but emitted the
# command into prose instead.

# Fenced code block fences (```bash / ```sh / ```shell / ```console / ```$).
_FENCE_HINT_RE = re.compile(
    r"```(?:bash|sh|shell|zsh|console|cmd|ps1?|powershell|sql|python|py|node|js|ts)\b",
    re.IGNORECASE,
)

# Shell-prompt-style command lines.
_SHELL_PROMPT_RE = re.compile(r"(?m)^\s*[\$\#\>▶]\s+\S")

# Imperative verbs commonly preceding a "let me run" intent in Chinese & English.
_IMPERATIVE_HINT_RE = re.compile(
    r"(?i)\b(?:let me (?:run|check|look|inspect|search|find|read)|"
    r"i(?:'ll| will) (?:run|check|look|inspect|search|find|read|execute)|"
    r"running|executing|searching|checking)\b"
    r"|(?:我来|让我|我先|我去|我会)(?:运行|执行|看看|检查|搜索|查看|读)",
)


def _looks_like_intended_tool_use(text: str) -> bool:
    """True when *text* contains positive evidence of attempted tool use.

    We use this to gate the idle-turn warning so a polite greeting doesn't
    get flagged as a missing tool call.
    """
    if not text:
        return False
    if _FENCE_HINT_RE.search(text):
        return True
    if _SHELL_PROMPT_RE.search(text):
        return True
    if _IMPERATIVE_HINT_RE.search(text):
        return True
    if _FUNCTION_EQ_OPEN_RE.search(text):
        return True
    return False


# Match a fenced shell block, body captured non-greedily.  Body ends at
# the *earliest* of:
#
#  1. ``\u0060\u0060\u0060`` *not* followed by a shell-language tag — a real
#     closing fence; consume it (so the substitute also drops it).
#  2. (lookahead) a blank line — paragraph break, treated as the
#     implicit end of an *unclosed* phantom-tool block since models
#     never embed prose-paragraphs *inside* a real bash command.
#  3. (lookahead) ``\u0060\u0060\u0060<lang>`` — opener of the *next* fenced
#     block; don't consume, leave it for the next iteration.
#  4. End-of-string — models routinely truncate the line they were
#     about to call a tool on.
#
# The negative lookahead in branch 1 is critical: without it the
# non-greedy ``.*?`` would happily stop at the *opening* ``\u0060\u0060\u0060`` of
# the *next* shell fence and substitution would leave a stranded
# ``bash\n...`` behind that fails to re-match.  Whether a given block
# is considered "trailing" is decided by callers via the
# post-fence-prose budget — see :func:`_extract_trailing_command_fence`.
_SHELL_LANG_GROUP = r"(?:bash|sh|shell|zsh|console|cmd|ps1?|powershell)"
_COMMAND_FENCE_RE = re.compile(
    r"(?is)```" + _SHELL_LANG_GROUP + r"\b"
    r"[ \t]*\n?(.*?)"
    r"(?:\n?```(?!" + _SHELL_LANG_GROUP + r"\b)"
    r"|(?=\n[ \t]*\n)"
    r"|(?=\n?```" + _SHELL_LANG_GROUP + r"\b)"
    r"|\Z)"
)
# Allow a small amount of trailing prose after the fence (e.g. "I'll
# run that for you.") — beyond this the fence is considered embedded
# in the response, not a tool intent.
_TRAILING_FENCE_TAIL_BUDGET = 80


def _extract_trailing_command_fence(text: str) -> str | None:
    """Pull a single-line shell command out of a trailing fenced block.

    Returns the command body (newlines collapsed to spaces, trimmed)
    when *text* ends with ``\u0060\u0060\u0060bash <command>\u0060\u0060\u0060`` —
    closing fence optional, since models routinely truncate at the
    moment they were *about* to invoke a tool.  Returns ``None`` when:

    * No fenced shell block is present.
    * A fence exists but is followed by substantial prose (it's
      embedded in the response as an example, not a tool intent).
    * The captured body is empty / pure whitespace.
    """
    if not text or "```" not in text:
        return None
    match: re.Match[str] | None = None
    for m in _COMMAND_FENCE_RE.finditer(text):
        match = m
    if match is None:
        return None
    if len(text[match.end():].strip()) > _TRAILING_FENCE_TAIL_BUDGET:
        return None
    body = (match.group(1) or "").strip()
    if not body:
        return None
    return " ".join(line.strip() for line in body.splitlines() if line.strip())


def _has_unclosed_command_fence(text: str) -> bool:
    """True if *text* contains a shell code fence missing its close.

    The streaming preview uses this to decide whether to hide a
    trailing ``\u0060\u0060\u0060bash …`` block in flight.  Returns ``True`` only
    when at least one fenced shell block is still open — i.e. the
    model has typed ``\u0060\u0060\u0060bash`` but not the matching closing
    ``\u0060\u0060\u0060`` yet.  Closed examples (both fences present) return
    ``False`` so legitimate code-block content stays visible.
    """
    if not text or "```" not in text:
        return False
    open_fences = len(
        re.findall(
            r"(?im)```(?:bash|sh|shell|zsh|console|cmd|ps1?|powershell)\b",
            text,
        )
    )
    if open_fences == 0:
        return False
    total_fences = text.count("```")
    return total_fences < open_fences * 2


def strip_trailing_command_fence(text: str) -> str:
    """Strip a trailing fenced shell block from *text*.

    Pairs with :func:`_extract_trailing_command_fence` — call this to
    drop the dangling code fence from the visible assistant body so
    the user doesn't see a half-rendered ``\u0060\u0060\u0060bash`` while the
    parsed command is surfaced via the phantom-tool diagnostic.
    Conservatively returns *text* unchanged when the fence isn't at
    the tail of the message.
    """
    if not text or "```" not in text:
        return text
    match: re.Match[str] | None = None
    for m in _COMMAND_FENCE_RE.finditer(text):
        match = m
    if match is None:
        return text
    if len(text[match.end():].strip()) > _TRAILING_FENCE_TAIL_BUDGET:
        return text
    return text[: match.start()].rstrip()


def _extract_all_command_fences(text: str) -> list[str]:
    """Return *every* shell command body in *text*, in source order.

    Each entry is one fenced block, with internal newlines collapsed
    to spaces so the diagnostic can render it on a single line.  Used
    by ``end_stream`` when the model emitted multiple ``\u0060\u0060\u0060bash`` blocks
    and *no* tool was dispatched: we strip them all and surface them
    via the phantom-tool diagnostic so the response body shows only
    the model's prose narration, claude-code-style.
    """
    if not text or "```" not in text:
        return []
    out: list[str] = []
    for match in _COMMAND_FENCE_RE.finditer(text):
        body = (match.group(1) or "").strip()
        if not body:
            continue
        flat = " ".join(line.strip() for line in body.splitlines() if line.strip())
        if flat:
            out.append(flat)
    return out


def strip_all_command_fences(text: str) -> str:
    """Remove every shell code-fence block from *text*.

    Aggressive companion to :func:`strip_trailing_command_fence` for
    multi-fence phantom-tool cases (Kimi-style).  Strips the fenced
    block plus any immediately preceding ``intro:`` / ``intro：``
    punctuation so the visible prose flows naturally instead of
    showing "我先看看：" floating on its own line.

    The caller must gate this behind the no-tools-dispatched +
    intent-verbs heuristic — bare ``\u0060\u0060\u0060bash`` examples in a long
    response would otherwise be incorrectly stripped.
    """
    if not text or "```" not in text:
        return text
    # Pass 1: drop every fence we can spot.  The negative-lookahead
    # in ``_COMMAND_FENCE_RE`` makes substitution stable across
    # multi-block input — substituting the first match leaves the
    # next fence intact for the next iteration.
    cleaned = _COMMAND_FENCE_RE.sub("", text)
    # Pass 2: clean up dangling lead-in punctuation ("我先看看：" /
    # "Let me run:") that's now sitting on its own line.  We require
    # the colon to be at end-of-line *and* the previous content to
    # not look like a real sentence end (e.g. "好。"), so we only
    # erase the trailing "短语：" punctuation, never a sentence period.
    cleaned = re.sub(
        r"(?m)([^\s:：])[ \t]*[：:][ \t]*$",
        r"\1。",
        cleaned,
    )
    # Collapse the blank-line runs the strip can leave behind.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.rstrip()


# ---------------------------------------------------------------------------
# Inline tool-tag intent extraction
# ---------------------------------------------------------------------------
#
# When the model emits ``<invoke name="…">…<parameter name="…">…</parameter>``
# inline (instead of populating the structured ``tool_calls`` field), we
# parse out a (name, args) pair so the user sees *what the model tried to
# do* rather than just "2420 chars stripped".  The structured call never
# ran, but at least the user can tell whether the loss matters.

_INVOKE_BLOCK_RE = re.compile(
    r'(?is)<invoke\b[^>]*name="([^"]+)"[^>]*>(.*?)</invoke\s*>'
)
_INVOKE_PARAM_RE = re.compile(
    r'(?is)<parameter\b[^>]*name="([^"]+)"[^>]*>(.*?)</parameter\s*>'
)
_TOOL_CALL_JSON_RE = re.compile(
    r"(?is)<tool_call\b[^>]*>\s*(\{.*?\})\s*</tool_call\s*>"
)


@dataclass(slots=True)
class _InlineToolIntent:
    name: str
    args: dict[str, Any]


def _extract_inline_tool_intent(raw_text: str) -> _InlineToolIntent | None:
    """Best-effort: pull the first ``<invoke>`` or ``<tool_call>{json}`` from *raw_text*.

    Returns ``None`` if the text doesn't carry a parseable tool intent.
    Used by the phantom-tool diagnostic so we can render
    ``model attempted: Reading file foo.py`` instead of an opaque
    "chars stripped" line.
    """
    if not raw_text:
        return None

    # Prefer Anthropic-style <invoke name="..."><parameter name="...">...
    invoke_match = _INVOKE_BLOCK_RE.search(raw_text)
    if invoke_match is not None:
        name = (invoke_match.group(1) or "").strip()
        body = invoke_match.group(2) or ""
        args: dict[str, Any] = {}
        for param_match in _INVOKE_PARAM_RE.finditer(body):
            key = (param_match.group(1) or "").strip()
            value = (param_match.group(2) or "").strip()
            if key:
                args[key] = value
        if name:
            return _InlineToolIntent(name=name, args=args)

    # Fall back to OpenAI-ish <tool_call>{"name":..., "arguments":...}</tool_call>
    json_match = _TOOL_CALL_JSON_RE.search(raw_text)
    if json_match is not None:
        try:
            payload = json.loads(json_match.group(1))
        except Exception:  # noqa: BLE001
            return None
        name = str(payload.get("name") or payload.get("tool") or "").strip()
        raw_args = payload.get("arguments") or payload.get("parameters") or {}
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except Exception:  # noqa: BLE001
                raw_args = {"_raw": raw_args}
        if not isinstance(raw_args, dict):
            raw_args = {"_raw": raw_args}
        if name:
            return _InlineToolIntent(name=name, args=raw_args)

    # Last-ditch: the non-standard ``<function=NAME> <body>`` syntax.
    # No closing tag, body extends until the next ``<function=`` or
    # end-of-text.  We treat the body as a single positional argument
    # under a ``"command"`` key so :func:`_detail_for_args` renders it
    # cleanly on the ``└ attempted`` line.  The *first* match wins to
    # mirror the XML branch above.
    eq_match = _FUNCTION_EQ_TAG_RE.search(raw_text)
    if eq_match is not None:
        name = (eq_match.group(1) or "").strip()
        body = (eq_match.group(2) or "").strip()
        if name:
            args = {"command": " ".join(body.split())} if body else {}
            return _InlineToolIntent(name=name, args=args)

    return None


# ---------------------------------------------------------------------------
# Reasoning excerpt helper
# ---------------------------------------------------------------------------

_REASONING_EXCERPT_LIMIT = 96


def _render_reasoning_excerpt(text: str) -> ConsoleRenderable:
    """Build the dim ``thinking: …`` sub-line shown under the activity bar.

    Truncates to a single line (``_REASONING_EXCERPT_LIMIT`` chars) and
    keeps the *tail* of the excerpt because the most recent reasoning
    fragment is the most informative — older content already produced
    visible body once the model started generating its answer.
    """
    flat = (text or "").replace("\n", " ").replace("\r", " ").strip()
    if len(flat) > _REASONING_EXCERPT_LIMIT:
        flat = "…" + flat[-(_REASONING_EXCERPT_LIMIT - 1):]
    line = Text()
    line.append(f"  {icon('thinking') or '·'} ", style=f"dim {AETHER_DIM}")
    line.append("thinking: ", style=f"dim {AETHER_DIM}")
    line.append(flat, style=f"dim {AETHER_TEXT}")
    return line


# ---------------------------------------------------------------------------
# Live turn surface — Group(streaming_body, activity_bar)
# ---------------------------------------------------------------------------

class _TurnSurface:
    """Rich renderable held by the turn-scoped ``Live``.

    Pulled by Rich on every refresh frame (20 Hz).  Reads CLIUI's current
    ``_stream_buffer`` + ``_turn_state`` so the rendered surface reflects
    live state without us having to call ``Live.update()`` every tick.

    Visibility rule:

      * **While streaming visible prose** the bar hides — the moving
        text is itself the activity indicator and a redundant
        ``Forging (12s · thought for 9s)`` glued to the bottom of the
        markdown reads as "stuck".
      * **Between iterations / during tool use / while thinking** the
        bar is the only visual signal that anything is happening, so
        we show it.
    """

    def __init__(self, ui: "CLIUI") -> None:
        self._ui = ui

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        ui = self._ui

        # Streaming body — only present while a stream is open and has
        # visible content (after stripping inline tool tags).  Rendering
        # it inside Live lets each delta trigger an in-place repaint via
        # Rich's auto-refresh.
        #
        # Tail-cropping rule (this matters!):
        #   Rich's Live region with content that grows past the terminal
        #   viewport leaks "frames" into scrollback because the cursor
        #   clear-and-redraw sequence can only reach lines that are still
        #   visible — anything that scrolled off the top is permanently
        #   stuck in scrollback.  As streaming continues, each new frame
        #   is taller than the previous, the top portion of every frame
        #   ends up in scrollback, and the user sees the same content
        #   repeated many times.
        #
        #   We bound the rendered surface ourselves to `options.height`
        #   lines (with a safety margin) and prefer the *latest* lines
        #   (tail crop) so the user sees new tokens as they arrive.  The
        #   full content is never lost — ``end_stream`` flushes the
        #   complete Markdown to scrollback exactly once when the stream
        #   completes.
        streamed_body: ConsoleRenderable | None = None
        if ui._stream.active and ui._stream_buffer:
            raw = "".join(ui._stream_buffer)
            cleaned = strip_tool_blocks(raw)
            if cleaned.strip():
                # Reserve a few rows for the activity bar / reasoning
                # excerpt / hint lines so the last row isn't pinned
                # against the prompt.  Minimum 3 rows so very small
                # terminals still show *something*.  ``options.height``
                # is ``None`` when this surface is rendered outside a
                # live region (tests / non-TTY paths) — fall back to
                # the console's actual height, then to a sane default
                # of 24 rows.
                effective_height = (
                    options.height
                    or getattr(console, "height", None)
                    or 24
                )
                tail_lines = max(3, int(effective_height) - 2)
                lines = cleaned.split("\n")
                if len(lines) > tail_lines:
                    lines = lines[-tail_lines:]
                    cleaned = "\n".join(lines)
                streamed_body = ui._render_assistant(cleaned)

        # We used to render an in-flight ``● Searching for 1 pattern…
        # ⎿ "foo" in src/`` line here, mirroring claude-code's
        # CollapsedReadSearchContent.  The codex-style per-call
        # rendering (``render_tool_call`` writing ``● Running command
        # \n  └ git status`` to scrollback at dispatch time, plus the
        # activity bar at the bottom carrying the ``Running command…
        # $ git status`` verb) now covers both the "what's running"
        # and "what just finished" signals — a third in-flight
        # surface above the bar would just duplicate the activity
        # verb.  We keep the tracker (counts feed stats + the bar's
        # mode transitions) but no longer surface its headline here.

        # Reasoning excerpt — single-line dim hint shown beneath the
        # activity bar when the model is producing chain-of-thought
        # tokens (either through a separate reasoning channel or via
        # ``<thinking>`` blocks in the visible body).  Without this the
        # user sees "↑ N tokens" but no body, with no clue the model is
        # actually thinking.
        reasoning_renderable: ConsoleRenderable | None = None
        excerpt = ui._stream.reasoning_excerpt if ui._stream.active else ""
        if excerpt:
            reasoning_renderable = _render_reasoning_excerpt(excerpt)

        if streamed_body is not None:
            # Streaming visible prose — show only the body; the bar
            # would compete with the moving text for attention.
            yield streamed_body
            return

        yield ActivityBar(ui._turn_state)
        if reasoning_renderable is not None:
            yield reasoning_renderable


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
        # Off-turn fallback only — slash commands ("/model") may want a
        # spinner without opening a full turn.  In-turn status flows
        # through ``self._turn_state.verb`` and the activity bar.
        self._status: Status | None = None
        self._status_message: str = ""
        self._stream = _StreamState()
        self._stream_buffer: list[str] = []
        # Most recent finalised stream text from this turn — used by
        # ``_render_idle_turn_hint`` after ``end_stream`` has already
        # cleared ``_stream_buffer``.  Reset at ``begin_turn``.
        self._last_streamed_text: str = ""
        # Parsed command body extracted from a trailing fenced shell
        # block when the model wrote ``\u0060\u0060\u0060bash <cmd>`` in prose
        # instead of dispatching a tool.  Stashed by ``end_stream`` and
        # consumed by ``_render_idle_turn_hint`` so the diagnostic can
        # show "└ attempted: $ <cmd>" — same shape we use for XML tool
        # tags, which gives the user a single, consistent hint across
        # both phantom-tool failure modes.  Reset at ``begin_turn``.
        self._last_trailing_command: str | None = None
        # Full list of phantom commands when the model emits 2+ bash
        # blocks in a single stream (Kimi-style "我先看看：\u0060\u0060\u0060bash …
        # 实际上让我用更系统的方式查看：\u0060\u0060\u0060bash …").  Populated
        # alongside ``_last_trailing_command`` so the phantom-tool
        # diagnostic can list each attempt instead of pretending only
        # the first one happened.  Reset at ``begin_turn``.
        self._last_attempted_commands: list[str] = []
        # Set to ``True`` by ``end_stream`` whenever it renders a
        # phantom-tool hint (XML / JSON / function-eq tags stripped).
        # ``end_turn`` reads this to suppress the idle-turn warning,
        # which would otherwise repeat the same diagnostic in slightly
        # different wording right above the footer.  Reset at
        # ``begin_turn``.
        self._phantom_hint_rendered: bool = False
        #  / : when the engine *synthesizes* structured
        # tool_calls from prose intent, ``end_turn`` is told via the
        # ``phantom_synth_notes`` kwarg.  We surface a single dim
        # ``↻ synthesized N call(s) from prose`` line so the user
        # knows the model misbehaved without the loud warning that
        # used to fire ("model emitted inline tool tags").  The
        # warning is suppressed entirely when synthesis covered every
        # phantom block.  Reset at ``begin_turn``.
        self._pending_phantom_strip: dict[str, Any] | None = None
        # Turn-scoped: opened in ``begin_turn`` and torn down in
        # ``end_turn``.  Hosts ``_TurnSurface`` which renders the
        # streamed body + activity bar each refresh.  ``None`` between
        # turns and during slash-command execution.
        self._turn_live: Live | None = None
        self._turn_state = TurnState()
        # Counters captured at the start of each streaming session within
        # a turn.  Used by the phantom-tool diagnostic to distinguish
        # "this stream produced no structured calls" from "the whole
        # turn never dispatched a tool".
        self._tool_calls_at_turn_start: int = 0
        self.stats = TurnStats()
        # Rotated every time a turn begins so the spinner doesn't feel
        # mechanical.  Mirrors Claude Code's "Forging / Channeling /
        # Pondering …" vibe.
        self._verb_cursor = 0
        # Coalesces consecutive tool dispatches inside a single LLM
        # iteration into one rolling line ("Searching for 2 patterns,
        # reading 1 file…").  Mirrors claude-code's
        # ``CollapsedReadSearchContent`` — without it every individual
        # call would print its own ``● Reading file`` block, which gets
        # noisy fast for tool-heavy turns.
        self.tool_groups = ToolGroupTracker(sink=self._flush_tool_group)
        # Last block kind we wrote to scrollback (``user`` / ``assistant``
        # / ``tool_group`` / ``footer`` / ``other``).  Drives the
        # blank-line spacer logic in ``_ensure_block_spacer`` so we can
        # insert a single empty row between visually distinct blocks
        # without hard-coding ``console.print()`` calls everywhere.
        self._last_block_kind: str = ""
        # Set to True by :class:`aether.cli.app.AetherApp` when the
        # bottom region (activity bar + active-group preview + input
        # frame) is owned by a long-lived ``prompt_toolkit``
        # Application.  In that mode we **must not** create our own
        # ``rich.live.Live`` region or ``rich.status.Status`` spinner
        # because both would clash with prompt_toolkit's bottom layout
        # via ``patch_stdout(raw=True)``.  AetherApp pulls the activity
        # bar / active-group rendering directly from ``self._turn_state``
        # and ``self.tool_groups`` instead.
        self.managed_externally: bool = False

    # ------------------------------ spacing --------------------------------
    #
    # Visually distinct blocks (user echo, assistant prose, tool group
    # headlines, the turn footer) share a single blank-line policy: emit
    # at most one blank row between any two adjacent blocks.  Centralising
    # this in :meth:`_ensure_block_spacer` keeps the rules coherent and
    # avoids the "two blocks glued together → caller sprinkles
    # ``console.print()`` calls all over the place" pattern.

    # Block kinds that count toward the spacer policy.  Anything not in
    # this set (info / warn / debug lines) is treated as transient and
    # doesn't reset the policy.
    #
    # ``tool_result`` is deliberately *absent*: result lines
    # (``    ⎿ <output>``) belong directly under the call's
    # ``  └ <detail>`` line with no blank gap, so we never want a
    # spacer between a ``tool_call`` block and its trailing output.
    # ``tool_call`` is included so back-to-back dispatches and the
    # transition into / out of an assistant block get one blank row.
    _SPACED_BLOCK_KINDS: frozenset[str] = frozenset({
        "user", "assistant", "tool_group", "tool_call", "footer",
    })

    def _ensure_block_spacer(self, next_kind: str) -> None:
        """Print one blank row when the previous block was visually heavy.

        Caller passes the kind of the block they're *about* to render;
        we print a separator if and only if the previous spaced block
        is something we want to push apart from the new one (e.g. user
        echo → assistant prose → tool group → footer transitions).

        Special case: consecutive ``tool_call`` blocks **don't** get a
        separator.  With the codex-style single-line per-call header
        (``● Read foo.py``), a blank row between every back-to-back
        call is what makes a long exploratory read/list/edit burst
        look "scattered" — the spacer takes more visual real estate
        than the calls themselves.  Tight stacking reads as one
        coherent activity block, which matches what the agent is
        actually doing semantically.  We still keep the spacer for
        the *transition* into / out of a tool burst (user → tool,
        tool → assistant, …).
        """
        prev = self._last_block_kind
        if not prev or prev not in self._SPACED_BLOCK_KINDS or next_kind not in self._SPACED_BLOCK_KINDS:
            return
        if prev == next_kind == "tool_call":
            return
        try:
            self.console.print()
        except Exception:  # noqa: BLE001 — never let a spacer kill the run
            pass

    def _record_block(self, kind: str) -> None:
        if kind in self._SPACED_BLOCK_KINDS:
            self._last_block_kind = kind

    def _flush_tool_group(self, renderable: ConsoleRenderable) -> None:
        """Sink wired into :class:`ToolGroupTracker` — prints the explore tree.

        The tracker passes its rendered output (a Rich :class:`Text`
        that includes the ``● Explored`` umbrella header followed by
        the per-call sub-call tree) and we land it in scrollback as
        a single visually coherent block.  Same blank-row spacer
        policy used by user/assistant blocks so an explore tree
        never glues to the assistant prelude directly above it.
        """
        self._ensure_block_spacer("tool_group")
        self.console.print(renderable)
        self._record_block("tool_group")

    # ------------------------------ status ---------------------------------

    def set_status(self, message: str, *, spinner: str = "dots") -> None:
        """Set the active verb on the activity bar (or open a fallback spinner).

        During a turn the verb routes to :class:`TurnState` so the
        always-on activity bar picks it up on its next refresh — no
        nested ``rich.status.Status`` is needed.  Outside a turn (slash
        commands like ``/model``) we fall back to the classic spinner.

        When :class:`aether.cli.app.AetherApp` owns the bottom region
        (``managed_externally=True``) we route exclusively through
        ``self._turn_state.verb`` because a ``rich.status.Status``
        spinner would smear ANSI escape sequences across the
        prompt_toolkit-managed input frame.
        """
        self._status_message = message
        # Trim trailing ellipsis — the bar's parenthesised suffix already
        # conveys "in progress" and we don't want "Reading file……" on
        # tight terminals.
        verb = message.rstrip().rstrip("…").rstrip()
        if self._turn_live is not None or self.managed_externally:
            self._turn_state.verb = verb
            return
        if self._status is not None:
            try:
                self._status.update(status=f"[aether.status]{message}[/]", spinner=spinner)
            except Exception:        # noqa: BLE001
                self._status.update(status=f"[aether.status]{message}[/]")
            return
        self._status = self.console.status(
            f"[aether.status]{message}[/]",
            spinner=spinner,
            spinner_style=f"{AETHER_PRIMARY}",
        )
        self._status.start()

    def clear_status(self, *, clear_message: bool = True) -> None:
        if self._turn_live is not None or self.managed_externally:
            if clear_message:
                self._turn_state.verb = ""
                self._status_message = ""
            return
        if self._status is not None:
            try:
                self._status.stop()
            finally:
                self._status = None
        if clear_message:
            self._status_message = ""

    # ------------------------------ stream ---------------------------------

    def begin_stream(self) -> None:
        """Mark a streaming session as active so the bar surface includes the body."""
        if self._stream.active:
            return
        # Stop any off-turn fallback spinner (defensive — should not
        # normally be active inside a turn).
        if self._status is not None:
            try:
                self._status.stop()
            finally:
                self._status = None

        self._stream_buffer = []
        self._stream.active = True
        self._stream.char_count = 0
        self._stream.line_count = 1
        self._stream.stripped_chars = 0
        self._stream.tool_calls_at_start = self.stats.tool_calls
        self._stream.inline_activity = ""
        self._stream.reasoning_chars = 0
        self._stream.reasoning_excerpt = ""

        # If no Live is running (e.g. caller didn't open a turn — used
        # by tests / non-TTY paths) leave a blank line and we'll print
        # cleaned deltas inline as they arrive.
        if self._turn_live is None:
            self.console.print()

    def stream_delta(self, delta: str) -> None:
        if not delta:
            return
        if not self._stream.active:
            self.begin_stream()

        self._stream_buffer.append(delta)
        self._stream.char_count += len(delta)
        self.stats.streamed_chars += len(delta)
        self._turn_state.response_chars = self._stream.char_count
        if "\n" in delta:
            self._stream.line_count += delta.count("\n")

        if self._turn_live is not None or self.managed_externally:
            # Either the Rich Live region or the prompt_toolkit AetherApp
            # owns the bottom region — recompute stripped_chars + inline
            # activity + reasoning excerpt so the renderer sees fresh
            # values on its next tick.  We deliberately do NOT echo
            # cleaned deltas to stdout here: in Live mode the surface
            # renders the markdown buffer itself, in managed mode the
            # activity bar's ``↓ N tokens`` counter ticks while the full
            # formatted body is flushed once at ``end_stream``.  Echoing
            # raw deltas would race the bottom region and produce
            # smeared output (verb fragments overlapping prose).
            try:
                raw = "".join(self._stream_buffer)
                cleaned = strip_tool_blocks(raw)
                self._stream.inline_activity = _inline_activity_from_text(raw)
                self._stream.stripped_chars = max(0, len(raw) - len(cleaned))
                # Capture <thinking> blocks into the reasoning excerpt so
                # the activity bar's sub-line shows what the model is
                # mulling — otherwise these tokens count toward "↑ N
                # tokens" but never become visible.
                thinking_text = extract_thinking_blocks(raw)
                if thinking_text:
                    self._stream.reasoning_excerpt = thinking_text
                if self._stream.inline_activity:
                    self._turn_state.verb = self._stream.inline_activity
                    self._turn_state.mode = MODE_TOOL_USE
                # Flip from "thinking" to "responding" only once the
                # cleaned visible body has started arriving — purely
                # reasoning/thinking tokens shouldn't reset the timer.
                if cleaned.strip() and self._turn_state.mode == MODE_THINKING:
                    self._turn_state.mark_first_response_token()
            except Exception:        # noqa: BLE001 — never let render-prep kill the stream
                pass
        else:
            # Fallback path when no turn is open AND no AetherApp is
            # managing the bottom region — echo the cleaned delta
            # directly so unit tests / non-TTY use still see output.
            cleaned_delta = strip_tool_blocks(delta)
            if cleaned_delta:
                if self._turn_state.mode == MODE_THINKING:
                    self._turn_state.mark_first_response_token()
                self.console.out(cleaned_delta, highlight=False, end="")

    def stream_reasoning_delta(self, delta: str) -> None:
        """Append a delta from the provider's reasoning/thinking channel.

        Reasoning content is **never** flushed to scrollback — it lives
        only as a single dim ``thinking: …`` sub-line beneath the
        activity bar so the user can see the model is producing
        chain-of-thought even before the visible body starts arriving.

        Providers route here when they emit a separate ``reasoning``
        channel (DeepSeek-R1 / o1-style ``reasoning_content`` deltas).
        For providers that wrap chain-of-thought in ``<thinking>`` tags
        inside the main content stream, the existing ``stream_delta``
        path catches it via :func:`extract_thinking_blocks`.
        """
        if not delta:
            return
        if not self._stream.active:
            self.begin_stream()

        self._stream.reasoning_chars += len(delta)
        # Keep a rolling tail of the reasoning excerpt — the most recent
        # fragment is the most useful one to show.  We bound the in-
        # memory excerpt to avoid unbounded growth on very long traces.
        existing = self._stream.reasoning_excerpt
        combined = (existing + delta) if existing else delta
        max_excerpt = max(_REASONING_EXCERPT_LIMIT * 4, 256)
        if len(combined) > max_excerpt:
            combined = combined[-max_excerpt:]
        self._stream.reasoning_excerpt = combined

    def end_stream(self) -> None:
        """Finalise the current streaming session.

        Flushes the streamed markdown to scrollback (so subsequent
        ``console.print`` for tool panels lands below it), runs the
        phantom-tool diagnostic, and resets ``_stream``.  Leaves the
        turn-scoped Live running — the next iteration's stream will
        re-attach to the same surface.
        """
        if not self._stream.active:
            return

        # Differentiate "this stream produced no structured calls" from
        # "this turn never dispatched anything" — the wording in the
        # diagnostic depends on it.
        stream_dispatched_tools = (
            self.stats.tool_calls > self._stream.tool_calls_at_start
        )
        turn_dispatched_tools = (
            self.stats.tool_calls > self._tool_calls_at_turn_start
        )
        raw_stream_text = "".join(self._stream_buffer)
        cleaned = strip_tool_blocks(raw_stream_text)
        # Recompute the stripped-char count from the actual buffer
        # rather than trusting ``self._stream.stripped_chars`` — that
        # field is updated by ``stream_delta`` and may be stale when
        # tests / non-streaming paths feed the buffer directly.  The
        # phantom-tool diagnostic gate uses this number; getting it
        # wrong silently swallows the warning.
        stripped = max(self._stream.stripped_chars, len(raw_stream_text) - len(cleaned))

        # Phantom command intent — if this iteration produced *no*
        # structured tool dispatch but the visible body carries one or
        # more ``\u0060\u0060\u0060bash <cmd>`` blocks AND the prose contains
        # imperative verbs ("let me run / 我来 / I'll check"), treat
        # each fenced block as a failed tool invocation: strip every
        # block from the body so the response shows only narration
        # (claude-code-style), and stash the parsed commands for the
        # phantom-tool diagnostic to surface as ``└ attempted: $ <cmd>``.
        #
        # We use the aggressive multi-fence stripper here rather than
        # the trailing-only variant — Kimi-class models routinely emit
        # 2-3 bash blocks interleaved with prose ("我先看看：\u0060\u0060\u0060bash …
        # 实际上让我用更系统的方式查看：\u0060\u0060\u0060bash …"), and the trailing
        # heuristic was leaving the earlier ones on screen.
        all_attempts: list[str] = []
        if not stream_dispatched_tools and _looks_like_intended_tool_use(cleaned):
            all_attempts = _extract_all_command_fences(cleaned)
            if all_attempts:
                cleaned = strip_all_command_fences(cleaned)
        if all_attempts:
            # Stash the *first* command on ``_last_trailing_command``
            # for the existing single-line idle hint, and keep the
            # full list around so ``_render_phantom_tool_hint`` can
            # decide whether to summarise ("attempted 3 commands").
            self._last_trailing_command = all_attempts[0]
            self._last_attempted_commands = all_attempts

        # Flush the final markdown into scrollback.  Rich's Live correctly
        # hoists this above the live region so it lands above the bar;
        # :class:`AetherApp`'s ``patch_stdout(raw=True)`` does the same
        # thing relative to the prompt_toolkit layout.
        if cleaned.strip():
            if self._turn_live is not None or self.managed_externally:
                try:
                    self._ensure_block_spacer("assistant")
                    self.console.print(self._render_assistant(cleaned))
                    self._record_block("assistant")
                except Exception:    # noqa: BLE001
                    pass
            else:
                # No turn-scoped Live (test / non-TTY path) — newline only.
                self.console.print()

        # Stash the stream text so ``end_turn`` can run the
        # idle-turn heuristic *after* this method clears the buffer.
        self._last_streamed_text = raw_stream_text
        # Reset stream state BEFORE running the diagnostic so the live
        # surface re-renders without the stale buffer.
        self._stream = _StreamState()
        self._stream_buffer = []
        # The bar should reflect "between tool dispatches" so the verb
        # rotates back to a thinking word; the middleware will overwrite
        # this on the next iteration boundary, but we keep the bar from
        # showing a stale "responding" verb in the gap.
        self._turn_state.mode = MODE_THINKING

        # Phantom-tool diagnostic —  /  deferred path.
        # We can't decide at stream-end whether the warning is needed
        # because the engine's synthesis pass runs *after* this method
        # returns: a stripped block might still get rescued into a
        # structured ``tool_calls`` dispatch.  Stash the parameters and
        # let ``end_turn`` make the final call once it knows whether
        # synthesis covered the strip.  This eliminates the old "loud
        # warning shown above a successful tool dispatch" UX glitch.
        if stripped >= 80 and not stream_dispatched_tools:
            self._pending_phantom_strip = {
                "stripped_chars": stripped,
                "raw_text": raw_stream_text,
                "cleaned_text": cleaned,
                "turn_already_dispatched": turn_dispatched_tools,
            }

    def _render_phantom_synth_note(
        self,
        *,
        count: int,
        notes: list[str],
    ) -> None:
        """Render the soft "↻ synthesized N call(s) from prose" hint.

        Fires once per turn whenever
        :class:`AgentEngine` rescued a phantom tool intent into a
        structured ``ToolCall`` and dispatched it.  We deliberately
        keep the styling subtle (dim ``↻`` glyph, no warning colour)
        because the call *did* run — the user just gets a small
        breadcrumb that the model emitted prose tags and the engine
        repaired them.  The loud "model emitted inline tool tags"
        warning is reserved for the case where synthesis was
        impossible (no matching tool registered) and the run-loop
        ended up sending a corrective message instead.
        """
        if count <= 0:
            return
        line = Text()
        line.append("  ", style="")
        line.append("↻ ", style=f"bold {AETHER_DIM}")
        plural = "" if count == 1 else "s"
        line.append(
            f"synthesized {count} tool call{plural} from prose",
            style=f"dim {AETHER_DIM}",
        )
        self.console.print(line)
        for raw_note in (notes or [])[:3]:
            note = (raw_note or "").strip()
            if not note:
                continue
            sub = Text()
            sub.append("    ⎿ ", style=f"dim {AETHER_DIM}")
            sub.append(_truncate_inline(note), style=f"dim {AETHER_TEXT}")
            self.console.print(sub)
        if len(notes or []) > 3:
            extra = Text()
            extra.append("    ⎿ ", style=f"dim {AETHER_DIM}")
            extra.append(
                f"… (+{len(notes) - 3} more)",
                style=f"dim {AETHER_DIM}",
            )
            self.console.print(extra)

    def _render_phantom_tool_hint(
        self,
        stripped_chars: int,
        *,
        raw_text: str = "",
        cleaned_text: str = "",
        turn_already_dispatched: bool = False,
    ) -> None:
        """Warn the user when an inline tool-tag block was stripped silently.

        Three pieces of information matter for the user:

        1. **What was lost** — render the model's *intent* (parsed name +
           args) instead of just "N chars stripped".  This way the user
           can tell at a glance whether a missing tool call mattered.
        2. **Why** — the model wrote XML tags into prose instead of
           populating the structured ``tool_calls`` field, so nothing
           ran.  Different wording when the turn already had successful
           dispatches earlier (mid-turn fallback) vs. an entire-turn miss.
        3. **A way out** — strengthen the system prompt or pick a model
           with native tool use.

        Auto-shows the raw preview when stripping ate so much that the
        cleaned reply is essentially empty (visible <20 chars) — that's
        the case where the turn looked "silenced" to the user.
        """
        intent = _extract_inline_tool_intent(raw_text)

        # Headline.  When the model fell back mid-turn (i.e. earlier
        # iterations succeeded) we say so explicitly — the previous
        # wording "no structured tool_calls were dispatched" was
        # confusing because the user had just seen one work.
        if turn_already_dispatched:
            headline_msg = "model fell back to inline tool tags"
            tail_msg = " — earlier dispatches succeeded but this one was lost."
        else:
            headline_msg = "model emitted inline tool tags"
            tail_msg = " — no structured tool_calls were dispatched."

        line = Text()
        line.append("  ", style="")
        line.append(f"{icon('warn')} ", style=f"bold {AETHER_WARNING}")
        line.append(f"{headline_msg} ", style=AETHER_WARNING)
        line.append(
            f"({stripped_chars} chars stripped)",
            style=f"dim {AETHER_DIM}",
        )
        line.append(tail_msg, style=AETHER_WARNING)
        self.console.print(line)

        # Surface the model's *intent* — the most important signal.
        # When parsing fails we still want the hint visible, just
        # without the "Reading file: foo.py" line.
        rendered_attempt = False
        if intent is not None:
            verb = _verb_for_tool(intent.name)
            detail = _truncate_inline(_detail_for_args(intent.args)) if intent.args else ""
            attempted = Text()
            attempted.append("    └ attempted: ", style=f"dim {AETHER_DIM}")
            attempted.append(verb, style=f"bold {AETHER_TEXT}")
            if detail:
                attempted.append("  ", style="")
                attempted.append(detail, style=f"dim {AETHER_TEXT}")
            self.console.print(attempted)
            rendered_attempt = True

        # Bash-fence attempts captured by ``end_stream`` — list each
        # one so the user sees the full intended workflow.  When an
        # XML/JSON intent was already shown above, the bash list is
        # appended underneath; otherwise it serves as the primary
        # attempted-action display.
        bash_attempts = list(self._last_attempted_commands)
        for idx, cmd in enumerate(bash_attempts):
            line = Text()
            if not rendered_attempt and idx == 0:
                line.append("    └ attempted: ", style=f"dim {AETHER_DIM}")
            else:
                line.append("      ", style="")
            line.append("$ ", style=f"bold {AETHER_TEXT}")
            line.append(_truncate_inline(cmd), style=AETHER_TEXT)
            self.console.print(line)
            rendered_attempt = True

        hint = Text(
            "    try /system to ask the model to use the structured "
            "tool_calls field, or pick a model that supports native tool use.",
            style=f"dim {AETHER_DIM}",
        )
        self.console.print(hint)

        # Keep the raw preview as an explicit debug/verbose affordance.
        # In normal mode the user asked for the tool activity to live in
        # the bar, not for the stripped XML/function payload to flood the
        # transcript area.
        if not self.verbose:
            return
        if not _should_render_inline_tool_preview(cleaned_text, stripped_chars):
            return
        if not _has_xml_style_inline_tool_markup(raw_text):
            return

        preview, truncated = _inline_tool_preview(raw_text)
        if not preview:
            return

        title = "raw model output preview"
        if truncated:
            title += " (truncated)"
        self.console.print(
            Panel(
                Text(preview, style=AETHER_DIM),
                title=title,
                border_style=AETHER_BORDER,
                expand=True,
            )
        )

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

        Spacer policy (centralised in :meth:`_ensure_block_spacer`)
        guarantees a single blank row separates this user echo from the
        previous assistant block / footer / tool group, and from the
        first assistant block that comes next.  Without this each new
        prompt felt glued to the previous response.
        """
        self._ensure_block_spacer("user")
        text = Text()
        text.append(f"{icon('user')} ", style=f"bold {AETHER_PRIMARY}")
        text.append(message.replace("\n", "\n  "), style=f"{AETHER_TEXT}")
        self.console.print(text)
        self._record_block("user")

    def render_assistant_block(self, content: str) -> None:
        """Render a non-streamed final assistant message (fallback path)."""
        raw_content = content or ""
        cleaned = strip_tool_blocks(raw_content)
        stripped = max(0, len(raw_content) - len(cleaned))
        turn_already_dispatched = (
            self.stats.tool_calls > self._tool_calls_at_turn_start
        )

        # Mirror ``end_stream``'s phantom-command guard for the
        # non-streaming path: when the model wrote ``\u0060\u0060\u0060bash <cmd>``
        # in prose without dispatching a tool, strip the trailing
        # fence from the visible body and stash the parsed command
        # for ``_render_idle_turn_hint`` to surface as ``└ attempted``.
        if not turn_already_dispatched and _looks_like_intended_tool_use(cleaned):
            trailing_command = _extract_trailing_command_fence(cleaned)
            if trailing_command:
                cleaned = strip_trailing_command_fence(cleaned)
                self._last_trailing_command = trailing_command

        if not cleaned.strip():
            if stripped >= 80:
                self._pending_phantom_strip = {
                    "stripped_chars": stripped,
                    "raw_text": raw_content,
                    "cleaned_text": cleaned,
                    "turn_already_dispatched": turn_already_dispatched,
                }
            return
        self._ensure_block_spacer("assistant")
        self.console.print(self._render_assistant(cleaned))
        self._record_block("assistant")

        if stripped >= 80:
            self._pending_phantom_strip = {
                "stripped_chars": stripped,
                "raw_text": raw_content,
                "cleaned_text": cleaned,
                "turn_already_dispatched": turn_already_dispatched,
            }

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

    def _render_stream_surface(self, text: str, *, activity: str = "") -> ConsoleRenderable:
        status_message = activity or self._status_message
        if not status_message:
            return self._render_assistant(text)

        header = Text()
        if activity:
            header.append(f"{icon('tool')} ", style=f"bold {AETHER_PRIMARY_DIM}")
        else:
            header.append(f"{icon('thinking')} ", style=f"bold {AETHER_PRIMARY_DIM}")
        header.append(status_message, style=f"dim {AETHER_DIM}")

        if not text.strip():
            return header

        return Group(header, self._render_assistant(text))

    def _refresh_live_stream(self) -> None:
        if self._live is None:
            return
        try:
            raw = "".join(self._stream_buffer)
            cleaned = strip_tool_blocks(raw)
            self._stream.inline_activity = _inline_activity_from_text(raw)
            self._stream.stripped_chars = max(0, len(raw) - len(cleaned))
            self._live.update(
                self._render_stream_surface(
                    cleaned,
                    activity=self._stream.inline_activity,
                )
            )
        except Exception:        # noqa: BLE001 — never let render kill the stream
            pass

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
        """Per-call header rendering — single-line ``● <Verb> <detail>``.

        Always renders (no longer verbose-gated).  Each tool dispatch
        produces ONE line in scrollback at call-start time::

            ● Read backend/harness/aether/cli/ui.py
            ● Bash git status
            ● Edit foo.py

        :meth:`render_tool_result` appends a ``    ⎿ <output>`` block
        below when the tool finishes — but only for output-bearing
        categories (Bash, Web, Search) where the result text is what
        the user actually wants to see.  Read / List / Edit / Write
        results stay silent because the *content* is for the model
        to consume; the user-relevant info (filename) is already on
        the call line above.

        Note: stats accounting (``self.stats.tool_calls += 1``) lives in
        :class:`CLIUIMiddleware` so this method only owns rendering.
        """
        del tool_call_id
        self.clear_status()
        if self._stream.active:
            self.end_stream()

        verb = _verb_for_tool(name)
        detail = _truncate_inline(_detail_for_args(args)) if args else ""

        self._ensure_block_spacer("tool_call")
        head = Text()
        head.append(f"{icon('assistant')} ", style=f"bold {AETHER_PRIMARY}")
        head.append(verb, style=f"bold {AETHER_TEXT}")
        if detail:
            head.append(" ", style="")
            head.append(detail, style=f"dim {AETHER_TEXT}")
        self.console.print(head)
        self._record_block("tool_call")

    # Categories whose result content we render below the call header.
    # Other categories (READ / LIST / WRITE / EDIT / SEARCH / SUBAGENT
    # / MCP) stay silent on success — their content is fed back into
    # the model, not the user, and the call line itself already says
    # *what* was read / listed / edited.  Errors always render
    # regardless of category.
    _RESULT_VERBOSE_CATEGORIES: frozenset[ToolCategory] = frozenset({
        ToolCategory.BASH,
        ToolCategory.WEB,
    })

    def render_tool_result(
        self,
        name: str,
        content: str,
        *,
        is_error: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Print the codex-style ``    ⎿ <output>`` tree under each call.

        Shape::

            ● Read backend/harness/aether/cli/ui.py
            ● Bash git status
                ⎿ On branch master
                  nothing to commit, working tree clean
            ● Edit foo.py

        Output-bearing tools (Bash / Web) get a ``    ⎿ <output>``
        block with up to ``_OUTPUT_PREVIEW_MAX_LINES`` lines plus a
        ``… (+N more lines)`` trailer if we had to truncate.

        Read / List / Write / Edit / Search / Subagent / MCP stay
        SILENT on success — the user wanted "filename-level" info,
        which is already on the call header above.  Showing the
        actual file contents / listings / search hits would just
        flood the transcript (claude-code and codex both suppress
        these by default for the same reason).

        Errors *always* render so the user can see why a call
        failed regardless of category.

        Note: stats accounting (``self.stats.tool_errors``) lives in
        :class:`CLIUIMiddleware` so this method only owns rendering.
        """
        del metadata
        self.clear_status()

        display_lines, summary = _format_output_preview(content or "")

        if is_error:
            # Errors *always* render — we synthesise a "(no message)"
            # placeholder when the tool returned nothing so the user
            # can still see something failed.
            if not display_lines:
                display_lines = [(content or "(no message)").strip() or "(no message)"]
                summary = ""

            head = Text()
            head.append(_RESULT_PREFIX, style=AETHER_DIM)
            head.append(f"{icon('error')} ", style=f"bold {AETHER_ERROR}")
            head.append(display_lines[0], style=AETHER_ERROR)
            self.console.print(head)
            for line in display_lines[1:]:
                cont = Text()
                cont.append(_RESULT_CONT, style="")
                cont.append(line, style=AETHER_ERROR)
                self.console.print(cont)
            if summary:
                tail = Text()
                tail.append(_RESULT_CONT, style="")
                tail.append(summary, style=f"dim {AETHER_DIM}")
                self.console.print(tail)
            self._record_block("tool_result")
            return

        # Suppress success output for read-style tools.  The user
        # asked for "filename-level" detail; the path is already on
        # the call header line and the contents are for the model.
        if category_for(name) not in self._RESULT_VERBOSE_CATEGORIES:
            return

        # Success path: stay silent for empty content too (Bash
        # commands like ``git config`` regularly return "").
        if not display_lines:
            return

        head = Text()
        head.append(_RESULT_PREFIX, style=AETHER_DIM)
        head.append(display_lines[0], style=f"dim {AETHER_TEXT}")
        self.console.print(head)
        for line in display_lines[1:]:
            cont = Text()
            cont.append(_RESULT_CONT, style="")
            cont.append(line, style=f"dim {AETHER_TEXT}")
            self.console.print(cont)
        if summary:
            tail = Text()
            tail.append(_RESULT_CONT, style="")
            tail.append(summary, style=f"dim {AETHER_DIM}")
            self.console.print(tail)
        self._record_block("tool_result")

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
        """Open a turn-scoped Live region that hosts streaming + the activity bar.

        From this point until ``end_turn`` is called, every
        ``console.print`` lands above the live region and the activity
        bar at the bottom is auto-refreshed at 20 Hz.  Streamed deltas
        feed into the same surface so the body stays directly above the
        bar instead of competing with a separate spinner.
        """
        self.stats = TurnStats()
        self._tool_calls_at_turn_start = 0  # we just reset stats above
        self._last_streamed_text = ""
        self._last_trailing_command = None
        self._last_attempted_commands = []
        self._phantom_hint_rendered = False
        self._pending_phantom_strip = None
        self._turn_state.reset()
        self._turn_state.verb = self._next_spinner_verb()
        self._turn_state.mark_thinking_start()
        self._status_message = self._turn_state.verb
        # Reset the tool-group tracker — leftover state from a prior
        # turn would otherwise leak into the new turn's first iteration.
        self.tool_groups.discard_active()

        # Stop any leftover off-turn fallback spinner.
        if self._status is not None:
            try:
                self._status.stop()
            finally:
                self._status = None

        if self._turn_live is not None:
            # Defensive — should not happen.  Tear down the previous
            # surface so we don't leak a Rich Live thread.
            try:
                self._turn_live.stop()
            except Exception:        # noqa: BLE001
                pass
            self._turn_live = None

        # When :class:`AetherApp` owns the bottom region we MUST NOT
        # spawn a Rich ``Live`` — the two layers would race for the
        # terminal cursor and produce smeared output on every refresh.
        # AetherApp polls ``self._turn_state`` and ``self.tool_groups``
        # directly via its own FormattedTextControl renderers.
        if self.managed_externally:
            return

        try:
            self._turn_live = Live(
                _TurnSurface(self),
                console=self.console,
                refresh_per_second=20,
                transient=True,
                # ``vertical_overflow="crop"`` is a line-wrap-level safety
                # net — the line-level tail-crop in
                # ``_TurnSurface.__rich_console__`` already keeps the
                # source markdown bounded to ``terminal_height - 1`` lines,
                # but a single very wide line can wrap onto extra display
                # rows and still overflow.  ``crop`` ensures the Live
                # region never exceeds the viewport even in that edge
                # case, so we never get scrollback duplication.  The full
                # content lands in scrollback exactly once via
                # ``end_stream``'s ``console.print(_render_assistant(…))``.
                vertical_overflow="crop",
            )
            self._turn_live.start()
        except Exception:            # noqa: BLE001 — Live is best-effort
            self._turn_live = None

    def end_turn(
        self,
        *,
        status: str,
        exit_reason: str,
        iterations: int,
        error: str | None = None,
        phantom_synth_notes: list[str] | None = None,
        phantom_synth_count: int = 0,
    ) -> None:
        if self._stream.active:
            self.end_stream()
        # Flush any unresolved tool group so its past-tense headline
        # lands in scrollback before the turn footer.  Errors paths
        # have already discarded the tracker; this is a no-op then.
        self.tool_groups.flush_active()
        # Now that the stream (if any) is flushed to scrollback, take down
        # the live surface so the activity bar disappears cleanly before
        # we print the turn footer.
        if self._turn_live is not None:
            try:
                self._turn_state.verb = ""  # last frame: blank the bar
                self._turn_live.stop()
            except Exception:        # noqa: BLE001
                pass
            self._turn_live = None
        if self._status is not None:
            try:
                self._status.stop()
            finally:
                self._status = None
        self._status_message = ""

        self.stats.iterations = iterations
        self.stats.elapsed_sec = max(0.0, time.monotonic() - self.stats.started_at)

        # Detect "did the agent actually do anything?".  We previously
        # fired this for any short reply (<200 chars, 0 tool calls)
        # which produced false positives for greetings, clarifications,
        # and short answers.  Now we require positive evidence the
        # model *intended* a tool call but emitted it in prose.
        #
        # Two distinct signals qualify as evidence:
        #
        # 1. Streamed body that *looks* like attempted tool use (fenced
        #    shell block, ``$``-prefixed line, "I'll run/我来" verb).
        # 2. ``end_stream`` / ``render_assistant_block`` already parsed
        #    a trailing command fence out of the body — definitive
        #    proof the model wrote a tool call in prose.  This second
        #    branch matters for the non-streaming fallback path where
        #    ``_last_streamed_text`` is empty.
        #
        # ``_phantom_hint_rendered`` deduplicates: when ``end_stream``
        # already showed the structured-tag phantom warning we skip
        # this one because both diagnostics convey the same thing —
        # firing both would look like a duplicate-bug regression.
        last_text = self._last_streamed_text
        streaming_intent = (
            self.stats.streamed_chars > 0
            and _looks_like_intended_tool_use(last_text)
        )
        parsed_intent = self._last_trailing_command is not None

        #  / : when the engine synthesized real
        # ``ToolCall``s from prose intents, render a single dim
        # acknowledgement line and *drop* any pending phantom warning
        # — the call did dispatch, the warning would be misleading.
        if phantom_synth_count > 0:
            self._render_phantom_synth_note(
                count=phantom_synth_count,
                notes=phantom_synth_notes or [],
            )
            self._pending_phantom_strip = None
            self._phantom_hint_rendered = True

        # Loud warning is now deferred from ``end_stream`` /
        # ``render_assistant_block``: only fire if the strip was *not*
        # rescued by synthesis above.  Surface it just before the
        # footer so the visual ordering is "assistant body → ●
        # dispatched group(s) → warning → footer".
        if (
            self._pending_phantom_strip is not None
            and self.stats.tool_calls == self._tool_calls_at_turn_start
        ):
            payload = self._pending_phantom_strip
            self._render_phantom_tool_hint(
                int(payload.get("stripped_chars", 0) or 0),
                raw_text=str(payload.get("raw_text", "") or ""),
                cleaned_text=str(payload.get("cleaned_text", "") or ""),
                turn_already_dispatched=bool(payload.get("turn_already_dispatched")),
            )
            self._phantom_hint_rendered = True
        self._pending_phantom_strip = None

        if (
            status == "COMPLETED"
            and self.stats.tool_calls == 0
            and (streaming_intent or parsed_intent)
            and not self._phantom_hint_rendered
        ):
            self._render_idle_turn_hint()

        # Render the always-on one-line turn footer.  Format::
        #
        #   ✓ done   ·   ↻ 5 iters   ·   ⚙ 7 tools   ·   2 errors   ·   68.9s
        #
        # Earlier versions glued the icon directly to the number
        # (``↻5``, ``⚙7/2 err``) which the user reported as "looks
        # overlapping, can't tell what each field is".  We now space
        # the icon away from the value AND tag every field with a
        # word label so the line is self-documenting.  The ``·``
        # separator is the same one used in the banner / hint line
        # so the visual style is consistent.
        sep = ("  " + (icon("dot") or "·") + "  ", AETHER_BORDER)

        line = Text()
        line.append("  ", style="")
        if status == "COMPLETED":
            line.append(f"{icon('success')} done", style=f"bold {AETHER_SUCCESS}")
        elif status == "INTERRUPTED":
            line.append(f"{icon('interrupt')} interrupted", style=f"bold {AETHER_WARNING}")
        elif status == "MAX_ITERATIONS":
            line.append(f"{icon('warn')} max iterations", style=f"bold {AETHER_WARNING}")
        else:
            line.append(f"{icon('error')} failed", style=f"bold {AETHER_ERROR}")

        line.append(*sep)
        iter_label = "iter" if iterations == 1 else "iters"
        line.append(f"{icon('iter')} {iterations} {iter_label}", style=AETHER_DIM)

        if self.stats.tool_calls:
            line.append(*sep)
            tool_label = "tool" if self.stats.tool_calls == 1 else "tools"
            line.append(
                f"{icon('tool')} {self.stats.tool_calls} {tool_label}",
                style=AETHER_DIM,
            )

        if self.stats.tool_errors:
            line.append(*sep)
            err_label = "error" if self.stats.tool_errors == 1 else "errors"
            line.append(
                f"{self.stats.tool_errors} {err_label}",
                style=f"bold {AETHER_ERROR}",
            )

        line.append(*sep)
        line.append(f"{self.stats.elapsed_sec:.1f}s", style=AETHER_DIM)

        # ``exit_reason`` is technical jargon ("natural_completion",
        # "tool_calls", …) that's only useful when debugging the engine —
        # show it in verbose mode and on non-success terminal states.
        if self.verbose or status != "COMPLETED":
            line.append(*sep)
            line.append(
                exit_reason.lower().replace("_", " "),
                style=f"dim {AETHER_DIM}",
            )

        self._ensure_block_spacer("footer")
        self.console.print(line)
        self._record_block("footer")
        if error:
            self.error(error)

    def _render_idle_turn_hint(self) -> None:
        """Warn when the model said something but never dispatched a tool.

        Only fires when the assistant text contains positive evidence of
        intent-to-call-a-tool (fenced shell block, ``$``-prefixed line,
        an "I'll run/我来运行/…" verb).  Polite greetings, clarifications,
        and short answers stay silent.

        When ``end_stream`` parsed a trailing fenced shell block out of
        the response (and stripped it from the body to avoid rendering
        a dangling code fence), surface the captured command on a
        ``└ attempted: $ <cmd>`` line.  That mirrors what
        :meth:`_render_phantom_tool_hint` does for inline XML tool
        tags so both failure modes look the same to the user — the
        critical question "what did the model try to do?" is
        answered consistently.
        """
        warn = Text()
        warn.append("  ", style="")
        warn.append(f"{icon('warn')} ", style=f"bold {AETHER_WARNING}")
        warn.append(
            "no tools were dispatched this turn",
            style=AETHER_WARNING,
        )
        warn.append(
            "  — the model wrote a command in prose instead of calling a tool.",
            style=f"dim {AETHER_DIM}",
        )
        self.console.print(warn)

        commands = list(self._last_attempted_commands)
        if not commands and self._last_trailing_command:
            commands = [self._last_trailing_command]
        if commands:
            # When the model emits multiple bash blocks in one turn,
            # render each on its own ``└ attempted: $ <cmd>`` line so
            # the user sees the full intended workflow — single-line
            # rendering would have hidden the 2nd / 3rd commands and
            # made it look like only one shell intent was lost.
            for idx, cmd in enumerate(commands):
                line = Text()
                line.append("    └ " if idx == 0 else "      ", style=f"dim {AETHER_DIM}")
                if idx == 0:
                    line.append("attempted: ", style=f"dim {AETHER_DIM}")
                line.append("$ ", style=f"bold {AETHER_TEXT}")
                line.append(_truncate_inline(cmd), style=AETHER_TEXT)
                self.console.print(line)

            hint = Text(
                "    try /system to ask the model to call tools instead of writing shell.",
                style=f"dim {AETHER_DIM}",
            )
            self.console.print(hint)

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

    def bump_response_chars(self, delta: str) -> None:
        """Advance the live token estimator without rendering ``delta``.

        pair with
        :attr:`EngineRequest.stream_silent_callback`.  Providers call
        this for streamed content that contributes to model output
        length but should NOT appear in the visible body — primarily
        OpenAI's ``delta.tool_calls.function.arguments`` fragments and
        Anthropic's ``input_json_delta`` events.

        Why a separate path (vs. just calling
        :meth:`stream_delta`)?  ``stream_delta`` appends to
        ``_stream_buffer`` so the markdown preview can re-render the
        cleaned body each tick; tool-arg JSON would land verbatim in
        that buffer and either pollute the body or force every
        consumer to filter it out.  Keeping the two channels distinct
        mirrors claude-code, where ``onUpdateLength`` (counter) is
        independent of the visible message store.

        Counts also flow into ``stats.streamed_chars`` so the per-turn
        footer / debug overlays see the same figure the activity bar
        uses.  No-op for empty / non-string deltas so providers don't
        need to filter at the call site.
        """
        if not isinstance(delta, str) or not delta:
            return
        n = len(delta)
        # Don't append to ``_stream_buffer`` — the body preview must
        # not see this content.  Updating ``response_chars`` directly
        # is enough for ``ActivityBar`` to tick.
        self._stream.char_count += n
        self.stats.streamed_chars += n
        self._turn_state.response_chars = self._stream.char_count

    def make_stream_silent_callback(self):
        """Return a closure suitable for ``EngineRequest.stream_silent_callback``.

        Symmetric with :meth:`make_stream_callback` — kept as a
        separate factory so REPL / app integrations can pass either
        callback (or both) without coupling.
        """
        def _cb(delta: str) -> None:
            self.bump_response_chars(delta)
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
