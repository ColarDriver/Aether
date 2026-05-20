"""Tool capability classification for scheduling decisions.

The classifier is intentionally conservative. Unknown tools are sequential by
default; only built-ins with read-only, independent behaviour are marked as
parallel safe.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCapabilities:
    read_only: bool = False
    mutates_files: bool = False
    interactive: bool = False
    requires_permission: bool = False
    path_scoped: bool = False
    parallel_safe: bool = False
    cheap: bool = False


_READ_ONLY_PARALLEL: frozenset[str] = frozenset(
    {
        "read_file",
        "list_dir",
        "grep",
        "glob",
        "web_fetch",
        "web_search",
        "task_output",
    }
)

_PATH_SCOPED: frozenset[str] = frozenset(
    {
        "read_file",
        "list_dir",
        "grep",
        "glob",
    }
)

_MUTATING_FILE_TOOLS: frozenset[str] = frozenset(
    {
        "write_file",
        "file_edit",
        "notebook_edit",
    }
)

_INTERACTIVE_TOOLS: frozenset[str] = frozenset(
    {
        "ask_user_question",
        "exit_plan_mode",
    }
)

_SEQUENTIAL_TOOLS: frozenset[str] = frozenset(
    {
        "shell",
        "task",
        "send_message",
        "task_stop",
        "web_browser",
        "lsp",
        "skill",
        "enter_plan_mode",
        "exit_plan_mode",
        "todo_write",
        "memory_write",
        "memory_update",
        "memory_forget",
    }
)

_CHEAP_TOOLS: frozenset[str] = frozenset(
    {
        "update_todo",
        "todo_write",
        "memory",
        "memory_write",
        "memory_read",
        "skill_manage",
        "session_search",
    }
)


def capabilities_for_tool_name(name: str) -> ToolCapabilities:
    normalized = normalize_tool_name(name)
    if normalized in _READ_ONLY_PARALLEL:
        return ToolCapabilities(
            read_only=True,
            path_scoped=normalized in _PATH_SCOPED,
            parallel_safe=True,
            cheap=normalized in _CHEAP_TOOLS,
        )
    if normalized in _MUTATING_FILE_TOOLS:
        return ToolCapabilities(
            mutates_files=True,
            requires_permission=True,
            path_scoped=True,
            parallel_safe=False,
            cheap=normalized in _CHEAP_TOOLS,
        )
    if normalized in _INTERACTIVE_TOOLS:
        return ToolCapabilities(
            read_only=True,
            interactive=True,
            parallel_safe=False,
            cheap=normalized in _CHEAP_TOOLS,
        )
    if normalized in _SEQUENTIAL_TOOLS:
        return ToolCapabilities(
            read_only=normalized in {"memory_read"},
            mutates_files=normalized in {"shell", "todo_write"},
            interactive=normalized in _INTERACTIVE_TOOLS,
            requires_permission=normalized in {"shell", "task_stop"},
            parallel_safe=False,
            cheap=normalized in _CHEAP_TOOLS,
        )
    return ToolCapabilities(cheap=normalized in _CHEAP_TOOLS)


def is_parallel_safe_tool(name: str) -> bool:
    return capabilities_for_tool_name(name).parallel_safe


def extract_tool_scope_path(
    name: str,
    arguments: Mapping[str, Any] | None,
    cwd: Path,
) -> Path | None:
    capabilities = capabilities_for_tool_name(name)
    if not capabilities.path_scoped or not isinstance(arguments, Mapping):
        return None
    for key in ("path", "file_path", "target"):
        raw = arguments.get(key)
        if isinstance(raw, str) and raw.strip():
            return normalize_scope_path(raw, cwd)
    return None


def normalize_scope_path(raw_path: str, cwd: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    return candidate.resolve(strict=False)


def normalize_tool_name(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_")


__all__ = [
    "ToolCapabilities",
    "capabilities_for_tool_name",
    "extract_tool_scope_path",
    "is_parallel_safe_tool",
    "normalize_scope_path",
    "normalize_tool_name",
]
