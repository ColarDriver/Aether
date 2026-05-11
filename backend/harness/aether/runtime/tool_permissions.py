"""Runtime contracts and policy helpers for tool permission checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from aether.runtime.contracts import ToolCall, ToolResult


class ToolPermissionMode(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class ToolPermissionDecisionType(str, Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"
    ABORT = "abort"


@dataclass(slots=True, frozen=True)
class ToolPermissionPreview:
    title: str
    subtitle: str | None = None
    body: str | None = None
    diff: str | None = None
    path: str | None = None
    command: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolPermissionRule:
    tool_name: str
    behavior: ToolPermissionMode = ToolPermissionMode.ALLOW
    scope: str = "session"
    path_prefix: str | None = None
    command_prefix: str | None = None
    reason: str | None = None


@dataclass(slots=True, frozen=True)
class ToolPermissionRequest:
    session_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    category: str
    risk: str
    preview: ToolPermissionPreview | None = None
    reason: str | None = None
    allow_session: bool = True


@dataclass(slots=True, frozen=True)
class ToolPermissionDecision:
    type: ToolPermissionDecisionType
    updated_arguments: dict[str, Any] | None = None
    feedback: str | None = None
    rule: ToolPermissionRule | None = None
    source: str = "user"


@runtime_checkable
class ToolPermissionPrompter(Protocol):
    def is_interactive(self) -> bool: ...

    def request_tool_permission(
        self,
        request: ToolPermissionRequest,
        *,
        timeout: float | None = None,
    ) -> ToolPermissionDecision: ...


DANGEROUS_TOOLS: frozenset[str] = frozenset(
    {
        "shell",
        "write_file",
        "file_edit",
        "notebook_edit",
        "todo_write",
        "task",
        "task_stop",
    }
)

READONLY_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "list_dir",
        "grep",
        "glob",
        "web_fetch",
        "web_search",
        "task_output",
        "skill",
        "lsp",
    }
)


def default_permission_stats(*, enabled: bool = True) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "asked": 0,
        "allowed_once": 0,
        "allowed_session": 0,
        "denied": 0,
        "aborted": 0,
        "non_interactive_denied": 0,
        "session_rules_added": 0,
    }


def normalize_permission_mode(value: Any, *, default: ToolPermissionMode) -> ToolPermissionMode:
    try:
        return ToolPermissionMode(str(value).strip().lower())
    except Exception:
        return default


def is_dangerous_tool(tool_name: str) -> bool:
    return tool_name in DANGEROUS_TOOLS


def category_for_tool(tool_name: str) -> str:
    if tool_name == "shell":
        return "shell"
    if tool_name in {"write_file", "file_edit", "notebook_edit"}:
        return "write"
    if tool_name in {"task", "task_stop"}:
        return "delegate"
    if tool_name == "todo_write":
        return "state"
    if tool_name in READONLY_TOOLS:
        return "read"
    return "tool"


def risk_for_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name != "shell":
        return "write" if is_dangerous_tool(tool_name) else "read"
    command = str(arguments.get("command") or "").strip()
    lowered = command.lower()
    high_markers = ("rm ", "sudo ", "chmod ", "chown ", "mkfs", "dd ", ">")
    medium_markers = ("git push", "git commit", "pip install", "npm install", "uv add")
    if any(marker in lowered for marker in high_markers):
        return "high"
    if any(marker in lowered for marker in medium_markers):
        return "medium"
    return "shell"


def make_permission_request(
    call: ToolCall,
    *,
    session_id: str,
    preview: ToolPermissionPreview | None = None,
    reason: str | None = None,
    allow_session: bool = True,
) -> ToolPermissionRequest:
    args = dict(call.arguments or {})
    return ToolPermissionRequest(
        session_id=session_id,
        tool_call_id=call.id,
        tool_name=call.name,
        arguments=args,
        category=category_for_tool(call.name),
        risk=risk_for_tool(call.name, args),
        preview=preview,
        reason=reason,
        allow_session=allow_session,
    )


def build_fallback_preview(call: ToolCall) -> ToolPermissionPreview:
    args = dict(call.arguments or {})
    title = "Use tool"
    body = None
    path = _path_from_arguments(args)
    command = str(args.get("command") or "").strip() if call.name == "shell" else None
    if call.name == "shell":
        title = "Run command"
    elif call.name in {"file_edit", "notebook_edit"}:
        title = "Edit file"
    elif call.name == "write_file":
        title = "Write file"
    if not command and args:
        body = _summarize_arguments(args)
    return ToolPermissionPreview(
        title=title,
        subtitle=path,
        body=body,
        path=path,
        command=command or None,
    )


def find_matching_rule(
    request: ToolPermissionRequest,
    rules: list[ToolPermissionRule],
) -> ToolPermissionRule | None:
    matching = [rule for rule in rules if _rule_matches(rule, request)]
    for rule in matching:
        if rule.behavior == ToolPermissionMode.DENY:
            return rule
    return matching[0] if matching else None


def build_session_rule_for_request(
    request: ToolPermissionRequest,
    *,
    behavior: ToolPermissionMode = ToolPermissionMode.ALLOW,
) -> ToolPermissionRule:
    preview = request.preview
    path = preview.path if preview and preview.path else _path_from_arguments(request.arguments)
    command = preview.command if preview and preview.command else str(request.arguments.get("command") or "")
    command_prefix = _shell_command_prefix(command) if request.tool_name == "shell" else None
    return ToolPermissionRule(
        tool_name=request.tool_name,
        behavior=behavior,
        scope="session",
        path_prefix=_normalize_path(path) if path and request.tool_name != "shell" else None,
        command_prefix=command_prefix,
        reason="allowed by user for this session",
    )


def build_permission_denied_result(
    request: ToolPermissionRequest,
    decision: ToolPermissionDecision,
) -> ToolResult:
    if decision.type == ToolPermissionDecisionType.ABORT:
        content = "permission prompt aborted before executing tool"
    else:
        content = "permission denied by user before executing tool"
    if decision.feedback:
        content += f": {decision.feedback}"
    return ToolResult(
        tool_call_id=request.tool_call_id,
        name=request.tool_name,
        content=content,
        is_error=True,
        metadata={
            "permission_denied": True,
            "permission_decision": decision.type.value,
            "permission_source": decision.source,
            "tool_executed": False,
        },
    )


def _rule_matches(rule: ToolPermissionRule, request: ToolPermissionRequest) -> bool:
    if rule.tool_name != request.tool_name:
        return False
    if rule.path_prefix:
        preview_path = request.preview.path if request.preview and request.preview.path else None
        candidate = _normalize_path(preview_path or _path_from_arguments(request.arguments))
        return bool(candidate) and candidate.startswith(rule.path_prefix)
    if rule.command_prefix:
        command = request.preview.command if request.preview and request.preview.command else None
        candidate = str(command or request.arguments.get("command") or "").strip()
        return candidate.startswith(rule.command_prefix)
    return True


def _path_from_arguments(arguments: dict[str, Any]) -> str | None:
    value = arguments.get("path")
    if value is None:
        value = arguments.get("file_path")
    if value is None:
        return None
    return str(value)


def _normalize_path(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return str(Path(path).expanduser().resolve(strict=False))
    except Exception:
        return str(path)


def _shell_command_prefix(command: str | None) -> str | None:
    if not command:
        return None
    stripped = command.strip()
    if not stripped:
        return None
    separators = ("&&", "||", ";", "|")
    first = stripped
    for separator in separators:
        if separator in first:
            first = first.split(separator, 1)[0].strip()
    parts = first.split()
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return " ".join(parts[:2])


def _summarize_arguments(arguments: dict[str, Any], *, max_chars: int = 1200) -> str:
    parts: list[str] = []
    for key, value in sorted(arguments.items()):
        text = repr(value)
        if len(text) > 240:
            text = text[:237] + "..."
        parts.append(f"{key}: {text}")
    body = "\n".join(parts)
    if len(body) > max_chars:
        return body[: max_chars - 3] + "..."
    return body


__all__ = [
    "DANGEROUS_TOOLS",
    "READONLY_TOOLS",
    "ToolPermissionDecision",
    "ToolPermissionDecisionType",
    "ToolPermissionMode",
    "ToolPermissionPreview",
    "ToolPermissionPrompter",
    "ToolPermissionRequest",
    "ToolPermissionRule",
    "build_fallback_preview",
    "build_permission_denied_result",
    "build_session_rule_for_request",
    "category_for_tool",
    "default_permission_stats",
    "find_matching_rule",
    "is_dangerous_tool",
    "make_permission_request",
    "normalize_permission_mode",
    "risk_for_tool",
]
