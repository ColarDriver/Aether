"""Built-in memory tools — Sprint 8 / PR 8.6.

Five tools that let the model explicitly read and write durable project
memory via the ``ProjectMemoryStore``.

Read-class tools (``memory_read``, ``memory_list``) are cheap and always
allowed.  Write-class tools (``memory_write``, ``memory_update``,
``memory_forget``) go through the write policy and permission gate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aether.memory.project_store import ProjectMemoryStore
from aether.memory.write_policy import check_write_policy
from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor


def _get_store(context: TurnContext) -> ProjectMemoryStore | None:
    return context.metadata.get("_project_memory_store")


def _get_config_value(context: TurnContext, key: str, default: Any = None) -> Any:
    config = context.metadata.get("_engine_config")
    if config is not None:
        return getattr(config, key, default)
    return default


def _error(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=f"error: {message}",
        is_error=True,
    )


def _no_store(call: ToolCall) -> ToolResult:
    return _error(call, "Project memory store is not available.")


_MAX_SUMMARY_CHARS = 120
_MAX_READ_RESULTS = 10
_MAX_RESULT_CHARS = 480


class MemoryReadTool(ToolExecutor):
    """Query project memory and return short summaries."""

    def __init__(self) -> None:
        self._descriptor = ToolDescriptor(
            name="memory_read",
            description=(
                "Search project memory for entries matching a query. "
                "Returns short summaries (id, topic, kind, and a text "
                "preview). Use this to recall project decisions, "
                "architecture notes, or known pitfalls."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — keywords or topic name.",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional topic filter (e.g. 'decisions', 'architecture').",
                    },
                },
                "required": ["query"],
            },
            required=["query"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        store = _get_store(context)
        if store is None:
            return _no_store(call)

        args = call.arguments or {}
        query = str(args.get("query") or "").strip().lower()
        topic = args.get("topic")

        try:
            entries = store.read_entries(
                topic=topic if topic else None,
                sanitize=True,
            )
        except Exception as exc:
            return _error(call, f"Failed to read memory: {type(exc).__name__}")

        if query:
            entries = tuple(
                e for e in entries
                if query in e.text.lower()
                or query in e.topic.lower()
                or any(query in t.lower() for t in e.tags)
            )

        entries = entries[:_MAX_READ_RESULTS]
        if not entries:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content="No matching memory entries found.",
            )

        lines: list[str] = []
        remaining = len(entries)
        for entry in entries:
            preview = entry.text[:_MAX_SUMMARY_CHARS]
            if len(entry.text) > _MAX_SUMMARY_CHARS:
                preview = preview.rstrip() + "..."
            line = f"- {entry.id} [{entry.topic}/{entry.kind.value}]: {preview}"
            if lines and len("\n".join(lines)) + len(line) + 1 > _MAX_RESULT_CHARS:
                lines.append(f"... ({remaining} more entries)")
                break
            lines.append(line)
            remaining -= 1

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content="\n".join(lines),
        )


class MemoryListTool(ToolExecutor):
    """List project memory topic entries (metadata only)."""

    def __init__(self) -> None:
        self._descriptor = ToolDescriptor(
            name="memory_list",
            description=(
                "List project memory entries showing metadata only "
                "(id, topic, kind, tags, updated_at). Does not return "
                "entry text — use memory_read for content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Optional topic filter.",
                    },
                },
            },
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        store = _get_store(context)
        if store is None:
            return _no_store(call)

        args = call.arguments or {}
        topic = args.get("topic")

        try:
            entries = store.read_entries(
                topic=topic if topic else None,
                sanitize=True,
            )
        except Exception as exc:
            return _error(call, f"Failed to list memory: {type(exc).__name__}")

        if not entries:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content="No memory entries found.",
            )

        lines: list[str] = []
        remaining = len(entries)
        for entry in entries:
            tags = ", ".join(entry.tags) if entry.tags else ""
            line = (
                f"- {entry.id} | topic={entry.topic} kind={entry.kind.value}"
                f" tags=[{tags}] updated={entry.updated_at or 'unknown'}"
            )
            if lines and len("\n".join(lines)) + len(line) + 1 > _MAX_RESULT_CHARS:
                lines.append(f"... ({remaining} more entries)")
                break
            lines.append(line)
            remaining -= 1

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content="\n".join(lines),
        )


class MemoryWriteTool(ToolExecutor):
    """Write a new project memory entry."""

    def __init__(self) -> None:
        self._descriptor = ToolDescriptor(
            name="memory_write",
            description=(
                "Write a new durable project memory entry. Requires a "
                "reason explaining why this memory is being saved. Only "
                "project-scope memory is allowed by default."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic (e.g. 'decisions', 'architecture', 'pitfalls', 'workflows').",
                    },
                    "text": {
                        "type": "string",
                        "description": "Memory content (max 2000 chars).",
                    },
                    "kind": {
                        "type": "string",
                        "description": "Entry kind (e.g. 'decision', 'project_fact', 'constraint', 'reference', 'warning').",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this memory is being saved (required for audit).",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Memory scope. Default 'project'. Only 'user' in personal_assistant mode.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for retrieval.",
                    },
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional related file paths.",
                    },
                    "confidence": {
                        "type": "string",
                        "description": "Confidence level: low, medium, high.",
                    },
                },
                "required": ["topic", "text", "reason"],
            },
            required=["topic", "text", "reason"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        store = _get_store(context)
        if store is None:
            return _no_store(call)

        args = call.arguments or {}
        scope = str(args.get("scope") or "project")
        text = str(args.get("text") or "")
        topic = str(args.get("topic") or "")
        reason = args.get("reason")
        kind = str(args.get("kind") or "project_fact")
        tags = tuple(str(t) for t in (args.get("tags") or []))
        paths = tuple(str(p) for p in (args.get("paths") or []))
        confidence = str(args.get("confidence") or "medium")

        mode = str(_get_config_value(context, "memory_mode", "project"))
        auto_write = bool(_get_config_value(context, "memory_auto_write_enabled", False))

        policy = check_write_policy(
            scope=scope, text=text, mode=mode,
            auto_write_enabled=auto_write, reason=reason,
        )
        if not policy.allowed:
            return _error(call, f"Write denied: {policy.reason}")

        result = store.write_entry(
            topic=topic,
            text=text,
            kind=kind,
            tags=tags,
            paths=paths,
            confidence=confidence,
            source_session=context.session_id,
        )

        if not result.success:
            return _error(call, f"Write failed: {result.error}")

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=(
                f"Memory written:\n"
                f"- id: {result.entry_id}\n"
                f"- scope: {scope}\n"
                f"- topic: {topic}\n"
                f"- source: {result.path}"
            ),
        )


class MemoryUpdateTool(ToolExecutor):
    """Update an existing project memory entry."""

    def __init__(self) -> None:
        self._descriptor = ToolDescriptor(
            name="memory_update",
            description=(
                "Update an existing project memory entry by id. "
                "The entry text is replaced; created_at is preserved."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "string",
                        "description": "ID of the entry to update.",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Topic containing the entry.",
                    },
                    "text": {
                        "type": "string",
                        "description": "New text content.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this entry is being updated.",
                    },
                },
                "required": ["entry_id", "topic", "text", "reason"],
            },
            required=["entry_id", "topic", "text", "reason"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        store = _get_store(context)
        if store is None:
            return _no_store(call)

        args = call.arguments or {}
        entry_id = str(args.get("entry_id") or "")
        topic = str(args.get("topic") or "")
        text = str(args.get("text") or "")
        reason = args.get("reason")

        mode = str(_get_config_value(context, "memory_mode", "project"))
        auto_write = bool(_get_config_value(context, "memory_auto_write_enabled", False))

        policy = check_write_policy(
            scope="project", text=text, mode=mode,
            auto_write_enabled=auto_write, reason=reason,
        )
        if not policy.allowed:
            return _error(call, f"Update denied: {policy.reason}")

        result = store.write_entry(
            topic=topic,
            text=text,
            entry_id=entry_id,
            source_session=context.session_id,
        )

        if not result.success:
            return _error(call, f"Update failed: {result.error}")

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=(
                f"Memory updated:\n"
                f"- id: {result.entry_id}\n"
                f"- topic: {topic}\n"
                f"- source: {result.path}"
            ),
        )


class MemoryForgetTool(ToolExecutor):
    """Mark a project memory entry as deleted (tombstone)."""

    def __init__(self) -> None:
        self._descriptor = ToolDescriptor(
            name="memory_forget",
            description=(
                "Mark a project memory entry as deleted by writing a "
                "tombstone. The entry is not physically removed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "string",
                        "description": "ID of the entry to forget.",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Topic containing the entry.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this entry is being forgotten.",
                    },
                },
                "required": ["entry_id", "topic", "reason"],
            },
            required=["entry_id", "topic", "reason"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        store = _get_store(context)
        if store is None:
            return _no_store(call)

        args = call.arguments or {}
        entry_id = str(args.get("entry_id") or "")
        topic = str(args.get("topic") or "")
        reason = str(args.get("reason") or "")

        mode = str(_get_config_value(context, "memory_mode", "project"))
        auto_write = bool(_get_config_value(context, "memory_auto_write_enabled", False))

        policy = check_write_policy(
            scope="project", text="[deleted]", mode=mode,
            auto_write_enabled=auto_write, reason=reason,
        )
        if not policy.allowed:
            return _error(call, f"Forget denied: {policy.reason}")

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        tombstone_text = f"[deleted] reason: {reason}"

        result = store.write_entry(
            topic=topic,
            text=tombstone_text,
            entry_id=entry_id,
            kind="warning",
            tags=("deleted",),
            source_session=context.session_id,
            reject_secrets=False,
        )

        if not result.success:
            return _error(call, f"Forget failed: {result.error}")

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=(
                f"Memory forgotten (tombstone):\n"
                f"- id: {result.entry_id}\n"
                f"- topic: {topic}\n"
                f"- deleted_at: {now}"
            ),
        )


__all__ = [
    "MemoryForgetTool",
    "MemoryListTool",
    "MemoryReadTool",
    "MemoryUpdateTool",
    "MemoryWriteTool",
]
