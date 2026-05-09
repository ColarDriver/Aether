"""Built-in ``todo_write`` tool — Sprint 3.5 / PR 3.5.3.

Lets the model maintain a session-scoped task checklist.  Closes the
loop opened by Sprint 3 / PR 3.2 \u2014 ``EngineConfig.cheap_tool_names``
already lists ``todo_write`` / ``update_todo`` as cheap-refundable
names, so a turn whose only tool call is ``todo_write`` no longer
consumes an iteration slot.

State model
-----------
The complete ``todos`` array is **replaced** on every call (not patched
or appended).  This matches claude-code's contract and keeps the model
prompt simple: it always sends "here is the current full list".  The
store is a module-level dict keyed by ``session_id``; concurrent
sessions can't see each other's todos.

When **all** todos reach a terminal status (``completed`` or
``cancelled``) the store entry is cleared automatically \u2014 the model
implicitly signals "nothing left to track" by sending an all-done
list, and we don't want stale done items lingering across turns.

The current process-local dict is intentionally simple; a future
sprint can swap it for a persistent ``SessionStore`` integration
without touching the tool surface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aether.runtime.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor


_VALID_STATUSES = frozenset({"pending", "in_progress", "completed", "cancelled"})

# Module-level session-keyed store.  Keeping it module-private (no
# ``__all__`` export) signals the dict is implementation detail; tests
# that need it import the helper functions explicitly.
_TODO_STORE: Dict[str, List[Dict[str, Any]]] = {}


def get_session_todos(session_id: str) -> List[Dict[str, Any]]:
    """Return a *copy* of the current todo list for the session.

    Returning a copy (not the live list) is intentional \u2014 callers
    that introspect or test the store should not be able to mutate
    state by accident.
    """
    return [dict(t) for t in _TODO_STORE.get(session_id, [])]


def set_session_todos(session_id: str, todos: List[Dict[str, Any]]) -> None:
    """Replace the session's todo list with ``todos``.

    Empty lists deliberately remain in the store as ``[]`` rather than
    being deleted: an explicit empty write is distinguishable from
    "session never wrote any todos" via the ``in`` operator.
    """
    _TODO_STORE[session_id] = [dict(t) for t in todos]


def clear_session_todos(session_id: str) -> None:
    """Drop the session entry entirely.  Used by tests; the tool itself
    stores ``[]`` rather than deleting so cross-turn observers see
    consistent shapes."""
    _TODO_STORE.pop(session_id, None)


class TodoWriteTool(ToolExecutor):
    """Maintain a session-scoped task checklist."""

    def __init__(self) -> None:
        self._descriptor = ToolDescriptor(
            name="todo_write",
            description=(
                "Replace the session task checklist with the supplied list. "
                "Always send the COMPLETE updated list, not a delta. Items "
                "have ``id`` (stable identifier), ``content`` (one-line "
                "description), and ``status`` (one of ``pending``, "
                "``in_progress``, ``completed``, ``cancelled``). When the "
                "entire list reaches a terminal status the checklist is "
                "cleared automatically. This tool is cheap \u2014 a turn whose "
                "only tool call is ``todo_write`` does not consume an "
                "iteration slot."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": (
                            "Complete updated todo list (replaces the prior "
                            "list)."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Stable identifier across turns.",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "One-line task description.",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": sorted(_VALID_STATUSES),
                                },
                            },
                            "required": ["id", "content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
            required=["todos"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        todos = args.get("todos", [])
        if not isinstance(todos, list):
            return _error(call, "'todos' must be a list")

        # Validate every item up-front before mutating store \u2014 partial
        # writes are worse than rejected writes here.
        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                return _error(call, f"todos[{i}] must be an object")
            for required in ("id", "content", "status"):
                if required not in item:
                    return _error(
                        call,
                        f"todos[{i}] missing required field: {required!r}",
                    )
            if item["status"] not in _VALID_STATUSES:
                return _error(
                    call,
                    (
                        f"todos[{i}].status invalid: {item['status']!r} \u2014 "
                        f"must be one of {sorted(_VALID_STATUSES)}"
                    ),
                )
            if not isinstance(item["id"], str) or not item["id"].strip():
                return _error(call, f"todos[{i}].id must be a non-empty string")
            if not isinstance(item["content"], str):
                return _error(call, f"todos[{i}].content must be a string")

        terminal_statuses = {"completed", "cancelled"}
        all_done = bool(todos) and all(
            t["status"] in terminal_statuses for t in todos
        )
        new_todos: List[Dict[str, Any]] = [] if all_done else todos
        set_session_todos(context.session_id, new_todos)

        pending = sum(1 for t in todos if t["status"] == "pending")
        in_progress = sum(1 for t in todos if t["status"] == "in_progress")
        completed = sum(1 for t in todos if t["status"] == "completed")
        cancelled = sum(1 for t in todos if t["status"] == "cancelled")
        msg = (
            f"todos updated: {len(todos)} total "
            f"({pending} pending, {in_progress} in progress, "
            f"{completed} completed, {cancelled} cancelled)"
        )
        if all_done:
            msg += "\n[cleared \u2014 all tasks complete]"

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=msg,
            is_error=False,
            metadata={
                "todo_count": len(todos),
                "pending": pending,
                "in_progress": in_progress,
                "completed": completed,
                "cancelled": cancelled,
                "cleared": all_done,
            },
        )


def _error(
    call: ToolCall,
    message: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=f"error: {message}",
        is_error=True,
        metadata=metadata or {},
    )


__all__ = [
    "TodoWriteTool",
    "get_session_todos",
    "set_session_todos",
    "clear_session_todos",
]
