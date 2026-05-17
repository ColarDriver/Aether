"""Built-in ``send_message`` tool — PR 10.7.

Queue an additional user-role message for a running async subagent.
The child drains its pending queue at the next iteration boundary
(see :meth:`AgentEngine._drain_pending_messages`) and the message
appears as a fresh ``user`` turn in the child's conversation.

Constraints:

- Target task must be in ``RUNNING`` status; sending to a terminal
  task returns a clear error so the model can adjust.
- Same gateway process only — cross-process / remote agent routing
  is intentionally out of scope for this PR.

Catalog resolution mirrors :class:`TaskOutputTool`: constructor
injection wins, then ``context.metadata['_task_store']`` (set by
:meth:`AgentEngine._prepare_turn_entry`).  Without a store the tool
returns a clear "not configured" error rather than silently
degrading.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.tasks import TaskStatus, TaskStore
from aether.tools.base import ToolDescriptor, ToolExecutor


class SendMessageTool(ToolExecutor):
    """Queue a user-role message for a running async subagent."""

    NAME = "send_message"

    def __init__(self, task_store: TaskStore | None = None) -> None:
        self._task_store = task_store
        self._descriptor = ToolDescriptor(
            name=self.NAME,
            description=(
                "Queue an additional user-role message for a running "
                "async subagent.  The child sees it at its next "
                "iteration boundary as a user turn.  Use this to add "
                "clarifications, redirect work, or share results from "
                "peer tasks.  Returns immediately; does not wait for "
                "the child to read the message.  Target task must be "
                "in 'running' status — sending to a terminal task "
                "returns an error."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": (
                            "Target task_id (returned by `task` when "
                            "``run_in_background=true``)."
                        ),
                    },
                    "message": {
                        "type": "string",
                        "description": (
                            "Content to deliver as a user-role turn to "
                            "the subagent."
                        ),
                    },
                    "summary": {
                        "type": "string",
                        "description": (
                            "Optional 5-10 word preview, currently "
                            "telemetry-only."
                        ),
                    },
                },
                "required": ["to", "message"],
            },
            required=["to", "message"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        to = args.get("to")
        message = args.get("message")
        if not isinstance(to, str) or not to.strip():
            return _error(call, "'to' is required and must be a non-empty string")
        if not isinstance(message, str) or not message.strip():
            return _error(
                call, "'message' is required and must be a non-empty string"
            )
        to = to.strip()
        message = message.strip()

        summary_arg = args.get("summary")
        if summary_arg is not None and not isinstance(summary_arg, str):
            return _error(call, "'summary' must be a string when provided")
        summary = summary_arg.strip() if isinstance(summary_arg, str) else None

        store = self._resolve_store(context)
        if store is None:
            return _error(
                call,
                "send_message unavailable: no TaskStore configured "
                "(check EngineConfig.task_store_enabled).",
            )

        record = store.read(to)
        if record is None:
            return _error(call, f"unknown task_id: {to!r}")
        if record.status != TaskStatus.RUNNING:
            return _error(
                call,
                f"cannot send to {to!r}: status is "
                f"{record.status.value!r} (must be 'running')",
            )

        store.enqueue_pending_message(to, message)
        body = (
            f"Queued. Subagent {to} will see the message at its "
            f"next iteration boundary."
        )
        metadata: Dict[str, Any] = {
            "to": to,
            "queued_chars": len(message),
        }
        if summary is not None:
            metadata["summary"] = summary
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=body,
            is_error=False,
            metadata=metadata,
        )

    # ------------------------------------------------------------ helpers

    def _resolve_store(self, context: TurnContext) -> TaskStore | None:
        if self._task_store is not None:
            return self._task_store
        injected = context.metadata.get("_task_store") if context.metadata else None
        return injected if isinstance(injected, TaskStore) else None


def _error(
    call: ToolCall,
    message: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=message,
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["SendMessageTool"]
