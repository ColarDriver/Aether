"""Built-in ``task_output`` tool — Sprint 3.5 / PR 3.5.6 (v1 stub).

In Aether's v1 subagent model the parent agent dispatches a task via
:class:`AgentTool` and *blocks* until the child returns its summary.
There is no streaming output channel exposed yet, so polling for
"give me what task X has produced so far" is meaningless: by the time
the model gets a chance to call :class:`TaskOutputTool` the dispatch
call has already returned.

We still ship the tool so the model's prompt can mention it without
producing "tool not found" errors, and so the upgrade path to async
fan-out (Sprint 4 / 5) keeps the same name.  The behaviour is a
structured "not supported in sync mode" refusal that points the model
back to ``AgentTool``'s direct return value.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor


class TaskOutputTool(ToolExecutor):
    """v1 stub: returns a structured 'not supported' error."""

    NAME = "task_output"

    def __init__(self) -> None:
        self._descriptor = ToolDescriptor(
            name=self.NAME,
            description=(
                "Fetch incremental output from a running subagent task. "
                "Reserved for the future async subagent model — in the "
                "current synchronous implementation the parent always "
                "blocks on the dispatch call, so this tool returns a "
                "'not supported' error. Read the task result directly "
                "from the task tool's return value instead."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Identifier returned by a prior task call.",
                    },
                    "since_offset": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "Reserved for the future streaming variant.",
                    },
                },
                "required": ["task_id"],
            },
            required=["task_id"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        task_id = args.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            return _error(call, "'task_id' is required and must be a non-empty string")
        return _error(
            call,
            (
                "task_output is not supported in synchronous subagent mode. "
                "Subagent tasks complete before returning to the parent — "
                "read the task result from the corresponding `task` tool "
                "call output instead. (task_id={tid!r})"
            ).format(tid=task_id),
            metadata={"task_id": task_id},
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
        content=message,
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["TaskOutputTool"]
