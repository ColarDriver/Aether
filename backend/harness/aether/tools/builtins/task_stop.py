"""Built-in ``task_stop`` tool — Sprint 3.5 / PR 3.5.6.

Best-effort cooperative cancellation for a running subagent.  Routes
to :meth:`SubagentManager.stop_task`, which sets a per-task
``threading.Event`` and calls ``AgentEngine.interrupt`` on the active
child.  In the current synchronous dispatch model the only realistic
caller is a *deeper* subagent stopping a *peer*; the parent agent in
the same turn already blocks until its dispatch returns and so cannot
race a stop call against the child's run loop.

Manager resolution mirrors :class:`AgentTool`:

1. ``context.metadata['_subagent_manager']``
2. Constructor injection
3. ``parent_agent.subagent_manager`` (via ``_parent_agent`` in metadata)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from aether.runtime.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor


class TaskStopTool(ToolExecutor):
    """Send a stop signal to a running subagent task."""

    NAME = "task_stop"

    def __init__(self, *, subagent_manager: Any | None = None) -> None:
        self._subagent_manager = subagent_manager
        self._descriptor = ToolDescriptor(
            name=self.NAME,
            description=(
                "Request graceful interruption of a running subagent task. "
                "Best-effort: the child observes the signal at its next "
                "iteration boundary and stops. Returns success when the "
                "signal was delivered, an error if the task has already "
                "finished or was never registered."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Identifier returned by a prior task call.",
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
        task_id = task_id.strip()

        config = context.metadata.get("_engine_config") if context.metadata else None
        if not bool(getattr(config, "allow_subagent_dispatch", True)):
            return _error(call, "subagent dispatch is disabled by configuration")

        manager = self._subagent_manager or (
            context.metadata.get("_subagent_manager") if context.metadata else None
        )
        if manager is None:
            parent = context.metadata.get("_parent_agent") if context.metadata else None
            manager = getattr(parent, "subagent_manager", None) if parent is not None else None
        if manager is None or not hasattr(manager, "stop_task"):
            return _error(
                call,
                "task_stop unavailable: SubagentManager is not configured "
                "or does not support stop_task",
                metadata={"task_id": task_id},
            )

        ok = bool(manager.stop_task(task_id))
        if ok:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"stop signal sent to task {task_id}",
                is_error=False,
                metadata={"task_id": task_id, "delivered": True},
            )
        return _error(
            call,
            f"task {task_id} not found or already complete",
            metadata={"task_id": task_id, "delivered": False},
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


__all__ = ["TaskStopTool"]
