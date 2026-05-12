"""Built-in ``enter_plan_mode`` tool — Sprint 3.5 / PR 3.5.7.

Switches the current session into *plan mode*.  Once in plan mode the
write-class tools (``shell``, ``write_file``, ``file_edit``,
``notebook_edit``, ``todo_write``, ``task``, ``task_stop``) are
intercepted by :mod:`aether.tools.registry` and return a structured
refusal until the model calls :class:`ExitPlanModeTool` and the user
approves the plan.

Plan mode is not allowed inside subagent contexts: a subagent that
flipped its own session mode could subvert the parent's plan/approve
contract.  The check happens at execute-time by inspecting
``context.metadata['_parent_agent'].delegate_depth``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.session.session_state import SessionMode, set_mode
from aether.tools.base import ToolDescriptor, ToolExecutor


_PLAN_INSTRUCTIONS = (
    "Entered plan mode. Until exit_plan_mode is approved:\n"
    "1. Use read_file / grep / glob / list_dir / web_fetch / web_search "
    "freely to understand the problem.\n"
    "2. Write-class tools (shell, write_file, file_edit, notebook_edit, "
    "todo_write, task, task_stop) are blocked and will return errors.\n"
    "3. Use ask_user_question if you need clarification.\n"
    "4. When you have a concrete plan, call exit_plan_mode with the plan "
    "as a markdown string. The user will review and approve before any "
    "write tool runs again.\n"
    "Stay focused on understanding before changing."
)


class EnterPlanModeTool(ToolExecutor):
    """Flip the session into plan mode and return guidance."""

    NAME = "enter_plan_mode"

    def __init__(self) -> None:
        self._descriptor = ToolDescriptor(
            name=self.NAME,
            description=(
                "Switch the current session into 'plan mode' so the model "
                "can explore freely without making changes. Read tools "
                "remain available; write tools (shell / write_file / "
                "file_edit / notebook_edit / todo_write / task) start "
                "returning a structured refusal until exit_plan_mode is "
                "approved by the user. Cannot be used inside subagent "
                "contexts."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            required=[],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        parent_agent = context.metadata.get("_parent_agent") if context.metadata else None
        delegate_depth = int(getattr(parent_agent, "delegate_depth", 0) or 0)
        if delegate_depth > 0:
            return _error(
                call,
                "enter_plan_mode is not allowed inside subagent contexts; "
                "the top-level agent owns the plan/approve contract",
            )

        config = context.metadata.get("_engine_config") if context.metadata else None
        if not bool(getattr(config, "plan_mode_enabled", True)):
            return _error(call, "plan mode is disabled by configuration")

        if not context.session_id:
            return _error(call, "session_id missing on TurnContext; cannot set mode")

        set_mode(context.session_id, SessionMode.PLAN)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=_PLAN_INSTRUCTIONS,
            is_error=False,
            metadata={"new_mode": SessionMode.PLAN.value},
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


__all__ = ["EnterPlanModeTool"]
