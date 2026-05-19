"""Built-in ``enter_plan_mode`` tool.

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


_PLAN_INSTRUCTIONS_TEMPLATE = (
    "Entered plan mode. Plan file: {plan_path}\n"
    "\n"
    "Follow the 5-phase plan workflow until exit_plan_mode is approved:\n"
    "\n"
    "Phase 1 -- Initial Understanding. Dispatch up to 3 ``Explore`` "
    "subagents in parallel via ``task(subagent_type=\"Explore\", ...)``, "
    "or use read_file / list_dir / grep / glob / web_fetch / web_search "
    "directly. Reuse existing utilities before proposing new code.\n"
    "Phase 2 -- Design. Launch a ``Plan`` subagent via "
    "``task(subagent_type=\"Plan\", ...)`` for non-trivial work; "
    "skip only for typos and single-line fixes.\n"
    "Phase 3 -- Review. Read the critical files yourself; use "
    "ask_user_question to clarify requirements (NOT to request plan "
    "approval).\n"
    "Phase 4 -- Final Plan. Write the plan to the file above with "
    "write_file (create) or file_edit (revise). Include Context, "
    "Approach, Files, Reused utilities, and Verification sections.\n"
    "Phase 5 -- Call exit_plan_mode. End your turn with either "
    "ask_user_question or exit_plan_mode -- not prose.\n"
    "\n"
    "Write-class tools (shell, write_file [except the plan file], "
    "file_edit [except the plan file], notebook_edit, todo_write, "
    "task with non-read-only subagent_type, task_stop, memory_write, "
    "memory_update, memory_forget) are blocked and return errors. "
    "read_file / list_dir / grep / glob are tool calls, not shell "
    "commands -- never invoke them via shell."
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
        _persist_session_mode(context.session_id, SessionMode.PLAN.value)
        plan_path = _plan_path_for_message(context.session_id)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=_PLAN_INSTRUCTIONS_TEMPLATE.format(plan_path=plan_path),
            is_error=False,
            metadata={
                "new_mode": SessionMode.PLAN.value,
                "plan_path": plan_path,
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
        content=message,
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["EnterPlanModeTool"]


def _persist_session_mode(session_id: str, mode: str) -> None:
    try:
        from aether.cli.sessions import load_session, save_session
    except Exception:  # pragma: no cover - defensive
        return
    record = load_session(session_id)
    if record is None:
        return
    record.mode = mode
    save_session(record)


def _plan_path_for_message(session_id: str) -> str:
    try:
        from aether.runtime.session.plan_artifact import get_plan_path
    except Exception:  # pragma: no cover - defensive
        return "(unavailable)"
    try:
        return str(get_plan_path(session_id))
    except ValueError:
        return "(unavailable)"
