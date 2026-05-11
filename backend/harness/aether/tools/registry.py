"""Tool registry and dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from aether.runtime.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.task_cleanup import acquire_task_resource_for_executor
from aether.tools.base import ToolDescriptor, ToolExecutor, UnknownToolError


# Sprint 3.5 / PR 3.5.7 — write-class tools that must be blocked while
# the session is in plan mode.  ``EnterPlanModeTool`` flips
# ``aether.runtime.session_state`` to ``"plan"``; ``ExitPlanModeTool``
# flips it back after user approval.  Read-only tools (read_file,
# grep, glob, list_dir, web_fetch, web_search, skill, ask_user_question,
# enter_plan_mode, exit_plan_mode, task_output) are intentionally NOT
# in this set so the model can keep exploring while planning.
WRITE_TOOLS_BLOCKED_IN_PLAN: frozenset[str] = frozenset(
    {
        "shell",
        "write_file",
        "file_edit",
        "notebook_edit",
        "todo_write",
        # subagent dispatch is also blocked — a subagent could trivially
        # mutate state on our behalf, defeating plan-mode's purpose.
        "task",
        "task_stop",
    }
)


def _check_plan_mode_block(name: str, context: TurnContext) -> str | None:
    """Return a human-friendly refusal string when ``name`` must be
    blocked because the session is in plan mode.  Returns ``None`` to
    allow the call through (the common case)."""
    try:
        from aether.runtime.session_state import get_mode, SessionMode
    except Exception:
        return None
    session_id = getattr(context, "session_id", "") or ""
    if not session_id:
        return None
    if get_mode(session_id) != SessionMode.PLAN.value:
        return None
    if name not in WRITE_TOOLS_BLOCKED_IN_PLAN:
        return None
    return (
        f"tool {name!r} is blocked while the session is in plan mode. "
        "Call exit_plan_mode with your concrete plan to request user "
        "approval before resuming write actions."
    )


@dataclass(slots=True)
class ToolRegistry:
    _tools: Dict[str, ToolExecutor] = field(default_factory=dict)

    def register(self, executor: ToolExecutor) -> None:
        name = executor.descriptor.name
        self._tools[name] = executor

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolExecutor:
        if name not in self._tools:
            raise UnknownToolError(name)
        return self._tools[name]

    def get_descriptor(self, name: str) -> ToolDescriptor:
        return self.get(name).descriptor

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def list_descriptors(self) -> List[ToolDescriptor]:
        return [tool.descriptor for tool in self._tools.values()]

    def dispatch(self, call: ToolCall, context: TurnContext) -> ToolResult:
        executor = self.get(call.name)
        executor.validate(call)
        # Sprint 3.5 / PR 3.5.7 — gate write-class tools when the
        # session is in plan mode.  We return a structured ToolResult
        # rather than raising so the model sees a normal "this tool
        # refused" message and can correct course (typically by
        # calling ``exit_plan_mode``).
        refusal = _check_plan_mode_block(call.name, context)
        if refusal is not None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=refusal,
                is_error=True,
                metadata={"plan_mode_blocked": True},
            )
        acquire_task_resource_for_executor(
            executor,
            task_id=context.task_id,
            context_metadata=context.metadata,
        )
        result = executor.execute(call, context)
        if (
            getattr(executor, "interrupt_behavior", "block") == "cancel"
            and context.interrupt_signal is not None
            and context.interrupt_signal.is_aborted()
        ):
            result.metadata.setdefault("interrupted", True)
        return result
