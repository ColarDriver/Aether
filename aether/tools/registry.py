"""Tool registry and dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.tools.task_cleanup import acquire_task_resource_for_executor
from aether.tools.base import ToolDescriptor, ToolExecutor, UnknownToolError


# Write-class tools that must be blocked while the session is in plan
# mode. ``EnterPlanModeTool`` flips
# ``aether.runtime.session.session_state`` to ``"plan"``; ``ExitPlanModeTool``
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
        # Write-class memory tools.
        "memory_write",
        "memory_update",
        "memory_forget",
    }
)


_PLAN_FILE_WRITE_TOOLS: frozenset[str] = frozenset({"write_file", "file_edit"})


@dataclass(slots=True, frozen=True)
class PlanModeBlock:
    message: str
    metadata: dict[str, Any]


@dataclass(slots=True, frozen=True)
class PlanModePlanFileWrite:
    plan_path: str


def _plan_mode_session_id(context: TurnContext) -> str | None:
    try:
        from aether.runtime.session.session_state import get_mode, SessionMode
    except Exception:
        return None
    session_id = getattr(context, "session_id", "") or ""
    if not session_id:
        return None
    if get_mode(session_id) != SessionMode.PLAN.value:
        return None
    return session_id


def _check_plan_mode_block(
    name: str,
    context: TurnContext,
    arguments: Mapping[str, Any] | None = None,
) -> PlanModeBlock | None:
    """Return a human-friendly refusal string when ``name`` must be
    blocked because the session is in plan mode.  Returns ``None`` to
    allow the call through (the common case)."""
    session_id = _plan_mode_session_id(context)
    if session_id is None:
        return None
    if name not in WRITE_TOOLS_BLOCKED_IN_PLAN:
        return None
    if _plan_file_write_metadata(name, context, arguments) is not None:
        return None

    allowed_path = _allowed_plan_path_for_message(session_id)
    metadata: dict[str, Any] = {
        "plan_mode_blocked": True,
        "tool_executed": False,
    }
    if allowed_path:
        metadata["allowed_plan_path"] = allowed_path
    message = (
        f"tool {name!r} is blocked while the session is in plan mode. "
        "The only write target allowed in plan mode is the current "
        "session plan file"
    )
    message += f": {allowed_path}." if allowed_path else "."
    message += (
        " Revise the plan file, then call exit_plan_mode to request "
        "user approval before resuming implementation actions."
    )
    return PlanModeBlock(message=message, metadata=metadata)


def _allowed_plan_path_for_message(session_id: str) -> str | None:
    try:
        from aether.runtime.session.plan_artifact import get_plan_path
    except Exception:
        return None
    try:
        return str(get_plan_path(session_id))
    except ValueError:
        return None


def _plan_file_write_metadata(
    name: str,
    context: TurnContext,
    arguments: Mapping[str, Any] | None = None,
) -> PlanModePlanFileWrite | None:
    session_id = _plan_mode_session_id(context)
    if session_id is None or name not in _PLAN_FILE_WRITE_TOOLS:
        return None
    if not isinstance(arguments, Mapping):
        return None
    raw_path = arguments.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    try:
        from aether.runtime.session.plan_artifact import get_plan_path
    except Exception:
        return None
    try:
        expected = get_plan_path(session_id)
    except ValueError:
        return None
    if _path_is_symlink_escape(expected):
        return None
    target = _normalise_tool_path(raw_path)
    expected_norm = _normalise_tool_path(str(expected))
    if target is None or expected_norm is None or target != expected_norm:
        return None
    return PlanModePlanFileWrite(plan_path=str(expected))


def _normalise_tool_path(raw_path: str) -> Path | None:
    try:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        parent = candidate.parent.resolve(strict=False)
        return parent / candidate.name
    except (OSError, RuntimeError, ValueError):
        return None


def _path_is_symlink_escape(path: Path) -> bool:
    try:
        if path.exists() and path.is_symlink():
            return True
        parent = path.parent
        if parent.exists() and parent.is_symlink():
            return True
    except OSError:
        return True
    return False


def plan_mode_plan_file_write(
    name: str,
    context: TurnContext,
    arguments: Mapping[str, Any] | None = None,
) -> PlanModePlanFileWrite | None:
    """Return metadata when a write tool is allowed only because it
    targets the current session plan file in plan mode."""
    return _plan_file_write_metadata(name, context, arguments)


def check_plan_mode_block(
    name: str,
    context: TurnContext,
    arguments: Mapping[str, Any] | None = None,
) -> PlanModeBlock | None:
    """Public wrapper for the engine permission gate.

    ``ToolRegistry.dispatch`` keeps its own call as a final safety net,
    but the engine checks this before prompting so plan mode does not
    show a misleading "approve write" dialog for an action that will be
    blocked regardless.
    """
    return _check_plan_mode_block(name, context, arguments)


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
        # Gate write-class tools when the session is in plan mode. We
        # return a structured ToolResult
        # rather than raising so the model sees a normal "this tool
        # refused" message and can correct course (typically by
        # calling ``exit_plan_mode``).
        refusal = _check_plan_mode_block(call.name, context, call.arguments)
        if refusal is not None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=refusal.message,
                is_error=True,
                metadata=refusal.metadata,
            )
        plan_file_write = plan_mode_plan_file_write(
            call.name, context, call.arguments
        )
        acquire_task_resource_for_executor(
            executor,
            task_id=context.task_id,
            context_metadata=context.metadata,
        )
        result = executor.execute(call, context)
        if plan_file_write is not None and not result.is_error:
            result.metadata.setdefault("plan_mode_plan_file_write", True)
            result.metadata.setdefault("plan_path", plan_file_write.plan_path)
        if (
            getattr(executor, "interrupt_behavior", "block") == "cancel"
            and context.interrupt_signal is not None
            and context.interrupt_signal.is_aborted()
        ):
            result.metadata.setdefault("interrupted", True)
        return result
