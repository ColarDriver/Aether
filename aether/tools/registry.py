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
        # subagent dispatch is blocked by default — a write-capable
        # subagent could trivially mutate state on our behalf,
        # defeating plan-mode's purpose.  Read-only subagent types
        # (``Explore``, ``Plan``) are carved out via
        # ``_PLAN_MODE_ALLOWED_SUBAGENT_TYPES`` below so the 5-phase
        # plan workflow can still dispatch them.
        "task",
        "task_stop",
        # Write-class memory tools.
        "memory_write",
        "memory_update",
        "memory_forget",
    }
)


_PLAN_FILE_WRITE_TOOLS: frozenset[str] = frozenset({"write_file", "file_edit"})

# Subagent types whose own tool whitelist makes them safe to dispatch
# from inside plan mode. ``Explore`` is read-only by definition;
# ``Plan`` may write only to the designated plan file (its own
# write_file is further constrained by the parent's plan-file gate).
# Defined in ``aether/agents/types/builtin.py``.
_PLAN_MODE_ALLOWED_SUBAGENT_TYPES: frozenset[str] = frozenset(
    {"Explore", "Plan"}
)


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
    # Carve-out: ``task`` is in the blocklist (a write-capable subagent
    # could mutate state for us) but read-only subagent types are safe
    # — their own tool whitelist enforces the constraint.  Letting
    # ``Explore`` / ``Plan`` through is what makes the 5-phase plan
    # workflow actually executable.
    if name == "task" and _is_allowed_plan_mode_subagent(arguments):
        return None

    allowed_path = _allowed_plan_path_for_message(session_id)
    metadata: dict[str, Any] = {
        "plan_mode_blocked": True,
        "tool_executed": False,
    }
    if allowed_path:
        metadata["allowed_plan_path"] = allowed_path
    if name == "shell":
        # ``shell`` is the single tool the model most often regresses
        # to in plan mode — it's where the ``read_file <path>`` /
        # ``pip install`` mistakes from the original bug report all
        # originated.  Spell out the correct routing in the refusal so
        # the model can self-correct on its next turn.
        message = (
            "tool 'shell' is blocked while the session is in plan mode. "
            "If you wanted to read a file, call read_file. To list a "
            "directory, call list_dir. To search content, call grep. "
            "To find files by name, call glob. These are tool calls, "
            "not shell binaries — do not invoke them via the shell "
            "tool. To run anything that mutates state, finalize the "
            "plan file"
        )
        message += f" ({allowed_path})" if allowed_path else ""
        message += (
            " and call exit_plan_mode to request user approval before "
            "resuming implementation actions."
        )
    elif name == "task":
        # Task dispatch landed here because the requested
        # ``subagent_type`` wasn't in the read-only carve-out.  Point
        # the model at the allowed subagent types so it doesn't just
        # retry blindly.
        allowed_types = ", ".join(sorted(_PLAN_MODE_ALLOWED_SUBAGENT_TYPES))
        message = (
            "tool 'task' is blocked in plan mode for this subagent_type. "
            f"Only read-only subagent types are allowed: {allowed_types}. "
            "Dispatch one of those, or do the exploration directly with "
            "read_file / list_dir / grep / glob. To resume general "
            "subagent dispatch, finalize the plan file"
        )
        message += f" ({allowed_path})" if allowed_path else ""
        message += (
            " and call exit_plan_mode to request user approval."
        )
    else:
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


def _is_allowed_plan_mode_subagent(
    arguments: Mapping[str, Any] | None,
) -> bool:
    """Return True when a ``task`` call should be allowed through the
    plan-mode gate because its ``subagent_type`` is read-only."""
    if not isinstance(arguments, Mapping):
        return False
    raw = arguments.get("subagent_type")
    if not isinstance(raw, str):
        return False
    return raw.strip() in _PLAN_MODE_ALLOWED_SUBAGENT_TYPES


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
