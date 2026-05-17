"""Built-in ``exit_plan_mode`` tool.

Renders the model's proposed plan to the user via the configured
:class:`aether.cli.approval_prompter.Prompter` and switches the session
back to *agent mode* iff the user approves.  When the user rejects (or
no interactive prompter is available), the session stays in plan mode
and the tool returns a structured refusal so the model can revise.

Prompter resolution order:
1. Optional constructor injection (``ExitPlanModeTool(prompter=...)``).
2. ``context.metadata['_approval_prompter']`` (set by
   :meth:`AgentEngine._prepare_turn_entry` from
   ``EngineRequest.approval_prompter``).

When neither is available the tool degrades gracefully: it logs the
plan to the result and **does not** approve.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import logging

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.session.plan_artifact import get_plan_path, read_plan, write_plan
from aether.runtime.session.session_state import SessionMode, get_mode, set_mode
from aether.tools.base import ToolDescriptor, ToolExecutor


_logger = logging.getLogger(__name__)


class ExitPlanModeTool(ToolExecutor):
    """Present the plan, ask the user to approve, return to agent mode."""

    NAME = "exit_plan_mode"

    def __init__(self, prompter: Any | None = None) -> None:
        self._prompter = prompter
        self._descriptor = ToolDescriptor(
            name=self.NAME,
            description=(
                "Present the current session plan to the user for "
                "approval. Prefer writing the plan to the session plan "
                "file first; the optional plan argument is a compatibility "
                "path that replaces the artifact before approval. On "
                "approval the session returns to agent mode and write "
                "tools are re-enabled; on rejection the session stays in "
                "plan mode for revision."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": (
                            "Optional markdown plan. Prefer writing the "
                            "plan to the session plan file; if provided, "
                            "this content replaces the artifact before "
                            "approval."
                        ),
                    },
                },
                "required": [],
            },
            required=[],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        plan_arg = args.get("plan")
        if plan_arg is not None and not isinstance(plan_arg, str):
            return _error(call, "'plan', if provided, must be a string")

        config = context.metadata.get("_engine_config") if context.metadata else None
        if not bool(getattr(config, "plan_mode_enabled", True)):
            return _error(call, "plan mode is disabled by configuration")

        if not context.session_id:
            return _error(call, "session_id missing on TurnContext; cannot resolve mode")

        if get_mode(context.session_id) != SessionMode.PLAN.value:
            return _error(
                call,
                "exit_plan_mode called but the session is not in plan mode; "
                "use enter_plan_mode first",
            )

        plan_path_str = str(get_plan_path(context.session_id))
        plan: str | None = None
        if isinstance(plan_arg, str) and plan_arg.strip():
            plan = plan_arg
            try:
                plan_path = write_plan(context.session_id, plan)
            except Exception as exc:  # noqa: BLE001 - logged + non-fatal
                _logger.warning(
                    "exit_plan_mode: failed to persist plan for session %s: %s",
                    context.session_id,
                    exc,
                )
            else:
                plan_path_str = str(plan_path)
        elif plan_arg is not None:
            return _error(call, "'plan', if provided, must be a non-empty string")

        if plan is None:
            plan = read_plan(context.session_id)
            if plan is None or not plan.strip():
                return _error(
                    call,
                    "no plan artifact found. Write your plan to the session "
                    f"plan file first ({plan_path_str}), or pass a non-empty "
                    "'plan' argument.",
                    metadata={"plan_path": plan_path_str},
                )

        prompter = self._prompter or (
            context.metadata.get("_approval_prompter") if context.metadata else None
        )
        if prompter is None or not getattr(prompter, "confirm_plan", None):
            return _error(
                call,
                "no approval prompter is configured; plan cannot be "
                "approved interactively. Stay in plan mode or finalise "
                "the plan as a normal text response for the user to "
                "review out-of-band.",
                metadata={"plan_preview": plan[:240], "plan_path": plan_path_str},
            )

        try:
            approved, updated_plan = _confirm_plan(
                prompter,
                plan,
                context=context,
                plan_path=plan_path_str,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return _error(call, f"approval prompter failed: {exc}")
        if updated_plan is not None and updated_plan.strip():
            plan = updated_plan
            try:
                plan_path = write_plan(context.session_id, plan)
            except Exception as exc:  # noqa: BLE001 - logged + non-fatal
                _logger.warning(
                    "exit_plan_mode: failed to persist updated plan for session %s: %s",
                    context.session_id,
                    exc,
                )
            else:
                plan_path_str = str(plan_path)

        if approved:
            set_mode(context.session_id, SessionMode.AGENT)
            _persist_session_mode(context.session_id, SessionMode.AGENT.value)
            approve_meta: dict[str, Any] = {
                "approved": True,
                "new_mode": SessionMode.AGENT.value,
                "plan_path": plan_path_str,
                "plan": plan,
            }
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=(
                    "Plan approved. Returning to agent mode.\n\n"
                    f"Your plan has been saved to: {plan_path_str}\n\n"
                    f"## Approved Plan:\n{plan}"
                ),
                is_error=False,
                metadata=approve_meta,
            )
        reject_meta: dict[str, Any] = {
            "approved": False,
            "new_mode": SessionMode.PLAN.value,
            "plan_path": plan_path_str,
        }
        _persist_session_mode(context.session_id, SessionMode.PLAN.value)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content="Plan rejected. Revise the plan file and call exit_plan_mode again.",
            is_error=False,
            metadata=reject_meta,
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


__all__ = ["ExitPlanModeTool"]


def _confirm_plan(
    prompter: Any,
    plan: str,
    *,
    context: TurnContext,
    plan_path: str,
) -> tuple[bool, str | None]:
    try:
        result = prompter.confirm_plan(plan, context=context, plan_path=plan_path)
    except TypeError:
        result = prompter.confirm_plan(plan, context=context)
    if isinstance(result, dict):
        updated_input = result.get("updated_input")
        updated_plan = None
        if isinstance(updated_input, dict) and isinstance(updated_input.get("plan"), str):
            updated_plan = updated_input["plan"]
        elif isinstance(result.get("plan"), str):
            updated_plan = result["plan"]
        return bool(result.get("confirmed")), updated_plan
    return bool(result), None


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
