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
from aether.runtime.session.plan_artifact import write_plan
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
                "Present a finalised plan (markdown string) to the user "
                "for approval. On approval the session returns to agent "
                "mode and write tools are re-enabled; on rejection the "
                "session stays in plan mode for revision."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": (
                            "Markdown-formatted plan describing the steps "
                            "the agent intends to take. Be specific about "
                            "files / commands / follow-ups."
                        ),
                    },
                },
                "required": ["plan"],
            },
            required=["plan"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        plan = args.get("plan")
        if not isinstance(plan, str) or not plan.strip():
            return _error(call, "'plan' is required and must be a non-empty string")

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

        # PR 12.4: persist the plan BEFORE prompting so a reject still
        # leaves the artifact on disk for /plan to display and for the
        # model to iterate on.  Failure here is non-fatal: we log and
        # continue so a transient FS hiccup can't block plan approval.
        plan_path_str: str | None = None
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
                metadata={"plan_preview": plan[:240]},
            )

        try:
            approved = bool(prompter.confirm_plan(plan, context=context))
        except Exception as exc:  # pragma: no cover - defensive
            return _error(call, f"approval prompter failed: {exc}")

        if approved:
            set_mode(context.session_id, SessionMode.AGENT)
            approve_meta: dict[str, Any] = {
                "approved": True,
                "new_mode": SessionMode.AGENT.value,
            }
            if plan_path_str is not None:
                approve_meta["plan_path"] = plan_path_str
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=(
                    "Plan approved. Returning to agent mode — proceed with "
                    "the implementation. Stick to the plan you outlined "
                    "and use todo_write to track progress."
                ),
                is_error=False,
                metadata=approve_meta,
            )
        reject_meta: dict[str, Any] = {
            "approved": False,
            "new_mode": SessionMode.PLAN.value,
        }
        if plan_path_str is not None:
            reject_meta["plan_path"] = plan_path_str
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=(
                "User did not approve the plan. Session remains in plan "
                "mode. Revise the plan based on the feedback (consider "
                "ask_user_question for clarification) and call "
                "exit_plan_mode again with the updated version."
            ),
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
