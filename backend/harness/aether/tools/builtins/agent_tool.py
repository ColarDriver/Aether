"""Built-in ``task`` (subagent dispatch) tool — Sprint 3.5 / PR 3.5.6.

Synchronously dispatches a subagent task via the parent agent's
:class:`SubagentManager`.  v1 is intentionally **synchronous** — the
caller blocks until the child returns its summary.  Async fan-out
(claude-code-style ``TaskOutputTool`` polling) is left for a future
sprint; today's :class:`TaskOutputTool` returns a clear "not supported
in sync mode" message.

Resolution order for the parent agent reference:

1. ``context.metadata['_parent_agent']`` (set by ``_prepare_turn_entry``)
2. Constructor injection (``AgentTool(parent_agent=...)``) — used by
   tests so they don't need to set up a real engine.

The same fallback applies to ``_subagent_manager``.

Subagents inherit the parent's tool registry / middleware pipeline by
default (see :class:`DefaultSubagentBuilder`); the new task gets a
fresh ``EngineRequest`` whose ``user_message`` is the model-supplied
``prompt``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from aether.config.schema import ModelCallConfig
from aether.runtime.contracts import EngineRequest, ToolCall, ToolResult, TurnContext
from aether.subagents.contracts import SubagentResult, SubagentStatus, SubagentTask
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool

logger = logging.getLogger(__name__)


class AgentTool(ToolExecutor):
    """Dispatch a subagent task and return its summary."""

    NAME = "task"
    MAX_RESULT_CHARS = 60_000

    def __init__(
        self,
        *,
        parent_agent: Any | None = None,
        subagent_manager: Any | None = None,
    ) -> None:
        self._parent_agent = parent_agent
        self._subagent_manager = subagent_manager
        self._descriptor = ToolDescriptor(
            name=self.NAME,
            description=(
                "Dispatch a subagent to handle an isolated, self-contained "
                "task. The subagent runs synchronously and returns its "
                "summary; its turn history does NOT pollute the parent "
                "context. Use this to fan-out I/O-heavy work (e.g. "
                "'inspect 30 files for pattern X') so the parent stays "
                "focused on planning. The prompt should be self-contained — "
                "the subagent does not see your conversation history."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "subagent_type": {
                        "type": "string",
                        "default": "general-purpose",
                        "description": (
                            "Which subagent persona to use. Defaults to "
                            "'general-purpose'."
                        ),
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Self-contained task description for the "
                            "subagent. Be specific about what you need."
                        ),
                    },
                    "expected_output": {
                        "type": "string",
                        "description": (
                            "Brief description of the desired output shape "
                            "(e.g. 'a list of file paths', 'JSON object "
                            "with fields foo/bar')."
                        ),
                    },
                },
                "required": ["prompt"],
            },
            required=["prompt"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        prompt = args.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return _error(call, "'prompt' is required and must be a non-empty string")
        subagent_type = str(args.get("subagent_type") or "general-purpose").strip() or "general-purpose"
        expected_output = args.get("expected_output")
        if expected_output is not None and not isinstance(expected_output, str):
            return _error(call, "'expected_output' must be a string when provided")

        config = context.metadata.get("_engine_config") if context.metadata else None
        if not bool(getattr(config, "allow_subagent_dispatch", True)):
            return _error(call, "subagent dispatch is disabled by configuration")

        parent = self._parent_agent or (
            context.metadata.get("_parent_agent") if context.metadata else None
        )
        if parent is None:
            return _error(
                call,
                "AgentTool not wired: _parent_agent missing from context",
            )

        manager = self._subagent_manager or (
            context.metadata.get("_subagent_manager") if context.metadata else None
        )
        if manager is None:
            manager = getattr(parent, "subagent_manager", None)
        if manager is None:
            return _error(
                call,
                "AgentTool not wired: SubagentManager is not configured on "
                "the parent agent",
            )

        task_id = f"task-{uuid.uuid4().hex[:10]}"
        goal = (expected_output or prompt)[:200]
        request = EngineRequest(
            session_id=f"{context.session_id}::sub::{task_id}",
            user_message=prompt,
            messages=[],
            model_config=ModelCallConfig(),
            metadata={
                "subagent_type": subagent_type,
                "parent_session_id": context.session_id,
                "parent_task_id": task_id,
            },
            interrupt_signal=context.interrupt_signal,
        )
        task = SubagentTask(
            task_id=task_id,
            goal=goal,
            request=request,
            metadata={
                "subagent_type": subagent_type,
                "expected_output": expected_output,
                "run_in_background": False,
            },
        )

        try:
            result = manager.run_task(parent=parent, task=task)
        except RuntimeError as exc:
            return _error(call, f"subagent dispatch failed: {exc}", metadata={"task_id": task_id})
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("subagent crashed")
            return _error(call, f"subagent crashed: {exc}", metadata={"task_id": task_id})

        body = self._format_result(result, expected_output=expected_output, task_id=task_id)
        content = maybe_spill_for_tool(
            body,
            call=call,
            context=context,
            max_chars=self.MAX_RESULT_CHARS,
            extension="md",
            full_lines=body.count("\n") + 1,
        )
        is_error = result.status != SubagentStatus.COMPLETED
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=content,
            is_error=is_error,
            metadata={
                "task_id": task_id,
                "subagent_type": subagent_type,
                "subagent_status": result.status.value
                if isinstance(result.status, SubagentStatus)
                else str(result.status),
                "duration_seconds": result.duration_seconds,
            },
        )

    @staticmethod
    def _format_result(
        result: SubagentResult,
        *,
        expected_output: Optional[str],
        task_id: str,
    ) -> str:
        status_value = (
            result.status.value if isinstance(result.status, SubagentStatus) else str(result.status)
        )
        lines = [
            "# Subagent task complete",
            f"- task_id: {task_id}",
            f"- subagent_id: {result.metadata.get('subagent_id', '?')}",
            f"- status: {status_value}",
            f"- duration: {result.duration_seconds:.1f}s",
        ]
        if expected_output:
            lines.append(f"- expected_output: {expected_output}")
        if result.error:
            lines.append("")
            lines.append("## Error")
            lines.append(result.error)
        if result.summary:
            lines.append("")
            lines.append("## Summary")
            lines.append(result.summary)
        return "\n".join(lines) + "\n"


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


__all__ = ["AgentTool"]
