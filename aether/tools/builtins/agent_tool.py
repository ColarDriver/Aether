"""Built-in ``task`` (subagent dispatch) tool.

Synchronously dispatches a subagent task via the parent agent's
:class:`SubagentManager`. The implementation is intentionally
**synchronous** — the caller blocks until the child returns its
summary. Async fan-out (``TaskOutputTool`` polling) is left for a
future implementation; the current :class:`TaskOutputTool` returns a
clear "not supported in sync mode" message.

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
from typing import TYPE_CHECKING, Any, Dict, Optional

from aether.config.schema import ModelCallConfig
from aether.runtime.core.contracts import EngineRequest, ToolCall, ToolResult, TurnContext
from aether.subagents.contracts import SubagentResult, SubagentStatus, SubagentTask
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool

if TYPE_CHECKING:
    from aether.agents.types import AgentTypeRegistry

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
        agent_type_registry: "AgentTypeRegistry | None" = None,
    ) -> None:
        self._parent_agent = parent_agent
        self._subagent_manager = subagent_manager
        self._agent_type_registry = agent_type_registry
        self._descriptor: ToolDescriptor | None = None

    @property
    def descriptor(self) -> ToolDescriptor:
        if self._descriptor is None:
            self._descriptor = self._build_descriptor()
        return self._descriptor

    def _build_descriptor(self) -> ToolDescriptor:
        registry = self._agent_type_registry
        if registry is not None:
            types = registry.list_all()
            type_names = sorted({t.agent_type for t in types})
            type_desc_lines = "\n".join(
                f"  - {t.agent_type}: {t.description}"
                for t in sorted(types, key=lambda x: x.agent_type)
            )
            subagent_type_field: dict[str, Any] = {
                "type": "string",
                "default": "general-purpose",
                "enum": type_names,
                "description": (
                    "Which subagent persona to use. Available:\n" + type_desc_lines
                ),
            }
        else:
            subagent_type_field = {
                "type": "string",
                "default": "general-purpose",
                "description": (
                    "Which subagent persona to use. Defaults to "
                    "'general-purpose'."
                ),
            }

        return ToolDescriptor(
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
                    "subagent_type": subagent_type_field,
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
                    "model": {
                        "type": "string",
                        "enum": ["sonnet", "opus", "haiku", "inherit"],
                        "default": "inherit",
                        "description": (
                            "Optional model override for this subagent. "
                            "Takes precedence over the agent type "
                            "definition's model. If omitted (or "
                            "'inherit'), uses the type definition's model "
                            "or inherits from the parent."
                        ),
                    },
                    "run_in_background": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Set to true to launch the subagent on a "
                            "background thread.  The tool returns "
                            "immediately with a task_id; poll progress "
                            "via `task_output(task_id=...)`, deliver "
                            "additional instructions via "
                            "`send_message(to=task_id, ...)`, and cancel "
                            "via `task_stop(task_id=...)`.  Lifecycle "
                            "state lands at "
                            "~/.aether/tasks/{task_id}/."
                        ),
                    },
                },
                "required": ["prompt"],
            },
            required=["prompt"],
        )

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        prompt = args.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return _error(call, "'prompt' is required and must be a non-empty string")
        subagent_type = str(args.get("subagent_type") or "general-purpose").strip() or "general-purpose"
        expected_output = args.get("expected_output")
        if expected_output is not None and not isinstance(expected_output, str):
            return _error(call, "'expected_output' must be a string when provided")

        model_arg = args.get("model")
        if model_arg is not None and not isinstance(model_arg, str):
            return _error(call, "'model' must be a string when provided")
        model_override: str | None = None
        if isinstance(model_arg, str):
            normalized_model = model_arg.strip()
            if normalized_model and normalized_model.lower() != "inherit":
                model_override = normalized_model

        run_in_background_arg = args.get("run_in_background", False)
        if run_in_background_arg is not None and not isinstance(run_in_background_arg, bool):
            return _error(call, "'run_in_background' must be a boolean when provided")
        run_in_background = bool(run_in_background_arg)

        registry = self._agent_type_registry or (
            context.metadata.get("_agent_type_registry") if context.metadata else None
        )
        definition = None
        if registry is not None:
            definition = registry.get(subagent_type)
            if definition is None:
                available = ", ".join(sorted(t.agent_type for t in registry.list_all()))
                return _error(
                    call,
                    f"unknown subagent_type: {subagent_type!r}. Available: {available}",
                )

        # A type definition can force async even when the caller didn't
        # ask for it (e.g. long-running workers that only make sense in
        # the background).
        if definition is not None and getattr(definition, "background", False):
            run_in_background = True

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
                # PR 10.7: surface our own task_id into the child's
                # ``context.task_id`` so the engine's pending-message
                # drain (``_drain_pending_messages``) can read this
                # task's queue in TaskStore.
                "task_id": task_id,
            },
            interrupt_signal=context.interrupt_signal,
        )
        task_metadata: Dict[str, Any] = {
            "subagent_type": subagent_type,
            "expected_output": expected_output,
            "run_in_background": run_in_background,
            "_agent_type_def": definition,
        }
        if model_override is not None:
            task_metadata["model_override"] = model_override
        task = SubagentTask(
            task_id=task_id,
            goal=goal,
            request=request,
            metadata=task_metadata,
        )

        if run_in_background:
            if not bool(getattr(config, "subagent_async_enabled", True)):
                return _error(
                    call,
                    "async subagent dispatch is disabled by configuration",
                    metadata={"task_id": task_id},
                )
            try:
                returned_id = manager.run_task_async(parent=parent, task=task)
            except RuntimeError as exc:
                return _error(
                    call,
                    f"async subagent dispatch failed: {exc}",
                    metadata={"task_id": task_id},
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("async subagent dispatch crashed")
                return _error(
                    call,
                    f"async subagent dispatch crashed: {exc}",
                    metadata={"task_id": task_id},
                )
            return self._format_async_launched(
                call=call,
                task_id=returned_id,
                subagent_type=subagent_type,
                manager=manager,
                parent=parent,
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
    def _format_async_launched(
        *,
        call: ToolCall,
        task_id: str,
        subagent_type: str,
        manager: Any,
        parent: Any,
    ) -> ToolResult:
        store = getattr(parent, "_task_store", None)
        output_path: str | None = None
        result_path: str | None = None
        if store is not None:
            try:
                root = store.root / task_id  # type: ignore[union-attr]
                output_path = str(root / "output.log")
                result_path = str(root / "result.json")
            except Exception:  # pragma: no cover - defensive
                output_path = None
                result_path = None

        body_lines = [
            "# Subagent launched in background",
            f"- task_id: {task_id}",
            f"- subagent_type: {subagent_type}",
        ]
        if output_path:
            body_lines.append(f"- output_path: {output_path}")
        body_lines.append("")
        body_lines.append(
            f"Use `task_output(task_id=\"{task_id}\")` to poll progress; "
            f"`send_message(to=\"{task_id}\", message=...)` to send "
            f"additional instructions; `task_stop(task_id=\"{task_id}\")` "
            f"to cancel."
        )
        body = "\n".join(body_lines) + "\n"
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=body,
            is_error=False,
            metadata={
                "status": "async_launched",
                "task_id": task_id,
                "subagent_type": subagent_type,
                "output_path": output_path,
                "result_path": result_path,
                "background": True,
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
