"""Default subagent builder implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from aether.runtime.control.interrupt_signal import InterruptSignal
from aether.config.schema import EngineConfig
from aether.subagents.builder import SubagentBuilder
from aether.subagents.contracts import SubagentTask
from aether.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from aether.agents.core.agent import AgentEngine


class DefaultSubagentBuilder(SubagentBuilder):
    """Build child agents by inheriting parent dependencies and config."""

    def __init__(self, inherit_tools: bool = True, inherit_middlewares: bool = True) -> None:
        self.inherit_tools = inherit_tools
        self.inherit_middlewares = inherit_middlewares

    def build_child(
        self, parent: AgentEngine, task: SubagentTask, child_depth: int
    ) -> AgentEngine:
        from aether.agents.core.agent import AgentEngine

        provider = task.provider or parent.services.provider
        agent_type_def = task.metadata.get("_agent_type_def")
        tool_registry = parent.services.tool_registry if self.inherit_tools else None
        if tool_registry is not None:
            allowed_tools = getattr(agent_type_def, "tools", None)
            disallowed_tools = tuple(getattr(agent_type_def, "disallowed_tools", ()) or ())
            if allowed_tools is not None or disallowed_tools:
                tool_registry = _filter_tool_registry(
                    tool_registry,
                    allowed_tools=allowed_tools,
                    disallowed_tools=disallowed_tools,
                )
        system_prompt = getattr(agent_type_def, "system_prompt", None)
        if isinstance(system_prompt, str) and system_prompt.strip():
            existing = task.request.system_message
            task.request.system_message = (
                f"{system_prompt.strip()}\n\n{existing.strip()}"
                if isinstance(existing, str) and existing.strip()
                else system_prompt.strip()
            )

        child_config = EngineConfig(
            max_iterations=task.max_iterations
            if task.max_iterations is not None
            else parent.config.max_iterations,
            fail_on_tool_error=parent.config.fail_on_tool_error,
            raise_on_middleware_error=parent.config.raise_on_middleware_error,
            fail_on_unknown_tool=parent.config.fail_on_unknown_tool,
            enable_todo_hydration=parent.config.enable_todo_hydration,
            memory_nudge_interval=parent.config.memory_nudge_interval,
            skill_nudge_interval=parent.config.skill_nudge_interval,
            skill_listing_enabled=parent.config.skill_listing_enabled,
            skill_listing_token_budget=parent.config.skill_listing_token_budget,
            verification_directive_enabled=parent.config.verification_directive_enabled,
            faithful_reporting_enabled=parent.config.faithful_reporting_enabled,
            verifier_gate_enabled=parent.config.verifier_gate_enabled,
            verifier_gate_file_threshold=parent.config.verifier_gate_file_threshold,
            memory_enabled=parent.config.memory_enabled,
            memory_mode=parent.config.memory_mode,
            memory_token_budget_pct=parent.config.memory_token_budget_pct,
            memory_token_budget_max=parent.config.memory_token_budget_max,
            memory_block_token_max=parent.config.memory_block_token_max,
            memory_retrieval_timeout_ms=parent.config.memory_retrieval_timeout_ms,
            memory_project_store_enabled=parent.config.memory_project_store_enabled,
            memory_user_profile_enabled=parent.config.memory_user_profile_enabled,
            memory_auto_write_enabled=parent.config.memory_auto_write_enabled,
            memory_llm_rerank_enabled=parent.config.memory_llm_rerank_enabled,
            memory_debug_log_content=parent.config.memory_debug_log_content,
            agent_type_search_paths=parent.config.agent_type_search_paths,
            agent_type_registry_enabled=parent.config.agent_type_registry_enabled,
        )

        child = AgentEngine(
            provider=provider,
            tool_registry=tool_registry,
            middleware_pipeline=parent.services.middleware_pipeline if self.inherit_middlewares else None,
            config=child_config,
            interrupt_controller=parent.services.interrupt_controller,
            logger=parent.services.logger,
            delegate_depth=child_depth,
            subagent_id=f"subagent-{uuid4().hex[:8]}",
            parent_subagent_id=parent.subagent_id,
            subagent_manager=parent.subagent_manager,
            session_store=parent._session_store,
            hooks=parent._hooks,
            memory_provider=parent.services.memory_provider,
            skill_catalog=parent._skill_catalog,
            agent_type_registry=parent._agent_type_registry,
            diagnostic_tracker=getattr(parent, "_diagnostic_tracker", None),
        )
        parent_signal = None
        if getattr(task.request, "interrupt_signal", None) is not None:
            parent_signal = task.request.interrupt_signal
        else:
            parent_session_id = getattr(parent, "_current_session_id", None)
            if parent_session_id is not None:
                parent_signal = parent.services.interrupt_controller.signal_for(parent_session_id)
        if parent_signal is not None:
            async_mode = bool(task.metadata.get("run_in_background", False))
            task.request.interrupt_signal = InterruptSignal() if async_mode else InterruptSignal(parent=parent_signal)
        return child


def _filter_tool_registry(
    registry: ToolRegistry,
    *,
    allowed_tools: tuple[str, ...] | None,
    disallowed_tools: tuple[str, ...],
) -> ToolRegistry:
    allowed = set(allowed_tools) if allowed_tools is not None else set(registry.list_names())
    denied = set(disallowed_tools)
    filtered = ToolRegistry()
    for name in registry.list_names():
        if name not in allowed or name in denied:
            continue
        filtered.register(registry.get(name))
    return filtered
