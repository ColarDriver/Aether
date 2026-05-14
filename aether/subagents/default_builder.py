"""Default subagent builder implementation."""

from __future__ import annotations

from uuid import uuid4

from aether.agents.core.agent import AgentEngine
from aether.runtime.control.interrupt_signal import InterruptSignal
from aether.config.schema import EngineConfig
from aether.subagents.builder import SubagentBuilder
from aether.subagents.contracts import SubagentTask


class DefaultSubagentBuilder(SubagentBuilder):
    """Build child agents by inheriting parent dependencies and config."""

    def __init__(self, inherit_tools: bool = True, inherit_middlewares: bool = True) -> None:
        self.inherit_tools = inherit_tools
        self.inherit_middlewares = inherit_middlewares

    def build_child(
        self, parent: AgentEngine, task: SubagentTask, child_depth: int
    ) -> AgentEngine:
        provider = task.provider or parent.services.provider

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
        )

        child = AgentEngine(
            provider=provider,
            tool_registry=parent.services.tool_registry if self.inherit_tools else None,
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
        )
        parent_signal = None
        if getattr(task.request, "interrupt_signal", None) is not None:
            parent_signal = task.request.interrupt_signal
        elif getattr(parent, "_current_session_id", None):
            parent_signal = parent.services.interrupt_controller.signal_for(parent._current_session_id)
        if parent_signal is not None:
            async_mode = bool(task.metadata.get("run_in_background", False))
            task.request.interrupt_signal = InterruptSignal() if async_mode else InterruptSignal(parent=parent_signal)
        return child
