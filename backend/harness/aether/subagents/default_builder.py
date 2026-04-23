"""Default subagent builder implementation."""

from __future__ import annotations

from uuid import uuid4

from aether.agents.core.agent import AgentEngine
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
        )
        return child
