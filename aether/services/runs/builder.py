"""Agent run dependency builders."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from aether.agents.core.agent import AgentEngine
from aether.agents.middlewares.pipeline import MiddlewarePipeline
from aether.agents.types import AgentTypeRegistry
from aether.cli.sessions import SessionRecord
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import EngineRequest
from aether.runtime.tasks import TaskStore
from aether.runtime.tools.skill_catalog import build_default_skill_catalog
from aether.services.runs.contracts import AgentRunOptions, AgentRunRequest
from aether.services.runs.handles import RunHandle
from aether.subagents import SubagentManager
from aether.tools.registry import ToolRegistry

ProviderFactory = Callable[[SessionRecord], ModelProvider]
ConfigFactory = Callable[[AgentRunOptions], EngineConfig]
ToolRegistryFactory = Callable[[], ToolRegistry | None]


class RunDependencyBuilder:
    """Build the runtime pieces needed for one agent run."""

    def __init__(
        self,
        *,
        provider_factory: ProviderFactory | None = None,
        config_factory: ConfigFactory | None = None,
        tool_registry_factory: ToolRegistryFactory | None = None,
    ) -> None:
        self._provider_factory = provider_factory
        self._config_factory = config_factory
        self._tool_registry_factory = tool_registry_factory

    def build_provider(self, record: SessionRecord) -> ModelProvider:
        if self._provider_factory is not None:
            return self._provider_factory(record)
        from aether.cli.providers import build_provider

        return build_provider(
            record.provider,
            model=record.model,
            api_key=os.getenv("AETHER_API_KEY"),
            base_url=record.base_url,
        )

    def build_engine_config(self, options: AgentRunOptions) -> EngineConfig:
        if self._config_factory is not None:
            config = self._config_factory(options)
        else:
            config = EngineConfig()
            if isinstance(options.max_iterations, int):
                config.max_iterations = options.max_iterations
            config.tool_permissions_enabled = True
            config.skill_listing_enabled = True
            config.agent_type_registry_enabled = True
        if options.disable_builtin_tools is True:
            config.use_builtin_tools = False
        return config

    def build_tool_registry(self) -> ToolRegistry | None:
        if self._tool_registry_factory is not None:
            return self._tool_registry_factory()
        return None

    def build_skill_catalog(self, config: EngineConfig):
        return build_default_skill_catalog(config)

    def build_agent_type_registry(self, config: EngineConfig) -> AgentTypeRegistry:
        return AgentTypeRegistry(
            search_paths=config.agent_type_search_paths
            or AgentEngine._default_agent_type_search_paths()
        )

    def build_task_store(self, config: EngineConfig) -> TaskStore | None:
        if not getattr(config, "task_store_enabled", True):
            return None
        task_store_path = getattr(config, "task_store_path", None)
        return TaskStore(task_store_path) if task_store_path else TaskStore()

    def build_subagent_manager(self, config: EngineConfig) -> SubagentManager | None:
        if not getattr(config, "allow_subagent_dispatch", True):
            return None
        return SubagentManager(
            max_concurrent_background=getattr(config, "max_concurrent_background", 8),
        )

    def build_engine_request(
        self,
        *,
        record: SessionRecord,
        request: AgentRunRequest,
        run_id: str,
        handle: RunHandle,
        stream_callback: Callable[[str], None],
        stream_silent_callback: Callable[[str], None],
        loop_state_callback: Callable[[Any], None],
    ) -> EngineRequest:
        options = request.options
        user_message_metadata: dict[str, Any] = {}
        if request.attachments:
            user_message_metadata["displayAttachments"] = list(request.attachments)
        return EngineRequest(
            session_id=record.session_id,
            user_message=request.user_message,
            user_message_metadata=user_message_metadata,
            system_message=(
                options.system_override
                if options.system_override is not None
                else record.system_prompt
            ),
            stream_callback=stream_callback,
            stream_silent_callback=stream_silent_callback,
            messages=list(record.messages),
            model_config=ModelCallConfig(
                temperature=options.temperature,
                max_tokens=options.max_tokens,
            ),
            metadata={"run_id": run_id, "_loop_state_callback": loop_state_callback},
            approval_prompter=request.approval_prompter,
            tool_permission_prompter=request.tool_permission_prompter,
            interrupt_signal=handle.interrupt_signal,
        )

    def build_engine(
        self,
        *,
        provider: ModelProvider,
        tool_registry: ToolRegistry | None,
        middleware_pipeline: MiddlewarePipeline,
        config: EngineConfig,
        hooks: Any,
        skill_catalog: Any,
        agent_type_registry: AgentTypeRegistry,
        subagent_manager: SubagentManager | None,
        task_store: TaskStore | None,
    ) -> AgentEngine:
        return AgentEngine(
            provider,
            tool_registry=tool_registry,
            middleware_pipeline=middleware_pipeline,
            config=config,
            hooks=hooks,
            skill_catalog=skill_catalog,
            agent_type_registry=agent_type_registry,
            subagent_manager=subagent_manager,
            task_store=task_store,
        )


__all__ = [
    "ConfigFactory",
    "ProviderFactory",
    "RunDependencyBuilder",
    "ToolRegistryFactory",
]
