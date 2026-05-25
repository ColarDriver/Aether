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

_ATTACHMENT_CONTEXT_TOTAL_CHAR_LIMIT = 120_000
_ATTACHMENT_CONTEXT_ITEM_CHAR_LIMIT = 40_000


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
        attachments = list(request.attachments)
        attachment_context = _render_attachment_context(attachments)
        if attachments:
            user_message_metadata["displayAttachments"] = [
                _display_attachment(attachment) for attachment in attachments
            ]
        metadata: dict[str, Any] = {"run_id": run_id, "_loop_state_callback": loop_state_callback}
        if attachment_context:
            metadata["_llm_attachment_context"] = attachment_context
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
            metadata=metadata,
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


def _render_attachment_context(attachments: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    remaining = _ATTACHMENT_CONTEXT_TOTAL_CHAR_LIMIT
    for index, attachment in enumerate(attachments, start=1):
        if remaining <= 0:
            break
        item = _render_attachment_item(index, attachment, remaining)
        if not item:
            continue
        rendered.append(item)
        remaining -= len(item)
    if not rendered:
        return ""
    return (
        "<attached_context>\n"
        "The user attached the following files or workspace context for this turn. "
        "Treat this as user-provided context when answering.\n\n"
        + "\n\n".join(rendered)
        + "\n</attached_context>"
    )


def _render_attachment_item(index: int, attachment: dict[str, Any], remaining: int) -> str:
    label = _attachment_label(attachment)
    attachment_type = str(attachment.get("type") or "file")
    language = _attachment_language(attachment)
    content = _attachment_content(attachment)
    metadata = [f"## {index}. {label}", f"type: {attachment_type}"]
    for key, label_name in (("path", "path"), ("mimeType", "mime_type"), ("url", "url"), ("note", "note")):
        value = attachment.get(key)
        if isinstance(value, str) and value.strip():
            metadata.append(f"{label_name}: {value.strip()}")
    error = attachment.get("_llm_error")
    if isinstance(error, str) and error.strip():
        metadata.append("content_error: " + error.strip())
    if bool(attachment.get("_llm_binary")) or bool(attachment.get("isDirectory")):
        metadata.append("content: omitted")
    if bool(attachment.get("_llm_truncated")):
        metadata.append("content_truncated: true")
    header = "\n".join(metadata)
    if not content:
        if attachment_type == "image":
            return _truncate_item(header + "\ncontent: image data is attached for UI display but is not text-rendered here.", remaining)
        if error:
            return _truncate_item(header, remaining)
        return ""
    limited_content = content[:_ATTACHMENT_CONTEXT_ITEM_CHAR_LIMIT]
    suffix = ""
    if len(content) > len(limited_content):
        suffix = "\n...[attachment content truncated]"
    body = header + "\n\n```" + language + "\n" + limited_content.rstrip() + suffix + "\n```"
    return _truncate_item(body, remaining)


def _attachment_content(attachment: dict[str, Any]) -> str:
    for key in ("_llm_content", "data", "quote"):
        value = attachment.get(key)
        if isinstance(value, str) and value.strip():
            if key == "data" and value.startswith("data:"):
                return ""
            return value
    return ""


def _attachment_label(attachment: dict[str, Any]) -> str:
    for key in ("name", "path", "url"):
        value = attachment.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(attachment.get("type") or "attachment")


def _attachment_language(attachment: dict[str, Any]) -> str:
    value = attachment.get("_llm_language") or attachment.get("language")
    if isinstance(value, str) and value.strip():
        return value.strip().replace("`", "") or "text"
    mime_type = attachment.get("mimeType")
    if isinstance(mime_type, str):
        if "json" in mime_type:
            return "json"
        if "markdown" in mime_type:
            return "markdown"
        if mime_type.startswith("text/"):
            return "text"
    name = attachment.get("name") or attachment.get("path")
    if isinstance(name, str):
        suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        return {
            "md": "markdown",
            "markdown": "markdown",
            "py": "python",
            "ts": "typescript",
            "tsx": "tsx",
            "js": "javascript",
            "jsx": "jsx",
            "json": "json",
            "yaml": "yaml",
            "yml": "yaml",
            "toml": "toml",
            "sh": "shell",
        }.get(suffix, "text")
    return "text"


def _truncate_item(value: str, remaining: int) -> str:
    if len(value) <= remaining:
        return value
    if remaining <= 40:
        return ""
    return value[: remaining - 34].rstrip() + "\n...[attached context limit reached]"


def _display_attachment(attachment: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in attachment.items() if not str(key).startswith("_llm_")}


__all__ = [
    "ConfigFactory",
    "ProviderFactory",
    "RunDependencyBuilder",
    "ToolRegistryFactory",
]
