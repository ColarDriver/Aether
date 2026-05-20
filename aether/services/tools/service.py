"""Tool service implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from aether.services.tools.contracts import ToolCatalog, ToolGroup, ToolSummary
from aether.tools.base import ToolDescriptor


RegistryFactory = Callable[[], Any]

_GROUP_ORDER: tuple[str, ...] = (
    "filesystem",
    "shell",
    "web",
    "subagent",
    "interaction",
    "planning",
    "skills",
    "diagnostics",
    "memory",
    "other",
)

_TOOL_GROUPS: dict[str, str] = {
    "read_file": "filesystem",
    "write_file": "filesystem",
    "list_dir": "filesystem",
    "grep": "filesystem",
    "glob": "filesystem",
    "file_edit": "filesystem",
    "notebook_edit": "filesystem",
    "shell": "shell",
    "web_fetch": "web",
    "web_search": "web",
    "web_browser": "web",
    "task": "subagent",
    "task_output": "subagent",
    "task_stop": "subagent",
    "send_message": "subagent",
    "ask_user_question": "interaction",
    "enter_plan_mode": "planning",
    "exit_plan_mode": "planning",
    "todo_write": "planning",
    "skill": "skills",
    "lsp": "diagnostics",
    "memory_read": "memory",
    "memory_list": "memory",
    "memory_write": "memory",
    "memory_update": "memory",
    "memory_forget": "memory",
}


class ToolService:
    """Read-only view over the built-in tool registry."""

    def __init__(self, *, registry_factory: RegistryFactory | None = None) -> None:
        self._registry_factory = registry_factory

    def list_tools(self) -> ToolCatalog:
        tools = [_descriptor_to_summary(descriptor) for descriptor in self._descriptors()]
        tools.sort(key=lambda item: item.name)
        return ToolCatalog(tools=tools)

    def list_groups(self) -> list[ToolGroup]:
        grouped: dict[str, list[ToolSummary]] = {name: [] for name in _GROUP_ORDER}
        for tool in self.list_tools().tools:
            grouped[_TOOL_GROUPS.get(tool.name, "other")].append(tool)
        return [
            ToolGroup(name=name, tools=tools)
            for name in _GROUP_ORDER
            if (tools := grouped[name])
        ]

    def get_tool(self, name: str) -> ToolSummary | None:
        normalized = name.strip()
        if not normalized:
            return None
        for tool in self.list_tools().tools:
            if tool.name == normalized:
                return tool
        return None

    def _descriptors(self) -> list[ToolDescriptor]:
        registry = self._build_registry()
        descriptors = registry.list_descriptors()
        return list(descriptors)

    def _build_registry(self) -> Any:
        if self._registry_factory is not None:
            return self._registry_factory()
        # Keep this lazy so importing services does not eagerly construct
        # heavyweight browser/LSP resources during gateway boot.
        from aether.tools.builtins import build_default_tool_registry

        return build_default_tool_registry()


def _descriptor_to_summary(descriptor: ToolDescriptor) -> ToolSummary:
    payload = asdict(descriptor)
    return ToolSummary(
        name=str(payload.get("name") or ""),
        description=str(payload.get("description") or ""),
        parameters=dict(payload.get("parameters") or {}),
        required=list(payload.get("required") or []),
        enabled=True,
    )


__all__ = ["ToolService"]
