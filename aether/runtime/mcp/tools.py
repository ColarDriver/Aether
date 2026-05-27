"""Tool registry bridge for MCP runtime tools."""

from __future__ import annotations

from typing import Any

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.mcp.runtime import DiscoveredMcpTool, McpRuntime
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class McpToolExecutor(ToolExecutor):
    """Expose one MCP server tool as a native Aether tool."""

    interrupt_behavior = "cancel"

    def __init__(self, *, runtime: McpRuntime, tool: DiscoveredMcpTool) -> None:
        self._runtime = runtime
        self._tool = tool
        self._descriptor = ToolDescriptor(
            name=tool.name,
            description=tool.description or f"MCP tool {tool.local_name} from server {tool.server}.",
            parameters=dict(tool.parameters or {"type": "object", "properties": {}}),
            required=list(tool.required),
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        result = self._runtime.call_tool(self._tool, call.arguments or {})
        metadata: dict[str, Any] = dict(result.metadata)
        metadata.setdefault("mcp_server", self._tool.server)
        metadata.setdefault("mcp_tool", self._tool.local_name)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=result.content,
            is_error=result.is_error,
            metadata=metadata,
        )


def register_mcp_tools(registry: ToolRegistry, runtime: McpRuntime) -> list[str]:
    """Discover and register configured MCP tools.

    Discovery failures are recorded in ``runtime.status()`` and do not raise,
    keeping Aether usable when an optional MCP server is down.
    """

    registered: list[str] = []
    for tool in runtime.discover_tools():
        if not tool.enabled:
            continue
        registry.register(McpToolExecutor(runtime=runtime, tool=tool))
        registered.append(tool.name)
    return registered


__all__ = ["McpToolExecutor", "register_mcp_tools"]
