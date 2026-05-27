from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from aether.config.schema import EngineConfig
from aether.runtime.core.contracts import ToolCall, TurnContext
from aether.runtime.mcp.config import McpServerConfig
from aether.runtime.mcp.runtime import McpRuntime
from aether.tools.builtins import build_default_tool_registry


@dataclass
class _RawTool:
    name: str
    description: str
    inputSchema: dict[str, Any]


@dataclass
class _RawText:
    text: str
    type: str = "text"


@dataclass
class _RawCallResult:
    content: list[_RawText]
    structuredContent: dict[str, Any] | None = None
    isError: bool = False


@dataclass
class _RawResource:
    uri: str
    name: str
    mimeType: str
    description: str = ""


@dataclass
class _RawResourceRead:
    contents: list[_RawText]


class _FakeMcpClient:
    def __init__(self) -> None:
        self.called: list[tuple[str, str, dict[str, Any]]] = []

    async def list_tools(self, config: McpServerConfig) -> Sequence[Any]:
        return [
            _RawTool(
                name="read_file",
                description=f"Read through {config.name}",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        ]

    async def call_tool(
        self,
        config: McpServerConfig,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        self.called.append((config.name, tool_name, dict(arguments)))
        return _RawCallResult(
            content=[_RawText(text="contents for " + str(arguments.get("path")))],
            structuredContent={"path": arguments.get("path")},
        )

    async def list_resources(self, config: McpServerConfig) -> Sequence[Any]:
        return [
            _RawResource(
                uri="file:///README.md",
                name="README.md",
                mimeType="text/markdown",
                description="Project readme",
            )
        ]

    async def read_resource(self, config: McpServerConfig, uri: str) -> Any:
        return _RawResourceRead(contents=[_RawText(text="# README")])


def _runtime(client: _FakeMcpClient | None = None) -> McpRuntime:
    return McpRuntime(
        servers=[McpServerConfig(name="filesystem", command="fake-mcp")],
        client=client or _FakeMcpClient(),
    )


def test_discovers_mcp_tools_with_stable_aether_tool_names() -> None:
    runtime = _runtime()

    tools = runtime.discover_tools()
    status = runtime.status()

    assert [tool.name for tool in tools] == ["mcp__filesystem__read_file"]
    assert tools[0].server == "filesystem"
    assert tools[0].local_name == "read_file"
    assert tools[0].required == ("path",)
    assert status.enabled is True
    assert status.status == "available"
    assert status.servers[0].tools_count == 1


def test_calls_discovered_mcp_tool_and_preserves_metadata() -> None:
    client = _FakeMcpClient()
    runtime = _runtime(client)
    tool = runtime.discover_tools()[0]

    result = runtime.call_tool(tool, {"path": "README.md"})

    assert result.is_error is False
    assert result.content == "contents for README.md"
    assert result.metadata["mcp"] == {"server": "filesystem", "tool": "read_file"}
    assert result.metadata["structured_content"] == {"path": "README.md"}
    assert client.called == [("filesystem", "read_file", {"path": "README.md"})]


def test_lists_and_reads_mcp_resources() -> None:
    runtime = _runtime()

    resources = runtime.list_resources()
    read = runtime.read_resource(server="filesystem", uri="file:///README.md")

    assert [(item.server, item.uri, item.name, item.mime_type) for item in resources] == [
        ("filesystem", "file:///README.md", "README.md", "text/markdown")
    ]
    assert read.server == "filesystem"
    assert read.uri == "file:///README.md"
    assert read.contents[0].text == "# README"


def test_default_tool_registry_registers_and_dispatches_mcp_tools() -> None:
    client = _FakeMcpClient()
    registry = build_default_tool_registry(
        config=EngineConfig(mcp_enabled=True, mcp_servers={"filesystem": {"command": "fake-mcp"}}),
        mcp_runtime=_runtime(client),
    )

    result = registry.dispatch(
        ToolCall(
            id="call-1",
            name="mcp__filesystem__read_file",
            arguments={"path": "README.md"},
        ),
        TurnContext(session_id="session-1", iteration=0),
    )

    assert result.tool_call_id == "call-1"
    assert result.name == "mcp__filesystem__read_file"
    assert result.content == "contents for README.md"
    assert result.metadata["mcp_server"] == "filesystem"
    assert result.metadata["mcp_tool"] == "read_file"
    assert client.called == [("filesystem", "read_file", {"path": "README.md"})]
