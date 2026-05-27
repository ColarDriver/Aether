from __future__ import annotations

from dataclasses import asdict
import json

import pytest

from aether.runtime.mcp.config import McpServerConfig
from aether.runtime.mcp.runtime import McpRuntime
from aether.services.tools import ToolService
from aether.services.tools.contracts import McpResourceContent, McpResourceReadResult, McpResourceSummary
from aether.tools.base import ToolDescriptor


class _Registry:
    def __init__(self, descriptors: list[ToolDescriptor]) -> None:
        self._descriptors = descriptors

    def list_descriptors(self) -> list[ToolDescriptor]:
        return self._descriptors


def test_tool_service_lists_builtin_descriptors() -> None:
    catalog = ToolService().list_tools()
    names = {tool.name for tool in catalog.tools}

    assert "read_file" in names
    assert "shell" in names
    assert "web_search" in names
    assert catalog.tools == sorted(catalog.tools, key=lambda item: item.name)


def test_tool_grouping_is_deterministic() -> None:
    service = ToolService(
        registry_factory=lambda: _Registry(
            [
                ToolDescriptor(name="shell"),
                ToolDescriptor(name="read_file"),
                ToolDescriptor(name="memory_read"),
                ToolDescriptor(name="custom_tool"),
            ]
        )
    )

    groups = service.list_groups()

    assert [(group.name, [tool.name for tool in group.tools]) for group in groups] == [
        ("filesystem", ["read_file"]),
        ("shell", ["shell"]),
        ("memory", ["memory_read"]),
        ("other", ["custom_tool"]),
    ]


def test_tool_summary_is_display_safe_and_lookup_works() -> None:
    service = ToolService(
        registry_factory=lambda: _Registry(
            [
                ToolDescriptor(
                    name="read_file",
                    description="Read a file",
                    parameters={"type": "object"},
                    required=["path"],
                )
            ]
        )
    )

    summary = service.get_tool("read_file")

    assert summary is not None
    assert asdict(summary) == {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object"},
        "required": ["path"],
        "enabled": True,
    }
    assert "executor" not in repr(summary).lower()
    assert service.get_tool("missing") is None


def test_mcp_status_groups_imported_mcp_tools_by_server() -> None:
    service = ToolService(
        registry_factory=lambda: _Registry(
            [
                ToolDescriptor(name="read_file"),
                ToolDescriptor(name="mcp__filesystem__read_file", description="MCP read"),
                ToolDescriptor(name="mcp__browser__screenshot", description="MCP screenshot"),
                ToolDescriptor(name="mcp__filesystem__list_dir", description="MCP list"),
            ]
        )
    )

    status = service.mcp_status()

    assert status.enabled is True
    assert status.status == "available"
    assert [(server.name, server.tools_count) for server in status.servers] == [
        ("browser", 1),
        ("filesystem", 2),
    ]
    assert [(tool.server, tool.local_name) for tool in status.imported_tools] == [
        ("browser", "screenshot"),
        ("filesystem", "list_dir"),
        ("filesystem", "read_file"),
    ]


def test_mcp_resources_report_unavailable_without_resource_provider() -> None:
    service = ToolService(
        registry_factory=lambda: _Registry([ToolDescriptor(name="mcp__filesystem__read_file")])
    )

    resources = service.mcp_resources()

    assert resources.enabled is True
    assert resources.status == "not_available"
    assert resources.resources == []
    assert "does not expose MCP resource browsing" in resources.message


def test_mcp_resources_filters_provider_resources_by_server() -> None:
    service = ToolService(
        registry_factory=lambda: _Registry([ToolDescriptor(name="mcp__filesystem__read_file")]),
        mcp_resource_provider=lambda server: [
            McpResourceSummary(server="filesystem", uri="file:///README.md", name="README.md"),
            McpResourceSummary(server="browser", uri="browser://page", name="Page"),
        ],
    )

    resources = service.mcp_resources(server="filesystem")

    assert resources.status == "available"
    assert [(item.server, item.uri) for item in resources.resources] == [("filesystem", "file:///README.md")]


def test_mcp_resource_read_reports_unavailable_without_reader() -> None:
    service = ToolService(
        registry_factory=lambda: _Registry([ToolDescriptor(name="mcp__filesystem__read_file")])
    )

    result = service.read_mcp_resource(server="filesystem", uri="file:///README.md")

    assert result.enabled is True
    assert result.status == "not_available"
    assert result.server == "filesystem"
    assert result.uri == "file:///README.md"
    assert result.contents == []


def test_mcp_resource_read_uses_runtime_reader() -> None:
    service = ToolService(
        registry_factory=lambda: _Registry([ToolDescriptor(name="mcp__filesystem__read_file")]),
        mcp_resource_reader=lambda server, uri: McpResourceReadResult(
            enabled=True,
            status="available",
            message="Read resource.",
            server=server,
            uri=uri,
            name="README.md",
            mime_type="text/markdown",
            contents=[McpResourceContent(type="text", text="# README", mime_type="text/markdown", uri=uri)],
        ),
    )

    result = service.read_mcp_resource(server="filesystem", uri="file:///README.md")

    assert result.status == "available"
    assert result.name == "README.md"
    assert result.contents[0].text == "# README"


def test_mcp_status_reports_not_configured_without_mcp_tools() -> None:
    service = ToolService(registry_factory=lambda: _Registry([ToolDescriptor(name="read_file")]))

    status = service.mcp_status()

    assert status.enabled is False
    assert status.status == "not_configured"
    assert status.servers == []
    assert status.imported_tools == []


def test_mcp_config_upsert_lists_and_deletes_managed_servers(tmp_path) -> None:
    service = ToolService(environ={"AETHER_HOME": str(tmp_path)})

    empty = service.mcp_config()
    assert empty.exists is False
    assert empty.servers == []

    saved = service.upsert_mcp_server(
        name="local fs",
        command="node",
        args=["server.js", "--root", "${WORKSPACE_ROOT}"],
        env={"TOKEN": "${MCP_TOKEN}"},
        timeout=5,
        connect_timeout=2,
    )

    assert saved.ok is True
    assert saved.server is not None
    assert saved.server.name == "local_fs"
    assert saved.server.command == "node"
    assert saved.server.args == ["server.js", "--root", "${WORKSPACE_ROOT}"]
    assert saved.server.env_keys == ["TOKEN"]

    config_path = tmp_path / "mcp_servers.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["mcp_servers"]["local_fs"]["command"] == "node"
    assert payload["mcp_servers"]["local_fs"]["env"] == {"TOKEN": "${MCP_TOKEN}"}

    listed = service.mcp_config()
    assert listed.exists is True
    assert [(server.name, server.transport, server.env_keys) for server in listed.servers] == [
        ("local_fs", "stdio", ["TOKEN"])
    ]

    deleted = service.delete_mcp_server(name="local fs")
    assert deleted.ok is True
    assert service.mcp_config().servers == []


def test_mcp_config_upsert_validates_transport_and_targets(tmp_path) -> None:
    service = ToolService(environ={"AETHER_HOME": str(tmp_path)})

    with pytest.raises(Exception, match="either command or url"):
        service.upsert_mcp_server(name="bad", command="node", url="https://example.test/mcp")

    with pytest.raises(Exception, match="Command-based MCP servers"):
        service.upsert_mcp_server(name="bad", command="node", transport="http")

    with pytest.raises(Exception, match="http://"):
        service.upsert_mcp_server(name="bad", url="ftp://example.test/mcp")


class _RuntimeClient:
    async def list_tools(self, config):
        return [
            {
                "name": "read_file",
                "description": "Read file",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]

    async def call_tool(self, config, tool_name, arguments):
        return {"content": [{"type": "text", "text": "ok"}]}

    async def list_resources(self, config):
        return [
            {
                "uri": "file:///README.md",
                "name": "README.md",
                "mimeType": "text/markdown",
            }
        ]

    async def read_resource(self, config, uri):
        return {
            "contents": [
                {
                    "type": "text",
                    "text": "# README",
                    "mimeType": "text/markdown",
                    "uri": uri,
                }
            ]
        }


def _mcp_runtime() -> McpRuntime:
    return McpRuntime(
        servers=[McpServerConfig(name="filesystem", command="fake-mcp")],
        client=_RuntimeClient(),
    )


def test_mcp_status_prefers_configured_runtime_over_inferred_registry_names() -> None:
    service = ToolService(
        registry_factory=lambda: _Registry([ToolDescriptor(name="read_file")]),
        mcp_runtime=_mcp_runtime(),
    )

    status = service.mcp_status()

    assert status.enabled is True
    assert status.status == "available"
    assert [(server.name, server.tools_count) for server in status.servers] == [("filesystem", 1)]
    assert [(tool.name, tool.server, tool.local_name) for tool in status.imported_tools] == [
        ("mcp__filesystem__read_file", "filesystem", "read_file")
    ]


def test_mcp_resources_and_reads_use_configured_runtime() -> None:
    service = ToolService(
        registry_factory=lambda: _Registry([ToolDescriptor(name="read_file")]),
        mcp_runtime=_mcp_runtime(),
    )

    resources = service.mcp_resources(server="filesystem")
    read = service.read_mcp_resource(server="filesystem", uri="file:///README.md")

    assert resources.status == "available"
    assert [(item.server, item.uri, item.name) for item in resources.resources] == [
        ("filesystem", "file:///README.md", "README.md")
    ]
    assert read.status == "available"
    assert read.server == "filesystem"
    assert read.contents[0].text == "# README"


class _FakeToolResult:
    def __init__(self) -> None:
        self.content = "# Web search: docs\n\nFound 1 results:\n"
        self.is_error = False
        self.metadata = {"result_count": 1}


class _FakeWebSearchTool:
    def execute(self, call, context):
        assert call.name == "web_search"
        assert call.arguments["query"] == "docs"
        assert context.metadata["_engine_config"].web_search_provider == "brave"
        assert context.metadata["_engine_config"].web_search_api_key == "test-key"
        return _FakeToolResult()


def test_web_search_status_reports_missing_invalid_and_ready_states() -> None:
    missing = ToolService(environ={}).web_search_status()
    assert missing.enabled is False
    assert missing.status == "missing_credential"
    assert missing.provider == "brave"

    invalid = ToolService(environ={"WEB_SEARCH_PROVIDER": "serpapi", "WEB_SEARCH_API_KEY": "key"}).web_search_status()
    assert invalid.enabled is False
    assert invalid.status == "invalid_provider"

    ready = ToolService(environ={"WEB_SEARCH_PROVIDER": "brave", "WEB_SEARCH_API_KEY": "key"}).web_search_status()
    assert ready.enabled is True
    assert ready.status == "ready"
    assert ready.supported_providers == ["bocha", "brave", "tavily"]


def test_web_search_test_uses_configured_provider_without_exposing_secret() -> None:
    service = ToolService(
        environ={"WEB_SEARCH_PROVIDER": "brave", "WEB_SEARCH_API_KEY": "test-key"},
        web_search_tool_factory=lambda: _FakeWebSearchTool(),
    )

    result = service.test_web_search(query="docs", max_results=1)

    assert result.ok is True
    assert result.provider == "brave"
    assert result.result_count == 1
    assert "test-key" not in result.content_preview


def test_web_search_test_short_circuits_when_not_configured() -> None:
    result = ToolService(environ={}).test_web_search(query="docs")

    assert result.ok is False
    assert result.error == "missing_credential"
