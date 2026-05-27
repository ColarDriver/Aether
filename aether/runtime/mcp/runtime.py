"""Runtime client for external MCP servers.

This module intentionally keeps MCP as an optional integration. If the Python
``mcp`` SDK is missing or a server is misconfigured, Aether reports that status
through web/service surfaces and keeps the normal tool registry usable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Protocol
import threading

from aether.runtime.credentials.redaction import redact_mapping, redact_text
from aether.runtime.mcp.config import (
    McpServerConfig,
    load_mcp_server_configs,
    sanitize_mcp_name_component,
)


@dataclass(frozen=True, slots=True)
class DiscoveredMcpTool:
    name: str
    server: str
    local_name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    enabled: bool = True
    status: str = "available"
    error: str | None = None


@dataclass(frozen=True, slots=True)
class McpResourceRecord:
    server: str
    uri: str
    name: str
    mime_type: str | None = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class McpContentRecord:
    type: str
    text: str | None = None
    blob: str | None = None
    mime_type: str | None = None
    uri: str | None = None


@dataclass(frozen=True, slots=True)
class McpResourceReadRecord:
    server: str
    uri: str
    name: str | None = None
    mime_type: str | None = None
    contents: tuple[McpContentRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class McpToolCallRecord:
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class McpServerRuntimeStatus:
    name: str
    status: str
    message: str = ""
    tools_count: int = 0
    resources_count: int = 0
    credential_status: str = "unknown"


@dataclass(frozen=True, slots=True)
class McpRuntimeStatus:
    enabled: bool
    status: str
    message: str
    servers: tuple[McpServerRuntimeStatus, ...] = ()
    imported_tools: tuple[DiscoveredMcpTool, ...] = ()


class McpClientAdapter(Protocol):
    async def list_tools(self, config: McpServerConfig) -> Sequence[Any]: ...

    async def call_tool(
        self,
        config: McpServerConfig,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Any: ...

    async def list_resources(self, config: McpServerConfig) -> Sequence[Any]: ...

    async def read_resource(self, config: McpServerConfig, uri: str) -> Any: ...


class McpRuntime:
    """Discover MCP tools and invoke them as native Aether tool executors."""

    def __init__(
        self,
        *,
        servers: Sequence[McpServerConfig] | None = None,
        server_source: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
        aether_home: Path | None = None,
        client: McpClientAdapter | None = None,
    ) -> None:
        self._environ = environ
        self._servers = list(servers) if servers is not None else load_mcp_server_configs(
            server_source,
            environ=environ,
            aether_home=aether_home,
        )
        self._client: McpClientAdapter = client or _SdkMcpClient()
        self._lock = threading.RLock()
        self._discovered: dict[str, DiscoveredMcpTool] = {}
        self._server_status: dict[str, McpServerRuntimeStatus] = {}
        self._resources: dict[str, tuple[McpResourceRecord, ...]] = {}

    @property
    def configured_servers(self) -> tuple[McpServerConfig, ...]:
        return tuple(self._servers)

    def status(self) -> McpRuntimeStatus:
        if not self._servers:
            return McpRuntimeStatus(
                enabled=False,
                status="not_configured",
                message="No MCP servers are configured for this Aether runtime.",
            )
        tools = self.discover_tools()
        with self._lock:
            statuses = tuple(
                self._server_status.get(server.name)
                or McpServerRuntimeStatus(name=server.name, status="unknown")
                for server in self._servers
            )
        enabled = any(item.status == "available" for item in statuses)
        status = "available" if enabled else "error"
        if enabled and any(item.status != "available" for item in statuses):
            status = "partial"
        return McpRuntimeStatus(
            enabled=enabled,
            status=status,
            message=_status_message(statuses),
            servers=statuses,
            imported_tools=tuple(tools),
        )

    def discover_tools(self, *, force: bool = False) -> tuple[DiscoveredMcpTool, ...]:
        with self._lock:
            if self._discovered and not force:
                return tuple(sorted(self._discovered.values(), key=lambda item: item.name))

        discovered: dict[str, DiscoveredMcpTool] = {}
        statuses: dict[str, McpServerRuntimeStatus] = {}
        for server in self._servers:
            try:
                raw_tools = _run_async(
                    self._client.list_tools(server),
                    timeout=server.connect_timeout,
                )
                tools = [_tool_from_raw(server, raw_tool) for raw_tool in raw_tools]
                for tool in tools:
                    discovered[tool.name] = tool
                statuses[server.name] = McpServerRuntimeStatus(
                    name=server.name,
                    status="available",
                    message=f"{len(tools)} MCP tool(s) discovered.",
                    tools_count=len(tools),
                    credential_status=_credential_status(server),
                )
            except Exception as exc:  # noqa: BLE001 - status surface, not fatal
                statuses[server.name] = McpServerRuntimeStatus(
                    name=server.name,
                    status="error",
                    message=redact_text(str(exc) or exc.__class__.__name__),
                    credential_status=_credential_status(server),
                )

        with self._lock:
            self._discovered = discovered
            self._server_status = statuses
        return tuple(sorted(discovered.values(), key=lambda item: item.name))

    def call_tool(
        self,
        tool: DiscoveredMcpTool,
        arguments: Mapping[str, Any],
    ) -> McpToolCallRecord:
        server = self._server_config(tool.server)
        if server is None:
            return McpToolCallRecord(
                content=f"MCP server {tool.server!r} is not configured.",
                is_error=True,
                metadata={"mcp": {"server": tool.server, "tool": tool.local_name}},
            )
        try:
            raw = _run_async(
                self._client.call_tool(server, tool.local_name, dict(arguments)),
                timeout=server.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            return McpToolCallRecord(
                content=redact_text(str(exc) or exc.__class__.__name__),
                is_error=True,
                metadata={"mcp": {"server": tool.server, "tool": tool.local_name}},
            )
        return _tool_call_from_raw(server.name, tool.local_name, raw)

    def list_resources(self, *, server: str | None = None) -> tuple[McpResourceRecord, ...]:
        configs = [
            item for item in self._servers
            if server is None or item.name == server
        ]
        records: list[McpResourceRecord] = []
        for config in configs:
            try:
                raw_resources = _run_async(
                    self._client.list_resources(config),
                    timeout=config.connect_timeout,
                )
            except Exception:
                continue
            resources = [_resource_from_raw(config.name, item) for item in raw_resources]
            records.extend(resources)
            with self._lock:
                self._resources[config.name] = tuple(resources)
                current = self._server_status.get(config.name)
                if current is not None:
                    self._server_status[config.name] = McpServerRuntimeStatus(
                        name=current.name,
                        status=current.status,
                        message=current.message,
                        tools_count=current.tools_count,
                        resources_count=len(resources),
                        credential_status=current.credential_status,
                    )
        return tuple(records)

    def read_resource(self, *, server: str, uri: str) -> McpResourceReadRecord:
        config = self._server_config(server)
        if config is None:
            raise ValueError(f"MCP server {server!r} is not configured")
        raw = _run_async(
            self._client.read_resource(config, uri),
            timeout=config.timeout,
        )
        return _resource_read_from_raw(server, uri, raw)

    def _server_config(self, server: str) -> McpServerConfig | None:
        for item in self._servers:
            if item.name == server:
                return item
        return None


class _SdkMcpClient:
    """Thin adapter over the optional Python MCP SDK."""

    async def list_tools(self, config: McpServerConfig) -> Sequence[Any]:
        async with _sdk_session(config) as session:
            result = await session.list_tools()
        return getattr(result, "tools", result) or []

    async def call_tool(
        self,
        config: McpServerConfig,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        async with _sdk_session(config) as session:
            return await session.call_tool(tool_name, dict(arguments))

    async def list_resources(self, config: McpServerConfig) -> Sequence[Any]:
        async with _sdk_session(config) as session:
            result = await session.list_resources()
        return getattr(result, "resources", result) or []

    async def read_resource(self, config: McpServerConfig, uri: str) -> Any:
        async with _sdk_session(config) as session:
            return await session.read_resource(uri)


@asynccontextmanager
async def _sdk_session(config: McpServerConfig):
    try:
        mcp_module = importlib.import_module("mcp")
        stdio_module = importlib.import_module("mcp.client.stdio")
        ClientSession = getattr(mcp_module, "ClientSession")
        StdioServerParameters = getattr(mcp_module, "StdioServerParameters")
        stdio_client = getattr(stdio_module, "stdio_client")
    except ImportError as exc:
        raise RuntimeError("MCP SDK is not installed. Install the 'mcp' Python package to enable MCP servers.") from exc

    if config.url:
        async with _remote_client(config) as streams:
            read_stream, write_stream = streams
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=config.connect_timeout)
                yield session
        return

    if not config.command:
        raise RuntimeError(f"MCP server {config.name!r} has no command configured")
    params = StdioServerParameters(
        command=config.command,
        args=list(config.args),
        env=_safe_stdio_env(config.env),
    )
    async with stdio_client(params, errlog=sys.stderr) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await asyncio.wait_for(session.initialize(), timeout=config.connect_timeout)
            yield session


@asynccontextmanager
async def _remote_client(config: McpServerConfig):
    if not config.url:
        raise RuntimeError("remote MCP client requires url")
    if config.transport == "sse":
        try:
            sse_module = importlib.import_module("mcp.client.sse")
            sse_client = getattr(sse_module, "sse_client")
        except ImportError as exc:
            raise RuntimeError("MCP SDK SSE client is not available") from exc
        async with sse_client(
            config.url,
            headers=dict(config.headers),
            timeout=config.connect_timeout,
            sse_read_timeout=config.timeout,
        ) as (read_stream, write_stream):
            yield read_stream, write_stream
        return

    try:
        http_module = importlib.import_module("mcp.client.streamable_http")
        streamablehttp_client = getattr(http_module, "streamablehttp_client")
    except ImportError as exc:
        raise RuntimeError("MCP SDK streamable HTTP client is not available") from exc
    async with streamablehttp_client(
        config.url,
        headers=dict(config.headers),
        timeout=config.connect_timeout,
        sse_read_timeout=config.timeout,
    ) as (read_stream, write_stream, _session_id_callback):
        yield read_stream, write_stream


_SAFE_ENV_KEYS = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR"}


def _safe_stdio_env(extra: Mapping[str, str]) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key in _SAFE_ENV_KEYS or key.startswith("XDG_")
    }
    env.update(dict(extra))
    return env


def _run_async(coro, *, timeout: float):
    async def _with_timeout():
        return await asyncio.wait_for(coro, timeout=timeout)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_with_timeout())

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(_with_timeout())).result(timeout=timeout + 1)


def _tool_from_raw(server: McpServerConfig, raw: Any) -> DiscoveredMcpTool:
    local_name = str(_attr_or_item(raw, "name") or "")
    safe_server = sanitize_mcp_name_component(server.name)
    safe_tool = sanitize_mcp_name_component(local_name)
    schema = _attr_or_item(raw, "inputSchema")
    if schema is None:
        schema = _attr_or_item(raw, "input_schema")
    params = _json_schema(schema)
    required_raw = params.get("required")
    required = required_raw if isinstance(required_raw, list) else []
    return DiscoveredMcpTool(
        name=f"mcp__{safe_server}__{safe_tool}",
        server=server.name,
        local_name=local_name,
        description=str(_attr_or_item(raw, "description") or ""),
        parameters=params,
        required=tuple(str(item) for item in required),
    )


def _tool_call_from_raw(server: str, tool_name: str, raw: Any) -> McpToolCallRecord:
    content_items = _attr_or_item(raw, "content") or []
    structured = _attr_or_item(raw, "structuredContent")
    if structured is None:
        structured = _attr_or_item(raw, "structured_content")
    text_parts: list[str] = []
    metadata: dict[str, Any] = {"mcp": {"server": server, "tool": tool_name}}
    if structured not in (None, {}):
        metadata["structured_content"] = redact_mapping(structured) if isinstance(structured, Mapping) else structured
    for item in content_items if isinstance(content_items, Sequence) and not isinstance(content_items, (str, bytes)) else [content_items]:
        text = _attr_or_item(item, "text")
        if isinstance(text, str) and text:
            text_parts.append(text)
            continue
        data = _attr_or_item(item, "data")
        mime_type = _attr_or_item(item, "mimeType") or _attr_or_item(item, "mime_type")
        if isinstance(data, str) and isinstance(mime_type, str):
            text_parts.append(f"[MCP content: {mime_type}, {len(data)} base64 chars]")
    if structured not in (None, {}) and not text_parts:
        text_parts.append(json.dumps(structured, ensure_ascii=False, indent=2, default=str))
    text = "\n".join(text_parts).strip()
    is_error = bool(_attr_or_item(raw, "isError") or _attr_or_item(raw, "is_error"))
    return McpToolCallRecord(
        content=redact_text(text or "(MCP tool returned no content)"),
        is_error=is_error,
        metadata=metadata,
    )


def _resource_from_raw(server: str, raw: Any) -> McpResourceRecord:
    uri = str(_attr_or_item(raw, "uri") or "")
    name = str(_attr_or_item(raw, "name") or uri)
    mime_type = _attr_or_item(raw, "mimeType") or _attr_or_item(raw, "mime_type")
    description = str(_attr_or_item(raw, "description") or "")
    return McpResourceRecord(
        server=server,
        uri=uri,
        name=name,
        mime_type=mime_type if isinstance(mime_type, str) else None,
        description=description,
    )


def _resource_read_from_raw(server: str, uri: str, raw: Any) -> McpResourceReadRecord:
    contents = _attr_or_item(raw, "contents") or _attr_or_item(raw, "content") or []
    records = tuple(
        _content_from_raw(item)
        for item in contents if item is not None
    ) if isinstance(contents, Sequence) and not isinstance(contents, (str, bytes)) else (_content_from_raw(contents),)
    return McpResourceReadRecord(server=server, uri=uri, contents=records)


def _content_from_raw(raw: Any) -> McpContentRecord:
    content_type = str(_attr_or_item(raw, "type") or "text")
    text = _attr_or_item(raw, "text")
    blob = _attr_or_item(raw, "blob")
    mime_type = _attr_or_item(raw, "mimeType") or _attr_or_item(raw, "mime_type")
    uri = _attr_or_item(raw, "uri")
    return McpContentRecord(
        type=content_type,
        text=redact_text(text) if isinstance(text, str) else None,
        blob=blob if isinstance(blob, str) else None,
        mime_type=mime_type if isinstance(mime_type, str) else None,
        uri=uri if isinstance(uri, str) else None,
    )


def _json_schema(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        schema = dict(value)
    elif hasattr(value, "model_dump"):
        schema = dict(value.model_dump())
    elif hasattr(value, "dict"):
        schema = dict(value.dict())
    else:
        schema = {}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


def _attr_or_item(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _credential_status(server: McpServerConfig) -> str:
    redacted = redact_mapping({"env": dict(server.env), "headers": dict(server.headers)})
    raw = {"env": dict(server.env), "headers": dict(server.headers)}
    return "configured" if redacted != raw else "none"


def _status_message(statuses: Sequence[McpServerRuntimeStatus]) -> str:
    if not statuses:
        return "No MCP servers are configured for this Aether runtime."
    available = [item for item in statuses if item.status == "available"]
    if len(available) == len(statuses):
        return f"{len(statuses)} MCP server(s) available."
    if available:
        return f"{len(available)}/{len(statuses)} MCP server(s) available."
    return "MCP servers are configured but none are currently available."


_DEFAULT_RUNTIME: McpRuntime | None = None
_DEFAULT_RUNTIME_LOCK = threading.Lock()


def get_default_mcp_runtime(
    *,
    server_source: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> McpRuntime:
    global _DEFAULT_RUNTIME
    with _DEFAULT_RUNTIME_LOCK:
        if _DEFAULT_RUNTIME is None or server_source is not None or environ is not None:
            _DEFAULT_RUNTIME = McpRuntime(server_source=server_source, environ=environ)
        return _DEFAULT_RUNTIME


def reset_default_mcp_runtime() -> None:
    global _DEFAULT_RUNTIME
    with _DEFAULT_RUNTIME_LOCK:
        _DEFAULT_RUNTIME = None


__all__ = [
    "DiscoveredMcpTool",
    "McpClientAdapter",
    "McpContentRecord",
    "McpResourceReadRecord",
    "McpResourceRecord",
    "McpRuntime",
    "McpRuntimeStatus",
    "McpServerRuntimeStatus",
    "McpToolCallRecord",
    "get_default_mcp_runtime",
    "reset_default_mcp_runtime",
]
