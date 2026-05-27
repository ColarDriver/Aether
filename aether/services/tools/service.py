"""Tool service implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

from aether.runtime.mcp import McpRuntime, get_default_mcp_runtime, reset_default_mcp_runtime
from aether.runtime.mcp.config import sanitize_mcp_name_component
from aether.services.common import ServiceValidationError
from aether.services.tools.contracts import (
    McpConfigList,
    McpConfigMutationResult,
    McpConfiguredServer,
    McpImportedTool,
    McpResourceContent,
    McpResourceList,
    McpResourceReadResult,
    McpResourceSummary,
    McpServerSummary,
    McpStatus,
    ToolCatalog,
    ToolGroup,
    ToolSummary,
    WebSearchStatus,
    WebSearchTestResult,
)
from aether.tools.base import ToolDescriptor


RegistryFactory = Callable[[], Any]
WebSearchToolFactory = Callable[[], Any]
McpResourceProvider = Callable[[str | None], Sequence[McpResourceSummary] | McpResourceList]
McpResourceReader = Callable[[str, str], McpResourceReadResult]

_WEB_SEARCH_SUPPORTED_PROVIDERS = {"brave", "tavily", "bocha"}

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

    def __init__(
        self,
        *,
        registry_factory: RegistryFactory | None = None,
        environ: Mapping[str, str] | None = None,
        web_search_tool_factory: WebSearchToolFactory | None = None,
        mcp_resource_provider: McpResourceProvider | None = None,
        mcp_resource_reader: McpResourceReader | None = None,
        mcp_runtime: McpRuntime | None = None,
    ) -> None:
        self._registry_factory = registry_factory
        self._environ = environ
        self._web_search_tool_factory = web_search_tool_factory
        self._mcp_resource_provider = mcp_resource_provider
        self._mcp_resource_reader = mcp_resource_reader
        self._mcp_runtime = mcp_runtime

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

    def web_search_status(self) -> WebSearchStatus:
        env = _env(self._environ)
        provider = _resolve_web_search_provider(env)
        api_key = _resolve_web_search_api_key(env)
        if provider not in _WEB_SEARCH_SUPPORTED_PROVIDERS:
            return WebSearchStatus(
                enabled=False,
                provider=provider,
                supported_providers=sorted(_WEB_SEARCH_SUPPORTED_PROVIDERS),
                api_key_configured=bool(api_key),
                api_key_source="env" if api_key else None,
                status="invalid_provider",
                message=(
                    f"WEB_SEARCH_PROVIDER={provider!r} is not supported. "
                    "Use brave, tavily, or bocha."
                ),
            )
        if not api_key:
            return WebSearchStatus(
                enabled=False,
                provider=provider,
                supported_providers=sorted(_WEB_SEARCH_SUPPORTED_PROVIDERS),
                api_key_configured=False,
                status="missing_credential",
                message="Set WEB_SEARCH_API_KEY to enable local web_search.",
            )
        return WebSearchStatus(
            enabled=True,
            provider=provider,
            supported_providers=sorted(_WEB_SEARCH_SUPPORTED_PROVIDERS),
            api_key_configured=True,
            api_key_source="env",
            status="ready",
            message=f"Local web_search is configured for {provider}.",
        )

    def test_web_search(self, *, query: str = "Aether web search test", max_results: int = 1) -> WebSearchTestResult:
        normalized_query = query.strip() or "Aether web search test"
        status = self.web_search_status()
        if not status.enabled:
            return WebSearchTestResult(
                ok=False,
                provider=status.provider,
                query=normalized_query,
                message=status.message,
                error=status.status,
            )
        env = _env(self._environ)
        api_key = _resolve_web_search_api_key(env) or ""
        try:
            max_result_count = max(1, min(int(max_results), 5))
        except (TypeError, ValueError):
            max_result_count = 1
        try:
            from aether.config.schema import EngineConfig
            from aether.runtime.core.contracts import ToolCall, TurnContext
            from aether.tools.builtins.web_search import WebSearchTool

            tool = self._web_search_tool_factory() if self._web_search_tool_factory is not None else WebSearchTool()
            result = tool.execute(
                ToolCall(
                    id="web-search-test",
                    name="web_search",
                    arguments={"query": normalized_query, "max_results": max_result_count},
                ),
                TurnContext(
                    session_id="web-search-test",
                    iteration=0,
                    metadata={
                        "_engine_config": EngineConfig(
                            web_search_provider=status.provider,
                            web_search_api_key=api_key,
                        )
                    },
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return WebSearchTestResult(
                ok=False,
                provider=status.provider,
                query=normalized_query,
                message=f"web_search test failed: {exc}",
                error=type(exc).__name__,
            )
        content = str(result.content or "")
        return WebSearchTestResult(
            ok=not bool(result.is_error),
            provider=status.provider,
            query=normalized_query,
            result_count=int(result.metadata.get("result_count") or 0) if isinstance(result.metadata, dict) else 0,
            message="web_search test succeeded." if not result.is_error else _truncate(content, 240),
            content_preview=_truncate(content, 800),
            error=_truncate(content, 240) if result.is_error else None,
        )

    def mcp_config(self) -> McpConfigList:
        path = _mcp_managed_config_path(_env(self._environ))
        payload = _read_managed_mcp_payload(path)
        servers = _raw_mcp_servers(payload)
        return McpConfigList(
            config_path=str(path),
            exists=path.exists(),
            servers=[
                _configured_server_summary(name, raw)
                for name, raw in sorted(servers.items(), key=lambda item: sanitize_mcp_name_component(str(item[0])))
                if isinstance(raw, Mapping)
            ],
        )

    def upsert_mcp_server(
        self,
        *,
        name: str,
        command: str | None = None,
        args: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        url: str | None = None,
        headers: Mapping[str, str] | None = None,
        transport: str | None = None,
        timeout: float | None = None,
        connect_timeout: float | None = None,
        enabled: bool = True,
    ) -> McpConfigMutationResult:
        server_name = sanitize_mcp_name_component(_require_text(name, "name"))
        command_value = _optional_text(command)
        url_value = _optional_text(url)
        if command_value and url_value:
            raise ServiceValidationError(
                "MCP server must use either command or url, not both.",
                details={"name": server_name},
            )
        if not command_value and not url_value:
            raise ServiceValidationError(
                "MCP server requires a command or url.",
                details={"name": server_name},
            )
        transport_value = _normalize_mcp_transport(transport, remote=bool(url_value))
        server_payload: dict[str, Any] = {
            "enabled": bool(enabled),
            "transport": transport_value,
        }
        if command_value:
            server_payload["command"] = command_value
            server_payload["args"] = [str(item) for item in (args or []) if str(item).strip()]
            if env:
                server_payload["env"] = _string_dict(env)
        if url_value:
            server_payload["url"] = _validate_mcp_url(url_value)
            if headers:
                server_payload["headers"] = _string_dict(headers)
        if timeout is not None:
            server_payload["timeout"] = _positive_float(timeout, "timeout")
        if connect_timeout is not None:
            server_payload["connect_timeout"] = _positive_float(connect_timeout, "connect_timeout")

        path = _mcp_managed_config_path(_env(self._environ))
        payload = _read_managed_mcp_payload(path)
        servers = dict(_raw_mcp_servers(payload))
        servers[server_name] = server_payload
        _write_managed_mcp_payload(path, servers)
        reset_default_mcp_runtime()
        return McpConfigMutationResult(
            ok=True,
            config_path=str(path),
            message=f"MCP server {server_name!r} saved.",
            server=_configured_server_summary(server_name, server_payload),
        )

    def delete_mcp_server(self, *, name: str) -> McpConfigMutationResult:
        server_name = sanitize_mcp_name_component(_require_text(name, "name"))
        path = _mcp_managed_config_path(_env(self._environ))
        payload = _read_managed_mcp_payload(path)
        servers = dict(_raw_mcp_servers(payload))
        existed = server_name in servers
        servers.pop(server_name, None)
        _write_managed_mcp_payload(path, servers)
        reset_default_mcp_runtime()
        return McpConfigMutationResult(
            ok=True,
            config_path=str(path),
            message=(
                f"MCP server {server_name!r} deleted."
                if existed
                else f"MCP server {server_name!r} was not configured."
            ),
        )

    def refresh_mcp_runtime(self) -> McpStatus:
        if self._mcp_runtime is not None:
            self._mcp_runtime.discover_tools(force=True)
        else:
            reset_default_mcp_runtime()
        return self.mcp_status()

    def mcp_resources(self, *, server: str | None = None) -> McpResourceList:
        status = self.mcp_status()
        requested_server = server.strip() if isinstance(server, str) and server.strip() else None
        if not status.enabled:
            return McpResourceList(
                enabled=False,
                status=status.status,
                message=status.message,
            )
        if requested_server and all(item.name != requested_server for item in status.servers):
            return McpResourceList(
                enabled=False,
                status="server_not_found",
                message=f"MCP server {requested_server!r} is not configured for this runtime.",
            )
        runtime = self._get_mcp_runtime()
        if runtime is not None:
            try:
                resources = [
                    McpResourceSummary(
                        server=resource.server,
                        uri=resource.uri,
                        name=resource.name,
                        mime_type=resource.mime_type,
                        description=resource.description,
                    )
                    for resource in runtime.list_resources(server=requested_server)
                ]
            except Exception as exc:  # noqa: BLE001
                return McpResourceList(
                    enabled=True,
                    status="error",
                    message=f"MCP resource provider failed: {exc}",
                )
            return McpResourceList(
                enabled=True,
                status="available",
                message=f"{len(resources)} MCP resource(s) available." if resources else "No MCP resources are available.",
                resources=resources,
            )

        if self._mcp_resource_provider is None:
            scope = f" for server {requested_server}" if requested_server else ""
            return McpResourceList(
                enabled=True,
                status="not_available",
                message=(
                    "MCP tools are exposed through the tool catalog"
                    + scope
                    + ", but this Aether runtime does not expose MCP resource browsing yet."
                ),
            )
        try:
            provided = self._mcp_resource_provider(requested_server)
        except Exception as exc:  # noqa: BLE001
            return McpResourceList(
                enabled=True,
                status="error",
                message=f"MCP resource provider failed: {exc}",
            )
        if isinstance(provided, McpResourceList):
            if requested_server is None:
                return provided
            return McpResourceList(
                enabled=provided.enabled,
                status=provided.status,
                message=provided.message,
                resources=[resource for resource in provided.resources if resource.server == requested_server],
            )
        resources = list(provided)
        if requested_server is not None:
            resources = [resource for resource in resources if resource.server == requested_server]
        return McpResourceList(
            enabled=True,
            status="available",
            message=f"{len(resources)} MCP resource(s) available." if resources else "No MCP resources are available.",
            resources=resources,
        )

    def read_mcp_resource(self, *, server: str, uri: str) -> McpResourceReadResult:
        requested_server = server.strip() if isinstance(server, str) else ""
        requested_uri = uri.strip() if isinstance(uri, str) else ""
        if not requested_server:
            return McpResourceReadResult(
                enabled=False,
                status="invalid_request",
                message="MCP server is required.",
                server=requested_server,
                uri=requested_uri,
            )
        if not requested_uri:
            return McpResourceReadResult(
                enabled=False,
                status="invalid_request",
                message="MCP resource URI is required.",
                server=requested_server,
                uri=requested_uri,
            )

        status = self.mcp_status()
        if not status.enabled:
            return McpResourceReadResult(
                enabled=False,
                status=status.status,
                message=status.message,
                server=requested_server,
                uri=requested_uri,
            )
        if all(item.name != requested_server for item in status.servers):
            return McpResourceReadResult(
                enabled=False,
                status="server_not_found",
                message=f"MCP server {requested_server!r} is not configured for this runtime.",
                server=requested_server,
                uri=requested_uri,
            )
        runtime = self._get_mcp_runtime()
        if runtime is not None:
            try:
                result = runtime.read_resource(server=requested_server, uri=requested_uri)
            except Exception as exc:  # noqa: BLE001
                return McpResourceReadResult(
                    enabled=True,
                    status="error",
                    message=f"MCP resource reader failed: {exc}",
                    server=requested_server,
                    uri=requested_uri,
                )
            return McpResourceReadResult(
                enabled=True,
                status="available",
                message="MCP resource read succeeded.",
                server=result.server,
                uri=result.uri,
                name=result.name,
                mime_type=result.mime_type,
                contents=[
                    McpResourceContent(
                        type=content.type,
                        text=content.text,
                        blob=content.blob,
                        mime_type=content.mime_type,
                        uri=content.uri,
                    )
                    for content in result.contents
                ],
            )

        if self._mcp_resource_reader is None:
            return McpResourceReadResult(
                enabled=True,
                status="not_available",
                message=(
                    "MCP resource listing is visible through the web console, "
                    "but this Aether runtime does not expose MCP resource reads yet."
                ),
                server=requested_server,
                uri=requested_uri,
            )

        try:
            result = self._mcp_resource_reader(requested_server, requested_uri)
        except Exception as exc:  # noqa: BLE001
            return McpResourceReadResult(
                enabled=True,
                status="error",
                message=f"MCP resource reader failed: {exc}",
                server=requested_server,
                uri=requested_uri,
            )
        return McpResourceReadResult(
            enabled=result.enabled,
            status=result.status,
            message=result.message,
            server=result.server or requested_server,
            uri=result.uri or requested_uri,
            name=result.name,
            mime_type=result.mime_type,
            contents=[_mcp_content_from_value(content) for content in result.contents],
        )


    def mcp_status(self) -> McpStatus:
        runtime = self._get_mcp_runtime()
        if runtime is not None:
            status = runtime.status()
            return McpStatus(
                enabled=status.enabled,
                status=status.status,
                message=status.message,
                servers=[
                    McpServerSummary(
                        name=server.name,
                        status=server.status,
                        tools_count=server.tools_count,
                        resources_count=server.resources_count,
                        credential_status=server.credential_status,
                    )
                    for server in status.servers
                ],
                imported_tools=[
                    McpImportedTool(
                        name=tool.name,
                        server=tool.server,
                        local_name=tool.local_name,
                        description=tool.description,
                        enabled=tool.enabled,
                    )
                    for tool in status.imported_tools
                ],
            )

        imported_tools: list[McpImportedTool] = []
        counts_by_server: dict[str, int] = {}
        for tool in self.list_tools().tools:
            parsed = _parse_mcp_tool_name(tool.name)
            if parsed is None:
                continue
            server, local_name = parsed
            counts_by_server[server] = counts_by_server.get(server, 0) + 1
            imported_tools.append(
                McpImportedTool(
                    name=tool.name,
                    server=server,
                    local_name=local_name,
                    description=tool.description,
                    enabled=tool.enabled,
                )
            )
        servers = [
            McpServerSummary(
                name=server,
                status="available",
                tools_count=tools_count,
                resources_count=0,
                credential_status="unknown",
            )
            for server, tools_count in sorted(counts_by_server.items())
        ]
        if not servers:
            return McpStatus(
                enabled=False,
                status="not_configured",
                message="No MCP servers are configured for this Aether runtime.",
            )
        return McpStatus(
            enabled=True,
            status="available",
            message=f"{len(servers)} MCP server(s) exposed through the tool catalog.",
            servers=servers,
            imported_tools=sorted(imported_tools, key=lambda item: (item.server, item.local_name)),
        )

    def _descriptors(self) -> list[ToolDescriptor]:
        registry = self._build_registry()
        descriptors = registry.list_descriptors()
        return list(descriptors)

    def _build_registry(self) -> Any:
        if self._registry_factory is not None:
            return self._registry_factory()
        # Keep this lazy so importing services does not eagerly construct
        # heavyweight browser/LSP resources during gateway boot.
        from aether.config.schema import EngineConfig
        from aether.tools.builtins import build_default_tool_registry

        return build_default_tool_registry(config=EngineConfig())

    def _get_mcp_runtime(self) -> McpRuntime | None:
        runtime = self._mcp_runtime or get_default_mcp_runtime()
        if not runtime.configured_servers:
            return None
        return runtime


def _env(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return environ if environ is not None else os.environ


def _mcp_managed_config_path(env: Mapping[str, str]) -> Path:
    raw = env.get("AETHER_MCP_CONFIG")
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser()
    home = env.get("AETHER_HOME")
    root = Path(home).expanduser() if isinstance(home, str) and home.strip() else Path.home() / ".aether"
    return root / "mcp_servers.json"


def _read_managed_mcp_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"mcp_servers": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ServiceValidationError(
            f"MCP config file is not valid JSON: {path}",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise ServiceValidationError(
            f"MCP config file must contain a JSON object: {path}",
            details={"path": str(path)},
        )
    return payload


def _raw_mcp_servers(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    servers = payload.get("mcp_servers") if "mcp_servers" in payload else payload
    return servers if isinstance(servers, Mapping) else {}


def _write_managed_mcp_payload(path: Path, servers: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mcp_servers": dict(sorted(servers.items()))}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configured_server_summary(name: str, raw: Mapping[str, Any]) -> McpConfiguredServer:
    raw_env = raw.get("env")
    raw_headers = raw.get("headers")
    raw_args = raw.get("args")
    env: Mapping[str, Any] = raw_env if isinstance(raw_env, Mapping) else {}
    headers: Mapping[str, Any] = raw_headers if isinstance(raw_headers, Mapping) else {}
    args: list[Any] = raw_args if isinstance(raw_args, list) else []
    return McpConfiguredServer(
        name=sanitize_mcp_name_component(str(name)),
        enabled=bool(raw.get("enabled", True)),
        transport=str(raw.get("transport") or ("http" if raw.get("url") else "stdio")).strip().lower(),
        command=_optional_text(raw.get("command")),
        args=[str(item) for item in args if item is not None],
        url=_optional_text(raw.get("url")),
        env_keys=sorted(str(key) for key in env.keys()),
        header_keys=sorted(str(key) for key in headers.keys()),
        timeout=_float_or_none(raw.get("timeout")),
        connect_timeout=_float_or_none(raw.get("connect_timeout")),
        source="file",
    )


def _require_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServiceValidationError(f"{field} is required", details={field: value})
    return value.strip()


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _string_dict(value: Mapping[str, str]) -> dict[str, str]:
    return {
        str(key): str(item)
        for key, item in value.items()
        if str(key).strip() and item is not None
    }


def _normalize_mcp_transport(value: str | None, *, remote: bool) -> str:
    normalized = (value or ("http" if remote else "stdio")).strip().lower()
    if normalized not in {"stdio", "http", "sse"}:
        raise ServiceValidationError(
            "MCP transport must be one of stdio, http, or sse.",
            details={"transport": value},
        )
    if remote and normalized == "stdio":
        raise ServiceValidationError(
            "Remote MCP servers must use http or sse transport.",
            details={"transport": value},
        )
    if not remote and normalized != "stdio":
        raise ServiceValidationError(
            "Command-based MCP servers must use stdio transport.",
            details={"transport": value},
        )
    return normalized


def _validate_mcp_url(value: str) -> str:
    if not (value.startswith("http://") or value.startswith("https://")):
        raise ServiceValidationError(
            "MCP url must start with http:// or https://.",
            details={"url": value},
        )
    return value


def _positive_float(value: float, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ServiceValidationError(f"{field} must be a positive number", details={field: value}) from exc
    if result <= 0:
        raise ServiceValidationError(f"{field} must be a positive number", details={field: value})
    return result


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _resolve_web_search_provider(env: Mapping[str, str]) -> str:
    return str(env.get("WEB_SEARCH_PROVIDER") or "brave").strip().lower().replace("_", "-")


def _resolve_web_search_api_key(env: Mapping[str, str]) -> str | None:
    value = env.get("WEB_SEARCH_API_KEY")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "..."


def _parse_mcp_tool_name(name: str) -> tuple[str, str] | None:
    parts = name.split("__", 2)
    if len(parts) != 3 or parts[0] != "mcp" or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


def _mcp_content_from_value(value: McpResourceContent | Mapping[str, Any]) -> McpResourceContent:
    if isinstance(value, McpResourceContent):
        return value
    content_type = str(value.get("type") or "text").strip() or "text"
    text = value.get("text")
    blob = value.get("blob")
    mime_type = value.get("mime_type") or value.get("mimeType")
    uri = value.get("uri")
    return McpResourceContent(
        type=content_type,
        text=text if isinstance(text, str) else None,
        blob=blob if isinstance(blob, str) else None,
        mime_type=mime_type if isinstance(mime_type, str) else None,
        uri=uri if isinstance(uri, str) else None,
    )


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
