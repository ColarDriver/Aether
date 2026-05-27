"""Tool service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolSummary:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class ToolCatalog:
    tools: list[ToolSummary] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ToolGroup:
    name: str
    tools: list[ToolSummary] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ToolAvailability:
    name: str
    enabled: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class McpImportedTool:
    name: str
    server: str
    local_name: str
    description: str = ""
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class McpServerSummary:
    name: str
    status: str
    tools_count: int = 0
    resources_count: int = 0
    credential_status: str = "unknown"


@dataclass(frozen=True, slots=True)
class McpConfiguredServer:
    name: str
    enabled: bool = True
    transport: str = "stdio"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env_keys: list[str] = field(default_factory=list)
    header_keys: list[str] = field(default_factory=list)
    timeout: float | None = None
    connect_timeout: float | None = None
    source: str = "file"


@dataclass(frozen=True, slots=True)
class McpConfigList:
    config_path: str
    exists: bool
    servers: list[McpConfiguredServer] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class McpConfigMutationResult:
    ok: bool
    config_path: str
    message: str
    server: McpConfiguredServer | None = None


@dataclass(frozen=True, slots=True)
class McpStatus:
    enabled: bool
    status: str
    message: str
    servers: list[McpServerSummary] = field(default_factory=list)
    imported_tools: list[McpImportedTool] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class McpResourceSummary:
    server: str
    uri: str
    name: str
    mime_type: str | None = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class McpResourceList:
    enabled: bool
    status: str
    message: str
    resources: list[McpResourceSummary] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class McpResourceContent:
    type: str
    text: str | None = None
    blob: str | None = None
    mime_type: str | None = None
    uri: str | None = None


@dataclass(frozen=True, slots=True)
class McpResourceReadResult:
    enabled: bool
    status: str
    message: str
    server: str
    uri: str
    name: str | None = None
    mime_type: str | None = None
    contents: list[McpResourceContent] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class WebSearchStatus:
    enabled: bool
    provider: str
    supported_providers: list[str] = field(default_factory=list)
    api_key_configured: bool = False
    credential_name: str = "WEB_SEARCH_API_KEY"
    api_key_source: str | None = None
    status: str = "unknown"
    message: str = ""


@dataclass(frozen=True, slots=True)
class WebSearchTestResult:
    ok: bool
    provider: str
    query: str
    result_count: int = 0
    message: str = ""
    content_preview: str = ""
    error: str | None = None


__all__ = [
    "McpConfigList",
    "McpConfigMutationResult",
    "McpConfiguredServer",
    "McpImportedTool",
    "McpResourceContent",
    "McpResourceList",
    "McpResourceReadResult",
    "McpResourceSummary",
    "McpServerSummary",
    "McpStatus",
    "ToolAvailability",
    "ToolCatalog",
    "ToolGroup",
    "WebSearchStatus",
    "WebSearchTestResult",
    "ToolSummary",
]
