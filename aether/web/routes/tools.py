"""Tool catalog routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from aether.web.serializers import to_jsonable

router = APIRouter()


class WebSearchTestBody(BaseModel):
    query: str = "Aether web search test"
    max_results: int = 1


class McpServerBody(BaseModel):
    name: str
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}
    url: str | None = None
    headers: dict[str, str] = {}
    transport: str | None = None
    timeout: float | None = None
    connect_timeout: float | None = None
    enabled: bool = True


@router.get("/api/tools")
async def tools(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.tools.list_tools())


@router.get("/api/tools/groups")
async def tool_groups(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return {"groups": to_jsonable(services.tools.list_groups())}


@router.get("/api/mcp/status")
async def mcp_status(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.tools.mcp_status())


@router.get("/api/mcp/config")
async def mcp_config(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.tools.mcp_config())


@router.put("/api/mcp/servers")
async def mcp_server_upsert(request: Request, body: McpServerBody) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.tools.upsert_mcp_server(
            name=body.name,
            command=body.command,
            args=body.args,
            env=body.env,
            url=body.url,
            headers=body.headers,
            transport=body.transport,
            timeout=body.timeout,
            connect_timeout=body.connect_timeout,
            enabled=body.enabled,
        )
    )


@router.delete("/api/mcp/servers/{name}")
async def mcp_server_delete(request: Request, name: str) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.tools.delete_mcp_server(name=name))


@router.post("/api/mcp/refresh")
async def mcp_refresh(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.tools.refresh_mcp_runtime())


@router.get("/api/mcp/resources")
async def mcp_resources(request: Request, server: str | None = None) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.tools.mcp_resources(server=server))


@router.get("/api/mcp/resources/read")
async def mcp_resource_read(request: Request, server: str, uri: str) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.tools.read_mcp_resource(server=server, uri=uri))


@router.get("/api/web-search/status")
async def web_search_status(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.tools.web_search_status())


@router.post("/api/web-search/test")
async def web_search_test(request: Request, body: WebSearchTestBody) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.tools.test_web_search(query=body.query, max_results=body.max_results))


__all__ = ["router"]
