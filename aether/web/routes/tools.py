"""Tool catalog routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from aether.web.serializers import to_jsonable

router = APIRouter()


@router.get("/api/tools")
async def tools(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.tools.list_tools())


@router.get("/api/tools/groups")
async def tool_groups(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return {"groups": to_jsonable(services.tools.list_groups())}


__all__ = ["router"]
