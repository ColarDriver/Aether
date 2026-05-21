"""Workspace browser routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from aether.web.serializers import to_jsonable

router = APIRouter()


@router.get("/api/workspace/tree")
async def workspace_tree(request: Request, path: str = "") -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.tree(path))


@router.get("/api/workspace/file")
async def workspace_file(request: Request, path: str) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.read_file(path))


@router.get("/api/workspace/search")
async def workspace_search(request: Request, q: str = "", limit: int = 100) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.search(q, limit=limit))


__all__ = ["router"]
