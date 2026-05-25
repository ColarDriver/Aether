"""Workspace browser routes."""

from __future__ import annotations

from pydantic import BaseModel
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from aether.web.serializers import to_jsonable

router = APIRouter()


class WorkspaceFileSaveRequest(BaseModel):
    path: str
    content: str


@router.get("/api/workspace/tree")
async def workspace_tree(request: Request, path: str = "") -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.tree(path))


@router.get("/api/workspace/file")
async def workspace_file(request: Request, path: str) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.read_file(path))


@router.put("/api/workspace/file")
async def workspace_file_save(request: Request, payload: WorkspaceFileSaveRequest) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.write_file(payload.path, payload.content))


@router.get("/api/workspace/raw")
async def workspace_file_raw(request: Request, path: str) -> FileResponse:
    services = request.app.state.aether_services
    file_path = services.workspace.raw_file_path(path)
    return FileResponse(file_path, media_type=services.workspace.mime_type(path), filename=file_path.name)


@router.get("/api/workspace/search")
async def workspace_search(request: Request, q: str = "", limit: int = 100) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.search(q, limit=limit))


__all__ = ["router"]
