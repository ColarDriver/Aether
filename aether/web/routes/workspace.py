"""Workspace browser routes."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse

from aether.runtime.session.session_state import set_cwd
from aether.web.serializers import to_jsonable

router = APIRouter()

_WORKSPACE_ACTIVE_ROOT_PREF = "workspace.active_root"
_WORKSPACE_RECENT_ROOTS_PREF = "workspace.recent_roots"


class WorkspaceFileSaveRequest(BaseModel):
    path: str
    content: str


class WorkspaceFileCreateRequest(BaseModel):
    path: str
    content: str = ""


class WorkspaceDirectoryCreateRequest(BaseModel):
    path: str


class WorkspacePathRenameRequest(BaseModel):
    path: str
    new_path: str


class WorkspaceRootSwitchRequest(BaseModel):
    path: str
    session_id: str | None = None
    remember: bool = True


class WorkspaceGitRestoreRequest(BaseModel):
    path: str


class WorkspaceCheckpointCreateRequest(BaseModel):
    label: str | None = None


class WorkspaceCheckpointRestorePathsRequest(BaseModel):
    paths: list[str]


class WorkspaceChangesAcceptRequest(BaseModel):
    paths: list[str]


class WorkspaceChangesRejectRequest(BaseModel):
    paths: list[str]
    checkpoint_id: str | None = None
    expected_hashes: dict[str, str] | None = None


class WorkspaceChangesVerifyRequest(BaseModel):
    paths: list[str]
    command: list[str] | None = None
    timeout_seconds: float = 120.0


@router.get("/api/workspace/root")
async def workspace_root(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.root_info(recent_roots=_recent_roots(services)))


@router.put("/api/workspace/root")
async def workspace_root_switch(request: Request, payload: WorkspaceRootSwitchRequest) -> dict[str, object]:
    services = request.app.state.aether_services
    remembered = _recent_roots(services)
    previous_root = str(services.workspace.root)
    info = services.workspace.switch_root(payload.path, recent_roots=remembered)
    root = info.root
    if payload.remember:
        next_roots = _merge_recent_roots(root, [previous_root, *remembered])
        services.prefs.set(_WORKSPACE_ACTIVE_ROOT_PREF, root)
        services.prefs.set(_WORKSPACE_RECENT_ROOTS_PREF, next_roots)
        info = services.workspace.root_info(recent_roots=next_roots)
    if payload.session_id:
        set_cwd(payload.session_id, services.workspace.root)
    return to_jsonable(info)


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


@router.post("/api/workspace/file")
async def workspace_file_create(request: Request, payload: WorkspaceFileCreateRequest) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.create_file(payload.path, payload.content))


@router.post("/api/workspace/directory")
async def workspace_directory_create(request: Request, payload: WorkspaceDirectoryCreateRequest) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.create_directory(payload.path))


@router.patch("/api/workspace/path")
async def workspace_path_rename(request: Request, payload: WorkspacePathRenameRequest) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.rename_path(payload.path, payload.new_path))


@router.delete("/api/workspace/path", status_code=204)
async def workspace_path_delete(request: Request, path: str, recursive: bool = False) -> Response:
    services = request.app.state.aether_services
    services.workspace.delete_path(path, recursive=recursive)
    return Response(status_code=204)


@router.get("/api/workspace/raw")
async def workspace_file_raw(request: Request, path: str) -> FileResponse:
    services = request.app.state.aether_services
    file_path = services.workspace.raw_file_path(path)
    return FileResponse(file_path, media_type=services.workspace.mime_type(path), filename=file_path.name)


@router.get("/api/workspace/search")
async def workspace_search(request: Request, q: str = "", limit: int = 100) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.search(q, limit=limit))


@router.get("/api/workspace/git/status")
async def workspace_git_status(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.git_status())


@router.get("/api/workspace/git/diff")
async def workspace_git_diff(
    request: Request,
    path: str | None = None,
    staged: bool = False,
) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.git_diff(path=path, staged=staged))


@router.post("/api/workspace/git/restore")
async def workspace_git_restore(request: Request, payload: WorkspaceGitRestoreRequest) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.git_restore(payload.path))


@router.get("/api/workspace/changes")
async def workspace_changes(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.changes())


@router.post("/api/workspace/changes/accept")
async def workspace_changes_accept(request: Request, payload: WorkspaceChangesAcceptRequest) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.accept_changes(payload.paths))


@router.post("/api/workspace/changes/reject")
async def workspace_changes_reject(request: Request, payload: WorkspaceChangesRejectRequest) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.workspace.reject_changes(
            payload.paths,
            checkpoint_id=payload.checkpoint_id,
            expected_hashes=payload.expected_hashes,
        )
    )


@router.post("/api/workspace/changes/verify")
async def workspace_changes_verify(request: Request, payload: WorkspaceChangesVerifyRequest) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.workspace.verify_changes(
            payload.paths,
            command=payload.command,
            timeout_seconds=payload.timeout_seconds,
        )
    )


@router.get("/api/workspace/checkpoints")
async def workspace_checkpoints(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.list_checkpoints())


@router.post("/api/workspace/checkpoints")
async def workspace_checkpoint_create(request: Request, payload: WorkspaceCheckpointCreateRequest) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.create_checkpoint(label=payload.label))


@router.post("/api/workspace/checkpoints/{checkpoint_id}/restore")
async def workspace_checkpoint_restore(request: Request, checkpoint_id: str) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.restore_checkpoint(checkpoint_id))


@router.post("/api/workspace/checkpoints/{checkpoint_id}/restore-paths")
async def workspace_checkpoint_restore_paths(
    request: Request,
    checkpoint_id: str,
    payload: WorkspaceCheckpointRestorePathsRequest,
) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.workspace.restore_paths_since_checkpoint(checkpoint_id, payload.paths))


def _recent_roots(services: object) -> list[str]:
    prefs = getattr(services, "prefs", None)
    if prefs is None:
        return []
    value = prefs.get(_WORKSPACE_RECENT_ROOTS_PREF)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _merge_recent_roots(root: str, roots: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in [root, *roots]:
        if not isinstance(item, str) or not item.strip() or item in seen:
            continue
        try:
            normalized = str(Path(item).expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out[:12]


__all__ = ["router"]
