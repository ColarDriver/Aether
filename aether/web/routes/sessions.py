"""Session routes."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel
from typing import Any

from aether.services.common import ServiceNotFoundError, ServiceValidationError
from aether.runtime.session.session_state import get_cwd, set_cwd
from aether.services.sessions import (
    SessionCreateRequest,
    SessionDeleteRequest,
    SessionExportRequest,
    SessionForkRequest,
    SessionImportRequest,
    SessionRenameRequest,
    SessionResumeRequest,
    SessionRewindRequest,
    SessionUpdateRequest,
)
from aether.web.serializers import to_jsonable

router = APIRouter()


class SessionCreateBody(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    system_prompt: str | None = None
    session_id: str | None = None


class SessionUpdateBody(BaseModel):
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    system_prompt: str | None = None
    update_base_url: bool = False
    update_system_prompt: bool = False


class SessionForkBody(BaseModel):
    message_index: int
    new_session_id: str | None = None


class SessionRewindBody(BaseModel):
    message_index: int | None = None
    target_user_message_id: str | None = None
    user_message_index: int | None = None
    expected_content: str | None = None


class SessionActionBody(BaseModel):
    message_index: int | None = None
    target_user_message_id: str | None = None
    user_message_index: int | None = None
    expected_content: str | None = None
    checkpoint_id: str | None = None
    paths: list[str] | None = None
    new_session_id: str | None = None


class SessionRenameBody(BaseModel):
    new_session_id: str


class SessionPermissionModeBody(BaseModel):
    mode: str


class SessionImportBody(BaseModel):
    data: dict[str, Any]
    new_session_id: str | None = None
    overwrite: bool = False
    make_current: bool = True


@router.get("/api/sessions")
async def sessions(request: Request, limit: int | None = 50) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.sessions.list(limit=limit))


@router.post("/api/sessions")
async def create_session(request: Request, body: SessionCreateBody) -> dict[str, object]:
    services = request.app.state.aether_services
    info = services.sessions.create(
        SessionCreateRequest(
            provider=body.provider,
            model=body.model,
            base_url=body.base_url,
            system_prompt=body.system_prompt,
            session_id=body.session_id,
        )
    )
    _seed_workspace_cwd(services, info.session_id)
    return to_jsonable(services.sessions.info(services.sessions.resolve_record(info.session_id)))


@router.get("/api/sessions/current")
async def current_session(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    current = services.sessions.current()
    return {"session": to_jsonable(current) if current is not None else None}


@router.get("/api/sessions/search")
async def search_sessions(request: Request, q: str = "", limit: int | None = 50) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.sessions.search(q, limit=limit))


@router.post("/api/sessions/import")
async def import_session(request: Request, body: SessionImportBody) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.sessions.import_session(
            SessionImportRequest(
                data=body.data,
                new_session_id=body.new_session_id,
                overwrite=body.overwrite,
                make_current=body.make_current,
            )
        )
    )


@router.get("/api/sessions/{session_id}")
async def session_detail(request: Request, session_id: str) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.sessions.detail(session_id))


@router.patch("/api/sessions/{session_id}")
async def update_session(request: Request, session_id: str, body: SessionUpdateBody) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.sessions.update(
            SessionUpdateRequest(
                session_id=session_id,
                provider=body.provider,
                model=body.model,
                base_url=body.base_url,
                system_prompt=body.system_prompt,
                update_base_url=body.update_base_url,
                update_system_prompt=body.update_system_prompt,
            )
        )
    )


@router.get("/api/sessions/{session_id}/permission-mode")
async def session_permission_mode(request: Request, session_id: str) -> dict[str, object]:
    services = request.app.state.aether_services
    mode = services.sessions.permission_mode(session_id)
    return {"session_id": session_id, "mode": mode}


@router.put("/api/sessions/{session_id}/permission-mode")
async def session_permission_mode_set(request: Request, session_id: str, body: SessionPermissionModeBody) -> dict[str, object]:
    services = request.app.state.aether_services
    info = services.sessions.set_permission_mode(session_id, body.mode)
    return to_jsonable({"session_id": info.session_id, "mode": info.permission_mode, "info": info})


@router.post("/api/sessions/{session_id}/rename")
async def rename_session(request: Request, session_id: str, body: SessionRenameBody) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.sessions.rename(
            SessionRenameRequest(
                session_id=session_id,
                new_session_id=body.new_session_id,
            )
        )
    )


@router.get("/api/sessions/{session_id}/export")
async def export_session(request: Request, session_id: str) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.sessions.export(SessionExportRequest(session_id)))


@router.post("/api/sessions/{session_id}/resume")
async def resume_session(request: Request, session_id: str) -> dict[str, object]:
    services = request.app.state.aether_services
    result = services.sessions.resume(SessionResumeRequest(session_id))
    _seed_workspace_cwd(services, result.session_id)
    return to_jsonable(services.sessions.detail(result.session_id))


@router.post("/api/sessions/{session_id}/fork")
async def fork_session(request: Request, session_id: str, body: SessionForkBody) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.sessions.fork(
            SessionForkRequest(
                session_id_or_prefix=session_id,
                message_index=body.message_index,
                new_session_id=body.new_session_id,
            )
        )
    )


@router.post("/api/sessions/{session_id}/rewind")
async def rewind_session(request: Request, session_id: str, body: SessionRewindBody) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.sessions.rewind(
            SessionRewindRequest(
                session_id_or_prefix=session_id,
                message_index=body.message_index,
                target_user_message_id=body.target_user_message_id,
                user_message_index=body.user_message_index,
                expected_content=body.expected_content,
            )
        )
    )


@router.get("/api/sessions/{session_id}/turn-checkpoints")
async def session_turn_checkpoints(request: Request, session_id: str) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.sessions.turn_checkpoints(session_id))


@router.get("/api/sessions/{session_id}/turn-checkpoints/diff")
async def session_turn_checkpoint_diff(
    request: Request,
    session_id: str,
    path: str = Query(...),
    target_user_message_id: str | None = Query(default=None),
    targetUserMessageId: str | None = Query(default=None),
    user_message_index: int | None = Query(default=None),
    userMessageIndex: int | None = Query(default=None),
) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.sessions.turn_checkpoint_diff(
            session_id,
            path=path,
            target_user_message_id=target_user_message_id or targetUserMessageId,
            user_message_index=user_message_index if user_message_index is not None else userMessageIndex,
        )
    )


@router.get("/api/sessions/{session_id}/message-actions/{message_index}")
async def session_message_actions(request: Request, session_id: str, message_index: int) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.sessions.message_actions(session_id, message_index))


@router.post("/api/sessions/{session_id}/actions/fork")
async def session_action_fork(request: Request, session_id: str, body: SessionActionBody) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.sessions.fork(
            SessionForkRequest(
                session_id_or_prefix=session_id,
                message_index=body.message_index,
                target_user_message_id=body.target_user_message_id,
                user_message_index=body.user_message_index,
                expected_content=body.expected_content,
                new_session_id=body.new_session_id,
            )
        )
    )


@router.post("/api/sessions/{session_id}/actions/rewind")
async def session_action_rewind(request: Request, session_id: str, body: SessionActionBody) -> dict[str, object]:
    services = request.app.state.aether_services
    restore = _restore_workspace_for_action(services, body)
    result = services.sessions.rewind(
        SessionRewindRequest(
            session_id_or_prefix=session_id,
            message_index=body.message_index,
            target_user_message_id=body.target_user_message_id,
            user_message_index=body.user_message_index,
            expected_content=body.expected_content,
        )
    )
    return to_jsonable({"action": "rewind", "restore": restore, "result": result})


@router.post("/api/sessions/{session_id}/actions/undo-run")
async def session_action_undo_run(request: Request, session_id: str, body: SessionActionBody) -> dict[str, object]:
    services = request.app.state.aether_services
    restore = _restore_workspace_for_action(services, body)
    result = services.sessions.rewind(
        SessionRewindRequest(
            session_id_or_prefix=session_id,
            message_index=(body.message_index - 1) if body.message_index is not None else None,
            target_user_message_id=body.target_user_message_id,
            user_message_index=body.user_message_index,
            expected_content=body.expected_content,
            rewind_before_target=body.message_index is None,
        )
    )
    return to_jsonable({"action": "undo_run", "restore": restore, "result": result})


@router.post("/api/sessions/{session_id}/actions/retry")
async def session_action_retry(request: Request, session_id: str, body: SessionActionBody) -> dict[str, object]:
    services = request.app.state.aether_services
    restore = _restore_workspace_for_action(services, body)
    result = services.sessions.rewind(
        SessionRewindRequest(
            session_id_or_prefix=session_id,
            message_index=(body.message_index - 1) if body.message_index is not None else None,
            target_user_message_id=body.target_user_message_id,
            user_message_index=body.user_message_index,
            expected_content=body.expected_content,
            rewind_before_target=body.message_index is None,
        )
    )
    return to_jsonable({"action": "retry_prepared", "restore": restore, "result": result})


@router.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(request: Request, session_id: str) -> Response:
    services = request.app.state.aether_services
    deleted = services.sessions.delete(SessionDeleteRequest(session_id))
    if not deleted:
        raise ServiceNotFoundError(
            f"session not found: {session_id}",
            details={"session_id": session_id},
        )
    services.tasks.delete_session_tasks(session_id)
    return Response(status_code=204)


@router.get("/api/sessions/{session_id}/messages")
async def session_messages(request: Request, session_id: str) -> dict[str, object]:
    services = request.app.state.aether_services
    return {
        "session_id": session_id,
        "messages": to_jsonable(services.sessions.transcript(session_id)),
    }


def _seed_workspace_cwd(services: object, session_id: str) -> None:
    if get_cwd(session_id):
        return
    workspace = getattr(services, "workspace", None)
    root = getattr(workspace, "root", None)
    if root is not None:
        set_cwd(session_id, root)


def _restore_workspace_for_action(services: object, body: SessionActionBody) -> dict[str, object] | None:
    if not body.checkpoint_id:
        return None
    workspace = getattr(services, "workspace", None)
    if workspace is None:
        return None
    paths = body.paths
    if paths is None:
        try:
            changes = workspace.changes()
            paths = [change.path for change in changes.changes]
        except Exception:
            paths = []
    if paths:
        restored = workspace.restore_paths_since_checkpoint(body.checkpoint_id, paths)
        return {"checkpoint_id": body.checkpoint_id, "paths": paths, "status": to_jsonable(restored)}
    restored_checkpoint = workspace.restore_checkpoint(body.checkpoint_id)
    return {"checkpoint_id": body.checkpoint_id, "paths": [file.path for file in restored_checkpoint.files], "status": to_jsonable(restored_checkpoint)}


__all__ = ["router"]
