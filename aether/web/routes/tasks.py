"""Task observability routes."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from aether.web.serializers import to_jsonable

router = APIRouter()


class TaskMessageBody(BaseModel):
    message: str
    summary: str | None = None


@router.get("/api/tasks")
async def tasks(
    request: Request,
    session_id: str | None = Query(default=None),
    active_only: bool = Query(default=False),
    limit: int = Query(default=50),
    include_output_tail: bool = Query(default=False),
) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.tasks.list_tasks(
            session_id=session_id,
            active_only=active_only,
            limit=limit,
            include_output_tail=include_output_tail,
        )
    )


@router.get("/api/sessions/{session_id}/tasks")
async def session_tasks(
    session_id: str,
    request: Request,
    active_only: bool = Query(default=False),
    limit: int = Query(default=50),
    include_output_tail: bool = Query(default=False),
) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.tasks.list_tasks(
            session_id=session_id,
            active_only=active_only,
            limit=limit,
            include_output_tail=include_output_tail,
        )
    )


@router.get("/api/tasks/{task_id}")
async def task_detail(task_id: str, request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.tasks.get_task(task_id))


@router.get("/api/tasks/{task_id}/messages")
async def task_messages(
    task_id: str,
    request: Request,
    limit: int = Query(default=100),
) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.tasks.get_task_messages(task_id, limit=limit))


@router.post("/api/tasks/{task_id}/messages")
async def task_send_message(
    task_id: str,
    request: Request,
    body: TaskMessageBody,
) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.tasks.send_task_message(
            task_id,
            message=body.message,
            summary=body.summary,
        )
    )


@router.get("/api/tasks/{task_id}/children/messages")
async def task_child_messages(
    task_id: str,
    request: Request,
    limit: int = Query(default=50),
    per_task_limit: int = Query(default=25),
) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.tasks.get_child_task_messages(
            task_id,
            limit=limit,
            per_task_limit=per_task_limit,
        )
    )


@router.get("/api/tasks/{task_id}/result")
async def task_result(task_id: str, request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.tasks.get_task_result(task_id))


@router.post("/api/tasks/{task_id}/stop")
async def task_stop(task_id: str, request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.tasks.stop_task(
            task_id,
            stopper=services.runs.stop_task,
        )
    )


__all__ = ["router"]
