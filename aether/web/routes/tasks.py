"""Task observability routes."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from aether.web.serializers import to_jsonable

router = APIRouter()


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


__all__ = ["router"]
