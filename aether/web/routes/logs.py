"""Runtime log routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from aether.web.serializers import to_jsonable

router = APIRouter()


@router.get("/api/logs/files")
async def log_files(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return {"files": to_jsonable(services.logs.files())}


@router.get("/api/logs")
async def logs(
    request: Request,
    file: str = "gateway",
    lines: int = 100,
    level: str | None = None,
    component: str | None = None,
    search: str | None = None,
) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.logs.read(
            file=file,
            lines=lines,
            level=level,
            component=component,
            search=search,
        )
    )


__all__ = ["router"]
