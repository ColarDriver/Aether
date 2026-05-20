"""Diagnostics routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from aether.web.serializers import to_jsonable

router = APIRouter()


@router.get("/api/diagnostics")
async def diagnostics(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.diagnostics.status())


__all__ = ["router"]
