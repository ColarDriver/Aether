"""Analytics routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from aether.web.serializers import to_jsonable

router = APIRouter()


@router.get("/api/analytics")
async def analytics_report(request: Request, days: int = 30, limit: int = 20) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.analytics.report(days=days, session_limit=limit))


__all__ = ["router"]
