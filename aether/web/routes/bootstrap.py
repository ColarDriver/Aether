"""Bootstrap data for browser clients."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/bootstrap")
async def bootstrap(request: Request) -> dict[str, object]:
    return {
        "session_token": str(getattr(request.app.state, "aether_session_token", "")),
        "auth_enabled": bool(getattr(request.app.state, "aether_auth_enabled", False)),
        "web": {"enabled": True},
    }


__all__ = ["router"]
