"""Health and status routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from aether.web.serializers import to_jsonable

router = APIRouter()


@router.get("/api/status")
async def status() -> dict[str, object]:
    return {
        "ok": True,
        "name": "Aether",
        "version": _package_version(),
        "web": {"enabled": True},
    }


@router.get("/api/health")
async def health(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.health.status())


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("aether-harness")
    except Exception:
        return "unknown"


__all__ = ["router"]
