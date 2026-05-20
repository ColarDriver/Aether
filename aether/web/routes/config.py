"""Configuration and preference routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from aether.web.serializers import to_jsonable

router = APIRouter()


@router.get("/api/config")
async def config(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.config.effective())


@router.get("/api/config/defaults")
async def config_defaults(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.config.defaults())


@router.get("/api/config/paths")
async def config_paths(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.config.paths())


@router.get("/api/config/environment-paths")
async def environment_paths(request: Request) -> list[dict[str, object]]:
    services = request.app.state.aether_services
    return to_jsonable(services.config.environment_paths())


@router.get("/api/prefs")
async def prefs(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.prefs.all())


__all__ = ["router"]
