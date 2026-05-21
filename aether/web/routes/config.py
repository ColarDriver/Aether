"""Configuration and preference routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from aether.web.serializers import to_jsonable

router = APIRouter()


class PrefSetBody(BaseModel):
    key: str
    value: Any = None


class PrefKeyBody(BaseModel):
    key: str


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


@router.get("/api/prefs/{key}")
async def pref_get(request: Request, key: str) -> dict[str, object]:
    services = request.app.state.aether_services
    return {"key": key, "value": to_jsonable(services.prefs.get(key))}


@router.put("/api/prefs")
async def pref_set(request: Request, body: PrefSetBody) -> dict[str, object]:
    services = request.app.state.aether_services
    services.prefs.set(body.key, body.value)
    return {"ok": True, "key": body.key, "value": to_jsonable(services.prefs.get(body.key))}


@router.delete("/api/prefs")
async def pref_delete(request: Request, body: PrefKeyBody) -> dict[str, object]:
    services = request.app.state.aether_services
    deleted = services.prefs.delete(body.key)
    return {"ok": True, "key": body.key, "deleted": deleted}


__all__ = ["router"]
