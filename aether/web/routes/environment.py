"""Environment variable routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from aether.web.serializers import to_jsonable

router = APIRouter()


class EnvSetBody(BaseModel):
    key: str
    value: str


class EnvKeyBody(BaseModel):
    key: str


@router.get("/api/env")
async def environment_catalog(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.environment.catalog())


@router.put("/api/env")
async def set_environment_variable(request: Request, body: EnvSetBody) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.environment.set(body.key, body.value))


@router.delete("/api/env")
async def delete_environment_variable(request: Request, body: EnvKeyBody) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.environment.delete(body.key))


@router.post("/api/env/reveal")
async def reveal_environment_variable(request: Request, body: EnvKeyBody) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.environment.reveal(body.key))


__all__ = ["router"]
