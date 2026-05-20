"""Skill catalog routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from aether.services.common import ServiceNotFoundError
from aether.web.serializers import to_jsonable

router = APIRouter()


@router.get("/api/skills")
async def skills(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.skills.list_skills())


@router.get("/api/skills/{name}")
async def skill_detail(request: Request, name: str) -> dict[str, object]:
    services = request.app.state.aether_services
    skill = services.skills.get_skill(name)
    if skill is None:
        raise ServiceNotFoundError(
            f"skill not found: {name}",
            details={"name": name},
        )
    return to_jsonable(skill)


__all__ = ["router"]
