"""Slash command catalog routes."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from aether.cli.commands import catalog_entries
from aether.web.serializers import to_jsonable

router = APIRouter()


@router.get("/api/commands")
async def commands_catalog() -> dict[str, object]:
    return {"commands": to_jsonable([asdict(entry) for entry in catalog_entries()])}


__all__ = ["router"]
