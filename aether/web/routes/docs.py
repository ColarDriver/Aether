"""Documentation routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from aether.web.serializers import to_jsonable

router = APIRouter()


@router.get("/api/docs")
async def docs_index(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.docs.index())


@router.get("/api/docs/{doc_path:path}")
async def doc_content(request: Request, doc_path: str) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.docs.read(doc_path))


__all__ = ["router"]
