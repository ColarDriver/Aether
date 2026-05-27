"""Context compression routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from aether.services.context import ContextCompressRequest, ContextEstimateRequest
from aether.web.serializers import to_jsonable

router = APIRouter()


class ContextCompressBody(BaseModel):
    focus: str | None = None
    force: bool = True


class ContextEstimateBody(BaseModel):
    draft: str = ""
    attachments: list[dict[str, object]] = Field(default_factory=list)


@router.get("/api/context/{session_id}/status")
async def context_status(request: Request, session_id: str) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.context.status(session_id))


@router.post("/api/context/{session_id}/compress")
async def context_compress(request: Request, session_id: str, body: ContextCompressBody) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.context.compress(
            ContextCompressRequest(
                session_id=session_id,
                focus=body.focus,
                force=body.force,
            )
        )
    )


@router.post("/api/context/{session_id}/estimate")
async def context_estimate(request: Request, session_id: str, body: ContextEstimateBody) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.context.estimate(
            ContextEstimateRequest(
                session_id=session_id,
                draft=body.draft,
                attachments=body.attachments,
            )
        )
    )


__all__ = ["router"]
