"""Session routes."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from aether.services.common import ServiceNotFoundError
from aether.services.sessions import (
    SessionCreateRequest,
    SessionDeleteRequest,
    SessionResumeRequest,
)
from aether.web.serializers import to_jsonable

router = APIRouter()


class SessionCreateBody(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    system_prompt: str | None = None
    session_id: str | None = None


@router.get("/api/sessions")
async def sessions(request: Request, limit: int | None = 50) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.sessions.list(limit=limit))


@router.post("/api/sessions")
async def create_session(request: Request, body: SessionCreateBody) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.sessions.create(
            SessionCreateRequest(
                provider=body.provider,
                model=body.model,
                base_url=body.base_url,
                system_prompt=body.system_prompt,
                session_id=body.session_id,
            )
        )
    )


@router.get("/api/sessions/current")
async def current_session(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    current = services.sessions.current()
    return {"session": to_jsonable(current) if current is not None else None}


@router.get("/api/sessions/search")
async def search_sessions(request: Request, q: str = "", limit: int | None = 50) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.sessions.search(q, limit=limit))


@router.get("/api/sessions/{session_id}")
async def session_detail(request: Request, session_id: str) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.sessions.detail(session_id))


@router.post("/api/sessions/{session_id}/resume")
async def resume_session(request: Request, session_id: str) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.sessions.resume(SessionResumeRequest(session_id)))


@router.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(request: Request, session_id: str) -> Response:
    services = request.app.state.aether_services
    deleted = services.sessions.delete(SessionDeleteRequest(session_id))
    if not deleted:
        raise ServiceNotFoundError(
            f"session not found: {session_id}",
            details={"session_id": session_id},
        )
    return Response(status_code=204)


@router.get("/api/sessions/{session_id}/messages")
async def session_messages(request: Request, session_id: str) -> dict[str, object]:
    services = request.app.state.aether_services
    return {
        "session_id": session_id,
        "messages": to_jsonable(services.sessions.transcript(session_id)),
    }


__all__ = ["router"]
