"""Plan-mode routes for the browser console."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from aether.runtime.session.plan_artifact import clear_plan, get_plan_path, read_plan
from aether.runtime.session.session_state import get_mode
from aether.web.serializers import to_jsonable

router = APIRouter()


class PlanModeBody(BaseModel):
    mode: str


@router.get("/api/plan/{session_id}")
async def plan_current(request: Request, session_id: str) -> dict[str, Any]:
    services = request.app.state.aether_services
    record = services.sessions.resolve_record(session_id)
    return _plan_envelope(record.session_id)


@router.put("/api/plan/{session_id}/mode")
async def set_plan_mode(request: Request, session_id: str, body: PlanModeBody) -> dict[str, Any]:
    services = request.app.state.aether_services
    info = services.sessions.set_session_mode(session_id, body.mode)
    result = _plan_envelope(info.session_id)
    result["info"] = to_jsonable(info)
    return result


@router.post("/api/plan/{session_id}/clear")
async def plan_clear(request: Request, session_id: str) -> dict[str, Any]:
    services = request.app.state.aether_services
    record = services.sessions.resolve_record(session_id)
    clear_plan(record.session_id)
    info = services.sessions.set_session_mode(record.session_id, "agent")
    result = _plan_envelope(info.session_id)
    result["info"] = to_jsonable(info)
    return result


def _plan_envelope(session_id: str) -> dict[str, Any]:
    content = read_plan(session_id)
    return {
        "session_id": session_id,
        "mode": get_mode(session_id),
        "plan_path": str(get_plan_path(session_id)),
        "has_plan": content is not None,
        "plan_content": content,
    }


__all__ = ["router"]
