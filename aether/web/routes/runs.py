"""Run status and cancellation routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from aether.services.common import ServiceNotFoundError
from aether.services.runs import AgentRunCancelRequest
from aether.web.serializers import to_jsonable

router = APIRouter()


class RunCancelBody(BaseModel):
    run_id: str | None = None
    reason: str | None = None


@router.get("/api/runs/{run_or_session_id}")
async def run_status(request: Request, run_or_session_id: str) -> dict[str, object]:
    services = request.app.state.aether_services
    snapshot = services.runs.status(run_or_session_id)
    if snapshot is None:
        raise ServiceNotFoundError(
            f"run not found: {run_or_session_id}",
            details={"run_or_session_id": run_or_session_id},
        )
    return to_jsonable(snapshot)


@router.post("/api/runs/{session_id}/cancel")
async def cancel_run(
    request: Request,
    session_id: str,
    body: RunCancelBody | None = None,
) -> dict[str, object]:
    services = request.app.state.aether_services
    cancelled = services.runs.cancel(
        AgentRunCancelRequest(
            session_id=session_id,
            run_id=body.run_id if body is not None else None,
            reason=body.reason if body is not None else None,
        )
    )
    return {"cancelled": cancelled}


__all__ = ["router"]
