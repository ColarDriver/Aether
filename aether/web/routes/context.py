"""Context compression routes."""

from __future__ import annotations

from typing import Any

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
    attachments = _enrich_workspace_reference_attachments(
        body.attachments,
        getattr(services, "workspace", None),
    )
    return to_jsonable(
        services.context.estimate(
            ContextEstimateRequest(
                session_id=session_id,
                draft=body.draft,
                attachments=attachments,
            )
        )
    )


def _enrich_workspace_reference_attachments(
    attachments: list[dict[str, object]],
    workspace: Any,
) -> list[dict[str, Any]]:
    if workspace is None or not attachments:
        return [dict(attachment) for attachment in attachments]
    enriched: list[dict[str, Any]] = []
    for attachment in attachments:
        if attachment.get("note") != "workspace reference":
            enriched.append(dict(attachment))
            continue
        path = attachment.get("path")
        if not isinstance(path, str) or not path or attachment.get("isDirectory") is True:
            enriched.append(dict(attachment))
            continue
        next_attachment: dict[str, Any] = dict(attachment)
        try:
            workspace_file = workspace.read_file(path)
        except Exception as exc:  # noqa: BLE001 - context estimates should expose misses, not fail the whole request
            next_attachment["_llm_error"] = str(exc) or type(exc).__name__
        else:
            next_attachment.setdefault("name", workspace_file.name)
            next_attachment["path"] = workspace_file.path
            next_attachment["_llm_language"] = workspace_file.language
            next_attachment["_llm_truncated"] = bool(workspace_file.truncated)
            next_attachment["_llm_binary"] = bool(workspace_file.binary)
            if not workspace_file.binary:
                next_attachment["_llm_content"] = workspace_file.content
        enriched.append(next_attachment)
    return enriched


__all__ = ["router"]
