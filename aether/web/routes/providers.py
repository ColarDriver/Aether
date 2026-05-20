"""Provider and model routes."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from aether.services.providers import ProviderSelectionRequest
from aether.web.serializers import to_jsonable

router = APIRouter()


class ModelSelectBody(BaseModel):
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    persist_last_model: bool = False


@router.get("/api/providers")
async def providers(request: Request) -> dict[str, object]:
    services = request.app.state.aether_services
    return {"providers": to_jsonable(services.providers.list_providers())}


@router.get("/api/providers/current")
async def provider_current(
    request: Request,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.providers.runtime_current(
            provider=provider,
            model=model,
            base_url=base_url,
        )
    )


@router.get("/api/providers/{provider}/models")
async def provider_models(
    request: Request,
    provider: str,
    base_url: str | None = None,
) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(services.providers.list_models(provider, base_url=base_url))


@router.post("/api/model/select")
async def select_model(request: Request, body: ModelSelectBody) -> dict[str, object]:
    services = request.app.state.aether_services
    return to_jsonable(
        services.model_selection.select(
            ProviderSelectionRequest(
                provider=body.provider,
                model=body.model,
                base_url=body.base_url,
                persist_last_model=body.persist_last_model,
            )
        )
    )


@router.get("/api/model/auxiliary")
async def auxiliary_models(
    request: Request,
    slots: list[str] | None = Query(default=None),
) -> dict[str, object]:
    services = request.app.state.aether_services
    return {"slots": to_jsonable(services.providers.auxiliary_slots(slots=slots))}


__all__ = ["router"]
