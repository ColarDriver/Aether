"""``providers.*`` RPC methods backed by provider services."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from aether.gateway.dispatcher import method
from aether.gateway.handlers.schemas import ModelInfo, ProviderInfo
from aether.gateway.handlers.service_errors import service_error_to_gateway
from aether.gateway.handlers.state import get_current_session
from aether.gateway.protocol import (
    ERROR_APPLICATION,
    ERROR_INVALID_PARAMS,
    GatewayError,
)
from aether.services.common import ServiceError, ServiceValidationError
from aether.services.providers import (
    AuthService,
    ModelSummary,
    ProviderService,
    ProviderSummary,
    candidate_urls,
    extract_model_ids,
    suggest_base_url,
)


def providers_list(_params: dict[str, Any] | None) -> dict[str, Any]:
    try:
        providers = ProviderService().list_providers()
    except ServiceError as exc:
        raise service_error_to_gateway(exc) from exc
    return {"providers": [_provider_to_wire(provider) for provider in providers]}


def providers_models(params: dict[str, Any] | None) -> dict[str, Any]:
    if not params or not isinstance(params.get("provider"), str) or not params["provider"].strip():
        raise GatewayError(
            "providers.models requires non-empty string 'provider'",
            code=ERROR_INVALID_PARAMS,
        )
    provider = params["provider"].strip()
    base_url = params.get("base_url")
    if base_url is not None and not isinstance(base_url, str):
        raise GatewayError(
            "providers.models optional 'base_url' must be a string",
            code=ERROR_INVALID_PARAMS,
        )
    try:
        result = ProviderService(current_session_getter=get_current_session).list_models(
            provider,
            base_url=base_url,
            current_session_id=get_current_session(),
        )
    except ServiceValidationError as exc:
        if "unknown provider" in exc.message:
            raise GatewayError(
                exc.message,
                code=ERROR_APPLICATION,
                data=exc.details or None,
            ) from exc
        raise service_error_to_gateway(exc) from exc
    except ServiceError as exc:
        raise service_error_to_gateway(exc) from exc
    return {
        "models": [_model_to_wire(model) for model in result.models],
        "discovery": result.discovery.to_dict(),
    }


def provider_runtime_current(params: dict[str, Any] | None) -> dict[str, Any]:
    provider, model, base_url = _runtime_params(params, method_name="provider.runtime_current")
    try:
        runtime = ProviderService().runtime_current(
            provider=provider,
            model=model,
            base_url=base_url,
        )
    except ServiceError as exc:
        raise service_error_to_gateway(exc) from exc
    payload: dict[str, Any] = {
        "family": runtime.family,
        "provider_name": runtime.provider_name,
        "model": runtime.model,
        "source": runtime.source,
    }
    if runtime.base_url is not None:
        payload["base_url"] = runtime.base_url
    payload.update(runtime.extra)
    if runtime.credential is not None:
        credential = _credential_to_wire(runtime.credential)
        if not runtime.credential.configured:
            credential["names"] = list(runtime.api_key_env_names)
        payload["credential"] = credential
    return payload


def provider_credentials_status(params: dict[str, Any] | None) -> dict[str, Any]:
    provider, model, base_url = _runtime_params(params, method_name="provider.credentials_status")
    try:
        status = AuthService().credentials_status(
            provider=provider,
            model=model,
            base_url=base_url,
        )
    except ServiceError as exc:
        raise service_error_to_gateway(exc) from exc
    return {
        "family": status.family,
        "provider": status.provider,
        "credentials": [_credential_to_wire(credential) for credential in status.credentials],
    }


def provider_auxiliary_slots(params: dict[str, Any] | None) -> dict[str, Any]:
    requested = None
    if params and "slots" in params:
        raw_slots = params.get("slots")
        if not isinstance(raw_slots, list) or not all(isinstance(item, str) for item in raw_slots):
            raise GatewayError(
                "provider.auxiliary_slots optional 'slots' must be a list of strings",
                code=ERROR_INVALID_PARAMS,
            )
        requested = [item for item in raw_slots if item.strip()]
    try:
        slots = ProviderService().auxiliary_slots(requested)
    except ServiceError as exc:
        raise service_error_to_gateway(exc) from exc
    return {"slots": [asdict(slot) for slot in slots]}


def _provider_to_wire(provider: ProviderSummary) -> dict[str, Any]:
    return ProviderInfo(
        name=provider.name,
        display_name=provider.display_name,
        requires_api_key=provider.requires_api_key,
        default_base_url=provider.default_base_url,
    ).model_dump(mode="json", exclude_none=True)


def _model_to_wire(model: ModelSummary) -> dict[str, Any]:
    return ModelInfo(
        id=model.id,
        display_name=model.display_name,
        context_window=model.context_window,
    ).model_dump(mode="json", exclude_none=True)


def _credential_to_wire(credential: Any) -> dict[str, Any]:
    return {
        "source": credential.source,
        "name": credential.name,
        "configured": credential.configured,
        "redacted": credential.redacted,
    }


def _runtime_params(
    params: dict[str, Any] | None,
    *,
    method_name: str,
) -> tuple[str | None, str | None, str | None]:
    provider = None
    model = None
    base_url = None
    if params:
        for key in ("provider", "model", "base_url"):
            value = params.get(key)
            if value is not None and not isinstance(value, str):
                raise GatewayError(
                    f"{method_name} optional '{key}' must be a string",
                    code=ERROR_INVALID_PARAMS,
                )
        provider = params.get("provider")
        model = params.get("model")
        base_url = params.get("base_url")
    return provider, model, base_url


_candidate_urls = candidate_urls
_extract_model_ids = extract_model_ids
_suggest_base_url = suggest_base_url


def register() -> None:
    method("providers.list", long=False)(providers_list)
    method("providers.models", long=True)(providers_models)
    method("provider.runtime_current", long=False)(provider_runtime_current)
    method("provider.credentials_status", long=False)(provider_credentials_status)
    method("provider.auxiliary_slots", long=False)(provider_auxiliary_slots)


__all__ = [
    "_candidate_urls",
    "_extract_model_ids",
    "_suggest_base_url",
    "provider_auxiliary_slots",
    "provider_credentials_status",
    "provider_runtime_current",
    "providers_list",
    "providers_models",
    "register",
]
