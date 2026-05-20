"""``session.*`` RPC methods backed by SessionService."""

from __future__ import annotations

from typing import Any

from aether.gateway.dispatcher import method
from aether.gateway.handlers.service_errors import service_error_to_gateway
from aether.gateway.handlers.state import (
    get_current_session,
    set_current_session,
)
from aether.gateway.protocol import ERROR_INVALID_PARAMS, GatewayError
from aether.services.common import ServiceError
from aether.services.sessions import (
    SessionCreateRequest,
    SessionDeleteRequest,
    SessionResumeRequest,
    SessionService,
    SessionUpdateRequest,
    extract_tool_calls,
    iso_to_epoch,
    session_info_to_dict,
    transcript_message_to_dict,
)


def _service() -> SessionService:
    return SessionService(
        current_getter=get_current_session,
        current_setter=set_current_session,
    )


def _to_info(info: Any) -> dict[str, Any]:
    return session_info_to_dict(info)


def _to_transcript(message: Any) -> dict[str, Any]:
    return transcript_message_to_dict(message)


def _extract_tool_calls(msg: dict[str, Any]) -> list[Any]:
    return extract_tool_calls(msg)


def _iso_to_epoch(iso: str) -> float:
    return iso_to_epoch(iso)


def _require_str(params: dict[str, Any] | None, key: str, *, where: str) -> str:
    if not params or not isinstance(params.get(key), str) or not params[key].strip():
        raise GatewayError(
            f"{where} requires non-empty string '{key}'",
            code=ERROR_INVALID_PARAMS,
        )
    return params[key].strip()


def session_create(params: dict[str, Any] | None) -> dict[str, Any]:
    provider = _require_str(params, "provider", where="session.create")
    model = _require_str(params, "model", where="session.create")
    params = params or {}

    base_url = params.get("base_url") if isinstance(params.get("base_url"), str) else None
    system_prompt = params.get("system") if isinstance(params.get("system"), str) else None
    requested_session_id = params.get("session_id")
    if requested_session_id is not None:
        if not isinstance(requested_session_id, str) or not requested_session_id.strip():
            raise GatewayError(
                "session.create requires non-empty string 'session_id'",
                code=ERROR_INVALID_PARAMS,
            )
        requested_session_id = requested_session_id.strip()

    try:
        info = _service().create(
            SessionCreateRequest(
                session_id=requested_session_id,
                provider=provider,
                model=model,
                base_url=base_url,
                system_prompt=system_prompt,
            )
        )
    except ServiceError as exc:
        raise service_error_to_gateway(exc) from exc
    return {"session_id": info.session_id, "info": _to_info(info)}


def session_list(params: dict[str, Any] | None) -> dict[str, Any]:
    limit = params.get("limit") if params and isinstance(params.get("limit"), int) else None
    try:
        result = _service().list(limit=limit)
    except ServiceError as exc:
        raise service_error_to_gateway(exc) from exc
    return {"sessions": [_to_info(info) for info in result.sessions]}


def session_resume(params: dict[str, Any] | None) -> dict[str, Any]:
    session_id = _require_str(params, "session_id", where="session.resume")
    try:
        result = _service().resume(SessionResumeRequest(session_id))
    except ServiceError as exc:
        raise service_error_to_gateway(exc) from exc
    return {
        "info": _to_info(result.info),
        "messages": [_to_transcript(message) for message in result.messages],
    }


def session_update(params: dict[str, Any] | None) -> dict[str, Any]:
    session_id = _require_str(params, "session_id", where="session.update")
    params = params or {}
    provider = _optional_non_empty_str(params, "provider", where="session.update") if "provider" in params else None
    model = _optional_non_empty_str(params, "model", where="session.update") if "model" in params else None
    base_url = None
    update_base_url = "base_url" in params
    if update_base_url:
        value = params.get("base_url")
        if value is not None and not isinstance(value, str):
            raise GatewayError(
                "session.update requires string or null 'base_url'",
                code=ERROR_INVALID_PARAMS,
            )
        base_url = value
    system_prompt = None
    update_system_prompt = "system" in params
    if update_system_prompt:
        value = params.get("system")
        if value is not None and not isinstance(value, str):
            raise GatewayError(
                "session.update requires string or null 'system'",
                code=ERROR_INVALID_PARAMS,
            )
        system_prompt = value

    try:
        info = _service().update(
            SessionUpdateRequest(
                session_id=session_id,
                provider=provider,
                model=model,
                base_url=base_url,
                system_prompt=system_prompt,
                update_base_url=update_base_url,
                update_system_prompt=update_system_prompt,
            )
        )
    except ServiceError as exc:
        raise service_error_to_gateway(exc) from exc
    return {"session_id": info.session_id, "info": _to_info(info)}


def session_delete(params: dict[str, Any] | None) -> dict[str, Any]:
    session_id = _require_str(params, "session_id", where="session.delete")
    try:
        deleted = _service().delete(SessionDeleteRequest(session_id))
    except ServiceError as exc:
        raise service_error_to_gateway(exc) from exc
    return {"deleted": bool(deleted)}


def session_current(_params: dict[str, Any] | None) -> dict[str, Any]:
    try:
        current = _service().current()
    except ServiceError as exc:
        raise service_error_to_gateway(exc) from exc
    if current is None:
        return {"session_id": None}
    return {"session_id": current.session_id, "info": _to_info(current.info)}


def _optional_non_empty_str(params: dict[str, Any], key: str, *, where: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GatewayError(
            f"{where} requires non-empty string '{key}'",
            code=ERROR_INVALID_PARAMS,
        )
    return value.strip()


def register() -> None:
    method("session.create", long=False)(session_create)
    method("session.list", long=True)(session_list)
    method("session.resume", long=True)(session_resume)
    method("session.update", long=False)(session_update)
    method("session.delete", long=False)(session_delete)
    method("session.current", long=False)(session_current)


__all__ = [
    "_extract_tool_calls",
    "_iso_to_epoch",
    "_to_info",
    "_to_transcript",
    "register",
    "session_create",
    "session_current",
    "session_delete",
    "session_list",
    "session_resume",
    "session_update",
]
