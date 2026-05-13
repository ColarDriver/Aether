"""Client responses to server-initiated reverse-RPC requests."""

from __future__ import annotations

from typing import Any

from aether.gateway import reverse_rpc
from aether.gateway.dispatcher import method
from aether.gateway.protocol import ERROR_INVALID_PARAMS, GatewayError


def approval_response(params: dict[str, Any] | None) -> dict[str, Any]:
    request_id = _require_response_id(params, where="approval.response")
    if params and "error" in params:
        _fail_or_raise(request_id, params.get("error"))
    else:
        payload = _payload_without_id(params)
        _complete_or_raise(request_id, payload)
    return {"ok": True}


def permission_response(params: dict[str, Any] | None) -> dict[str, Any]:
    request_id = _require_response_id(params, where="permission.response")
    if params and "error" in params:
        _fail_or_raise(request_id, params.get("error"))
    else:
        payload = _payload_without_id(params)
        _complete_or_raise(request_id, payload)
    return {"ok": True}


def _require_response_id(params: dict[str, Any] | None, *, where: str) -> str:
    if params is None:
        raise GatewayError(f"{where} requires params", code=ERROR_INVALID_PARAMS)
    request_id = params.get("id")
    if request_id is None:
        # Reverse response envelopes are parsed into RpcRequest(id=srv_...),
        # so explicit method callers may omit params.id and let the dispatcher
        # pass the id via current_request_id().
        from aether.gateway.dispatcher import current_request_id

        request_id = current_request_id()
    if not isinstance(request_id, str) or not request_id.startswith("srv_"):
        raise GatewayError(
            f"{where} requires server request id",
            code=ERROR_INVALID_PARAMS,
            data={"reason": "missing_or_invalid_id"},
        )
    return request_id


def _payload_without_id(params: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(params or {})
    payload.pop("id", None)
    return payload


def _complete_or_raise(request_id: str, payload: dict[str, Any]) -> None:
    try:
        reverse_rpc.complete(request_id, payload)
    except reverse_rpc.UnknownPendingResponse as exc:
        raise GatewayError(
            "unknown_pending",
            code=ERROR_INVALID_PARAMS,
            data={"id": request_id},
        ) from exc


def _fail_or_raise(request_id: str, error: Any) -> None:
    message = "reverse RPC error response"
    if isinstance(error, dict) and error.get("message"):
        message = str(error["message"])
    try:
        reverse_rpc.fail(request_id, RuntimeError(message))
    except reverse_rpc.UnknownPendingResponse as exc:
        raise GatewayError(
            "unknown_pending",
            code=ERROR_INVALID_PARAMS,
            data={"id": request_id},
        ) from exc


def register() -> None:
    method("approval.response", long=False)(approval_response)
    method("permission.response", long=False)(permission_response)


__all__ = [
    "approval_response",
    "permission_response",
    "register",
]
