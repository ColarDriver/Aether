"""``prefs.*`` RPC methods.

Thin wrappers around :mod:`aether.cli.prefs`.  Supports a small "nested
key" syntax so callers can read or update the ``last_model_by_provider``
sub-dict without round-tripping the whole prefs blob:

* ``prefs.get {"key": "last_model_by_provider.claude"}``
  → ``{"value": "claude-sonnet-4-6"}``
* ``prefs.set {"key": "last_model_by_provider.claude", "value": "claude-haiku-4-5-20251001"}``
  → updates that one slot atomically

Any other ``key`` round-trips through the ``unknown`` round-trip bucket
that :class:`~aether.cli.prefs.Prefs` already preserves — useful for
forward-compat values the TS client wants to set without the Python
side caring.
"""

from __future__ import annotations

from typing import Any

from aether.gateway.dispatcher import method
from aether.gateway.handlers.service_errors import service_error_to_gateway
from aether.gateway.protocol import ERROR_INVALID_PARAMS, GatewayError
from aether.services.common import ServiceError
from aether.services.config import PrefsService


_LAST_MODEL_PREFIX = "last_model_by_provider"
_LAST_MODEL_DOT = f"{_LAST_MODEL_PREFIX}."


def _require_key(params: dict[str, Any] | None, *, where: str) -> str:
    if not params or not isinstance(params.get("key"), str) or not params["key"].strip():
        raise GatewayError(
            f"{where} requires non-empty string 'key'",
            code=ERROR_INVALID_PARAMS,
        )
    return params["key"].strip()


def prefs_get(params: dict[str, Any] | None) -> dict[str, Any]:
    key = _require_key(params, where="prefs.get")
    try:
        return {"value": PrefsService().get(key)}
    except ServiceError as exc:
        raise service_error_to_gateway(exc) from exc


def prefs_set(params: dict[str, Any] | None) -> dict[str, Any]:
    key = _require_key(params, where="prefs.set")
    value = (params or {}).get("value")
    try:
        PrefsService().set(key, value)
    except ServiceError as exc:
        raise service_error_to_gateway(exc) from exc
    return {"ok": True}


def prefs_all(_params: dict[str, Any] | None) -> dict[str, Any]:
    return {"prefs": PrefsService().all()}


def register() -> None:
    """Register ``prefs.*`` handlers on the dispatcher.  Idempotent."""
    method("prefs.get", long=False)(prefs_get)
    method("prefs.set", long=False)(prefs_set)
    method("prefs.all", long=False)(prefs_all)


__all__ = [
    "prefs_all",
    "prefs_get",
    "prefs_set",
    "register",
]
