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

from aether.cli.prefs import load_prefs, save_prefs
from aether.gateway.dispatcher import method
from aether.gateway.protocol import ERROR_INVALID_PARAMS, GatewayError


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
    prefs = load_prefs()

    if key == _LAST_MODEL_PREFIX:
        return {"value": dict(prefs.last_model_by_provider)}
    if key.startswith(_LAST_MODEL_DOT):
        provider = key[len(_LAST_MODEL_DOT):]
        return {"value": prefs.last_model_by_provider.get(provider)}
    if key == "version":
        return {"value": prefs.version}

    # Forward-compat: any other key is read out of the ``unknown``
    # bucket if it exists, else returns null.
    return {"value": prefs.unknown.get(key)}


def prefs_set(params: dict[str, Any] | None) -> dict[str, Any]:
    key = _require_key(params, where="prefs.set")
    value = (params or {}).get("value")
    prefs = load_prefs()

    if key == _LAST_MODEL_PREFIX:
        if not isinstance(value, dict):
            raise GatewayError(
                f"prefs.set('{_LAST_MODEL_PREFIX}') requires dict value",
                code=ERROR_INVALID_PARAMS,
            )
        prefs.last_model_by_provider = {
            str(k): str(v) for k, v in value.items() if v
        }
    elif key.startswith(_LAST_MODEL_DOT):
        provider = key[len(_LAST_MODEL_DOT):]
        if not provider:
            raise GatewayError(
                f"prefs.set requires provider in '{_LAST_MODEL_DOT}<provider>'",
                code=ERROR_INVALID_PARAMS,
            )
        if value is None or value == "":
            prefs.last_model_by_provider.pop(provider, None)
        else:
            prefs.last_model_by_provider[provider] = str(value)
    elif key == "version":
        # Version is managed by the store; reject mutation.
        raise GatewayError(
            "prefs.set cannot mutate 'version'",
            code=ERROR_INVALID_PARAMS,
        )
    else:
        if value is None:
            prefs.unknown.pop(key, None)
        else:
            prefs.unknown[key] = value

    save_prefs(prefs)
    return {"ok": True}


def prefs_all(_params: dict[str, Any] | None) -> dict[str, Any]:
    prefs = load_prefs()
    body: dict[str, Any] = {
        "last_model_by_provider": dict(prefs.last_model_by_provider),
        "version": prefs.version,
    }
    body.update(prefs.unknown)
    return {"prefs": body}


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
