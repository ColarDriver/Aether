"""Preference service implementation."""

from __future__ import annotations

from typing import Any

from aether.cli.prefs import get_last_model, load_prefs, save_prefs, set_last_model
from aether.services.common import ServiceValidationError

_LAST_MODEL_PREFIX = "last_model_by_provider"
_LAST_MODEL_DOT = f"{_LAST_MODEL_PREFIX}."


class PrefsService:
    def get(self, key: str) -> Any:
        key = _require_key(key)
        prefs = load_prefs()
        if key == _LAST_MODEL_PREFIX:
            return dict(prefs.last_model_by_provider)
        if key.startswith(_LAST_MODEL_DOT):
            provider = key[len(_LAST_MODEL_DOT) :]
            return prefs.last_model_by_provider.get(provider)
        if key == "version":
            return prefs.version
        return prefs.unknown.get(key)

    def set(self, key: str, value: Any) -> None:
        key = _require_key(key)
        prefs = load_prefs()
        if key == _LAST_MODEL_PREFIX:
            if not isinstance(value, dict):
                raise ServiceValidationError(
                    f"prefs.set('{_LAST_MODEL_PREFIX}') requires dict value",
                    details={"key": key},
                )
            prefs.last_model_by_provider = {
                str(item_key): str(item_value)
                for item_key, item_value in value.items()
                if item_value
            }
        elif key.startswith(_LAST_MODEL_DOT):
            provider = key[len(_LAST_MODEL_DOT) :]
            if not provider:
                raise ServiceValidationError(
                    f"prefs.set requires provider in '{_LAST_MODEL_DOT}<provider>'",
                    details={"key": key},
                )
            if value is None or value == "":
                prefs.last_model_by_provider.pop(provider, None)
            else:
                prefs.last_model_by_provider[provider] = str(value)
        elif key == "version":
            raise ServiceValidationError(
                "prefs.set cannot mutate 'version'",
                details={"key": key},
            )
        elif value is None:
            prefs.unknown.pop(key, None)
        else:
            prefs.unknown[key] = value
        save_prefs(prefs)

    def delete(self, key: str) -> bool:
        existed = self.get(key) is not None
        self.set(key, None)
        return existed

    def all(self) -> dict[str, Any]:
        prefs = load_prefs()
        body: dict[str, Any] = {
            "last_model_by_provider": dict(prefs.last_model_by_provider),
            "version": prefs.version,
        }
        body.update(prefs.unknown)
        return body

    def get_last_model(self, provider: str) -> str | None:
        return get_last_model(provider)

    def set_last_model(self, provider: str, model: str) -> None:
        set_last_model(provider, model)


def _require_key(key: str) -> str:
    if not isinstance(key, str) or not key.strip():
        raise ServiceValidationError(
            "prefs requires non-empty string 'key'",
            details={"field": "key"},
        )
    return key.strip()


__all__ = ["PrefsService"]
