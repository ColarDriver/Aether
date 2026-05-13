"""``providers.*`` RPC methods.

The provider catalog is intentionally static: providers without a
live ``/v1/models`` endpoint advertise a stable list of canonical
model IDs so the TS UI's ``/model`` picker can render something
without requiring credentials.  Future PRs may switch select
providers to live discovery (e.g. ``OpenAICompatibleModel.list_models``),
but PR 3 keeps the static path for predictability.
"""

from __future__ import annotations

from typing import Any

from aether.cli.providers import list_providers, resolve_provider_name
from aether.gateway.dispatcher import method
from aether.gateway.handlers.schemas import ModelInfo, ProviderInfo
from aether.gateway.protocol import (
    ERROR_APPLICATION,
    ERROR_INVALID_PARAMS,
    GatewayError,
)


# ── Display + credential metadata for every supported provider ──────


_PROVIDER_DISPLAY: dict[str, dict[str, Any]] = {
    "claude": {
        "display_name": "Anthropic Claude",
        "requires_api_key": True,
        "default_base_url": None,
    },
    "openai": {
        "display_name": "OpenAI-compatible",
        "requires_api_key": True,
        "default_base_url": "https://api.openai.com/v1",
    },
    "codex": {
        "display_name": "OpenAI Codex",
        "requires_api_key": True,
        "default_base_url": None,
    },
}


# ── Static model catalog ────────────────────────────────────────────
# Keep these IDs aligned with the model strings referenced in
# `aether/cli/providers.py:_DEFAULTS` and the wider codebase.  When a
# new canonical Claude / OpenAI / Codex model lands, add it here so
# the TS picker discovers it.


_MODEL_CATALOG: dict[str, list[ModelInfo]] = {
    "claude": [
        ModelInfo(id="claude-opus-4-7", display_name="Claude Opus 4.7", context_window=200_000),
        ModelInfo(id="claude-sonnet-4-6", display_name="Claude Sonnet 4.6", context_window=200_000),
        ModelInfo(
            id="claude-haiku-4-5-20251001",
            display_name="Claude Haiku 4.5",
            context_window=200_000,
        ),
    ],
    "openai": [
        ModelInfo(id="gpt-4o", display_name="GPT-4o", context_window=128_000),
        ModelInfo(id="gpt-4-turbo", display_name="GPT-4 Turbo", context_window=128_000),
        ModelInfo(id="gpt-4.1", display_name="GPT-4.1", context_window=128_000),
        ModelInfo(id="gpt-5", display_name="GPT-5", context_window=128_000),
    ],
    "codex": [
        ModelInfo(id="gpt-5.4", display_name="Codex GPT-5.4", context_window=128_000),
    ],
}


def providers_list(_params: dict[str, Any] | None) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for name in list_providers():
        meta = _PROVIDER_DISPLAY.get(name, {})
        info = ProviderInfo(
            name=name,
            display_name=meta.get("display_name", name),
            requires_api_key=bool(meta.get("requires_api_key", True)),
            default_base_url=meta.get("default_base_url"),
        )
        items.append(info.model_dump(mode="json", exclude_none=True))
    return {"providers": items}


def providers_models(params: dict[str, Any] | None) -> dict[str, Any]:
    if not params or not isinstance(params.get("provider"), str) or not params["provider"].strip():
        raise GatewayError(
            "providers.models requires non-empty string 'provider'",
            code=ERROR_INVALID_PARAMS,
        )
    raw_name = params["provider"].strip()
    resolved = resolve_provider_name(raw_name)
    catalog = _MODEL_CATALOG.get(resolved)
    if catalog is None:
        raise GatewayError(
            f"unknown provider: {raw_name}",
            code=ERROR_APPLICATION,
            data={"provider": raw_name, "resolved": resolved},
        )
    return {
        "models": [m.model_dump(mode="json", exclude_none=True) for m in catalog],
    }


def register() -> None:
    """Register ``providers.*`` handlers on the dispatcher.  Idempotent."""
    method("providers.list", long=False)(providers_list)
    method("providers.models", long=True)(providers_models)


__all__ = [
    "providers_list",
    "providers_models",
    "register",
]
