"""Provider factory for the Aether CLI."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from aether.config.provider_runtime import (
    provider_choice,
    resolve_main_provider_runtime,
)
from aether.models.provider.base import ModelProvider
from aether.runtime.credentials import default_credential_lookup


_DEFAULTS: dict[str, dict[str, Any]] = {
    "claude": {"model": "claude-sonnet-4-6", "max_tokens": 16384},
    "openai": {"model": "gpt-5.4", "base_url": "https://api.openai.com/v1"},
    "codex": {"model": "gpt-5.4", "reasoning_effort": "medium"},
}

PROVIDER_ALIASES: dict[str, str] = {
    "anthropic": "claude",
    "claude-code": "claude",
    "openai-compatible": "openai",
}


def resolve_provider_name(name: str) -> str:
    return PROVIDER_ALIASES.get(name.lower(), name.lower())


def build_provider(
    provider: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> ModelProvider:
    """Instantiate a ModelProvider from a provider name and optional overrides.

    Supported providers: claude, openai, codex
    Falls back to environment variables for credentials when not supplied.
    """
    runtime = resolve_main_provider_runtime(
        provider=provider,
        model=model,
        base_url=base_url,
    )
    credential_lookup = default_credential_lookup()
    name = runtime.provider_name

    if name == "claude":
        from aether.models.provider.claude import ClaudeChatModel

        d = _DEFAULTS["claude"]
        return ClaudeChatModel(
            model=runtime.model or d["model"],
            max_tokens=int(kwargs.pop("max_tokens", d["max_tokens"])),
            anthropic_api_key=api_key
            or _credential_value(credential_lookup.get_first(runtime.api_key_env_names)),
            **kwargs,
        )

    if name == "openai":
        from aether.models.provider.openai_compatible import OpenAICompatibleModel

        d = _DEFAULTS["openai"]
        resolved_key = api_key or _credential_value(
            credential_lookup.get_first(runtime.api_key_env_names)
        ) or ""
        if not resolved_key:
            raise ValueError(
                "OpenAI provider requires an API key. "
                "Set OPENAI_API_KEY or ANTHROPIC_AUTH_TOKEN or pass --api-key."
            )
        resolved_url = _prefer_configured_openai_api_root(runtime.base_url or d["base_url"])
        return OpenAICompatibleModel(
            model=runtime.model or d["model"],
            api_key=resolved_key,
            base_url=resolved_url,
            **kwargs,
        )

    if name == "codex":
        from aether.models.provider.codex import CodexChatModel

        d = _DEFAULTS["codex"]
        return CodexChatModel(
            model=runtime.model or d["model"],
            reasoning_effort=str(kwargs.pop("reasoning_effort", d["reasoning_effort"])),
            access_token=api_key
            or _credential_value(credential_lookup.get_first(runtime.api_key_env_names)),
            **kwargs,
        )

    raise ValueError(
        f"Unknown provider: {provider!r}. "
        f"Supported: claude, openai, codex (aliases: {', '.join(PROVIDER_ALIASES)})."
    )


def list_providers() -> list[str]:
    return list(_DEFAULTS.keys())


def get_provider_defaults(name: str) -> dict[str, Any]:
    """Return a copy of provider defaults for display and diagnostics."""
    choice = provider_choice(name)
    defaults = dict(_DEFAULTS.get(choice.provider_name, {}))
    defaults["model"] = choice.default_model
    return defaults


def _credential_value(value: object) -> str | None:
    raw = getattr(value, "value", None)
    return raw if isinstance(raw, str) and raw else None


def _prefer_configured_openai_api_root(
    base_url: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Use the configured API root when an older session stored only the host root.

    Session records persist ``base_url`` so old conversations can keep their
    provider settings.  Some earlier web flows saved the gateway root
    (``http://host:port``) while the OpenAI-compatible chat transport expects
    an API root (``http://host:port/v1``).  If the environment now points at
    the same root plus a conventional API suffix, prefer that env value for the
    run without mutating historical session files.
    """

    normalized = base_url.rstrip("/")
    env = os.environ if environ is None else environ
    for key in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
        configured = env.get(key)
        if not isinstance(configured, str) or not configured.strip():
            continue
        configured = configured.strip().rstrip("/")
        if configured == normalized:
            return configured
        if not configured.startswith(f"{normalized}/"):
            continue
        suffix = configured[len(normalized) :]
        if suffix in {"/v1", "/api", "/api/v1"} or suffix.endswith("/v1"):
            return configured
    return normalized
