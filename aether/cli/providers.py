"""Provider factory for the Aether CLI."""

from __future__ import annotations

import os
from typing import Any

from aether.models.provider.base import ModelProvider


_DEFAULTS: dict[str, dict[str, Any]] = {
    "claude": {"model": "claude-sonnet-4-6", "max_tokens": 16384},
    "openai": {"model": "claude-sonnet-4-6", "base_url": "https://api.openai.com/v1"},
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
    name = resolve_provider_name(provider)

    if name == "claude":
        from aether.models.provider.claude import ClaudeChatModel

        d = _DEFAULTS["claude"]
        return ClaudeChatModel(
            model=model or d["model"],
            max_tokens=int(kwargs.pop("max_tokens", d["max_tokens"])),
            anthropic_api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            **kwargs,
        )

    if name == "openai":
        from aether.models.provider.openai_compatible import OpenAICompatibleModel

        d = _DEFAULTS["openai"]
        resolved_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_AUTH_TOKEN")
            or ""
        )
        if not resolved_key:
            raise ValueError(
                "OpenAI provider requires an API key. "
                "Set OPENAI_API_KEY or ANTHROPIC_AUTH_TOKEN or pass --api-key."
            )
        resolved_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("ANTHROPIC_BASE_URL")
            or d["base_url"]
        )
        return OpenAICompatibleModel(
            model=model or os.getenv("AETHER_MODEL") or os.getenv("ANTHROPIC_MODEL", d["model"]),
            api_key=resolved_key,
            base_url=resolved_url,
            **kwargs,
        )

    if name == "codex":
        from aether.models.provider.codex import CodexChatModel

        d = _DEFAULTS["codex"]
        return CodexChatModel(
            model=model or d["model"],
            reasoning_effort=str(kwargs.pop("reasoning_effort", d["reasoning_effort"])),
            access_token=api_key or os.getenv("CODEX_ACCESS_TOKEN"),
            **kwargs,
        )

    raise ValueError(
        f"Unknown provider: {provider!r}. "
        f"Supported: claude, openai, codex (aliases: {', '.join(PROVIDER_ALIASES)})."
    )


def list_providers() -> list[str]:
    return list(_DEFAULTS.keys())
