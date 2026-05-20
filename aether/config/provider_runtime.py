"""Provider runtime configuration resolution.

This module centralises the mapping between user-facing provider families
(``AETHER_PROVIDER``) and the internal provider names used by the provider
factory.  It intentionally does not own credential lookup yet; Sprint 17.2
adds the credential source abstraction on top of the env names exposed here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from aether.runtime.credentials.redaction import redact_mapping


ProviderFamily = Literal["codex", "claude", "openai-compatible"]


@dataclass(frozen=True, slots=True)
class ProviderRuntimeChoice:
    """Canonical provider family mapping."""

    family: ProviderFamily
    provider_name: str
    default_model: str


@dataclass(frozen=True, slots=True)
class ProviderRuntimeConfig:
    """Resolved runtime provider/model configuration.

    Raw credential values are deliberately absent.  ``api_key_env_names`` names
    the env variables that a later credential source may inspect.
    """

    family: ProviderFamily
    provider_name: str
    model: str
    base_url: str | None = None
    api_key_env_names: tuple[str, ...] = ()
    model_env_names: tuple[str, ...] = ()
    base_url_env_names: tuple[str, ...] = ()
    source: str = "default"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def public_metadata(self) -> dict[str, Any]:
        """Return a status-safe snapshot without raw credential values."""

        return redact_mapping(
            {
                "family": self.family,
                "provider_name": self.provider_name,
                "model": self.model,
                "base_url": self.base_url,
                "api_key_env_names": self.api_key_env_names,
                "model_env_names": self.model_env_names,
                "base_url_env_names": self.base_url_env_names,
                "source": self.source,
                "extra": dict(self.extra),
            }
        )


_CHOICES: dict[ProviderFamily, ProviderRuntimeChoice] = {
    "codex": ProviderRuntimeChoice(
        family="codex",
        provider_name="codex",
        default_model="gpt-5.4",
    ),
    "claude": ProviderRuntimeChoice(
        family="claude",
        provider_name="claude",
        default_model="claude-sonnet-4-6",
    ),
    "openai-compatible": ProviderRuntimeChoice(
        family="openai-compatible",
        provider_name="openai",
        default_model="gpt-5.4",
    ),
}

_ALIASES: dict[str, ProviderFamily] = {
    "codex": "codex",
    "claude": "claude",
    "anthropic": "claude",
    "claude-code": "claude",
    "openai-compatible": "openai-compatible",
    "openai_compatible": "openai-compatible",
    "openai": "openai-compatible",
}

_MODEL_ENV_NAMES: dict[ProviderFamily, tuple[str, ...]] = {
    "codex": ("AETHER_MODEL", "CODEX_MODEL"),
    "claude": ("AETHER_MODEL", "ANTHROPIC_MODEL"),
    "openai-compatible": ("AETHER_MODEL", "OPENAI_MODEL", "ANTHROPIC_MODEL"),
}

_BASE_URL_ENV_NAMES: dict[ProviderFamily, tuple[str, ...]] = {
    "codex": (),
    "claude": (),
    "openai-compatible": ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"),
}

_API_KEY_ENV_NAMES: dict[ProviderFamily, tuple[str, ...]] = {
    "codex": ("CODEX_ACCESS_TOKEN", "CODEX_API_KEY"),
    "claude": ("ANTHROPIC_API_KEY",),
    "openai-compatible": ("OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
}

_DEFAULT_BASE_URLS: dict[ProviderFamily, str | None] = {
    "codex": None,
    "claude": None,
    "openai-compatible": "https://api.openai.com/v1",
}


def normalize_provider_family(value: str | None) -> ProviderFamily | None:
    """Return the canonical provider family for *value*.

    ``None`` and blank strings mean "unset" and return ``None``. Unknown
    non-empty values raise ``ValueError`` with user-facing text.
    """

    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(("codex", "claude", "openai-compatible"))
        raise ValueError(
            f"unknown provider family {value!r}; expected one of: {allowed}"
        ) from exc


def provider_choice(family: str | None) -> ProviderRuntimeChoice:
    canonical = normalize_provider_family(family) or "openai-compatible"
    return _CHOICES[canonical]


def resolve_provider_runtime_from_env(
    *,
    environ: Mapping[str, str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> ProviderRuntimeConfig:
    """Resolve provider runtime config from explicit values and env.

    Explicit arguments win over env. The default family remains
    ``openai-compatible`` to match the existing CLI help text.
    """

    env = environ if environ is not None else _empty_env_or_os()
    raw_provider = _first_non_empty(provider, env.get("AETHER_PROVIDER"))
    family = normalize_provider_family(raw_provider) or "openai-compatible"
    choice = _CHOICES[family]
    model_env_names = _MODEL_ENV_NAMES[family]
    base_url_env_names = _BASE_URL_ENV_NAMES[family]
    api_key_env_names = _API_KEY_ENV_NAMES[family]
    resolved_model = _first_non_empty(
        model,
        *(_env_value(env, key) for key in model_env_names),
        choice.default_model,
    )
    resolved_base_url = _first_non_empty(
        base_url,
        *(_env_value(env, key) for key in base_url_env_names),
        _DEFAULT_BASE_URLS[family],
    )
    source = "explicit" if provider or model or base_url else ("env" if raw_provider else "default")
    return ProviderRuntimeConfig(
        family=family,
        provider_name=choice.provider_name,
        model=resolved_model or choice.default_model,
        base_url=resolved_base_url,
        api_key_env_names=api_key_env_names,
        model_env_names=model_env_names,
        base_url_env_names=base_url_env_names,
        source=source,
    )


def resolve_main_provider_runtime(
    config: Any | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> ProviderRuntimeConfig:
    """Resolve the main provider runtime using config, explicit args, env.

    ``EngineConfig`` does not currently expose provider/model fields, but this
    helper also checks common attribute names so future config-file support can
    route through the same path without another resolver.
    """

    return resolve_provider_runtime_from_env(
        environ=environ,
        provider=_first_non_empty(provider, _config_str(config, "provider"), _config_str(config, "main_provider")),
        model=_first_non_empty(model, _config_str(config, "model"), _config_str(config, "main_model")),
        base_url=_first_non_empty(base_url, _config_str(config, "base_url"), _config_str(config, "main_base_url")),
    )


def provider_api_key_env_names(family_or_provider: str) -> tuple[str, ...]:
    family = normalize_provider_family(family_or_provider) or "openai-compatible"
    return _API_KEY_ENV_NAMES[family]


def provider_model_env_names(family_or_provider: str) -> tuple[str, ...]:
    family = normalize_provider_family(family_or_provider) or "openai-compatible"
    return _MODEL_ENV_NAMES[family]


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _env_value(environ: Mapping[str, str], key: str) -> str | None:
    value = environ.get(key)
    return value if isinstance(value, str) else None


def _config_str(config: Any | None, name: str) -> str | None:
    if config is None:
        return None
    value = getattr(config, name, None)
    return value if isinstance(value, str) else None


def _empty_env_or_os() -> Mapping[str, str]:
    import os

    return os.environ


__all__ = [
    "ProviderFamily",
    "ProviderRuntimeChoice",
    "ProviderRuntimeConfig",
    "normalize_provider_family",
    "provider_api_key_env_names",
    "provider_choice",
    "provider_model_env_names",
    "resolve_main_provider_runtime",
    "resolve_provider_runtime_from_env",
]
