"""Provider service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ProviderSummary:
    name: str
    display_name: str
    requires_api_key: bool
    default_base_url: str | None = None


@dataclass(frozen=True, slots=True)
class ModelSummary:
    id: str
    display_name: str
    context_window: int | None = None


@dataclass(frozen=True, slots=True)
class ModelDiscoveryStatus:
    kind: str
    source: str | None = None
    reason: str | None = None
    error: str | None = None
    base_url: str | None = None
    base_url_source: str | None = None
    count: int | None = None
    url: str | None = None
    suggested_base_url: str | None = None
    warning: str | None = None
    body_preview: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "source": self.source,
            "reason": self.reason,
            "error": self.error,
            "base_url": self.base_url,
            "base_url_source": self.base_url_source,
            "count": self.count,
            "url": self.url,
            "suggested_base_url": self.suggested_base_url,
            "warning": self.warning,
            "body_preview": self.body_preview,
        }
        payload.update(self.extra)
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True, slots=True)
class ProviderModelList:
    models: list[ModelSummary]
    discovery: ModelDiscoveryStatus


@dataclass(frozen=True, slots=True)
class CredentialStatus:
    source: str
    name: str
    configured: bool
    redacted: str = ""


@dataclass(frozen=True, slots=True)
class CredentialSetStatus:
    family: str
    provider: str
    credentials: list[CredentialStatus]


@dataclass(frozen=True, slots=True)
class ProviderRuntimeStatus:
    family: str
    provider_name: str
    model: str
    base_url: str | None
    api_key_env_names: tuple[str, ...]
    model_env_names: tuple[str, ...]
    base_url_env_names: tuple[str, ...]
    source: str
    credential: CredentialStatus | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProviderPreflightStatus:
    family: str
    provider_name: str
    model: str
    base_url: str | None
    chat_completions_url: str | None
    models_url: str | None
    status: str
    ready: bool
    credential: CredentialStatus | None = None
    discovery: ModelDiscoveryStatus | None = None
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AuxiliarySlotStatus:
    slot: str
    provider_family: str
    provider_name: str
    model: str
    inherited: bool
    source: str


@dataclass(frozen=True, slots=True)
class ProviderSelectionRequest:
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    persist_last_model: bool = False


@dataclass(frozen=True, slots=True)
class ProviderSelectionResult:
    provider: str
    family: str
    model: str
    base_url: str | None
    ready: bool
    missing_credentials: list[str] = field(default_factory=list)
    credential: CredentialStatus | None = None


__all__ = [
    "AuxiliarySlotStatus",
    "CredentialSetStatus",
    "CredentialStatus",
    "ModelDiscoveryStatus",
    "ModelSummary",
    "ProviderModelList",
    "ProviderPreflightStatus",
    "ProviderRuntimeStatus",
    "ProviderSelectionRequest",
    "ProviderSelectionResult",
    "ProviderSummary",
]
