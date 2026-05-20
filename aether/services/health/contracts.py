"""Health service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from aether.services.diagnostics import DiagnosticsStatus
from aether.services.providers import CredentialSetStatus


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    name: str
    available: bool
    status: str = "ok"
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    python_version: str
    platform: str
    implementation: str


@dataclass(frozen=True, slots=True)
class HealthStatus:
    status: str
    runtime: RuntimeStatus
    services: list[ServiceStatus] = field(default_factory=list)
    provider: CredentialSetStatus | None = None
    diagnostics: DiagnosticsStatus | None = None


__all__ = [
    "HealthStatus",
    "RuntimeStatus",
    "ServiceStatus",
]
