"""Health service implementation."""

from __future__ import annotations

import platform
import sys

from aether.services.diagnostics import DiagnosticsService, DiagnosticsStatus
from aether.services.health.contracts import HealthStatus, RuntimeStatus, ServiceStatus
from aether.services.providers import AuthService, CredentialSetStatus


class HealthService:
    """Aggregate public-safe runtime readiness."""

    def __init__(
        self,
        *,
        auth: AuthService | None = None,
        diagnostics: DiagnosticsService | None = None,
    ) -> None:
        self._auth = auth or AuthService()
        self._diagnostics = diagnostics or DiagnosticsService()

    def status(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> HealthStatus:
        services: list[ServiceStatus] = [
            ServiceStatus(name="services", available=True),
        ]
        provider_status = self._provider_status(
            services,
            provider=provider,
            model=model,
            base_url=base_url,
        )
        diagnostics_status = self._diagnostics_status(services)
        overall = "ok" if all(item.available for item in services) else "degraded"
        return HealthStatus(
            status=overall,
            runtime=RuntimeStatus(
                python_version=sys.version.split()[0],
                platform=platform.platform(),
                implementation=platform.python_implementation(),
            ),
            services=services,
            provider=provider_status,
            diagnostics=diagnostics_status,
        )

    def _provider_status(
        self,
        services: list[ServiceStatus],
        *,
        provider: str | None,
        model: str | None,
        base_url: str | None,
    ) -> CredentialSetStatus | None:
        try:
            credentials = self._auth.credentials_status(
                provider=provider,
                model=model,
                base_url=base_url,
            )
        except Exception as exc:  # noqa: BLE001
            services.append(
                ServiceStatus(
                    name="provider_auth",
                    available=False,
                    status="error",
                    detail=str(exc) or type(exc).__name__,
                )
            )
            return None
        ready = any(item.configured for item in credentials.credentials)
        services.append(
            ServiceStatus(
                name="provider_auth",
                available=ready,
                status="ok" if ready else "missing_credentials",
                detail=None if ready else "one or more provider credentials are not configured",
            )
        )
        return credentials

    def _diagnostics_status(self, services: list[ServiceStatus]) -> DiagnosticsStatus:
        status = self._diagnostics.status()
        services.append(
            ServiceStatus(
                name="diagnostics",
                available=True,
                status="enabled" if status.enabled else "disabled",
            )
        )
        return status


__all__ = ["HealthService"]
