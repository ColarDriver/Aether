from __future__ import annotations

from aether.services.diagnostics import DiagnosticsService
from aether.services.health import HealthService
from aether.services.providers import AuthService


def test_health_reports_runtime_and_missing_provider_readiness() -> None:
    health = HealthService(
        auth=AuthService(environ={}),
        diagnostics=DiagnosticsService(),
    ).status(provider="openai")

    assert health.runtime.python_version
    assert health.status == "degraded"
    assert health.provider is not None
    assert health.provider.provider == "openai"
    assert health.provider.credentials[0].configured is False
    assert any(service.name == "diagnostics" for service in health.services)


def test_health_output_is_public_safe_when_credentials_exist() -> None:
    health = HealthService(
        auth=AuthService(environ={"OPENAI_API_KEY": "sk-secret"}),
        diagnostics=DiagnosticsService(),
    ).status(provider="openai")

    assert health.status == "ok"
    assert health.provider is not None
    assert health.provider.credentials[0].configured is True
    assert "sk-secret" not in repr(health)
