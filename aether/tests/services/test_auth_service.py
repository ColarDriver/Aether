from __future__ import annotations

from aether.services.providers import AuthService


def test_auth_service_reports_missing_and_redacted_credentials() -> None:
    missing = AuthService(environ={}).credentials_status(provider="openai")
    configured = AuthService(environ={"OPENAI_API_KEY": "sk-secret"}).credentials_status(provider="openai")

    assert missing.provider == "openai"
    assert missing.credentials[0].configured is False
    assert configured.credentials[0].configured is True
    assert configured.credentials[0].redacted
    assert "sk-secret" not in repr(configured)
