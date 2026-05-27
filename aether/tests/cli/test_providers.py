from __future__ import annotations

from aether.cli.providers import (
    _prefer_configured_openai_api_root,
    build_provider,
)
from aether.models.provider.openai_compatible import OpenAICompatibleModel


def test_prefers_env_api_root_for_legacy_session_base_url() -> None:
    assert (
        _prefer_configured_openai_api_root(
            "http://gateway.local:8317",
            environ={"OPENAI_BASE_URL": "http://gateway.local:8317/v1"},
        )
        == "http://gateway.local:8317/v1"
    )


def test_keeps_explicit_base_url_when_env_points_elsewhere() -> None:
    assert (
        _prefer_configured_openai_api_root(
            "https://api.deepseek.com",
            environ={"OPENAI_BASE_URL": "http://gateway.local:8317/v1"},
        )
        == "https://api.deepseek.com"
    )


def test_build_provider_uses_env_api_root_for_legacy_session_base_url(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gateway.local:8317/v1")

    provider = build_provider(
        "openai",
        model="gpt-5.4",
        base_url="http://gateway.local:8317",
    )

    assert isinstance(provider, OpenAICompatibleModel)
    assert provider.base_url == "http://gateway.local:8317/v1"
