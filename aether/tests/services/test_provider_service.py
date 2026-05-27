from __future__ import annotations

import httpx
import pytest

from aether.services.common import ServiceValidationError
from aether.services.providers import (
    ProviderService,
    candidate_urls,
    extract_model_ids,
    suggest_base_url,
)


def test_provider_list_and_runtime_status_are_public_safe() -> None:
    service = ProviderService(environ={"OPENAI_API_KEY": "sk-secret"})

    providers = service.list_providers()
    runtime = service.runtime_current(provider="openai", model="gpt-test")

    assert [item.name for item in providers] == ["claude", "openai", "codex"]
    assert providers[1].display_name == "OpenAI-compatible"
    assert runtime.provider_name == "openai"
    assert runtime.model == "gpt-test"
    assert runtime.credential is not None
    assert runtime.credential.configured is True
    assert "sk-secret" not in repr(runtime)


def test_list_models_falls_back_to_static_catalog_without_credentials() -> None:
    service = ProviderService(environ={})

    result = service.list_models("openai")

    assert result.discovery.kind == "static"
    assert result.discovery.reason == "no_credentials"
    assert "gpt-5" in [model.id for model in result.models]


def test_preflight_reports_missing_credentials_before_runs() -> None:
    service = ProviderService(environ={})

    result = service.preflight(provider="openai", model="gpt-5.4")

    assert result.status == "error"
    assert result.ready is False
    assert result.chat_completions_url == "https://api.openai.com/v1/chat/completions"
    assert result.credential is not None
    assert result.credential.configured is False
    assert result.issues == ["Missing provider credential: OPENAI_API_KEY or ANTHROPIC_AUTH_TOKEN."]
    assert "OPENAI_API_KEY" in result.suggestions[0]


def test_openai_live_discovery_uses_tolerant_payload_shapes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer sk-test"
        return httpx.Response(200, json={"models": [{"id": "z"}, {"name": "a"}]})

    transport = httpx.MockTransport(handler)
    service = ProviderService(
        environ={"OPENAI_API_KEY": "sk-test"},
        http_client_factory=lambda: httpx.Client(transport=transport),
    )

    result = service.list_models("openai", base_url="https://example.test")

    assert [model.id for model in result.models] == ["a", "z"]
    assert result.discovery.kind == "live"
    assert result.discovery.count == 2
    assert result.discovery.url == "https://example.test/models"


def test_preflight_surfaces_suggested_openai_api_root() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer sk-test"
        if str(request.url) == "https://example.test/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gpt-5.4"}]})
        return httpx.Response(404, text="404 page not found")

    transport = httpx.MockTransport(handler)
    service = ProviderService(
        environ={"OPENAI_API_KEY": "sk-test"},
        http_client_factory=lambda: httpx.Client(transport=transport),
    )

    result = service.preflight(
        provider="openai",
        model="gpt-5.4",
        base_url="https://example.test",
    )

    assert result.status == "warning"
    assert result.ready is True
    assert result.chat_completions_url == "https://example.test/chat/completions"
    assert result.models_url == "https://example.test/v1/models"
    assert result.discovery is not None
    assert result.discovery.suggested_base_url == "https://example.test/v1"
    assert result.issues == ["Configured base URL appears to be missing the provider API root."]
    assert "https://example.test/v1" in result.suggestions[0]


def test_discovery_helpers_match_gateway_semantics() -> None:
    assert candidate_urls("https://example.test/v1") == [
        "https://example.test/v1/models",
        "https://example.test/v1/v1/models",
        "https://example.test/v1/api/models",
    ]
    assert extract_model_ids({"data": [{"id": "one"}, {"model": "two"}]}) == ["one", "two"]
    assert extract_model_ids({"models": ["one"]}) == ["one"]
    assert suggest_base_url("https://example.test", "https://example.test/v1/models") == "https://example.test/v1"


def test_auxiliary_slots_and_unknown_provider_validation() -> None:
    service = ProviderService(environ={})

    slots = service.auxiliary_slots(["subagent"])
    assert slots[0].slot == "subagent"

    with pytest.raises(ServiceValidationError):
        service.auxiliary_slots(["bad-slot"])
    with pytest.raises(ServiceValidationError):
        service.list_models("missing-provider")
