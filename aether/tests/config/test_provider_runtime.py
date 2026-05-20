from __future__ import annotations

from dataclasses import dataclass

import pytest

from aether.config.provider_runtime import (
    normalize_provider_family,
    provider_choice,
    resolve_main_provider_runtime,
    resolve_provider_runtime_from_env,
)


def test_env_provider_codex_default_model() -> None:
    runtime = resolve_provider_runtime_from_env(environ={"AETHER_PROVIDER": "codex"})

    assert runtime.family == "codex"
    assert runtime.provider_name == "codex"
    assert runtime.model == "gpt-5.4"
    assert runtime.api_key_env_names == ("CODEX_ACCESS_TOKEN", "CODEX_API_KEY")


def test_env_provider_claude_default_model() -> None:
    runtime = resolve_provider_runtime_from_env(environ={"AETHER_PROVIDER": "claude"})

    assert runtime.family == "claude"
    assert runtime.provider_name == "claude"
    assert runtime.model == "claude-sonnet-4-6"
    assert runtime.api_key_env_names == ("ANTHROPIC_API_KEY",)


def test_env_provider_openai_compatible_default_model_and_url() -> None:
    runtime = resolve_provider_runtime_from_env(
        environ={"AETHER_PROVIDER": "openai-compatible"}
    )

    assert runtime.family == "openai-compatible"
    assert runtime.provider_name == "openai"
    assert runtime.model == "gpt-5.4"
    assert runtime.base_url == "https://api.openai.com/v1"
    assert runtime.api_key_env_names == ("OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def test_missing_provider_defaults_to_openai_compatible() -> None:
    runtime = resolve_provider_runtime_from_env(environ={})

    assert runtime.family == "openai-compatible"
    assert runtime.provider_name == "openai"
    assert runtime.model == "gpt-5.4"


def test_openai_aliases_normalize_to_openai_compatible() -> None:
    assert normalize_provider_family("openai") == "openai-compatible"
    assert normalize_provider_family("openai_compatible") == "openai-compatible"
    assert normalize_provider_family("openai-compatible") == "openai-compatible"


def test_invalid_provider_has_clear_error() -> None:
    with pytest.raises(ValueError, match="unknown provider family 'sonnect'"):
        resolve_provider_runtime_from_env(environ={"AETHER_PROVIDER": "sonnect"})


@dataclass(slots=True)
class _Config:
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None


def test_explicit_config_beats_env() -> None:
    runtime = resolve_main_provider_runtime(
        _Config(provider="claude", model="claude-custom"),
        environ={"AETHER_PROVIDER": "codex", "AETHER_MODEL": "gpt-env"},
    )

    assert runtime.family == "claude"
    assert runtime.provider_name == "claude"
    assert runtime.model == "claude-custom"


def test_explicit_arguments_beat_config_and_env() -> None:
    runtime = resolve_main_provider_runtime(
        _Config(provider="claude", model="claude-custom"),
        environ={"AETHER_PROVIDER": "codex"},
        provider="openai-compatible",
        model="gpt-explicit",
        base_url="https://proxy.test/v1",
    )

    assert runtime.family == "openai-compatible"
    assert runtime.provider_name == "openai"
    assert runtime.model == "gpt-explicit"
    assert runtime.base_url == "https://proxy.test/v1"


def test_provider_specific_model_envs_are_preserved() -> None:
    claude = resolve_provider_runtime_from_env(
        environ={"AETHER_PROVIDER": "claude", "ANTHROPIC_MODEL": "claude-env"}
    )
    openai = resolve_provider_runtime_from_env(
        environ={"AETHER_PROVIDER": "openai-compatible", "OPENAI_MODEL": "gpt-env"}
    )

    assert claude.model == "claude-env"
    assert openai.model == "gpt-env"


def test_provider_choice_maps_internal_provider_name() -> None:
    assert provider_choice("openai-compatible").provider_name == "openai"
    assert provider_choice("codex").provider_name == "codex"


def test_runtime_repr_has_no_secret_value_field() -> None:
    runtime = resolve_provider_runtime_from_env(
        environ={"AETHER_PROVIDER": "openai-compatible", "OPENAI_API_KEY": "sk-secret"}
    )

    assert "sk-secret" not in repr(runtime)
    assert "OPENAI_API_KEY" in repr(runtime)
