from __future__ import annotations

from dataclasses import dataclass

import pytest

from aether.config.auxiliary_slots import resolve_auxiliary_slot, resolve_subagent_slot


def test_default_subagent_slot_inherits_when_requested() -> None:
    slot = resolve_subagent_slot(environ={}, inherit_if_unconfigured=True)

    assert slot.inherited is True
    assert slot.model == "inherit"


def test_default_subagent_slot_uses_safe_default_when_not_inheriting() -> None:
    slot = resolve_subagent_slot(environ={})

    assert slot.inherited is False
    assert slot.provider_family == "openai-compatible"
    assert slot.provider_name == "openai"
    assert slot.model == "gpt"


def test_env_subagent_slot_provider_and_model() -> None:
    slot = resolve_subagent_slot(
        environ={
            "AETHER_AUX_SUBAGENT_PROVIDER": "claude",
            "AETHER_AUX_SUBAGENT_MODEL": "sonnet",
        }
    )

    assert slot.provider_family == "claude"
    assert slot.provider_name == "claude"
    assert slot.model == "sonnet"
    assert slot.source == "env"


def test_caller_override_wins() -> None:
    slot = resolve_subagent_slot(
        environ={
            "AETHER_AUX_SUBAGENT_PROVIDER": "claude",
            "AETHER_AUX_SUBAGENT_MODEL": "sonnet",
        },
        provider="openai-compatible",
        model="gpt-custom",
    )

    assert slot.provider_name == "openai"
    assert slot.model == "gpt-custom"
    assert slot.source == "explicit"


@dataclass(slots=True)
class _Config:
    aux_subagent_provider: str | None = None
    aux_subagent_model: str | None = None


def test_config_slot_beats_env() -> None:
    slot = resolve_subagent_slot(
        config=_Config(aux_subagent_provider="codex", aux_subagent_model="gpt"),
        environ={"AETHER_AUX_SUBAGENT_PROVIDER": "claude"},
    )

    assert slot.provider_name == "codex"
    assert slot.model == "gpt"
    assert slot.source == "config"


def test_main_provider_env_can_seed_slot_when_slot_env_missing() -> None:
    slot = resolve_subagent_slot(environ={"AETHER_PROVIDER": "claude"})

    assert slot.provider_name == "claude"
    assert slot.model == "sonnet"


def test_deprecated_subagent_env_aliases_are_supported() -> None:
    slot = resolve_subagent_slot(
        environ={
            "AETHER_SUBAGENT_PROVIDER": "openai-compatible",
            "AETHER_SUBAGENT_MODEL": "gpt",
        }
    )

    assert slot.provider_name == "openai"
    assert slot.model == "gpt"


def test_invalid_slot_provider_returns_clear_error() -> None:
    with pytest.raises(ValueError, match="unknown provider family 'sonnect'"):
        resolve_subagent_slot(environ={"AETHER_AUX_SUBAGENT_PROVIDER": "sonnect"})


def test_compression_slot_resolves_but_is_not_engine_wired() -> None:
    slot = resolve_auxiliary_slot(
        "compression",
        environ={
            "AETHER_AUX_COMPRESSION_PROVIDER": "openai-compatible",
            "AETHER_AUX_COMPRESSION_MODEL": "gpt-5.4",
        },
    )

    assert slot.slot == "compression"
    assert slot.provider_name == "openai"
    assert slot.model == "gpt-5.4"
